"""A fail-closed gate for LlamaIndex's local ``FunctionTool`` call surfaces.

LlamaIndex invokes a FunctionTool's callback only after its local function has run, so a
callback cannot enforce a policy. ``gated_tool`` instead returns a FunctionTool subclass
whose ``call`` and ``acall`` methods prepare the arguments before delegating to the stock
implementation. Only wrappers returned by this factory are covered; hosted tools and
unwrapped FunctionTools are outside this adapter's boundary.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, Callable, List, Sequence

from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import ToolOutput

from recusal import decide
from recusal.invocation import detached, json_data, reached_parameters, unsupported_signature

Policy = Callable[[str, dict], List[Any]]


class GatedFunctionTool(FunctionTool):
    """A stock FunctionTool that judges the exact named data it will receive.

    Context is host input rather than model input and is deliberately omitted from policy
    evidence. Positional arguments are refused because a policy cannot judge them by name.
    A function whose signature, wrappers, defaults, or binding cannot be reproduced is
    refused before its policy or body runs.
    """

    def __init__(self, source: FunctionTool, policy: Policy) -> None:
        if type(source) is not FunctionTool:
            raise TypeError("gated_tool supports a stock llama_index FunctionTool only")
        real_fn = source.real_fn
        function_args: dict[str, Any]
        if inspect.iscoroutinefunction(real_fn):
            function_args = {"async_fn": real_fn}
        else:
            function_args = {"fn": real_fn}
        super().__init__(
            metadata=source.metadata,
            callback=source._callback,
            async_callback=source._async_callback,
            partial_params=copy.deepcopy(source.partial_params),
            **function_args,
        )
        self._recusal_policy = policy
        self._recusal_gated = True

    def _refusal(self, reason: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> ToolOutput:
        return ToolOutput(
            content=f"Recusal refused `{self.metadata.get_name()}`: {reason}",
            tool_name=self.metadata.get_name(),
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=None,
            is_error=True,
        )

    def _prepare(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        if args:
            return (
                None,
                "positional arguments are unsupported; every argument must be judged by name",
            )

        context_name = self.ctx_param_name if self.requires_context else None
        if (
            context_name is not None
            and context_name not in kwargs
            and context_name not in self.partial_params
        ):
            return None, "Context is required for this tool"

        try:
            why = unsupported_signature(self.real_fn, skip=(context_name,) if context_name else ())
            if why is not None:
                return None, f"its signature cannot be judged ({why})"
            levels = reached_parameters(self.real_fn)
        except Exception as exc:  # the callable cannot be examined faithfully
            return None, f"its signature cannot be judged ({type(exc).__name__})"

        merged = {**self._field_defaults, **self.partial_params, **kwargs}
        names = {}
        for level in levels:
            for name, parameter in level.items():
                if name != context_name:
                    names.setdefault(name, parameter)
        unknown = sorted(set(merged) - set(names) - ({context_name} if context_name else set()))
        if unknown:
            return None, f"unknown parameter(s) {unknown!r}"

        for name, parameter in names.items():
            if name in merged:
                continue
            if parameter.default is parameter.empty:
                return None, f"required parameter `{name}` was not supplied"
            if not json_data(parameter.default):
                return None, f"`{name}` has a default outside the JSON domain"
            merged[name] = copy.deepcopy(parameter.default)

        # Bind with the public signature too: it catches any framework-specific binding
        # rule that the reached-callable walk did not express.
        try:
            inspect.signature(self.real_fn).bind(**merged)
        except TypeError as exc:
            return None, f"arguments do not bind to the tool ({exc})"

        evidence = {name: value for name, value in merged.items() if name != context_name}
        try:
            decision, reason = decide(
                self.metadata.get_name(), detached(evidence), self._recusal_policy, safe_errors=True
            )
        except Exception as exc:
            return (
                None,
                f"failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})",
            )
        if decision == "deny":
            return None, reason
        return merged, None

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        prepared, reason = self._prepare(args, kwargs)
        if reason is not None:
            return self._refusal(reason, args, kwargs)
        assert prepared is not None
        return super().call(**prepared)

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        prepared, reason = self._prepare(args, kwargs)
        if reason is not None:
            return self._refusal(reason, args, kwargs)
        assert prepared is not None
        return await super().acall(**prepared)


def gated_tool(tool: FunctionTool, policy: Policy) -> GatedFunctionTool:
    """Return an explicit, fail-closed wrapper for one local ``FunctionTool``."""
    return GatedFunctionTool(tool, policy)


def is_gated(tools: Sequence[object]) -> bool:
    """True only when every supplied local tool came from :func:`gated_tool`.

    This checks only the list it is handed. It cannot see a tool added later or a
    provider-hosted tool that never appears in this list.
    """
    return bool(tools) and all(isinstance(tool, GatedFunctionTool) for tool in tools)
