"""
What a gate can judge: the data-only invocation, held in one place.

A gate in front of a tool call makes one promise: the arguments the policy judged are the
arguments the tool ran with. Keeping it takes the same few decisions in every framework,
and they are decisions about data, not about any framework:

- which parameter annotations a gate can judge by value (:func:`supported_annotation`);
- which values are JSON data, so a default can be completed and judged
  (:func:`json_data`);
- whether every parameter of a callable can be judged by name
  (:func:`unsupported_signature`);
- how to hand a policy a copy it can write to without reaching what executes
  (:func:`detached`).

The framework gates under ``examples/`` each carried their own copy of these. They import
them from here instead, so the supported set is defined once and a fix reaches every gate
that uses it. Nothing here imports a framework, reads a file or calls a policy; every
function is a pure function of its argument. Standard library only.

What stays in each gate is the part that is the framework's: where its hook sits, how it
prepares an invocation, and which of its own parameters it injects.
"""

import inspect
import math
import types
import typing
from enum import Enum
from typing import Any, Callable, Optional, Tuple

#: The scalar types a judged parameter may be annotated with.
SCALARS: Tuple[type, ...] = (str, int, float, bool)

# The PEP 604 union (``int | None``) has its own type from Python 3.10; 3.9 has none.
_UNION_TYPE = getattr(types, "UnionType", None)


def supported_annotation(annotation: Any) -> bool:
    """True for an annotation whose preparation is a pure function of the value: ``str``,
    ``int``, ``float``, ``bool``, ``Optional`` of one, ``list[T]`` and ``dict[str, T]`` with
    ``T`` supported in turn. Everything else (no annotation, ``Any``, a model, an ``Enum``, an
    ``Annotated``, a dataclass, a bare ``list``/``dict``) is unsupported."""
    if annotation in SCALARS:
        return True
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Union or (_UNION_TYPE is not None and origin is _UNION_TYPE):
        rest = [a for a in args if a is not type(None)]
        return len(rest) == 1 and supported_annotation(rest[0])
    if origin is list:
        return len(args) == 1 and supported_annotation(args[0])
    if origin is dict:
        return len(args) == 2 and args[0] is str and supported_annotation(args[1])
    return False


def json_data(value: Any) -> bool:
    """True for a value in the JSON domain (what a default may be)."""
    if value is None or isinstance(value, SCALARS):
        return True
    if isinstance(value, list):
        return all(json_data(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and json_data(v) for k, v in value.items())
    return False


def unsupported_signature(func: Callable[..., Any], skip: Tuple[str, ...] = ()) -> Optional[str]:
    """Why this callable's parameters cannot all be judged, or None if they can.

    ``skip`` names the parameters the framework injects itself (a context object, a run
    manager); they are not judged and not checked. Name them explicitly: a parameter that
    is merely absent from a schema is not one of them."""
    try:
        parameters = inspect.signature(func).parameters
        # resolves string annotations (PEP 563) and keeps Annotated so it can be refused
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception:  # noqa: BLE001, an uninspectable callable cannot be judged
        return "its signature cannot be inspected"
    for name, parameter in parameters.items():
        if name in skip:
            continue
        annotation = hints.get(name, parameter.empty)
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            return f"`{name}` is variadic; every argument the tool receives must be judged by name"
        if parameter.kind is parameter.POSITIONAL_ONLY:
            return f"`{name}` is positional-only and cannot be judged by name"
        if not supported_annotation(annotation):
            return f"`{name}` is not annotated with a supported data type"
        if parameter.default is not parameter.empty and not json_data(parameter.default):
            return f"`{name}` has a default outside the JSON domain"
    return None


def detached(value: Any, ancestors: Tuple[int, ...] = ()) -> Any:
    """Detach the supported data domain without invoking application copying/serialization.

    Returns a copy the policy can write to without reaching the original, or raises
    ``TypeError`` for a value that is not data. A caller turns that into a refusal.

    Exact container/scalar types avoid subclasses with custom iteration or copy behavior.
    Every Enum member is refused, whatever its value: a member is one shared object, and
    attributes its class sets on it (an ``__init__`` that stores a dict, say) would be the
    same object at policy and at execution. Passing only ``.value`` would hide that state
    from the policy while the tool still used it, so that is no repair either.
    """
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError("non-finite numbers are not JSON data")
        return value
    if isinstance(value, Enum):
        raise TypeError(
            "Enum arguments are unsupported; declare data-only parameters "
            "and resolve application types inside the tool body"
        )
    if type(value) not in (dict, list):
        raise TypeError("arguments must be data-only")
    if id(value) in ancestors:
        raise TypeError("cyclic arguments are not JSON data")
    ancestors = (*ancestors, id(value))
    if type(value) is list:
        return [detached(item, ancestors) for item in value]
    if any(type(key) is not str for key in value):
        raise TypeError("argument keys must be strings")
    return {key: detached(item, ancestors) for key, item in value.items()}
