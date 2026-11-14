"""Recusal for the OpenAI API: judge every function call in a loop you own.

    from recusal_openai import gate_function_calls

    response = client.responses.create(model=..., input=..., tools=...)
    answers = gate_function_calls(response, {"run_sql": run_sql}, policy)
    # pass ``answers`` as the next request's input

``gate_tool_calls(completion, tools, policy)`` is the same gate for Chat Completions.
``policy(tool_name, tool_input)`` is an ordinary Recusal policy returning findings. A
refused call is answered with Recusal's reason under the call's own id, and its tool does
not run. ``audited_function_calls`` and ``audited_tool_calls`` are the same loops with
each decision on a hash-chained audit log.

The decision is made by ``recusal.decide``. This package adds only what is the OpenAI
API's: where the calls are in a response, how each is answered, and its call id.
"""

from ._audit import SURFACE_CHAT, SURFACE_RESPONSES, audited_function_calls, audited_tool_calls
from ._gate import (
    PASSIVE_OUTPUT_TYPES,
    Policy,
    Tools,
    gate_function_calls,
    gate_tool_calls,
    judge_and_run,
)

__version__ = "0.1.0"

__all__ = [
    "PASSIVE_OUTPUT_TYPES",
    "SURFACE_CHAT",
    "SURFACE_RESPONSES",
    "Policy",
    "Tools",
    "__version__",
    "audited_function_calls",
    "audited_tool_calls",
    "gate_function_calls",
    "gate_tool_calls",
    "judge_and_run",
]
