# recusal-llamaindex (draft)

An experimental Recusal adapter for explicit local LlamaIndex `FunctionTool` wrappers.
It judges named arguments, including ordinary Python defaults, before the function runs.

It does **not** cover unwrapped tools, provider-hosted tools, or tools added after an
application calls `is_gated`. This draft has not yet been accepted or published.

For `llama-index-core==0.14.25`, the workflow-agent execution route calls a
`FunctionTool` through `acall()`. The draft has an offline regression test for that
route; it does not yet claim coverage for every agent, integration, or provider route.
That route overwrites a caller-supplied context name with its host `Context` before the
tool runs; the adapter never exposes that host value to the policy. Direct FunctionTool
calls preserve their values rather than coercing them, so the policy and function see the
same types on that surface.
