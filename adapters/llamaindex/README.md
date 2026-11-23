# recusal-llamaindex (draft)

An experimental Recusal adapter for explicit local LlamaIndex `FunctionTool` wrappers.
It judges named arguments, including ordinary Python defaults, before the function runs.

It does **not** cover unwrapped tools, provider-hosted tools, or tools added after an
application calls `is_gated`. This draft has not yet been accepted or published.
