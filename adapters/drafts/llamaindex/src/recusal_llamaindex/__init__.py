"""Recusal gate for explicit LlamaIndex ``FunctionTool`` wrappers."""

from ._gate import GatedFunctionTool, Policy, gated_tool, is_gated

__version__ = "0.1.0"

__all__ = ["GatedFunctionTool", "Policy", "__version__", "gated_tool", "is_gated"]
