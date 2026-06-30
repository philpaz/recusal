"""
Recusal, LIVE Claude tool-call gate (real Anthropic SDK, manual agent loop).

This actually drives Claude. It gives the model a CRM tool and a task that leads
it to write to the *wrong* customer; Recusal adjudicates the proposed call and
**refuses before the tool runs**, handing Claude a reason. Claude then adapts.

Setup:
    pip install anthropic
    ant auth login           # or: export ANTHROPIC_API_KEY=...

Run:
    python examples/claude_agent_live.py

(For the deterministic, no-API version used in CI and for screenshots, see
examples/claude_refusal.py.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding  # noqa: E402
from recusal.claude import gate_tool_use  # noqa: E402

MODEL = "claude-opus-4-8"
MAX_TURNS = 6  # bound the loop, agents shouldn't run unbounded

ACTIVE_CUSTOMER = {"customer_id": "C1001", "name": "Bob Smith"}

SYSTEM = (
    "You are a CRM assistant. The active customer in this session is "
    f"{ACTIVE_CUSTOMER['name']} (customer_id {ACTIVE_CUSTOMER['customer_id']}). "
    "When the user asks to change a record, use the update_customer_record tool."
)

TOOLS = [
    {
        "name": "update_customer_record",
        "description": "Update a single field on a customer's record.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer to update, e.g. C1001",
                },
                "field": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["customer_id", "field", "value"],
            "additionalProperties": False,
        },
    }
]


def gather_evidence(tool_input: dict) -> list:
    """A write must target the active customer, an invariant the model can't self-enforce."""
    target = tool_input.get("customer_id")
    active = ACTIVE_CUSTOMER["customer_id"]
    if target != active:
        return [
            Finding.fail(
                "subject_match",
                severity="CRITICAL",
                message=f"write targets {target} but the active customer this turn is {active}",
                target=target,
                active=active,
            )
        ]
    return [Finding.ok("subject_match", severity="CRITICAL", target=target)]


def execute_tool(tool_input: dict) -> str:
    return f"OK: set {tool_input['field']}={tool_input['value']} on {tool_input['customer_id']}"


def main():
    try:
        import anthropic
    except ImportError:
        print("This demo needs the Anthropic SDK:\n    pip install anthropic")
        return

    try:
        client = anthropic.Anthropic()  # auth via ANTHROPIC_API_KEY or `ant auth login`
    except Exception as exc:  # noqa: BLE001
        print(
            f"Could not initialise the Anthropic client: {exc}\n"
            "Set ANTHROPIC_API_KEY or run `ant auth login`."
        )
        return

    messages = [
        {
            "role": "user",
            "content": "Please set the loyalty_tier to Gold for customer C-9988.",
        }
    ]

    print(f"RECUSAL - live Claude gate ({MODEL})")
    print(f"Active customer: {ACTIVE_CUSTOMER['name']} ({ACTIVE_CUSTOMER['customer_id']})")
    print(f"User: {messages[0]['content']}\n")

    for _turn in range(MAX_TURNS):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"API call failed: {exc}")
            return

        for block in resp.content:
            if block.type == "text" and block.text.strip():
                print(f"Claude: {block.text.strip()}\n")

        if resp.stop_reason != "tool_use":
            break

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": resp.content})

        results = []
        for tool in tool_uses:
            print(f"Claude wants to call: {tool.name}({tool.input})")
            allow, refusal = gate_tool_use(
                tool.id, gather_evidence(tool.input), tool_name=tool.name
            )
            if not allow:
                print(f"  RECUSAL -> REFUSED: {refusal['content']}\n")
                results.append(refusal)
            else:
                out = execute_tool(tool.input)
                print(f"  RECUSAL -> ALLOWED. {out}\n")
                results.append({"type": "tool_result", "tool_use_id": tool.id, "content": out})

        messages.append({"role": "user", "content": results})

    print("--- done ---")


if __name__ == "__main__":
    main()
