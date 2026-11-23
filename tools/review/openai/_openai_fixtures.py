"""What the ``recusal-openai`` review scripts share: this checkout on ``sys.path``, SDK
objects built without a network, and audit logs in a temporary directory.

Not a script. ``corpus.py`` and ``keyword_matrix.py`` import it.

The code under test is the one in this checkout, never an installed copy: ``load()``
puts the repository root and ``adapters/openai/src`` first on ``sys.path`` and stops if
``recusal`` or ``recusal_openai`` was imported from anywhere else. The OpenAI SDK is the
one installed in the interpreter; the review ran against the version locked in
``requirements/adapter-openai.txt``, and each script prints the version it found.

Two ways of building a response are used, because the gate meets both:

- validated: ``Response.model_validate`` and ``ChatCompletion.model_validate``, which
  reject a field of the wrong type;
- as received: ``openai._models.construct_type``, the function the SDK itself uses to
  build a response from the bytes the API sent. It validates nothing, so a name that is
  a list, arguments that are bytes or a status the SDK does not know reach the gate as
  they are. That function is private to the SDK, which is one reason the SDK version is
  pinned.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ADAPTER_SRC = ROOT / "adapters" / "openai" / "src"
REQUIREMENTS = "requirements/adapter-openai.txt"

_RESPONSE = {
    "id": "resp_1",
    "created_at": 0,
    "model": "m",
    "object": "response",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


def load() -> SimpleNamespace:
    """Import the SDK and the gate from this checkout; exit with a message if either is
    missing or was imported from somewhere else."""
    for entry in (str(ROOT), str(ADAPTER_SRC)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    try:
        import openai
        from openai._models import construct_type
        from openai.types.chat import ChatCompletion
        from openai.types.responses import Response
    except ImportError as exc:
        raise SystemExit(
            f"the OpenAI SDK is not installed ({exc}); install the locked requirements: "
            f"python -m pip install --require-hashes -r {REQUIREMENTS}"
        ) from None
    import recusal_openai

    import recusal
    from recusal import Finding
    from recusal.audit import AuditLog

    for module in (recusal, recusal_openai):
        where = Path(module.__file__ or "").resolve()
        if ROOT not in where.parents:
            raise SystemExit(
                f"{module.__name__} was imported from {where}, not from this checkout "
                f"({ROOT}); the review scripts test the code beside them"
            )
    return SimpleNamespace(
        AuditLog=AuditLog,
        ChatCompletion=ChatCompletion,
        Finding=Finding,
        Response=Response,
        construct_type=construct_type,
        audited_function_calls=recusal_openai.audited_function_calls,
        audited_tool_calls=recusal_openai.audited_tool_calls,
        gate_function_calls=recusal_openai.gate_function_calls,
        gate_tool_calls=recusal_openai.gate_tool_calls,
        judge_and_run=recusal_openai.judge_and_run,
        versions={
            "openai": openai.__version__,
            "recusal": recusal.__version__,
            "recusal_openai": recusal_openai.__version__,
        },
    )


def function_call(call_id: Any, name: Any, arguments: Any, **fields: Any) -> dict:
    """One Responses ``function_call`` output item, as a dict."""
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        **fields,
    }


def response(sdk: SimpleNamespace, items: list, validated: bool, **fields: Any) -> Any:
    """A Responses API response holding ``items``, validated or as received."""
    value = {**_RESPONSE, "output": list(items), **fields}
    if validated:
        return sdk.Response.model_validate(value)
    return sdk.construct_type(type_=sdk.Response, value=value)


def completion(
    sdk: SimpleNamespace, calls: Any, validated: bool, choice: Any = None, **message: Any
) -> Any:
    """A Chat Completions completion with one choice whose message carries ``calls``, a
    list of ``(id, name, arguments)``, or no ``tool_calls`` field when ``calls`` is None.
    ``choice`` is the fields of the choice itself: ``finish_reason``, or none of it."""
    body: dict = {"role": "assistant", "content": None, **message}
    if calls is not None:
        body["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
            for call_id, name, arguments in calls
        ]
    fields = {"finish_reason": "tool_calls"} if choice is None else choice
    value = {
        "id": "c",
        "created": 0,
        "model": "m",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": body, **fields}],
    }
    if validated:
        return sdk.ChatCompletion.model_validate(value)
    return sdk.construct_type(type_=sdk.ChatCompletion, value=value)


class Logs:
    """Audit logs on disk in one temporary directory, removed on exit.

    The logs are files, as an application's are, so an entry the log cannot write (a
    lone surrogate in a field, for one) behaves here as it would there.
    """

    def __init__(self, sdk: SimpleNamespace) -> None:
        self._sdk = sdk
        self._directory = Path(tempfile.mkdtemp(prefix="recusal-review-"))
        self._count = 0

    def __enter__(self) -> "Logs":
        return self

    def __exit__(self, *_exc: Any) -> None:
        shutil.rmtree(self._directory, ignore_errors=True)

    def new(self) -> tuple:
        """A new empty log and the path of its file."""
        self._count += 1
        path = self._directory / f"{self._count}.jsonl"
        return self._sdk.AuditLog(path=str(path)), path

    @staticmethod
    def entries(path: Path) -> list:
        """The entries on file at ``path``; the file is then removed."""
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as handle:
            found = [json.loads(line) for line in handle if line.strip()]
        try:
            path.unlink()
        except OSError:
            pass
        return found
