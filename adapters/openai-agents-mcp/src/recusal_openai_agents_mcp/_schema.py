"""A tool's listed input schema: strict validation of a call's arguments against all of it,
and every default it lists written out.

The gate sees what is sent to a server, and the tool's listed ``inputSchema`` is the only
statement of what the server then does with it. A server converts a value of another type
(the string ``"false"`` to a boolean) and fills what a call leaves out from its default, at
any depth, after the policy judged. So before the policy runs, :meth:`Listed.complete`

1. validates the arguments against the whole schema with ``jsonschema``, in the draft the
   schema declares (2020-12 when it declares none), formats not asserted. JSON Schema
   validation does not convert: a string is not a boolean. It is made stricter in two
   places where JSON Schema is looser than what a server then runs: a float is not an
   ``integer`` (``5.0``), and an ``enum`` or ``const`` value must be of the same JSON type
   as the listed one (``1.0`` and ``true`` are not ``1``). A ``pattern`` that Python's
   ``re`` cannot compile is not asserted;
2. refuses a string argument whose text is JSON for null, a list or an object where the
   property's schema allows another JSON type than string: the MCP Python SDK's server
   decodes such text and runs with that value;
3. writes every default the schema lists for a property the arguments leave out, at every
   depth, repeated until nothing changes;
4. validates the completed arguments again.

Where a ``default`` is read. For a property the arguments leave out: on the property's own
schema; or, when that lists none, on the schema its ``$ref`` names, on a member of its
``allOf`` and on an alternative of its ``anyOf`` or ``oneOf``, where all that are listed
must be one value. For an item a list leaves out (``prefixItems``) the call is refused. A
``default`` on a schema whose value is always present (the schema of the arguments
themselves, of the items of a list, of ``additionalProperties``) has nothing to be applied
to, since JSON has no way to send a member without a value. A ``default`` anywhere else,
under a keyword the walk does not follow or under a keyword the gate does not know,
refuses the schema.

Whatever it cannot check or complete is refused, with a reason that carries no text of the
model's: a schema that is not a JSON object or not a valid schema, a draft other than
draft-07, 2019-09 and 2020-12, a ``$ref`` that does not resolve inside the schema, a schema
that refers to itself without a value in between, a default where the walk does not go, a
left-out property whose alternatives list different defaults, a value for which the
schema's alternatives would write different defaults, and arguments that cost more to
check than the work the gate allows for their size.

The work is bounded. Nothing is validated twice for the same schema and the same value
while the arguments stand as they are, the defaults for one schema and one place in the
arguments are worked out once in a pass, and every step is counted against an allowance
that grows with the number of values in the arguments.

Two kinds of listed default cannot be sent. One that is not valid against its own
property schema (``{"default": null, "type": "string"}``, which Pydantic publishes for
``q: str = None``): the server runs with it when the parameter is left out and would
reject it if it were sent. And a top-level default that is itself JSON text of the kind
step 2 refuses (``mode: Optional[str] = "null"``): the server runs with the string when
the parameter is left out and would decode it if it were sent. Both are in the arguments
the policy judges and not in the arguments that are sent.
"""

from __future__ import annotations

import copy
import functools
import json
import re
from contextvars import ContextVar
from typing import Any, Iterable, Optional
from urllib.parse import unquote

from jsonschema import FormatChecker, validators  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError, ValidationError  # type: ignore[import-untyped]
from referencing import Registry

from recusal.invocation import detached as _detached

#: The JSON type each Python type of decoded JSON data is.
_JSON_TYPES = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
    type(None): "null",
}
_EVERY_TYPE = frozenset(_JSON_TYPES.values())

#: Keywords whose value is a map of names to subschemas, a list of subschemas, a subschema.
_MAPS = ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas")
_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SINGLE = (
    "items",
    "additionalProperties",
    "additionalItems",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contentSchema",
)
#: Of those, the ones the walk that writes defaults does not go through. A default under
#: one of them refuses the schema.
_UNWALKED = (
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contentSchema",
)
#: Keywords whose value is data and may hold anything, a member named ``default`` included.
_VALUES = ("default", "const", "enum", "examples", "example")
#: References the gate does not resolve.
_DYNAMIC = ("$dynamicRef", "$recursiveRef")
#: Keywords that apply a subschema to the very value the schema that holds them is applied
#: to. A schema that leads back to itself through these alone never finishes validating.
_SAME_VALUE = ("allOf", "anyOf", "oneOf", "not", "if", "then", "else", "dependentSchemas")

#: How many times the defaults are written before the gate gives up on a fixed point.
_MAX_PASSES = 16

#: The work one call may cost: this many steps, and this many more for every value in its
#: arguments (an object, a list, and each member, item and scalar in them). A step is one
#: keyword of the schema applied to one value, or one schema whose defaults are worked out
#: for one place in the arguments. The largest ordinary calls measured (the tools of a
#: server built with the MCP Python SDK, the listed tools of the reference GitHub server,
#: and a recursive model nested to the 64-level limit) spend under a twentieth of this.
_STEPS = 10_000
_STEPS_PER_VALUE = 1_000

_SCHEMA = "the server's listed schema for this tool"

_TOO_COSTLY = f"the arguments are too costly to check against {_SCHEMA}"


class Refused(Exception):
    """The arguments cannot be tied to what the server would run; ``reason`` says why."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _TooCostly(Exception):
    """The work a call is allowed is spent."""


class _Work:
    """What one call has cost so far, and what is already worked out for its arguments as
    they stand."""

    def __init__(self, allowed: int):
        self.allowed = allowed
        self.steps = 0
        #: (keyword, schema, value) that raised no error; each entry holds the schema and
        #: the value, so the identities it is keyed by cannot be given to other objects
        self.valid: dict[tuple, tuple] = {}
        #: (schema, value) -> whether the value is valid against the schema, kept the same way
        self.answers: dict[tuple, tuple] = {}
        #: (schema, place in the arguments) -> the defaults to write there
        self.plans: dict[tuple, dict] = {}

    def step(self) -> None:
        self.steps += 1
        if self.steps > self.allowed:
            raise _TooCostly

    def forget(self) -> None:
        """The arguments were written to: nothing worked out for them stands."""
        self.valid.clear()
        self.answers.clear()
        self.plans.clear()


#: The call being completed, which the keyword functions below count their steps against.
#: A context variable, so two calls in two threads or two tasks never share one.
_WORK: ContextVar[Optional[_Work]] = ContextVar("recusal_openai_agents_mcp_work", default=None)


def _count(value: Any) -> int:
    """How many values ``value`` holds, itself included."""
    if type(value) is dict:
        return 1 + sum(_count(member) for member in value.values())
    if type(value) is list:
        return 1 + sum(_count(item) for item in value)
    return 1


def _same(a: Any, b: Any) -> bool:
    """Equal, and of the same JSON type all the way down (``1`` is not ``True`` or ``1.0``)."""
    if type(a) is not type(b):
        return False
    if type(a) is list:
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if type(a) is dict:
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return bool(a == b)


def _enum(validator: Any, choices: Any, instance: Any, schema: Any) -> Iterable[Any]:
    if not any(_same(instance, choice) for choice in choices):
        yield ValidationError("not one of the listed values")


def _const(validator: Any, fixed: Any, instance: Any, schema: Any) -> Iterable[Any]:
    if not _same(instance, fixed):
        yield ValidationError("not the fixed value")


@functools.lru_cache(maxsize=1024)
def _compiled(pattern: Any) -> Any:
    """The pattern as Python's ``re`` reads it, or None when it cannot (``\\p{L}`` and the
    other ECMA 262 forms Python has no reading of)."""
    try:
        return re.compile(pattern)
    except (re.error, TypeError, ValueError, OverflowError, RecursionError):
        return None


def _pattern(validator: Any, pattern: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """``pattern`` where Python can read it. A pattern it cannot compile is a constraint
    the server checks and the gate does not, as a ``format`` is."""
    regex = _compiled(pattern)
    if regex is not None and type(instance) is str and not regex.search(instance):
        yield ValidationError("does not match the pattern")


def _is_integer(checker: Any, instance: Any) -> bool:
    return type(instance) is int


def _passes(validator: Any, instance: Any, schema: Any) -> bool:
    """Whether ``instance`` is valid against ``schema``, stopping at the first error, and
    worked out once for a schema and a value while the arguments stand as they are."""
    work = _WORK.get()
    key = (id(schema), id(instance))
    if work is not None:
        known = work.answers.get(key)
        if known is not None and known[0] is schema and known[1] is instance:
            return bool(known[2])
    answer = next(iter(validator.descend(instance, schema)), None) is None
    if work is not None:
        work.answers[key] = (schema, instance, answer)
    return answer


def _any_of(validator: Any, alternatives: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """``anyOf`` by whether each alternative is valid, which is all it asks. ``jsonschema``
    gathers every error of every alternative that fails, and under a schema that refers
    to itself that is the same alternatives against the same values again at every level."""
    if not any(_passes(validator, instance, each) for each in alternatives):
        yield ValidationError("matches none of the alternatives")


def _one_of(validator: Any, alternatives: Any, instance: Any, schema: Any) -> Iterable[Any]:
    if sum(1 for each in alternatives if _passes(validator, instance, each)) != 1:
        yield ValidationError("does not match exactly one of the alternatives")


def _measured(keyword: str, check: Any) -> Any:
    """``check``, one keyword's validation, counted against the call's allowance and not
    run again for a schema and a value it has already passed for."""

    def measured(validator: Any, value: Any, instance: Any, schema: Any) -> Iterable[Any]:
        work = _WORK.get()
        if work is None:
            yield from check(validator, value, instance, schema) or ()
            return
        work.step()
        key = (keyword, id(schema), id(instance))
        known = work.valid.get(key)
        if known is not None and known[0] is schema and known[1] is instance:
            return
        clean = True
        for error in check(validator, value, instance, schema) or ():
            clean = False
            yield error
        if clean:
            work.valid[key] = (schema, instance)

    return measured


def _strict(base: Any) -> Any:
    """``base`` with the checks a server's conversion makes necessary, a ``pattern`` that
    is skipped where Python cannot read it, and every keyword counted."""
    keywords = {
        **base.VALIDATORS,
        "enum": _enum,
        "const": _const,
        "pattern": _pattern,
        "anyOf": _any_of,
        "oneOf": _one_of,
    }
    return validators.extend(
        base,
        validators={keyword: _measured(keyword, check) for keyword, check in keywords.items()},
        type_checker=base.TYPE_CHECKER.redefine("integer", _is_integer),
    )


#: The drafts the gate reads, each with the stricter checks.
_DRAFTS = {
    base: _strict(base)
    for base in (
        validators.Draft7Validator,
        validators.Draft201909Validator,
        validators.Draft202012Validator,
    )
}

#: The ``$schema`` each draft is declared by, without the scheme and the trailing ``#``:
#: servers write both ``http`` and ``https``, with and without the ``#``.
_DIALECTS = {
    "json-schema.org/draft-07/schema": validators.Draft7Validator,
    "json-schema.org/draft/2019-09/schema": validators.Draft201909Validator,
    "json-schema.org/draft/2020-12/schema": validators.Draft202012Validator,
}


def _draft(declared: Any) -> Any:
    """The draft a ``$schema`` value declares, or None for one the gate does not read."""
    if type(declared) is not str:
        return None
    scheme, _, rest = declared.partition("://")
    if scheme not in ("http", "https"):
        return None
    return _DIALECTS.get(rest[:-1] if rest.endswith("#") else rest)


def _resolve(root: Any, ref: Any) -> Any:
    """The subschema a ``$ref`` names inside ``root``, or :class:`Refused`."""
    node = root
    if type(ref) is str and (ref == "#" or ref.startswith("#/")):
        for token in ref[2:].split("/") if ref != "#" else ():
            token = unquote(token).replace("~1", "/").replace("~0", "~")
            if type(node) is dict and token in node:
                node = node[token]
            elif type(node) is list and token.isdigit() and int(token) < len(node):
                node = node[int(token)]
            else:
                node = None
                break
        if type(node) in (dict, bool):
            return node
    raise Refused(
        f"{_SCHEMA} has a `$ref` that does not resolve inside the schema itself, so the "
        "gate cannot check the arguments against it"
    )


def _holds_default(value: Any) -> bool:
    """Whether data holds an object with a member named ``default``, at any depth."""
    if type(value) is dict:
        return "default" in value or any(_holds_default(member) for member in value.values())
    if type(value) is list:
        return any(_holds_default(item) for item in value)
    return False


def _endless(node: Any, root: Any, state: dict[int, bool]) -> bool:
    """Whether ``node`` leads back to itself through ``$ref`` and the keywords that apply
    a subschema to the same value: validating any value against it would never finish.
    A ``$ref`` that does not resolve inside ``root`` is refused here (:func:`_resolve`)."""
    if type(node) is not dict:
        return False
    if id(node) in state:
        return state[id(node)]  # True while the node is still being followed
    state[id(node)] = True
    onward: list[Any] = []
    if "$ref" in node:
        onward.append(_resolve(root, node["$ref"]))
    for keyword in (*_SAME_VALUE, "dependencies"):
        member = node.get(keyword)
        if type(member) is dict and keyword in ("dependentSchemas", "dependencies"):
            onward.extend(member.values())
        elif type(member) is list:
            onward.extend(member)
        else:
            onward.append(member)
    found = any(_endless(each, root, state) for each in onward)
    state[id(node)] = found
    return found


def _scan(
    node: Any, root: Any, names: set[str], state: dict[int, bool], under: Optional[str] = None
) -> None:
    """Refuse a schema the gate cannot check or complete, and collect the property names
    it lists (the only argument names a refusal may quote).

    Every keyword of every subschema is one of four kinds, and a ``default`` is refused
    wherever it is not read: a keyword the walk that writes defaults follows (its
    subschemas are scanned in turn), one it does not follow (a ``default`` anywhere under
    it is refused), one whose value is data (not looked into), and any other keyword,
    which the gate does not know: it is refused when its value holds a ``default``, so a
    keyword a later draft or a vendor adds cannot carry one past the gate."""
    if type(node) is not dict:
        return
    for keyword in _DYNAMIC:
        if keyword in node:
            raise Refused(f"{_SCHEMA} uses `{keyword}`, which the gate does not resolve")
    if node is not root and "$id" in node:
        raise Refused(
            f"{_SCHEMA} embeds a schema with an `$id` of its own, which the gate does not "
            "resolve references against"
        )
    if _endless(node, root, state):
        raise Refused(f"{_SCHEMA} refers to itself without end")
    if under is not None and "default" in node:
        raise Refused(
            f"{_SCHEMA} lists a default under `{under}`, where the gate cannot tell "
            "whether the server would apply it"
        )
    for keyword, member in node.items():
        inner = keyword if keyword in _UNWALKED else under
        if keyword in _MAPS or keyword == "dependencies":
            if type(member) is dict:
                if keyword == "properties":
                    names.update(name for name in member if type(name) is str)
                if keyword == "patternProperties":
                    if any(_compiled(pattern) is None for pattern in member):
                        raise Refused(
                            f"{_SCHEMA} names members by a `patternProperties` pattern that "
                            "Python's `re` cannot read, so the gate cannot tell which "
                            "members it governs"
                        )
                for each in member.values():
                    _scan(each, root, names, state, inner)
        elif keyword in _LISTS or keyword in _SINGLE:
            for each in member if type(member) is list else (member,):
                _scan(each, root, names, state, inner)
        elif keyword not in _VALUES and _holds_default(member):
            raise Refused(
                f"{_SCHEMA} lists a default under a keyword the gate does not read, where "
                "it cannot tell whether the server would apply it"
            )


def _quoted(names: Iterable[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _declared(types: Any) -> list[str]:
    return [types] if type(types) is str else [name for name in types if type(name) is str]


def _json_text(text: str) -> bool:
    """Whether a string is JSON text for null, a list or an object: what the MCP Python
    SDK's server decodes and runs in the string's place. Text for a boolean, a number or
    a string it keeps a string, and so does the gate; text that is not JSON is not."""
    try:
        decoded = json.loads(text)
    except (ValueError, RecursionError):
        return False
    return decoded is None or type(decoded) in (list, dict)


class Listed:
    """One tool's input schema as a server listed it."""

    def __init__(self, schema: Any):
        self.schema = schema
        self._validator: Any = None
        self._failure: Optional[Refused] = None
        self._names: set[str] = set()

    # --- the schema itself must be checkable --------------------------------------------------

    def _prepared(self) -> Any:
        """The validator for this schema, or :class:`Refused` when the gate cannot read it.
        Worked out once for a listing."""
        if self._failure is not None:
            raise self._failure
        if self._validator is None:
            try:
                self._validator = self._prepare()
            except Refused as refused:
                self._failure = refused
                raise
        return self._validator

    def _prepare(self) -> Any:
        schema = self.schema
        if type(schema) is not dict:
            raise Refused(f"{_SCHEMA} is not a JSON object")
        try:
            _detached(schema)
        except (TypeError, RecursionError):
            raise Refused(f"{_SCHEMA} is not JSON data") from None
        base = _draft(schema["$schema"]) if "$schema" in schema else validators.Draft202012Validator
        if base not in _DRAFTS:
            raise Refused(
                f"{_SCHEMA} declares a JSON Schema draft the gate does not read (it reads "
                "draft-07, 2019-09 and 2020-12)"
            )
        try:
            _scan(schema, schema, self._names, {})
            # a pattern is checked for being one Python can read where it is used, not here:
            # one that is valid ECMA 262 and not valid Python does not make a schema invalid
            formats = FormatChecker(())
            formats.checkers = {
                name: check
                for name, check in base.FORMAT_CHECKER.checkers.items()
                if name != "regex"
            }
            _DRAFTS[base].check_schema(schema, format_checker=formats)
        except SchemaError:
            raise Refused(f"{_SCHEMA} is not a valid JSON Schema") from None
        except RecursionError:
            raise Refused(f"{_SCHEMA} is nested too deeply to read") from None
        self._modern = base is validators.Draft202012Validator
        # in draft-07 a `$ref` is read alone: the keywords beside it are not applied
        self._ref_alone = base is validators.Draft7Validator
        # an empty registry: nothing outside the schema is ever fetched
        return _DRAFTS[base](schema, registry=Registry())

    # --- validation -----------------------------------------------------------------------------

    def _invalid(self, validator: Any, instance: Any, schema: Any) -> bool:
        return not _passes(validator, instance, schema)

    def _path(self, parts: Iterable[Any]) -> str:
        """A place in the arguments, written only from the property names the schema lists
        and from positions in a list; a key the schema does not list is ``*``."""
        text = ""
        for part in parts:
            if type(part) is int:
                text += f"[{part}]"
            else:
                text += ("." if text else "") + (part if part in self._names else "*")
        return text

    def _why(self, error: Any) -> str:
        """The reason for one validation error, with nothing of the instance in it but its
        JSON type."""
        path = self._path(error.absolute_path)
        subject, plural = (f"argument `{path}`", False) if path else ("the arguments", True)
        keyword, value = error.validator, error.validator_value
        strict = "the gate sends only arguments that are valid against it as they stand"
        typed: Optional[list[str]] = None
        if keyword == "type":
            typed = _declared(value)
        elif keyword in ("anyOf", "oneOf") and all(
            type(branch) is dict and "type" in branch for branch in value
        ):
            typed = list(dict.fromkeys(n for branch in value for n in _declared(branch["type"])))
            actual = _JSON_TYPES[type(error.instance)]
            if actual in typed or (actual == "integer" and "number" in typed):
                typed = None  # the JSON type is listed: another constraint failed
        if typed is not None:
            actual = _JSON_TYPES[type(error.instance)]
            return (
                f"{subject} {'are' if plural else 'is'} of JSON type {actual} where the "
                f"server's schema declares {' or '.join(typed)}; the server would not run "
                "with the value the policy judged. Send the declared JSON type."
            )
        if keyword == "enum":
            detail = f"{'are' if plural else 'is'} not one of the values the server's schema lists"
        elif keyword == "const":
            detail = f"{'are' if plural else 'is'} not the value the server's schema fixes"
        elif keyword == "required":
            missing = [name for name in value if name not in error.instance]
            detail = (
                f"leave{'' if plural else 's'} out {_quoted(missing)}, which the server's "
                "schema requires"
            )
        elif keyword == "additionalProperties":
            detail = f"hold{'' if plural else 's'} a member the server's schema does not allow"
        elif keyword in ("anyOf", "oneOf"):
            detail = (
                f"match{'' if plural else 'es'} "
                f"{'none' if keyword == 'anyOf' else 'not exactly one'} of the alternatives "
                "the server's schema lists there"
            )
        else:
            detail = f"do{'' if plural else 'es'} not meet `{keyword}` in the server's schema"
        return f"{subject} {detail}; {strict}"

    def _validate(self, validator: Any, instance: Any) -> None:
        errors = sorted(
            validator.iter_errors(instance),
            key=lambda error: ([str(part) for part in error.absolute_path], str(error.validator)),
        )
        if errors:
            raise Refused(self._why(errors[0]))

    # --- text a server may read as JSON ---------------------------------------------------------

    def _types(self, schema: Any, seen: tuple = ()) -> frozenset[str]:
        """The JSON types a schema can allow, read from its ``type``, ``const`` and
        ``enum``, the schema its ``$ref`` names, its ``allOf`` members and its ``anyOf``
        and ``oneOf`` alternatives. A schema that says nothing allows every type."""
        if schema is False:
            return frozenset()
        if type(schema) is not dict or id(schema) in seen:
            return _EVERY_TYPE
        seen = (*seen, id(schema))
        if "$ref" in schema:
            named = self._types(_resolve(self.schema, schema["$ref"]), seen)
            if self._ref_alone:
                return named
        else:
            named = _EVERY_TYPE
        found = named
        declared = schema.get("type")
        if type(declared) in (str, list):
            found &= frozenset(_declared(declared))
        if "const" in schema:
            found &= {_JSON_TYPES[type(schema["const"])]}
        if type(schema.get("enum")) is list:
            found &= {_JSON_TYPES[type(choice)] for choice in schema["enum"]}
        for member in schema.get("allOf") or ():
            found &= self._types(member, seen)
        for keyword in ("anyOf", "oneOf"):
            alternatives = schema.get(keyword)
            if type(alternatives) is list:
                allowed: frozenset[str] = frozenset()
                for alternative in alternatives:
                    allowed |= self._types(alternative, seen)
                found &= allowed
        return found

    def _refuse_json_text(self, arguments: dict[str, Any]) -> None:
        """Refuse a string argument a server may run as another value.

        The MCP Python SDK's server decodes the text of a string argument as JSON when the
        parameter is not annotated exactly ``str``, and runs with what it decoded when
        that is null, a list or an object: ``"null"`` for a parameter typed
        ``Optional[str]`` runs as ``None``. The listed schema of such a parameter allows
        another JSON type than string, which is what the gate can see, so for a property
        whose schema does, a string that is JSON text for null, a list or an object is
        refused. Text for a boolean or a number that SDK keeps a string (``"true"`` runs
        as ``"true"``), so it is judged and sent as the string it is. A property whose
        schema allows only strings keeps every string."""
        properties = self.schema.get("properties")
        if type(properties) is not dict:
            return
        for name in sorted(name for name in arguments if name in properties):
            if self._decoded(properties[name], arguments[name]):
                raise Refused(
                    f"argument `{self._path((name,))}` is a string that holds JSON text "
                    "(null, a list or an object) where the server's schema allows "
                    "another JSON type than string; a server may read the text as JSON and "
                    "run with that value, not with the string the policy judged. Send the "
                    "value as JSON, not as text."
                )

    def _decoded(self, schema: Any, value: Any) -> bool:
        """Whether ``value``, sent for a top-level property with this schema, is a string
        a server may decode and run as another value: JSON text for null, a list or an
        object, where the schema allows another JSON type than string."""
        if type(value) is not str or self._types(schema) <= {"string"}:
            return False
        return _json_text(value)

    # --- defaults ------------------------------------------------------------------------------

    def _default(self, work: _Work, schema: Any, path: tuple, seen: tuple = ()) -> tuple[bool, Any]:
        """Whether the schema of a property the arguments leave out (at ``path``) lists a
        default, and the default.

        Its own ``default`` is the one. When it lists none itself, the defaults listed by
        the schema its ``$ref`` names, by the members of its ``allOf`` and by the
        alternatives of its ``anyOf`` and ``oneOf`` (each read the same way) must be one
        value, of the same JSON type all the way down; different ones are refused, since
        the gate cannot tell which the server would take."""
        if type(schema) is not dict or id(schema) in seen:
            return False, None
        work.step()
        if "default" in schema:
            return True, schema["default"]
        seen = (*seen, id(schema))
        inner: list[Any] = []
        if "$ref" in schema:
            inner.append(_resolve(self.schema, schema["$ref"]))
        for keyword in ("allOf", "anyOf", "oneOf"):
            members = schema.get(keyword)
            if type(members) is list:
                inner.extend(members)
        listed = [self._default(work, member, path, seen) for member in inner]
        found = [default for has_default, default in listed if has_default]
        if any(not _same(default, found[0]) for default in found[1:]):
            raise Refused(
                f"argument `{self._path(path)}` is left out, and the alternatives the "
                "server's schema lists for it list different defaults, so the gate cannot "
                "tell what the server would run with"
            )
        return (True, found[0]) if found else (False, None)

    def _join(self, found: dict, more: dict) -> None:
        """Add the defaults ``more`` to ``found``. Two parts of the schema that list
        different defaults for one place are refused."""
        for path, write in more.items():
            first = found.setdefault(path, write)
            if first is not write and not (_same(first[0], write[0]) and first[1] == write[1]):
                raise Refused(
                    f"argument `{self._path(path)}` is left out, and the server's schema "
                    "lists different defaults for it, so the gate cannot tell what the "
                    "server would run with"
                )

    def _writes(
        self, validator: Any, work: _Work, value: Any, schema: Any, path: tuple, held: dict
    ) -> dict[tuple, tuple[Any, bool]]:
        """The defaults ``schema`` lists for what ``value`` (at ``path`` in the arguments)
        leaves out: a map of each place to ``(the default, whether it can be sent)``.
        Nothing is written here.

        Followed: ``$ref``, ``allOf``, the alternative of ``anyOf`` and ``oneOf`` that
        validates, ``properties``, ``additionalProperties``, ``items`` and ``prefixItems``
        (in the older drafts a list of ``items`` and ``additionalItems``). The keywords
        beside a ``$ref`` are followed in every draft: a default that is written is also
        sent, so writing one the server would not have applied changes nothing it runs.

        Worked out once for a schema and a place while the arguments stand as they are:
        a recursive schema reaches the same place through every alternative that names it."""
        if type(schema) is not dict or type(value) not in (dict, list):
            return {}
        key = (id(schema), path)
        known = work.plans.get(key)
        if known is not None:
            return known
        work.step()
        found: dict[tuple, tuple[Any, bool]] = {}
        work.plans[key] = found
        if "$ref" in schema:
            target = _resolve(self.schema, schema["$ref"])
            self._join(found, self._writes(validator, work, value, target, path, held))
        for branch in schema.get("allOf") or ():
            self._join(found, self._writes(validator, work, value, branch, path, held))
        for keyword in ("anyOf", "oneOf"):
            branches = schema.get(keyword)
            if type(branches) is list:
                self._join(found, self._one_of(validator, work, value, branches, path, held))
        if type(value) is dict:
            self._join(found, self._members(validator, work, value, schema, path, held))
        else:
            self._join(found, self._items(validator, work, value, schema, path, held))
        return found

    def _one_of(
        self, validator: Any, work: _Work, value: Any, branches: list, path: tuple, held: dict
    ) -> dict[tuple, tuple[Any, bool]]:
        """The defaults of the alternative that validates ``value``. When more than one
        does, they must all list the same defaults, or the call is refused as ambiguous."""
        valid = [branch for branch in branches if not self._invalid(validator, value, branch)]
        plans = [self._writes(validator, work, value, branch, path, held) for branch in valid]
        for other in plans[1:]:
            if other.keys() != plans[0].keys() or any(
                not _same(write[0], plans[0][place][0]) or write[1] != plans[0][place][1]
                for place, write in other.items()
            ):
                where = self._path(path)
                raise Refused(
                    (f"argument `{where}` matches" if where else "the arguments match")
                    + " more than one of the alternatives the server's schema lists, and "
                    "they list different defaults, so the gate cannot tell what the server "
                    "would run with"
                )
        return plans[0] if plans else {}

    def _members(
        self, validator: Any, work: _Work, value: dict, schema: dict, path: tuple, held: dict
    ) -> dict[tuple, tuple[Any, bool]]:
        found: dict[tuple, tuple[Any, bool]] = {}
        listed = schema.get("properties")
        properties = listed if type(listed) is dict else {}
        for name, member in properties.items():
            here = (*path, name)
            if name in value:
                self._join(found, self._writes(validator, work, value[name], member, here, held))
                continue
            has_default, default = self._default(work, member, here)
            if not has_default or here in held:
                continue
            default = _detached(default)  # a copy: the schema was checked to be JSON data
            # the server runs with its default as listed, and would not if it were sent one
            # that is not valid there (it would reject it) or one that is text it decodes
            sendable = not self._invalid(validator, default, member) and not (
                not path and self._decoded(member, default)
            )
            self._join(found, {here: (default, sendable)})
        additional = schema.get("additionalProperties")
        if type(additional) is dict:
            patterns = schema.get("patternProperties")
            patterns = patterns if type(patterns) is dict else {}
            for name, member in value.items():
                if name not in properties and not any(re.search(p, name) for p in patterns):
                    here = (*path, name)
                    self._join(found, self._writes(validator, work, member, additional, here, held))
        return found

    def _items(
        self, validator: Any, work: _Work, value: list, schema: dict, path: tuple, held: dict
    ) -> dict[tuple, tuple[Any, bool]]:
        items = schema.get("items")
        if self._modern:
            prefix, rest = schema.get("prefixItems"), items
        elif type(items) is list:
            prefix, rest = items, schema.get("additionalItems")
        else:
            prefix, rest = None, items
        prefix = prefix if type(prefix) is list else []
        if any(self._default(work, member, path)[0] for member in prefix[len(value) :]):
            raise Refused(
                f"argument `{self._path(path)}` leaves out an item the server's schema lists "
                "a default for, which the gate cannot write in place"
            )
        found: dict[tuple, tuple[Any, bool]] = {}
        for index, item in enumerate(value):
            member = prefix[index] if index < len(prefix) else rest
            self._join(found, self._writes(validator, work, item, member, (*path, index), held))
        return found

    # --- the whole of it ------------------------------------------------------------------------

    def _complete(
        self, validator: Any, work: _Work, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[tuple, Any]]:
        """``(sent, held)``: the arguments with every sendable default written, and the
        defaults that cannot be sent, each under its place."""
        sent = copy.deepcopy(arguments)
        self._validate(validator, sent)
        self._refuse_json_text(sent)
        held: dict[tuple, Any] = {}
        for _ in range(_MAX_PASSES):
            writes = self._writes(validator, work, sent, self.schema, (), held)
            if not writes:
                break
            for path, (default, sendable) in writes.items():
                if not sendable:
                    held[path] = default
                    continue
                parent: Any = sent
                for part in path[:-1]:
                    parent = parent[part]
                parent[path[-1]] = default
            work.forget()
        else:
            raise Refused(f"{_SCHEMA} lists defaults that never settle")
        self._validate(validator, sent)
        return sent, held

    def complete(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(judged, sent)`` for a call's ``arguments``, which must be a detached copy of
        JSON data: what the server will run with and what it is sent. The two differ only
        by a listed default the server would not run with if it were sent (one that is not
        valid against its own property schema, or text the server would decode), which is
        judged and not sent. Raises :class:`Refused` for arguments that are not valid
        against the schema as they stand, for a string a server may read as JSON, for
        arguments too costly to check, and for a schema the gate cannot check or complete
        them against."""
        validator = self._prepared()
        work = _Work(_STEPS + _STEPS_PER_VALUE * _count(arguments))
        token = _WORK.set(work)
        try:
            sent, held = self._complete(validator, work, arguments)
        except Refused:
            raise
        except _TooCostly:
            raise Refused(_TOO_COSTLY) from None
        except RecursionError:
            raise Refused(
                f"the arguments are nested too deeply to check against {_SCHEMA} within "
                "the interpreter's recursion limit"
            ) from None
        except Exception as exc:  # noqa: BLE001, a schema the gate cannot apply is refused
            raise Refused(
                f"{_SCHEMA} could not be applied to the arguments ({type(exc).__name__})"
            ) from None
        finally:
            _WORK.reset(token)
        judged = copy.deepcopy(sent)
        for path, default in sorted(held.items(), key=lambda item: len(item[0])):
            parent: Any = judged
            for part in path[:-1]:
                parent = parent[part]
            parent[path[-1]] = default
        return judged, sent
