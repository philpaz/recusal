"""A tool's listed input schema: strict validation of a call's arguments against all of it,
the two refusals that close what a schema cannot say about a server, and every default it
lists written out.

The gate sees what is sent to a server, and the tool's listed ``inputSchema`` is the only
statement of what the server then does with it. A server converts a value of another type
(the string ``"false"`` to a boolean), picks one of several object forms by rules of its
own, and fills what a call leaves out from its default, at any depth, after the policy
judged. So before the policy runs, :meth:`Listed.complete`

1. validates the arguments against the whole schema with ``jsonschema``, in the draft the
   schema declares (2020-12 when it declares none), formats not asserted. JSON Schema
   validation does not convert: a string is not a boolean. It is made stricter in three
   places where JSON Schema is looser than what a server then runs: a float is not an
   ``integer`` (``5.0``), an ``enum`` or ``const`` value must be of the same JSON type as
   the listed one (``1.0`` and ``true`` are not ``1``), and for ``uniqueItems`` two numbers
   that are equal are one item whatever their JSON type (``1``, ``1.0`` and ``true``, which
   a Python set holds once). A server's regular expression is never evaluated: ``pattern``
   is not asserted, and a member governed by ``patternProperties`` must meet the schema of
   every pattern listed there, since the gate does not work out which one matches its name;
2. walks the arguments along the schema and refuses, at every depth,

   * an object where the schema offers more than one object form (two or more alternatives
     of an ``anyOf`` or ``oneOf`` that admit an object), unless every form fixes one
     same-named property to a ``const`` of its own and the object carries that property
     with the value of exactly one of them; and a list where it offers more than one list
     form. Which form a server runs is decided by the server's own rules, which a schema
     does not state;
   * a string whose text a server reads as another type the schema also admits there: the
     text of a number or a word for a boolean where the schema names that type beside
     string, and, for an argument itself, JSON text for null, a list or an object where
     the schema admits another JSON type than string (the MCP Python SDK's server decodes
     such text);
3. writes every default the schema lists for a property the arguments leave out, at every
   depth, repeated (with step 2) until nothing changes;
4. validates the completed arguments again.

Where a ``default`` is read. For a property the arguments leave out: on the property's own
schema; or, when that lists none, on the schema its ``$ref`` names, on a member of its
``allOf`` and on an alternative of its ``anyOf`` or ``oneOf``, where all that are listed
must be one value. For an item a list leaves out (``prefixItems``) the call is refused. A
``default`` on a schema whose value is always present (the schema of the arguments
themselves, of the items of a list, of ``additionalProperties``) has nothing to be applied
to, since JSON has no way to send a member without a value. A ``default`` anywhere else,
under a keyword the walk does not write defaults through or under a keyword the gate does
not know, refuses the schema.

Whatever it cannot check or complete is refused, with a reason that carries no text of the
model's: a schema that is not a JSON object or not a valid schema, a draft other than
draft-07, 2019-09 and 2020-12, a ``$ref`` that does not resolve inside the schema, a schema
that refers to itself without a value in between, ``unevaluatedProperties`` together with
``patternProperties``, a default where the walk does not write one, a left-out property
whose alternatives list different defaults, a listed default with a member its own schema
does not allow, and arguments that cost more to check than the work the gate allows for
their size.

The work is bounded by counting it, never by a timer. A step is one keyword of the schema
applied to one value (and one more for each member or item the value holds directly), one
value visited by the walk, one value read for a comparison (``enum``, ``const``,
``uniqueItems``), or one part of the place of a validation error. Nothing is validated
twice for the same schema and the same value while the arguments stand as they are, a
value is walked once for a schema, and every step is counted against an allowance that
grows with the number of values in the arguments. What one step costs beyond that is
bounded by the size of the keyword in the listed schema, which the model does not choose.
What is read from the schema alone (the JSON types a schema admits, the default it lists,
the forms of a union) is worked out once when the listing is read.

Some listed defaults cannot be sent. One that is not valid against its own property schema
(``{"default": null, "type": "string"}``, which Pydantic publishes for ``q: str = None``),
and one that is text step 2 would refuse if a call sent it (``mode: Optional[str] =
"null"``). Both are in the arguments the policy judges, as listed, and not in the
arguments that are sent. A default that holds an object or a list where the schema offers
more than one form refuses the call that leaves its property out.
"""

from __future__ import annotations

import copy
import json
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
_NUMBERS = frozenset(("integer", "number"))

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
#: Of those, the ones the walk does not write defaults through. A default under one of
#: them refuses the schema.
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

#: The words Pydantic reads as a boolean in a string, whatever their case.
_BOOLEAN_WORDS = frozenset(
    ("true", "false", "1", "0", "yes", "no", "on", "off", "t", "f", "y", "n")
)

#: How many times the defaults are written before the gate gives up on a fixed point.
_MAX_PASSES = 16

#: The work one call may cost: this many steps, and this many more for every value in its
#: arguments (an object, a list, and each member, item and scalar in them). What a step is
#: the module's own text says. The largest ordinary calls measured (the tools of a server
#: built with the MCP Python SDK, the listed tools of the reference GitHub server, and a
#: recursive model nested to the 64-level limit) spend under a twentieth of this.
_STEPS = 10_000
_STEPS_PER_VALUE = 1_000

_SCHEMA = "the server's listed schema for this tool"

_TOO_COSTLY = f"the arguments are too costly to check against {_SCHEMA}"

#: What a schema lists as the default of a property: none, one value, or different ones.
_NONE, _ONE, _DIFFERENT = "none", "one", "different"


class Refused(Exception):
    """The arguments cannot be tied to what the server would run; ``reason`` says why."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Forms(Refused):
    """An object or a list stands where the schema offers more than one form for it."""


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
        #: (schema, value) the walk has been through, kept the same way
        self.walked: dict[tuple, Any] = {}
        #: a list or an object -> what it is compared by, kept the same way
        self.keys: dict[tuple, tuple] = {}
        #: the choices of an ``enum`` -> what each is compared by (the schema's own data)
        self.choices: dict[int, tuple] = {}
        #: a string -> what a server may read its text as (a string never changes)
        self.reads: dict[int, tuple[str, frozenset[str]]] = {}

    def step(self, count: int = 1) -> None:
        self.steps += count
        if self.steps > self.allowed:
            raise _TooCostly

    def forget(self) -> None:
        """The arguments were written to: nothing worked out for them stands."""
        self.valid.clear()
        self.answers.clear()
        self.walked.clear()
        self.keys.clear()


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


def _key(work: Optional[_Work], value: Any, exact: bool) -> Any:
    """What a value is compared by: two values have the same key when they are the same
    JSON data. ``exact`` keeps the JSON types apart, as :func:`_same` does; otherwise two
    numbers that are equal are the same whatever their type, a boolean among them. One
    step for every value read, and a list or an object is read once while the arguments
    stand as they are."""
    if work is not None:
        work.step()
    kind = type(value)
    if kind is not list and kind is not dict:
        if exact or kind is str or value is None:
            return (kind, value)
        return (float, value)
    if work is not None:
        known = work.keys.get((id(value), exact))
        if known is not None and known[0] is value:
            return known[1]
    key: Any
    if kind is list:
        key = (list, tuple(_key(work, item, exact) for item in value))
    else:
        key = (dict, frozenset((name, _key(work, each, exact)) for name, each in value.items()))
    if work is not None:
        work.keys[(id(value), exact)] = (value, key)
    return key


def _enum(validator: Any, choices: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """``enum`` by the JSON type too. The choices are read once for a call, so a long list
    of them costs a call its length once and not once for every value checked."""
    work = _WORK.get()
    known = None if work is None else work.choices.get(id(choices))
    if known is None or known[0] is not choices:
        known = (choices, frozenset(_key(work, choice, True) for choice in choices))
        if work is not None:
            work.choices[id(choices)] = known
    if _key(work, instance, True) not in known[1]:
        yield ValidationError("not one of the listed values")


def _const(validator: Any, fixed: Any, instance: Any, schema: Any) -> Iterable[Any]:
    work = _WORK.get()
    if _key(work, instance, True) != _key(work, fixed, True):
        yield ValidationError("not the fixed value")


def _unique_items(validator: Any, wanted: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """``uniqueItems`` in one reading of the list. Two numbers that are equal are the
    same item whatever their JSON type: a server that makes a set of them keeps one."""
    if wanted and type(instance) is list:
        work = _WORK.get()
        seen = set()
        for item in instance:
            key = _key(work, item, False)
            if key in seen:
                yield ValidationError("holds the same item twice")
                return
            seen.add(key)


def _not_evaluated(validator: Any, value: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """``pattern``: a server's regular expression is never run on text the model chose,
    which can take it longer than any allowance. A constraint the server checks and the
    gate does not, as a ``format`` is."""
    return ()


def _listed_properties(schema: Any) -> dict[str, Any]:
    properties = schema.get("properties")
    return properties if type(properties) is dict else {}


def _pattern_properties(validator: Any, patterns: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """``patternProperties`` without evaluating a pattern: the gate does not work out
    which pattern a member's name matches, so every member must meet the schema of every
    pattern listed. A member that does is valid whichever of them govern it."""
    if type(instance) is dict:
        for name, member in instance.items():
            for subschema in patterns.values():
                yield from validator.descend(member, subschema, path=name)


def _additional_properties(
    validator: Any, additional: Any, instance: Any, schema: Any
) -> Iterable[Any]:
    """``additionalProperties`` without evaluating a pattern. Beside ``patternProperties``
    the gate cannot tell which members are additional: every member ``properties`` does
    not name must meet the schema, and where that is ``false`` nothing is asserted (the
    server rejects a member no pattern matches)."""
    if type(instance) is not dict:
        return
    listed = _listed_properties(schema)
    extras = [name for name in instance if name not in listed]
    if schema.get("patternProperties"):
        if additional is False:
            return
    elif additional is False:
        if extras:
            yield ValidationError("holds a member that is not allowed")
        return
    for name in extras:
        yield from validator.descend(instance[name], additional, path=name)


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
    run again for a schema and a value it has already passed for. A keyword applied to a
    list or an object costs one step more for each item or member it holds directly."""

    def measured(validator: Any, value: Any, instance: Any, schema: Any) -> Iterable[Any]:
        work = _WORK.get()
        if work is None:
            yield from check(validator, value, instance, schema) or ()
            return
        work.step(1 + (len(instance) if type(instance) in (dict, list) else 0))
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
    """``base`` with the checks a server's conversion makes necessary, no regular
    expression evaluated, the comparisons read once, and every keyword counted."""
    keywords = {
        **base.VALIDATORS,
        "enum": _enum,
        "const": _const,
        "uniqueItems": _unique_items,
        "pattern": _not_evaluated,
        "patternProperties": _pattern_properties,
        "additionalProperties": _additional_properties,
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


class _Read:
    """What reading a schema found: the property names it lists (the only argument names
    a refusal may quote), its subschemas, the keywords it uses, and which of its
    subschemas lead back to themselves."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.nodes: list[dict] = []
        self.keywords: set[str] = set()
        self.state: dict[int, bool] = {}


def _scan(node: Any, root: Any, read: _Read, under: Optional[str] = None) -> None:
    """Refuse a schema the gate cannot check or complete, and collect what it holds.

    Every keyword of every subschema is one of four kinds, and a ``default`` is refused
    wherever it is not read: a keyword the walk that writes defaults follows (its
    subschemas are scanned in turn), one it does not write through (a ``default``
    anywhere under it is refused), one whose value is data (not looked into), and any
    other keyword, which the gate does not know: it is refused when its value holds a
    ``default``, so a keyword a later draft or a vendor adds cannot carry one past the
    gate. ``additionalProperties`` beside ``patternProperties`` is one the walk does not
    write through: without evaluating the patterns it cannot tell which members it
    governs."""
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
    if _endless(node, root, read.state):
        raise Refused(f"{_SCHEMA} refers to itself without end")
    if under is not None and "default" in node:
        raise Refused(
            f"{_SCHEMA} lists a default under `{under}`, where the gate cannot tell "
            "whether the server would apply it"
        )
    read.nodes.append(node)
    for keyword, member in node.items():
        inner = keyword if keyword in _UNWALKED else under
        if keyword == "additionalProperties" and node.get("patternProperties"):
            inner = "additionalProperties` beside `patternProperties"
        if keyword in _MAPS or keyword == "dependencies":
            if type(member) is dict:
                if keyword == "properties":
                    read.names.update(name for name in member if type(name) is str)
                if member:
                    read.keywords.add(keyword)
                for each in member.values():
                    _scan(each, root, read, inner)
        elif keyword in _LISTS or keyword in _SINGLE:
            read.keywords.add(keyword)
            for each in member if type(member) is list else (member,):
                _scan(each, root, read, inner)
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
    SDK's server decodes and runs in the string's place when it is an argument. Text that
    is not JSON is not."""
    try:
        decoded = json.loads(text)
    except (ValueError, RecursionError):
        return False
    return decoded is None or type(decoded) in (list, dict)


def _number_text(text: str) -> bool:
    """Whether Pydantic may read a string as a number. Wider than its own rule, never
    narrower: the text Python reads as a float once the white space around it is gone,
    with or without its underscores (Pydantic drops single underscores between any two
    characters, Python only between digits), and with or without a sign and the zeros
    that lead it (for an integer Pydantic drops leading zeros and then reads a sign:
    ``"0-5"`` is ``-5``)."""
    rest = text.lstrip("+-").lstrip("0_")
    for candidate in (text, text.replace("_", ""), rest, rest.replace("_", "")):
        try:
            float(candidate)
        except (ValueError, OverflowError):
            continue
        return True
    return False


def _reads(work: _Work, text: str) -> frozenset[str]:
    """What a server may read a string's text as, other than the string: ``number``,
    ``boolean`` (Pydantic's lax conversion, at any depth) and ``json`` (null, a list or
    an object, which the MCP Python SDK's server decodes for an argument). Worked out
    once for a string."""
    known = work.reads.get(id(text))
    if known is not None and known[0] is text:
        return known[1]
    bare = text.strip()
    found = set()
    if bare.lower() in _BOOLEAN_WORDS:
        found.add("boolean")
    if _number_text(bare):
        found.add("number")
    if _json_text(text):
        found.add("json")
    reads = frozenset(found)
    work.reads[id(text)] = (text, reads)
    return reads


class Listed:
    """One tool's input schema as a server listed it."""

    def __init__(self, schema: Any):
        self.schema = schema
        self._validator: Any = None
        self._failure: Optional[Refused] = None
        self._names: set[str] = set()
        #: what is read from the schema alone, each under the identity of its subschema
        #: (the schema is kept whole, so an identity is never given to another object)
        self._typed: dict[int, tuple[frozenset[str], frozenset[str]]] = {}
        self._defaults: dict[int, tuple[str, Any]] = {}
        self._unions: dict[tuple[int, str], tuple[list, list]] = {}

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
        self._modern = base is validators.Draft202012Validator
        # in draft-07 a `$ref` is read alone: the keywords beside it are not applied
        self._ref_alone = base is validators.Draft7Validator
        read = _Read()
        try:
            _scan(schema, schema, read)
            if {"unevaluatedProperties", "patternProperties"} <= read.keywords:
                raise Refused(
                    f"{_SCHEMA} uses `unevaluatedProperties` together with "
                    "`patternProperties`, which the gate could apply only by evaluating "
                    "the server's patterns"
                )
            # a pattern is never compiled, here or for a call: one that is valid ECMA 262
            # and not valid Python does not make a schema invalid
            formats = FormatChecker(())
            formats.checkers = {
                name: check
                for name, check in base.FORMAT_CHECKER.checkers.items()
                if name != "regex"
            }
            _DRAFTS[base].check_schema(schema, format_checker=formats)
            # what the schema alone says, read once and not for every call
            for node in read.nodes:
                self._types(node)
                self._listed(node)
                for keyword in ("anyOf", "oneOf"):
                    if type(node.get(keyword)) is list:
                        self._forms(node[keyword], "object")
                        self._forms(node[keyword], "array")
        except SchemaError:
            raise Refused(f"{_SCHEMA} is not a valid JSON Schema") from None
        except RecursionError:
            raise Refused(f"{_SCHEMA} is nested too deeply to read") from None
        self._names = read.names
        # an empty registry: nothing outside the schema is ever fetched
        return _DRAFTS[base](schema, registry=Registry())

    # --- what the schema alone says -------------------------------------------------------------

    def _types(self, schema: Any) -> tuple[frozenset[str], frozenset[str]]:
        """``(admitted, named)``: the JSON types a schema can admit, and the ones it names.

        Both are read from its ``type``, ``const`` and ``enum``, the schema its ``$ref``
        names, its ``allOf`` members and its ``anyOf`` and ``oneOf`` alternatives. A
        schema that says nothing admits every type and names none; ``number`` admits an
        integer; and a schema that is ``not`` anything (``{"not": {}}``, which zod
        publishes beside the schema of an optional member) admits none. Worked out once
        for a subschema: the scan has refused a schema that leads back to itself through
        these keywords, so the reading ends."""
        if schema is False:
            return frozenset(), frozenset()
        if type(schema) is not dict:
            return _EVERY_TYPE, frozenset()
        known = self._typed.get(id(schema))
        if known is not None:
            return known
        admitted: frozenset[str] = _EVERY_TYPE
        named: frozenset[str] = frozenset()
        alone = False
        if "$ref" in schema:
            admitted, named = self._types(_resolve(self.schema, schema["$ref"]))
            alone = self._ref_alone
        if not alone:
            said: list[frozenset[str]] = []
            declared = schema.get("type")
            if type(declared) in (str, list):
                listed = frozenset(_declared(declared))
                said.append(listed | {"integer"} if "number" in listed else listed)
            if "const" in schema:
                said.append(frozenset((_JSON_TYPES[type(schema["const"])],)))
            if type(schema.get("enum")) is list:
                said.append(frozenset(_JSON_TYPES[type(choice)] for choice in schema["enum"]))
            if "not" in schema and schema["not"] in ({}, True):
                admitted = frozenset()
            for member in schema.get("allOf") or ():
                inner, inner_named = self._types(member)
                admitted &= inner
                named |= inner_named
            for keyword in ("anyOf", "oneOf"):
                alternatives = schema.get(keyword)
                if type(alternatives) is list:
                    any_of: frozenset[str] = frozenset()
                    for alternative in alternatives:
                        inner, inner_named = self._types(alternative)
                        any_of |= inner
                        named |= inner_named
                    admitted &= any_of
            for each in said:
                admitted &= each
                named |= each
        self._typed[id(schema)] = (admitted, named)
        return admitted, named

    def _listed(self, schema: Any) -> tuple[str, Any]:
        """Whether a property's schema lists a default, and the default.

        Its own ``default`` is the one. When it lists none itself, the defaults listed by
        the schema its ``$ref`` names, by the members of its ``allOf`` and by the
        alternatives of its ``anyOf`` and ``oneOf`` (each read the same way) must be one
        value, of the same JSON type all the way down; different ones cannot be applied,
        since the gate cannot tell which the server would take. Worked out once for a
        subschema."""
        if type(schema) is not dict:
            return _NONE, None
        known = self._defaults.get(id(schema))
        if known is not None:
            return known
        answer: tuple[str, Any]
        if "default" in schema:
            answer = (_ONE, schema["default"])
        else:
            inner: list[Any] = []
            if "$ref" in schema:
                inner.append(_resolve(self.schema, schema["$ref"]))
            for keyword in ("allOf", "anyOf", "oneOf"):
                members = schema.get(keyword)
                if type(members) is list:
                    inner.extend(members)
            listed = [self._listed(member) for member in inner]
            found = [default for state, default in listed if state == _ONE]
            if any(state == _DIFFERENT for state, _ in listed) or any(
                not _same(default, found[0]) for default in found[1:]
            ):
                answer = (_DIFFERENT, None)
            else:
                answer = (_ONE, found[0]) if found else (_NONE, None)
        self._defaults[id(schema)] = answer
        return answer

    def _fixed(self, schema: Any, seen: set[int]) -> tuple[bool, Any]:
        """Whether a property's schema fixes one value (a ``const``, or an ``enum`` of
        one), on itself or on what its ``$ref`` and ``allOf`` name, and the value. Each
        subschema is read once (``seen``), however many ways lead to it."""
        if type(schema) is not dict or id(schema) in seen:
            return False, None
        seen.add(id(schema))
        inner: list[Any] = []
        if "$ref" in schema:
            inner.append(_resolve(self.schema, schema["$ref"]))
        if not (inner and self._ref_alone):
            if "const" in schema:
                return True, schema["const"]
            choices = schema.get("enum")
            if type(choices) is list and len(choices) == 1:
                return True, choices[0]
            inner.extend(schema.get("allOf") or ())
        for each in inner:
            fixed, value = self._fixed(each, seen)
            if fixed:
                return True, value
        return False, None

    def _tags(self, schema: Any, found: dict[str, Any], unfit: set[str], seen: set[int]) -> None:
        """Collect into ``found`` the properties one object form fixes to a value, read
        from its ``properties`` and those of what its ``$ref`` and ``allOf`` name. A
        property fixed to two different values is ``unfit`` to tell the form by."""
        if type(schema) is not dict or id(schema) in seen:
            return
        seen.add(id(schema))
        if "$ref" in schema:
            self._tags(_resolve(self.schema, schema["$ref"]), found, unfit, seen)
            if self._ref_alone:
                return
        for member in schema.get("allOf") or ():
            self._tags(member, found, unfit, seen)
        for name, member in _listed_properties(schema).items():
            fixed, value = self._fixed(member, set())
            if fixed:
                if name in found and not _same(found[name], value):
                    unfit.add(name)
                found.setdefault(name, value)

    def _forms(self, alternatives: list, wanted: str) -> tuple[list, list]:
        """``(forms, tags)`` for the alternatives of one ``anyOf`` or ``oneOf``: the ones
        that admit a value of JSON type ``wanted`` (``object`` or ``array``), and, for two
        or more object forms, each property all of them fix to a value of its own, with
        those values in the order of the forms. Worked out once for a subschema."""
        known = self._unions.get((id(alternatives), wanted))
        if known is not None:
            return known
        forms = [each for each in alternatives if wanted in self._types(each)[0]]
        tags: list[tuple[str, list]] = []
        if wanted == "object" and len(forms) > 1:
            fixed: list[dict[str, Any]] = []
            for form in forms:
                found: dict[str, Any] = {}
                unfit: set[str] = set()
                self._tags(form, found, unfit, set())
                fixed.append({name: tag for name, tag in found.items() if name not in unfit})
            for name in fixed[0]:
                if all(name in each for each in fixed):
                    values = [each[name] for each in fixed]
                    # apart as a server compares them: `1`, `1.0` and `true` are one tag
                    if len({_key(None, value, False) for value in values}) == len(values):
                        tags.append((name, values))
        self._unions[(id(alternatives), wanted)] = (forms, tags)
        return forms, tags

    # --- validation -----------------------------------------------------------------------------

    def _path(self, link: Any) -> str:
        """A place in the arguments, written only from the property names the schema lists
        and from positions in a list; a key the schema does not list is ``*``. ``link``
        is the place as the walk holds it (``(the place above, the last part)``, None for
        the arguments themselves) or the parts in order."""
        parts: list[Any] = []
        if type(link) is tuple:
            while link is not None:
                link, part = link
                parts.append(part)
            parts.reverse()
        elif link is not None:
            parts = list(link)
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
        path = self._path(list(error.absolute_path))
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
        work = _WORK.get()

        def place(error: Any) -> tuple[list[str], str]:
            parts = []
            for part in error.absolute_path:
                if work is not None:
                    work.step()
                parts.append(str(part))
            return parts, str(error.validator)

        first = min(validator.iter_errors(instance), key=place, default=None)
        if first is not None:
            raise Refused(self._why(first))

    # --- what a schema cannot say about a server ------------------------------------------------

    def _subject(self, link: Any) -> tuple[str, str]:
        path = self._path(link)
        return (f"argument `{path}`", "is") if path else ("the arguments", "are")

    def _text(self, work: _Work, text: str, schema: Any, link: Any) -> None:
        """Refuse a string where the schema also admits the type a server reads its text as.

        One rule for two readings. Pydantic's lax conversion reads the text of a number
        as the number and a word for a boolean as the boolean, at any depth, wherever the
        value is not taken as a string first: for a schema that names ``integer``,
        ``number`` or ``boolean`` beside string (``Union[int, date]`` lists an integer and
        a string with a format), the gate cannot tell. And the MCP Python SDK's server
        decodes an argument whose text is JSON for null, a list or an object when the
        parameter is not annotated exactly ``str``, which is a schema that admits another
        JSON type than string. A schema that admits only strings keeps every string, and
        one that names no type keeps the text of a number and of a boolean."""
        admitted, named = self._types(schema)
        reads = _reads(work, text)
        stated = admitted & named
        if "number" in reads and stated & _NUMBERS:
            held, allows, read = "the text of a number", "a number as well as a string", "a number"
        elif "boolean" in reads and "boolean" in stated:
            held, allows, read = (
                "a word for a boolean",
                "a boolean as well as a string",
                "a boolean",
            )
        elif "json" in reads and link[0] is None and not admitted <= {"string"}:
            held, allows, read = (
                "JSON text (null, a list or an object)",
                "another JSON type than string",
                "JSON",
            )
        else:
            return
        raise Refused(
            f"argument `{self._path(link)}` is a string that holds {held} where the "
            f"server's schema allows {allows}; a server may read the text as {read} and "
            "run with that value, not with the string the policy judged. Send the value "
            "as JSON, not as text."
        )

    def _chosen(self, value: Any, alternatives: list, link: Any) -> list:
        """The alternatives of an ``anyOf`` or ``oneOf`` to walk a list or an object
        along: the one form that admits it, or, for an object, the one of several whose
        tag it carries. More than one form with nothing to tell them apart is refused:
        a server built on Pydantic takes the member with more of the fields that are set,
        one built on zod the first that parses, dropping the members it does not know,
        and neither rule is in the schema."""
        forms, tags = self._forms(alternatives, "object" if type(value) is dict else "array")
        if len(forms) < 2:
            return forms
        subject, verb = self._subject(link)
        if type(value) is list:
            raise _Forms(
                f"{subject} {verb} a list, and the server's schema offers more than one "
                "list form there (alternatives of an `anyOf` or `oneOf`); the gate cannot "
                "tell which of them the server will run, so it cannot show the policy "
                "what runs. The server can publish one list form, or separate tools."
            )
        for name, values in tags:
            if name in value:
                hits = [form for form, tag in zip(forms, values) if _same(tag, value[name])]
                if len(hits) == 1:
                    return hits
        told = (
            f" The forms are told apart by {_quoted(name for name, _ in tags)}, and the "
            "object carries the value of none of them there."
            if tags
            else ""
        )
        raise _Forms(
            f"{subject} {verb} an object, and the server's schema offers more than one "
            "object form there (alternatives of an `anyOf` or `oneOf`); the gate cannot "
            "tell which of them the server will run, so it cannot show the policy what "
            f"runs.{told} The server can publish a discriminated union (every form "
            "fixing one same-named property to a `const` of its own, which the call "
            "then carries), or separate tools."
        )

    # --- the walk: the two refusals, and the defaults -------------------------------------------

    def _walk(
        self,
        validator: Any,
        work: _Work,
        value: Any,
        schema: Any,
        link: Any,
        found: Optional[dict],
        held: dict,
    ) -> None:
        """Walk ``value`` (at ``link`` in the arguments) along ``schema``: refuse what
        :meth:`_text` and :meth:`_chosen` refuse, and note in ``found`` the default of
        every property an object leaves out (``found`` None: refuse only, note nothing).
        Nothing is written here.

        Followed: ``$ref``, ``allOf``, the form of ``anyOf`` and ``oneOf`` that
        :meth:`_chosen` names, ``properties``, ``additionalProperties``, ``items`` and
        ``prefixItems`` (in the older drafts a list of ``items`` and ``additionalItems``).
        Followed for the refusals alone, whether or not they apply to the value, since no
        default may stand under them: ``then``, ``else``, ``dependentSchemas`` and the
        draft-07 ``dependencies``, ``patternProperties``, ``unevaluatedProperties``,
        ``contains`` and ``unevaluatedItems``. The keywords beside a ``$ref`` are
        followed in every draft: a default that is written is also sent, so writing one
        the server would not have applied changes nothing it runs.

        A string is read where it stands, against the schema of its place, and not again
        against each part of that schema. A list or an object is walked once for a schema
        while the arguments stand as they are: a recursive schema reaches the same place
        through every alternative that names it."""
        work.step()
        if type(schema) is not dict:
            return
        kind = type(value)
        if kind is str:
            self._text(work, value, schema, link)
            return
        if kind is not dict and kind is not list:
            return
        key = (id(schema), id(value))
        known = work.walked.get(key)
        if known is not None and known[0] is schema and known[1] is value:
            return
        work.walked[key] = (schema, value)
        onward: list[Any] = []
        if "$ref" in schema:
            onward.append(_resolve(self.schema, schema["$ref"]))
        onward.extend(schema.get("allOf") or ())
        for keyword in ("anyOf", "oneOf"):
            alternatives = schema.get(keyword)
            if type(alternatives) is list:
                onward.extend(self._chosen(value, alternatives, link))
        onward.extend(schema[keyword] for keyword in ("then", "else") if keyword in schema)
        for keyword in ("dependentSchemas", "dependencies"):
            members = schema.get(keyword)
            if type(members) is dict:
                onward.extend(members.values())
        for each in onward:
            self._walk(validator, work, value, each, link, found, held)
        if kind is dict:
            self._members(validator, work, value, schema, link, found, held)
        else:
            self._items(validator, work, value, schema, link, found, held)

    def _members(
        self,
        validator: Any,
        work: _Work,
        value: dict,
        schema: dict,
        link: Any,
        found: Optional[dict],
        held: dict,
    ) -> None:
        properties = _listed_properties(schema)
        for name, member in properties.items():
            if name in value:
                self._walk(validator, work, value[name], member, (link, name), found, held)
            elif found is not None:
                self._plan(validator, work, value, name, member, link, found, held)
        patterns = schema.get("patternProperties")
        every = list(patterns.values()) if type(patterns) is dict else []
        rest = [
            schema[keyword]
            for keyword in ("additionalProperties", "unevaluatedProperties")
            if type(schema.get(keyword)) is dict
        ]
        if every or rest:
            for name, member in value.items():
                # every pattern's schema, as validation asks, and what governs the
                # members `properties` does not name
                for each in every if name in properties else every + rest:
                    self._walk(validator, work, member, each, (link, name), found, held)

    def _plan(
        self,
        validator: Any,
        work: _Work,
        value: dict,
        name: str,
        member: Any,
        link: Any,
        found: dict,
        held: dict,
    ) -> None:
        """Note the default ``member`` lists for the property ``name`` that ``value`` (at
        ``link``) leaves out, and whether it can be sent."""
        state, default = self._listed(member)
        here = (link, name)
        if state == _DIFFERENT:
            raise Refused(
                f"argument `{self._path(here)}` is left out, and the alternatives the "
                "server's schema lists for it list different defaults, so the gate cannot "
                "tell what the server would run with"
            )
        if state == _NONE or (id(value), name) in held:
            return
        default = _detached(default)  # a copy: the schema was checked to be JSON data
        errors = list(validator.descend(default, member))
        if self._stray(validator, errors):
            raise Refused(
                f"argument `{self._path(here)}` is left out, and the default the server's "
                "schema lists for it has a member its own schema does not allow there (a "
                "Pydantic model whose fields have aliases lists its default under other "
                "names than its schema), so the gate cannot tell what the server would "
                "run with"
            )
        # the server runs with its default as listed, and would not if it were sent one
        # that is not valid there (it would reject it), or text it would read as another
        # type. Which form of several a server takes its default for is not in the schema
        sendable = not errors
        if sendable:
            try:
                self._walk(validator, work, default, member, here, None, held)
            except _Forms:
                raise Refused(
                    f"argument `{self._path(here)}` is left out, and the default the "
                    "server's schema lists for it holds an object or a list where the "
                    "schema offers more than one form (alternatives of an `anyOf` or "
                    "`oneOf`), so the gate cannot tell what the server would run with"
                ) from None
            except Refused:
                sendable = False
        first = found.get((id(value), name))
        if first is None:
            found[(id(value), name)] = (value, name, default, sendable)
        elif not (_same(first[2], default) and first[3] == sendable):
            raise Refused(
                f"argument `{self._path(here)}` is left out, and the server's schema "
                "lists different defaults for it, so the gate cannot tell what the "
                "server would run with"
            )

    def _stray(self, validator: Any, errors: Iterable[Any]) -> bool:
        """Whether a default fails its own schema by a member that schema does not
        allow, under an alternative too."""
        for error in errors:
            if error.validator in ("additionalProperties", "unevaluatedProperties"):
                return True
            if error.validator in ("anyOf", "oneOf") and any(
                self._stray(validator, validator.descend(error.instance, alternative))
                for alternative in error.validator_value
            ):
                return True
        return False

    def _items(
        self,
        validator: Any,
        work: _Work,
        value: list,
        schema: dict,
        link: Any,
        found: Optional[dict],
        held: dict,
    ) -> None:
        items = schema.get("items")
        if self._modern:
            prefix, rest = schema.get("prefixItems"), items
        elif type(items) is list:
            prefix, rest = items, schema.get("additionalItems")
        else:
            prefix, rest = None, items
        prefix = prefix if type(prefix) is list else []
        if found is not None and any(
            self._listed(member)[0] != _NONE for member in prefix[len(value) :]
        ):
            raise Refused(
                f"argument `{self._path(link)}` leaves out an item the server's schema lists "
                "a default for, which the gate cannot write in place"
            )
        others = [
            schema[keyword] for keyword in ("contains", "unevaluatedItems") if keyword in schema
        ]
        for index, item in enumerate(value):
            here = (link, index)
            member = prefix[index] if index < len(prefix) else rest
            self._walk(validator, work, item, member, here, found, held)
            for each in others:
                self._walk(validator, work, item, each, here, found, held)

    # --- the whole of it ------------------------------------------------------------------------

    def _complete(
        self, validator: Any, work: _Work, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(judged, sent)``: the arguments with every listed default, and with every
        default that can be sent."""
        # a copy in which no list or object stands in two places: the walk knows a place
        # by the object that is there
        sent = _detached(arguments)
        self._validate(validator, sent)
        held: dict[tuple, tuple] = {}
        for _ in range(_MAX_PASSES):
            found: dict[tuple, tuple] = {}
            self._walk(validator, work, sent, self.schema, None, found, held)
            if not found:
                break
            for key, (parent, name, default, sendable) in found.items():
                if sendable:
                    parent[name] = default
                else:
                    held[key] = (parent, name, default)
            work.forget()
        else:
            raise Refused(f"{_SCHEMA} lists defaults that never settle")
        self._validate(validator, sent)
        # a default that is not sent stands where its property is left out, so none
        # stands under another: each is put in for the copy and taken out again
        for parent, name, default in held.values():
            parent[name] = default
        judged = copy.deepcopy(sent)
        for parent, name, _ in held.values():
            del parent[name]
        return judged, sent

    def complete(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(judged, sent)`` for a call's ``arguments``, which must be JSON data: what
        the policy judges and what the server is sent. The two differ only by a listed
        default that cannot be sent (one that is not valid against its own property
        schema, or text a server would read as another type), which is judged as listed
        and not sent.
        Raises :class:`Refused` for arguments that are not valid against the schema as
        they stand, for an object or a list the schema offers more than one form for,
        for a string a server may read as another type the schema admits, for arguments
        too costly to check, and for a schema the gate cannot check or complete them
        against."""
        validator = self._prepared()
        work = _Work(_STEPS + _STEPS_PER_VALUE * _count(arguments))
        token = _WORK.set(work)
        try:
            return self._complete(validator, work, arguments)
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
