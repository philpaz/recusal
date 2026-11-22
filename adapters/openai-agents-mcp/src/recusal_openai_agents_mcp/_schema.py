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
   as the listed one (``1.0`` and ``true`` are not ``1``);
2. writes every default the schema lists for a property the arguments leave out, at every
   depth, repeated until nothing changes;
3. validates the completed arguments again.

Whatever it cannot check or complete is refused, with a reason that carries no text of the
model's: a schema that is not a JSON object or not a valid schema, a draft other than
draft-07, 2019-09 and 2020-12, a ``$ref`` that does not resolve inside the schema, a default
where the walk does not go, and a value for which the schema's alternatives would write
different defaults.

One listed default cannot be sent: one that is not valid against its own property schema
(``{"default": null, "type": "string"}``, which Pydantic publishes for ``q: str = None``).
The server runs with it when the parameter is left out and would reject it if it were
sent, so it is in the arguments the policy judges and not in the arguments that are sent.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable, Optional
from urllib.parse import unquote

from jsonschema import validators  # type: ignore[import-untyped]
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
)
#: Of those, the ones the walk that writes defaults does not go through.
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
)
#: References the gate does not resolve.
_DYNAMIC = ("$dynamicRef", "$recursiveRef")

#: How many times the defaults are written before the gate gives up on a fixed point, and
#: how many schemas deep one value is followed (a ``$ref`` that leads back to itself).
_MAX_PASSES = 16
_MAX_SCHEMA_DEPTH = 128

_SCHEMA = "the server's listed schema for this tool"


class Refused(Exception):
    """The arguments cannot be tied to what the server would run; ``reason`` says why."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


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


def _is_integer(checker: Any, instance: Any) -> bool:
    return type(instance) is int


def _strict(base: Any) -> Any:
    """``base`` with the three checks a server's conversion makes necessary."""
    return validators.extend(
        base,
        validators={"enum": _enum, "const": _const},
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


def _scan(node: Any, root: Any, names: set[str], under: Optional[str] = None) -> None:
    """Refuse a schema the gate cannot check or complete, and collect the property names
    it lists (the only argument names a refusal may quote)."""
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
    if "$ref" in node:
        _resolve(root, node["$ref"])
    if under is not None and "default" in node:
        raise Refused(
            f"{_SCHEMA} lists a default under `{under}`, where the gate cannot tell "
            "whether the server would apply it"
        )
    for keyword in (*_MAPS, "dependencies"):
        members = node.get(keyword)
        if type(members) is dict:
            if keyword == "properties":
                names.update(name for name in members if type(name) is str)
            inner = keyword if keyword in _UNWALKED else under
            for member in members.values():
                _scan(member, root, names, inner)
    for keyword in (*_LISTS, *_SINGLE):
        inner = keyword if keyword in _UNWALKED else under
        member = node.get(keyword)
        for each in member if type(member) is list else (member,):
            _scan(each, root, names, inner)


def _quoted(names: Iterable[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _declared(types: Any) -> list[str]:
    return [types] if type(types) is str else [name for name in types if type(name) is str]


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
        base = None
        if "$schema" not in schema:
            base = validators.Draft202012Validator
        elif type(schema["$schema"]) is str:
            base = validators.validator_for(schema, default=None)
        if base not in _DRAFTS:
            raise Refused(
                f"{_SCHEMA} declares a JSON Schema draft the gate does not read (it reads "
                "draft-07, 2019-09 and 2020-12)"
            )
        try:
            _scan(schema, schema, self._names)
            _DRAFTS[base].check_schema(schema)
        except SchemaError:
            raise Refused(f"{_SCHEMA} is not a valid JSON Schema") from None
        except RecursionError:
            raise Refused(f"{_SCHEMA} is nested too deeply to read") from None
        self._modern = base is validators.Draft202012Validator
        # an empty registry: nothing outside the schema is ever fetched
        return _DRAFTS[base](schema, registry=Registry())

    # --- validation -----------------------------------------------------------------------------

    def _invalid(self, validator: Any, instance: Any, schema: Any) -> bool:
        return next(iter(validator.descend(instance, schema)), None) is not None

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

    # --- defaults ------------------------------------------------------------------------------

    def _default(self, schema: Any, depth: int = 0) -> tuple[bool, Any]:
        """Whether a property's schema lists a default, and the default: its own, or the
        one the schema its ``$ref`` names lists."""
        if type(schema) is not dict or depth > _MAX_SCHEMA_DEPTH:
            return False, None
        if "default" in schema:
            return True, schema["default"]
        if "$ref" in schema:
            return self._default(_resolve(self.schema, schema["$ref"]), depth + 1)
        return False, None

    def _fill(
        self, validator: Any, value: Any, schema: Any, path: tuple, held: dict, depth: int = 0
    ) -> bool:
        """Write the defaults ``schema`` lists for what ``value`` leaves out, in place, and
        say whether anything was written. A default that could not be sent goes to
        ``held`` under its path in place of the arguments.

        Followed: ``$ref``, ``allOf``, the one alternative of ``anyOf`` and ``oneOf`` that
        validates, ``properties``, ``additionalProperties``, ``items`` and ``prefixItems``
        (in the older drafts a list of ``items`` and ``additionalItems``). The keywords
        beside a ``$ref`` are followed in every draft: a default that is written is also
        sent, so writing one the server would not have applied changes nothing it runs."""
        if type(schema) is not dict:
            return False
        if depth > _MAX_SCHEMA_DEPTH:
            raise Refused(f"{_SCHEMA} refers to itself without end")
        changed = False
        if "$ref" in schema:
            target = _resolve(self.schema, schema["$ref"])
            changed |= self._fill(validator, value, target, path, held, depth + 1)
        for branch in schema.get("allOf") or ():
            changed |= self._fill(validator, value, branch, path, held, depth + 1)
        for keyword in ("anyOf", "oneOf"):
            branches = schema.get(keyword)
            if type(branches) is list and type(value) in (dict, list):
                changed |= self._fill_one_of(validator, value, branches, path, held, depth)
        if type(value) is dict:
            changed |= self._fill_members(validator, value, schema, path, held, depth)
        elif type(value) is list:
            changed |= self._fill_items(validator, value, schema, path, held, depth)
        return changed

    def _fill_one_of(
        self, validator: Any, value: Any, branches: list, path: tuple, held: dict, depth: int
    ) -> bool:
        """Follow the alternative that validates ``value``. When more than one does, they
        must all write the same defaults, or the call is refused as ambiguous."""
        valid = [branch for branch in branches if not self._invalid(validator, value, branch)]
        if not valid:
            return False
        outcomes = []
        for branch in valid if len(valid) > 1 else ():
            trial, trial_held = copy.deepcopy(value), dict(held)
            self._fill(validator, trial, branch, path, trial_held, depth + 1)
            outcomes.append((trial, trial_held))
        for trial, trial_held in outcomes[1:]:
            first, first_held = outcomes[0]
            if not _same(trial, first) or not _same(trial_held, first_held):
                where = self._path(path)
                raise Refused(
                    (f"argument `{where}` matches" if where else "the arguments match")
                    + " more than one of the alternatives the server's schema lists, and "
                    "they list different defaults, so the gate cannot tell what the server "
                    "would run with"
                )
        return self._fill(validator, value, valid[0], path, held, depth + 1)

    def _fill_members(
        self, validator: Any, value: dict, schema: dict, path: tuple, held: dict, depth: int
    ) -> bool:
        changed = False
        listed = schema.get("properties")
        properties = listed if type(listed) is dict else {}
        for name, member in properties.items():
            here = (*path, name)
            if name in value:
                changed |= self._fill(validator, value[name], member, here, held, depth + 1)
                continue
            has_default, default = self._default(member)
            if not has_default or here in held:
                continue
            default = _detached(default)  # a copy: the schema was checked to be JSON data
            if self._invalid(validator, default, member):
                held[here] = default  # the server runs with it and would reject it if sent
            else:
                value[name] = default
            changed = True
        additional = schema.get("additionalProperties")
        if type(additional) is dict:
            patterns = schema.get("patternProperties")
            patterns = patterns if type(patterns) is dict else {}
            for name, member in value.items():
                if name not in properties and not any(re.search(p, name) for p in patterns):
                    changed |= self._fill(
                        validator, member, additional, (*path, name), held, depth + 1
                    )
        return changed

    def _fill_items(
        self, validator: Any, value: list, schema: dict, path: tuple, held: dict, depth: int
    ) -> bool:
        items = schema.get("items")
        if self._modern:
            prefix, rest = schema.get("prefixItems"), items
        elif type(items) is list:
            prefix, rest = items, schema.get("additionalItems")
        else:
            prefix, rest = None, items
        prefix = prefix if type(prefix) is list else []
        if any(self._default(member)[0] for member in prefix[len(value) :]):
            raise Refused(
                f"argument `{self._path(path)}` leaves out an item the server's schema lists "
                "a default for, which the gate cannot write in place"
            )
        changed = False
        for index, item in enumerate(value):
            member = prefix[index] if index < len(prefix) else rest
            changed |= self._fill(validator, item, member, (*path, index), held, depth + 1)
        return changed

    # --- the whole of it ------------------------------------------------------------------------

    def complete(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(judged, sent)`` for a call's ``arguments``, which must be a detached copy of
        JSON data: what the server will run with and what it is sent. The two differ only
        by a listed default that is not valid against its own property schema, which is
        judged and not sent. Raises :class:`Refused` for arguments that are not valid
        against the schema as they stand, and for a schema the gate cannot check or
        complete them against."""
        validator = self._prepared()
        try:
            self._validate(validator, arguments)
            sent = copy.deepcopy(arguments)
            held: dict[tuple, Any] = {}
            for _ in range(_MAX_PASSES):
                if not self._fill(validator, sent, self.schema, (), held):
                    break
            else:
                raise Refused(f"{_SCHEMA} lists defaults that never settle")
            self._validate(validator, sent)
        except Refused:
            raise
        except RecursionError:
            raise Refused(f"{_SCHEMA} refers to itself without end") from None
        except Exception as exc:  # noqa: BLE001, a schema the gate cannot apply is refused
            raise Refused(
                f"{_SCHEMA} could not be applied to the arguments ({type(exc).__name__})"
            ) from None
        judged = copy.deepcopy(sent)
        for path, default in sorted(held.items(), key=lambda item: len(item[0])):
            parent: Any = judged
            for part in path[:-1]:
                parent = parent[part]
            parent[path[-1]] = default
        return judged, sent
