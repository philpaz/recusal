"""A tool's listed input schema: strict validation of a call's arguments against all of it,
the two refusals that close what a schema cannot say about a server, and every default it
lists written out.

The gate sees what is sent to a server, and the tool's listed ``inputSchema`` is the only
statement of what the server then does with it. A server converts a value of another type
(the string ``"false"`` to a boolean), picks one of several object forms by rules of its
own, and fills what a call leaves out from its default, at any depth, after the policy
judged. So before the policy runs, :meth:`Listed.complete`

1. validates the arguments against the whole schema, in the draft the schema declares
   (2020-12 when it declares none), formats not asserted. ``jsonschema`` applies the
   schema and resolves its references; each keyword is checked by a function of this
   module, which asserts what ``jsonschema`` 4.26 asserts for it and writes no message
   from the value (:data:`_UNMET`). JSON Schema
   validation does not convert: a string is not a boolean. It is made stricter in three
   places where JSON Schema is looser than what a server then runs: a float is not an
   ``integer`` (``5.0``), an ``enum`` or ``const`` value must be of the same JSON type as
   the listed one (``1.0`` and ``true`` are not ``1``), and for ``uniqueItems`` two numbers
   that are equal are one item whatever their JSON type (``1``, ``1.0`` and ``true``, which
   a Python set holds once). A server's regular expression is never evaluated: ``pattern``
   is not asserted, and a member governed by ``patternProperties`` must meet the schema of
   every pattern listed there, since the gate does not work out which one matches its name.
   All of that holds where a keyword adds a demand. Under the four keywords that turn a
   failure into a pass (``not``, the condition of an ``if``, ``oneOf``, ``maxContains``)
   a stricter reading would admit what the schema forbids and a keyword left to the
   server would decide the result, so what stands under them is read by two readings
   that bound every reading a server can make, and a call they answer differently for
   is refused as not decided (the three readings, below). What ``unevaluatedProperties``
   and ``unevaluatedItems`` count as evaluated is worked out here and is a bound for
   each reading too (:func:`_evaluated`): the branch of an ``if`` is not a demand that
   only adds, so where its condition is not decided the two branches are both read;
2. walks the arguments along the schema and refuses, at every depth,

   * an object where the schema offers more than one object form (two or more alternatives
     of an ``anyOf`` or ``oneOf`` that admit an object), unless every form fixes one
     same-named property to a ``const`` of its own and the object carries that property
     with the value of exactly one of them; and a list where it offers more than one list
     form. Which form a server runs is decided by the server's own rules, which a schema
     does not state;
   * a string whose text a server reads as another type the schema also admits there: the
     text of a number or a word for a boolean where the schema names that type beside
     string (a number as Pydantic reads one, and as JavaScript's ``Number()`` does, which
     is what a zod ``z.coerce.number()`` runs), and, for an argument itself, JSON text
     for null, a list or an object where the schema admits another JSON type than string
     (the MCP Python SDK's server decodes such text);
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
draft-07, 2019-09 and 2020-12, a subschema that declares a draft of its own, a ``$ref``
that does not resolve inside the schema (read by ``referencing``, as the validation
reads it: :class:`_References`; what a ``$ref`` names is read by these rules wherever in
the schema it stands, :meth:`Listed._prepare`), a schema
that refers to itself without a value in between, ``unevaluatedProperties`` together with
``patternProperties``, a default where the walk does not write one, a left-out property
whose alternatives list different defaults, a listed default with a member its own schema
does not allow, and arguments that cost more to check than the work the gate allows a
call.

The work is bounded by counting it, never by a timer. A step is one keyword of the schema
applied to one value (and one more for each member or item the value holds directly), one
value visited by the walk, one value read for a comparison (``enum``, ``const``,
``uniqueItems``), or one part of the place of a validation error; and for
``unevaluatedProperties`` and ``unevaluatedItems``, which are checked by
comparing every member or item with every other, the square of their number. What a
wide schema costs is counted too: a step for each subschema of an ``allOf``, ``anyOf``,
``oneOf`` or ``patternProperties`` that checks nothing (:func:`_silent`), for each name
read of a ``dependentRequired`` list, and for each value a default holds beyond the
first, each time it is written; and the names a schema lists are never read one by one
for an object (:func:`_both`, :func:`_lacks`, :meth:`Listed._left_out`). What the other
keywords evaluated is worked out once for a schema and a value, a step for each
subschema gone into that nothing else counted (:func:`_evaluated`), and a ``$ref`` has a
longest length (:data:`LONGEST_REFERENCE`), since a reference is looked up again each
time it is applied. Not counted: reading the listing itself, once, at the first call
after it. Nothing is
validated twice for the same schema and the same value while the arguments stand as they
are, a value is walked once for a schema, and every step is counted against an allowance
that grows with the number of values in the arguments up to a ceiling
(:data:`STEP_CEILING`, which an application can raise for a connection). What one step
costs beyond its count is bounded by the size of the keyword in the listed schema, which
the model does not choose, and by the length of the string or the size of the number it
reads; never by what the value holds below it, since no keyword writes a value out
(:data:`_UNMET`). That length is not bounded by the steps, and neither are the values in
an argument the schema does not name, which cost no step and which the gate still copies
and counts. So the arguments of one call have a ceiling on their size too
(:data:`SIZE_CEILING`, which an application can raise as well), checked before anything
else reads them (:func:`measure`). The two ceilings together are what bounds the time
one call can take: an ordinary schema spends between 4 and a few hundred steps on each
value, so without them the time grows with whatever the model sends. The same reading
refuses an integer of more than 4,300 digits (:data:`LONGEST_INTEGER`): writing one out
as text is quadratic in its digits where the interpreter sets no limit on them, which
no ceiling on the size bounds, so the gate writes no integer out and hands none of
more digits on. What the gate itself reads of the arguments before and after the check
(their size and depth, the copy, the count of their values, the keys they are compared
by, the text of a refusal) it reads without recursion: on Python 3.11 and later the
cost of a function that recurses moves with where in the interpreter's stack its
deepest calls fall, and the depth of nesting is the sender's to choose. The copy is the
kernel's (``recusal.invocation.detached``, which does not call itself from ``recusal``
0.16.0, the oldest this package installs with), so what is JSON data is said in one
place, with the one exception that function states: it tells a value's type by
comparing types, so a value of a class whose metaclass answers ``==`` for ``str`` is
handed back as it is and not copied, which no JSON decoder builds. The validation and
the walk recurse with the schema, as deep as the arguments nest under it, and are
refused where the interpreter's recursion limit ends them. The items
``uniqueItems`` compares are gathered into a set by keys that hold no number, only text:
the hash of a number is the sender's to choose, and the hash of a string is keyed by the
interpreter (:func:`_key`).
What is read from the schema alone (the JSON types a schema admits, the default it lists,
the forms of a union) is worked out once, at the first call after each listing.

Some listed defaults cannot be sent. One that is not valid against its own property schema
(``{"default": null, "type": "string"}``, which Pydantic publishes for ``q: str = None``),
and one that is text step 2 would refuse if a call sent it (``mode: Optional[str] =
"null"``). Both are in the arguments the policy judges, as listed, and not in the
arguments that are sent. A default that holds an object or a list where the schema offers
more than one form refuses the call that leaves its property out.
"""

from __future__ import annotations

import json
import operator
from asyncio import CancelledError
from bisect import bisect_right
from contextvars import ContextVar
from fractions import Fraction
from itertools import chain
from json.encoder import encode_basestring as _as_json
from typing import Any, Iterable, Iterator, Optional

from jsonschema import FormatChecker, validators  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError, ValidationError  # type: ignore[import-untyped]
from referencing import Registry
from referencing import jsonschema as _specifications

from recusal.invocation import detached

#: What is never turned into a refusal: the interpreter or the event loop stopping the
#: work. Every other exception that reaches the gate, one that is not an ``Exception``
#: among them (a panic in a compiled dependency arrives as a ``BaseException``), is a
#: refusal of the call.
PASSED_ON = (KeyboardInterrupt, SystemExit, GeneratorExit, CancelledError)

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

#: The white space JavaScript drops from the ends of a string before it reads a number,
#: and more: what Python's ``str.strip`` drops (which JavaScript drops too, but for
#: U+001C to U+001F and U+0085) and the byte order mark, which only JavaScript drops.
_SPACE = "".join(
    map(
        chr,
        (
            *range(0x09, 0x0E),
            *range(0x1C, 0x21),
            0x85,
            0xA0,
            0x1680,
            *range(0x2000, 0x200B),
            0x2028,
            0x2029,
            0x202F,
            0x205F,
            0x3000,
            0xFEFF,
        ),
    )
)

#: How many times the defaults are written before the gate gives up on a fixed point.
_MAX_PASSES = 16

#: The work one call may cost: this many steps, and this many more for every value in its
#: arguments (an object, a list, and each member, item and scalar in them), up to the
#: ceiling below. What a step is the module's own text says. The ordinary calls measured
#: (``tools/measure_mcp_cost.py``) spend from 4 steps for each value (a list of integers)
#: to 123 (a list of a discriminated union of ten models) and 323 (of thirty models): up
#: to a third of what a value adds.
_STEPS = 10_000
_STEPS_PER_VALUE = 1_000

#: The most steps one call may cost, however many values its arguments hold, unless the
#: application names another ceiling for a connection (``step_ceiling=`` on
#: ``gate_mcp_server``, ``gated_agent`` and ``audited_agent``). The steps allowed for
#: each value do not bound the work of a call, which grows with what the model sends;
#: this does, and with the ceiling on the size of the arguments (``SIZE_CEILING``, below)
#: it bounds the time.
STEP_CEILING = 200_000

_SCHEMA = "the server's listed schema for this tool"

_TOO_COSTLY = f"the arguments are too costly to check against {_SCHEMA}"


def ceiling_of(value: Any) -> int:
    """``value`` as a ceiling on the steps of one call: a whole number of at least 1."""
    if type(value) is not int or value < 1:
        raise ValueError(
            "Recusal: step_ceiling is the most steps the gate spends checking the "
            "arguments of one call, a whole number of at least 1 (the default is "
            f"{STEP_CEILING:,})"
        )
    return value


#: The most characters the arguments of one call may be, written as JSON text in its most
#: compact form (:func:`measure`), unless the application names another ceiling for a
#: connection (``size_ceiling=`` on ``gate_mcp_server``, ``gated_agent`` and
#: ``audited_agent``). The steps do not bound the size: a step that reads a string costs
#: the string's length, and a value in an argument the schema does not name costs no
#: step at all, while the gate copies and counts every value it is handed. This does,
#: and it is checked before anything else reads the arguments.
SIZE_CEILING = 1_000_000

#: A string is counted as JSON text this many characters at a time: each part is written
#: out once as JSON text (which is where an escape becomes two characters or six) and
#: measured, so what is written at any one time is at most six times this, whatever the
#: string and whatever the ceiling.
_WINDOW = 65_536

#: The most decimal digits an integer in the arguments of one call may have. It is the
#: limit Python sets by default on writing an integer as text (3.11 and later, and 3.10
#: from 3.10.7), which is quadratic in the digits before 3.12: an interpreter without the
#: limit (an older one, or one where it is switched off) would write an integer of any
#: length in the audit record's fingerprint and wherever a policy or a server writes it.
#: The gate refuses a longer one on every interpreter, whatever its limit is set to, and
#: writes none out itself (:func:`measure`).
LONGEST_INTEGER = 4_300

#: The most characters a ``$ref`` of a listed schema may have. ``jsonschema`` looks a
#: reference up again each time it applies one to a value, which costs in proportion to
#: its length and is counted as one step: a pointer of a million characters, applied to
#: forty thousand values, held the event loop a minute under both ceilings. An ordinary
#: reference (``#/$defs/Name``) is a few dozen characters.
LONGEST_REFERENCE = 1_024

#: The least integer of more digits than that, and its bits: no integer of fewer bits has
#: more digits, and every integer of more bits has.
_TOO_LONG = 10**LONGEST_INTEGER
_TOO_LONG_BITS = _TOO_LONG.bit_length()

#: The powers of ten up to 10**18: an integer below the last has as many digits as there
#: are powers here that are not above it, which one search finds.
_POWERS = tuple(10**exponent for exponent in range(19))
_BEYOND = _POWERS[-1]

#: What :func:`measure` finds.
FITS, OVER, LONG, DEEP = "", "size", "integer", "depth"


def size_ceiling_of(value: Any) -> int:
    """``value`` as a ceiling on the size of one call's arguments: a whole number of at
    least 1."""
    if type(value) is not int or value < 1:
        raise ValueError(
            "Recusal: size_ceiling is the most characters the arguments of one call may "
            "be as JSON text, a whole number of at least 1 (the default is "
            f"{SIZE_CEILING:,})"
        )
    return value


def measure(value: Any, ceiling: int, deepest: int = 64) -> str:
    """How ``value`` stands against what the gate checks before anything else reads the
    arguments of a call, in one reading that stops at the first of the two it refuses
    unread:

    * :data:`OVER`: it is more than ``ceiling`` characters as JSON text in its most
      compact form: no white space, and every character written as it is but for the
      ones JSON has to escape (what ``json.dumps(value, separators=(",", ":"),
      ensure_ascii=False)`` writes, counted in characters, of which one outside ASCII is
      more than one byte);
    * :data:`LONG`: it holds an integer of more than :data:`LONGEST_INTEGER` digits;
    * :data:`DEEP`: it is neither, and holds a value more than ``deepest`` levels down,
      which the gate refuses next;
    * :data:`FITS`: none of these.

    Nothing recurses, and the reading stops at the ceiling: a string longer than what is
    left is not read, and the members of a list or an object that holds more of them
    than characters are left are not looked at, so this costs in proportion to the
    ceiling whatever it is handed. A string that may fit is counted by writing it out
    as JSON text, :data:`_WINDOW` characters at a time. Two things the gate refuses next
    are not counted exactly. Whatever is not JSON data counts as one character. And
    what a list or an object more than ``deepest`` levels down holds is not read (the
    gate reads no deeper, and data that holds itself has no end).

    An integer is never written out. One of up to 18 digits is looked up among the
    powers of ten. The digits of a longer one are counted from its bits: an integer
    of ``b`` bits is at least ``2**(b - 1)`` and less than ``2**b``, and the logarithm
    of 2 lies between 0.30102 and 0.30103, so it has at least
    ``(b - 1) * 30102 // 100000 + 1`` digits and at most ``b * 30103 // 100000 + 1``.
    The second never undercounts, and for an integer of up to :data:`LONGEST_INTEGER`
    digits the two differ by one at the most; where they differ the integer is
    compared with the power of ten between them, so the count is exact. Whether an
    integer has more digits than that is read from its bits too, and from one
    comparison with ``10**LONGEST_INTEGER`` where its bits are that number's."""
    left = ceiling
    deep = False
    levels: list[Iterator[Any]] = [iter((value,))]
    while levels:
        for each in levels[-1]:
            kind = type(each)
            if kind is str:
                size = len(each)
                left -= size + 2
                if left < 0:
                    return OVER
                if size <= _WINDOW:
                    left -= len(_as_json(each)) - size - 2
                else:
                    for start in range(0, size, _WINDOW):
                        part = each[start : start + _WINDOW]
                        left -= len(_as_json(part)) - len(part) - 2
                        if left < 0:
                            return OVER
            elif kind is dict or kind is list:
                # the brackets, a comma between two members, a colon in each member of
                # an object; and every member or item is one character at the least
                count = len(each)
                left -= 2 + max(count - 1, 0) + (count if kind is dict else 0)
                if left < count:
                    return OVER
                if count:
                    if len(levels) > deepest:
                        deep = True  # its members are more than ``deepest`` levels down
                    if len(levels) <= deepest + 1:
                        levels.append(chain(each, each.values()) if kind is dict else iter(each))
                        break
            elif kind is int:
                if -10 < each < 10:
                    left -= 1 + (each < 0)
                elif -_BEYOND < each < _BEYOND:
                    left -= bisect_right(_POWERS, abs(each)) + (each < 0)
                else:
                    bits = each.bit_length()
                    if bits >= _TOO_LONG_BITS and (each >= _TOO_LONG or each <= -_TOO_LONG):
                        return LONG
                    digits = bits * 30103 // 100000 + 1
                    if digits != (bits - 1) * 30102 // 100000 + 1 and abs(each) < 10 ** (
                        digits - 1
                    ):
                        digits -= 1
                    left -= digits + (each < 0)
            elif kind is float:
                left -= len(repr(each))
            elif each is None or each is True:
                left -= 4
            elif each is False:
                left -= 5
            else:
                left -= 1
            if left < 0:
                return OVER
        else:
            levels.pop()
    return DEEP if deep else FITS


def too_large(value: Any, ceiling: int, deepest: int = 64) -> bool:
    """Whether ``value`` is more than ``ceiling`` characters as JSON text in its most
    compact form (:func:`measure`, which says how that is counted). False where the
    reading stopped before the ceiling at an integer of more than
    :data:`LONGEST_INTEGER` digits, which the gate refuses for that."""
    return measure(value, ceiling, deepest) == OVER


def too_large_reason(ceiling: int) -> str:
    """The reason for a call whose arguments are over the ceiling on their size."""
    return (
        f"the arguments are more than {ceiling:,} characters as JSON text, the ceiling "
        "the gate sets on the size of one call's arguments so that the time a call takes "
        "does not grow with what is sent; the gate has not read them. Send less in one "
        "call. Where calls of this size are expected, the application can raise the "
        "ceiling: `size_ceiling=` on `gate_mcp_server`, `gated_agent` or `audited_agent`."
    )


def long_integer_reason() -> str:
    """The reason for a call whose arguments hold an integer of more digits than the
    gate reads."""
    return (
        f"the arguments hold an integer of more than {LONGEST_INTEGER:,} digits, the most "
        "an integer in one call's arguments may have (the limit Python sets by default on "
        "writing an integer as text, which the gate applies on every interpreter); the "
        "gate has read no further. Send a number of fewer digits, or the digits as a "
        "string where the server's schema takes a string."
    )


def _too_costly(allowed: int, ceiling: int) -> str:
    """The reason for a call whose check was stopped at ``allowed`` steps: which of the
    two limits stopped it, and what the application can do about that one. A call the
    ceiling stopped may be over the steps allowed for its values too, which the gate
    cannot know without spending them, so that reason says when raising helps."""
    if allowed >= ceiling:
        return (
            f"{_TOO_COSTLY}: the check was stopped at {ceiling:,} steps, the ceiling the "
            "gate sets on the steps of one call. Send fewer values in one call. Where "
            "calls of this size are expected, the application can raise the ceiling: "
            "`step_ceiling=` on `gate_mcp_server`, `gated_agent` or `audited_agent`. "
            f"Raising it does not help a call that costs more than {_STEPS:,} steps and "
            f"{_STEPS_PER_VALUE:,} more for every value in its arguments, which the "
            "gate refuses under any ceiling."
        )
    return (
        f"{_TOO_COSTLY}: the check costs more than the {_STEPS:,} steps and "
        f"{_STEPS_PER_VALUE:,} more for every value in the arguments that the gate allows "
        "a call. That is what the schema costs for each value here, which raising the "
        "gate's ceiling on a call (`step_ceiling=`) does not change."
    )


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


class _NoFloat(Exception):
    """A ``multipleOf`` could not be checked: a number is too large to be a float.
    ``sent`` says whether that number is the argument's or the schema's."""

    def __init__(self, sent: bool):
        super().__init__()
        self.sent = sent


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
        #: what was met, in order, that two readings of the schema answer differently
        #: (:func:`_doubt`), and each keyword the gate could not decide for it
        #: (:func:`_undecided`). Each note is ``(the kinds of doubt, the keyword not
        #: decided or None)``. What is worked out once is kept with ONE note, the notes
        #: it made gathered (:func:`_gathered`), and adds that one when it is used
        #: again: the list grows by what is done, never by what a note stands for
        self.notes: list[tuple] = []
        #: what the listing worked out once for every call (:meth:`Listed._prepare`): the
        #: subschemas that hold no keyword the validation checks, and what each
        #: ``const`` and ``enum`` is compared by when the JSON types are not told apart
        self.silent: dict[int, int] = {}
        self.loose: dict[int, tuple] = {}
        #: (reading, items or members, schema, value) -> what the keywords of the
        #: schema evaluated of the value (:func:`_evaluated`), kept the same way
        self.evaluated: dict[tuple, tuple] = {}

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
        self.evaluated.clear()
        del self.notes[:]


#: The call being completed, which the keyword functions below count their steps against.
#: A context variable, so two calls in two threads or two tasks never share one.
_WORK: ContextVar[Optional[_Work]] = ContextVar("recusal_openai_agents_mcp_work", default=None)


def _count(value: Any) -> int:
    """How many values ``value`` holds, itself included. Without recursion, as every
    reading of the arguments that is the gate's own is (the module text says why); the
    data must not hold itself, which a copy never does."""
    count = 1
    held = [value]
    while held:
        each = held.pop()
        kind = type(each)
        if kind is dict:
            each = each.values()
        elif kind is not list:
            continue
        count += len(each)
        for inner in each:
            kind = type(inner)
            if kind is dict or kind is list:
                held.append(inner)
    return count


def _same(a: Any, b: Any) -> bool:
    """Equal, and of the same JSON type all the way down (``1`` is not ``True`` or ``1.0``)."""
    pairs = [(a, b)]
    while pairs:
        a, b = pairs.pop()
        kind = type(a)
        if kind is not type(b):
            return False
        if kind is list:
            if len(a) != len(b):
                return False
            pairs.extend(zip(a, b))
        elif kind is dict:
            if a.keys() != b.keys():
                return False
            pairs.extend((a[name], b[name]) for name in a)
        elif not a == b:
            return False
    return True


def _number(value: Any) -> str:
    """The text two numbers share exactly when they are equal, whatever their JSON type
    (``1``, ``1.0`` and ``true`` are ``0x1``): an integer, and a float or a boolean that
    equals one, in hexadecimal, and any other float as ``float.hex`` writes it. Both are
    written in proportion to the number's size."""
    if type(value) is float:
        if not value.is_integer():
            return value.hex()
        value = int(value)
    return hex(value)


def _key(work: Optional[_Work], value: Any, exact: bool) -> Any:
    """What a value is compared by: two values have the same key when they are the same
    JSON data. ``exact`` keeps the JSON types apart, as :func:`_same` does; otherwise two
    numbers that are equal are the same whatever their type, a boolean among them. One
    step for every value read, and a list or an object is read once while the arguments
    stand as they are.

    The keys that are not ``exact`` are the ones gathered into a set (``uniqueItems``),
    and a set costs one step for each member only while their hashes differ. The hash
    of a number is the sender's to choose: an integer below 2**61 - 1 hashes to itself
    and a larger one to its remainder, and the hash of a tuple is worked out from the
    hashes of its members with no secret in it. So integers, and lists of small
    integers, can be chosen to share one hash, and the set becomes a scan. Such a key
    therefore holds no number: a number is held as its text (:func:`_number`), and the
    hash of a string is keyed by the interpreter, as the hash of every member name is.
    An ``exact`` key is never gathered: it is looked up once among the values of the
    schema's own ``enum``, or compared with its ``const``.

    A list or an object is read without recursion (the module text says why): the key of
    each one inside it is finished before the one that holds it, as a recursion would."""
    #: for each list or object being read: itself, what is left of it, the keys of its
    #: members so far, and the name it has in the object that holds it
    levels: list[tuple[Any, Iterator[Any], list, Any]] = []
    members: Iterator[Any] = iter((value,))
    parts: list = []
    named = False
    while True:
        for each in members:
            if named:
                name, each = each
            if work is not None:
                work.step()
            kind = type(each)
            key: Any
            if kind is not list and kind is not dict:
                if exact or kind is str or each is None:
                    key = (kind, each)
                else:
                    key = (float, _number(each))
            else:
                known = None if work is None else work.keys.get((id(each), exact))
                if known is not None and known[0] is each:
                    key = known[1]
                else:
                    levels.append((each, members, parts, name if named else None))
                    named = kind is dict
                    members = iter(each.items()) if named else iter(each)
                    parts = []
                    break
            parts.append((name, key) if named else key)
        else:
            if not levels:
                return parts[0]
            source, members, above, name = levels.pop()
            key = (dict, frozenset(parts)) if named else (list, tuple(parts))
            if work is not None:
                work.keys[(id(source), exact)] = (source, key)
            parts = above
            named = bool(levels) and type(levels[-1][0]) is dict
            parts.append((name, key) if named else key)


#: The message of every validation error the gate makes. ``jsonschema`` writes the value
#: that failed into the message of each error (``f"{instance!r} is not of type ..."``)
#: when the error is made, for the errors nothing reads too: an ``anyOf`` asks each
#: alternative for its first error and drops it. That text is as long as the value and
#: everything under it, and under a schema that nests it is written again at every
#: level, which no step counts. The gate reads no message (:meth:`Listed._why` reads the
#: keyword, its value, the value's JSON type and the place), so no keyword it runs
#: writes one: every keyword is a function of this module that makes its error with
#: this text (:func:`_strict`), and so is the ``false`` schema (:func:`_own`).
_UNMET = "a keyword of the listed schema is not met"


def _unmet(**details: Any) -> Any:
    """A validation error, with nothing of the value in its message."""
    return ValidationError(_UNMET, **details)


# --- three readings of one schema -----------------------------------------------------------------
#
# The gate reads a schema more strictly than JSON Schema in three places (a float is not
# an ``integer``; ``enum`` and ``const`` by the JSON type; ``uniqueItems`` by equal
# numbers) and leaves three keywords to the server (``pattern``, ``format``, and which
# members a ``patternProperties`` governs). Where a keyword only ever adds a demand, a
# stricter reading refuses more and a keyword left out demands less, and both are what
# the page says. Four keywords turn a failure into a pass: ``not``, the condition of an
# ``if``, ``oneOf`` (a second alternative that is met makes the whole fail) and
# ``maxContains``. Under those the stricter reading ADMITS what the schema forbids
# (``{"not": {"const": 0}}`` let ``0.0`` through), and a keyword that is never
# evaluated decides the result (``{"not": {"pattern": ...}}`` refused every string).
#
# So what stands under one of the four is read twice, by two readings that bound every
# reading a server can make: ``_LEAST``, which is met only where every reading is met
# (the stricter three, and a keyword left to the server counted as not met), and
# ``_MOST``, which is met wherever any reading is (JSON Schema's own three, loosened
# once more so that ``true`` and ``1`` are equal as a server written in Python finds
# them, and a keyword left to the server counted as met). Each takes the other for what
# stands under a ``not`` of its own. Where the two agree, that is the answer. Where
# they differ, the gate cannot tell whether the keyword is met and refuses the call,
# saying which keyword and what the answer rests on. Everything else is read as before
# (``_GATE``).
_GATE, _LEAST, _MOST = "gate", "least", "most"


def _doubt(kind: str) -> None:
    """Note that something was met which the two bounding readings answer differently:
    ``number`` (a float that is a whole number, where ``integer`` is asked), ``equal``
    (values equal as numbers and of different JSON types), or the keyword left to the
    server (``pattern``, ``format``, ``patternProperties``)."""
    work = _WORK.get()
    if work is not None:
        work.notes.append(((kind,), None))


def _mark() -> int:
    """Where the notes of the call stand now."""
    work = _WORK.get()
    return len(work.notes) if work is not None else 0


def _settled(mark: int) -> bool:
    """Whether nothing was noted since ``mark``: what was read since is read the same
    by every reading. Never so outside a call, where nothing is noted."""
    work = _WORK.get()
    return work is not None and len(work.notes) == mark


def _kinds(mark: int) -> tuple[str, ...]:
    """What was noted since ``mark`` that the readings answer differently, each once."""
    work = _WORK.get()
    if work is None:
        return ()
    return tuple(dict.fromkeys(kind for note in work.notes[mark:] for kind in note[0]))


def _gathered(work: _Work, mark: int) -> tuple:
    """The notes made since ``mark`` as one note, put in their place: every kind of
    doubt among them, each once, and the last keyword that was not decided. Returned
    as what is kept with an answer that is worked out once: no note, or that one.

    A note stands for a doubt, and a reason needs to know which kinds of doubt there
    were and which keyword was not decided, not how many times each was met. Kept one
    by one, the notes of an answer were written into the list again every time the
    answer was used again, and kept again by whatever used it: definitions that each
    name the next one twice doubled them at every level (67 million notes and 4 GB
    for a schema of 1,920 characters and an argument of one character), and one
    definition of many doubts named many times multiplied them. Gathered, an answer
    used again adds one note, so the list is no longer than what was done to make
    it: a step, or one turn of a loop a step covers, for each note."""
    made = work.notes[mark:]
    if not made:
        return ()
    if len(made) > 1:
        kinds = tuple(dict.fromkeys(kind for note in made for kind in note[0]))
        last = next((note[1] for note in reversed(made) if note[1] is not None), None)
        work.notes[mark:] = [(kinds, last)]
    return (work.notes[mark],)


def _undecided(position: str, kinds: tuple[str, ...], of: str = "{}") -> Any:
    """The error of a keyword the gate could not decide: ``position`` is the keyword,
    and ``kinds`` what its answer rests on (:func:`_doubt`). ``of`` says what of the
    value the error is left at stands under that keyword, where it is not the value
    itself: the value's own subject goes where its ``{}`` is (``an item of {}``,
    ``a member name of {}``). It is noted, so
    that a keyword which drops the errors under it (an ``anyOf`` asks each alternative
    only whether it is met) can still say that one of them was not decided."""
    error = _unmet()
    error.recusal_undecided = (position, kinds, of)
    work = _WORK.get()
    if work is not None:
        work.notes.append(((), (position, kinds, of)))
    return error


def _failed(mark: int, of: str = "{}") -> Any:
    """The error of a keyword that is not met by what it asked of the schemas under it
    since ``mark``: its own, or, where one of those was not decided, that one's again,
    since "not met" would then say more than the gate knows. ``of`` is what of the
    value the keyword asked those schemas about, where that is not the value itself:
    the error is left at the value, and its reason must not say the value stands
    under a keyword that one of its items does. The last one noted is the one said:
    it is what the keyword asked last."""
    work = _WORK.get()
    if work is not None:
        for note in reversed(work.notes[mark:]):
            if note[1] is not None:
                position, kinds, inner = note[1]
                return _undecided(position, kinds, of.format(inner))
    return _unmet()


def _moot(work: _Work, mark: int) -> None:
    """Drop what was noted since ``mark`` as not decided: the keyword or the schema that
    asked is met all the same (another alternative of an ``anyOf`` was), so it is the
    cause of nothing, and a refusal further out must not name it. What the bounding
    readings noted stays."""
    if len(work.notes) > mark:
        work.notes[mark:] = [(note[0], None) for note in work.notes[mark:] if note[0]]


#: The keyword a value stands under that the gate could not decide, and what it could
#: not tell, as the reason says it (:meth:`Listed._why`).
_UNDECIDED = {
    "not": (
        "a `not` in the server's schema, and the gate cannot tell whether the schema "
        "under it is met, which is what a `not` forbids."
    ),
    "if": (
        "an `if` in the server's schema, and the gate cannot tell whether its condition "
        "is met, so it cannot tell whether `then` or `else` applies; arguments that "
        "meet both are sent, and these do not."
    ),
    "oneOf": (
        "a `oneOf` in the server's schema, which asks that exactly one alternative is "
        "met, and the gate cannot tell whether a second one is."
    ),
    "maxContains": (
        "a `maxContains` in the server's schema, and the gate cannot tell whether more "
        "items than it allows meet `contains`."
    ),
}

#: What an answer the gate could not decide rests on (:func:`_doubt`), as the reason
#: says it. The numbers written here are examples, never the argument's own.
_RESTS_ON = {
    "number": (
        "whether a whole number sent as a float (`5.0`) is an `integer`, which it is to "
        "JSON Schema and is not to the gate"
    ),
    "equal": (
        "whether a value is equal to one of another JSON type (`0.0` and `0`, `true` and "
        "`1`), which JSON Schema, the gate and a server do not all answer alike"
    ),
    "pattern": (
        "a `pattern`, which the gate does not evaluate (it never runs a server's "
        "regular expression on text the model chose)"
    ),
    "format": "a `format`, which the gate does not check",
    "patternProperties": (
        "`patternProperties`, whose patterns the gate does not evaluate, so it does not "
        "know which members they govern"
    ),
}

#: How a validator is built from another: the fields ``jsonschema`` itself carries over
#: when it descends (``evolve``), by attribute and by the name the class takes each as.
#: Read from the class as ``attrs`` describes it, which is where ``jsonschema`` reads
#: them from.
_CARRIED = [
    (field.name, field.alias)
    for field in validators.Draft202012Validator.__attrs_attrs__
    if field.init
]


def _as(validator: Any, reading: Any) -> Any:
    """``validator`` as a validator of another reading: the same schema, the same
    place in it for a ``$ref`` to be resolved from, another set of checks."""
    return reading(**{name: getattr(validator, attribute) for attribute, name in _CARRIED})


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
        yield _unmet()


def _const(validator: Any, fixed: Any, instance: Any, schema: Any) -> Iterable[Any]:
    work = _WORK.get()
    if _key(work, instance, True) != _key(work, fixed, True):
        yield _unmet()


def _unique_items(validator: Any, wanted: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """``uniqueItems`` in one reading of the list. Two numbers that are equal are the
    same item whatever their JSON type: a server that makes a set of them keeps one."""
    if wanted and type(instance) is list:
        work = _WORK.get()
        seen = set()
        for item in instance:
            key = _key(work, item, False)
            if key in seen:
                yield _unmet()
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
        _silent(patterns, len(instance))
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
            yield _unmet()
        return
    for name in extras:
        yield from validator.descend(instance[name], additional, path=name)


def _valid(validator: Any, instance: Any, schema: Any) -> bool:
    """Whether ``instance`` is valid against ``schema``, stopping at the first error."""
    return next(iter(validator.descend(instance, schema)), None) is None


# --- the keywords `jsonschema` checks the same way, written without its messages ----------------
#
# Each asserts what the function of `jsonschema` 4.26 for that keyword asserts, in the
# same order, and counts the same steps: only the message differs. A value is of a JSON
# type as the validator's type checker says (``validator.is_type``).


def _type(validator: Any, types: Any, instance: Any, schema: Any) -> Iterable[Any]:
    names = [types] if isinstance(types, str) else types
    if not any(validator.is_type(instance, name) for name in names):
        yield _unmet()


def _both(listed: Any, instance: dict) -> Iterable[Any]:
    """The names that are in a map of the schema and in an object of the arguments, read
    from whichever of the two holds fewer: a map of thousands of names costs an object
    of three members three lookups, which the step counted for each member covers. In
    the order of the schema where the object is not the smaller."""
    if len(instance) < len(listed):
        return [name for name in instance if name in listed]
    return [name for name in listed if name in instance]


def _properties(validator: Any, properties: Any, instance: Any, schema: Any) -> Iterable[Any]:
    if validator.is_type(instance, "object"):
        for name in _both(properties, instance):
            yield from validator.descend(
                instance[name], properties[name], path=name, schema_path=name
            )


def _lacks(needed: Any, instance: dict, counted: bool) -> bool:
    """Whether a list of names holds one the object does not have. It stops at the
    first, and the names of such a list are all different (the schema is checked for
    that), so it reads at most one name more than the object has members. That is
    covered by the step a keyword counts for each member where the keyword reads one
    list (``required``); where it reads a list for each of several members
    (``dependentRequired``) every name read is ``counted`` as a step."""
    work = _WORK.get() if counted else None
    for name in needed:
        if work is not None:
            work.step()
        if name not in instance:
            return True
    return False


def _required(validator: Any, required: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """``required``: one error, however many names are left out (the reason names all
    of them, :meth:`Listed._why`)."""
    if validator.is_type(instance, "object") and _lacks(required, instance, False):
        yield _unmet()


def _property_names(validator: Any, names: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """``propertyNames``: every member's name meets the schema. The error is this
    keyword's own, whatever keyword of that schema a name did not meet: the name is not
    a value with a place of its own in the arguments, so an error of ``enum`` or
    ``maxLength`` left at the object would say the object is not one of the listed
    values, or is too long. One that was not decided is passed on as it is."""
    if validator.is_type(instance, "object"):
        if names is False:
            if instance:
                yield _unmet()
            return
        failed = False
        for name in instance:
            for error in validator.descend(instance=name, schema=names):
                open_ = getattr(error, "recusal_undecided", None)
                if open_ is not None:
                    error.recusal_undecided = (open_[0], open_[1], "a member name of {}")
                    yield error
                    return
                failed = True
                break
            if failed:
                yield _unmet()
                return


def _dependent_required(
    validator: Any, dependents: Any, instance: Any, schema: Any
) -> Iterable[Any]:
    if validator.is_type(instance, "object"):
        for name in _both(dependents, instance):
            if _lacks(dependents[name], instance, True):
                yield _unmet()


def _dependent_schemas(
    validator: Any, dependents: Any, instance: Any, schema: Any
) -> Iterable[Any]:
    if validator.is_type(instance, "object"):
        for name in _both(dependents, instance):
            yield from validator.descend(instance, dependents[name], schema_path=name)


def _dependencies(validator: Any, dependencies: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """The ``dependencies`` of draft-07: a list of names, or a schema."""
    if not validator.is_type(instance, "object"):
        return
    for name in _both(dependencies, instance):
        dependency = dependencies[name]
        if validator.is_type(dependency, "array"):
            if _lacks(dependency, instance, True):
                yield _unmet()
        else:
            yield from validator.descend(instance, dependency, schema_path=name)


def _silent(listed: Any, times: int = 1) -> None:
    """Count what a keyword that holds many subschemas costs beyond its own step: one
    step for each of them that holds no keyword the validation checks (``{}``, ``true``,
    ``false``, a title alone), ``times`` over. Such a subschema counts no step of its
    own when the keyword descends into it, and a schema can list as many as it likes
    (an ``allOf`` of a thousand), so without this the steps would not count that work.
    How many of them a list or a map holds is worked out once for a listing."""
    work = _WORK.get()
    if work is not None:
        count = work.silent.get(id(listed))
        if count:
            work.step(count * times)


def _all_of(validator: Any, members: Any, instance: Any, schema: Any) -> Iterable[Any]:
    _silent(members)
    for index, subschema in enumerate(members):
        yield from validator.descend(instance, subschema, schema_path=index)


def _decide(
    validator: Any, sides: dict, instance: Any, schema: Any
) -> tuple[Optional[bool], tuple[str, ...]]:
    """Whether ``instance`` is valid against ``schema`` by both bounding readings:
    ``(the answer, ())`` where they agree, ``(None, what they differ over)`` where they
    do not. The reading that is met most is made first, and where it met nothing the
    two answer differently the other would do the same and is not made."""
    mark = _mark()
    most = _passes(_as(validator, sides[_MOST]), instance, schema)
    if _settled(mark):
        return most, ()
    if _passes(_as(validator, sides[_LEAST]), instance, schema) == most:
        return most, ()
    return None, _kinds(mark)


def _condition(validator: Any, sides: dict, instance: Any, condition: Any) -> Optional[bool]:
    """Whether the condition of an ``if`` is met, for the reading ``validator`` makes:
    True or False where both bounding readings answer alike, None where they do not.
    For the gate's own reading that is noted as an answer it could not decide."""
    if validator.RECUSAL_READING == _GATE:
        answer, kinds = _decide(validator, sides, instance, condition)
        if answer is None:
            work = _WORK.get()
            if work is not None:
                work.notes.append(((), ("if", kinds, "{}")))
        return answer
    mark = _mark()
    own = _passes(validator, instance, condition)
    if _settled(mark):
        return own
    other = _MOST if validator.RECUSAL_READING == _LEAST else _LEAST
    return own if _passes(_as(validator, sides[other]), instance, condition) == own else None


def _not(sides: dict) -> Any:
    """``not``, as the gate reads it: not met where every reading finds the schema
    under it met, met where none does, and not decided between the two."""

    def check(validator: Any, refused: Any, instance: Any, schema: Any) -> Iterable[Any]:
        answer, kinds = _decide(validator, sides, instance, refused)
        if answer is None:
            yield _undecided("not", kinds)
        elif answer:
            yield _unmet()

    return check


def _if(sides: dict) -> Any:
    """``if``, as the gate reads it. Where the condition is not decided the gate cannot
    tell which branch applies: arguments that meet both branches are valid either way,
    and any others are refused as not decided, never for what one branch asks."""

    def check(validator: Any, condition: Any, instance: Any, schema: Any) -> Iterable[Any]:
        mark = _mark()
        answer = _condition(validator, sides, instance, condition)
        if answer is None:
            if not all(
                _valid(validator, instance, schema[branch])
                for branch in ("then", "else")
                if branch in schema
            ):
                yield _failed(mark)
        elif answer:
            if "then" in schema:
                yield from validator.descend(instance, schema["then"], schema_path="then")
        elif "else" in schema:
            yield from validator.descend(instance, schema["else"], schema_path="else")

    return check


def _not_bound(reading: str, sides: dict) -> Any:
    """``not`` for a bounding reading: met where the other one finds the schema under it
    not met."""
    other = _MOST if reading == _LEAST else _LEAST

    def check(validator: Any, refused: Any, instance: Any, schema: Any) -> Iterable[Any]:
        if _passes(_as(validator, sides[other]), instance, refused):
            yield _unmet()

    return check


def _if_bound(reading: str, sides: dict) -> Any:
    """``if`` for a bounding reading. Where both readings answer the condition alike
    that branch applies. Where they do not, the reading that is met least asks for both
    branches and the one that is met most for either."""

    def check(validator: Any, condition: Any, instance: Any, schema: Any) -> Iterable[Any]:
        own = _condition(validator, sides, instance, condition)
        if own is None:
            met = [
                _valid(validator, instance, schema[branch])
                for branch in ("then", "else")
                if branch in schema
            ]
            if not (all(met) if reading == _LEAST else any(met) or len(met) < 2):
                yield _unmet()
            return
        if own:
            if "then" in schema:
                yield from validator.descend(instance, schema["then"], schema_path="then")
        elif "else" in schema:
            yield from validator.descend(instance, schema["else"], schema_path="else")

    return check


def _prefix_items(validator: Any, prefix: Any, instance: Any, schema: Any) -> Iterable[Any]:
    if validator.is_type(instance, "array"):
        for (index, item), subschema in zip(enumerate(instance), prefix):
            yield from validator.descend(
                instance=item, schema=subschema, schema_path=index, path=index
            )


def _items(validator: Any, items: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """The ``items`` of 2020-12: the schema of every item after ``prefixItems``."""
    if not validator.is_type(instance, "array"):
        return
    prefix = len(schema.get("prefixItems", []))
    total = len(instance)
    if total <= prefix:
        return
    if items is False:
        yield _unmet()
        return
    for index in range(prefix, total):
        yield from validator.descend(instance=instance[index], schema=items, path=index)


def _items_or_list(validator: Any, items: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """The ``items`` of draft-07 and 2019-09: one schema for every item, or a list of
    schemas for the first ones."""
    if not validator.is_type(instance, "array"):
        return
    if validator.is_type(items, "array"):
        for (index, item), subschema in zip(enumerate(instance), items):
            yield from validator.descend(item, subschema, path=index, schema_path=index)
    else:
        for index, item in enumerate(instance):
            yield from validator.descend(item, items, path=index)


def _additional_items(validator: Any, additional: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """The ``additionalItems`` of draft-07 and 2019-09: the items after a list of
    ``items``."""
    if not validator.is_type(instance, "array") or validator.is_type(
        schema.get("items", {}), "object"
    ):
        return
    listed = len(schema.get("items", []))
    if validator.is_type(additional, "object"):
        for index, item in enumerate(instance[listed:], start=listed):
            yield from validator.descend(item, additional, path=index)
    elif not additional and len(instance) > listed:
        yield _unmet()


def _contains(reading: str, sides: dict) -> Any:
    """The ``contains`` of 2019-09 and 2020-12, with ``minContains`` and ``maxContains``.

    How many items are enough is asked of the reading itself: one more that is met
    only helps. How many are too many (``maxContains``) is where an item that is met
    makes the whole fail, so it is asked of the reading that finds the most met (for
    the reading that is met least) or the least (for the one that is met most), and
    the gate's own reading is not met where even the least count is over, met where
    the most is not, and not decided between the two."""

    def count(validator: Any, items: Any, contains: Any) -> int:
        return sum(1 for each in items if _valid(validator, each, contains))

    def check(validator: Any, contains: Any, instance: Any, schema: Any) -> Iterable[Any]:
        if not validator.is_type(instance, "array"):
            return
        least = schema.get("minContains", 1)
        mark = _mark()
        matches = count(validator, instance, contains)
        if "maxContains" in schema:
            most = schema["maxContains"]
            if reading == _GATE:
                beside = _mark()
                upper = count(_as(validator, sides[_MOST]), instance, contains)
                if upper > most:
                    if _settled(beside) or (
                        count(_as(validator, sides[_LEAST]), instance, contains) > most
                    ):
                        yield _unmet(validator="maxContains", validator_value=most)
                    else:
                        error = _undecided("maxContains", _kinds(beside), "the items of {}")
                        error.validator, error.validator_value = "maxContains", most
                        yield error
                    return
            else:
                other = matches
                if not _settled(mark):
                    turned = sides[_MOST if reading == _LEAST else _LEAST]
                    other = count(_as(validator, turned), instance, contains)
                if other > most:
                    yield _unmet(validator="maxContains", validator_value=most)
                    return
        if matches < least:
            if not matches:
                yield _failed(mark, "an item of {}")
            else:
                yield _unmet(validator="minContains", validator_value=least)

    return check


def _contains_one(validator: Any, contains: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """The ``contains`` of draft-07: at least one item is valid."""
    mark = _mark()
    if validator.is_type(instance, "array") and not any(
        _valid(validator, each, contains) for each in instance
    ):
        yield _failed(mark, "an item of {}")


def _sized(kind: str, beyond: Any) -> Any:
    """``minLength``, ``maxItems`` and the like: how many characters, items or members a
    value of JSON type ``kind`` holds, beside the listed number."""

    def sized(validator: Any, limit: Any, instance: Any, schema: Any) -> Iterable[Any]:
        if validator.is_type(instance, kind) and beyond(len(instance), limit):
            yield _unmet()

    return sized


def _bounded(beyond: Any) -> Any:
    """``minimum``, ``maximum`` and their exclusive forms: a number beside the listed one."""

    def bounded(validator: Any, limit: Any, instance: Any, schema: Any) -> Iterable[Any]:
        if validator.is_type(instance, "number") and beyond(instance, limit):
            yield _unmet()

    return bounded


def _multiple_of(validator: Any, factor: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """``multipleOf``, which divides as floats when the listed number is a float, and as
    fractions where the quotient is too large to be an integer. An integer of more than
    about 308 digits has no float, and Python raises ``OverflowError`` there: for an
    argument that large beside a ``multipleOf`` that is a fraction, or for a
    ``multipleOf`` that large beside an argument that is a fraction. Which of the two it
    was is kept for the reason."""
    if not validator.is_type(instance, "number"):
        return
    try:
        if isinstance(factor, float):
            quotient = instance / factor
            try:
                failed = int(quotient) != quotient
            except OverflowError:
                failed = (Fraction(instance) / Fraction(factor)).denominator != 1
        else:
            failed = bool(instance % factor)
    except OverflowError:
        raise _NoFloat(type(instance) is int and type(factor) is float) from None
    if failed:
        yield _unmet()


_ITEMS_NOT_READ = (
    f"{_SCHEMA} uses `unevaluatedItems` beside an `items` that is `true` or `false` in a "
    "2019-09 schema, which the gate does not read (`jsonschema` raises on it)"
)


def _evaluated(
    validator: Any, sides: dict, modern: bool, items: bool, instance: Any, schema: Any, memo: dict
) -> frozenset:
    """The members of an object (``items`` False: their names) or the items of a list
    (their positions) that the keywords of ``schema`` evaluated, as ``jsonschema`` 4.26
    gathers them for ``unevaluatedProperties`` and ``unevaluatedItems`` in 2019-09
    and in 2020-12 (``modern``), keyword for keyword, with three differences.

    It is a bound. ``jsonschema`` takes the branch of an ``if`` by its own answer to
    the condition, and a branch is not a demand that only adds: a reading that finds
    the condition met where another does not reads other members as evaluated, so
    neither is within the other. Here, where the two bounding readings answer the
    condition differently (:func:`_condition`), the reading that is met most counts
    what either branch evaluated, and the other two only what both did. Everything
    else that is asked of the schemas under it (which alternatives are met, which
    members meet ``additionalProperties``) is asked of the reading itself, which only
    ever evaluates more for the reading that is met more.

    It is worked out once for a schema and a value while the arguments stand as they
    are (``memo``, the call's own where there is a call). ``jsonschema`` follows a
    ``$ref``, a ``dependentSchemas`` and the branches of an ``if`` again each time
    one is named, with no validation in between, so definitions that each name the
    next one twice were read two to the power of their number of times.

    And it counts: a step for each subschema it goes into that nothing else counted
    (what a ``$ref`` names, a member of ``dependentSchemas``, a condition and its
    branches; an alternative that checks nothing, :func:`_silent`), and with it a
    step for each member or item of the value, since what is found evaluated there
    is kept for the call. The alternatives
    themselves are validated, which counts their keywords, and the members and items
    are covered by the square the keyword counts first (:func:`_squared`).

    No pattern is evaluated: a schema that uses ``unevaluatedProperties`` together with
    ``patternProperties`` is refused when it is read. Should one reach here, the
    members a pattern governs are every member for the reading that is met most and
    none for the others."""
    if type(schema) is not dict:
        return frozenset()
    work = _WORK.get()
    reading = validator.RECUSAL_READING
    key = (reading, items, id(schema), id(instance))
    kept = memo if work is None else work.evaluated
    known = kept.get(key)
    if known is not None and known[0] is schema and known[1] is instance:
        if work is not None:
            work.notes.extend(known[3])
        return frozenset(known[2])
    mark = _mark()
    found: set = set()

    def within(subschema: Any, using: Any = validator) -> frozenset:
        if work is not None:
            # one step, and one for each member or item: what it finds evaluated is
            # kept for the call, and can be as large as the value
            work.step(1 + len(instance))
        return _evaluated(using, sides, modern, items, instance, subschema, memo)

    ref = schema.get("$ref")
    if ref is not None:
        resolved = validator._resolver.lookup(ref)
        found |= within(
            resolved.contents,
            validator.evolve(schema=resolved.contents, _resolver=resolved.resolver),
        )
    if items:
        everything = range(len(instance))
        if "items" in schema:
            listed = schema["items"]
            if modern or "additionalItems" in schema or type(listed) is dict:
                found.update(everything)
            elif type(listed) is list:
                found.update(range(min(len(listed), len(instance))))
            else:
                raise Refused(_ITEMS_NOT_READ)
        if modern and type(schema.get("prefixItems")) is list:
            found.update(range(min(len(schema["prefixItems"]), len(instance))))
        for keyword in ("contains", "unevaluatedItems"):
            if keyword in schema:
                for index, item in enumerate(instance):
                    if _valid(validator, item, schema[keyword]):
                        found.add(index)
    else:
        for keyword in ("properties", "additionalProperties", "unevaluatedProperties"):
            if keyword not in schema:
                continue
            listed = schema[keyword]
            if modern and keyword != "properties":
                # the members whose value meets it
                for name, member in instance.items():
                    if _valid(validator, member, listed):
                        found.add(name)
            elif type(listed) is dict:
                # the names it lists (for 2019-09 `jsonschema` reads the other two
                # keywords this way too, by the names of their schema's own keywords)
                found.update(_both(listed, instance))
            elif listed is True and not modern:
                found.update(instance)
        if schema.get("patternProperties") and reading == _MOST:
            found.update(instance)
        dependents = schema.get("dependentSchemas")
        if type(dependents) is dict:
            for name in _both(dependents, instance):
                found |= within(dependents[name])
    for keyword in ("allOf", "oneOf", "anyOf"):
        members = schema.get(keyword)
        if type(members) is list:
            _silent(members)
            for subschema in members:
                if _valid(validator, instance, subschema):
                    found |= _evaluated(validator, sides, modern, items, instance, subschema, memo)
    if "if" in schema:
        if work is not None:
            work.step()
        answer = _condition(validator, sides, instance, schema["if"])
        met: frozenset = frozenset()
        unmet: frozenset = frozenset()
        if answer is not False:
            met = within(schema["if"])
            if "then" in schema:
                met |= within(schema["then"])
        if answer is not True and "else" in schema:
            unmet = within(schema["else"])
        if answer is None and reading != _MOST:
            found |= met & unmet
        else:
            found |= met | unmet
    result = frozenset(found)
    kept[key] = (schema, instance, result, _gathered(work, mark) if work is not None else ())
    return result


def _unevaluated_items(sides: dict, modern: bool) -> Any:
    """``unevaluatedItems``: no item is left that no other keyword evaluated
    (:func:`_evaluated`)."""

    def unevaluated_items(validator: Any, value: Any, instance: Any, schema: Any) -> Iterable[Any]:
        if not validator.is_type(instance, "array"):
            return
        mark = _mark()
        known = _evaluated(validator, sides, modern, True, instance, schema, {})
        if len(known) < len(instance):
            yield _failed(mark, "{}, or an item of it,")

    return unevaluated_items


def _unevaluated_properties(sides: dict, modern: bool) -> Any:
    """``unevaluatedProperties``: every member no other keyword evaluated meets the
    schema (:func:`_evaluated`)."""

    def unevaluated_properties(
        validator: Any, value: Any, instance: Any, schema: Any
    ) -> Iterable[Any]:
        if not validator.is_type(instance, "object"):
            return
        mark = _mark()
        known = _evaluated(validator, sides, modern, False, instance, schema, {})
        failed = False
        for name in instance:
            if name not in known:
                for _ in validator.descend(instance[name], value, path=name, schema_path=name):
                    failed = True
        if failed:
            yield _failed(mark, "{}, or a member of it,")

    return unevaluated_properties


def _squared(check: Any, kind: Any) -> Any:
    """``check``, an ``unevaluatedProperties`` or ``unevaluatedItems``, counted for what it
    costs: the square of the members or items of the value, before it runs, so that a
    value too large for the allowance is refused without the reading. Every keyword of
    the schema is asked about every member (which of them meet ``additionalProperties``,
    which a ``contains`` is met by), and that is counted here once and not for each
    keyword; it is also what the keyword cost when ``jsonschema`` gathered what was
    evaluated into a list and looked every member up in it."""

    def squared(validator: Any, value: Any, instance: Any, schema: Any) -> Iterable[Any]:
        work = _WORK.get()
        if work is not None and type(instance) is kind:
            work.step(len(instance) ** 2)
        return check(validator, value, instance, schema) or ()

    return squared


def _unread(keyword: str) -> Any:
    """A keyword ``jsonschema`` checks and this module has no function for: a schema
    that uses it is refused when a call reaches it. None of ``jsonschema`` 4.26 is
    left to this but the two references the scan refuses first; a later ``jsonschema``
    that adds a keyword gets this, and never a function that writes a message."""

    def unread(validator: Any, value: Any, instance: Any, schema: Any) -> Iterable[Any]:
        raise Refused(f"{_SCHEMA} uses `{keyword}`, which the gate has no check for")

    return unread


def _is_integer(checker: Any, instance: Any) -> bool:
    return type(instance) is int


def _passes(validator: Any, instance: Any, schema: Any) -> bool:
    """Whether ``instance`` is valid against ``schema``, stopping at the first error, and
    worked out once for a schema and a value while the arguments stand as they are."""
    work = _WORK.get()
    if work is None:
        return _valid(validator, instance, schema)
    key = (validator.RECUSAL_READING, id(schema), id(instance))
    known = work.answers.get(key)
    if known is not None and known[0] is schema and known[1] is instance:
        work.notes.extend(known[3])
        return bool(known[2])
    mark = len(work.notes)
    answer = _valid(validator, instance, schema)
    if answer and key[0] == _GATE:
        _moot(work, mark)
    work.answers[key] = (schema, instance, answer, _gathered(work, mark))
    return answer


def _any_of(validator: Any, alternatives: Any, instance: Any, schema: Any) -> Iterable[Any]:
    """``anyOf`` by whether each alternative is valid, which is all it asks. ``jsonschema``
    gathers every error of every alternative that fails, and under a schema that refers
    to itself that is the same alternatives against the same values again at every level."""
    _silent(alternatives)
    mark = _mark()
    if not any(_passes(validator, instance, each) for each in alternatives):
        yield _failed(mark)


def _one_of(sides: dict) -> Any:
    """``oneOf``, as the gate reads it: exactly one alternative is met. That one is
    asked of the gate's own reading, as an ``anyOf`` asks. That there is no second is
    where an alternative that is met makes the whole fail: it is not met where two are
    met by every reading, and not decided where a second may be."""

    def check(validator: Any, alternatives: Any, instance: Any, schema: Any) -> Iterable[Any]:
        _silent(alternatives)
        mark = _mark()
        upper = _as(validator, sides[_MOST])
        may = [each for each in alternatives if _passes(upper, instance, each)]
        if _settled(mark):
            # nothing was met that the readings answer differently: this is every
            # reading's answer, the gate's own included
            if len(may) != 1:
                yield _unmet()
            return
        if len(may) > 1:
            lower = _as(validator, sides[_LEAST])
            if sum(1 for each in may if _passes(lower, instance, each)) > 1:
                yield _unmet()
            else:
                yield _undecided("oneOf", _kinds(mark))
            return
        beside = _mark()
        if not may or not _passes(validator, instance, may[0]):
            yield _failed(beside)

    return check


def _one_of_bound(reading: str, sides: dict) -> Any:
    """``oneOf`` for a bounding reading: one alternative at least is met by this
    reading, and one at most by the other."""

    def check(validator: Any, alternatives: Any, instance: Any, schema: Any) -> Iterable[Any]:
        _silent(alternatives)
        mark = _mark()
        own = sum(1 for each in alternatives if _passes(validator, instance, each))
        other = own
        if own and not _settled(mark):
            turned = _as(validator, sides[_MOST if reading == _LEAST else _LEAST])
            other = sum(1 for each in alternatives if _passes(turned, instance, each))
        if own < 1 or other > 1:
            yield _unmet()

    return check


def _type_bound(reading: str) -> Any:
    """``type`` for a bounding reading: a float that is a whole number is an
    ``integer`` to JSON Schema and not to the gate."""

    def check(validator: Any, types: Any, instance: Any, schema: Any) -> Iterable[Any]:
        names = [types] if isinstance(types, str) else types
        if any(validator.is_type(instance, name) for name in names):
            return
        if type(instance) is float and instance.is_integer() and "integer" in names:
            _doubt("number")
            if reading == _MOST:
                return
        yield _unmet()

    return check


def _loosely(instance: Any) -> Any:
    """What a value of the arguments is compared by when the JSON types are not told
    apart (``1``, ``1.0`` and ``true`` are one). A list or an object is read once for
    a call and counted; a number is not read at all."""
    return _key(_WORK.get() if type(instance) in (list, dict) else None, instance, False)


def _fixed_bound(keyword: str, reading: str) -> Any:
    """``const`` and ``enum`` for a bounding reading. A value that is the listed one, or
    one of them, with the same JSON types all the way down, is that for every reading.
    One that is equal only as a number (``0.0`` for ``0``, ``true`` for ``1``) is that
    for JSON Schema or for a server that compares the two in Python, and not for the
    gate. What the listed values are compared by is worked out once for a listing."""
    index = 0 if keyword == "const" else 1

    def check(validator: Any, listed: Any, instance: Any, schema: Any) -> Iterable[Any]:
        work = _WORK.get()
        strictly = _const if keyword == "const" else _enum
        if next(iter(strictly(validator, listed, instance, schema)), None) is None:
            return
        if type(instance) is not str and instance is not None:
            known = None if work is None else work.loose.get(id(schema))
            if known is not None and known[0] is schema:
                wanted = known[1][index]
            elif keyword == "const":
                wanted = frozenset((_key(None, listed, False),))
            else:
                wanted = frozenset(_key(None, choice, False) for choice in listed)
            if _loosely(instance) in wanted:
                _doubt("equal")
                if reading == _MOST:
                    return
        yield _unmet()

    return check


def _unique_items_bound(reading: str) -> Any:
    """``uniqueItems`` for a bounding reading. Items that are all different as the gate
    compares them (equal numbers are one item whatever their JSON type) are different
    for every reading. Two that are equal only so (``1`` and ``true``) are one item for
    a server that makes a set of them and two for JSON Schema."""

    def check(validator: Any, wanted: Any, instance: Any, schema: Any) -> Iterable[Any]:
        if next(iter(_unique_items(validator, wanted, instance, schema)), None) is None:
            return
        work = _WORK.get()
        seen = set()
        for item in instance:
            key = _key(work, item, True)
            if key in seen:
                yield _unmet()
                return
            seen.add(key)
        _doubt("equal")
        if reading == _LEAST:
            yield _unmet()

    return check


def _left_to_the_server(keyword: str, reading: str) -> Any:
    """``pattern`` and ``format`` for a bounding reading: the gate evaluates neither, so
    for a value one could apply to (a string, for ``pattern``) it is met by the reading
    that is met most and not by the one that is met least."""

    def check(validator: Any, value: Any, instance: Any, schema: Any) -> Iterable[Any]:
        if keyword == "pattern" and type(instance) is not str:
            return
        _doubt(keyword)
        if reading == _LEAST:
            yield _unmet()

    return check


def _pattern_properties_bound(reading: str) -> Any:
    """``patternProperties`` for a bounding reading: which members a pattern governs is
    not worked out, so the reading that is met least asks every member to meet the
    schema of every pattern, and the one that is met most asks nothing."""

    def check(validator: Any, patterns: Any, instance: Any, schema: Any) -> Iterable[Any]:
        if type(instance) is dict and instance and patterns:
            _doubt("patternProperties")
            if reading == _LEAST:
                yield from _pattern_properties(validator, patterns, instance, schema)

    return check


def _additional_properties_bound(reading: str) -> Any:
    """``additionalProperties`` for a bounding reading. Beside ``patternProperties`` the
    members it governs are not known: the reading that is met least asks every member
    ``properties`` does not name to meet it, and the one that is met most asks
    nothing."""

    def check(validator: Any, additional: Any, instance: Any, schema: Any) -> Iterable[Any]:
        if type(instance) is not dict:
            return
        if not schema.get("patternProperties"):
            yield from _additional_properties(validator, additional, instance, schema)
            return
        listed = _listed_properties(schema)
        extras = [name for name in instance if name not in listed]
        if not extras:
            return
        _doubt("patternProperties")
        if reading == _LEAST:
            if additional is False:
                yield _unmet()
                return
            for name in extras:
                yield from validator.descend(instance[name], additional, path=name)

    return check


def _measured(keyword: str, check: Any, reading: str) -> Any:
    """``check``, one keyword's validation, counted against the call's allowance and not
    run again for a schema and a value it has already passed for (by the same reading).
    A keyword applied to a list or an object costs one step more for each item or member
    it holds directly. What a check that passed noted is noted again where it is not
    run again."""

    def measured(validator: Any, value: Any, instance: Any, schema: Any) -> Iterable[Any]:
        work = _WORK.get()
        if work is None:
            yield from check(validator, value, instance, schema) or ()
            return
        work.step(1 + (len(instance) if type(instance) in (dict, list) else 0))
        key = (reading, keyword, id(schema), id(instance))
        known = work.valid.get(key)
        if known is not None and known[0] is schema and known[1] is instance:
            work.notes.extend(known[2])
            return
        mark = len(work.notes)
        clean = True
        for error in check(validator, value, instance, schema) or ():
            clean = False
            yield error
        if clean:
            if reading == _GATE:
                _moot(work, mark)
            work.valid[key] = (schema, instance, _gathered(work, mark))

    return measured


#: The keywords every draft the gate reads checks the same way.
_KEYWORDS: dict[str, Any] = {
    "additionalItems": _additional_items,
    "additionalProperties": _additional_properties,
    "allOf": _all_of,
    "anyOf": _any_of,
    "const": _const,
    "dependencies": _dependencies,
    "dependentRequired": _dependent_required,
    "dependentSchemas": _dependent_schemas,
    "enum": _enum,
    "exclusiveMaximum": _bounded(operator.ge),
    "exclusiveMinimum": _bounded(operator.le),
    "format": _not_evaluated,
    "maxItems": _sized("array", operator.gt),
    "maxLength": _sized("string", operator.gt),
    "maxProperties": _sized("object", operator.gt),
    "maximum": _bounded(operator.gt),
    "minItems": _sized("array", operator.lt),
    "minLength": _sized("string", operator.lt),
    "minProperties": _sized("object", operator.lt),
    "minimum": _bounded(operator.lt),
    "multipleOf": _multiple_of,
    "pattern": _not_evaluated,
    "patternProperties": _pattern_properties,
    "prefixItems": _prefix_items,
    "properties": _properties,
    "propertyNames": _property_names,
    "required": _required,
    "type": _type,
    "uniqueItems": _unique_items,
}

#: The keywords a draft checks its own way: ``contains`` (None for the one with
#: ``maxContains``, which each reading has its own check for) and ``items``.
_BY_DRAFT: dict[Any, dict[str, Any]] = {
    validators.Draft7Validator: {"contains": _contains_one, "items": _items_or_list},
    validators.Draft201909Validator: {
        "contains": None,
        "items": _items_or_list,
    },
    validators.Draft202012Validator: {
        "contains": None,
        "items": _items,
    },
}


def _checks(base: Any, reading: str, sides: dict) -> dict[str, Any]:
    """The function for every keyword ``base`` checks, for one reading: this module's
    own for each, but for ``$ref``, which is the one of ``jsonschema`` (it resolves the
    reference and makes no error itself). A keyword this module has no function for
    refuses the schema that uses it (:func:`_unread`)."""
    own = {**_KEYWORDS, **_BY_DRAFT[base]}
    if own["contains"] is None:
        own["contains"] = _contains(reading, sides)
    if reading == _GATE:
        own.update({"not": _not(sides), "if": _if(sides), "oneOf": _one_of(sides)})
    else:
        own.update(
            {
                "not": _not_bound(reading, sides),
                "if": _if_bound(reading, sides),
                "oneOf": _one_of_bound(reading, sides),
                "type": _type_bound(reading),
                "const": _fixed_bound("const", reading),
                "enum": _fixed_bound("enum", reading),
                "uniqueItems": _unique_items_bound(reading),
                "pattern": _left_to_the_server("pattern", reading),
                "format": _left_to_the_server("format", reading),
                "patternProperties": _pattern_properties_bound(reading),
                "additionalProperties": _additional_properties_bound(reading),
            }
        )
    modern = base is validators.Draft202012Validator
    own["unevaluatedItems"] = _squared(_unevaluated_items(sides, modern), list)
    own["unevaluatedProperties"] = _squared(_unevaluated_properties(sides, modern), dict)
    checks = {}
    for keyword, check in base.VALIDATORS.items():
        checks[keyword] = check if keyword == "$ref" else own.get(keyword) or _unread(keyword)
    return checks


def _never(instance: Any) -> Any:
    """The error of the ``false`` schema, as ``jsonschema`` sets it but for the message."""
    return _unmet(validator=None, validator_value=None, instance=instance, schema=False)


def _own(strict: Any) -> Any:
    """``strict``, a validator class made here, with three of its methods wrapped.

    ``jsonschema`` answers the ``false`` schema inside ``descend`` and ``iter_errors``,
    with a message that holds the value: both answer it here first, with the gate's
    own error. And ``evolve``, which every descent goes through, picks the validator
    class by the ``$schema`` of the subschema it is handed, which would be the class
    of ``jsonschema`` with none of the checks above: a subschema that declares a draft
    is refused when the schema is read, the root's own declaration is not handed to the
    validator (:meth:`Listed._prepare`), and anything that still left this class is
    refused here."""
    descend, iter_errors, evolve = strict.descend, strict.iter_errors, strict.evolve

    def quiet_descend(self: Any, instance: Any, schema: Any, *args: Any, **kwargs: Any) -> Any:
        if schema is False:
            error = _never(instance)
            # `jsonschema` leaves this error at the list or the object that holds the
            # value; the reason names the value the schema allows none for
            path = args[0] if args else kwargs.get("path")
            if path is not None:
                error.path.appendleft(path)
            return iter((error,))
        return descend(self, instance, schema, *args, **kwargs)

    def quiet_iter_errors(self: Any, instance: Any, *args: Any, **kwargs: Any) -> Any:
        if self.schema is False and not args and not kwargs:
            return iter((_never(instance),))
        return iter_errors(self, instance, *args, **kwargs)

    def same_evolve(self: Any, **changes: Any) -> Any:
        evolved = evolve(self, **changes)
        if type(evolved) is not strict:
            raise Refused(
                f"{_SCHEMA} declares a JSON Schema draft on a subschema, which the gate "
                "does not read"
            )
        return evolved

    strict.descend = quiet_descend
    strict.iter_errors = quiet_iter_errors
    strict.evolve = same_evolve
    return strict


def _strict(base: Any) -> Any:
    """``base`` with the checks a server's conversion makes necessary, no regular
    expression evaluated, the comparisons read once, no message written from a value,
    and every keyword counted: the gate's own reading, which holds the two that bound
    it (``RECUSAL_SIDES``) for what stands under a keyword that turns a failure into a
    pass."""
    sides: dict[str, Any] = {}

    def made(reading: str) -> Any:
        strict = _own(
            validators.extend(
                base,
                validators={
                    keyword: _measured(keyword, check, reading)
                    for keyword, check in _checks(base, reading, sides).items()
                },
                type_checker=base.TYPE_CHECKER.redefine("integer", _is_integer),
            )
        )
        strict.RECUSAL_READING = reading
        return strict

    gate = made(_GATE)
    sides[_LEAST], sides[_MOST] = made(_LEAST), made(_MOST)
    gate.RECUSAL_SIDES = sides
    return gate


#: The drafts the gate reads, each with the stricter checks.
_DRAFTS = {base: _strict(base) for base in _BY_DRAFT}

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


#: The specification ``referencing`` reads a schema of each draft by: what
#: ``jsonschema`` resolves a ``$ref`` with for that draft.
_SPECIFICATIONS = {
    validators.Draft7Validator: _specifications.DRAFT7,
    validators.Draft201909Validator: _specifications.DRAFT201909,
    validators.Draft202012Validator: _specifications.DRAFT202012,
}


class _References:
    """The subschema each ``$ref`` of one schema names, read by ``referencing``: the
    library ``jsonschema`` resolves a ``$ref`` with when it validates, built the way
    ``jsonschema`` builds it (the schema as the root resource of an empty registry).

    One reading, so that what the gate reads from the schema a ``$ref`` names (its
    types, its default, the forms of a union, where the walk goes) is read from the
    subschema the validation is made against. A reading of the gate's own differed from
    it: for a pointer with a percent-escape that decodes to ``/``, which the library
    decodes before it splits the pointer, and for the position of an item of a list,
    which the library reads with ``int`` (a sign, a digit that is not ASCII). Each
    ``$ref`` is resolved once for a listing."""

    def __init__(self, schema: Any, base: Any):
        resource = _SPECIFICATIONS[base].create_resource(schema)
        self._resolver = Registry().resolver_with_root(resource)
        self._known: dict[str, Any] = {}

    def __call__(self, ref: Any) -> Any:
        """The subschema ``ref`` names, or :class:`Refused` for one that is not a
        pointer into the schema itself or that the library does not resolve to a
        schema."""
        if type(ref) is str:
            if ref in self._known:
                return self._known[ref]
            if len(ref) > LONGEST_REFERENCE:
                raise Refused(
                    f"{_SCHEMA} has a `$ref` of more than {LONGEST_REFERENCE:,} characters, "
                    "which the gate does not read: a reference is looked up again each "
                    "time it is applied to a value, in proportion to its length"
                )
            if ref == "#" or ref.startswith("#/"):
                try:
                    node = self._resolver.lookup(ref).contents
                except RecursionError:
                    raise  # the stack, not the reference: :meth:`Listed._prepared`
                except Exception:  # noqa: BLE001, whatever the library gives up with
                    node = None
                if type(node) in (dict, bool):
                    self._known[ref] = node
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


def _names(value: Any) -> bool:
    """Whether data is a list of strings: names, which hold nothing."""
    return type(value) is list and all(type(name) is str for name in value)


def _only_names(keyword: str, member: Any) -> bool:
    """Whether the value of a keyword that is not a subschema is of a shape that holds
    names and no schema, so that a member NAMED ``default`` in it is a name and not the
    keyword. Two shapes, each read exactly:

    * ``dependentRequired``, which the gate checks: a map from a property's name to a
      list of names (a property may be called ``default``);
    * ``discriminator``, the OpenAPI annotation Pydantic publishes beside the ``oneOf``
      of a tagged union: ``propertyName``, a string, and ``mapping``, a map from each
      tag VALUE to the reference of its form, a string (a tag may be ``"default"``).
      The gate does not read it: the forms are told apart by their ``const``.

    Anything else under either keyword, and every keyword the gate does not know, is
    looked into for a ``default`` as before."""
    if type(member) is not dict:
        return False
    if keyword == "dependentRequired":
        return all(_names(needed) for needed in member.values())
    if keyword == "discriminator":
        return all(
            type(value) is str
            or (
                name == "mapping"
                and type(value) is dict
                and all(type(target) is str for target in value.values())
            )
            for name, value in member.items()
        )
    return False


def _endless(node: Any, resolve: Any, state: dict[int, bool]) -> bool:
    """Whether ``node`` leads back to itself through ``$ref`` and the keywords that apply
    a subschema to the same value: validating any value against it would never finish.
    A ``$ref`` that does not resolve inside the schema is refused here (``resolve``,
    a :class:`_References`)."""
    if type(node) is not dict:
        return False
    if id(node) in state:
        return state[id(node)]  # True while the node is still being followed
    state[id(node)] = True
    onward: list[Any] = []
    if "$ref" in node:
        onward.append(resolve(node["$ref"]))
    for keyword in (*_SAME_VALUE, "dependencies"):
        member = node.get(keyword)
        if type(member) is dict and keyword in ("dependentSchemas", "dependencies"):
            onward.extend(member.values())
        elif type(member) is list:
            onward.extend(member)
        else:
            onward.append(member)
    found = any(_endless(each, resolve, state) for each in onward)
    state[id(node)] = found
    return found


class _Read:
    """What reading a schema found: the property names it lists (the only argument names
    a refusal may quote), its subschemas, the keywords it uses, and which of its
    subschemas lead back to themselves."""

    def __init__(self, resolve: Any) -> None:
        #: the subschema a ``$ref`` names (:class:`_References`)
        self.resolve = resolve
        self.names: set[str] = set()
        self.nodes: list[dict] = []
        #: the identity of every subschema read (the schema is kept whole, so none is
        #: given to another object)
        self.scanned: set[int] = set()
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
    gate (a member NAMED ``default`` in a map of names is not one: :func:`_only_names`).
    ``additionalProperties`` beside ``patternProperties`` is one the walk does not
    write through: without evaluating the patterns it cannot tell which members it
    governs.

    This reads the subschemas the keywords it knows hold. A ``$ref`` may name any other
    place in the schema; :meth:`Listed._prepare` hands each such place to this in turn,
    so that nothing is validated against that was not read here."""
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
    if node is not root and "$schema" in node:
        # `jsonschema` would check the values under it with its own keywords for that
        # draft, and none of the gate's (`_own`)
        raise Refused(
            f"{_SCHEMA} embeds a schema that declares a draft of its own (`$schema`), "
            "which the gate does not read"
        )
    if _endless(node, read.resolve, read.state):
        raise Refused(f"{_SCHEMA} refers to itself without end")
    if under is not None and "default" in node:
        raise Refused(
            f"{_SCHEMA} lists a default under `{under}`, where the gate cannot tell "
            "whether the server would apply it"
        )
    read.nodes.append(node)
    read.scanned.add(id(node))
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
        elif keyword not in _VALUES and not _only_names(keyword, member) and _holds_default(member):
            raise Refused(
                f"{_SCHEMA} lists a default under a keyword the gate does not read, where "
                "it cannot tell whether the server would apply it"
            )


def _quoted(names: Iterable[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _declared(types: Any) -> list[str]:
    return [types] if type(types) is str else [name for name in types if type(name) is str]


def _digits(text: str) -> int:
    """What the decoder of :func:`_json_text` makes of the digits of an integer: nothing
    of their value, which is not asked for. An integer of more digits than Python's
    default limit is given up on, as the decoder itself gives up where that limit is
    set; where it is not (an older Python, or the limit switched off) making the
    integer takes time that grows with the square of its digits, so no integer is made
    here on any interpreter."""
    if len(text) - (text[0] == "-") > LONGEST_INTEGER:
        raise ValueError("an integer of more digits than the gate reads")
    return 0


#: The decoder of :func:`_json_text`, made once: ``json.loads`` builds a decoder at every
#: call that names a ``parse_int``.
_DECODER = json.JSONDecoder(parse_int=_digits)


def _json_text(text: str) -> bool:
    """Whether a string is JSON text for null, a list or an object: what the MCP Python
    SDK's server decodes and runs in the string's place when it is an argument. Text that
    is not JSON is not.

    The server decodes under its own interpreter, and this one has limits another does
    not share: how deep a list or an object may be nested (about 1,000 levels before
    Python 3.12, about 3,000 in 3.12 and 3.13, more after) and how many digits an
    integer may have (4,300 by default, which is what the gate reads on every
    interpreter: :func:`_digits`). Text this interpreter gives up on for such a limit
    is text a server may still decode, so it is JSON text here when it opens a list or
    an object. Text that opens neither could decode only to a number, which a server
    keeps a string: a long run of digits is judged and sent as the string it is."""
    try:
        decoded = _DECODER.decode(text)
    except json.JSONDecodeError:
        return False  # not JSON, for any interpreter
    except (ValueError, RecursionError):
        return text.lstrip()[:1] in ("[", "{")
    return decoded is None or type(decoded) in (list, dict)


def _number_text(text: str) -> bool:
    """Whether Pydantic may read a string as a number, the white space around it already
    gone. Wider than its own rule, never narrower: the text Python reads as a float,
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


def _javascript_number_text(text: str) -> bool:
    """Whether JavaScript's ``Number()`` reads a string as a number, the white space
    around it already gone: what a zod ``z.coerce.number()`` runs on a string. Wider
    than its rule, never narrower: nothing at all (which it reads as 0), the digits of
    another base after ``0x``, ``0o`` or ``0b`` (as Python reads them, so with a sign
    and with underscores too), and a decimal number, which is text Python reads as a
    float."""
    if not text:
        return True
    if text.lstrip("+-")[:2].lower() in ("0x", "0o", "0b"):
        try:
            int(text, 0)
        except ValueError:
            pass
        else:
            return True
    return _number_text(text)


def _reads(work: _Work, text: str) -> frozenset[str]:
    """What a server may read a string's text as, other than the string: ``number``
    (Pydantic's lax conversion, at any depth), ``javascript`` (a number by the rules of
    JavaScript's ``Number()`` and not by Pydantic's), ``boolean`` (Pydantic's conversion
    again) and ``json`` (null, a list or an object, which the MCP Python SDK's server
    decodes for an argument). Worked out once for a string."""
    known = work.reads.get(id(text))
    if known is not None and known[0] is text:
        return known[1]
    bare = text.strip()
    found = set()
    if bare.lower() in _BOOLEAN_WORDS:
        found.add("boolean")
    if _number_text(bare):
        found.add("number")
    elif _javascript_number_text(text.strip(_SPACE)):
        found.add("javascript")
    if _json_text(text):
        found.add("json")
    reads = frozenset(found)
    work.reads[id(text)] = (text, reads)
    return reads


class Listed:
    """One tool's input schema as a server listed it."""

    def __init__(self, schema: Any, ceiling: int = STEP_CEILING):
        self.schema = schema
        #: the most steps the check of one call may cost, whatever its arguments hold
        self._ceiling = ceiling_of(ceiling)
        self._validator: Any = None
        self._failure: Optional[Refused] = None
        #: the subschema each ``$ref`` names, once the draft is known (:meth:`_prepare`)
        self._resolve: Any = None
        self._names: set[str] = set()
        #: what is read from the schema alone, each under the identity of its subschema
        #: (the schema is kept whole, so an identity is never given to another object)
        self._typed: dict[int, tuple[frozenset[str], frozenset[str]]] = {}
        self._defaults: dict[int, tuple[str, Any]] = {}
        self._unions: dict[tuple[int, str], tuple[list, list, dict]] = {}
        self._planned: dict[int, tuple[dict, list]] = {}
        self._lasts: dict[int, tuple[list, int]] = {}
        #: for a call's work (:class:`_Work`): the subschemas that hold no keyword the
        #: validation checks, and what each ``const`` and ``enum`` is compared by when
        #: the JSON types are not told apart
        self._silent: dict[int, int] = {}
        self._loose: dict[int, tuple] = {}

    # --- the schema itself must be checkable --------------------------------------------------

    def _prepared(self) -> Any:
        """The validator for this schema, or :class:`Refused` when the gate cannot read it.
        Worked out once for a listing.

        What the schema itself is refused for is kept, and said again for every call.
        A schema the interpreter's recursion limit stopped the reading of is refused
        for this call alone: where the limit is reached depends on how deep in the
        application's own stack the gate was called, so the next call may read it."""
        if self._failure is not None:
            raise self._failure
        if self._validator is None:
            try:
                self._validator = self._prepare()
            except Refused as refused:
                self._failure = refused
                raise
            except RecursionError:
                raise Refused(
                    f"{_SCHEMA} is nested too deeply to read within what is left of the "
                    "interpreter's recursion limit where the gate was called"
                ) from None
        return self._validator

    def _prepare(self) -> Any:
        schema = self.schema
        if type(schema) is not dict:
            raise Refused(f"{_SCHEMA} is not a JSON object")
        try:
            detached(schema)
        except TypeError:
            raise Refused(f"{_SCHEMA} is not JSON data") from None
        base = _draft(schema["$schema"]) if "$schema" in schema else validators.Draft202012Validator
        if base not in _DRAFTS:
            raise Refused(
                f"{_SCHEMA} declares a JSON Schema draft the gate does not read (it reads "
                "draft-07, 2019-09 and 2020-12)"
            )
        try:
            self._resolve = _References(schema, base)
        except RecursionError:
            raise
        except Exception:  # noqa: BLE001, an `$id` that is not text: no draft allows one
            raise Refused(f"{_SCHEMA} is not a valid JSON Schema") from None
        self._modern = base is validators.Draft202012Validator
        # in draft-07 a `$ref` is read alone: the keywords beside it are not applied
        self._ref_alone = base is validators.Draft7Validator
        read = _Read(self._resolve)
        try:
            _scan(schema, schema, read)
            # A `$ref` may name any place in the schema, and the scan reads subschemas
            # only where a keyword it knows holds them. What a reference names
            # elsewhere (under a keyword of a vendor's, in `examples`, a map of
            # `properties` taken as a schema) is validated against all the same, so it
            # is read here, by the same rules, as a subschema that stands on its own
            # like one of `$defs`; and so is whatever those refer to in turn. Each is
            # read once. Nothing of the schema is then checked against that was not read.
            elsewhere: list[dict] = []
            index = 0
            while index < len(read.nodes):
                node = read.nodes[index]
                index += 1
                if "$ref" in node:
                    target = self._resolve(node["$ref"])
                    if type(target) is dict and id(target) not in read.scanned:
                        elsewhere.append(target)
                        _scan(target, schema, read)
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
            # what a `$ref` names outside the keywords that hold schemas is no part of
            # what that checked: each is checked as the schema it is used as
            for target in elsewhere:
                _DRAFTS[base].check_schema(target, format_checker=formats)
            # what the schema alone says, read once and not for every call
            checked = _DRAFTS[base].VALIDATORS.keys()
            silent: dict[int, int] = {}
            for node in read.nodes:
                for keyword in ("allOf", "anyOf", "oneOf", "patternProperties"):
                    several: Any = node.get(keyword)
                    if type(several) not in (list, dict):
                        continue
                    nothing = 0
                    for inner in several.values() if type(several) is dict else several:
                        if type(inner) is not dict or not inner.keys() & checked:
                            nothing += 1
                    if nothing:
                        silent[id(several)] = nothing
                self._types(node)
                self._listed(node)
                for keyword in ("anyOf", "oneOf"):
                    if type(node.get(keyword)) is list:
                        self._forms(node[keyword], "object")
                        self._forms(node[keyword], "array")
                if "const" in node or type(node.get("enum")) is list:
                    self._loose[id(node)] = (
                        node,
                        (
                            frozenset((_key(None, node["const"], False),))
                            if "const" in node
                            else None,
                            frozenset(_key(None, choice, False) for choice in node["enum"])
                            if type(node.get("enum")) is list
                            else None,
                        ),
                    )
                self._left_out(node)
            self._silent = silent
        except SchemaError:
            raise Refused(f"{_SCHEMA} is not a valid JSON Schema") from None
        self._names = read.names
        # The validator is handed the schema without its `$schema`: the draft is chosen
        # above, and `jsonschema` chooses again, by that keyword, for every schema a
        # `$ref` leads to. The values under a `$ref` back to a root that declares a draft
        # (`"#"`) would be checked by the keywords of `jsonschema` for that draft, with
        # none of the gate's.
        rooted = {keyword: member for keyword, member in schema.items() if keyword != "$schema"}
        # an empty registry: nothing outside the schema is ever fetched
        return _DRAFTS[base](rooted, registry=Registry())

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
            admitted, named = self._types(self._resolve(schema["$ref"]))
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
                inner.append(self._resolve(schema["$ref"]))
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

    def _left_out(self, schema: dict) -> list:
        """``(name, its schema, the values its default holds beyond the first)`` for
        every property of a subschema that lists a default (or lists different ones,
        which refuses the call that leaves it out): what the walk asks about for an
        object, in the order of the schema. Worked out once for a subschema."""
        known = self._planned.get(id(schema))
        if known is not None and known[0] is schema:
            return known[1]
        pairs: list = []
        for name, member in _listed_properties(schema).items():
            state, default = self._listed(member)
            if state != _NONE:
                pairs.append((name, member, _count(default) - 1 if state == _ONE else 0))
        self._planned[id(schema)] = (schema, pairs)
        return pairs

    def _last_default(self, prefix: list) -> int:
        """The position of the last item of a list of item schemas that lists a default,
        or -1. Worked out once for a list."""
        known = self._lasts.get(id(prefix))
        if known is not None and known[0] is prefix:
            return int(known[1])
        last = -1
        for index, member in enumerate(prefix):
            if self._listed(member)[0] != _NONE:
                last = index
        self._lasts[id(prefix)] = (prefix, last)
        return last

    def _fixed(self, schema: Any, seen: set[int]) -> tuple[bool, Any]:
        """Whether a property's schema fixes one value (a ``const``, or an ``enum`` of
        one), on itself or on what its ``$ref`` and ``allOf`` name, and the value. Each
        subschema is read once (``seen``), however many ways lead to it."""
        if type(schema) is not dict or id(schema) in seen:
            return False, None
        seen.add(id(schema))
        inner: list[Any] = []
        if "$ref" in schema:
            inner.append(self._resolve(schema["$ref"]))
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
            self._tags(self._resolve(schema["$ref"]), found, unfit, seen)
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

    def _forms(self, alternatives: list, wanted: str) -> tuple[list, list, dict]:
        """``(forms, tags, told)`` for the alternatives of one ``anyOf`` or ``oneOf``: the
        ones that admit a value of JSON type ``wanted`` (``object`` or ``array``), and,
        for two or more object forms, each property all of them fix to a value of its
        own, with those values in the order of the forms; and, for each such property,
        the form each of its values tells (by the value as :func:`_same` compares it),
        so that an object's form is looked up and not searched for among them. Worked
        out once for a subschema."""
        known = self._unions.get((id(alternatives), wanted))
        if known is not None:
            return known
        forms = [each for each in alternatives if wanted in self._types(each)[0]]
        tags: list[tuple[str, list]] = []
        told: dict[str, dict] = {}
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
                        told[name] = {
                            _key(None, value, True): form for form, value in zip(forms, values)
                        }
        self._unions[(id(alternatives), wanted)] = (forms, tags, told)
        return forms, tags, told

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
        undecided = getattr(error, "recusal_undecided", None)
        if undecided is not None:
            position, kinds, of = undecided
            rests = [_RESTS_ON[kind] for kind in kinds if kind in _RESTS_ON]
            if of != "{}":
                subject, plural = of.format(subject), of.startswith("the items of")
            return (
                f"{subject} {'stand' if plural else 'stands'} under {_UNDECIDED[position]} "
                f"That rests on {'; and on '.join(rests) or 'what the gate does not settle'}. "
                "The gate sends only arguments it can show to be valid against the "
                "server's schema as they stand, and it cannot show that here."
            )
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
        elif keyword == "propertyNames":
            detail = (
                f"hold{'' if plural else 's'} a member whose name does not meet what the "
                "server's schema asks of member names there (`propertyNames`)"
            )
        elif keyword in ("anyOf", "oneOf"):
            detail = (
                f"match{'' if plural else 'es'} "
                f"{'none' if keyword == 'anyOf' else 'not exactly one'} of the alternatives "
                "the server's schema lists there"
            )
        elif keyword is None:
            # the `false` schema: no keyword failed, the schema allows no value at all
            detail = (
                f"{'stand' if plural else 'stands'} where the server's schema allows no "
                "value (`false`)"
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
        a string with a format), the gate cannot tell. A zod ``z.coerce.number()`` beside
        a string lists the same two types and reads the text by JavaScript's ``Number()``,
        so what that reads as a number is refused there too. And the MCP Python SDK's
        server decodes an argument whose text is JSON for null, a list or an object when
        the parameter is not annotated exactly ``str``, which is a schema that admits
        another JSON type than string. A schema that admits only strings keeps every
        string, and one that names no type keeps the text of a number and of a boolean.

        The reason says both ways out, since the gate cannot tell which was meant: the
        JSON value where the other type was meant, and, where the text itself was (a
        postal code, ``"no"`` as a word), that the server's author can declare the
        parameter a string only."""
        admitted, named = self._types(schema)
        reads = _reads(work, text)
        stated = admitted & named
        if "number" in reads and stated & _NUMBERS:
            held, allows, read = "the text of a number", "a number as well as a string", "a number"
            meant = "a number is meant, send a JSON number"
        elif "javascript" in reads and stated & _NUMBERS:
            held, allows, read = (
                "text a server written in JavaScript reads as a number (an empty string or "
                "white space as 0, the digits after `0x`, `0o` or `0b` as an integer, a "
                "number after a byte order mark)",
                "a number as well as a string",
                "a number",
            )
            meant = "a number is meant, send a JSON number"
        elif "boolean" in reads and "boolean" in stated:
            held, allows, read = (
                "a word for a boolean",
                "a boolean as well as a string",
                "a boolean",
            )
            meant = "a boolean is meant, send JSON `true` or `false`"
        elif "json" in reads and link[0] is None and not admitted <= {"string"}:
            held, allows, read = (
                "JSON text (null, a list or an object)",
                "another JSON type than string",
                "JSON",
            )
            meant = "null, a list or an object is meant, send that JSON value"
        else:
            return
        raise Refused(
            f"argument `{self._path(link)}` is a string that holds {held} where the "
            f"server's schema allows {allows}; a server may read the text as {read} and "
            f"run with that value, not with the string the policy judged. If {meant}, not "
            "text. If the text itself is meant (an identifier, a code, a word), the gate "
            "cannot send it while the schema allows both: the server's author can declare "
            "this parameter a string only (`str`, or `z.string()` with zod), and the gate "
            "then takes every string there as the string it is."
        )

    def _chosen(self, value: Any, alternatives: list, link: Any) -> list:
        """The alternatives of an ``anyOf`` or ``oneOf`` to walk a list or an object
        along: the one form that admits it, or, for an object, the one of several whose
        tag it carries. More than one form with nothing to tell them apart is refused:
        a server built on Pydantic takes the member with more of the fields that are set,
        one built on zod the first that parses, dropping the members it does not know,
        and neither rule is in the schema."""
        forms, tags, told = self._forms(alternatives, "object" if type(value) is dict else "array")
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
        # the form whose tag the object carries, looked up by the tag's value: one
        # lookup for each tag the object carries, however many forms there are
        work = _WORK.get()
        for name in _both(told, value):
            tag = value[name]
            form = told[name].get(_key(work if type(tag) in (list, dict) else None, tag, True))
            if form is not None:
                return [form]
        said = (
            f" The forms are told apart by {_quoted(name for name, _ in tags)}, and the "
            "object carries the value of none of them there."
            if tags
            else ""
        )
        raise _Forms(
            f"{subject} {verb} an object, and the server's schema offers more than one "
            "object form there (alternatives of an `anyOf` or `oneOf`); the gate cannot "
            "tell which of them the server will run, so it cannot show the policy what "
            f"runs.{said} The server can publish a discriminated union (every form "
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
        followed in every draft: a default that is written is also sent and judged, so
        the policy is shown what the server receives. Writing one is not without
        effect on a server: a default that is written is a value the call set, and a
        server that asks which fields a call set (a Pydantic model's
        ``model_fields_set``, ``model_dump(exclude_unset=True)``) no longer sees it as
        left out. The package page states that consequence.

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
            onward.append(self._resolve(schema["$ref"]))
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
        # the members the object has, then the defaults of the ones it leaves out: each
        # read from what is as long as the object, or as the defaults the schema lists
        # there, and never from every name the schema lists (a model of two thousand
        # optional fields costs an empty object nothing)
        for name in _both(properties, value):
            self._walk(validator, work, value[name], properties[name], (link, name), found, held)
        if found is not None and properties:
            for name, member, extra in self._left_out(schema):
                if name not in value:
                    if (id(value), name) in held:
                        work.step()  # found again at every pass, and not planned again
                        continue
                    # a default is copied whole into every object that leaves it out:
                    # one step for each value it holds beyond the first, which the
                    # walk of it counts
                    work.step(extra)
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
        default = detached(default)  # the schema was checked to be JSON data
        try:
            errors = list(validator.descend(default, member))
            stray = self._stray(validator, errors)
        except _NoFloat:
            # both numbers are the schema's here: its default and its `multipleOf`
            raise Refused(
                f"argument `{self._path(here)}` is left out, and the default the server's "
                "schema lists for it cannot be checked against a `multipleOf` listed "
                "there, one of the two being too large to be a float, so the gate "
                "cannot tell what the server would run with"
            ) from None
        if stray:
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
        if found is not None and self._last_default(prefix) >= len(value):
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
        self, validator: Any, work: _Work, sent: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(judged, sent)``: the arguments with every listed default, and with every
        default that can be sent. ``sent`` is a copy, which is written to."""
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
        judged = detached(sent)
        for parent, name, _ in held.values():
            del parent[name]
        return judged, sent

    def complete(
        self, arguments: dict[str, Any], owned: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(judged, sent)`` for a call's ``arguments``, which must be JSON data: what
        the policy judges and what the server is sent. The two differ only by a listed
        default that cannot be sent (one that is not valid against its own property
        schema, or text a server would read as another type), which is judged as listed
        and not sent. The arguments are copied first, unless they are ``owned``: a copy
        the caller made for this call with ``recusal.invocation.detached`` and hands
        over, which is written to and returned as what is sent.
        Raises :class:`Refused` for arguments that are not valid against the schema as
        they stand, for an object or a list the schema offers more than one form for,
        for a string a server may read as another type the schema admits, for arguments
        too costly to check, and for a schema the gate cannot check or complete them
        against."""
        work: Any = None
        token: Any = None
        try:
            validator = self._prepared()
            # a copy in which no list or object stands in two places: the walk knows a
            # place by the object that is there
            sent = arguments if owned else detached(arguments)
            work = _Work(min(_STEPS + _STEPS_PER_VALUE * _count(sent), self._ceiling))
            work.silent, work.loose = self._silent, self._loose
            token = _WORK.set(work)
            return self._complete(validator, work, sent)
        except Refused:
            raise
        except _TooCostly:
            raise Refused(_too_costly(work.allowed, self._ceiling)) from None
        except RecursionError:
            raise Refused(
                f"the arguments are nested too deeply to check against {_SCHEMA} within "
                "what is left of the interpreter's recursion limit where the gate was "
                "called"
            ) from None
        except _NoFloat as cause:
            if cause.sent:
                raise Refused(
                    "the arguments hold an integer too large to be a float (more than "
                    "about 308 digits) where the server's schema lists a `multipleOf` "
                    "that is a fraction, which is checked by dividing the two as floats, "
                    "so the gate cannot check it. Send a number a float can hold."
                ) from None
            raise Refused(
                f"{_SCHEMA} lists a `multipleOf` too large to be a float, which the gate "
                "cannot check a number that is not an integer against"
            ) from None
        except PASSED_ON:
            raise
        except BaseException as exc:  # noqa: BLE001, what the gate cannot check is refused
            raise Refused(
                f"the arguments could not be checked against {_SCHEMA} ({type(exc).__name__})"
            ) from None
        finally:
            if token is not None:
                _WORK.reset(token)
