<p align="center">
  <a href="https://recusal.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
      <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
    </picture>
  </a>
</p>

# recusal-openai-agents-mcp

[![Website](https://img.shields.io/badge/website-recusal.io-2453c4)](https://recusal.io)
[![PyPI](https://img.shields.io/pypi/v/recusal-openai-agents-mcp)](https://pypi.org/project/recusal-openai-agents-mcp/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/philpaz/recusal/badge)](https://scorecard.dev/viewer/?uri=github.com/philpaz/recusal)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

The [Recusal](https://github.com/philpaz/recusal) gate for MCP tools on an
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) agent: every call to a
local MCP server, and every hosted MCP call OpenAI asks approval for, is judged by your
policy first.

**New to Recusal?** It checks each tool call an AI agent proposes against your policy and
refuses the unsafe or unapproved ones before the tool runs. No model makes that decision.
A policy is a plain Python function: it is given the tool's name and its arguments and
returns a list of findings. An empty list lets the call run; a failing finding refuses it
and gives the reason.
[Writing a policy](https://github.com/philpaz/recusal#write-the-policy-two-postures-one-principle)
covers the two postures, allowlist and deny-list.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/gate-position.svg">
    <img alt="Before the tool runs. An agent proposes a tool call. Recusal, your policy with no model in it, decides. With no objection the tool runs. On a failing finding, or when the policy itself raises, the call is refused: the tool never runs and the reason goes back to the model. Every decision can be written to a hash-chained audit log, if you pass one." src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/gate-position-light.svg" width="880">
  </picture>
</p>

This is a separate distribution from `recusal`. The kernel has zero runtime dependencies
and a frozen compatibility perimeter; this package carries the SDK and MCP dependencies
and its own version, so a release of either never forces a release of the kernel.

It builds on
[`recusal-openai-agents`](https://github.com/philpaz/recusal/tree/main/adapters/openai-agents),
the function-tool gate, and depends on that package.

Install it from PyPI. `recusal-openai-agents`, which this package requires, comes with it:

```bash
pip install recusal-openai-agents-mcp
```

Requires Python 3.10+, `recusal>=0.16.0`, `recusal-openai-agents>=0.2.2`,
`openai-agents>=0.24.0`, `mcp>=2.3.0`, `jsonschema>=4.26.0` and `referencing>=0.37.0` (the
MCP SDK requires `jsonschema` itself, and `referencing` comes with `jsonschema`). It is
tested on every pull request against exactly `openai-agents==0.24.0` with
`openai==3.49.0`, and weekly against the newest releases.

The package ships inline types (`py.typed`) for your editor and type checker.

## Use

```python
from agents import Agent
from recusal import Finding
from recusal_openai_agents_mcp import gated_agent


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


agent = gated_agent(Agent(name="assistant", tools=[...], mcp_servers=[server]), policy)
```

`gated_agent` gates all three kinds of tool at once: function tools (through
`recusal-openai-agents`), every server in `mcp_servers`, and every `HostedMCPTool`.
`gate_mcp_server(server, policy)` and `gate_hosted_mcp(tool, policy)` gate one of each.

**Local MCP servers are gated in place, at `server.call_tool`.** That is the one point
where the SDK hands the server its own tool name and the arguments that are sent to it,
after any approval a person gave. A refusal returns an MCP error result carrying
Recusal's reason, and the server is never reached. The policy sees the name the server
executes under, whatever public name the model called it by, so a policy keyed by tool
name must name the server's own tools.

**The policy judges what is sent, checked against and completed from the server's listed
schema.** A server does three things with what it receives that the gate cannot watch: it
fills what the call leaves out from its defaults, it converts a value of another type
(the string `"false"` to a boolean, `"500"` to a number), and where its schema offers
several object forms it picks one by a rule of its own. The tool's listed `inputSchema` is
the only statement of what the server accepts, and it states none of those rules, so
before the policy runs the gate:

- **requires the arguments to be valid against the whole listed schema, as they stand.**
  They are validated with [`jsonschema`](https://pypi.org/project/jsonschema/), in the
  draft the schema declares (2020-12 when it declares none; draft-07 and 2019-09 are read
  too, each under its `http` or `https` address, with or without the trailing `#`):
  `anyOf`, `oneOf` and `allOf`, a `$ref` into the schema's own `$defs` or `definitions`,
  `items`, `prefixItems`, `additionalProperties`, `required`, `enum`, ranges and lengths.
  A `format` is not asserted. JSON Schema validation does not convert, so a string is not
  a boolean or a number. The gate is stricter in three places where a server would still
  convert: `5.0` is not an `integer`; an `enum` or `const` value must be of the same JSON
  type as the listed one (`1.0` and `true` are not `1`); and for `uniqueItems` two equal
  numbers are one item whatever their JSON type (`1`, `1.0` and `true`, which a Python
  set holds once). A JSON string sent for a parameter the schema types as an object or a
  list is refused like any other value of another type; the MCP Python SDK's server
  would parse it into one.
- **never sends what the listed schema forbids because it read a keyword more strictly.**
  A stricter reading refuses more only where a keyword adds a demand. Four keywords turn
  a failure into a pass: `not`, the condition of an `if`, `oneOf` (a second alternative
  that is met makes it fail) and `maxContains`. Under those, reading `5.0` as no
  `integer` would admit what the schema forbids (`{"type": "number", "not": {"const":
  0}}` and `0.0`), and a keyword the gate leaves to the server would decide the result.
  So what stands under one of the four is read by two readings that bound every reading
  a server can make: one that is met only where every reading is (the stricter three,
  and `pattern`, `format` and the members a `patternProperties` governs counted as not
  met), and one that is met wherever any is (JSON Schema's own three, with `true` and
  `1` equal as a server written in Python finds them, and those keywords counted as
  met). Where the two agree, that is the answer. Where they differ the gate cannot tell
  whether the keyword is met, and the call is refused with a reason that names the
  keyword (`not`, `if`, `oneOf`, `maxContains`) and what the answer rests on (a whole
  number sent as a float, a value equal to one of another JSON type, a `pattern`, a
  `format`, `patternProperties`). An `if` whose condition is not decided still sends
  arguments that meet both of its branches; and a `oneOf` asks only that no second
  alternative may be met, so a tagged union whose forms hold a `pattern` or a `format`
  is called as before. The same holds for what `unevaluatedProperties` and
  `unevaluatedItems` count as evaluated, which the gate works out itself: which
  members an `if` evaluated depends on its branch, so where the condition is not
  decided only what both branches evaluated counts (and, for the reading that is met
  wherever any is, what either did). What the gate sends is therefore valid for
  `jsonschema` as it ships, for every keyword the gate evaluates, with one exception
  that is `jsonschema`'s own: `uniqueItems`. The specification asks that no two items
  are equal, and `jsonschema` 4.26.0 sorts the items and compares neighbours, so a list
  whose equal items do not sort together (`[[true], [1], [true]]`) is unique for it.
  Under a `not` the gate, which compares every item with every other, then sends what
  `jsonschema` rejects. A test compares 245,248 calls with it, each schema alone and
  under each of the four keywords, in the three drafts, and an `if` beside an
  `unevaluated` keyword in every such position: every call the gate sends is valid for
  `jsonschema` with `uniqueItems` compared pairwise, and the 104 of the calls the gate
  sends that `jsonschema` as it ships rejects each hold such a list
  (`test_what_the_gate_sends_is_valid_for_jsonschema_as_it_ships`). The cost: a float
  that is a whole number, `true` beside a `const` of `1`, and any string under a `not`
  or an `if` that holds a `pattern`, are refused there although a server may accept
  them, and so is an object beside an `if` on a pattern that `unevaluatedProperties`
  does not find evaluated by both branches.
- **never evaluates a server's regular expression.** `pattern` is not checked: the server
  checks it. A pattern can take longer on a few dozen characters than any limit, and the
  text is the model's. The same holds for the names of members: under `propertyNames` a
  `pattern` is not checked, and beside `patternProperties` the gate does not work out
  which pattern a member's name matches, so every member of the object must meet the
  schema of every pattern listed there, a member `properties` does not name must also
  meet `additionalProperties` when that is a schema, and `additionalProperties: false`
  asserts nothing there (the server rejects a member no pattern matches). That holds
  where the keyword adds a demand; under a `not`, an `if`, a `oneOf` or a `maxContains`
  a keyword that is not evaluated is neither met nor not met (above). A test counts it:
  across 13,041 calls against schemas that hold a pattern in every keyword that takes
  one, in every such position, and across the 245,248 above, nothing hands Python's `re`
  module a pattern while the gate reads a listing or checks a call
  (`test_no_pattern_of_a_listed_schema_is_handed_to_the_re_module`). What that does not
  say: a listing is checked against the metaschema of its draft, and the metaschema's
  own fixed patterns (for the text of an `$anchor`) are run on the listed schema's
  text, which is the server's and not the model's
  (`test_the_patterns_run_while_a_listing_is_read_are_the_metaschemas_own`).
- **refuses an object where the schema offers more than one object form, and a list
  where it offers more than one list form.** Two or more alternatives of an `anyOf` or
  `oneOf` that admit an object are what a `Union` of two models and a zod `z.union` of
  two objects publish. Which form runs is the server's own rule, and it is not JSON
  Schema's: a server built on Pydantic takes the member with more of the fields that are
  set and converts strings to fit it, and one built on zod takes the first member that
  parses and drops the members that one does not know, although it lists
  `additionalProperties: false`. So an object for such an argument is refused whatever
  it holds, at any depth, with one exception: where every object form fixes one
  same-named property to a `const` (or an `enum` of one value) of its own, no two the
  same, and the object carries that property with the value of exactly one form, that
  form alone is used. A list has no such tag: where two or more alternatives admit a
  list, a list is refused. One object form, or one list form, beside alternatives that
  are not one (`Optional[Model]`, `Model | str`, `list[Model] | None`) is unaffected.
- **refuses a string where the schema also admits the type a server reads its text as.**
  One rule for three readings. Pydantic's conversion reads the text of a number as the
  number and a word for a boolean (`true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`,
  `t`, `f`, `y`, `n`, in any case) as the boolean, at any depth, unless the value is
  taken as a string first: `"1700000000"` for `when: Union[int, date]` runs as the
  integer. The schema of such a parameter names `integer`, `number` or `boolean` beside
  string (a string with a `format` counts), which is what the gate can see, so there a
  string that reads as a number (anything Python's `float` accepts once the white space
  around it and its underscores are gone, `inf` and `nan` included) or as a boolean is
  refused. A zod `z.coerce.number()` beside a string
  (`z.union([z.coerce.number(), z.string()])`) lists the same two types and reads the
  text with JavaScript's `Number()`, so beside a number type the gate also refuses what
  that reads as a number and Pydantic does not: an empty string and white space (which
  run as 0), the digits of another base after `0x`, `0o` or `0b` (`"0x1F"` runs as 31),
  and a number after a byte order mark. And the MCP Python SDK's server decodes an
  argument whose text is JSON for null, a list or an object when the parameter is not
  annotated exactly `str`: `{"scope": "null"}` for `scope: Optional[str]` runs as
  `None`. So for an argument whose schema admits another JSON type than string (an
  `anyOf` or a `type` list with `null` in it, a schema with no `type`), such text is
  refused. The server decodes under its own Python, whose limits are not the gate's:
  text the gate's Python gives up on for a limit of its own (a list or an object nested
  deeper than about 1,000 levels before Python 3.12 and about 3,000 in 3.12, or an
  integer of more than 4,300 digits) is refused like other JSON text when it opens a
  list or an object, because a server on another Python may decode it. A long run of
  digits alone is the string it is: a server keeps a number a string.
  A schema that admits only strings keeps every string, one that names no type
  keeps the text of a number and of a boolean (a server does not convert for a
  parameter typed `Any`), and JSON text further in than an argument is the string it
  is, which no SDK decodes.

  The gate cannot tell whether the other type or the text itself was meant, so the
  reason gives both ways out. Where the other type was meant, send it as JSON: a JSON
  number, `true` or `false`, the list or the object itself. Where the text was meant
  (the identifier `"007"`, the postal code `"01234"`, the word `"no"`), the call cannot
  be made as the schema stands: the server's author can declare the parameter a string
  only (`str`, `z.string()`), and every string is then judged and sent as the string it
  is. The rule is applied whatever the server is built with, because the listing does
  not say: a zod server never decodes JSON text and keeps a string for
  `z.union([z.number(), z.string()])`, and the gate refuses there all the same.
- **writes every default the schema lists for a property the arguments leave out, at
  every depth.** The gate walks the arguments with the schema, through `$ref`,
  `properties`, `items`, `prefixItems` (in the older drafts a list of `items`, and
  `additionalItems`), `additionalProperties`, `allOf`, and the one form of an `anyOf` or
  `oneOf` that admits the object or the list there (above), and adds each property that
  is left out and has a default. The default is the property's own `default`; when its
  schema lists none itself, it is the one listed by the schema its `$ref` names, by a
  member of its `allOf` or by an alternative of its `anyOf` or `oneOf` (zod publishes
  `z.boolean().default(true).nullable()` that way), where all that are listed must be
  the same value. A default that is an object is completed in turn. The completed
  arguments are checked again, by all of the above.

The policy judges the completed arguments. They are what is sent, with one exception.

**Writing the defaults changes what a server sees as set.** The gate writes the listed
defaults so that the policy judges what the server would run with: a server fills what a
call leaves out after the gate has judged, and a policy that was shown `{"path": "/"}`
has not judged the `recursive: true` the server then adds. Since what is judged is what
is sent, the written defaults are in the call the server receives, and a server cannot
tell a default the gate wrote from a value the model sent. For most tools that changes
nothing: the server would have filled the same value. It does change a tool that asks
which fields the call set. A Pydantic model knows that (`model_fields_set`), and a
partial-update tool that writes `patch.model_dump(exclude_unset=True)` over a record
changes only the fields the call mentions. Called through the gate with `{"patch":
{"phone": "555-0199"}}`, where `name` and `email` are `Optional[str] = None`, it is sent
`{"patch": {"phone": "555-0199", "name": null, "email": null}}`: a field the call did
not mention is then a field the call set, and the tool clears the name and the email.
The policy is shown exactly that call, so a policy can refuse it, and an audit entry
records it; the gate does not prevent it. A server that tells "left out" from "set to
the default" should not be called through this gate as it stands, or should list no
default for those fields (a field with no listed default is never written). The gate
has no way to send the model's own arguments and still show the policy what runs: that
would rest on the server applying exactly the defaults it lists, which the listing does
not promise (the exception below gives cases where it does not).

**The exception: a listed default that cannot be sent is judged, as listed, and not
sent.** There are two kinds. One that is not valid against its own property schema:
Pydantic publishes `{"default": null, "type": "string"}` for a parameter written
`q: str = None`, and the server would reject the `null` if it were sent. And one that
is text a call would be refused for sending: `mode: Optional[str] = "null"`, which the
server would decode to `None`, and the text of a number where a number is listed too.
The policy is given the default as the schema lists it (`q: None`, `mode: "null"`), the
audit entry is bound to it, and the server is sent the arguments without it. This rests
on the listing being what the server runs with when the parameter is left out. That is
so for a server built with the MCP Python SDK, which lists the default it uses. It is
not always so for one built with zod, which derives the default another way in these
cases: `z.number().default(Infinity)` lists `default: null` and runs `Infinity`;
`z.number().min(5).default(1).optional()` lists `default: 1` and runs with the parameter
absent; a `Set` or a `Map` default is listed as `{}`; and an object default that fails
its own schema is parsed again through that schema, which fills or drops members. The
gate cannot tell these listings from Pydantic's, so it judges the default as listed.

**What the gate cannot check or complete, it refuses** before the policy runs, with a
reason the model reads. The reason names the place in the arguments (`opts.recursive`,
`items[1]`), written only from property names the schema lists and positions in a list (a
key the schema does not list is written `*`), and says what the schema requires there. It
carries no argument text. Refused:

- arguments that are not valid against the schema. A value of another JSON type is told
  the types the schema declares: send the declared type, and an object as an object, not
  as JSON text. A value outside an `enum`, a `required` property left out, a member that
  `additionalProperties: false` forbids and any other constraint each say so. A member
  whose NAME does not meet `propertyNames` (what Pydantic lists for
  `dict[Literal["cpu", "ram"], int]`) is said to be that, and the name is not written;
- arguments that stand under a `not`, an `if`, a `oneOf` or a `maxContains` the gate
  cannot decide for them (above);
- an object where the schema offers more than one object form and no tag tells which,
  and a list where it offers more than one list form (above);
- a string whose text a server reads as another type the schema also admits there
  (above);
- a schema that is not a JSON object, is not a valid JSON Schema, or declares a draft
  other than draft-07, 2019-09 and 2020-12;
- a `$ref` that does not resolve inside the listed schema itself. A remote reference is
  never fetched, and a reference to an `$anchor` is not resolved. `$dynamicRef`,
  `$recursiveRef` and a subschema with an `$id` of its own are refused too. A `$ref` is
  read by `referencing`, the library the validation resolves it with, so what the gate
  reads from the schema a `$ref` names (its types, its default, the forms of a union)
  is read from the schema the arguments are validated against, whatever the pointer
  holds (a percent-escape that decodes to `/`, a position in a list written with a
  sign). A `$ref` may name any place in the schema, also one under a keyword that holds
  no schemas (a vendor's `x-shapes`, an entry of `examples`, a map of `properties` taken
  as a schema): whatever a `$ref` names is read by the same rules as a subschema under
  `$defs`, and is checked to be a valid schema, before anything is validated against
  it;
- a `default` listed where the gate does not read one, because it cannot tell whether
  the server would apply it: under `if`, `then`, `else`, `not`, `patternProperties`,
  `additionalProperties` beside `patternProperties`, `dependentSchemas`, `dependencies`,
  `contains`, `propertyNames`, `unevaluatedProperties`, `unevaluatedItems` and
  `contentSchema`, and under any keyword the gate does not know (a vendor's `x-shape`, a
  keyword of a later draft). `const`, `enum`, `examples`, `example` and `default` hold
  data and are not looked into. A member NAMED `default` is not the keyword where the
  value holds names and no schema, in the two shapes the gate reads exactly: a
  `dependentRequired` (a property may be called `default`), and the `discriminator`
  Pydantic publishes beside a tagged union, whose `mapping` is keyed by the tag values
  (a tag may be `"default"`);
- a property left out whose alternatives list different defaults
  (`{"anyOf": [{"type": "boolean", "default": true}, {"type": "string", "default":
  "no"}]}`), and a place two parts of the schema list different defaults for;
- a property left out whose listed default has a member its own schema does not allow.
  A Pydantic model whose fields have aliases, used as a parameter's default with
  `extra="forbid"`, lists that default under the field names and its schema under the
  aliases, so the gate cannot tell what the server would run with;
- a property left out whose listed default holds an object or a list where the schema
  offers more than one form: Pydantic runs with such a default as listed, and zod
  parses it again through the union;
- a list that leaves out an item a `prefixItems` entry lists a default for;
- a schema that leads back to itself through `$ref`, `allOf`, `anyOf`, `oneOf`, `not`,
  `if`, `then`, `else`, `dependentSchemas` or the draft-07 `dependencies` alone, with no
  value in between (`{"$ref": "#"}`), or whose defaults never settle;
- a schema that uses `unevaluatedProperties` together with `patternProperties`, which
  could be applied only by evaluating the patterns, wherever in the schema the two
  stand (a `patternProperties` only a `$ref` reaches among them); a 2019-09 schema
  with `unevaluatedItems` beside an `items` that is `true` or `false`, which
  `jsonschema` raises on; and arguments that `unevaluatedProperties` or
  `unevaluatedItems` forbid beside an `if` whose condition is not decided, unless both
  branches evaluate what they hold (refused as not decided, like the four keywords);
- a `$ref` of more than 1,024 characters: a reference is looked up again each time
  it is applied to a value, in proportion to its length, which no step counts;
- a tool the server's listing names more than once: a call carries the name alone, the
  SDK checks its required parameters against the first listing of a name, and the gate
  reads one schema for a tool and does not choose between two;
- a tool the server object keeps under another schema than the listing it last handed
  out, where listing again does not settle it (below);
- arguments that cost more to check than the gate allows a call (below);
- an integer too large to be a float (more than about 308 digits) where the schema
  lists a `multipleOf` that is a fraction, which is checked by dividing the two as
  floats. The reason says so, and to send a number a float can hold;
- arguments that are not JSON data (a non-finite number such as `1e400`), or that hold a
  value more than 64 levels below the arguments object;
- arguments of more than 1,000,000 characters as JSON text, and arguments that hold an
  integer of more than 4,300 digits (below);
- a value where the schema allows none (a `false` subschema), which the reason names as
  that.

Nothing leaves the gate but its answer. An exception inside it (in the validation
library, in the policy, in writing the record) is a refusal of the call, with a reason
that names the kind of exception and nothing of the arguments, and is one deny entry on
an audit log where the entry can be written. That holds for an exception that is not a
Python `Exception` too: a panic in a compiled dependency arrives as a `BaseException`.
Only what stops the interpreter or the event loop is passed on: `KeyboardInterrupt`,
`SystemExit`, `GeneratorExit` and `asyncio.CancelledError`.

An argument the schema neither names nor forbids is judged and sent as it is.

The two refusals for forms and for text are also made under the keywords no default is
written through (`then`, `else`, `dependentSchemas`, `dependencies`, `patternProperties`,
`unevaluatedProperties`, `contains`, `unevaluatedItems`), whether or not the keyword
applies to the value.

A `default` is never applied where a value is always present: on the schema of the
arguments themselves (the gate always sends an object), of the items of a list and of
`additionalProperties`. JSON cannot send a member or an item without a value.

**The work is bounded by counting it, not by a timer.** A step is one keyword of the
schema applied to one value (and one more for each member or item that value holds
directly), one value visited by the walk above, one value read for a comparison (`enum`,
`const`, `uniqueItems`), or one part of the place of a validation error. And what a wide
schema costs beyond that: one step for each subschema an `allOf`, `anyOf`, `oneOf` or
`patternProperties` holds that checks nothing (`{}`, `true`, `false`, a title alone),
which would otherwise cost a descent and count nothing; one for each name read of a
`dependentRequired` list; one for each value a default holds beyond the first, each time
it is written; and one for each default that cannot be sent, at every pass. The names a
schema lists are never read one by one for an object: the members an object has are
looked up among them, a `required` list is read to the first name that is missing, the
defaults of a subschema and the form a tag names are worked out once for a listing. So a
model of two thousand optional fields costs an empty object what a model of four does.
What the other keywords evaluated, for `unevaluatedProperties` and `unevaluatedItems`,
is worked out by the gate once for a schema and a value, with a step for each subschema
it goes into that nothing else counted (`jsonschema` followed a `$ref` again each time
it was named, so definitions that each name the next one twice were read two to the
power of their number of times). And a `$ref` may be 1,024 characters at the most, since
a reference is looked up again each time it is applied. One thing a listed schema can
make long is still not counted: reading the listing itself, once, when the first call
after it is checked (its check against the metaschema, and the gate's own reading),
which costs in proportion to the size of the listed schema.
`unevaluatedProperties` and `unevaluatedItems` are counted as the square of the number
of members or items they are applied to, because they are checked by comparing
each with every other. Every step is
counted against an allowance of 10,000 steps and 1,000 more for every value in the
arguments (an object, a list, and each member, item and scalar in them), up to a
ceiling of 200,000 steps for one call (`recusal_openai_agents_mcp.STEP_CEILING`), which
an application can raise (below). Nothing is
validated twice for the same schema and the same value while the arguments stand as they
are, a list or an object is walked once for a schema, and the choices of an `enum` are
read once for a call, so a recursive schema (a model with `child: Optional[Node]`), a
list of 12,000 ordinary items under `uniqueItems` and 20,000 values against an `enum`
of 600 each cost in proportion to the arguments. What a single step costs beyond its
count is bounded by the size of that keyword in the listed schema and by the length of
the string or the size of the number it reads, and never by what the value holds below
it. `jsonschema` writes the value that failed into the message of every error it makes,
the ones an `anyOf` asks an alternative for and drops among them, and the gate reads no
message; so `jsonschema` applies the schema and resolves its references, and each
keyword is checked by a function of the gate's own that writes none. What the schema
alone says (the types
a subschema admits, the default it lists, the forms of a union) is read once, at the
first call after each listing. The ordinary calls measured spend from 4 steps for each
value (a list of integers) to 19 (a list of a model of four fields), 37 (a recursive
model), 123 (a list of a discriminated union of ten models) and 323 (of thirty models):
up to a third of the 1,000 a value adds. A call over the allowance is refused before the
policy runs, with a reason that begins "the arguments are too costly to check against
the server's listed schema for this tool" and says which of the two limits stopped the
check, and is one deny entry on an audit log.

**The two ceilings on one call, what they cost, and how to raise them.** The steps
allowed for each value do not bound the time a call takes: with them alone the time
grows with whatever the model sends, and the check runs on the event loop (next
section). Two ceilings bound it. A call whose check would cost more than 200,000 steps
is refused, whatever it holds, and the reason says that the ceiling stopped it and how
the application raises it. And a call whose arguments are more than 1,000,000
characters as JSON text (`recusal_openai_agents_mcp.SIZE_CEILING`) is refused before
anything else reads them, with a reason that says so and how the application raises
that ceiling; it is one deny entry on an audit log. Neither ceiling alone bounds the
time. The steps do not bound the size: a step that reads a string costs the string's
length, and values in an argument the schema does not name cost no steps, while the gate
still copies and counts them. And the size does not bound the steps: 809 items of a
ten-model discriminated union are 11 KB and cost 200,000. The size is the length of the
arguments written as JSON text in its most compact form (no white space; a character
outside ASCII counts as one, and is more than one byte), and measuring it stops at the
ceiling.

Beside the two ceilings stands one limit an application cannot raise: a call whose
arguments hold an integer of more than 4,300 digits is refused in the same reading,
before anything else reads them, and is one deny entry on an audit log. That is the
limit Python sets by default on writing an integer as text (3.11 and later, and 3.10
from 3.10.7), and under it the SDK never hands the gate a longer integer, because
decoding the model's JSON refuses it first. On a Python without the limit (3.10.0 to
3.10.6, or one where it is switched off with `PYTHONINTMAXSTRDIGITS=0` or
`sys.set_int_max_str_digits(0)`) the SDK decodes an integer of any length, and writing
one out takes time that grows with the square of its digits before Python 3.12, which
the ceiling on the size does not bound. The gate reads the number of digits of a long
integer from its bits and writes no integer out, so every interpreter refuses what the
default one refuses, and nothing after the gate (the fingerprint of an audit entry, a
policy, the server) is handed an integer of the call's arguments with more digits than
that.

The cost of the two ceilings is that a large ordinary call is refused. The largest
calls the default step ceiling completes, measured, each under the ceiling on the size:

| Arguments | Largest call completed | Size as JSON |
|---|---|---|
| `list[int]` | 49,997 items | 289 KB |
| `list[str]`, short strings | 49,997 items | 439 KB |
| `dict[str, Optional[float]]` | 39,997 members | 509 KB |
| `dict[str, Optional[str]]`, short strings | 39,997 members | 698 KB |
| `list[Union[int, str]]`, short strings | 33,331 items | 289 KB |
| `list[Union[int, float, bool, str, None]]`, short strings | 24,998 items | 214 KB |
| a recursive JSON type (`list[Json]` among its members), 60 lists deep around numbers | 12,420 numbers | 50 KB |
| `list[Item]` (a model of four fields, two defaults written) | 3,571 items | 110 KB |
| a recursive model (`kids: list[Node]`), its children sent as `{}` | 5,403 children | 16 KB |
| a list of a discriminated union of ten models, items `{"kind": ...}` | 809 items | 11 KB |
| a list of a discriminated union of thirty models | 308 items | 4.5 KB |

An application whose ordinary calls are larger than that raises the ceilings for the
connection:

```python
agent = gated_agent(
    Agent(name="assistant", mcp_servers=[server]),
    policy,
    step_ceiling=1_000_000,
    size_ceiling=5_000_000,
)
```

`gate_mcp_server` and `audited_agent` take the same two keywords, each a whole number. A
server object carries one of each, as it carries one policy: gating it again with
another raises `ValueError`, and `gated_agent` then gates none of the agent's servers.
The time one call can hold the event loop grows with each ceiling in proportion (the
time tables below). Raising the step ceiling does not help a call that costs more than
10,000 steps and 1,000 more for each of its values, which none of the ordinary schemas
measured reaches and a hand-written one can (`unevaluatedProperties`, below). The gate
cannot tell that of a call the ceiling stopped, so the reason for the ceiling says it,
and the reason for the steps allowed for each value says that raising the ceiling does
not change it.

The items `uniqueItems` compares are gathered into a set, and a set costs one step for
each member only while their hashes differ. The hash of a number is the sender's to
choose: every multiple of 2^61 - 1 has the same one, and lists of small integers can be
chosen to share one too. So the gate compares a number by its text (equal numbers have
the same text whatever their JSON type), and the hash of a string is keyed by Python
with a secret of the process, unless hash randomization is switched off
(`PYTHONHASHSEED`). Integers of one hash now cost what as many ordinary ones do
(below).

**What the check costs in time.** It is written in Python and runs on the event loop
inside `call_tool`, so for as long as it takes nothing else on that loop runs. It takes
longer than the server's own validation, in proportion to the arguments up to the two
ceilings. Measured with
[`tools/measure_mcp_cost.py`](https://github.com/philpaz/recusal/blob/main/tools/measure_mcp_cost.py)
on one laptop (Windows 11, Python 3.12.5), the best of three runs, the arguments
decoded from JSON text as the SDK hands them to the gate. The figures in the three
tables are of the code as it stood before the eighth review round (commit `fdd84b0`
of this repository, where the tables were last written). That round changed
how the gate reads the arguments outside the check (their measure, the copy and the
count, which no longer recurse) and not what a step is, and it found that the place a
call is made from moves these times (below the third table). The ninth round changed
how a `oneOf`, a `not`, an `if` and a `maxContains` are read (by a second validator
class, and twice where the two readings differ) and how the names a schema lists are
looked up; the largest call of each row and the steps it spends are unchanged
(`tools/measure_mcp_cost.py` reproduces them on this code), and the times of the rows
with a discriminated union, which is a `oneOf`, are the ones this most likely moved.
The tenth round changed how `unevaluatedProperties` and `unevaluatedItems` are read
(by the gate itself, where `jsonschema` gathered what was evaluated), which the
figures for `unevaluatedProperties` in the text below the second table are of. The
tables and the times in the text around them have not been measured again on this
code: 0.2.1 is released with them as they are, and they are what is still to be
done. The first two tables time the
check itself (`Listed.complete`: the validation, the two refusals and the defaults);
what the rest of the gate and an audit log add is in the third. First the largest call
of each shape that the default step ceiling completes (the table above), which is the
longest the check of one such call takes:

| Arguments, at the default ceiling | The gate | Pydantic's validation of the same value |
|---|---|---|
| `list[int]`, 49,997 items | 249 ms | 0.4 ms |
| `list[str]`, 49,997 items | 573 ms | 0.5 ms |
| `dict[str, Optional[float]]`, 39,997 members | 398 ms | 1.5 ms |
| `dict[str, Optional[str]]`, 39,997 members | 609 ms | 1.6 ms |
| `list[Union[int, str]]`, 33,331 short strings | 718 ms | 1.8 ms |
| `list[Union[int, float, bool, str, None]]`, 24,998 short strings | 837 ms | 3.6 ms |
| a recursive JSON type, 60 lists deep around 12,420 numbers | 576 ms | 1.4 ms |
| `list[Item]` (a model of four fields, two defaults written), 3,571 items | 267 ms | 1.2 ms |
| a recursive model, 5,403 children | 369 ms | 4.9 ms |
| a list of a discriminated union of ten models, 809 items | 353 ms | 0.3 ms |
| a list of a discriminated union of thirty models, 308 items | 350 ms | 0.2 ms |

A call one item larger is refused, and takes as long to refuse. A hand-written schema
that reaches one subschema 4,096 ways (twelve levels of `allOf`) took 726 ms to be
stopped at the ceiling, on 5,897 bytes of arguments: the ceiling bounds the steps, and
a step took between 1.2 and 4.2 microseconds across these cases, the least for a
number and the most for a short string beside other types in a union, which the gate
reads for what a server could read it as. A check that fails costs what the steps
count and no more: a recursive model nested 60 deep whose last node is not valid and
holds 250,000 children (750 KB) is refused after 497 steps
in 306 ms, all of it copying and counting the values, where every level used to write
out everything below it (about 1,300 ms for the same call). One string of 1,000,000
characters is 12 steps for an `Optional[str]` and took between 13 ms (letters) and
35 ms (byte order marks) to check; where the schema lists ten alternatives and none is
a string, it is refused in under 1 ms without being read. The time grows in proportion
to the length: with the ceiling on the size raised that far, 10,000,000 characters took
122 ms to 358 ms and 50,000,000 letters 626 ms. Then the same shapes with the step
ceiling raised (`step_ceiling=`), which is what raising it costs:

| Arguments, the ceiling raised | The gate | Pydantic's validation of the same value |
|---|---|---|
| `list[int]`, 10,000 items | 37 ms | 0.1 ms |
| `list[str]`, 10,000 items | 116 ms | 0.1 ms |
| `dict[str, Optional[float]]`, 10,000 members | 77 ms | 0.3 ms |
| `list[Union[int, float, bool, str, None]]`, 10,000 short strings | 328 ms | 3.1 ms |
| `list[Item]`, 10,000 items | 772 ms | 3.5 ms |
| `list[Item]`, 50,000 items (1.6 MB: the ceiling on the size raised too) | 3,799 ms | 18.1 ms |
| a recursive model, 10,000 children | 607 ms | 8.9 ms |
| a list of a discriminated union of ten models, 10,000 items (140 KB) | 4,935 ms | 3.9 ms |
| a list of a discriminated union of thirty models, 10,000 items (147 KB) | 11,725 ms | 4.1 ms |

That is between 68 and 2,853 times Pydantic's own time across the cases the script
times beside it, and the gate's time grew in proportion: five times the items took
between 4.6 and 6.1 times as long. Under `uniqueItems`, 20,000 ordinary integers took
93 ms and 20,000 integers of one hash 104 ms at the default ceiling, and with the
ceiling raised 100,000 took 533 ms and 513 ms. Under `unevaluatedProperties` on a
hand-written schema, 443 members of one object are the most the default ceiling
completes: 400 took 3 ms and spent 163,216 steps, 500 were refused in 1 ms and 30,000
in 77 ms. With the ceiling raised, 1,000 members took 10 ms and spent 1,008,016 of the
1,012,000 steps allowed, and 1,100 were refused in 3 ms for costing more than 1,000
steps for each value.

**What the rest of the gate adds, and the longest a call takes at the ceiling on its
size.** Before the check the gate measures the arguments and copies them, and after it
the policy is handed a copy of its own; with an audit log, the fingerprint of the
arguments is written. All of that is in proportion to the size of the arguments (the
number of values in them, and the length of their strings), whether or not the schema
names them: values in an argument the schema does not name cost no steps, and still
cost that time. The ceiling on the size is what
bounds it. Through `call_tool` on a gated server object, with both ceilings as they are
by default:

| Arguments | The check alone | `call_tool` | `call_tool` with an audit log |
|---|---|---|---|
| `list[int]`, 49,997 items (the first table) | 178 ms | 187 ms | 224 ms |
| `list[str]`, 49,997 items | 511 ms | 515 ms | 523 ms |
| `dict[str, Optional[float]]`, 39,997 members | 346 ms | 385 ms | 425 ms |
| `list[Union[int, float, bool, str, None]]`, 24,998 short strings | 828 ms | 847 ms | 851 ms |
| at the ceiling on the size: one string of letters | 13 ms | 13 ms | 16 ms |
| at the ceiling: one string of byte order marks | 35 ms | 37 ms | 47 ms |
| at the ceiling: a list of `0`, in an argument the schema does not name | 109 ms | 184 ms | 442 ms |
| at the ceiling: a list of `{}`, in an argument the schema does not name | 336 ms | 439 ms | 942 ms |
| at the ceiling: the same list 60 lists deep | 497 ms | 608 ms | 1,111 ms |
| at the ceiling: a list of `{}` 60 lists deep, for a parameter typed `Any` | 501 ms | 609 ms | 1,105 ms |
| at the ceiling: a recursive model 60 deep whose last node is not valid, its children `{}` (refused) | 360 ms | 465 ms | 2,399 ms |
| at the ceiling: a list of `0` for `list[int]` (refused at the step ceiling) | 54 ms | 125 ms | 392 ms |

A call over the ceiling on the size is refused as soon as the reading reaches the
ceiling. Where one string, or the number of members of one list or object, is over it
alone, nothing more is read: a string of 50,000,000 characters and a list of 2,000,000
empty objects were each refused in under 3 ms, where a 50,000,000-character string took
626 ms to check with the ceiling raised. A call a little over the ceiling is known to
be over only once most of it is read (500,001 numbers, 333,400 empty objects), so
refusing it costs about what measuring a call at the ceiling costs, which is part of
every `call_tool` figure above. Arguments that are not JSON data (a number too large
for a float decodes to one that is not finite) are read three times before they are
refused: to measure them, to find that out, and to write the text the refusal is bound
to. The tool times both (`a little over`, `not JSON data`).

The two bounds add up: one call can spend 200,000 steps on an argument the schema names
and carry what is left of 1,000,000 characters in one it does not, so the longest one
call can take is of the order of the longest row of the first table and the longest of
this one together, and longer from an unlucky place in the stack (next). The tool times
that call (`the two ceilings together`).

**The place a call is made from moves these times.** On Python 3.11 and later CPython
keeps the frames of Python functions in blocks of 16 KB, and a function called where a
block ends maps a new block and frees it again when it returns, at every call. The
check is made of such calls, so the same arguments take longer when the gate is called
from some depths of the application's own stack than from others, and each figure
above is the time from the one place the tool measured it from.
`python tools/measure_mcp_cost.py stack` makes four of these calls from each of 144
depths of plain Python frames and prints the least, the median and the most; this
page does not state that range yet. What the gate itself reads of the arguments (their
measure, the copy, the count of their values, the text a refusal is bound to) it reads
without recursion and without a Python call for each value, so the depth the sender
nests the arguments at does not move those. The copy is the kernel's
(`recusal.invocation.detached`), which does not call itself from `recusal` 0.16.0, the
oldest kernel this package installs with. The validation recurses with a recursive
schema, as deep as the arguments nest under it, and so does the kernel's fingerprint
of the arguments for an audit entry: there the sender's nesting moves the place too.

**What the rules above mean for ordinary schemas:**

- A parameter typed as a `Union` of two models, or as a zod `z.union` of two objects,
  cannot be called through the gate with an object, whatever the object holds, and
  `extra="forbid"` or `.strict()` does not change that. Give every model a `Literal` tag
  field of its own (with or without `Field(discriminator=...)`), or use
  `z.discriminatedUnion`, and have the call carry the tag; or publish one tool for each
  form. A tag field with a default must still be sent: without it the gate cannot tell
  the forms apart.
- A parameter typed `list[A] | list[B]`, or a list beside `Any`, cannot be called with a
  list. Publish `list[Union[A, B]]` with tagged members, or separate tools.
- A parameter typed `Union[int, str]`, `Union[float, str]` or `Union[bool, str]` (zod:
  `z.union([z.number(), z.string()])`) cannot be sent a string that reads as a number or
  as a boolean, although such a server would keep it a string: its schema is the same as
  that of `Union[int, date]`, for which the server converts. A `Decimal` parameter lists
  a number beside a string too: send the number. Such a parameter cannot be sent an
  identifier that is all digits (`"007"`, a postal code), a word Pydantic reads as a
  boolean (`"no"`, `"on"`, `"t"`), an empty string or a `0x` literal: if text is what
  the parameter takes, type it `str` (`z.string()`).
- A string parameter that also admits null (`Optional[str]`, `z.string().nullable()`),
  or any other JSON type, cannot be sent the text `"null"`, `"[]"` or `'{"a": 1}'`. That
  holds on a zod server too, which would keep the text.
- A hand-written `anyOf` or `oneOf` of `required` lists (one of `a` and `b` is
  required: `"anyOf": [{"required": ["a"]}, {"required": ["b"]}]`) is two alternatives
  that admit an object, so the object is refused whatever it holds; at the top of a
  tool's schema that is every call. Written as a condition
  (`"if": {"not": {"required": ["a"]}}, "then": {"required": ["b"]}`) it is one form.
- An object under `unevaluatedProperties`, or a list under `unevaluatedItems`, costs the
  square of its members or items, so 443 of them is the most a call can hold there at
  the default ceiling, and about 1,000 with the ceiling raised, when the call holds
  little else. No SDK publishes these keywords; hand-written schemas do.
- A large call is refused at the ceiling on the steps of one call, although the server
  would run it: more than about 50,000 numbers or short strings, 3,571 records of a
  four-field model, 809 items of a ten-model discriminated union (the table above). An
  application that expects such calls raises the ceiling (`step_ceiling=`).
- A call whose arguments are more than 1,000,000 characters as JSON text is refused,
  whatever the schema, although the server would run it: a document or a file sent
  whole as one string, for one. An application that expects such calls raises that
  ceiling (`size_ceiling=`).
- A schema that declares a draft on one of its subschemas (a `$schema` anywhere but on
  the schema of the arguments) is refused, as one that embeds an `$id` is.
- A `NamedTuple` or tuple parameter with a defaulted last member publishes a
  `prefixItems` entry with a default, so a call that leaves that item out is refused;
  send every item.
- A zod object publishes `additionalProperties: false` unless it is written with
  `.passthrough()`, so an argument its schema does not name is refused.
- OpenAPI's `nullable: true` is not JSON Schema: the gate reads the declared `type`, so
  `null` for `{"type": "string", "nullable": true}` is refused. A schema that means null
  lists it (`"type": ["string", "null"]`, or an `anyOf` with `{"type": "null"}`).
- A recursive model nested as deep as the 64-level limit on the arguments is completed
  where the interpreter's recursion limit leaves room for it: the validation recurses
  with the schema, a number of Python frames for each level the arguments nest, on top
  of the application's own stack. Under Python's default limit of 1,000 frames, a
  model nested the full 64 levels was completed when the gate was called from up to
  210 plain Python frames below the top of a script on Python 3.12.5, and from up to
  79 on Python 3.10.22 (`python tools/measure_mcp_cost.py stack` prints it); from
  deeper the call is refused, with a reason that names the recursion limit, and
  arguments nested less deeply leave more room. An application that calls the gate
  from deep in its own stack, or lowers the limit, meets that refusal sooner.

To read the schema the gate also wraps the server's `list_tools`, in place. `list_tools`
returns what the server listed, unchanged, and the gate keeps each tool's schema from the
most recent listing: each listing replaces the one before, so a call to a tool the server
no longer lists is refused, and so is a call to a tool the listing names more than
once. A call to a tool that is not in the most recent listing makes
the gate call `list_tools()` itself, with no run context and no agent, once for that
call. A tool that is still not listed, and a server whose tools cannot be listed that
way, is refused.

The gate judges a call against the listing the agent was handed. The SDK's own server
classes hand out a listing they fetched without keeping it when another fetch was
published first, or when the cache was invalidated while the fetch was in flight
(`invalidate_tools_cache()`): the agent then holds one schema and the server object
(`cached_tools`) another. Where the two differ for a tool, the gate lists the tools
again when that tool is called and judges by the fresh listing; where that cannot be
had, the call is refused, with a reason that says the two listings differ.

A `tool_filter` hides tools from one agent; the schemas are the server's. The SDK's own
server classes keep the listing as it was fetched, before the filter (`cached_tools`),
and that whole listing is what the gate keeps. So two agents that share one server
object under a dynamic filter each reach the tools they were listed, and the gate's own
`list_tools()`, which such a filter cannot answer without an agent, still leaves it the
fetched listing. The gate does not enforce a filter: the SDK never routes a model's call
to a tool the filter hides, and a call your own code makes to one through
`server.call_tool` is judged by the policy like any other. A server class of your own
that keeps no such listing (`cached_tools` is `None`, the base class's answer) leaves
the gate with the tools of the latest listing as filtered for one agent: give each agent
its own server object there.

**A server carries one policy.** The gate is installed on the server object, so every
agent that lists that object shares it. Gating it again with the same policy is a no-op;
gating it with a different policy raises `ValueError`. Use a separate server object per
policy.

**Hosted MCP is approval-gated, not execution-gated.** Those calls run on OpenAI's
servers. `gate_hosted_mcp` returns a copy whose `require_approval` is `"always"` and
whose approval callback is Recusal's: a refusal answers `{"approve": False, "reason":
...}`, so the hosted call is not made. A hosted tool whose `require_approval` is anything
but `"always"` is reported as ungated. The policy sees the arguments as the model sent
them: the hosted server's schema is not read, so the check above does not apply.

**Build the agent completely, then gate it.** A tool or a server appended to the gated
agent afterwards is not gated, and `is_gated(agent)` then returns `False`. The gated copy
has its own `tools`, `mcp_servers` and `handoffs` lists, so a later change to the
original agent's lists does not reach it. The servers themselves are shared with the
original agent and stay gated for both. A sub-agent behind `as_tool` is accepted when it
is gated itself with this package's `gated_agent`, its own servers and hosted tools
included.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_openai_agents_mcp import audited_agent

log = AuditLog(path="decisions.jsonl")
agent = audited_agent(Agent(name="assistant", mcp_servers=[server]), policy, log)
```

One log covers the three seams, each under its own surface label:

| Seam | Surface | Call id on the entry |
|---|---|---|
| Function tools | `openai_agents.tool_input_guardrail` | the SDK's tool call id |
| Local MCP server | `openai_agents.mcp.call_tool` | none: the SDK hands `call_tool` no id; the server is `control.mcp_server` |
| Hosted MCP | `openai_agents.hosted_mcp.approval` | the approval request id; the server label is `control.mcp_server` |

Each entry holds a SHA-256 fingerprint of the arguments, never the arguments. For a local
server's judged call that is the fingerprint of the arguments as judged: the arguments
sent, completed with the listed defaults, and with any listed default that is not sent
(above). A call the gate refuses before the policy runs (arguments the schema does
not allow, an object or a list the schema offers more than one form for, a string a
server may read as another value, arguments too costly to check, a schema the gate
cannot check, a tool the server does not list, arguments that are not JSON data) is one
entry too, a deny, bound to the arguments as the SDK handed them.
Arguments that are not JSON data have no fingerprint of their own, so that entry is
bound to the text the gate writes them as, under the key `input`: JSON without
spaces, a non-finite number as `nan`, `inf` or `-inf`, an object that is not data as its
type's name in angle brackets, and `...` for anything more than 64 levels down. A
call refused for the size of its arguments (above) is one deny entry too, and the gate
does not read arguments it refuses for their size, so that entry holds no fingerprint
of them: it is bound to the text
`<arguments of more than 1,000,000 characters as JSON text>` (with the ceiling of the
connection), under the key `input`. A call refused for an integer of more than 4,300
digits is bound the same way, to
`<arguments that hold an integer of more than 4,300 digits>`: a fingerprint would write
the integer out. And a call refused because an exception left the gate's own reading
of it is bound to `<arguments the gate could not judge>`. If the
log cannot be written the call is refused. Check the log with
`recusal audit verify decisions.jsonl`.

## What the gate does not see

- **A parameter the call leaves out that has no listed default.** It is the server's to
  fill, and the schema does not say with what. Most optional parameters of hand-written
  schemas are like this, and so is every Pydantic `default_factory` default, which is not
  published. A policy that depends on such a parameter should refuse a call that leaves
  it out.
- **A server whose code does more than its schema says.** The gate settles what is
  sent, and that it is valid against the listed schema. What a server's own code then
  makes of a value is not in the schema. Cases seen on servers built with the official
  SDKs (`mcp` 2.3.0, and the TypeScript SDK with zod):
  - **A parameter without an annotation.** The MCP Python SDK lists `type: string` for it
    and then decodes the string as JSON, so `"null"` runs as `None` and `"[]"` as a
    list. A schema that admits only strings keeps every string, so the gate does not
    refuse these. Annotate the parameter `str`, and the server keeps the string.
  - **A type the schema leaves out.** `scope: str | SkipJsonSchema[None]` lists
    `type: string` only, and the server still decodes `"null"` and runs with `None`, for
    the same reason.
  - **A string the schema says is encoded.** A Pydantic `Json[...]` parameter lists a
    string with `contentMediaType: application/json`: the policy judges the string, and
    the server runs with what it decodes from it. `contentMediaType`, `contentEncoding`
    and `contentSchema` are not applied.
  - **A string with a `format`, which the type behind it rewrites.** A `format` is not
    asserted, and the server's type normalises the text it is given. A numeric string
    for a date: `"0"` for a parameter typed `date` runs as 1970-01-01 and
    `"1700000000"` for a `datetime` as that timestamp. A URL:
    `"HTTP://X/a/../admin"` for a parameter typed `AnyUrl` runs as `http://x/admin`.
    A path: `""` for a parameter typed `Path` runs as `.`. A UUID:
    `"{12345678-1234-5678-1234-56781234ABCD}"`, in braces and capitals, runs as
    `12345678-1234-5678-1234-56781234abcd`. What to do: a policy that compares such a
    string (a host, a path prefix, an identifier) normalises it the way the server's
    type does before it compares, or refuses any text that is not already in the
    normal form.
  - **A number a Pydantic server reads as a date, a time, an address, a complex
    number or an enum member.** In a
    union read the default way (no `union_mode`), a number that is not exactly the
    type of another member runs as the first member that converts it. For
    `when: Union[datetime, Decimal]` the listing is a string with a format beside a
    number, and the number `5` runs as the `datetime` 1970-01-01T00:00:05, `1.5`
    likewise. A `time` beside a `Decimal` runs both numbers as a `time` too, and a
    `datetime` beside an `IntEnum` runs `1` as the `datetime`. For a `date` beside a
    `Decimal`, only a whole number of days runs as a `date` (`0` as 1970-01-01 and
    `86400` as the day after); `5` and `1.5` run as the `Decimal`. A number also runs
    as an `IPv4Address` (`5` as `0.0.0.5`) or as a `complex` (`5` as `5+0j`) where
    that type stands before a `Decimal` or an `IntEnum`. With the members the other
    way round
    (`Union[Decimal, datetime]`) the number runs as the `Decimal`, and the listing
    holds the same types. What to do: the policy judges the number it was given, so
    for such a tool it applies the server's conversion, or refuses a number where it
    expects the text of a date.
  - **`1`, `0`, `1.0` and `0.0` that run as booleans, and `true` that runs as `1`.**
    In some unions
    with a boolean member a Pydantic server runs `1` as `True` and `0` as `False`:
    `force: Union[bool, Decimal]` does (`2` runs as the `Decimal`), and `1` runs as
    `True` for `Union[bool, IntEnum]`, `Literal[True] | int` and
    `Literal[True] | float` too. Where the listing admits a float there, the floats
    do the same: `1.0` runs as `True` for `Union[bool, Decimal]` and
    `Literal[True] | float`, and `0.0` and `-0.0` as `False` for `Union[bool, Decimal]`.
    The other way round, `Literal[1, 2] | bool` runs
    `true` as the integer `1`. Each value is valid against
    the listed schema as it stands, so the gate sends it. What to do: for a tool
    whose schema lists a boolean beside a number, the policy treats `1`, `1.0` and
    `true` (and `0`, `0.0` and `false`) as one value, or refuses the one it does not
    expect.
  - **A validator or a constraint that changes the value.** A string constraint that
    strips and lowers (`StringConstraints(strip_whitespace=True, to_lower=True)`,
    `z.string().trim().toLowerCase()`) runs `"  /ETC "` as `"/etc"`. This also changes a
    default the gate wrote: Pydantic does not validate a default when the parameter is
    left out, so `name = "X "` runs as `"X "` without the gate, and as `"x"` with it,
    because the gate sent the default and the server validated what it was sent. The
    policy judged `"X "`.
  - **A fallback for a value that fails.** zod's `.catch(...)` is in no schema:
    `z.boolean().catch(undefined).or(z.null())` runs `null` as undefined.
  - **A default the server derives another way than it lists.** A listed default that
    cannot be sent is judged as listed (above): zod's `default(Infinity)`, such a
    default under `.optional()` (which runs with the parameter absent), a `Set` or `Map`
    default and an object default parsed again run as something else. So does a default
    a Pydantic model validates (`validate_default=True`).
  - **A zod `z.coerce` beside another type.** `z.union([z.coerce.boolean(), z.string()])`
    lists `boolean` and `string` and runs every string as a boolean: `"abc"` as `true`,
    `""` as `false`. The gate refuses only the words Pydantic reads as a boolean there,
    and `"abc"` is judged as the string. `z.coerce.boolean().or(z.number())` lists
    `boolean` and `number` and runs the number `5` as `true`. And
    `z.record(z.coerce.number(), ...)` lists an object with no key type and reads its
    keys as numbers, so `{"1": 1, "01": 2}` runs as one member. (What
    `z.coerce.number()` beside a string reads as a number is refused, above.) What to
    do: keep `z.coerce` out of a union and out of the keys of a record on a server
    behind the gate, or have the policy refuse a string, and a key that is not written
    the one way a number is, for that tool.
  - **The keys of an object, which a Pydantic server converts.** `quotas: dict[int, int]`
    lists `additionalProperties: {"type": "integer"}` and no key type. Pydantic reads
    the keys `"1"` and `"01"` as the one integer 1, so `{"1": 5, "01": 1000000}` is
    judged with two members and runs as `{1: 1000000}`; `dict[bool, str]` reads `"true"`
    and `"yes"` as one key. What to do: type the keys `str` on the server, or have the
    policy refuse a key that is not written the one way (for an integer key,
    `str(int(key)) == key`).
  - **A member a Pydantic server ignores, or reads from another place.** A dataclass
    field declared `field(init=False)` is listed and never read:
    `{"job": {"path": "/data", "dry_run": true}}` runs with `dry_run` false. A field with
    `validation_alias=AliasPath("options", "dry_run")` is listed under its own name and
    read from `options.dry_run`, a member the schema does not name: the listed member
    is ignored and the unlisted one decides. What to do: a policy for such a tool has
    to be written from the server's code, since the listing does not say which member
    is read.
  - **A union a Pydantic server reads left to right.** With
    `Field(union_mode="left_to_right")` on `Union[date, float]` the listing is a string
    with a format beside a number, and the number `0` runs as the date 1970-01-01,
    although a float is a member. (Without `union_mode` a number that is exactly the
    type of a member runs as that member; one that is not is converted, above.) What
    to do: the policy judges the number it was given, so for such a tool it has to
    apply the server's own conversion, or refuse the type it does not expect.
- **An integer too large for a float.** A JSON integer above 2^53 sent for a `number`
  parameter is judged as the exact integer; a server that reads the parameter as a float
  runs the nearest float. A server written in TypeScript holds every number as a float,
  so there the same is true of an `integer` parameter (`z.number().int()`):
  `9007199254740993` runs as `9007199254740992`. What to do: a policy that depends on
  the exact value refuses an integer above 2^53 (`9007199254740992`).
- **Whether the server reads a member its schema does not name there.** An argument, or
  a member of an object at any depth, that the schema neither names nor forbids at that
  place is judged and sent as it is, a string that holds JSON text or the text of a
  number included: the text check reads the schema of the place, and there is none. A
  server that ignores such a member (a Pydantic model does, a zod intersection drops it)
  runs without it.
- **A schema that changes after it was listed.** The gate checks against the schema of
  the most recent listing made through the gated server object. A server that changes a
  tool between that listing and the call, and a listing the SDK serves from its cache
  (`cache_tools_list=True`), leave the gate with the older schema.
- **Request metadata.** What a server's `tool_meta_resolver` returns is sent as `_meta`
  beside the arguments. It is not given to the policy and is not in the fingerprint.
- **Hosted MCP: the server's defaults and type conversion.** The hosted server's schema
  is not read, so the policy sees the arguments as the model sent them. Which of two
  duplicate keys in the argument text the hosted server reads is not seen either: the
  policy is given the last one, as `json.loads` reads it.
- **A hosted call OpenAI does not ask approval for.** That is why the gate sets
  `require_approval` to `"always"`.
- **A hosted approval the SDK answers from its own memory.** The gate keeps no state
  between turns. When the provider repeats the id and the arguments of an approval
  request the gate already allowed, the SDK approves it again without asking the
  callback, so there are two approvals for one consultation of the policy and one entry.
  A policy that counts calls counts consultations, not approvals.
- **What the tool's own approval callback answered** after Recusal allowed a call. The
  entry records Recusal's decision, not that callback's.
- **A call the SDK rejects before `call_tool`.** Arguments for a local MCP tool that are
  malformed JSON, or not a JSON object, are rejected by the SDK itself, and the model
  reads the SDK's tool error. Arguments that hold a lone surrogate character raise
  `UnicodeEncodeError` inside the SDK and stop the run. Nothing runs and there is no
  entry.
- **Handoffs, an agent with a `prompt`, an agent whose `model_settings` put tools into the
  request, an ungated sub-agent behind `as_tool`, and `Agent` subclasses.** `gated_agent`
  raises for these at construction. A prompt stored on OpenAI's servers can configure
  tools, hosted MCP among them, that are not in the agent's `tools` list, and so can
  `model_settings.extra_body` or `extra_args` with a `tools`, `prompt`, `plugins`,
  `web_search_options` or `mcp_servers` key.
- **Model settings given to a run.** Only the agent's own `model_settings` are read when
  it is gated. `RunConfig(model_settings=...)` is merged in when a run starts, and an
  `extra_body` there is not seen.
- **Legacy `functions` definitions in `extra_body`.** A `functions` key there is sent in
  the body of the request and is not refused. It only describes functions to the model:
  the SDK runs a call by looking its name up among the agent's own tools, so a call to a
  name that is not one of them raises `ModelBehaviorError` and nothing runs, and a call
  to a name that is one passes through that tool's gate.

**Beyond the gate.** This gate judges the tool calls the framework routes through it;
the paths around it inside the framework are listed above. Like every Recusal gate, it
does not stop an agent that can reach a resource another way: a credential with wider
rights, a shell or network path outside the agent, or a file it can write that
configures the gate. Use it as one layer of several: least-privilege credentials for the
agent, a sandbox or container, limits on its network egress, and an audit anchor the
writer cannot reach ([the limits in
full](https://github.com/philpaz/recusal/blob/main/SECURITY.md#things-worth-knowing)).

## Status

**In-depth review under way.** Its fixes so far are released in 0.2.1, and the review continues. Recusal is a Beta (`0.x`) open-source project in active
development: it is not a finished or certified security product, and it is provided under
Apache-2.0 without warranty.
[What the reviews are, and where every adapter stands](https://github.com/philpaz/recusal#project-status).

**Help make it better.** Read the code, test this adapter against your own stack and tell
us what you find. Reviews, bug reports and pull requests are all welcome:
[how to contribute](https://github.com/philpaz/recusal/blob/main/CONTRIBUTING.md), [open issues](https://github.com/philpaz/recusal/issues).
Please report a security gap [privately](https://github.com/philpaz/recusal/security/advisories/new), not in a public
issue.

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_openai_agents_mcp.__all__`. It uses only the public
names of `recusal-openai-agents`.

A runnable offline demonstration, with an in-process MCP server and no API key, is
[`examples/openai_agents_mcp_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_agents_mcp_gate.py)
and
[`examples/openai_agents_mcp_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_agents_mcp_audit.py).

## More

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/adapter-map.svg">
    <img alt="One kernel, eight adapters. Each adapter package depends on recusal: recusal-crewai, recusal-google-adk, recusal-langchain, recusal-langgraph, recusal-openai, recusal-openai-agents, recusal-openai-agents-mcp and recusal-pydantic-ai. recusal-langchain also depends on recusal-langgraph, and recusal-openai-agents-mcp on recusal-openai-agents. The kernel depends on nothing." src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/adapter-map-light.svg" width="880">
  </picture>
</p>

An arrow means "depends on". No arrow leaves the kernel, so a framework's breaking
change is answered by a release of that framework's adapter, not by a release of the
kernel.

- [Framework notes for MCP tools on the OpenAI Agents SDK](https://github.com/philpaz/recusal/blob/main/docs/FRAMEWORKS.md#openai-agents-sdk-mcp-tools):
  the detail behind this page.
- [All eight adapter packages](https://github.com/philpaz/recusal/tree/main/adapters) and
  [the kernel](https://github.com/philpaz/recusal), which has zero runtime dependencies.
- [Verifying a release](https://github.com/philpaz/recusal/blob/main/docs/VERIFY.md#adapter-packages):
  each release is built in CI, signed with Sigstore and published to PyPI by trusted
  publishing.
- [Changelog](https://github.com/philpaz/recusal/blob/main/CHANGELOG.md),
  [issues](https://github.com/philpaz/recusal/issues),
  [security policy](https://github.com/philpaz/recusal/blob/main/SECURITY.md) and
  [support](https://github.com/philpaz/recusal/blob/main/SUPPORT.md).

Licensed under Apache-2.0.
