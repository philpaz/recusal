# Review log

This is the record of the in-depth review of each Recusal package, round by round: what
each round found, what changed, what is still open, and where you can help. It covers
2026-11-19 to 2026-11-22, the date of the newest commit it cites.

The threat model is in two parts. The model is untrusted: everything it produces is input
to the gate. The application's own code is trusted but ordinary: a decorator around a
tool, an optional audit log, a keyword spelled wrong
([the threat model](REVIEW.md#the-threat-model)). A round is one pass over one package
against that model, by reading the code and by running it against the pinned version of
the framework ([what a review round is](REVIEW.md#what-a-review-round-is)).

Every fixed finding below was reproduced before it was fixed, and the fix is pinned by a
test. The reviews are the project's own work, and a review finds what it looks for.
Recusal is a Beta (`0.x`) open-source project in active development. It is not a finished
or certified security product.

How to check this page:

- A released fix is in the [changelog](../CHANGELOG.md) under the package's name.
- A commit is named by its short id. `git show <id>` prints its message. Three branches
  were merged as one commit each; the commits of their rounds are kept on the branches
  `review-history/pydantic-ai`, `review-history/openai-agents` (with
  `recusal-openai-agents-mcp`), `review-history/langgraph-langchain` and
  `review-history/kernel` (fetch them first: `git fetch origin "review-history/*"`).
- A release date is the date of the tagged commit: `git log -1 --format=%ci <tag>`.
- A fix that is not on `main` yet is named with its pull request. Its changelog entry and
  its tests are on that pull request's branch.
- This page states no count of generated cases. Several rounds ran generated test
  spaces, and the scripts that produce them are in the repository for `recusal-openai`
  only ([`tools/review/`](../tools/review)). Adding the others is one of the ways to
  help.

## The kernel: `recusal`

**Where it stands.** `recusal` 0.15.0 is released (tag `v0.15.0`, 2026-11-21). Two more
changes are in review in [#299](https://github.com/philpaz/recusal/pull/299). They are
not merged and not released.

The part of the kernel under review is `recusal.invocation`
([source](../recusal/invocation.py)), which adapters use to read the parameters of a
decorated tool. `recusal-crewai` still carries its own copy.

**The rounds.**

- **With the Google ADK fix, 2026-11-20** (commit `77c3fb2`). `unsupported_signature`
  read a `functools.wraps` wrapper as the function it wraps, so a parameter the wrapper
  adds ran with its default while a gate judged a call without it. The kernel now walks
  every callable a call reaches (`reached_parameters`) and refuses such a parameter and a
  declared `__signature__`. `json_data` takes only the exact JSON types and finite
  floats. Released in 0.15.0.
- **Round one of that change** (commit `2ff9a03`). A forwarding wrapper that names a
  parameter the function also takes kept the argument, and the function ran with its own
  default. Refused now. Pinned by
  `test_a_wrapper_that_takes_a_parameter_the_function_also_takes_is_refused` in
  [`tests/test_invocation.py`](../tests/test_invocation.py). Released in 0.15.0.
- **In review, #299** (commit `293f22f`). Three adapter reviews found the same gap: a
  decorator that applies a second wrapper inside itself calls it from its closure, and
  the walk never read that wrapper. A `functools.singledispatch` function runs an
  implementation chosen by the type of an argument, and the walk read only the base one.
  The change refuses both (`HiddenWrapper`, `DispatchedAtCallTime`), reads nothing it
  cannot read with certainty, and lists what it does not find. The commit message says
  four reviews shaped it.

**Open items.**

- The review after those four held #299 on two items. Importing `recusal.invocation`
  raised in a few environments where 0.15.0 imports: a replaced `functools.partial`, a
  `sys.modules["functools"]` that is not a module, and an audit hook that raises on
  `__code__`. And the newest code, which counts receivers, is not pinned by tests. A fix
  for both is being written and is not committed.
- Round twelve of `recusal-pydantic-ai` found that the walk does not read under
  `functools.partialmethod` over a callable object. See that package's open items. The
  decision whether the fix goes in the kernel is open.
- The changelog entry in #299 lists what the walk does not find by design, under "Not
  found, by design". Those are stated limits, not fixed.

**Where you can help.**

- Read `own_parameters`, `_next_callables` and `reached_parameters` in
  [`recusal/invocation.py`](../recusal/invocation.py) and tell us a callable they read
  wrongly.
- Hand `recusal.invocation.unsupported_signature` a tool wrapped the way your code wraps
  tools, with the decorators of the libraries you use, and report an answer that is
  wrong.
- Read #299 and its "Not found, by design" list, and say which of those shapes exist in
  real code you know.

## `recusal-crewai`

**Where it stands.** `recusal-crewai` 0.2.1 is released (tag `recusal-crewai-v0.2.1`,
2026-11-20). It is tested with the versions in
[`adapters/crewai/tested-with.txt`](../adapters/crewai/tested-with.txt).
[Package](../adapters/crewai/README.md).

**The rounds.** The commits name the fifth to the ninth review. The fixes of the reviews
before the fifth are in one commit. All are in the changelog entry for 0.2.1.

- **Before the fifth review** (commit `47ba412`, 2026-11-19). CrewAI finds a tool by its
  normalised name in more than one place, and the places disagree when two tools
  normalise alike: the policy judged one tool's arguments while another tool ran with
  its own defaults. The gate now reads, in the CrewAI function about to run the tool,
  which tool that is, and refuses a call it cannot tie to it. Two earlier attempts were
  each broken by the next review and were never released.
- **Fifth review** (commit `a0b57ec`). A hook that replaces the argument dict had its
  replacement judged while CrewAI ran its own dict. Refused on every route. A declared
  `__signature__` further down a callable is refused. The tests of this round are under
  the comment "round five" in [`tests/test_crewai_gate.py`](../tests/test_crewai_gate.py).
- **Sixth review** (commit `e40ffd8`). A hook that points the context at another tool had
  that tool judged while CrewAI ran the one it selected. A `(*args, **kwargs)` tracer
  around a `_run` hid a default the schema omits. Both refused.
- **Seventh review** (commit `a50abbf`). Two refusals without cause: a default that is
  not a valid value of its own type (`limit: int = None`), and a `wrapt` wrapper around
  CrewAI's hook entry, which made the gate refuse every call. Both run judged now.
- **Eighth review** (commit `c6aaf75`). A tracer behind a `functools.partial` or a
  callable object's `__call__` hid the wrapped function's defaults. The gate now follows
  every callable a call reaches. A default must be of the exact JSON types.
- **Ninth review** (commit `5db2d7c`). CrewAI validates the arguments once more after the
  gate. The gate now judges arguments only when validating them again gives them back
  exactly, and refuses a schema whose validation would change them.
- **After the ninth** (commit `fd18711`). Documentation only: which MCP tools are refused,
  and where an error raised by a hook is outside the gate.

**Open items.**

- No open finding of a review is recorded in the repository for this package.
- The package still carries its own copy of the walk (`_reached_parameters` in
  [`_gate.py`](../adapters/crewai/src/recusal_crewai/_gate.py)). The changelog entry for
  0.15.0 says it moves to the kernel's walk in its next release. That release is not
  made.
- An MCP tool with an `enum` parameter is refused on every call, because CrewAI builds
  the parameter as a `Literal`. The README says so.

**Where you can help.**

- Review `_calling_site` and `_identity_problem` in `_gate.py`. They read CrewAI's own
  functions to find the tool about to run, and tests pin each local variable and
  attribute they read there.
- Run the adapter's tests against a CrewAI release newer than the pinned one and report
  what fails.
- The scripts of this package's review are not in the repository, so it has no section in
  [REVIEW.md](REVIEW.md). Writing a mutation script for `_gate.py`, as
  [`tools/review/openai/mutate.py`](../tools/review/openai/mutate.py) does for
  `recusal-openai`, would give it one.

## `recusal-google-adk`

**Where it stands.** `recusal-google-adk` 0.2.1 is released (tag
`recusal-google-adk-v0.2.1`, 2026-11-22). Its gate is the one 0.2.0 shipped (tag
`recusal-google-adk-v0.2.0`, 2026-11-21): 0.2.1 adds annotations only. It is tested with
the versions in
[`adapters/google-adk/tested-with.txt`](../adapters/google-adk/tested-with.txt).
[Package](../adapters/google-adk/README.md).

**The rounds.** The commits name six reviews and two confirming rounds, and the tests in
[`tests/test_google_adk_gate.py`](../tests/test_google_adk_gate.py) carry the same labels
in their docstrings ("Review round 1" to "Review round 6", "Confirming round"). All are
in the changelog entry for 0.2.0.

- **The findings recorded on 2026-11-19** (commit `77c3fb2`). Tools the model provider
  runs on its own side never reach a callback, and the gate accepted them. A
  `FunctionTool` subclass could change the arguments after the policy judged them. Every
  `AgentTool` call was refused. Fixed: default-deny on tool kinds, the tool's run path
  held to `FunctionTool`'s own, a plain `AgentTool` judged.
- **First review** (commit `2ff9a03`). Under `RecusalPlugin` with
  `include_plugins=False`, a sub-agent's calls ran unjudged. An `AgentTool` call is now
  judged only when the sub-agent's own calls are judged too.
- **Second review** (commit `af3e42e`). A plugin is handed tool calls only, so a `task`
  delegation and a `code_executor` ran with no policy asked. The plugin now checks every
  agent a run can reach before the run starts.
- **Third review** (commit `eb409fd`). A continued session can start at a sub-agent, so
  the run check now walks the tree from its root. A run whose `RunConfig` sets
  `support_cfc` is refused.
- **Fourth review** (commit `bed63f0`). Judging the `set_model_response` call guarded
  nothing, because the model can give the same answer as plain text that no callback
  sees. An agent's answer is its output and is not judged. The README says so.
- **Fifth review** (commit `5861cd9`). With the ADK's argument validation off, its
  default, the policy judged the model's raw text while the function ran converted
  values. The gate now validates the arguments itself. A model object that turns on the
  provider's own web search is refused.
- **Sixth review** (commit `bb833ce`). A refusal without cause: a parameter whose default
  is `None` failed the gate's validation, so every call to such a tool was refused. It
  is judged as `None` now.
- **Confirming round** (commit `0b0cc58`). A `None` default left for Python to fill ran as
  whatever a wrapper declared for the same parameter. It is left out only when every
  callable that names the parameter declares `None`.
- **Second confirming round** (commit `dc5a365`). The provider web plugin was found as a
  list and not as a tuple. Fixed, with wording in the README.

**Open items.**

- No open finding of a review is recorded in the repository for this package, and no
  round after the second confirming round is.

**Where you can help.**

- Review `_prepared`, `_unseen_reason` and `_provider_tool_option` in
  [`_gate.py`](../adapters/google-adk/src/recusal_google_adk/_gate.py).
- The default-deny on tool kinds means a tool kind added in a later ADK release is
  refused. Run the tests against a newer ADK and tell us which kinds should be judged.
- The scripts of this package's review are not in the repository. A corpus or a mutation
  script for it would give it a section in [REVIEW.md](REVIEW.md).

## `recusal-openai`

**Where it stands.** `recusal-openai` 0.2.2 is released (tag `recusal-openai-v0.2.2`,
2026-11-22). [Package](../adapters/openai/README.md).

**The rounds.** Four rounds. This is the one package whose review scripts are in the
repository, so its rounds, its numbers and the commands that reproduce them are in
[REVIEW.md](REVIEW.md#recusal-openai), each finding with the test that pins it. In short:

- **Round one** (commit `d24c9c3`). The plain loop and the loop that writes an audit log
  did not decide the same way. A function call that carried a namespace ran the
  application's function of the same bare name. A call that was not finished ran.
- **Round two** (commit `b633a04`). A recording keyword without an audit log refused every
  call. A tool name the model chose was written into a refusal and an audit entry with
  no bound.
- **Round three** (commit `071f405`). The package still allowed a kernel without the walk
  of a decorated tool's wrappers. A call id the model chose had no bound.
- **Round four** (commit `b6ef47c`). The commit message says no bypass was found. Its
  tests pin two behaviours that do not change which calls run.

**Open items.**

- No open finding of a review is recorded in the repository for this package.
- The gate has no helper for streaming, and the corpus holds no streamed response.
  [REVIEW.md](REVIEW.md#what-it-does-not-show) says what else the numbers do not show.

**Where you can help.**

- Run the three scripts against your own checkout:
  `python tools/review/openai/corpus.py --summary`, and the same for `keyword_matrix.py`
  and `mutate.py`.
- Write a mutation of `_gate.py` that no test catches. `mutate.py --list` prints the ones
  that exist.
- Add a form of model output the corpus does not generate.

## `recusal-pydantic-ai`

**Where it stands.** `recusal-pydantic-ai` 0.2.0 is released (tag
`recusal-pydantic-ai-v0.2.0`, 2026-11-22), with the fixes of the rounds below, merged in
[#292](https://github.com/philpaz/recusal/pull/292). The review continues: its package
page states the open items below. [Package](../adapters/pydantic-ai/README.md).

**The rounds.** Eleven rounds have a commit each on `review-history/pydantic-ai`, after
the first commit. A twelfth found the open items below.

- **The findings recorded on 2026-11-19** (commit `ef48d32`). An output function ran with
  no policy asked. A toolset wrapper changed the arguments after the gate judged them. A
  call handed to the application to run left the run unjudged. All three are judged or
  refused now.
- **Round one** (commit `36d1c13`). A `Choices` output whose value is callable ran
  unjudged. A wrapper added to the run later sat outside the check. A request that
  offers the model a tool the provider runs is refused.
- **Round two** (commit `c6e026b`). No bypass against a plain setup, several in specific
  ones, most where another capability answers deferred calls. The gate now takes the
  deferred handler itself and judges first.
- **Round three** (commit `2173123`). A gate inside a `WrapperCapability` lost its place,
  so a hook listed after it rewrote the call. The gate now reads the order the run
  dispatches in and refuses a call when an application hook comes after it.
- **Round four** (commit `0d1600d`). A tool given its schema by hand is not validated, so
  a parameter the model left out ran with the function's default, which no policy had
  seen. The defaults are now written out and judged.
- **Round five** (commit `748c5b5`). The same gap was still open on one public form of the
  gate. A function that takes `**kwargs` behind such a tool is refused.
- **Round six** (commit `b49fad1`). Round five had reopened a gap: a keyword default on a
  decorator in the middle of a stack was read by nothing. Every callable the call
  reaches is read again.
- **Round seven** (commit `1778fcd`). When one decorator could not be inspected, the gate
  read only the outermost callable. A stack that cannot be walked is refused as a whole.
- **Round eight** (commit `eb609e4`). Three more ways for a wrapper's own parameter to
  reach a tool unjudged. The rule became: a wrapper must be transparent, or the call is
  refused before the policy runs.
- **Round nine** (commit `0ad5452`). A wrapper was matched by name while Pydantic AI passes
  by position in some cases, so the policy judged one path and the function wrote to
  another. Name and position must agree now.
- **Round ten** (commit `bff524f`). The commit message says no bypass a model can cause
  was found, and six that an ordinary setup causes, most behind a `functools.partial`
  with a keyword. It cites a generated space of decorator stacks. That script is not in
  the repository.
- **Round eleven** (commit `0c846ce`). A `functools.partial` that binds a value by position
  hid the leading parameter of every wrapper under it. Such a partial is allowed only
  over the function itself or over wrappers that take nothing but `*args` and `**kwargs`.

**Open items**, from round twelve. The wrong advice is fixed in 0.2.0; the others are
not, and the package page states the first two.

- **The most serious.** A tool built with `functools.partialmethod` over a callable
  object, such as a class-based decorator, is read one level deep. What is under the
  object is not read, so the rules above for wrappers are never asked there: the policy
  can judge one value while the tool runs another, and a default can run unjudged. Over
  a plain function the same `partialmethod` is read correctly. The gap is in the
  kernel's walk (`own_parameters`, `_next_callables`) and in the adapter's `_steps`.
- **Wrong advice, fixed in 0.2.0.** The README and two refusal reasons said to set
  `__name__` and `__doc__` on a partial. Pydantic AI needs `__name__` and
  `__qualname__`, which they now say.
- **Not judged, now stated on the package page.** A keyword a partial binds that names
  no parameter and lands in the function's `**kwargs`.
- **Documentation.** The docstring of `_with_defaults` does not list every rule it
  applies. The README reads as if a partial's own `__wrapped__` were followed. It does
  not say that decorators built on `wrapt` count as naming a parameter. One refusal
  reason describes a `functools.partial` where the developer wrote a `partialmethod`.
- **Tests.** Some checks are not pinned by a test: in the rule that the signature
  Pydantic AI validates against is the one the gate reads, and in `_as_bound`.

**Where you can help.**

- Read `_steps`, `_as_bound` and `_with_defaults` in
  [`_gate.py`](../adapters/pydantic-ai/src/recusal_pydantic_ai/_gate.py).
- Write the test that reproduces the `partialmethod` item, or a test for one of the
  unpinned checks.
- The generator of decorator stacks is not in the repository. Writing one under
  `tools/review/pydantic-ai/` would let anyone rerun it.
- Hand the gate a tool decorated the way your code decorates tools and report a refusal
  whose reason is not true of your tool.

## `recusal-langgraph` and `recusal-langchain`

The two packages share one gate: `recusal-langchain` uses the gate of
`recusal-langgraph`.

**Where it stands.** `recusal-langgraph` 0.3.0 is released (tag
`recusal-langgraph-v0.3.0`, 2026-11-22). `recusal-langchain` 0.2.0 is released (tag
`recusal-langchain-v0.2.0`, 2026-11-22). The fixes of both were merged in
[#298](https://github.com/philpaz/recusal/pull/298).
Packages: [LangGraph](../adapters/langgraph/README.md),
[LangChain](../adapters/langchain/README.md).

**The rounds.** Seven rounds, by the message of the commit that merged them (`51bbf68`).
The tests carry the labels of rounds one to six: in
[`tests/test_langgraph_gate.py`](../tests/test_langgraph_gate.py) one class for each, with
the round in its docstring, and in
[`tests/test_langchain_agent_gate.py`](../tests/test_langchain_agent_gate.py) in comments
and docstrings. The changelog entries are under `recusal-langgraph` 0.3.0 and
`recusal-langchain` 0.2.0. The entry says no bypass by the model against an ordinary
setup was found.

- **Round one** (`TestWhatRunsIsWhatWasJudged`). Values the function ran with that the
  policy never saw: a call id the model wrote, a wrapper's own default, the default
  behind an injected argument nothing filled. Ordinary tools that were always refused.
  For LangChain: the sync and the async middleware could not be used on one agent, and a
  tool the provider runs reached no middleware.
- **Round two** (`TestWhatTheGateCannotReadIsRefused`). Arguments handed on to code the
  gate could not read: a decorator built without `functools.wraps`, an `_arun` that hands
  the call to `_run`. And an answer that did not come from the execution the gate
  judged.
- **Round three** (`TestForwardersAndTheNamesLangChainKeeps`). A function that takes only
  `*args` and `**kwargs` under the schema LangChain infers from it. Refusal reasons that
  were not true of the call they refused.
- **Round four** (`TestAForwarderThatNamesOnlyWhatOthersFill`). A forwarder whose only
  named parameter is one the framework or the host fills.
- **Round five** (`TestAReservedNameIsNeverAParameterOfItsOwn`). One more forwarder of the
  same family. The family was closed by rule: a parameter the framework or the host
  fills is never the function's own.
- **Round six** (`TestTheFrameworksAnswerIsNeverSetAside`). Round five's rule read a
  parameter's raw annotation, so an alias or a forward reference was read wrongly. The
  rule now reads annotations as the frameworks do.
- **Round seven.** The commit message says it found no call a model can cause to run
  unjudged and none an ordinary setup causes.

**Open items.**

- The latest round found no security finding.
- What the gate does not read is listed in each README. For LangGraph that includes a
  function that names its parameters and then calls another function, a default an MCP
  server applies to an argument the model left out, and `runtime.tool_call_id`.
- A function that takes only `**kwargs` under a schema is refused, because the gate
  cannot tell it from a forwarding decorator. These ran judged before 0.3.0. The README
  says what to give the node instead.

**Where you can help.**

- Review `_read_to_the_end`, `_not_its_own` and `_filled_type` in
  [`_gate.py`](../adapters/langgraph/src/recusal_langgraph/_gate.py): the forwarder rule
  of rounds three to six.
- The gate depends on private LangChain and LangGraph names, which the README lists.
  [`tests/test_compat_langgraph.py`](../tests/test_compat_langgraph.py) holds the probes
  for them. If a public API can replace one of those names, tell us.
- Report a published tool or decorator that is refused and should not be, with the
  reason the gate gave.
- The scripts of this review are not in the repository.

## `recusal-openai-agents`

**Where it stands.** `recusal-openai-agents` 0.2.0 is released (tag
`recusal-openai-agents-v0.2.0`, 2026-11-22), with the fixes of the rounds below, merged in
[#297](https://github.com/philpaz/recusal/pull/297). The review continues: its package
page states the open items below. [Package](../adapters/openai-agents/README.md).

**The rounds.** `review-history/openai-agents` holds a first commit and the commits of
rounds two to four. A fifth round found the open items below.

- **The first commit** (`bfc41e5`). For one kind of tool the gate named every argument
  one parameter off, so the policy saw the SQL under the wrong name and it ran. The gate
  now reads the names from the SDK's own schema for the tool. A wrapper that adds or
  takes a parameter is refused when the agent is built. The gated copy of an agent
  shared its `handoffs` list with the original. It has its own now.
- **Round two** (commit `dcbe813`). A `functools.singledispatch` tool with a registered
  implementation, a mapping keyed by another type than `str`, and model settings that
  add tools through `extra_body` are refused when the agent is built.
- **Round three** (commit `bab48e3`). An agent used as a tool with a `parameters` model
  was judged on the model's raw JSON while the sub-agent received the validated input.
  The policy is now handed what the sub-agent is given.
- **Round four** (commit `e216933`). A single-dispatch function used as a
  method is recognised, and the refusal of an agent tool's `parameters` type names its
  reason.

**Open items**, from round five. All are about `functools.singledispatch`. None is fixed
in the code; 0.2.0 corrects the sentence the first item names and states the items on the
package page.

- **The most serious.** A single-dispatch tool with nothing registered is accepted. The
  SDK checks every call of a model turn before any tool body runs. So an implementation
  registered during the same turn, by another tool call or by the tool's own first use,
  runs without having been judged. The README and the changelog had said an
  implementation "registered afterwards refuses its next call", which is not true in that
  case; they now say so, and say not to register implementations while an agent runs.
  The decision is taken and not yet built: refuse every single-dispatch function when
  the tool is gated, whether or not anything is registered.
- A single-dispatch function that is a callable object's `__call__`, under a
  `functools.wraps` wrapper, is not found. `_wrapped_chain` does not follow `__call__`,
  and the kernel's walk does.
- A single-dispatch function is recognised by one code object read at import. If
  `functools` is reloaded, or `functools.singledispatch` was replaced before the import,
  one is not recognised, and one kind of replacement makes the package's import raise.
- **Messages and documentation.** `_agent_tool_parameters_reason` names the wrong cause
  for some fields: a pattern field, a list of nested models, a model with a validator.
  Two sentences of the README overstate what is found.
- **Tests.** Some checks are not pinned by a test, and the items above need tests.

**Where you can help.**

- Read `_wrapped_chain`, `_registered` and `_agent_tool_parameters_reason` in
  [`_gate.py`](../adapters/openai-agents/src/recusal_openai_agents/_gate.py).
- Write the test for the same-turn item: two calls in one turn, in both orders.
- Run the adapter's tests against an `openai-agents` release newer than the one in
  [`adapters/openai-agents/tested-with.txt`](../adapters/openai-agents/tested-with.txt).
- The scripts of this review are not in the repository.

## `recusal-openai-agents-mcp`

**Where it stands.** `recusal-openai-agents-mcp` 0.2.0 is released (tag
`recusal-openai-agents-mcp-v0.2.0`, 2026-11-22), with the fixes of the rounds below,
merged in [#297](https://github.com/philpaz/recusal/pull/297) with those of
`recusal-openai-agents`. It requires `recusal-openai-agents` 0.2.0. The review continues:
its package page states the open items below.
[Package](../adapters/openai-agents-mcp/README.md).

**The rounds.** The same commits as `recusal-openai-agents`, on
`review-history/openai-agents`: a first commit, rounds two to four, and a fifth round that
found the open items below.

- **The first commit** (`bfc41e5`). The gate judged the arguments as sent while the
  server converted types and filled its schema's defaults: `dry_run` sent as the string
  `"false"` passed a policy and ran as `False`, and an empty call ran the server's
  defaults. The arguments are now checked against the tool's listed schema and completed
  with its listed defaults, and the completed arguments are judged, recorded and sent.
- **Round two** (commit `dcbe813`). The first schema check was too weak: a value the
  server would convert one level down, and a default nested inside an object, ran
  unjudged. The gate now validates against the whole listed schema and writes every
  listed default at every depth.
- **Round three** (commit `bab48e3`). A default listed inside an alternative of a
  property's own `anyOf`, the shape zod publishes, ran unjudged. JSON text for a
  parameter that may be null ran as null. And checking a call against a schema that
  refers to itself could hold the event loop for minutes. All three fixed.
- **Round four** (commit `e216933`). The gate had predicted from the schema
  which form of a union a server runs, and the two SDKs do not follow the schema there.
  The class is closed by rule: an object is refused where the schema offers more than
  one object form, unless a tag tells them apart. The gate no longer evaluates a
  server's regular expressions on the model's text. The commit cites generated spaces of
  calls against a zod server and a Python server. Those scripts are not in the
  repository.

**Open items**, from round five, as 0.2.0 was released. The untrue sentence is corrected
in 0.2.0, and the 0.2.0 package page states the others; none of the others is fixed in
0.2.0. The changes that answer them, and the findings of a sixth round on those changes
(JSON text nested deeper, or holding a longer integer, than the gate's own Python
decodes ran as a list on a server under another Python; and the time one call takes
grew with what the model sent, which a ceiling on the steps of one call now bounds),
are in the changelog's Unreleased entry for 0.2.1, which is not released.

- **The most serious.** In 0.2.0 a model can still stall the agent's process. Under
  `uniqueItems`, integers chosen to share one hash make the uniqueness check slow, and
  the step allowance does not count that work (`_unique_items`, `_key` in `_schema.py`).
- **Not seen, now stated on the package page.** A zod server that converts text
  (`z.coerce`) beside another type in a union reads a string by JavaScript's rules, and
  the gate reads number text by Python's. A Pydantic server converts the keys of a `dict`
  parameter, and the key type is not in the listing. An integer too large for a double on
  a zod `integer` parameter, which the README now names beside a `number` parameter. A
  member the schema lists that a Pydantic server ignores or reads from elsewhere, and a
  union a Pydantic server reads left to right (`union_mode='left_to_right'`).
- **A sentence that was not true, corrected in 0.2.0.** The README, the module text and
  the changelog said the schema is read once, when the listing is read. It is read at the
  first call after each listing, which they now say.
- **Wrong advice.** The refusal reason "Send the value as JSON, not as text" is wrong
  where the text is not that number, such as a zip code. The reason should tell the
  server's author to type the parameter `str`.
- **Cost, to be stated.** The rule on JSON text also refuses strings on zod servers,
  which never decode them. Checking `unevaluatedProperties` on a very large hand-written
  schema is slow. The gate costs many times the server's own validation.
- **Tests.** Not pinned by a test: `_stray` with `unevaluatedProperties`, and the upper
  side of `_STEPS_PER_VALUE`.
- **Not verified.** Whether a hand-written `anyOf` of two `required` lists is refused by
  the union rule.

**Where you can help.**

- Read `_unique_items`, `_key`, `_number_text` and `_members` in `_schema.py` on the
  branch `fix-openai-agents-gates`.
- Point the gate at an MCP server you run, list its tools, and report a call that is
  refused though the server would run it, or a value the server ran that the policy was
  not shown.
- The two generators, for a zod server and for a Python server, are not in the
  repository. Writing them under `tools/review/` would let anyone rerun them.
- The changelog entry names the servers the findings were reproduced against: one built
  with the MCP Python SDK and one built with the TypeScript SDK and zod. A server built
  another way is not in the record. Try yours.

## How to help

- **Read the code.** Each section above names the functions that carry the rules. A gate
  is one file in each package: `adapters/<package>/src/<module>/_gate.py`.
- **Run the review scripts against your own stack.**
  [REVIEW.md](REVIEW.md#reproducing-a-number) has the commands. Today they exist for
  `recusal-openai`. Scripts for another package are a welcome contribution.
- **Report a security gap privately**, not in a public issue:
  <https://github.com/philpaz/recusal/security/advisories/new>.
- **Contribute.** [CONTRIBUTING.md](../CONTRIBUTING.md) says how, and what a good pull
  request looks like. The open items above are a list of places to start.
- **Credit.** Contributors are credited in the [changelog](../CHANGELOG.md) and in the
  [Contributors](../README.md#contributors) section of the README.
