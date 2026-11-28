# Review log

This is the record of the in-depth review of each Recusal package, round by round: what
each round found, what changed, what is still open, and where you can help. It covers
2026-11-19 to 2026-11-27, the date of the newest change it describes.

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
- This page states no count of generated cases, with one exception that says so
  where it stands: the confirmation pass on the kernel change after 0.16.0, whose
  counts are that pass's own report. Several rounds ran generated test spaces, and the
  scripts that produce them are in the repository for `recusal-openai` only
  ([`tools/review/`](../tools/review)). Adding the others is one of the ways to help.

## The kernel: `recusal`

**Where it stands.** `recusal` 0.16.0 is released (tag `v0.16.0`, 2026-11-26). It has
the change of the rounds below, merged in
[#299](https://github.com/philpaz/recusal/pull/299); `recusal` 0.15.1 (tag `v0.15.1`,
2026-11-23) has none of it, and its `recusal.invocation` is that of 0.15.0. The
commits named under the rounds below are the change as each round read it; they are
kept on the branches `review-history/kernel-299-pre-rebuild` (`293f22f`) and
`review-history/kernel-299-rounds-6-8` (all the others). A further change, which
answers what two adapter reviews found open in the walk of 0.16.0, was merged from the
pull request [#329](https://github.com/philpaz/recusal/pull/329) and is released in
`recusal` 0.17.0 (tag `v0.17.0`, 2026-11-27). A change that answers what two adapter
reviews found in 0.17.0 is in progress on the pull request
[#336](https://github.com/philpaz/recusal/pull/336), not merged; it is the last entry
under the rounds.

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
- **The change of #299, as first read** (commit `293f22f`; merged with the rounds after
  it, released in 0.16.0). Three adapter reviews found the same gap: a
  decorator that applies a second wrapper inside itself calls it from its closure, and
  the walk never read that wrapper. A `functools.singledispatch` function runs an
  implementation chosen by the type of an argument, and the walk read only the base one.
  The change refuses both (`HiddenWrapper`, `DispatchedAtCallTime`), reads nothing it
  cannot read with certainty, and lists what it does not find. The commit message says
  four reviews shaped it. The review after those four held it on two items: importing
  `recusal.invocation` raised in a few environments where 0.15.0 imports (a replaced
  `functools.partial`, a `sys.modules["functools"]` that is not a module, an audit hook
  that raises on `__code__`), and the code that counts receivers was not pinned by
  tests. Both are fixed in commit `f6162a3`, the change rebuilt on main: nothing is
  prepared at import, and the checks are settled once at the first walk.
- **`functools.partialmethod`** (commit `4650c71`). Round twelve of
  `recusal-pydantic-ai` found that the walk read one level for a `partialmethod` over a
  callable object and nothing under the object. Closed in the kernel, so that every
  adapter that uses the walk gets it, and in `recusal-crewai`'s own walk. The adapters
  report the kernel's named refusals as what they are.
- **A wrapper held in a module global** (commit `a4aaf5d`). A wrapper that calls a
  second wrapper, or a single-dispatch function, of the same function by its global
  name is refused like one that holds it in a closure. The `partialmethod` refusal got
  a public name (`PartialMethodSignature`), and `unsupported_signature` gives it before
  any other reason.
- **Kernel review round six** (commit `c40ee87`). No category 1. Four shapes the walk
  read wrongly, each accepted by 0.15.0 too: a `__wrapped__` carried by a
  `functools.partial` was not followed; `functools.wraps` over `functools.lru_cache`
  over a wrapper was read as the function the cache was named for; the `__call__` of a
  class-based decorator written inside the decorator was not read for what it holds;
  and a method that delegates to a free function under `functools.wraps` of it was read
  as handing its object on. All four are closed. One regression of the change itself
  was found and fixed: the `partialmethod` check asked `functools.partialmethod` about
  every callable, so a process with another object under that name refused every tool.
  Five sentences of the changelog and the docstrings were corrected. After it, a cache
  that was named again is read as the wrapper it calls, which is picked out of what the
  cache refers to by its place, in place of refusing every such cache (a caching
  decorator written as `functools.wraps(f)` over `functools.lru_cache` over a
  forwarding wrapper was refused though nothing unjudged ran); a cache whose function
  cannot be told has a refusal of its own (`UnreadCache`); and what a cache calls is
  told once per cache (commit `e9b3dcb`).
- **Kernel review round seven** (commit `e9b3dcb`). No category 1; round six's four
  confirmed closed, and nothing 0.15.0 refused accepted in the generated shapes, which
  went to three decorators deep and to async and generator wrappers. Two category 2. The fourth shape of round six came back when the free
  function wore one decorator that only passes its arguments on (a forwarding wrapper,
  a class-based decorator, a caching decorator): the method was accepted, and the free
  function ran its first parameter at a default no signature showed. And over a method,
  a cache named again that calls something other than a plain function (a
  `functools.partial`, a callable object, a second cache) was still read by its name.
  Four regressions of the change itself: `import gc` at the top of the module raised
  where a process had None under that name in `sys.modules`; an audit hook that stopped
  the listing of a cache with a `BaseException` stopped every tool with a cache; another
  function of `gc` under `get_referents` refused every cache; and on Python 3.9 to 3.11
  `list(d.items())` could raise out of the walk when a collection it set off changed
  the dictionary. Also: 41 clean shapes refused with a reason that did not help (a
  callable object that defines `__get__`), four surviving mutants, eight sentences that
  were untrue or incomplete, and a walk that cost 1.3 milliseconds for five stacked
  decorators in a module of 5,000 globals. Closed in commit `7ab31c3`: the name a bound
  argument was taken under goes with it through every layer that only passes arguments
  on, so the delegating method is refused under any number of them. A method over a
  cache named again is read as what the cache calls. The two imports 0.15.0 did not
  make are guarded and never made at a walk; a listing a hook stops, whatever it raises
  but an interrupt from the keyboard, and a replaced `gc.get_referents` leave the cache
  read by its name; a listing a collection interrupts is made again. What is kept
  between walks is a plain `dict` by `id()` beside a weak reference, in place of a
  `weakref.WeakKeyDictionary`. The module globals a wrapper names are looked up by
  name, with the names kept for each code object. The `ValueError` for a callable
  object that defines `__get__` says the cause and what to do.
- **Kernel review round eight** (commit `7ab31c3`, whose source is that of `322ee13`).
  No category 1; round seven's two category 2 and four regressions confirmed closed,
  and nothing 0.15.0 refused accepted in the generated shapes. Held on one category 2
  that 0.15.0 has too: a wrapper that keeps one bound argument moves every argument
  bound after it up one place, and the walk tested the parameter the kept argument
  would fill where the one left without a value is the one the LAST bound argument
  would fill. `functools.partial(tools.search, "acme")` over a method that delegates to
  `def search(tenant, region="eu", limit=10)` declared `(limit=10)`, was judged
  `{'limit': 5}` and ran `region='eu'`. The review's own generator had filed these rows
  as a stated limit, by a name in the row. Category 3: a false refusal on Python 3.9
  only (a decorator over the bound delegating method, which 3.9 shows with every
  parameter of the free function); a refusal of a facade that runs correctly, because
  it passes a field of its object where the other facade passes nothing; any built-in
  type put under `weakref.ref` was used as the weak reference type, which raised from
  the second walk; two costs not stated (the first walk of a new cache on a
  free-threaded build, and the memory entries of dead objects hold until the next
  sweep); seven surviving mutants of 28 (two test gaps, five lines that were not
  needed); and nine sentences that were untrue or incomplete.
- **Kernel review round nine** (commit `f712d61`). No category 1. Round eight's
  category 2 and its 3.9 false refusal confirmed closed; the change for a signature
  declared on the class of what a method binds and the `detached` that no longer calls
  itself confirmed (the second compared with the version that called itself: no
  difference in 105,587 cases on Python 3.12, 3.9 and 3.14 free-threaded). Nothing
  0.15.0 refused accepted in the generated shapes or the scans. Verdict: release, with
  one more member of the bound-argument family reported for the maintainer's choice,
  present in 0.15.0: a facade that hands its bound argument on BY KEYWORD over a
  wrapper that keeps a leading named parameter (declared `(limit=10)`, judged
  `{'limit': 5}`, ran `limit=10`; the same through `recusal_openai.judge_and_run`).
  Also, each constructed and present in 0.15.0: a `__signature__` a class answers for
  in `__getattr__` or `__getattribute__` (a forwarding proxy), and a
  `__text_signature__` on an object whose class defines `__get__`, are honoured by
  `inspect.signature` and were not seen by the check; a callable object whose
  metaclass declares `__signature__` was refused with a reason untrue of it. To state:
  `recusal-pydantic-ai`'s plain gate counted on the `RecursionError` that `detached`
  no longer raises (no call ran without its policy being called); and a sentence on
  `detached` was inexact (a value's metaclass `__eq__` runs when its type is told).
  Not covered: the timing and memory figures in the changelog, mutation testing,
  generator modes on the free-threaded build, pydantic-ai output functions with deep
  data, and the origin of exceptions that escape four framework runners at depth,
  which are the same beside 0.15.1.
- **Kernel review round ten** (commit `5fba40c`, the change that closes round nine).
  No category 1. Round nine's findings confirmed closed on Python 3.9 to 3.15, with no
  clean row newly refused in 71,474 generated shapes of the keyword hand-on. Verdict:
  release, after one defect of the change itself is fixed: the first form of the
  text-signature rule refused every callable written in C once it was bound as a
  method (`types.MethodType(len, obj)`, a `classmethod` of a built-in), with a reason
  untrue of it, which changed 2,243 of 2,715 outcomes of a scan of the standard
  library. One more member of the bound-argument family, present in 0.15.0: the
  keyword hand-on where the keeping wrapper is HELD by the facade, which the rule made
  where the walk binds never reaches. And, constructed and present in 0.15.0: an
  object that shows a code object, a name and defaults is read by `inspect.signature`
  as a function, which hides a default (no parameter: the walk read its `__call__`
  too). Three sentences inexact. Not covered: the two-deep generator modes, the
  hostile-environment scenarios and the probes of rounds one to five on this commit,
  a real Cython function with an embedded text signature, timing, mutation testing.
- **Kernel review round eleven** (commit `788d632`, the change that closes round ten).
  No category 1. Round ten's findings confirmed closed on Python 3.9 to 3.15: no
  outcome of a callable written in C differs from 0.15.0, bare or bound, in the
  standard library or the scanned packages; the held keyword hand-on is refused held
  through a cache, a `functools.partial`, a pass-through decorator and a cache named
  again; six clean method decorators that hold a second wrapper are read. Verdict:
  release, with the last member of the bound-argument family the review could find,
  present in 0.15.0: a held wrapper whose leading parameter is its own, has a default
  and is not declared by the tool, which runs with that default when the holder hands
  the bound argument on by keyword. The review stated the family in one sentence (the
  changelog quotes it) and the one shape of it that stays a stated limit. Also: the
  lookalike rule left out an object whose `__wrapped__` is None; and a limit that was
  nowhere stated, in ordinary code and in 0.15.0: the walk compares names, not
  defaults, so a wrapper that declares a parameter again with another default runs
  that default when the call leaves the parameter out. Not covered: the two-deep
  generator modes, the kernel's test file and the keyword hand-on generator on this
  commit, hostile-environment scenarios, timing, mutation testing.
- **Kernel review round twelve** (commit `47670de`, the change that closes round
  eleven). No category 1. Round eleven's findings confirmed closed. On this commit
  every generator mode was run to the end, 3,879,764 generated shapes in nine runs
  (one, two and three wrappers deep, async and generator wrappers, third-party
  decorators, Python 3.12 and one mode on 3.9): no verdict or outcome differs from
  round nine's, and nothing 0.15.0 refused is accepted; the scans of the standard
  library and the installed packages differ from 0.15.0 only in the single-dispatch
  functions refused on purpose and two objects that are not tool shapes. Verdict:
  release, after two small things. One defect of the change itself: the held rule
  counted the places a held `functools.partial` binds itself as places a holder could
  hand on by keyword, and refused a decorator that holds
  `functools.partial(retrying(function), 5)` with a reason untrue of it. And one more
  member of the bound-argument family, present in 0.15.0: a facade that takes its
  receiver in `*args` raises no name doubt, so the rule for a wrapper the walk reads,
  made only under a doubt, did not act, where the rule for a held wrapper did. The
  review tried the rule without the doubt on every callable; see the open items.

- **Kernel review round thirteen** (commit `65370a2`, the change that closes round
  twelve). No category 1. Round twelve's findings confirmed closed; the nine generator
  runs and the scans as in round twelve, with no verdict or outcome changed; ten
  ordinary shapes of decorated methods and partials read, eight of them on Python 3.9
  (the other two, a decorator over a bound method, not compared with 0.15.0 there).
  The reviewer corrected two of its own earlier counts: the kernel's tests it had
  reported as passing under its trial fixes in rounds nine and twelve had been run
  against the unpatched tree. Verdict: release, after one change. One more member of
  the bound-argument family, constructed and present in 0.15.0: the rule for a
  wrapper the walk reads tested the LAST place the bound arguments would fill, where
  the rule for a held wrapper tests each; with two arguments bound by one
  `functools.partial` and both handed on by keyword, the judged name in the first
  place was not seen. Apart from that the review found nothing outside the stated
  limit. Not covered: its trial of the fix on the two-deep modes and on 3.9, the
  kernel's tests on five of the eight interpreters, hostile-environment scenarios,
  timing, mutation testing.

- **Kernel review round fourteen** (commit `386e21b`, the change that closes round
  thirteen). No category 1. Round thirteen's finding confirmed closed on Python 3.9
  to 3.15; the nine generator runs and the scans unchanged; no clean shape newly
  refused in 71,474 generated shapes of the keyword hand-on or in 27 ordinary shapes
  of decorated methods and partials. The review compared the rule for a wrapper the
  walk reads with the rule for a held one: the same for a wrapper that passes the
  call on, with two differences left. One is a member of the bound-argument family,
  constructed and present in 0.15.0: a held wrapper that passes the call on to
  nothing, with a default of its own for its first parameter, under a holder that
  keeps its receiver, ran with that default; the same wrapper read by the walk was
  refused. The other: the held rule excepted the names a framework injects and the
  read rule did not. Verdict: release, and tag once the first is refused or stated.
  Apart from it the review found no shape that is neither refused nor inside the
  stated limit. Not covered: a trial of the refusal, the kernel's tests on five of
  the eight interpreters, hostile-environment scenarios, timing, mutation testing.

- **Kernel review round fifteen** (commit `e05977e`, the change that closes round
  fourteen). No category 1, no category 2, and the reviewer would tag the commit.
  Round fourteen's finding confirmed closed on Python 3.9 to 3.15. The rule for a
  wrapper the walk reads and the rule for a held one are the same for a callable that
  passes the call on; for one that passes the call on to nothing they differ in the
  one stated way (a defaulted place the tool does not declare is refused on a held
  wrapper always, and on one the walk reads only under a name doubt). The review
  found no shape, on either path, that is neither refused nor the stated limit in
  substance. In the nine generator runs no clean shape is newly refused and nothing
  0.15.0 refused is accepted; in each of the three two-deep runs, two shapes that
  were accepted while something unjudged ran are refused now. Of 56 method
  decorators that hold a second wrapper (14 decorators, among them
  `pydantic.validate_call`, `tenacity`, `wrapt` and `cachetools`, on four kinds of
  tool) one differs from round fourteen: the stated twin. Removing the exception for
  injected names changed no verdict found: only `recusal-google-adk` passes such
  names to the kernel. One sentence was narrower than the limit it states (the
  parameter left at its default need not be the leading one) and is widened. Not
  covered: the kernel's tests on five of the eight interpreters on this commit,
  hostile-environment scenarios, timing, mutation testing, and the 135 generated
  shapes of the stated limit one by one (classified by the wrapper they contain).

**Open items.**

- After round fifteen, only the sentence that states the limit was changed. No code
  has changed since the commit round fifteen reviewed.
- After round fourteen, a change reviewed in round fifteen closes what it found: a
  held wrapper's leading parameter in the place of a bound argument is refused where
  it has a default the tool does not declare, whether or not the wrapper passes the
  call on; and the held rule no longer excepts an injected name. Refused by it
  although it runs correctly, and stated in the changelog: a `functools.partial`
  that binds by position the defaulted first parameter of a function whose decorator
  holds a wrapper that names the parameter again.
- After round thirteen, a change reviewed in round fourteen closes what it found: on
  a wrapper the walk reads that passes the call on, every place the bound arguments
  would fill is tested.
- After round twelve, a change reviewed in round thirteen closes what it found: the
  places a held `functools.partial` or method fills itself are not in doubt; and a
  wrapper the walk reads that passes the call on is refused, doubt or no doubt, where
  the last parameter the bound arguments would fill on it is one the tool declares or
  has a default, which is the rule a held wrapper gets.
- Tried and not kept: refusing bound arguments wherever the last parameter they would
  fill has a default, on every callable and without a name doubt. It refused no clean
  generated shape, and closed the stated limit below as well. It also refused four
  shapes the kernel's own tests pin as ordinary, among them
  `functools.partial(traced(table), "orders")` over `def table(table="users", ...)`,
  a `functools.partial` of a decorated method that binds a defaulted parameter, and a
  facade that hands its receiver on to a function that takes it as a defaulted
  parameter. Those have the signature of the stated limit and run correctly. (The
  review reported the kernel's tests as passing under its trial; they had been run
  against the unpatched tree.)
- Stated limit, by design: a wrapper the walk reads that takes a bound argument in
  `*args`, or under the very name the function has for it, and keeps it or hands it
  on by keyword, over a function the walk reads that passes the call on to nothing,
  where a parameter the bound arguments would fill there has a default the tool does
  not declare (the first, or a later one, since a kept argument moves the ones bound
  after it): the function runs at that default.
  `functools.partial(dropping(table), "orders")`, where `dropping` calls
  `function(*args[1:], **kwargs)`, is read as the partial over a pass-through
  decorator is, and ran `table="users"`.
- After round eleven, a change reviewed in round twelve closes what it found: a held
  wrapper's leading parameter in the place of a bound argument is refused where it has
  a default, as where the tool declares it; a lookalike whose `__wrapped__` is None is
  refused; the limit on defaults is stated in the changelog.
- After round ten, a change reviewed in round eleven closes what it found: a method
  is asked for a text signature as what it binds; a held wrapper that passes the call
  on and has a parameter the tool declares in the place of a bound argument is
  refused; an object called through a `__call__` written in Python that shows a code
  object and names nothing as `__wrapped__` is refused; the three sentences are
  corrected.
- After round nine, the maintainer chose to close its findings before release, in a
  change reviewed in round ten: bound arguments under a doubt are refused where
  the last parameter they would fill is one the tool declares and the callable that
  has it passes the call on; the declared-signature check makes the lookup
  `inspect.signature` makes, so a proxy that answers for `__signature__` is refused
  and a signature only the metaclass declares is not; a declared `__text_signature__`
  is refused. Two more shapes than the review found are closed in it: a text signature
  on a plain function, which `inspect.signature` reads on Python 3.9 to 3.15 (and on
  any callable object from 3.13), and a `__signature__` set on a cache wrapper or a
  `staticmethod` object, which the walk looked through without asking. The
  `recusal-pydantic-ai` difference and the `detached` sentence are stated in the
  changelog.
- Not decided: whether `recusal-pydantic-ai`'s plain gate should bound the depth of
  the arguments it hands to a policy, now that the kernel does not stop them.
- Not decided: whether `detached` and `json_data` should tell a value's type without
  the comparison a metaclass can answer (`type(value) in (...)`). A model's arguments
  are parsed JSON and never of such a class.

- After round eight, a change committed as `837047e` (on top of the rebuilt round-seven
  change, `d4ff8da`) closes the review's findings. The walk
  refuses bound arguments under a doubt where the LAST parameter they would fill has a
  default the tool does not declare, and the doubt stays with the arguments bound after
  the doubted one and with the same arguments further in; one more shape of it than the
  review found is closed (a wrapper between the facade and the function that hands the
  next argument on, under one partial that binds both). A parameter the tool declares
  is not refused, unless a wrapper on the way (one the walk reads or one a decorator holds) takes it by name itself, which ends the
  false refusal on 3.9. Only the interpreter's own weak
  reference type is used. A cache laid out in neither known way is refused outright.
  Seven simplifications the review listed are applied. The two test gaps are pinned,
  and the five lines that were not needed are gone. The sentences are corrected and
  the two costs stated. Reviewed in round nine.
- Simplification 8 was tried and not kept: removing the legacy reading of a method over
  a cache named for a non-function lost the `DeclaredSignature` refusal of a class-level
  `__signature__` and accepted stated-limit holders that run an unjudged default. The
  legacy reading stays.
- Found while trying it, present in 0.15.0, and closed in a change reviewed in round
  nine: a callable object whose class declares `__signature__`, bound as a method
  (by its own `__get__`, by `types.MethodType`, by a `classmethod`, or under a
  `functools.wraps` wrapper used as a method), was read as the signature it declared,
  and a parameter that signature left out ran at its default. The walk now looks at
  what a method binds as it looks at the method. Refused although it runs correctly,
  by decision: a bound object that declares a signature, whatever the signature says,
  among them a method over a cache named for the method that calls such an object.
- Refused although it runs correctly, by decision: a facade whose method passes a value
  of its own in the place of the object
  (`return search(self.tenant, *args, **kwargs)` under `functools.wraps(search)`, the
  first parameter defaulted) has the signature of the facade that keeps the object, and
  the walk reads parameters, not bodies. 0.15.0 accepts it. The changelog, the message
  and the exception say so, with the remedy: register a plain function that names every
  parameter the tool takes and calls the facade.
- A wrapper that takes the object of a method under the same name as the first
  parameter of the function it wraps, and does not hand it on, is read as handing it
  on: the two cannot be told apart from signatures. Stated in the changelog; not fixed.
- The changelog entries for 0.16.0 list what the walk does not find by design, under
  "Not found, by design". Those are stated limits, not fixed.
- Not reviewed: stacks more than three decorators deep. Rounds seven and eight ran
  generated shapes to three deep, with async and generator wrappers.
- **After 0.16.0: what two adapter reviews found open in its walk** (the pull
  request #329, merged; released in 0.17.0).
  The review of `recusal-openai-agents` 0.2.2 and a review of `recusal-pydantic-ai`
  beside 0.16.0 found the same gaps by different routes, each reproduced on 0.16.0
  before it was changed:
  - A `weakref.proxy` on the way was read through as the object it stands for, so what
    the walk tells by type was not told: a cache named again behind it was read by its
    name, a dispatcher with an implementation registered as its base, and the
    `__call__` of a class-based decorator not at all. An object that answers for
    `__wrapped__` without holding one had the same effect: a proxy written in Python,
    and the ordinary class-based decorator that keeps the function in an attribute and
    hands every other attribute on to it with `__getattr__`. Both are refused
    (`WeakProxy`, `ProxiedCallable`), the second although such a decorator runs
    correctly; with `functools.update_wrapper(self, fn)` in its `__init__` it is read.
  - A single-dispatch function with nothing registered was read as its base, and a
    wrapper that registers an implementation on its first use ran it for the call that
    was judged. Refused (`OpenDispatch`), although one nothing is ever registered on
    runs correctly.
  - A held wrapper that takes the tool's parameters in another order was read as
    passing them on. Refused where it takes, in a place an argument passed by position
    can fill, another name than the tool declares there. Under a bound argument (a
    method's receiver, a positional argument of a partial) a leading place named as
    the tool names a parameter is counted as that parameter's, and a place named
    otherwise as the bound argument's.
  - A held wrapper with a parameter of its own was not found where its way to the
    function passed a class-based decorator, a callable object, a `functools.partial`
    that carries a `__wrapped__`, a weak proxy, a cache named again for another
    function, or a second held wrapper. Each is followed.
  - Two reasons were not true: a `DeclaredSignature` said nothing, and gates wrote "a
    `__signature__`" for a `__text_signature__`; and the reason for a bound argument
    said a parameter "runs at a default" where it has none. Both say what holds.
  - Two public functions are added so that an adapter needs no walk of its own beside
    the kernel's: `reached_callables` and `cache_target`. No adapter uses them yet, so
    `recusal-pydantic-ai` still refuses the stack the changelog names, which the kernel
    reads.
  - Pinned by the tests named `test_o_` and `test_p_` in
    [`tests/test_invocation.py`](../tests/test_invocation.py). The changelog entry
    for 0.17.0 lists what is refused although it runs correctly and what is
    still not found.
  - **One round on the change itself**, on its first form (commit `d67e744`). It found
    no tool that 0.16.0 refuses and the change reads, and these, each closed in the
    commit after it and pinned by the tests named `test_v_`:
    - The order rule was wrong under a bound argument. Under a method it took the
      first place of a held wrapper for the receiver's whatever its name, so a held
      `def second(force, sql)` whose decorator keeps the receiver was read and ran
      swapped, and the same decorator over `def second(sql, force)` was refused for a
      place that wrapper does not take. The rule above is the corrected one.
    - A held wrapper reached two ways, with another count of bound arguments each,
      was judged for the first way alone, so an attribute nothing calls could hide a
      parameter of the wrapper's own. It is judged for each way.
    - Read by 0.16.0 too, and closed: an object whose `__class__` shows a function, a
      method or a partial it is not; a subclass of `functools.partial` with a
      `__call__` of its own; a cache whose `__wrapped__` is None or deleted. And two
      exceptions that were no refusal: the `ReferenceError` of a proxy whose object is
      gone, and whatever a `__getattr__` raised for `__wrapped__`.
    - Refused although it ran correctly, and read again: a class-based decorator with
      some hundreds of plain attributes, a wrapper that holds hundreds of harmless
      callable objects (the search stopped at 512 values of any kind), and a held
      proxy of a partial or of a bound method the walk had read.
    - Sentences that were not true of the code, in the changelog, in this log and on
      the `recusal-openai-agents` page: corrected.
  - **A confirmation pass on the corrected change** (commit `9516c7f`), again a review
    of the change by the project and not one from outside it. It changed no behaviour:
    what it found is sentences and stated limits, corrected in the commit after it,
    which differs from `9516c7f` in `recusal/invocation.py` by docstrings and comments
    alone. The counts below are the pass's own report; its scripts are not in the
    repository.
    - What it ran again: the probes of the first pass, unchanged; and its sweep of
      generated stacks, 979,496 of them, compared three ways. Refused by 0.16.0 and
      read now: 0. Read again since the first form of the change: 6,266 stacks, in the
      two classes that were meant to be (a held wrapper in the tool's own order under
      a bound argument, and a held proxy of a partial or a method the walk stood on),
      none of them misrunning because of it. Accepted while a call misruns: 733 stacks
      of one stated class (a proxy that forwards a call and names nothing as
      `__wrapped__`, over a `functools.partial`), accepted by 0.16.0 too. It made
      25,620 calls of the public functions on callables whose hooks raise: none raised
      anything outside `DeclaredSignature`, `TypeError` and `ValueError`. It read
      library objects in four configurations of `wrapt` (1.17.2 and 2.5.2, each in its
      C and its Python build) and found one verdict change beyond those the changelog
      stated, the first item below.
    - What it found, each now stated in the changelog and the docstrings, and pinned
      where a test can hold it (`test_w_`): the list of what is refused although it
      runs correctly left out an object that shows another class, holds a `__wrapped__`
      of its own and names its parameters in its own `__call__` (a subclass of
      `wrapt.CallableObjectProxy`; refused in three of the four configurations, read
      in the Python build of 2.5.2); the name rule for a bound place reads two shapes
      that run otherwise, a holder that hands its receiver on to a wrapper whose
      leading place is named like a parameter of the tool, and a held wrapper with a
      required parameter of its own in the leading place; the bound of 16,384
      callables is on one search and not on a walk, which has no ceiling (it measured
      10, 40 and 160 callable attributes over one shared graph of 16,000 objects at
      0.75 s, 3.1 s and 12.2 s); a proxy whose object is gone is a `WeakProxy` as the
      tool and under a wrapper, and a plain `ValueError` as the function of a partial
      or of a method; three `__class__` properties written to deceive are still read,
      as in 0.16.0, because the walk's other type tests believe `__class__`; and any
      `Exception` raised while the walk reads, a fault of the kernel's own included,
      is reported as a `ValueError` that names the callable's hooks.
    - What it did not cover: Python 3.9 to 3.11, the free-threaded build, threads,
      the paths where the checks are off, the adapters' tests and messages, any corpus
      of real method decorators, depths four and five other than by sampling, more
      than one base signature, `skip` parameters, Django's lazy objects, and wrappers
      a C extension builds other than those of `wrapt`.
  - The generated stacks of the `recusal-openai-agents` review, and the three-layer
    stacks of the review of the change, were run again beside the corrected change;
    the scripts are not in the repository, so no count is stated here.
  - Still open: a wrapper held where no instance dictionary shows it (a slot, a wrapper
    a C extension builds), by an object that is not callable, in a container, in a
    class attribute or as an argument a `functools.partial` binds; a second wrapper
    that names nothing as `__wrapped__`; a wrapper built on first use; and the
    `__call__` of a held callable object that names nothing.
- **After 0.17.0: what two adapter reviews found in its walk and its messages** (in
  progress on the pull request [#336](https://github.com/philpaz/recusal/pull/336), not
  merged and not released). The review of
  `recusal-openai-agents` 0.2.3 and that of `recusal-pydantic-ai` 0.2.1, each beside
  `recusal` 0.17.0, reproduced each of these on 0.17.0; the pull request reduces each to
  a test without any framework:
  - A doubt about a bound argument was taken for settled by a reason that holds for a
    call by keyword alone. A facade that exposes a module function as a method and
    drops its object, under a wrapper that names the object `self` where the function
    names that place `database`, was read because `database` has no default, so a call
    by keyword that kept the object fails. A framework that hands every declared
    parameter in by position ran `execute(database='SELECT 1', sql='DELETE FROM
    audit_log', note='')` for judged `sql='SELECT 1'`, `note='DELETE FROM
    audit_log'`. The change refuses it unless the tool declares no parameter an
    argument passed by position can fill, and refuses the same shape in a wrapper a
    decorator holds. Refused although they run correctly: wrappers that take the bound
    argument under another name and do hand it on, listed in the changelog.
  - The same facade where it takes its object in `*args`, or under the name the
    function has for that place, runs shifted the same way and is still read: no name
    is in doubt, and the shape is that of every method decorator that hands its
    receiver on. Stated in the changelog, not closed.
  - A cache named again over a wrapper that names the receiver `self` and drops it
    (found by the `recusal-pydantic-ai` review) is that stated shape in the kernel,
    which reads it alike with the cache and without; the adapter's own stricter rule
    did not see the cache, and is being changed in the adapter.
  - The advice in the kernel's own refusal messages reopened the gap: "register a plain
    function that calls it", followed literally, gives a tool a gate reads by its
    parameters alone, and both reviews ran the refused hazard through one. The change
    rewrites every piece of advice to a remedy that removes the cause, each built and
    checked by a test.
  - Pinned by the tests named `test_x_` in
    [`tests/test_invocation.py`](../tests/test_invocation.py) on the pull request's
    branch.

**Where you can help.**

- Read `own_parameters`, `_next_callables` and `reached_parameters` in
  [`recusal/invocation.py`](../recusal/invocation.py) and tell us a callable they read
  wrongly.
- Hand `recusal.invocation.unsupported_signature` a tool wrapped the way your code wraps
  tools, with the decorators of the libraries you use, and report an answer that is
  wrong.
- Read the "Not found, by design" lists in the changelog entries for 0.16.0, and "Still
  not found, by design" in the kernel entry for 0.17.0, and say which of those
  shapes exist in real code you know.

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
  made. Since commit `4650c71` the gate also asks a kernel later than 0.15.1 for what
  that copy does not find; beside 0.15.1 or older, the four shapes of kernel review
  round six, a decorator that holds a second wrapper and a single-dispatch function are
  not found, and the package page says so.
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
  Fixed in the kernel's walk in commit `4650c71`, which is released in `recusal`
  0.16.0; 0.15.0 and 0.15.1, which this package also allows, have the gap.
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

**Where it stands.** `recusal-openai-agents` 0.2.1 is the release with the fixes of
rounds five and six, merged in [#308](https://github.com/philpaz/recusal/pull/308); 0.2.0
(tag `recusal-openai-agents-v0.2.0`, 2026-11-22) has the fixes of the rounds before them,
merged in [#297](https://github.com/philpaz/recusal/pull/297). The review continues: a
seventh round found the open items below, which the package page states. 0.2.2 is not a
review round: it requires `recusal` 0.16.0, whose walk closes the first two open items
of round seven, and gives each refusal of that walk a reason of its own (the changelog
entry for 0.2.2). 0.2.3 is not a review round either: it requires `recusal` 0.17.0,
whose walk and whose `cache_target` close the first three of the four items the review
of 0.2.2 left open, and gives the three refusals that walk adds reasons of their own
(the changelog entry for 0.2.3, under Unreleased).
[Package](../adapters/openai-agents/README.md).

**The rounds.** `review-history/openai-agents` holds a first commit and the commits of
rounds two to four. A fifth and a sixth round found what the next two entries close, and
a seventh found the open items.

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

- **Round five** found four things, all in 0.2.0, and the changelog entry for 0.2.1
  closes them. They are pinned by `TestFifthReview` in
  [`tests/test_openai_agents_gate.py`](../tests/test_openai_agents_gate.py).
  - The most serious. A single-dispatch tool with nothing registered was accepted. The
    SDK checks every call of a model turn before any tool body runs. So an
    implementation registered during the same turn, by another tool call or by the
    tool's own first use, ran without having been judged. Every single-dispatch function
    is now refused when the tool is gated, whether or not anything is registered.
  - A single-dispatch function that is a callable object's `__call__`, under a
    `functools.wraps` wrapper, was not found. The question is now asked of every
    callable `_reached_callables` finds, which follows `__call__`.
  - A single-dispatch function was recognised by one code object read at import. If
    `functools` was reloaded, or `functools.singledispatch` was replaced before the
    import, one was not recognised, and one kind of replacement made the package's
    import raise. It is now recognised in three ways, any one of which is enough, and
    whatever stands at `functools.singledispatch`, the import goes on.
  - `_agent_tool_parameters_reason` named the wrong cause for some fields: a pattern
    field, a list of nested models, a model with a validator. Two sentences of the
    README overstated what is found.

- **Round six**, on the changes that close round five, found seven things. The same
  changelog entry closes them, and they are pinned by `TestSixthReview` in the same
  file.
  - The most serious, and in 0.2.0 too. A `functools.partialmethod` hid the callable it
    holds. Read from a class, or from an object when the held callable has no `__get__`,
    it showed the gate the held callable's parameters, and nothing read what was under
    that callable. A single-dispatch function or a wrapper with a parameter of its own
    there ran with a value no policy judged. A callable that is or carries a
    partialmethod is now refused, whatever it holds.
  - A `weakref.proxy` of a single-dispatch function was accepted: the walk read
    `__wrapped__` through the proxy and stepped past the dispatcher. A weak proxy is now
    refused.
  - An ordinary event bus, a function that closes over a `dispatch` function over a
    `registry` dict, was refused as a single-dispatch function. The closure test now
    also asks for the two other names `functools` uses.
  - The README said a change to a tool's function is refused on every call. A change a
    tool body makes is seen from the next model turn, not by the other calls of the
    same turn. The README says so now.
  - Two sentences were not true: that the gate's walk takes every step the kernel's
    walk takes, and that no code of the object runs. Both now say exactly what is read
    and how.
  - The refusal's advice, "dispatch inside its body", could be read as "call the
    dispatcher from a plain function", which is not read. It now says to branch on the
    type.
  - Five mutations of the gate passed the suite. Each is now killed by a test.

- **Round seven**, on the changes that close round six, confirmed those closed and found
  no call a model can cause to run unjudged. It found two ordinary stacks of decorators
  that run a wrapper's own parameter unjudged. Both come from how the kernel's walk in
  `recusal` 0.15.0 and 0.15.1 reads them, both are in 0.2.0 too, and neither is fixed in
  0.2.1: the fix is in the kernel, merged in
  [#299](https://github.com/philpaz/recusal/pull/299) and released in `recusal` 0.16.0.
  They are the first two items below. It also found sentences that overstated what the
  walk reads, which 0.2.1 corrects, and refusal reasons whose advice or wording is the
  third and fourth items.

**Open items.**

- A `functools.partial` that carries a `__wrapped__` (a decorator written with
  `functools.update_wrapper` over a partial) was read as a pass-through, so a wrapper
  under it with a parameter of its own ran that parameter unjudged. Closed beside
  `recusal` 0.16.0, which 0.2.2 requires; open in 0.2.0 and 0.2.1 beside `recusal`
  0.15.0 or 0.15.1.
- `functools.wraps` written above `functools.lru_cache`: the cache names one function and
  calls the wrapper it was built over, which was not read. Closed and open as the item
  above.
- Four refusal reasons advised calling the function from a plain function and
  registering that one, without saying that what its body calls is not read, and a
  fifth, for a callable the gate cannot question, advised registering one plain
  function. In 0.2.2 every reason that gives that advice says so, and a test holds
  them to it.
- Two refusals named only an exception class (`DeclaredSignature` for the `decorator`
  package, `TypeError` for a wrapper over `wrapt.ObjectProxy`), with no cause or advice.
  In 0.2.2 a `DeclaredSignature` has a reason with both. The other is open.
- A single-dispatch function that a wrapper calls from a closure, a global or an
  attribute, and not through `__wrapped__`, is not found, and neither is one a plain
  function's body calls. The README lists it under what the gate does not see.
- A change a tool body makes to another tool's function is not seen by the other calls
  of the same model turn. The README lists it there too.
- The reasons given for the kernel's refusals (`DispatchedAtCallTime`, `HiddenWrapper`,
  `PartialMethodSignature`) were tested with stand-in classes of those names. In 0.2.2
  each class the walk of `recusal` 0.16.0 raises is tested with a real tool of that
  shape and the kernel's own class (`TestKernelRefusals` in
  [`tests/test_openai_agents_gate.py`](../tests/test_openai_agents_gate.py)). An
  `UnreadCache` is raised there by shortening what the cache lists, as the kernel's own
  test does: no supported interpreter gives one.
- Found by the review of 0.2.2 before its release, in 0.2.1 too, open beside
  `recusal` 0.16.0, and closed in 0.2.3, which requires `recusal` 0.17.0. The first
  two are closed by the gate's own walk, which now follows what a cache named again
  calls (the kernel's `cache_target`) and asks its own questions there, and by the
  kernel's walk of 0.17.0 (the last entry under the kernel's rounds above), which
  refuses them as well; the last two by that walk alone. Each is held as refused by a
  test named `test_closed_` in `TestStatedLimits` in the same file:
  - Behind a `functools.lru_cache` named again for another function, the gate's own
    questions were not asked in 0.2.2: its walk followed what the cache names. The
    kernel's walk of 0.16.0 read what the cache calls, and refused a single-dispatch
    function there only once an implementation was registered, and a weak proxy not at
    all. A single-dispatch function with nothing registered was gated where the wrapper
    that reaches it carries none of the dispatcher's attributes
    (`functools.partial(dispatcher)`, a `functools.wraps(dispatcher, updated=())`
    wrapper, a class-based decorator that sets only `__wrapped__`). An implementation
    registered after a call was judged ran for it, and the judged call itself could
    register it. The declared parameters ran with the judged values; a parameter the
    implementation adds ran at its default.
  - Behind such a cache, a `weakref.proxy` hid what stood behind it beside 0.16.0: a
    wrapper with a parameter of its own, with no dispatcher anywhere, and a dispatcher
    with an implementation registered before the tool was gated, which ran on every
    turn and was never refused.
  - A decorator that holds a `functools.wraps` wrapper which takes the tool's own
    parameters in another order was gated beside 0.16.0, and the function ran with the
    values swapped. The same wrapper on the `__wrapped__` chain was refused. This was in
    how the kernel read a held wrapper.
  - A decorator that holds a `functools.wraps` wrapper with a parameter of its own was
    gated beside 0.16.0 where the way from that wrapper to the function passes a
    class-based decorator, a callable object, a `functools.partial` that carries a
    `__wrapped__`, a weak proxy, a cache named again or a second held wrapper. The
    kernel's entries for 0.16.0 stated these as not found; its walk of 0.17.0 follows
    them.
- Refused although they run correctly beside `recusal` 0.16.0, and gated by 0.2.1
  beside `recusal` 0.15.1: a decorator that holds a cache named again for the function
  over anything but a plain function, a cache named again over a
  `functools.partial` of a second cache or of a `staticmethod` object, and a
  `functools.partial` of a `staticmethod` object under a decorator written with
  `functools.update_wrapper` over a `functools.partial` (under an ordinary
  `functools.wraps` decorator 0.2.1 refused that one too). The package page
  and the changelog entry for 0.2.2 say so, with the two the kernel's own entries
  state: a facade that does pass its bound argument on, and a declared signature that
  matches the real parameters.
- Found in review round nine of `recusal-openai-agents-mcp`, and closed in 0.2.2:
  arguments that did not arrive as JSON text (a `dict`, a list, a number) left the gate
  as a `TypeError`, on the hosted approval callback that package builds on this one's
  `refusal`. Nothing was approved, but it was an exception and not a refusal. The one
  place the package decodes arguments now refuses them before the policy runs, for the
  hosted seam, a function tool and an agent tool alike, and an empty `dict` or list,
  `0` and `False` are no longer read as no arguments. The type is asked exactly and the
  value is never asked whether it is true, so an instance of a subclass of `str` or
  `bytes` is refused too. Pinned by `TestArgumentsThatAreNotText` in the same file.
- Found by the confirmation review of 0.2.2, in 0.2.1 too, and open, as part of what
  stands behind a `weakref.proxy` behind a cache named again: a second cache named
  again that calls a `functools.partialmethod` through a `functools.partial` or a
  wrapper is gated, and the callable the partialmethod holds runs with a parameter of
  its own at its default. Built directly over the method, that second cache is refused,
  only because `functools.lru_cache` copies the partialmethod's mark onto it. Without a
  proxy of a cache between, the kernel's walk reaches the partialmethod and refuses it.
  Closed in 0.2.3: the gate's own walk follows what the outer cache calls and refuses
  the proxy, and the kernel's walk in 0.17.0 refuses it too. Tests in
  `TestStatedLimits` hold each shape as refused.
- The fourth item the review of 0.2.2 left open, and open in 0.2.3, narrower: what the
  kernel's walk raises for a callable it cannot read, other than by a class of its own,
  is refused with the name of the exception class alone (a cache named again over a
  `functools.partial` of a second cache: "cannot be inspected (ValueError)"). Beside
  `recusal` 0.17.0, 0.2.2 refused the three classes that kernel adds that way too
  (`WeakProxy`, `OpenDispatch`, `ProxiedCallable`); in 0.2.3 each has a reason, held by
  `TestKernelRefusals`.
- Found while 0.2.3 was prepared, in 0.2.2 too, and open: the kernel's stated limit of
  its name rule for a bound place reaches this gate. A held wrapper with a required
  parameter of its own in the leading place, under a method decorator that keeps the
  object, is gated, and the function runs with the judged values one place off (judged
  `sql="S"`, `force=True`; ran `sql=True`, `force=False`). The kernel's entry for 0.17.0
  states it; the package page states it under what the gate does not see, from 0.2.3.
- Refused although they run correctly beside `recusal` 0.17.0, and gated by 0.2.2 beside
  `recusal` 0.16.0, each run through the gate while 0.2.3 was prepared: a weak proxy of
  a plain function and a single-dispatch function nothing is registered on, each behind
  a cache named again; a class-based decorator that hands its attributes on with
  `__getattr__` over a function that names a `__wrapped__`; a decorator that holds a
  wrapper in another order and calls it by keyword; a decorator that holds a weak proxy
  of a helper; and a decorator that holds a cache named again for another function
  over a `functools.partial`. The package page and the changelog entry for 0.2.3 list
  them with what to write instead.
- Found by the reviews of 0.2.3 before its release, and answered in it:
  - A method facade that drops its object, over a function whose first parameter has no
    default, was gated (in 0.2.2 too) and ran with every value one place off, because
    the SDK passes every declared parameter by position. The kernel's walk of 0.17.0
    reads it. The gate refuses it itself where the dropped place is not named `self` or
    `cls`, and the same shape through a `functools.partial`. Where it is so named, a
    method decorator that forgets to pass the object on is gated and runs every value
    one place off (judged `sql="S"`, `force=True`; ran `self="S"`, `sql=True`,
    `force=False`): the signature cannot tell it from the ordinary decorated method. That
    is open, found by the confirmation review, and stated on the package page; held by
    `test_open_a_method_decorator_that_forgets_the_object`. Held by `test_closed_a_method_facade_that_drops_its_object` and
    `test_closed_a_partial_that_binds_an_argument_a_wrapper_drops` in `TestStatedLimits`.
  - Followed literally, the advice of 11 of 15 reasons gave a gated tool that ran the
    cause: it was to call the refused function from a plain function, whose body the
    gate never reads. Every reason now advises removing the cause, and
    `TestReasonAdvice` (14 tests) follows each piece of advice and runs the result.
  - A callable object whose class has a member named `partialmethod`, behind a cache
    named again, was refused, which 0.2.2 gated: the gate asks that name, which Python
    never reads, and now only of what the object itself holds.
  - Stated, not changed: a bounded cache named again over a callable that cannot be
    weakly referenced is listed on every check, once more by the gate than by 0.2.2.

**Where you can help.**

- Read `_steps`, `_reached_callables`, `_held_partialmethod`, `_dispatch_function`,
  `_dispatch_method` and `_agent_tool_parameters_reason` in
  [`_gate.py`](../adapters/openai-agents/src/recusal_openai_agents/_gate.py).
- Run the adapter's tests against an `openai-agents` release newer than the one in
  [`adapters/openai-agents/tested-with.txt`](../adapters/openai-agents/tested-with.txt).
- The scripts of this review are not in the repository.

## `recusal-openai-agents-mcp`

**Where it stands.** `recusal-openai-agents-mcp` 0.2.1 is the release with the fixes of
rounds five to ten, in [#309](https://github.com/philpaz/recusal/pull/309); 0.2.0 (tag
`recusal-openai-agents-mcp-v0.2.0`, 2026-11-22) has the fixes of the rounds before them,
merged in [#297](https://github.com/philpaz/recusal/pull/297) with those of
`recusal-openai-agents`. 0.2.1 requires `recusal` 0.16.0 and `recusal-openai-agents`
0.2.2 (0.2.0 required 0.15.0 and 0.2.0). The review continues: its package page states
what is open, and what each round did not cover is below.
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
grew with what the model sent, which a ceiling on the steps of one call now bounds)
and of a seventh round (a check that failed wrote out the whole value, at every level
of a schema that nests, which no step counted; a subschema that declares a draft, and
a `$ref` back to a root that declares one, were checked by none of the gate's own
keywords; and the steps did not bound the size of a call, which a ceiling on the size
of one call's arguments now does) and of an eighth round (on a Python without the
limit on the digits of an integer, one long integer made the size check quadratic,
so an integer of more than 4,300 digits is now refused from its bits on every Python;
the gate's own readings of the arguments recursed, so their time moved with the depth
the sender nested at; a call a little over the size ceiling was read almost to its
end; the gate read a `$ref` its own way; and three exceptions left the gate
uncaught) and of a ninth, a tenth and an eleventh round (below),
are in the changelog entry for 0.2.1. 0.2.1 copies a call's arguments with the
kernel's `recusal.invocation.detached`, as 0.2.0 does, and requires `recusal` 0.16.0,
the first whose copy does not recurse.

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

**Round nine** (2026-11-27, of the 0.2.1 changes as they stood after the eighth; the
fixes are in the same changelog entry).

- **What it found.** A `$ref` to a place no keyword the scan knows holds was validated
  against and never read, so a `patternProperties` there had `jsonschema` run the
  server's regular expression on member names the model chose, and a schema that never
  ends was not refused as one. The three stricter readings admitted what the schema
  forbids under `not`, as the condition of an `if`, in a `oneOf` and under
  `maxContains` (`0.0` under `{"not": {"const": 0}}`), and `pattern`, never evaluated,
  refused every string under a `not` with a reason that was not true. A member named
  `default` was taken for the keyword, which refused every call of a Pydantic tagged
  union with the tag `"default"`. A step did not count what a wide schema costs (a
  model of 2,000 optional fields, an `allOf` of 1,000 empty schemas). A member name
  that failed `propertyNames` was reported as the object failing `enum`. A server that
  lists one tool name twice was read by its last listing where the SDK calls its
  first. And, as a consequence nobody was told of: the defaults the gate writes are
  values the call set, which a Pydantic partial-update tool acts on.
- **What closed it.** Whatever a `$ref` names is read by the scan's rules, wherever it
  stands. What stands under one of the four keywords is read by two bounding readings,
  and a call they answer differently for is refused as not decided, with the keyword
  and the cause. Maps of names (`dependentRequired`, a `discriminator`) are read as
  names. The names a schema lists are looked up, not looped over, and the subschemas
  that check nothing, the names of a `dependentRequired` list and the values of a
  default are counted. The reason for `propertyNames` says it is a member's name. A
  tool listed twice is refused. The consequence of the written defaults is stated on
  the package page and the design is not changed. Tests: `TestNinthReview` in
  `tests/test_openai_agents_mcp_gate.py`, and in
  `tests/test_compat_openai_agents_mcp.py` a comparison with `jsonschema` as it ships
  (round ten extended it and gives its counts) and a count of what is handed to
  Python's `re` module (no pattern of a listed schema, across those and 13,041 more).
- **What it did not cover**, by its own list: the timing figures on the package page;
  the measuring tool's `stack` mode; Python 3.10; real transports (stdio, HTTP);
  `_audit.py` beyond the package's own tests; `tool_filter`, `cache_tools_list`,
  `MCPServerManager`, realtime agents, a resumed `RunState` and a rejected approval;
  the internals of `recusal_openai_agents`, and hosted MCP in a real run; a live zod
  server; the kernel's `decide` and its fingerprint; the pages sentence by sentence;
  the wide-schema cost on 0.2.0; and six cases of its fuzzing that it did not read.
- **Still open after it.** The steps did not count what `jsonschema` itself reads of a
  schema for `unevaluatedProperties` and `unevaluatedItems`, nor the walk of a `$ref`
  pointer of many parts (round ten closed the first, and bounded the second by the
  characters of a reference, not by its parts: see round eleven). A hosted approval whose `arguments`
  is not text raised `TypeError` in `recusal-openai-agents`, which this package builds
  on (found by this round, belongs to that package): `recusal-openai-agents` 0.2.2
  refuses such an approval, and 0.2.1 of this package requires it. The time tables on
  the package page are of the code before round eight.

**Round ten** (2026-11-27, of round nine's changes; the fixes are in the same
changelog entry). It confirmed every finding of round nine closed, found no ordinary
Pydantic schema refused that 0.2.0 sends, and compared 1,213,330 calls with
`jsonschema` without one the gate sends and `jsonschema` rejects, outside what follows.

- **What it found.** One shape the gate still sent although the schema forbids it: an
  `if` beside `unevaluatedProperties` or `unevaluatedItems`, under a keyword that turns
  a failure into a pass. What `jsonschema` gathers as evaluated takes the branch of
  the `if` on one reading's own answer, so neither of round nine's two readings was a
  bound there. One place where the gate and `jsonschema` as it ships disagree and the
  gate is right: `uniqueItems`, which `jsonschema` checks by sorting. Two ways a small
  listed schema held the event loop: definitions that each name the next one twice
  beside `unevaluatedProperties` (two minutes for 1.8 KB of schema), and a `$ref` of a
  million characters. A listing the SDK's server hands out and does not keep, which
  left the gate judging by the schema before it (suspected from the SDK's source,
  then reproduced). And sentences that said more than is so: what reaches Python's
  `re` module while a listing is read, what the SDK does with a tool listed twice,
  and a reason that named a list where an item of it was meant.
- **What closed it.** The gate works out what was evaluated itself, as a bound for
  each reading, once for a schema and a value, with steps; a `$ref` has a longest
  length; the gate reads the listing that was handed out beside the one the server
  object keeps, and lists again or refuses where they differ. The `uniqueItems`
  difference is stated on the package page and held by the comparison, which is now
  made with `jsonschema` compared pairwise for that keyword and counts the calls
  where `jsonschema` as it ships differs. The sentences are corrected. Tests:
  `TestTenthReview` in `tests/test_openai_agents_mcp_gate.py`, and in
  `tests/test_compat_openai_agents_mcp.py` the comparison (245,248 calls, a family
  for this shape among them) and a test of which patterns are run while a listing is
  read.
- **What it did not cover**, by its own list: timings beyond their order of
  magnitude, and the time tables on the package page; the measuring tool's `stack`
  mode; Python 3.10; real transports; a live zod server; `_audit.py`; hosted MCP in a
  real run; `tool_filter`, `cache_tools_list`, `MCPServerManager`, realtime agents, a
  resumed `RunState` and a rejected approval; a proof that the two readings are
  bounds; sentences outside the diff it reviewed; and the 2019-09 `unevaluatedItems`
  path where `jsonschema` raises `TypeError` on `items: true`.
- **Still open after it.** No proof of the bounds: the evidence is the comparison.
  Reading a listing is not counted in steps (once for a listing; round eleven says
  what it grows with). The time tables on the package page are of the code before
  round eight. And what round eleven found.

**Round eleven** (2026-11-27, of round ten's changes; the fixes are in the same
changelog entry).

- **What it confirmed.** The shape of round ten closed, over about 2.2 million calls
  (about 1.6 million on grids built by hand and about 0.6 million from two random
  generators), none of them sent although `jsonschema` rejects it; the handling
  of a listing that is handed out and not kept; the longest reference; the step counts
  the package page states; and no ordinary schema refused that was sent before.
- **What it found.** A list no step counted. A call notes each doubt the two readings
  meet, and an answer that is worked out once kept every note and wrote them all into
  the list again each time it was used again: definitions that each name the next one
  twice doubled them (67 million notes and a peak of 4.2 GB for 1,920 characters of
  schema and an argument of one character, in 169 steps), and one definition of many
  doubts named many times multiplied them. Present since round nine. One sentence of
  the package page that was not so (the gate judges by the most recent listing, not
  by the one an agent was handed), one about what is not counted that needed
  checking again, and a figure in the changelog.
- **What closed it.** An answer that is worked out once keeps one note, the kinds of
  doubt and the last keyword not decided, and adds that one when it is used again; a
  reason reads nothing else of the notes. Looking for the same fault elsewhere found
  one more collection that a reference fills for one step, what the keywords of a
  subschema evaluated, and a step is now counted for each member where one is gone
  into. The sentences are corrected. Tests: `TestEleventhReview` in
  `tests/test_openai_agents_mcp_gate.py`, which count the notes of a call and assert
  that they do not grow with the depth and grow by one for each naming.
- **What it did not cover**, by its own list: 21 of the 24 object parts and 14 of the
  16 list parts of its grid; the race inside the SDK's own server classes; Python
  3.10; real transports; a live zod server; `_audit.py`; timings beyond their order of
  magnitude; the doubling case at 28 definitions and more; the changes of
  `recusal-openai-agents` 0.2.2 beyond one probe; sentences outside the diff it
  reviewed; and a proof of the bounds.
- **The confirmation pass** (2026-11-27, of the fix above). It compared 2,234,030 calls
  line by line between the code before the fix and after: none differs in what is
  sent, in what is refused or in a reason. The doubling case ends with 3 notes at
  every depth it ran (10 to 26) and a peak of 96 MB. Six other routes were measured
  for memory, and nothing grew without steps. It found one cost the page did not
  state: a `$ref` of many pointer parts inside the 1,024 characters allowed (190
  parts took the check of 30,000 integers from 0.3 s to 3.2 s for the same steps;
  500 parts are refused when the listing is read). It is stated on the page and not
  changed. It did not cover: the doubling case on a Python other than 3.12 or at 28
  definitions and more; the rest of its grid; clean timings of its memory probe;
  real transports; a live zod server; `_audit.py`; the race inside the SDK's own
  server classes; and a proof that gathering the notes loses none.
- **Still open after it.** No proof of the bounds. Reading a listing, and the walk of
  a `$ref` pointer of many parts, are not counted in steps. The time tables on the
  package page were measured again on the code of 0.2.1.

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
