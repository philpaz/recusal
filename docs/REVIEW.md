# What a review consisted of

The [project status](../README.md#project-status) table says which packages have had the
in-depth review. This page says what that review was made of, method by method, and how
to reproduce every number it states. A package gets a section here when the scripts that
produce its numbers are in this repository. Until then this page says nothing about it.
The [review log](REVIEW-LOG.md) says what each round found, package by package.

Recusal is a Beta (`0.x`) open-source project in active development. It is not a
finished or certified security product. The reviews described here are the project's own
work, and a review finds what it looks for.

## What a review round is

A review round is one pass over one package, against the threat model below, by reading
the code and by running it:

1. Probes are written against the pinned version of the framework: short scripts that
   build the framework's own objects, hand them to the gate and print what ran, what was
   answered and what was recorded.
2. Whatever a probe finds is reproduced first, then fixed, then pinned by a test that
   fails without the fix. The fix is recorded in the
   [changelog](../CHANGELOG.md).
3. The tests are checked by mutation testing: each new check is switched off in a copy
   of the code, and the tests have to notice.

The next round starts again on the fixed code. The rounds continue until one finds no way
for a model to get a call past the gate.

## The threat model

- **The model is untrusted.** Everything it produces is input to the gate, and it
  chooses all of it: the tool's name, the arguments, the call id, the type of each item,
  and the fields that say whether a call is finished. A review asks what each of those
  can be made to do: run a tool the policy did not judge, run it with arguments the
  policy did not see, put the model's text into a refusal or an audit entry, or end the
  turn so that a call that ran is not answered or not recorded.
- **The application's own code is trusted but ordinary.** The tools, the policy and the
  keywords the application passes are not an attacker's. They are written the way
  people write them: a decorator around a tool, an audit log that is optional, a
  keyword spelled wrong. A review asks what the gate does with each of those, because
  an ordinary mistake that silently turns the gate off is a gap too.

Not in the model: someone who can change the application's code, its policy, or the
machine it runs on. [`SECURITY.md`](../SECURITY.md) says where that boundary is.

## The methods

- **Probes against the pinned framework version.** As above. Most are one question
  asked once, and they are not kept: what a probe found is kept, as a test.
- **A generated corpus, run through both loops and compared.** A script generates
  turns a model could send, from lists of names, argument forms and field values, and
  runs each one twice from the same input: through the plain loop and through the loop
  that writes an audit log. It then counts the turns where the two disagree on what
  ran, where a tool ran that the application does not have under that name, and where
  the entries on the log are not one per call. The method does not need to know the
  right answer for a case: it looks for a difference between two paths that are
  documented to decide the same way, and for effects that should never happen.
- **A keyword matrix.** The keywords an entry point takes are each given a short list
  of values (not passed, `None`, an ordinary value, values that should be refused), and
  every combination is called. The script counts the combinations that ran a tool and
  then raised, that recorded something for a call they refused, or that returned an
  answer without running a tool the policy allowed.
- **Mutation testing.** A mutation is one small edit that removes or weakens one check:
  a condition replaced by `False`, a bound moved by one, a value left out of a set. Each
  mutation is applied alone to a copy of the code and the package's tests are run
  against the copy. The mutation is caught if at least one test fails. A mutation that
  survives means no test depends on that check. The mutations are written by hand, one
  or more for each check a round added.
- **Decorated-tool shapes.** A tool is rarely a bare function. The review hands the gate
  tools wrapped the ways applications wrap them (a `functools.wraps` wrapper with a
  parameter of its own, one that keeps a parameter the function also takes, a decorator
  applied inside another, a cache) and compares the values the policy judged with the
  values each layer received. The shapes that mattered are tests, in the package's test
  files and in [`tests/test_invocation.py`](../tests/test_invocation.py) for the
  kernel's check.

## Reproducing a number

Every number in a package's section below is printed by a script under
[`tools/review/`](../tools/review). From a clone:

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install --require-hashes -r requirements/adapter-openai.txt
python tools/review/openai/corpus.py --summary
```

The requirements file is the package's hash-locked one: the framework at its pinned
version and the test tools. Nothing else is installed: the scripts put the checkout
itself first on the import path and stop if the code under test would come from anywhere
else. They use no API key and open no connection. A turn is built in memory by the
SDK's own types.

Each script prints one line of JSON with its counts and exits with status 1 if any
count of findings is not zero. Run without `--summary`, it also prints each finding.
`--count` prints how many cases there are without running any, and needs no SDK.
`--help` lists the rest, and the comment at the top of each script says exactly what it
does and what a finding would be.

## What none of this shows

- **A count of cases is not a proof.** A corpus is the inputs someone thought to
  generate. No finding in it means those inputs behave as checked, and nothing more.
- **A mutation list is not every possible edit.** It shows that the tests depend on the
  checks that have a mutation. A check nobody wrote a mutation for may be untested.
- **One version of the framework.** The numbers are for the framework version in the
  requirements file. A later version can add a field, an item type or a status that no
  case here has.
- **What the gate does not see stays unseen.** Each package's README lists what its
  gate refuses and what it cannot see. No amount of testing moves a call from the second
  list to the first.
- **The project's own work.** The code, the probes and the tests come from the same
  project. A reviewed package can still have gaps, and the way to find them is to read
  the code and try it against your own stack:
  [how to contribute](../CONTRIBUTING.md), and for a security gap, a
  [private report](https://github.com/philpaz/recusal/security/advisories/new).

## `recusal-openai`

The gate for tool-calling loops written on the OpenAI API
([package](../adapters/openai/README.md), [source](../adapters/openai/src/recusal_openai)).
Its in-depth review was four rounds, against `openai==3.49.0`. The fixes are released in
`recusal-openai` 0.2.2 and listed under that name in the [changelog](../CHANGELOG.md).
Each round's fixes are one commit in the history of `adapters/openai/src`, and the tests
added from round two on say which round in a comment or a docstring.

What the model controls here: a Responses `function_call` item's `name`, `arguments`,
`call_id`, `namespace` and `status`, the response's own `status`, the types of the other
items in the output, and for Chat Completions each tool call's `id`, name and arguments,
the legacy `function_call` field and the choice's `finish_reason`. What the application
controls: the tool table, how each tool is declared and decorated, the policy, and the
recording keywords (`audit`, `surface`, `actor`, `control`, `allow_on_pass`,
`tool_use_id`).

### The numbers

| Method | Count | Result | Command |
|---|---|---|---|
| Corpus, both loops | 44,148 cases: 258 single forms and a grid of 43,890 | no finding | `python tools/review/openai/corpus.py --summary` |
| Keyword matrix | 35,280 combinations: 7,560 for each of the two loops and 20,160 for `judge_and_run` | no finding | `python tools/review/openai/keyword_matrix.py --summary` |
| Mutation testing | 148 mutations: 141 in `_gate.py`, 6 in `_audit.py`, 1 in the package's `pyproject.toml` | all caught | `python tools/review/openai/mutate.py --summary` |

The counts are of the cases the scripts define, and
[`tests/test_review_tooling_openai.py`](../tests/test_review_tooling_openai.py) holds
this table and the package's README to them: it imports each script, counts, and fails
if a number written here is not the number computed. The results are for the code in
this repository as it stands. That test reruns them where the SDK is installed: the
whole keyword matrix, the 258 single forms, one case in 89 of the grid, and one
mutation in each of the three files. The full corpus and the full mutation run are the
commands in the table. They take minutes, and the test does not run them.

**The corpus** ([`corpus.py`](../tools/review/openai/corpus.py)). The single forms are
turns of one call, or a few of two: one list of ways to write the arguments (numbers
that are not JSON, nesting at and past the limit, duplicate keys, text that is almost
JSON), the tools of each kind the gate tells apart, names the application does not
have, and each `status`, `namespace` and `finish_reason` value in turn. The grid is
turns of two calls on responses as the SDK builds them from the wire, with no
validation. The first call is an ordinary allowed one. The second is every name in a
list (the application's tools of each kind, look-alike names, names that are not
strings) with every form of the arguments in a list (JSON text, and values that are not
text at all), under every combination of item status, namespace and response status for
Responses and every `finish_reason` for Chat Completions. No two cases in a set have the
same input. A finding is any of: the two loops disagree on what ran or on whether the
turn was answered; more tools ran than the turn has calls to tools the application has;
the audit entries are not one per call under the calls' own ids; the answers are not one
per call under the calls' own ids; the two loops' answers differ beyond the two wordings
they are documented to differ in; a turn the gate refused by raising had already run a
tool; an answer or an entry is long enough to be carrying the model's text back; the
text a tool or a policy raised with reaches an answer or an entry.

**The keyword matrix** ([`keyword_matrix.py`](../tools/review/openai/keyword_matrix.py)).
Every combination of values for `audit`, `surface`, `actor`, `control`, `allow_on_pass`
and `tool_use_id`, through `gate_function_calls`, `gate_tool_calls` and `judge_and_run`,
with a policy that allows everything, so only a keyword can refuse a call. A finding is
any of: the call raised after a tool had run; it raised and an entry was written; it
returned an answer and a tool did not run; with a log, the entries are not one per
call; the two loops treat the same keywords differently. The matrix does not say which
combinations should raise. The README states that, and the package's tests pin it.

**Mutation testing** ([`mutate.py`](../tools/review/openai/mutate.py)). The 148
mutations are listed in the script, each as the exact text it replaces.
`python tools/review/openai/mutate.py --list` prints them, and `--count` checks that
each one still applies to the source. The run copies the checkout to a temporary
directory, edits only the copy, and runs
[`tests/test_openai_responses_gate.py`](../tests/test_openai_responses_gate.py) and
[`tests/test_openai_responses_audit.py`](../tests/test_openai_responses_audit.py)
against it once per mutation. It refuses to report if those tests do not pass on the
unedited copy.

### The rounds

Each finding of rounds one to three below is fixed in 0.2.2, is in that release's
changelog entry, and is pinned by the test named, in
[`tests/test_openai_responses_gate.py`](../tests/test_openai_responses_gate.py) unless
another file is given. Each test named first appears in the commit of its round. The
changelog entry has the rest of what the rounds found.

- **Round one.** The two loops did not decide the same way: a policy written as
  `amount > 100` let `{"amount": NaN}` run in the plain loop while the audited loop
  refused it
  (`test_a_number_that_is_not_json_or_not_finite_is_refused_before_the_policy`). A
  function call that carried a namespace ran the application's function of the same
  bare name (`test_a_namespaced_function_call_is_refused_before_the_policy`). A call
  that was not finished ran, on the tool's defaults if it was cut off before its
  arguments (`test_an_unfinished_function_call_item_is_refused_and_the_finished_one_runs`,
  `test_every_tool_call_of_an_unfinished_chat_message_is_refused`).
- **Round two.** A recording keyword without an audit log refused every call as a
  policy error, though no policy had run (`test_no_audit_log_is_the_plain_gate`,
  `test_a_recording_keyword_without_an_audit_log_raises_before_anything_is_judged`). A
  tool name the model chose was written into a refusal and onto the audit entry as
  sent, with no bound
  (`test_a_name_the_application_has_no_tool_for_is_described_never_quoted`). A tool
  cached with `functools.lru_cache` and handed a list raised out of the turn
  (`test_a_cached_tool_handed_a_list_or_an_object_is_refused_before_the_policy`).
- **Round three.** The package still allowed a kernel that does not have the check of a
  decorated tool's wrappers
  (`test_the_package_requires_the_kernel_that_walks_every_callable`,
  `test_the_package_does_not_import_beside_a_kernel_without_the_walk`). With an audit
  log, a `surface=` or `control=` the kernel does not record refused every call and
  wrote no entry
  (`test_a_recording_value_decide_would_refuse_raises_naming_the_keyword`, and
  `test_a_control_that_is_not_a_dict_raises_and_writes_no_entry` in
  [`tests/test_openai_responses_audit.py`](../tests/test_openai_responses_audit.py)). A
  call id the model chose had no bound
  (`test_a_call_id_longer_than_256_characters_refuses_the_turn_before_anything_runs`).
- **Round four.** The tests that carry this round's label pin `judge_and_run` called
  directly (`test_judge_and_run_called_directly_answers_a_tool_error_only_when_asked`)
  and the error raised when the stack runs out while a recording keyword is checked
  (`test_a_stack_that_runs_out_while_a_recording_value_is_checked_is_not_the_values_fault`).
  Neither is in the changelog entry, and neither changes which calls run.

### What it does not show

Everything under [What none of this shows](#what-none-of-this-shows) applies. For this
package in particular:
[What the gate refuses, and what it does not see](../adapters/openai/README.md#what-the-gate-refuses-and-what-it-does-not-see)
lists what stays outside the gate whatever the tests say: what a wrapper does in its own
body, a decorator applied inside another with a kernel that does not detect it, hosted
tools, remote MCP approvals, and what a tool does after the gate allowed it. The corpus
holds no streamed response, because the gate takes complete responses only.
