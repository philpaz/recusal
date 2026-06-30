# FAQ

Short, direct answers to the questions people actually ask before adopting Recusal.
For the long-form version of the "why," read [`WHY.md`](WHY.md) and
[`../CONSTITUTION.md`](../CONSTITUTION.md).

---

## Why do I need this? My agent already works.

It works *until it doesn't*, and the failure is no longer a wrong sentence a human
reads, it's an **executed action**: a `DELETE` without a `WHERE`, an `rm -rf` outside
the intended directory, a write staged against the wrong customer, a runaway loop that
burns a budget overnight. The moment an agent *acts* through tool calls, the consequence
lands before anyone reviews it. Recusal is the deterministic check that sits in that
action path and can **refuse before the irreversible thing happens**. If your agent only
drafts text a human approves, you may not need it yet. If it takes real actions on real
systems, you do.

See [`WHY.md` §1-2](WHY.md) for the full argument.

## Doesn't Claude / Claude Code already do this?

Partly, and Anthropic is refreshingly candid about the limit. Claude Code's auto-mode
safety layer is a **same-family classifier**: a Claude-class model (Sonnet 4.6) judging another Claude
agent's actions. Anthropic itself states it is *"not a drop-in replacement for careful human
review on high-stakes infrastructure"* and carries an acknowledged 17% false-negative rate.
In *Trustworthy agents in practice* Anthropic also says the security of agents *"cannot be achieved by any single company"*,
an explicit invitation for an **independent** verifier.

That's the gap Recusal fills. The builder and an in-family reviewer share training data,
share blind spots, and can be argued out of a refusal by the same reasoning that produced
the action. Recusal puts **no model in the decision path**, so it can't be talked into it.
And a Claude Code `PreToolUse` `deny` from Recusal is honored **even under
`bypassPermissions`**, a control your users can't turn off by switching permission modes.

## What does "the autonomous agent flow needs this" actually mean?

An autonomous loop is *generate → act → observe → repeat*, often dozens of tool calls
deep with no human between steps. The thing generating each action is also, in most
stacks, the only thing judging whether it's safe. That's a builder grading its own work,
the single most common way autonomous agents fail (they declare success, confidently, on
work that doesn't hold; see the Anthropic `sys.exit(0)` study and the UC Berkeley 100%-by-
cheating result in [`WHY.md` §2](WHY.md)). Recusal inserts an **independent, deterministic
gate** into that loop:

> gather evidence → compute a verdict → allow / retry / refuse

On `RETRY` the agent gets the failure reasons as context and tries again; on `FAIL` the
action is refused terminally. The loop keeps its autonomy; it just can't certify itself.

## Isn't a deterministic gate too rigid? Why not an LLM judge?

An LLM judge is a *probabilistic* verifier: it gives different answers to the same input
on a different day or model version, can be prompt-injected, and can't be replayed. For
the thing empowered to **refuse**, that's the wrong trade. A deterministic verdict can be
unit-tested, diffed, logged, and explained to an auditor or an on-call engineer at 3am,
*same evidence in, same verdict out, forever.* You can still use a model **upstream** to
gather evidence; you just don't let it sit **inside** the decision.

## Where does the "evidence" come from? Does Recusal gather it?

No, and that separation is deliberate. Recusal **adjudicates** evidence; it doesn't
collect it. Evidence is anything that produces a `Finding`: preconditions, an allowlist, a
dry-run, a policy check, or the output of validators you already run (Great Expectations,
pytest, a linter, a schema check). You decide *what proves this action is safe*; Recusal
folds those findings into one verdict. See [`HOWTO.md`](HOWTO.md) and
[`EXTENDING.md`](EXTENDING.md).

## How is this different from guardrails / evals / observability?

- **Guardrails** (Guardrails AI, NeMo) filter I/O *content*, they don't adjudicate a work
  product or an action.
- **Eval libraries** (promptfoo, DeepEval) score **offline**, usually with an LLM judge,
  not in the live action path.
- **Observability** (Langfuse, AgentOps) **records** what happened, zero authority to stop
  anything.

Recusal is the small piece none of them are: an independent authority *in* the action path
that can say **no**. Full comparison: [`LANDSCAPE.md`](LANDSCAPE.md).

## Is it production-ready? What's the maturity?

It's early (`0.x`, Alpha) and honest about scope. What's proven end-to-end **today**: the
enforcement path on the real wire format, a real Claude Code hook, the real `PreToolUse`
JSON, a real `deny` Claude Code honors, and it governs *this* repository's own
development. See [`PROVEN.md`](PROVEN.md). What it does not yet claim: fleet-scale
production deployment. The core is zero-dependency stdlib and frozen/immutable by design,
which is exactly what you want from the part allowed to refuse.

## What are the dependencies? What Python versions?

**Zero runtime dependencies**, standard library only (`dataclasses` + `enum`). Python
**3.9+**, tested in CI on 3.9-3.13. The dev extras (`pytest`, `ruff`, `mypy`) are only for
contributing.

## What happens if my policy code crashes?

The Claude Code hook **fails closed**: a policy that raises becomes a `deny`, not a silent
allow. You can opt into fail-open (`fail_closed=False`) only if you understand the
trade-off. See [`../SECURITY.md`](../SECURITY.md).

## How do I add my own rules?

A check is just a function that returns `Finding`s; a policy is just a function that
returns a list of them. You never touch the core. Severity is the policy dial
(`CRITICAL→FAIL`, `ERROR→RETRY`, `WARNING/INFO→PASS`), chosen per call site. Full guide:
[`EXTENDING.md`](EXTENDING.md).

## The failure classifier is just keyword matching. Isn't that brittle?

It's deliberate, and the design owns the trade. `classify_failure` routes a failure by
explicit, ordered markers, *transparent, deterministic, and replaceable*, the same
properties the gate has. A model-based classifier would be the thing this project exists
to avoid in the decision path: non-reproducible and un-auditable. The classifier is built
to fail safe, security classes (policy violation, prompt injection) are matched first so
they're never misread as a generic error, and **anything unmatched falls through to
`ask-human`; it never guesses.** The taxonomy is plain data (`DEFAULT_TAXONOMY`): tighten a
marker, reorder precedence, or supply your own classes for your domain, without touching
the core. If you need fuzzy classification, run a model *upstream* to produce the failure
text; the routing of it stays deterministic.

## Does it only work with Claude?

No. The Claude adapters are conveniences; the enforcement core (`compute_verdict`,
`Finding`, the checks, the audit log) is zero-dependency and framework-neutral. See
[`../examples/agent_loop.py`](../examples/agent_loop.py), a complete gate in a plain
agent loop whose only import is `recusal`, no Claude and no SDK. The same `compute_verdict`
seam drops into LangGraph, the OpenAI Agents SDK, or a homegrown runtime unchanged.

## Why the name "Recusal"?

A judge **recuses** themselves from a case they can't impartially decide. The same
principle governs autonomous agents: the thing that *generates* the work must never be the
thing that *certifies* it. (The name "Verdict" was taken by an LLM-as-judge library, the
probabilistic opposite of what this is.)
