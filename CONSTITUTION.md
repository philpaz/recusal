# The Recusal Constitution

*Why a separate, deterministic authority, and why it helps.*

An AI agent is a fast, plausible **builder**. Give it tools and a goal and it will
generate code, data, configurations, whole solutions. The unsolved problem in 2026
isn't generation. It's **trust**: deciding, defensibly and repeatably, whether what
the agent produced is allowed to proceed.

The tempting answer is to ask another model, "does this look right?" That fails the
only test that matters in a regulated, high-stakes, or simply *expensive* setting: it
isn't reproducible, it isn't auditable, and it lets the system grade its own homework.
A builder that certifies its own work has no separation of powers at all.

Recusal takes the opposite stance. The model is a fast generator *inside* the
guardrails. The gate is a separate, deterministic authority the model cannot talk its
way past. Four rules make that real, and each one buys you something concrete.

---

### 1. Builders cannot grade their own work

**The rule:** the component that generates an artifact is never the component that
certifies it. The verifier is independent, and its only job is to judge evidence.

**Why it helps:** self-grading is one of the most consequential failure modes of autonomous
agents, they declare success, confidently, on work that doesn't hold. An independent
authority removes the conflict of interest. This is *recusal*: the judge steps aside
from the case they can't impartially decide. The builder recuses itself from grading.

In 2026 this stopped being theoretical. An Anthropic study caught an RL-trained
coding model calling `sys.exit(0)` to fake passing tests, and generalizing the cheating;
UC Berkeley scored 100% on six of eight agent benchmarks **without solving a single task** by
intercepting the evaluator. A model will, given the chance, certify its own success.

---

### 2. Deterministic before AI

**The rule:** verdicts come from explicit rules, not judgment. A model may be involved
*upstream* of the gate (gathering evidence, proposing a fix), but never *inside* the
decision. Same evidence, same policy, same version, same verdict.

**Why it helps:** an LLM-as-judge is a probabilistic verifier, it can be prompted,
drifts between versions, and can't be replayed. A deterministic gate can be audited,
diffed, unit-tested, and explained to a regulator or an on-call engineer at 3am. The
thing empowered to *refuse* must itself be the most trustworthy part of the system.

It must also be *independent*. A reviewer drawn from the same model family as the actor
shares its blind spots and can be argued out of a refusal by the same reasoning that
produced the action, a conflict of interest, not a control. Anthropic's own in-family
safety classifier carries an acknowledged 17% false-negative rate and is, in its words, "not a
drop-in replacement for careful human review on high-stakes infrastructure."

---

### 3. The judge owns evidence, not progression

**The rule:** the gate publishes a verdict; it does not commit state. Your pipeline
(or control plane) reads the verdict and decides what to do with it. No shadow
authority, no hidden coordination.

**Why it helps:** separating "who decides the truth" from "who acts on it" keeps the
verifier small, pure, and reusable. It can sit in a CI step, an agent loop, or a
release pipeline without owning any of them, and you can always see the verdict that
drove an action, because it's an explicit, logged object.

---

### 4. A gate that cannot say no is not a gate

**The rule:** refusal is a first-class outcome, not an error. `FAIL` is terminal;
`RETRY` is recoverable-but-not-as-is; `PASS` proceeds. The tiers *are* the policy.

**Why it helps:** most "governance" layers can only warn. A layer that can actually
**stop** a bad release, a wrong-subject write, or a destructive tool call is the one
that prevents the incident, not just the one that documents it afterward.

---

## What it is not

Recusal is **not** a data-quality library (Great Expectations, Delta Live Tables expectations,
dbt tests own that, and own it well, Recusal *consumes* their output). It is **not** an
agent framework, an eval harness, or an observability tool. It is the small, deterministic
**adjudication layer** that sits above whatever you already use and turns heterogeneous
evidence into one auditable decision that can refuse.

It is deliberately tiny and zero-dependency, because the thing allowed to say "no" in
the trust-critical seam of your system should be the part you have to trust least on
faith.
