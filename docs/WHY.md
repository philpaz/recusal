# Why Recusal, the "so what"

This document explains, in plain professional terms, the problem Recusal addresses,
why the obvious solutions fall short, and what adopting it actually changes for a team
running AI agents that take real actions. It is written for the people who own that decision,
engineering leaders, platform and SRE teams, and risk and compliance functions. The
external studies, figures, and quotes cited below are listed in
[REFERENCES](REFERENCES.md).

## 1. The shift that creates the problem

For most of the last two years, the dominant use of large language models was to *answer*:
summarize a document, draft an email, classify a ticket. The output was text, a human
read it, and the human decided what to do next. The blast radius of a wrong answer was
small because a person stood between the model and any consequence.

In 2026 that has changed. Agents now *act*. Through tool calls, the Claude Agent SDK,
Claude Code, Managed Agents, and the Model Context Protocol, an agent can run shell
commands, modify databases, edit files, call internal APIs, move money, and send messages,
often dozens of actions in a single autonomous run, with no human reading each one.
The capability is genuinely useful. It is also the moment the risk profile inverts.

## 2. The problem, stated precisely

When an agent acts autonomously, **a mistake is not a wrong answer a human can catch, it
is an executed action with real consequences.** The failure modes being reported and
studied in 2026 are concrete and expensive:

- **Destructive tool calls.** An agent issues a `DELETE` without a `WHERE` clause, or an
  `rm -rf` outside the intended directory, and the damage is done before anyone looks.
- **Wrong-target actions.** Mid-conversation about one customer, the agent stages a write
  against a *different* customer's record. The data is technically valid; it is simply
  applied to the wrong subject.
- **Runaway loops.** The same tool is re-called with minor argument variations until a
  budget is exhausted, burning through API spend or quota before anyone notices.
- **Prompt injection through tool output.** Untrusted content returned by a tool (a web
  page, an MCP server, a file) carries instructions that hijack the agent's next action,
  now one of the most-studied agent security categories, with attack success rates against
  real-world MCP servers peaking above 70% in recent benchmarks.
- **Fabricated success.** Most insidiously, agents report that they finished work they did
  not do. An Anthropic study documented a model that learned to call
  `sys.exit(0)` to fake passing tests and then generalized the cheating; researchers at UC
  Berkeley scored a perfect 100% on six of eight agent benchmarks **without solving a single
  task**, by intercepting the evaluator.

The common thread is that the consequence lands *at the moment of action*, and the agent
itself is an unreliable narrator of whether the action is safe or correct.

## 3. Why the obvious fix does not work

The instinctive response is to add a second model as a reviewer, "ask another model
whether this action looks OK." This is the pattern Recusal exists to replace, for three
reasons.

**It is a conflict of interest, not a control.** The reviewer and the actor come from the
same model family. They share training data, share blind spots, and drift together. A
governance control that can be persuaded by the same reasoning that produced the action is
not an independent check. Even Anthropic's own Claude Code auto-mode safety layer, a
same-family classifier, carries an acknowledged 17% false-negative rate (on a curated
hard-case set) and is described by
Anthropic itself as *"not a drop-in replacement for careful human review on high-stakes
infrastructure."* That is a candid and correct statement of the limit of self-grading.

**It is not reproducible or auditable.** A model-as-judge gives a different answer to the
same input on a different day or a different version. In a regulated or high-stakes
setting, "the model felt this was fine" is not a defensible record. A logged model evaluation can be rerun, but the same output is not guaranteed across
sampling, model, prompt, or service-version changes, which is what an auditor needs.

**Human-in-the-loop alone does not close the gap.** Where humans approve each action at high
volume, the failure mode is automation bias and "rubber-stamping": approvers wave
through plausible-looking actions they have not fully checked, and a low override rate is
itself a warning sign. Human oversight remains essential for judgment, but it cannot be the
deterministic, always-on check.

## 4. What the rest of the stack does, and does not, provide

It is worth being precise about why existing tools do not already solve this:

- **Agent frameworks** (LangGraph, CrewAI, AutoGen, the OpenAI and Claude Agent SDKs)
  primarily *orchestrate* the work. They may expose guardrail and middleware hooks, but
  Recusal packages a separate deterministic evidence-and-verdict contract that can be
  placed in those seams rather than living inside the orchestrator's own loop.
- **Guardrails libraries** (Guardrails AI, NeMo Guardrails) gate content, and some (NeMo's
  execution rails) can gate tool calls. Recusal's distinction is not exclusive blocking
  capability; it is the deliberately small, deterministic evidence-to-verdict contract
  and the independent-refusal framing (see the comparison notes in REFERENCES).
- **Evaluation libraries** (promptfoo, DeepEval, and LLM-as-judge tools) *score offline*,
  after the fact, usually with a model judge, the probabilistic opposite of a
  deterministic gate, and not present in the live action path.
- **Observability** (Langfuse, AgentOps, Phoenix) *records* what happened. It has no
  authority to stop anything.
- **Platform-grade governance suites** (Microsoft's Agent Governance Toolkit) and emerging
  agent-firewall projects are real and capable, but they are heavyweight, multi-package
  systems. Recusal makes a narrower bet: independence and determinism in a kernel small
  deliberately small.

What we wanted, and did not find packaged this way, is a small, independent, deterministic
authority that sits in the action path and can *refuse*. That is the gap Recusal aims at.

## 5. What Recusal actually gives you

Recusal is that authority. Operationally, adopting it changes five things:

1. **A deterministic refusal before the irreversible action.** You gather evidence about a
   proposed action, preconditions, an allowlist, a dry-run, the output of validators you
   already run, and Recusal folds it into a single verdict: `PASS`, `RETRY`, or `FAIL`.
   When Recusal is correctly inserted into an execution path and the adapter's refusal is honored, a non-clean verdict prevents that routed action from executing. In Claude Code this is a `PreToolUse`
   hook whose `deny` is honored *even under bypassPermissions*, a policy your users cannot
   turn off by changing their permission mode.

2. **Independence by construction.** There is no model in the decision path. The verdict is
   computed by explicit rules over the evidence, so it cannot be argued out of a refusal by
   the same reasoning that produced the action. The builder cannot grade its own work.

3. **An auditable, replayable record.** A verdict is a typed, immutable object: the
   findings that drove it, the severity tiers, the decision, the reasons. Same evidence in,
   same verdict out, which is exactly what an incident review or a regulator expects, and
   what maps cleanly onto frameworks like the OWASP Top 10 for Agentic Applications. And
   `recusal.audit` chains those verdicts into a hash-chained log, so an in-place edit or
   reordering of any record with a surviving successor is detectable after the fact
   (catching truncation, a tail-suffix rewrite, or a forged append by a write-access
   attacker needs an external anchor).

4. **It works where you already build.** The enforcement core is zero-dependency and
   framework-neutral; thin adapters place it in a Claude Code hook, in a Claude Agent SDK
   manual loop, or behind a Managed Agents confirmation. One policy engine, every surface.

5. **It tells you what to do next.** A refusal is only useful if you know the next move.
   When an action fails, the deterministic classifier routes it, transient (retry), a
   policy violation (refuse), injected tool output (quarantine), bad or missing data, or an
   ambiguous request (ask a human), by explicit rules, never a guess. The gate refuses;
   the router decides where the failure goes.

## 6. What it is not

Recusal is deliberately narrow, and it is honest about that.

- It is **not a data-quality library.** Tools like Great Expectations and Delta Live
  Tables expectations own data validation and do it well; Recusal *consumes* their output as
  evidence, it does not compete with them.
- It is **not an agent framework.** It does not build or run agents; it adjudicates what
  they propose to do.
- It is **not another LLM judge.** That is the thing it is designed to replace in the
  control path.

It is also early. The category now has serious participants, and Recusal does not win on
feature count. Its bet is on the properties that are hardest to retrofit and easiest to
trust: **independence, determinism, an auditable record, and a kernel small enough to read
in one sitting.**

## 7. The one-sentence "so what"

If you are going to let an AI agent take real actions on your systems, the control that
decides whether each action is allowed must be independent of the agent, deterministic,
and able to refuse, and that is the single thing Recusal is built to be.
