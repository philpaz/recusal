# Landscape, is anyone else doing this?

*Researched June 2026; re-check live before citing.*

Short answer: the category is active and filling fast. Most adjacent tools cover a
*different* layer (orchestration, content filtering, evaluation, observability), and real
peers now exist (AEGIS, Microsoft's toolkit). As far as we found, no small, zero-dependency
library takes this exact angle, but Recusal is new and unproven and these are mature,
widely-used projects. This page maps where it sits and how the approach differs; it is not
a ranking.

## The category is real (2026 validation)

- **Microsoft Agent Governance Toolkit**: open-sourced Apr 2, 2026; "runtime security for AI agents." [blog](https://opensource.microsoft.com/blog/2026/04/02/introducing-the-agent-governance-toolkit-open-source-runtime-security-for-ai-agents/) · [repo](https://github.com/microsoft/agent-governance-toolkit)
- **Galileo Agent Control**: "write behavioral policies once, enforce across deployments." [coverage](https://thenewstack.io/galileo-agent-control-open-source/)
- **OWASP Agentic Top 10** is now the reference risk taxonomy.
- The recurring theme across 2026 write-ups on agent governance is that responsible AI is increasingly enforced *at runtime* through infrastructure-level controls rather than in policy documents (our synthesis of the sources above and the emerging-peer projects below, not a single attributable quote). That runtime-enforcement angle is the one Recusal takes.
- Regulatory note: EU AI Act high-risk obligations were slated for Aug 2, 2026, but the **Digital Omnibus (agreed May 2026) defers them to Dec 2, 2027**. Sell on reliability and audit-readiness *now*; compliance is the long game. [timeline](https://artificialintelligenceact.eu/implementation-timeline/) · [deferral](https://knowledge.dlapiper.com/dlapiperknowledge/globalemploymentlatestdevelopments/2026/The-Digital-AI-Omnibus-Proposed-deferral-of-high-risk-AI-obligations-under-the-AI-Act)

## Who's adjacent, and what they *don't* do

Recusal's four ideas: **(1)** independent refusing verifier · **(2)** deterministic
failure-classification + routing · **(3)** runaway/bounded-autonomy controls · **(4)**
constitutional separation-of-powers model.

### Agent frameworks, *orchestrate; any governance is in-process and self-graded*
These build and run agents; adjudication is a different job. All are mature, widely-used tools.
| Project | How its focus differs from Recusal |
|---|---|
| [AutoGen](https://github.com/microsoft/autogen) | "critic" patterns are in-process and LLM-graded, not a separate deterministic authority. |
| [CrewAI](https://github.com/crewAIInc/crewAI) | `max_iter` bounds loops; no deterministic verifier or failure-router. |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | `max_turns` plus LLM/heuristic "guardrails," not a separate authority. |
| [Pydantic AI](https://github.com/pydantic/pydantic-ai) | output schema *validation*, not evidence adjudication. |
| [LangGraph](https://github.com/langchain-ai/langgraph) | closest in spirit (retry/rollback nodes), but governance is hand-rolled, not packaged. |

### Guardrails / safety, *filter content on the I/O, not the work product*
- [Guardrails AI](https://github.com/guardrails-ai/guardrails): output validation (RAIL validators).
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails): Colang conversation rails.
- Llama Guard (a probabilistic safety classifier) / [LlamaFirewall](https://arxiv.org/html/2505.03574v1): an open-source guardrail *framework* (PromptGuard 2 jailbreak detection, Agent Alignment Checks over chain-of-thought, and CodeShield static analysis), broader than a single classifier.
- [Invariant Labs](https://github.com/invariantlabs-ai/invariant): rule-based guardrail proxy; closest to "intercept and block," but a *security* gate, not a refusing certifier or failure-router.

### Eval / testing, *score offline, often LLM-as-judge*
- [promptfoo](https://github.com/promptfoo/promptfoo), [DeepEval](https://github.com/confident-ai/deepeval), [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai), Ragas, OpenAI Evals.
- ⚠️ [Verdict (Haize Labs)](https://github.com/haizelabs/verdict): "scaling judge-time compute," a compound **LLM-as-judge**. It is the **probabilistic opposite** of Recusal's deterministic verifier (and owns the name "Verdict", which is why this library is not called that).

### Observability, *primarily records and analyzes execution* (verify each product's current enforcement features before a categorical comparison)
- [Langfuse](https://github.com/langfuse/langfuse), Arize Phoenix, AgentOps, Helicone. Passive telemetry.

### The closest documented peers
- **[Microsoft Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit)**: a multi-package, multi-language platform (Agent OS / Mesh / Runtime / SRE / Compliance / Marketplace / Lightning / Agent Control Specification; Python/TS/Rust/Go/.NET). It covers **#3** (circuit breakers, SLOs, cascade handling) and partial compliance grading, but per its own docs does **not** package (1) an adjudicator that can *refuse to certify*, (2) deterministic failure-classification→remediator routing, or (4) a constitutional separation-of-powers model. It is OWASP/policy-engine framed (Cedar/OPA Rego) and platform-scale, the opposite of a zero-dependency kernel. It is itself deterministic, so that is shared ground, not a difference.

### Direct peers, the lane is filling
- **AEGIS** ([github](https://github.com/Justin0504/Aegis)): an OSS agent-firewall whose documentation describes pre-execution policy enforcement (YAML/AJV DSL), deterministic blocking, hash-chained + Merkle audit, a kill switch, and Claude Code + MCP adapters. Recusal does **not** compete on feature count; the documented difference in scope is Recusal's small zero-dependency evidence-to-verdict contract and independent-refusal framing. Verify AEGIS's current capabilities from its own documentation before comparing.
- **Anthropic's Claude Code auto mode**: a *same-family* safety layer: an injection probe on tool output plus a Sonnet-class transcript classifier judging actions pre-execution, with an admitted 17% false-negative rate and Anthropic's own note that it is "not a drop-in replacement for careful human review on high-stakes infrastructure." This is the conflict of interest Recusal exists to remove, a model from the same family grading the same family. In "Trustworthy agents in practice" Anthropic also states the security of agents "cannot be achieved by any single company", the ecosystem's explicit invitation for an independent verifier.

### Academic / niche on the exact thesis (no popular pip-install)
- "AgentCity: Constitutional Governance for Autonomous Agent Economies via Separation of Power" ([arXiv](https://arxiv.org/abs/2604.07007), an on-chain governance model); "Beyond Autonomy: A Dynamic Tiered AgentRunner" puts proposal, review, execution, and **verification** in independent agents with physically isolated boundaries ([arXiv](https://arxiv.org/abs/2605.10223)): conceptual overlap, but framework papers, not libraries.
- **LawClaw**: a single self-governing agent with a `constitution.md` + judicial layer; closest in spirit to #4, but a demo agent, not a reusable governance library.

## Which layer each one works at

Different tools solve different parts of the problem. This is a map of layers, not a
ranking, and Recusal is the new, unproven entry here; several of the others are mature and
widely used.

| Layer | What it does | Where Recusal sits differently |
|---|---|---|
| Agent frameworks | build and orchestrate agents | doesn't orchestrate; adjudicates a proposed action |
| Guardrails | filter input/output content | judges the work product or action, not the text |
| Eval libraries | score offline, often LLM-as-judge | decides in the live action path, deterministically |
| Observability | record what happened | can refuse before the action runs |
| MS Toolkit | enterprise governance platform | a small zero-dependency kernel, not a platform |
| Anthropic auto mode | same-family safety classifier | no model in the decision path |
| AEGIS | agent firewall: enforce + audit (per its documentation) | a documented peer; Recusal's difference is the small deterministic evidence-to-verdict contract, not feature parity |

Frameworks build, guardrails filter, evals score, observability records, and the safety
layer that ships with an agent is, by construction, a model from the same family judging
that family. Real OSS peers now exist (AEGIS especially), so this is a differentiation
question, not an empty category. Recusal's bet is narrow: an independent, deterministic
authority in the action path that can refuse, with no model in the decision and a kernel
deliberately small. Whether that is worth adopting over a more featureful
peer is a fair question to ask of a new, early-stage library.
