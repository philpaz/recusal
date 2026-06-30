# Landscape — is anyone else doing this?

*Researched June 2026. Star counts are approximate (±10–20%); re-check live before citing.*

Short answer: **the category is hot, and no small, zero-dependency, framework-agnostic
library owns the "separation-of-powers / refusing verifier / deterministic
failure-routing" thesis.** The adjacent tools each cover a *different* layer. The one
real peer — Microsoft's toolkit — is a heavyweight enterprise platform, which is exactly
what Recusal positions against.

## The category is real (2026 validation)

- **Microsoft Agent Governance Toolkit** — open-sourced Apr 2, 2026; "runtime security for AI agents." [blog](https://opensource.microsoft.com/blog/2026/04/02/introducing-the-agent-governance-toolkit-open-source-runtime-security-for-ai-agents/) · [repo](https://github.com/microsoft/agent-governance-toolkit)
- **Galileo Agent Control** — "write behavioral policies once, enforce across deployments." [coverage](https://thenewstack.io/galileo-agent-control-open-source/)
- **OWASP Agentic Top 10** is now the reference risk taxonomy.
- Consensus thesis: *"responsible AI in 2026 is enforced at runtime through infrastructure-level guarantees rather than in policy documents."* That sentence is Recusal's wedge.
- Regulatory note: EU AI Act high-risk obligations were slated for Aug 2, 2026, but the **Digital Omnibus (May 2026) defers them to Dec 2, 2027**. Sell on reliability and audit-readiness *now*; compliance is the long game. [timeline](https://artificialintelligenceact.eu/implementation-timeline/) · [deferral](https://knowledge.dlapiper.com/dlapiperknowledge/globalemploymentlatestdevelopments/2026/The-Digital-AI-Omnibus-Proposed-deferral-of-high-risk-AI-obligations-under-the-AI-Act)

## Who's adjacent, and what they *don't* do

Recusal's four ideas: **(1)** independent refusing verifier · **(2)** deterministic
failure-classification + routing · **(3)** runaway/bounded-autonomy controls · **(4)**
constitutional separation-of-powers model.

### Agent frameworks — *orchestrate; governance is in-process and self-graded*
| Project | ~Stars | Covers Recusal's thesis? |
|---|---:|---|
| [AutoGen](https://github.com/microsoft/autogen) | ~42k | No — "critic" patterns are in-process, LLM-graded, not a deterministic refusing authority. |
| [CrewAI](https://github.com/crewAIInc/crewAI) | ~31k | No — `max_iter` (weak #3); no deterministic verifier/router. |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | ~19k | Partial #3 (`max_turns`); "guardrails" are LLM/heuristic, not a separate authority. |
| [Pydantic AI](https://github.com/pydantic/pydantic-ai) | ~15k | No — output schema *validation*, not evidence adjudication. |
| [LangGraph](https://github.com/langchain-ai/langgraph) | ~13k | Closest in spirit (retry/rollback nodes) — but you hand-roll governance; nothing packaged. |

### Guardrails / safety — *filter content on the I/O, not the work product*
- [Guardrails AI](https://github.com/guardrails-ai/guardrails) (~6.6k) — output validation (RAIL validators).
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) (~6.5k) — Colang conversation rails.
- Llama Guard / [LlamaFirewall](https://arxiv.org/html/2505.03574v1) — probabilistic safety classifiers.
- [Invariant Labs](https://github.com/invariantlabs-ai/invariant) — rule-based guardrail proxy; closest to "intercept and block," but a *security* gate, not a refusing certifier or failure-router.

### Eval / testing — *score offline, often LLM-as-judge*
- [promptfoo](https://github.com/promptfoo/promptfoo) (~21.5k), [DeepEval](https://github.com/confident-ai/deepeval) (~15.7k), [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) (~2.1k), Ragas, OpenAI Evals.
- ⚠️ [Verdict (Haize Labs)](https://github.com/haizelabs/verdict) — "scaling judge-time compute," a compound **LLM-as-judge**. It is the **probabilistic opposite** of Recusal's deterministic verifier (and owns the name "Verdict" — which is why this library is not called that).

### Observability — *records; zero authority to stop or refuse*
- [Langfuse](https://github.com/langfuse/langfuse) (~21k), Arize Phoenix (~8k), AgentOps, Helicone. Passive telemetry.

### The real peer
- **[Microsoft Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit)** (~4.6k) — a 7-package, multi-language platform (Agent OS / Mesh / Runtime / SRE / Compliance / Marketplace / Lightning; Python/TS/Rust/Go/.NET). It covers **#3** (circuit breakers, SLOs, cascade handling) and partial compliance grading, but per its own docs does **not** package (1) an adjudicator that can *refuse to certify*, (2) deterministic failure-classification→remediator routing, or (4) a constitutional separation-of-powers model. It's OWASP/policy-engine framed (Cedar/OPA Rego), enterprise-platform — **the opposite of zero-dependency.**

### Direct peers — the lane is filling
- **AEGIS** ([github](https://github.com/Justin0504/Aegis)) — an OSS agent-firewall that already ships ~80% of an obvious MVP: pre-execution policy enforcement (YAML/AJV DSL), deterministic blocking, hash-chained + Merkle audit, a kill switch, and Claude Code + MCP adapters. This is the genuine competitor. Recusal does **not** try to out-feature it; it differentiates on **independence** ("a verifier the builder cannot influence"), determinism, and a kernel small enough to read in one sitting. (See also LoopRails.)
- **Anthropic's Claude Code auto mode** — a *same-family* safety layer: an injection probe on tool output plus a Sonnet-class transcript classifier judging actions pre-execution, with an admitted false-negative rate and Anthropic's own note that it is "not a drop-in replacement for human review on high-stakes infrastructure." This is the conflict of interest Recusal exists to remove — a model from the same family grading the same family. Anthropic also states layered defense "cannot be achieved by any single company" — the ecosystem's explicit invitation for an independent verifier.

### Academic / niche on the exact thesis (no popular pip-install)
- "AgentCity: Constitutional Governance via Separation of Power" ([arXiv](https://arxiv.org/html/2604.07007)); "Dynamic Tiered AgentRunner" physically isolates Worker/Critic/ToolGateway/**Verifier** into non-colluding processes ([arXiv](https://arxiv.org/html/2605.10223)) — strong conceptual overlap, but framework papers, not libraries.
- **LawClaw** — a single self-governing agent with a `constitution.md` + judicial layer; closest in spirit to #4, but a demo agent, not a reusable governance library.

## The gap Recusal fills

| Layer | What it does | Can it *refuse* a work product? | Deterministic? | Zero-dep? |
|---|---|:---:|:---:|:---:|
| Agent frameworks | build / orchestrate | self-graded | ❌ | ❌ |
| Guardrails | filter I/O content | content only | mixed | ❌ |
| Eval libraries | score offline | ❌ (no loop authority) | usually ❌ (LLM judge) | ❌ |
| Observability | record | ❌ | n/a | ❌ |
| MS Toolkit | enterprise platform | partial | partial | ❌ (7 packages) |
| Anthropic auto mode | same-family classifier | yes — but self-graded | ❌ (a model) | n/a |
| AEGIS (agent firewall) | enforce + audit | ✅ | ✅ | mostly |
| **Recusal** | **adjudicate evidence, refuse — independently** | ✅ | ✅ | ✅ |

Frameworks build, guardrails filter, evals score, observability records — and the safety
layer that ships with the agent is, by construction, a model from the same family grading
its own family. A serious OSS peer now exists (AEGIS), so this is a **differentiation**
game, not an empty category. Recusal's bet is the property that is hardest to retrofit and
easiest to trust: an **independent, deterministic** authority that sits in the action path
and can refuse — small enough to read in one sitting, with no model in the decision.
