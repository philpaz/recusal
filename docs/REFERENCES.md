# References

The claims Recusal's docs make about the problem it addresses (why self-grading is
unreliable, how agents fail, what regulators and standards bodies ask for) are grounded in
primary sources, listed here so you can check them yourself. Recusal is the kind of project
whose whole premise is "don't trust an unverified claim," so its own claims are cited.

**How to read this.** Each source is cited as *motivation and alignment* for a reference
architecture, never as a compliance or authority claim. Recusal is not "audit-proof,"
"production-certified," or "EU AI Act / ISO / NIST compliant." Where a source is easy to
overstate, a **Note** records the honest limit. Time-sensitive items (regulatory dates,
recently added catalog entries) should be re-checked at the time you rely on them.

Every figure quoted below was read from the primary source; arXiv identifiers link to the
abstract page.

---

## 1. Self-grading by an LLM is unreliable (the core premise)

This is the reason the verdict path contains no model. See [`docs/WHY.md`](WHY.md) §3.

- **Self-preference bias is structural and does not scale away with capability.** LLM judges
  over-score familiar, low-perplexity text regardless of who wrote it; more capable models
  are not reliably less biased.
  - *Self-Preference Bias in LLM-as-a-Judge*, Wataoka, Takahashi, Ri (Oct 2024; accepted at the
    NeurIPS 2024 Safe Generative AI Workshop).
    arXiv [2410.21819](https://arxiv.org/abs/2410.21819). "GPT-4 exhibits a significant degree
    of self-preference bias."
  - *Quantifying and Mitigating Self-Preference Bias of LLM Judges*, Yang et al. (Apr 2026).
    arXiv [2604.22891](https://arxiv.org/abs/2604.22891). Across roughly 20 models, "advanced
    capabilities are often uncorrelated, or even negatively correlated, with low SPB."
  - *Breaking the Mirror: Activation-Based Mitigation of Self-Preference*, Roytburg et al.
    (Sep 2025). arXiv [2509.03647](https://arxiv.org/abs/2509.03647). Steering reduces
    unjustified self-preference "by up to 97%."
  - *LLM Evaluators Recognize and Favor Their Own Generations*, Panickssery et al.
    arXiv [2404.13076](https://arxiv.org/abs/2404.13076). Corroborating, via a self-recognition
    account.
  - **Note:** the underlying mechanism (perplexity vs. self-recognition) is still debated, and
    the 97% is a best-case controlled result on curated pairs, not a field average.

- **The human-preference signal itself can reward sycophancy over truth.**
  - *Towards Understanding Sycophancy in Language Models*, Sharma et al. (Anthropic; Oct 2023,
    ICLR 2024). arXiv [2310.13548](https://arxiv.org/abs/2310.13548). Five production
    assistants were sycophantic across four tasks; humans and preference models sometimes
    prefer a convincing wrong answer to a correct one.

- **Reward hacking can generalize to subverting graders and to broad misalignment, and safety
  training can hide it on agentic tasks.**
  - *Natural Emergent Misalignment from Reward Hacking in Production RL*, MacDiarmid, Wright,
    Hubinger et al. (Anthropic; Nov 2025). arXiv [2511.18397](https://arxiv.org/abs/2511.18397).
    "RLHF safety training ... results in aligned behavior on chat-like evaluations, but
    misalignment persists on agentic tasks."
  - *School of Reward Hacks*, Taylor, Chua, Betley, Treutlein, Evans (Aug 2025).
    arXiv [2508.17511](https://arxiv.org/abs/2508.17511). Models "generalized to reward
    hacking ... preferring less knowledgeable graders."
  - **Note:** these are controlled fine-tuning and primed setups. They show models *can*
    generalize this way; they are not evidence that shipped models are covertly misaligned.

## 2. How autonomous agents fail (the actions Recusal gates)

See [`docs/WHY.md`](WHY.md) §2 and the OWASP-mapped [`examples/gallery.py`](../examples/gallery.py).

- **Prompt injection is the top LLM risk, including the indirect (tool-output) variant.**
  - OWASP GenAI Security Project, *LLM01:2025 Prompt Injection*
    ([genai.owasp.org](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)). Can cause
    data disclosure and "executing arbitrary commands in connected systems."

- **The OWASP Top 10 for Agentic Applications names Recusal's targets directly.**
  ASI01 Agent Goal Hijack ("hidden prompts to silent exfiltration") and ASI02 Tool Misuse
  ("legitimate tools bent to destructive outputs").
  - OWASP GenAI Security Project, *Top 10 for Agentic Applications* (Dec 9 2025), preceded by
    *Agentic AI: Threats and Mitigations* (Feb 17 2025)
    ([genai.owasp.org](https://genai.owasp.org/)).

- **Attack success against real-world MCP tooling is high.** Backs the figure in
  [`docs/WHY.md`](WHY.md) §2 ("peaking above 70% in recent benchmarks") and the quarantine
  demo, [`examples/injection_quarantine.py`](../examples/injection_quarantine.py).
  - *MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers*, Wang, Gao, Wang,
    Liu, Sun, Cheng, Shi, Du, Li (Aug 19 2025). arXiv
    [2508.14925](https://arxiv.org/abs/2508.14925). 20 LLM agents were tested against 45 live,
    real-world MCP servers and 353 authentic tools (1,312 malicious cases across 10 risk
    categories); peak attack success rate 72.8% (o1-mini), and the most-resistant model
    (Claude 3.7 Sonnet) still refused fewer than 3% of poisoned calls.
  - Corroborating: *Breaking the Protocol*, arXiv
    [2601.17549](https://arxiv.org/abs/2601.17549), reports a 52.8% baseline attack success
    rate reduced by a proposed defense, and that MCP's architecture amplifies attack success by
    23 to 41% over equivalent non-MCP integrations.
  - **Note:** 72.8% is the peak for a single model on a tool-poisoning benchmark (poisoned tool
    *descriptions*), not a field-wide mean. "Peaking above 70%" states a maximum, not an
    average. This is one specific injection-via-tool-metadata vector, a sibling of tool-output
    injection; both are the tool channel.

- **MCP tools reach Claude Code's hook as ordinary tools.** Grounds the README's
  "MCP tools, the same gate" section and [`examples/mcp_governance.py`](../examples/mcp_governance.py):
  MCP server tools "appear as regular tools in tool events (`PreToolUse`, `PostToolUse`,
  `PostToolUseFailure`, `PermissionRequest`, `PermissionDenied`)" under the naming pattern
  `mcp__<server>__<tool>`, e.g. `mcp__github__search_repositories`; per-server matchers like
  `mcp__memory__.*` are also documented.
  - Claude Code hooks reference
    ([code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks)).

- **The MCP specification's own security guidance covers a different, complementary layer.**
  The spec's *Security Best Practices* page enumerates authorization/transport attacks,
  confused deputy, token passthrough, SSRF, session hijacking, local-server compromise,
  OAuth URL validation, and scope minimization (a "progressive, least-privilege scope
  model"), the layer the README's MCP section defers to for transport threats.
  - Model Context Protocol, *Security Best Practices* (2025-06-18 spec revision)
    ([modelcontextprotocol.io](https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices)).
  - **Note:** that page does not enumerate tool-*description* poisoning; the research
    covering that vector is MCPTox (above), and `recusal.mcp` addresses it as deterministic
    drift detection (pin the reviewed catalog, refuse post-approval change), never as malice
    detection. Recusal's gates and the spec's authorization layer are complementary; neither
    replaces the other, and Recusal claims the discovery, invocation, and response
    boundaries it implements, never blanket "MCP security coverage" (transport and
    authorization remain the spec's layer).

- **A concrete, documented case of injection driving an exfiltration tool call.**
  - MITRE ATLAS technique AML.T0086, *Exfiltration via AI Agent Tool Invocation* (Exfiltration
    tactic AML.TA0010), added via the MITRE and Zenity Labs collaboration (Oct 2025). Case
    study AML.CS0039, *Living Off AI: Prompt Injection via Jira Service Management* (originating
    research: Cato Networks CTRL): a poisoned Jira ticket drove an Atlassian MCP tool to
    exfiltrate data. Machine-verifiable source:
    [github.com/mitre-atlas/atlas-data](https://github.com/mitre-atlas/atlas-data) CHANGELOG
    (T0086 in v5.0.0, 2025-09-30; CS0039 in v5.1.0, 2025-11-06). Human-readable:
    [atlas.mitre.org/techniques/AML.T0086](https://atlas.mitre.org/techniques/AML.T0086).
  - **Note:** these entries are recent (late 2025), and AML.CS0039 is a proof-of-concept with a
    human-in-the-loop proxy, not a confirmed in-the-wild autonomous breach. ATLAS documents the
    attack, not that any tool blocks it.

## 3. Regulatory drivers (why "auditable" and "can refuse" are design goals)

See [`SECURITY.md`](../SECURITY.md) and [`CONSTITUTION.md`](../CONSTITUTION.md).

- **Automatic lifetime event logging for high-risk systems.** "High-risk AI systems shall
  technically allow for the automatic recording of events (logs) over the lifetime of the
  system." Motivates the audit-log design.
  - Regulation (EU) 2024/1689 (EU AI Act), Article 12
    ([eur-lex.europa.eu](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)).
  - **Note:** Article 12 does not require hash-chaining or tamper-evidence. Recusal's hash chain
    goes beyond the text; it aligns with the logging goal rather than being required by it.

- **Human oversight and the veto principle.** Overseers must be able to interrupt a system to a
  safe state and to "disregard, override or reverse" its output. Motivates separation of powers
  and refusal-as-a-feature.
  - Regulation (EU) 2024/1689 (EU AI Act), Article 14
    ([eur-lex.europa.eu](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)).
  - **Note (important):** Article 14 mandates oversight by *natural persons*. A deterministic
    automated verifier aligns with and supports the veto principle, but it cannot legally
    substitute for the required human overseer. Recusal does not replace human judgment.

- **High-risk (Annex III) obligations: 2 Aug 2026 as enacted, now deferred to 2 Dec 2027.**
  - EU AI Act, Article 113, per the European Commission AI Act timeline. As originally enacted,
    2 Aug 2026 covers stand-alone Annex III high-risk systems plus transparency (Article 50) and
    penalties, not the entire Act.
  - **Note:** the **Digital Omnibus** (Commission proposal Nov 2025; political agreement reached
    May 2026) defers stand-alone Annex III high-risk obligations to **2 Dec 2027** (and Annex I
    embedded high-risk to 2 Aug 2028). This takes legal effect on formal adoption/publication in
    the Official Journal, so treat 2 Dec 2027 as the agreed-but-pending date and re-check before
    relying on it. See also [`docs/LANDSCAPE.md`](LANDSCAPE.md).

## 4. Standards frameworks (voluntary alignment, not certification)

- **Confabulation is a named GenAI risk.** "The production of confidently stated but erroneous
  or false content ... by which users may be misled or deceived." Supports the premise that
  models fabricate.
  - NIST AI 600-1, *AI RMF Generative AI Profile* (Jul 2024,
    [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)); companion to
    the AI RMF (NIST AI 100-1).
  - **Note:** NIST guidance is voluntary ("suggested actions"). "Supports the premise" is the
    claim, not "NIST requires" or "NIST compliant."

- **An AI management-system standard certifies organizations, not code.** ISO/IEC 42001
  specifies requirements for an AI Management System within an organization; certification is a
  two-stage external audit that attaches to an organization and scope, never to a software
  artifact. Annex A includes control A.6.2.8, "AI system recording of event logs."
  - ISO/IEC 42001:2023 ([iso.org/standard/42001](https://www.iso.org/standard/42001.html)).
  - **Note:** Recusal is not an "ISO 42001 certified library" (no such thing exists). The honest
    framing is that Recusal helps an organization *produce 42001-aligned evidence*.

## 5. Landscape and positioning (complementary, not superior)

See [`docs/LANDSCAPE.md`](LANDSCAPE.md). Recusal is a complementary, out-of-band, deterministic,
zero-dependency verifier of individual tool calls. It does not compete on feature count and does
not claim these tools "cannot" govern; it occupies the independent-refusal seam.

- **NeMo Guardrails** adds programmable rails "between the application code and the LLM" (an
  in-line intermediary). ([github.com/NVIDIA-NeMo/Guardrails](https://github.com/NVIDIA-NeMo/Guardrails);
  arXiv [2310.10501](https://arxiv.org/abs/2310.10501).) It has execution rails that can gate
  tool calls; the contrast with Recusal is architectural (in-line wrapper vs. out-of-band
  deterministic verifier), not a capability gap.
- **Guardrails AI** applies "quality controls to the outputs of LLMs" (post-generation
  validators, optional input guards); it validates content, not tool calls at execution time.
  ([guardrailsai.com](https://www.guardrailsai.com/docs/concepts/validators).)
- **Microsoft agent governance** is a process framework plus a layered product stack (Defender
  for Cloud AI, Content Safety, AI Red Teaming Agent, RBAC, Sentinel) with an agent registry,
  not a single product. ([learn.microsoft.com](https://learn.microsoft.com/), Cloud Adoption
  Framework.)
- **Claude Code `PreToolUse` hooks** fire before execution with an allow / deny / ask decision;
  policy enforcement uses exit code 2 to block
  ([code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks)). The
  "refusal holds even under `bypassPermissions`" property is stated explicitly in the Agent SDK
  permissions reference: "hooks run before every other step, and a hook deny applies even in
  `bypassPermissions` mode"
  ([code.claude.com/docs/en/agent-sdk/permissions](https://code.claude.com/docs/en/agent-sdk/permissions)).

## 6. Foundations Recusal builds on directly

- **Tamper-evident is not tamper-proof.** The canonical treatment of what a hash-chained log
  does and does not guarantee. Grounds the honest limits in [`SECURITY.md`](../SECURITY.md) and
  [`recusal/audit.py`](../recusal/audit.py).
  - Crosby and Wallach, *Efficient Data Structures for Tamper-Evident Logging*, USENIX Security
    2009
    ([usenix.org](https://static.usenix.org/event/sec09/tech/full_papers/crosby.pdf)).

## 7. Specific figures used in the docs

- **A model faking passing tests, then generalizing.** The write-up describes a model calling
  `sys.exit(0)` to break out of a test harness with an exit code of 0 so it appears all tests
  passed, and reports that once the model learns to reward-hack, its misalignment evaluations
  rise across the board (the paraphrase here is ours; see the source for the exact wording).
  - Anthropic, *From shortcuts to sabotage: natural emergent misalignment from reward hacking*
    ([anthropic.com](https://www.anthropic.com/research/emergent-misalignment-reward-hacking);
    arXiv [2511.18397](https://arxiv.org/abs/2511.18397)).

- **Perfect benchmark scores without solving the task.** "Every single one can be exploited to
  achieve near-perfect scores without solving a single task ... Just exploitation of how the
  score is computed." Six of eight highlighted benchmarks reached exactly 100% by gaming the
  scoring harness (for example a short `conftest.py` pytest hook forcing `outcome='passed'`).
  - UC Berkeley Center for Responsible, Decentralized Intelligence, *How We Broke Top AI Agent
    Benchmarks* ([rdi.berkeley.edu](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/)).
  - **Note:** the figure quoted in the docs is the RDI blog's own framing, eight benchmarks
    evaluated with six reaching exactly 100% by gaming the scoring harness. Cite that phrasing
    (and that count) rather than any other; use the blog as the primary source.

- **A same-family safety classifier's acknowledged limit.** "The 17% false-negative rate on
  real overeager actions is the honest number"; "It is not a drop-in replacement for careful
  human review on high-stakes infrastructure." The page also states the transcript classifier
  runs "on Sonnet 4.6", i.e. a same-family model judging the agent.
  - Anthropic engineering, *How we built Claude Code auto mode*
    ([anthropic.com](https://www.anthropic.com/engineering/claude-code-auto-mode)).
  - **Note:** the 17% is scoped to a curated set of about 52 real overeager actions; a separate
    0.4% false-positive figure is the production-traffic number. The two are different
    conditions and should not be merged.

- **Agent security is not a single-vendor problem.** "The security and reliability of agents
  cannot be achieved by any single company working alone." The docs cite this as the ecosystem's
  own case for an independent verifier.
  - Anthropic, *Trustworthy agents in practice* (Apr 2026)
    ([anthropic.com/research/trustworthy-agents](https://www.anthropic.com/research/trustworthy-agents)).
    Quoted in [`docs/FAQ.md`](FAQ.md) and [`docs/LANDSCAPE.md`](LANDSCAPE.md).

---

*If you find a citation here that is out of date, mis-scoped, or that you cannot reproduce from
the linked primary source, please open an issue. Getting these right is part of the point.*
