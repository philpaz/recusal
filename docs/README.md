# Recusal documentation

Deterministic governance for Claude agents, an independent verifier that can refuse to
certify a tool call *before* it runs. Start with whichever door fits you:

## Start here

- **[FAQ](FAQ.md)**: the questions people ask before adopting: *do I need this? doesn't
  Claude already do it? is it ready to use?* The fastest orientation.
- **[WHY](WHY.md)**: the long-form "so what," written for the people who own the decision
  to run agents that take real actions (engineering, platform/SRE, risk/compliance).
- **[../README](../README.md)**: the project overview, install, and copy-paste integrations.

## Understand the design

- **[../CONSTITUTION](../CONSTITUTION.md)**: the four rules and *why each one helps*:
  builders can't grade their own work · deterministic before AI · the judge owns evidence,
  not progression · a gate that can't say no isn't a gate.
- **[EVIDENCE](EVIDENCE.md)**: the contract: `Finding`, `Verdict`, `Severity`, `Decision`,
  and `compute_verdict`. Two objects and one function; the whole spine.

## Use it

- **[HOWTO](HOWTO.md)**: the three integration paths (Claude Code hook, Agent SDK loop,
  Managed Agents) plus direct adjudication, audit, and failure routing.
- **[EXAMPLE](EXAMPLE.md)**: one complete, copy-paste configuration for a real use case
  (a database-admin agent left in auto mode), shown end to end: install, hook, settings,
  and what the agent experiences when it tries something it shouldn't.
- **[COOKBOOK](COOKBOOK.md)**: copy-paste policies for the actions people actually gate:
  destructive shell, unscoped SQL, secret-file writes, wrong-subject writes, egress
  allowlists, injection quarantine, action budgets, quality gates. Lift one and adapt it.
- **[EXTENDING](EXTENDING.md)**: write your own checks, bundle policies, add an adapter for
  another framework, customize severities and the failure taxonomy.

## Evaluate it

- **[LANDSCAPE](LANDSCAPE.md)**: where Recusal sits among frameworks, guardrails, evals,
  observability, and the real peers (Microsoft's toolkit, AEGIS, Anthropic's auto mode).
- **[PROVEN](PROVEN.md)**: Recusal governs its own repository: a real Claude Code hook
  refusing real destructive tool calls, CI-locked and reproducible verbatim.

## Contribute

- **[../CONTRIBUTING](../CONTRIBUTING.md)** · **[../CODE_OF_CONDUCT](../CODE_OF_CONDUCT.md)**
  · **[../SECURITY](../SECURITY.md)** · **[../CHANGELOG](../CHANGELOG.md)**

---

The mental model, everywhere: **gather evidence → compute a verdict → act on it
(allow / retry / refuse).** No model in the decision path.
