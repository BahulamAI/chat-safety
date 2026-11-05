# chat-safety

`chat-safety` is an early-stage Python framework for typed risk assessments in conversational applications. The caller supplies content, context, and a risk registry; the framework returns model evidence. The caller owns thresholds, subfamily follow-up calls, agent nudges, and chat actions. The first consumer is planned to be Bahulam under workspace PRD-118; the package design is in workspace PRD-119.

## Status

The primary API now supports configurable top-N family reporting, caller-defined family/subfamily criteria, explicit subfamily assessment calls, and separate message/session scores. It has offline and mocked-HTTP tests. The earlier `SafetyGate`, fixed `Category` vocabulary, JSON action policies, and action-based session tracker remain optional compatibility APIs. There are **no approved production thresholds**, Bahulam assessment/nudge hooks, durable session-store adapter, or published release. The Jev adapter follows OpenRouter's documented [Decisions API](https://openrouter.ai/blog/insights/what-is-jev/) shape; production privacy controls remain to be verified.

## Package layout

| Module | Responsibility |
| --- | --- |
| `assessment.py` | Evidence-only input/result contracts, top-N rankings, and explicit family/subfamily calls |
| `registry.py` | Caller-defined family IDs, subfamily IDs, questions, bounds, and reporting defaults |
| `questions.py` | Provider-neutral binary question criteria |
| `adapters/jev_risk.py` | Jev adapter for the dynamic family assessment API |
| `models.py` | Stages, actions, risk tiers, input, signals, and decision types |
| `context.py` | Reject oversize content or history without silent truncation |
| `policy.py` | Versioned threshold bands supplied by the application |
| `config.py` | Strict loader for caller-owned JSON policy and category questions |
| `engine.py` | Hard rules, evaluator evidence, action composition, and safe fallback |
| `adapters/jev.py` | OpenRouter Jev Decisions API client and strict response parsing |
| `session.py` | Bounded session events, immutable state transitions, and durable-store interface |
| `evaluation.py` | Offline metrics for labeled cases |

## Family assessment and caller nudges

Use [risk-registry.json](examples/risk-registry.json) as a synthetic configuration example. `reporting.top_k_families` defaults to 3 and accepts any positive integer. Each call may override it with 1 or N. Custom family and subfamily IDs are validated identifiers defined in this file; they do not require a package release. Each definition includes separate message and session questions. Descriptions are metadata; the question instructions and criteria carry the model-facing meaning. The framework validates contracts and bounds, without interpreting the subject matter itself.

```python
from chat_safety import AssessmentInput, RiskAssessor, Stage, load_risk_registry
from chat_safety.adapters.jev_risk import JevRiskEvaluator

registry = load_risk_registry("/trusted/config/risk-registry.json")
assessor = RiskAssessor(registry, JevRiskEvaluator(api_key=key_from_secret_store))
request = AssessmentInput(
    request_id="event-8", registry_version=registry.version,
    stage=Stage.INPUT, content=content_to_assess,
)
families = await assessor.assess_families(request, top_k_families=1)
# The caller inspects status/scores and decides whether a second call is needed.
detail = await assessor.assess_subfamilies(request, family_ids=("health",))
```

`assess_families` evaluates every family in the configured registry. Its result contains all `scores` plus up to N `top_families` **per requested scope**. N greater than the registry size returns all families; rankings can include low scores and are not declarations of harm. Scores are independent criterion probabilities and do not sum to one. A mathematical intersection of leading families does not establish chat intent. The caller can inspect every score, including threshold crossings outside the displayed top N. Sorting is deterministic for the returned scores; Jev inference is not guaranteed to repeat exactly.

`assess_subfamilies` runs only when the caller invokes it. The caller selects one or more registered parents based on its own thresholds; the framework evaluates their children without changing the broad scores. There are no automatic follow-up calls, nudges, refusals, or allow decisions. The caller may keep expansion thresholds and trusted nudge templates in its own configuration. See [family_nudge.py](examples/family_nudge.py) for a runnable offline example that makes two explicit assessments and constructs a caller-owned nudge from the child results.

Results include `status`, evaluator/model snapshot, registry version and fingerprint, input fingerprint, stage, assessed family IDs, scopes, and session reference/revision. Provider errors or invalid/missing scores return `status="unavailable"` with empty scores, never zero-risk evidence or a chat disposition. Invalid request/configuration arguments raise before the provider call. `result.to_dict()` is JSON-compatible and excludes raw assessed content. The caller decides how to handle unavailable results.

For session assessment, supply both `AssessmentScope.MESSAGE` and `AssessmentScope.SESSION`, the trusted history/summary, and session reference/revision in `AssessmentInput`. User, assistant, tool proposal, and tool result summaries can contribute. The same content snapshot can be used for both family and subfamily calls; match fingerprints and revisions before using the responses. A changed draft/history needs a fresh assessment. Session questions evaluate the current trajectory in the provided context; the framework does not maintain a global user risk score or infer a trend solely from a score increase. The chat application persists evidence and actual outcomes with a revision check and owns summary building and retention.

The caller uses these signals to steer the agent with its own trusted instructions. A nudge is a mitigation attempt, and the caller can assess the resulting draft at `output_release` before releasing it. Tool execution permissions remain with the caller. Adding families or changing question wording requires application evaluation and calibration; this example registry is deliberately small and does not establish global safety coverage.

## Optional legacy policy gate

The following action-mapping API is retained for callers that explicitly select it. `RiskAssessor` never invokes it. The earlier `synthetic-policy.json` is an action-policy replay fixture; the primary family API uses `risk-registry.json`.

1. The application builds `SafetyInput` from verified identity, current content, bounded recent turns, and a server-produced summary. Never put secrets or unrelated history in the input.
2. `SafetyGate` checks context limits and hard rules. If they pass, an evaluator returns category signals. Jev is an optional evaluator, configured with application-owned questions.
3. `PolicyBundle` maps signals to `allow`, `constrain`, `review`, or `refuse`. `shadow` mode records recommendations; `enforce` mode makes them effective. The consuming application must enforce the returned action before model generation or output release.
4. Missing or invalid evaluator results take a degraded fallback. Low-risk text input is constrained with tools ineligible; high-risk input and tool/output stages are held for review.
5. An `allow` decision makes a request eligible for normal processing. **It never grants tool permission:** the application must still check actor, scope, resource, arguments, and approvals.

## Legacy policy inputs and session tracker

The trusted caller configures `SafetyGate` with a versioned `PolicyBundle`, hard rules, and an evaluator whose questions it owns. For every check, it passes a `SafetyInput` with deterministic arguments: stage, content or action summary, actor/surface, requested enabled focus categories, assessment scopes, and verified context. The evaluator asks its configured model questions and returns category signals. The gate maps those signals through policy and returns `SafetyDecision.recommended_action` and `effective_action` to the caller. The caller applies the effective action (`allow`, `constrain`, `review`, or `refuse`) in its own workflow. Model output cannot directly grant a tool permission or override a hard rule. Caller-provided focus categories cannot change the configured questions or thresholds.

### Caller-owned JSON policy

The caller may keep a versioned `chat-safety.policy/1` JSON file and load it with `load_policy_json`. [runtime-policy.json](examples/runtime-policy.json) is a **synthetic example**. It defines required and optional category IDs, a description and message/session Jev question for each enabled category, and threshold bands for every stage. `thresholds` apply to the latest event; `session_thresholds` apply to trajectory signals. A band gives `review_at`, `refuse_at`, and the action to return in each triggered band. The loader rejects unknown category IDs, missing definitions, missing stage/category coverage, bad threshold ordering, and duplicate JSON keys before a model call.

`Category` is the legacy action API's fixed identifier registry. Its restriction on new identifiers does not apply to `RiskRegistry`, whose family and subfamily IDs come from caller configuration. Hard rules and tool authorization remain application code, and credentials remain in the caller's secret store.

```python
from chat_safety import SafetyGate, load_policy_json
from chat_safety.adapters.jev import JevEvaluator

configured = load_policy_json("/trusted/config/chat-safety-policy.json")
evaluator = JevEvaluator(
    api_key=key_from_secret_store,
    questions=configured.message_questions,
    session_questions=configured.session_questions,
    mandatory_categories=configured.policy.required_categories,
)
gate = SafetyGate(configured.policy, evaluator, hard_rules=backend_hard_rules)
# decision = await gate.evaluate(caller_built_safety_input)
```

`SafetyInput.stage` identifies the event: user input, tool proposal, tool result, or assistant output release. `content` is the actual message or a bounded action/result summary. A policy has mandatory and enabled optional categories. `focus_categories` lets a caller add an enabled optional category such as `medical_advice`; it never narrows the mandatory set. A category name alone supplies no threshold or safe interpretation. The consuming product owns Jev question wording and threshold bands. Application sharing and export use backend authorization and privacy rules; they are not Jev stages.

To assess a conversation trajectory, set `assessment_scopes` to both `MESSAGE` and `SESSION`, and supply a server-owned `session_ref`, revision, bounded recent turns, and a server-produced safety summary. Recent turns may include user, assistant, tool proposal, and tool result events. The Jev adapter asks separately about the latest content and the trajectory in one request; the gate requires complete signals for both scopes and applies their configured thresholds. Missing session questions, thresholds, or signals yield a degraded hold. A benign latest message can therefore be held because the session context is concerning.

```python
from chat_safety import AssessmentScope, Category, RecentTurn, SafetyInput, Stage

request = SafetyInput(
    request_id="opaque-event-id",
    policy_version="example-policy/1",
    stage=Stage.INPUT,
    surface="chat",
    actor_ref="opaque-actor-id",
    content="What should I do next?",
    focus_categories=(Category.MEDICAL_ADVICE,),
    assessment_scopes=(AssessmentScope.MESSAGE, AssessmentScope.SESSION),
    session_ref="opaque-session-id",
    session_revision=7,
    safety_summary="Server-produced, bounded summary of earlier safety-relevant events.",
    recent_turns=(
        RecentTurn("user", "Bounded prior user text"),
        RecentTurn("assistant", "Bounded prior agent answer"),
        RecentTurn("tool_result", "Bounded tool-result summary"),
    ),
)
```

`SessionTracker` builds the next assessment from a `SessionSafetyState` and appends a redacted `SessionEvent` after the outcome is known. Events cover user messages, assistant output, tool proposals, and tool results. The state retains a bounded window, content hashes, decisions, outcomes, and a trusted summary. The package defines a `SessionStore` interface; Bahulam must implement durable storage and atomic compare-and-swap on the revision. A past refusal is context, not a permanent user label. Tool permission and exact user approval are separate backend checks. The `medical_advice` category is an experimental contract label for unsafe personalized guidance; it is not a judgment that general medical education or research is disallowed.

For a session, the caller loads a trusted state, passes the same per-event arguments through `SessionTracker.build_input`, and calls `SafetyGate.evaluate`. The tracker adds bounded history and requests both message and session signals once history exists. The caller enforces the returned action, then appends the **actual** outcome (`accepted`, `held`, `denied`, `executed`, or `released`) and persists the next revision atomically. A review or refusal cannot be recorded as accepted, executed, or released. See the runnable [caller session example](examples/caller_session.py); its scores and thresholds are synthetic. `state_to_record` and `state_from_record` serialize the bounded state without raw message content.

`ThresholdBand` numbers in tests are synthetic. Real thresholds require a human-labeled corpus, calibration, a locked test set, and approval by the consuming product's policy owners.

## Local checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python examples/family_nudge.py
PYTHONPATH=src python examples/minimal.py
PYTHONPATH=src python examples/caller_session.py
PYTHONPATH=src python -m chat_safety.eval_cli \
  --policy examples/synthetic-policy.json \
  --cases examples/synthetic-cases.jsonl \
  --list-decisions --fail-on-mismatch
```

The replay command is offline: it reads labels and previously recorded category probabilities, then runs the policy engine. It prints aggregate results, category precision and recall, slice summaries, and optional case IDs without printing case text. The sample policy and scores are **invented**, so a perfect replay score proves only that routing and fallback logic match the fixtures. The chats cover benign and authorized requests, progressive misuse across turns, refusal, supportive responses, tool and output checks, unscanned attachments, and provider failures. The sample thresholds are not approved for production.

To test Jev itself on the bundled **synthetic** chats, use a dedicated OpenRouter key and an explicit live command:

```bash
export CHAT_SAFETY_OPENROUTER_API_KEY='your-dedicated-test-key'
PYTHONPATH=src python examples/live_jev_eval.py --run-live
```

This sends the example chat text to OpenRouter and may incur charges. It refuses to run without the key and synthetic markers, and it prints case IDs, scores, decisions, and the returned model snapshot without printing prompts or the key. Any degraded case makes the command fail because fallback decisions cannot measure Jev accuracy. The questions are experimental. On 2026-10-30, after enabling `typesafe` for the Azure development OpenRouter key, Jev `typesafe/jev-1.14-20261014` returned scores for all 16 model cases: 15 expected actions matched, with the borderline fraud case allowed instead of reviewed. The sample policy had no production calibration; this is a connectivity and smoke-test result. For a real evaluation, keep the labeled corpus outside the public repository, split calibration and locked test sets by scenario family, and record model snapshot and question-set version with every score set.

To use the HTTP adapter, install the optional `jev` dependency (`pip install '.[jev]'`). The package does not read environment variables or send requests unless the caller constructs and invokes `JevEvaluator` with a key.

## License

MIT. Review dependency and provenance records before public publication.
