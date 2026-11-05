"""Bounded, backend-persisted safety state for a conversation trajectory.

This module does not store raw messages or provide a production database. A
caller supplies redacted summaries and persists each returned state using a
compare-and-swap operation on ``revision``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
from typing import Mapping, Protocol

from .models import (
    Action, AssessmentScope, Category, RecentTurn, RiskTier, SafetyDecision,
    SafetyInput, Signal, Stage, Mode, ScannerStatus,
)


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_FOR_STAGE = {
    Stage.INPUT: "user",
    Stage.TOOL_PROPOSAL: "tool_proposal",
    Stage.TOOL_RESULT: "tool_result",
    Stage.OUTPUT_RELEASE: "assistant",
}
_OUTCOMES_FOR_STAGE = {
    Stage.INPUT: {"accepted", "denied", "held"},
    Stage.TOOL_PROPOSAL: {"executed", "denied", "held"},
    Stage.TOOL_RESULT: {"accepted", "denied", "held"},
    Stage.OUTPUT_RELEASE: {"released", "denied", "held"},
}


class SessionConflict(ValueError):
    """A stale revision or out-of-order event was supplied."""


class EventOutcome(str, Enum):
    ACCEPTED = "accepted"
    DENIED = "denied"
    EXECUTED = "executed"
    HELD = "held"
    RELEASED = "released"


def content_digest(content: str) -> str:
    """Hash content for an event record without retaining that content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SessionEvent:
    sequence: int
    content_hash: str
    summary: str
    decision: SafetyDecision
    outcome: EventOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.decision, SafetyDecision) or not isinstance(self.outcome, EventOutcome):
            raise ValueError("event requires a safety decision and a typed outcome")
        if not isinstance(self.summary, str) or not isinstance(self.content_hash, str):
            raise ValueError("event summary and content hash must be strings")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("event sequence must be a positive integer")
        if not _HASH_RE.fullmatch(self.content_hash):
            raise ValueError("event content hash must be a lowercase SHA-256 digest")
        if not self.summary.strip():
            raise ValueError("event requires a redacted summary")
        if self.decision.stage not in _ROLE_FOR_STAGE:
            raise ValueError("unsupported session event stage")
        if self.outcome.value not in _OUTCOMES_FOR_STAGE[self.decision.stage]:
            raise ValueError("outcome is invalid for the event stage")
        if (self.decision.effective_action in {Action.REVIEW, Action.REFUSE}
                and self.outcome in {EventOutcome.ACCEPTED, EventOutcome.EXECUTED,
                                     EventOutcome.RELEASED}):
            raise ValueError("a held or refused event cannot be accepted, executed, or released")
        if (self.decision.stage is Stage.TOOL_PROPOSAL
                and self.outcome is EventOutcome.EXECUTED
                and self.decision.effective_action is not Action.ALLOW):
            raise ValueError("only a safety-allowed tool proposal may execute")

    @property
    def role(self) -> str:
        return _ROLE_FOR_STAGE[self.decision.stage]


@dataclass(frozen=True)
class SessionSafetyState:
    session_ref: str
    policy_version: str
    revision: int = 0
    last_sequence: int = 0
    summary: str = ""
    summary_version: str = ""
    recent_events: tuple[SessionEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) for value in (
            self.session_ref, self.policy_version, self.summary, self.summary_version
        )):
            raise ValueError("session references and summaries must be strings")
        if not self.session_ref.strip() or not self.policy_version.strip():
            raise ValueError("session and policy references are required")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (self.revision, self.last_sequence)):
            raise ValueError("revision and last sequence must be nonnegative integers")
        if self.last_sequence and not self.revision:
            raise ValueError("a session with events must have a revision")
        events = tuple(self.recent_events)
        if events and (events[-1].sequence != self.last_sequence or
                       any(left.sequence >= right.sequence for left, right in zip(events, events[1:]))):
            raise ValueError("recent events must be ordered and end at last sequence")
        if any(item.decision.policy_version != self.policy_version for item in events):
            raise ValueError("recent events must use the session policy version")
        object.__setattr__(self, "recent_events", events)


class SessionStore(Protocol):
    """Bahulam implements this with durable, atomic revision checks."""

    async def load(self, session_ref: str) -> SessionSafetyState | None: ...

    async def compare_and_swap(
        self, state: SessionSafetyState, *, expected_revision: int
    ) -> bool: ...


def state_to_record(state: SessionSafetyState) -> dict:
    """Produce a JSON-compatible record with summaries and codes, not raw content."""
    return {
        "schema": "chat-safety.session/1",
        "session_ref": state.session_ref,
        "policy_version": state.policy_version,
        "revision": state.revision,
        "last_sequence": state.last_sequence,
        "summary": state.summary,
        "summary_version": state.summary_version,
        "recent_events": [
            {
                "sequence": event.sequence,
                "content_hash": event.content_hash,
                "summary": event.summary,
                "outcome": event.outcome.value,
                "decision": {
                    "request_id": event.decision.request_id,
                    "policy_version": event.decision.policy_version,
                    "stage": event.decision.stage.value,
                    "mode": event.decision.mode.value,
                    "recommended_action": event.decision.recommended_action.value,
                    "effective_action": event.decision.effective_action.value,
                    "reason_code": event.decision.reason_code,
                    "public_reason_code": event.decision.public_reason_code,
                    "evaluator": event.decision.evaluator,
                    "model_version": event.decision.model_version,
                    "degraded": event.decision.degraded,
                    "tool_eligible": event.decision.tool_eligible,
                    "signals": [
                        {
                            "category": signal.category.value,
                            "scope": signal.scope.value,
                            "probability": signal.probability,
                            "confidence": signal.confidence,
                            "evidence_codes": list(signal.evidence_codes),
                        }
                        for signal in event.decision.signals
                    ],
                },
            }
            for event in state.recent_events
        ],
    }


def state_from_record(record: dict) -> SessionSafetyState:
    """Validate a persisted record before it enters the next model request."""
    if not isinstance(record, dict) or record.get("schema") != "chat-safety.session/1":
        raise ValueError("unsupported session state schema")
    try:
        events = []
        for raw_event in record["recent_events"]:
            raw_decision = raw_event["decision"]
            signals = tuple(
                Signal(
                    category=Category(item["category"]),
                    scope=AssessmentScope(item["scope"]),
                    probability=item["probability"],
                    confidence=item["confidence"],
                    evidence_codes=tuple(item["evidence_codes"]),
                )
                for item in raw_decision["signals"]
            )
            decision = SafetyDecision(
                request_id=raw_decision["request_id"],
                policy_version=raw_decision["policy_version"],
                stage=Stage(raw_decision["stage"]),
                mode=Mode(raw_decision["mode"]),
                recommended_action=Action(raw_decision["recommended_action"]),
                effective_action=Action(raw_decision["effective_action"]),
                reason_code=raw_decision["reason_code"],
                public_reason_code=raw_decision["public_reason_code"],
                evaluator=raw_decision["evaluator"],
                model_version=raw_decision["model_version"],
                degraded=raw_decision["degraded"],
                tool_eligible=raw_decision["tool_eligible"],
                signals=signals,
            )
            events.append(SessionEvent(
                sequence=raw_event["sequence"],
                content_hash=raw_event["content_hash"],
                summary=raw_event["summary"],
                decision=decision,
                outcome=EventOutcome(raw_event["outcome"]),
            ))
        return SessionSafetyState(
            session_ref=record["session_ref"],
            policy_version=record["policy_version"],
            revision=record["revision"],
            last_sequence=record["last_sequence"],
            summary=record["summary"],
            summary_version=record["summary_version"],
            recent_events=tuple(events),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid session state record") from exc


@dataclass(frozen=True)
class SessionTracker:
    max_recent_events: int = 8
    max_event_summary_chars: int = 3_000
    max_summary_chars: int = 2_000

    def __post_init__(self) -> None:
        if min(self.max_recent_events, self.max_event_summary_chars,
               self.max_summary_chars) <= 0:
            raise ValueError("session bounds must be positive")

    def build_input(
        self, state: SessionSafetyState, *, request_id: str, stage: Stage,
        surface: str, actor_ref: str, content: str,
        tenant_ref: str | None = None,
        focus_categories: tuple[Category, ...] = (),
        risk_tier: RiskTier = RiskTier.LOW,
        attachment_kinds: tuple[str, ...] = (),
        scanner_status: ScannerStatus = ScannerStatus.NOT_APPLICABLE,
        authorization_facts: Mapping[str, bool] | None = None,
    ) -> SafetyInput:
        """Build the next bounded assessment from a trusted state snapshot."""
        if len(state.summary) > self.max_summary_chars:
            raise ValueError("session summary exceeds configured bound")
        if len(state.recent_events) > self.max_recent_events:
            raise ValueError("session event count exceeds configured bound")
        if any(len(event.summary) > self.max_event_summary_chars
               for event in state.recent_events):
            raise ValueError("session event summary exceeds configured bound")
        has_history = bool(state.recent_events or state.summary)
        return SafetyInput(
            request_id=request_id,
            policy_version=state.policy_version,
            stage=stage,
            surface=surface,
            actor_ref=actor_ref,
            content=content,
            tenant_ref=tenant_ref,
            recent_turns=tuple(RecentTurn(event.role, event.summary)
                               for event in state.recent_events),
            safety_summary=state.summary,
            risk_tier=risk_tier,
            attachment_kinds=attachment_kinds,
            scanner_status=scanner_status,
            authorization_facts=authorization_facts or {},
            assessment_scopes=(AssessmentScope.MESSAGE, AssessmentScope.SESSION)
                if has_history else (AssessmentScope.MESSAGE,),
            focus_categories=focus_categories,
            session_ref=state.session_ref if has_history else None,
            session_revision=state.revision if has_history else None,
        )

    def append(
        self, state: SessionSafetyState, event: SessionEvent, *,
        expected_revision: int, next_summary: str,
        summary_version: str,
    ) -> SessionSafetyState:
        """Return a new state; the caller must persist it with CAS."""
        if expected_revision != state.revision or event.sequence != state.last_sequence + 1:
            raise SessionConflict("stale session revision or out-of-order event")
        if event.decision.policy_version != state.policy_version:
            raise SessionConflict("event policy version differs from the session")
        if any(previous.decision.request_id == event.decision.request_id
               for previous in state.recent_events):
            raise SessionConflict("duplicate event request ID")
        if len(event.summary) > self.max_event_summary_chars:
            raise ValueError("event summary exceeds configured bound")
        if len(next_summary) > self.max_summary_chars:
            raise ValueError("session summary exceeds configured bound")
        if next_summary and not summary_version.strip():
            raise ValueError("a nonempty summary requires a summary-builder version")
        return SessionSafetyState(
            session_ref=state.session_ref,
            policy_version=state.policy_version,
            revision=state.revision + 1,
            last_sequence=event.sequence,
            summary=next_summary,
            summary_version=summary_version,
            recent_events=(*state.recent_events, event)[-self.max_recent_events:],
        )
