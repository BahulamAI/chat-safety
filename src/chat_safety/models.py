"""Stable input, signal, and decision contracts for chat-safety/1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Mapping


_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class Stage(str, Enum):
    INPUT = "input"
    TOOL_PROPOSAL = "tool_proposal"
    TOOL_RESULT = "tool_result"
    OUTPUT_RELEASE = "output_release"


class Action(str, Enum):
    ALLOW = "allow"
    CONSTRAIN = "constrain"
    REVIEW = "review"
    REFUSE = "refuse"


class Mode(str, Enum):
    SHADOW = "shadow"
    ENFORCE = "enforce"


class RiskTier(str, Enum):
    LOW = "low"
    HIGH = "high"


class AssessmentScope(str, Enum):
    MESSAGE = "message"
    SESSION = "session"


class ScannerStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    CLEAN = "clean"
    PENDING = "pending"
    BLOCKED = "blocked"


class Category(str, Enum):
    CHILD_EXPLOITATION = "child_exploitation"
    NONCONSENSUAL_INTIMATE_CONTENT = "nonconsensual_intimate_content"
    VIOLENCE_WEAPONS = "violence_weapons"
    FRAUD_THEFT = "fraud_theft"
    UNAUTHORIZED_INTRUSION = "unauthorized_intrusion"
    SELF_HARM = "self_harm"
    HARASSMENT_HATE = "harassment_hate"
    PRIVACY_EXFILTRATION = "privacy_exfiltration"
    PROMPT_INJECTION = "prompt_injection"
    MEDICAL_ADVICE = "medical_advice"


@dataclass(frozen=True)
class RecentTurn:
    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant", "tool_proposal", "tool_result"}:
            raise ValueError("recent turn role must be user, assistant, or a tool event")


@dataclass(frozen=True)
class SafetyInput:
    request_id: str
    policy_version: str
    stage: Stage
    surface: str
    actor_ref: str
    content: str
    tenant_ref: str | None = None
    recent_turns: tuple[RecentTurn, ...] = ()
    safety_summary: str = ""
    risk_tier: RiskTier = RiskTier.LOW
    attachment_kinds: tuple[str, ...] = ()
    scanner_status: ScannerStatus = ScannerStatus.NOT_APPLICABLE
    authorization_facts: Mapping[str, bool] = field(default_factory=dict)
    assessment_scopes: tuple[AssessmentScope, ...] = (AssessmentScope.MESSAGE,)
    focus_categories: tuple[Category, ...] = ()
    session_ref: str | None = None
    session_revision: int | None = None

    def __post_init__(self) -> None:
        for value in (self.request_id, self.policy_version, self.surface, self.actor_ref):
            if not value or not value.strip():
                raise ValueError("request, policy, surface, and actor references are required")
        if not self.content or not self.content.strip():
            raise ValueError("content or action summary is required")
        if any(not isinstance(value, bool) for value in self.authorization_facts.values()):
            raise ValueError("authorization facts must be backend-verified booleans")
        if any(not isinstance(turn, RecentTurn) for turn in self.recent_turns):
            raise ValueError("recent turns must use RecentTurn")
        if self.attachment_kinds and self.scanner_status is ScannerStatus.NOT_APPLICABLE:
            raise ValueError("attachments require an explicit scanner status")
        if any(not isinstance(scope, AssessmentScope) for scope in self.assessment_scopes):
            raise ValueError("assessment scopes must use AssessmentScope")
        if not self.assessment_scopes or len(set(self.assessment_scopes)) != len(self.assessment_scopes):
            raise ValueError("assessment scopes must be nonempty and unique")
        if any(not isinstance(category, Category) for category in self.focus_categories):
            raise ValueError("focus categories must use Category")
        if len(set(self.focus_categories)) != len(self.focus_categories):
            raise ValueError("focus categories must be unique")
        if AssessmentScope.SESSION in self.assessment_scopes:
            if not self.session_ref or not self.session_ref.strip():
                raise ValueError("session assessment requires a server-owned session reference")
            if isinstance(self.session_revision, bool) or not isinstance(self.session_revision, int) or self.session_revision < 0:
                raise ValueError("session assessment requires a nonnegative session revision")
            if not self.recent_turns and not self.safety_summary:
                raise ValueError("session assessment requires bounded conversation context")
        object.__setattr__(self, "recent_turns", tuple(self.recent_turns))
        object.__setattr__(self, "attachment_kinds", tuple(self.attachment_kinds))
        object.__setattr__(self, "assessment_scopes", tuple(self.assessment_scopes))
        object.__setattr__(self, "focus_categories", tuple(self.focus_categories))
        object.__setattr__(self, "authorization_facts",
                           MappingProxyType(dict(self.authorization_facts)))


@dataclass(frozen=True)
class Signal:
    category: Category
    probability: float
    evidence_codes: tuple[str, ...] = ()
    confidence: float | None = None
    scope: AssessmentScope = AssessmentScope.MESSAGE

    def __post_init__(self) -> None:
        if not 0 <= self.probability <= 1:
            raise ValueError("signal probability must be between zero and one")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("signal confidence must be between zero and one")
        if any(not _CODE_RE.fullmatch(code) for code in self.evidence_codes):
            raise ValueError("evidence codes must be bounded symbolic codes")
        if not isinstance(self.scope, AssessmentScope):
            raise ValueError("signal scope must use AssessmentScope")
        object.__setattr__(self, "evidence_codes", tuple(self.evidence_codes))


@dataclass(frozen=True)
class Evaluation:
    signals: tuple[Signal, ...]
    evaluator: str
    model_version: str

    def __post_init__(self) -> None:
        if not self.signals or not self.evaluator or not self.model_version:
            raise ValueError("evaluation requires signals, evaluator, and model version")


@dataclass(frozen=True)
class SafetyDecision:
    request_id: str
    policy_version: str
    stage: Stage
    mode: Mode
    recommended_action: Action
    effective_action: Action
    reason_code: str
    public_reason_code: str | None = None
    signals: tuple[Signal, ...] = ()
    evaluator: str | None = None
    model_version: str | None = None
    degraded: bool = False
    tool_eligible: bool = False

    def __post_init__(self) -> None:
        if self.tool_eligible and self.effective_action is not Action.ALLOW:
            raise ValueError("blocked decisions cannot be eligible for tool authorization")
        if not _CODE_RE.fullmatch(self.reason_code):
            raise ValueError("reason code must be a bounded symbolic code")
        if self.public_reason_code is not None and not _CODE_RE.fullmatch(self.public_reason_code):
            raise ValueError("public reason code must be a bounded symbolic code")
