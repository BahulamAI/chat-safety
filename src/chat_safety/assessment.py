"""Evidence-only family and subfamily assessment; callers own all actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Mapping, Protocol

from .context import ContextLimits
from .models import AssessmentScope, RecentTurn, SafetyInput, Stage
from .questions import BinaryQuestion


_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("risk IDs must be lowercase identifiers of at most 64 characters")


def _positive_int(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("top_k_families must be a positive integer")


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RiskDefinition:
    description: str
    questions: Mapping[AssessmentScope, BinaryQuestion]

    def __post_init__(self) -> None:
        if (not isinstance(self.description, str) or not self.description.strip()
                or len(self.description) > 1_000):
            raise ValueError("risk description must be nonempty bounded text")
        if (set(self.questions) != set(AssessmentScope)
                or any(not isinstance(key, AssessmentScope) for key in self.questions)
                or any(not isinstance(item, BinaryQuestion) for item in self.questions.values())):
            raise ValueError("every risk needs typed message and session questions")
        object.__setattr__(self, "questions", MappingProxyType(dict(self.questions)))


@dataclass(frozen=True)
class RiskFamily(RiskDefinition):
    subfamilies: Mapping[str, RiskDefinition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        for key, value in self.subfamilies.items():
            _identifier(key)
            if type(value) is not RiskDefinition:
                raise ValueError("subfamilies must be leaf RiskDefinition values")
        object.__setattr__(self, "subfamilies", MappingProxyType(dict(self.subfamilies)))


@dataclass(frozen=True)
class RiskRegistry:
    version: str
    families: Mapping[str, RiskFamily]
    top_k_families: int = 3
    limits: ContextLimits = field(default_factory=ContextLimits)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("registry version is required")
        _positive_int(self.top_k_families)
        if not self.families or not isinstance(self.limits, ContextLimits):
            raise ValueError("a registry needs families and context limits")
        for key, value in self.families.items():
            _identifier(key)
            if not isinstance(value, RiskFamily):
                raise ValueError("families must use RiskFamily")
        object.__setattr__(self, "families", MappingProxyType(dict(self.families)))

    @property
    def fingerprint(self) -> str:
        def definition(item: RiskDefinition) -> dict:
            result = {"description": item.description,
                      "questions": {scope.value: asdict(question)
                                    for scope, question in item.questions.items()}}
            if isinstance(item, RiskFamily):
                result["subfamilies"] = {key: definition(child)
                                         for key, child in item.subfamilies.items()}
            return result
        return _digest({"version": self.version, "top_k_families": self.top_k_families,
                        "limits": asdict(self.limits),
                        "families": {key: definition(item) for key, item in self.families.items()}})


@dataclass(frozen=True)
class AssessmentInput:
    request_id: str
    registry_version: str
    stage: Stage
    content: str
    surface: str = "chat"
    assessment_scopes: tuple[AssessmentScope, ...] = (AssessmentScope.MESSAGE,)
    recent_turns: tuple[RecentTurn, ...] = ()
    safety_summary: str = ""
    session_ref: str | None = None
    session_revision: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, Stage):
            raise ValueError("stage must use Stage")
        if any(not isinstance(value, str) for value in (
            self.request_id, self.registry_version, self.content, self.surface, self.safety_summary
        )):
            raise ValueError("assessment content and references must be strings")
        if any(not isinstance(turn, RecentTurn) or not isinstance(turn.text, str)
               for turn in self.recent_turns):
            raise ValueError("recent turns must have typed roles and text")
        if (self.session_ref is None) != (self.session_revision is None):
            raise ValueError("session reference and revision must be supplied together")
        if self.session_ref is not None:
            if not isinstance(self.session_ref, str) or not self.session_ref.strip():
                raise ValueError("session reference must be nonempty text")
            if (isinstance(self.session_revision, bool) or not isinstance(self.session_revision, int)
                    or self.session_revision < 0):
                raise ValueError("session revision must be a nonnegative integer")
        # Reuse structural context validation; this never invokes SafetyGate or policy actions.
        context = self._context()
        object.__setattr__(self, "recent_turns", context.recent_turns)
        object.__setattr__(self, "assessment_scopes", context.assessment_scopes)

    def _context(self) -> SafetyInput:
        return SafetyInput(
            request_id=self.request_id, policy_version=self.registry_version,
            stage=self.stage, surface=self.surface, actor_ref="assessment",
            content=self.content, assessment_scopes=self.assessment_scopes,
            recent_turns=self.recent_turns, safety_summary=self.safety_summary,
            session_ref=self.session_ref, session_revision=self.session_revision,
        )

    def model_state(self) -> dict:
        return {"stage": self.stage.value, "surface": self.surface,
                "latest_content": self.content,
                "assessment_scopes": [scope.value for scope in self.assessment_scopes],
                "recent_turns": [asdict(turn) for turn in self.recent_turns],
                "safety_summary": self.safety_summary,
                "session_revision": self.session_revision}

    @property
    def fingerprint(self) -> str:
        return _digest({**self.model_state(), "session_ref": self.session_ref,
                        "registry_version": self.registry_version})


@dataclass(frozen=True)
class RiskQuestion:
    family_id: str
    scope: AssessmentScope
    question: BinaryQuestion
    subfamily_id: str | None = None

    @property
    def key(self) -> tuple:
        return self.scope, self.family_id, self.subfamily_id


@dataclass(frozen=True)
class RiskScore:
    family_id: str
    scope: AssessmentScope
    probability: float
    subfamily_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.family_id)
        if self.subfamily_id is not None:
            _identifier(self.subfamily_id)
        if not isinstance(self.scope, AssessmentScope):
            raise ValueError("score scope must use AssessmentScope")
        if (isinstance(self.probability, bool) or not isinstance(self.probability, (int, float))
                or not math.isfinite(self.probability) or not 0 <= self.probability <= 1):
            raise ValueError("risk probability must be a finite number between zero and one")

    @property
    def key(self) -> tuple:
        return self.scope, self.family_id, self.subfamily_id


@dataclass(frozen=True)
class RiskEvaluation:
    scores: tuple[RiskScore, ...]
    evaluator: str
    model_version: str


class RiskEvaluator(Protocol):
    async def evaluate(
        self, request: AssessmentInput, questions: tuple[RiskQuestion, ...]
    ) -> RiskEvaluation: ...


class AssessmentStatus(str, Enum):
    COMPLETE = "complete"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RiskAssessment:
    request_id: str
    registry_version: str
    registry_digest: str
    input_digest: str
    stage: Stage
    assessment_scopes: tuple[AssessmentScope, ...]
    session_ref: str | None
    session_revision: int | None
    level: str
    assessed_family_ids: tuple[str, ...]
    status: AssessmentStatus
    scores: tuple[RiskScore, ...] = ()
    top_families: tuple[RiskScore, ...] = ()
    top_k_families: int | None = None
    evaluator: str | None = None
    model_version: str | None = None
    error_code: str | None = None

    def to_dict(self) -> dict:
        """JSON-compatible result for caller storage or an API response; no raw content."""
        return {"schema": "chat-safety.assessment/1", **asdict(self)}


class RiskAssessor:
    def __init__(self, registry: RiskRegistry, evaluator: RiskEvaluator):
        self.registry = registry
        self.evaluator = evaluator

    async def assess_families(
        self, request: AssessmentInput, *, top_k_families: int | None = None,
    ) -> RiskAssessment:
        count = self.registry.top_k_families if top_k_families is None else top_k_families
        _positive_int(count)
        return await self._assess(request, tuple(sorted(self.registry.families)), "families", count)

    async def assess_subfamilies(
        self, request: AssessmentInput, *, family_ids: tuple[str, ...],
    ) -> RiskAssessment:
        if isinstance(family_ids, str) or not family_ids or len(set(family_ids)) != len(family_ids):
            raise ValueError("select a nonempty, unique collection of family IDs")
        for family_id in family_ids:
            if family_id not in self.registry.families:
                raise ValueError("requested family is not registered")
            if not self.registry.families[family_id].subfamilies:
                raise ValueError("requested family has no subfamilies")
        return await self._assess(request, tuple(sorted(family_ids)), "subfamilies", None)

    async def _assess(self, request, family_ids, level, top_k) -> RiskAssessment:
        if request.registry_version != self.registry.version:
            raise ValueError("request registry version does not match configuration")
        self.registry.limits.validate(request._context())
        questions = []
        for scope in request.assessment_scopes:
            for family_id in family_ids:
                family = self.registry.families[family_id]
                definitions = ({None: family} if level == "families" else family.subfamilies)
                for child_id, definition in definitions.items():
                    questions.append(RiskQuestion(family_id, scope, definition.questions[scope], child_id))
        metadata = dict(
            request_id=request.request_id, registry_version=self.registry.version,
            registry_digest=self.registry.fingerprint, input_digest=request.fingerprint,
            stage=request.stage, assessment_scopes=request.assessment_scopes,
            session_ref=request.session_ref, session_revision=request.session_revision,
            level=level, assessed_family_ids=family_ids, top_k_families=top_k,
        )
        try:
            evaluation = await self.evaluator.evaluate(request, tuple(questions))
            expected = {question.key for question in questions}
            scores = tuple(evaluation.scores)
            if (not all(isinstance(score, RiskScore) for score in scores)
                    or {score.key for score in scores} != expected or len(scores) != len(expected)
                    or not isinstance(evaluation.evaluator, str) or not evaluation.evaluator
                    or not isinstance(evaluation.model_version, str) or not evaluation.model_version):
                raise ValueError("invalid evaluator coverage or metadata")
        except Exception:
            # No fabricated zero scores, allow/refuse decision, retry, or second model call.
            return RiskAssessment(**metadata, status=AssessmentStatus.UNAVAILABLE,
                                  error_code="EVALUATION_UNAVAILABLE")
        ordered = tuple(sorted(scores, key=lambda score: (
            score.scope.value, -score.probability, score.family_id, score.subfamily_id or "")))
        top = tuple(score for scope in request.assessment_scopes
                    for score in tuple(s for s in ordered if s.scope is scope)[:top_k]) if top_k else ()
        return RiskAssessment(**metadata, status=AssessmentStatus.COMPLETE, scores=ordered,
                              top_families=top, evaluator=evaluation.evaluator,
                              model_version=evaluation.model_version)
