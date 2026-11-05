"""Versioned action thresholds; numbers are supplied by the consuming product."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .context import ContextLimits
from .models import Action, Category, Mode, Stage


@dataclass(frozen=True)
class ThresholdBand:
    review_at: float
    refuse_at: float
    middle_action: Action = Action.REVIEW
    upper_action: Action = Action.REFUSE

    def __post_init__(self) -> None:
        if not 0 <= self.review_at < self.refuse_at <= 1:
            raise ValueError("thresholds must satisfy 0 <= review_at < refuse_at <= 1")
        if self.middle_action is Action.ALLOW or self.upper_action is Action.ALLOW:
            raise ValueError("a triggered threshold cannot map to allow")
        priority = {Action.CONSTRAIN: 1, Action.REVIEW: 2, Action.REFUSE: 3}
        if priority[self.upper_action] < priority[self.middle_action]:
            raise ValueError("the upper threshold action cannot be less restrictive")

    def action_for(self, probability: float) -> Action:
        if probability >= self.refuse_at:
            return self.upper_action
        if probability >= self.review_at:
            return self.middle_action
        return Action.ALLOW


@dataclass(frozen=True)
class PolicyBundle:
    version: str
    mode: Mode
    thresholds: Mapping[Stage, Mapping[Category, ThresholdBand]]
    required_categories: frozenset[Category]
    limits: ContextLimits = field(default_factory=ContextLimits)
    session_thresholds: Mapping[Stage, Mapping[Category, ThresholdBand]] = field(default_factory=dict)
    optional_categories: frozenset[Category] = frozenset()

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("policy version is required")
        if not self.required_categories:
            raise ValueError("policy must declare its required category coverage")
        if self.required_categories & self.optional_categories:
            raise ValueError("required and optional categories must be disjoint")
        enabled_categories = self.required_categories | self.optional_categories
        for stage, bands in self.thresholds.items():
            if set(bands) != enabled_categories:
                raise ValueError(f"threshold coverage mismatch for {stage.value}")
        for stage, bands in self.session_thresholds.items():
            if set(bands) != enabled_categories:
                raise ValueError(f"session threshold coverage mismatch for {stage.value}")
        object.__setattr__(self, "thresholds", MappingProxyType({
            stage: MappingProxyType(dict(bands)) for stage, bands in self.thresholds.items()
        }))
        object.__setattr__(self, "session_thresholds", MappingProxyType({
            stage: MappingProxyType(dict(bands)) for stage, bands in self.session_thresholds.items()
        }))
        object.__setattr__(self, "required_categories", frozenset(self.required_categories))
        object.__setattr__(self, "optional_categories", frozenset(self.optional_categories))
