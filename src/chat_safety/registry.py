"""Strict JSON configuration for caller-defined family and subfamily criteria."""

from pathlib import Path

from .assessment import RiskDefinition, RiskFamily, RiskRegistry
from .config import _keys, _object, _question, _read_config_json
from .context import ContextLimits
from .models import AssessmentScope


def parse_risk_registry(value: dict) -> RiskRegistry:
    raw = _object(value, "registry")
    _keys(raw, "registry", {"schema", "version", "families"}, {"reporting", "limits", "note"})
    if raw["schema"] != "chat-safety.risks/1":
        raise ValueError("unsupported risk registry schema")
    reporting = _object(raw.get("reporting", {}), "reporting")
    _keys(reporting, "reporting", set(), {"top_k_families"})

    def definition(item, label, *, family=False):
        spec = _object(item, label)
        _keys(spec, label, {"description", "questions"}, {"subfamilies"} if family else set())
        questions = _object(spec["questions"], f"{label}.questions")
        _keys(questions, f"{label}.questions", {scope.value for scope in AssessmentScope})
        args = dict(description=spec["description"], questions={
            AssessmentScope(scope): _question(question, f"{label}.{scope}")
            for scope, question in questions.items()
        })
        if not family:
            return RiskDefinition(**args)
        children = _object(spec.get("subfamilies", {}), f"{label}.subfamilies")
        return RiskFamily(**args, subfamilies={
            key: definition(child, f"{label}.{key}") for key, child in children.items()
        })

    limits = _object(raw.get("limits", {}), "limits")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in limits.values()):
        raise ValueError("context limits must be integers")
    try:
        bounds = ContextLimits(**limits)
    except TypeError as exc:
        raise ValueError("invalid context limit setting") from exc
    return RiskRegistry(
        version=raw["version"], top_k_families=reporting.get("top_k_families", 3), limits=bounds,
        families={key: definition(item, f"families.{key}", family=True)
                  for key, item in _object(raw["families"], "families").items()},
    )


def load_risk_registry(path: str | Path) -> RiskRegistry:
    return parse_risk_registry(_read_config_json(path, max_bytes=2_097_152))
