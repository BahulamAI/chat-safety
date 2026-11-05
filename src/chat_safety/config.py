"""Load a caller-owned, versioned JSON policy and category question registry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .adapters.jev import NoulQuestion
from .context import ContextLimits
from .models import Action, Category, Mode, Stage
from .policy import PolicyBundle, ThresholdBand


_SCHEMA = "chat-safety.policy/1"
_MAX_CONFIG_BYTES = 262_144


@dataclass(frozen=True)
class CategoryDefinition:
    """Caller-maintained meaning and model criteria for a stable category ID."""

    description: str
    message_question: NoulQuestion
    session_question: NoulQuestion | None = None


@dataclass(frozen=True)
class PolicyConfiguration:
    policy: PolicyBundle
    categories: Mapping[Category, CategoryDefinition]

    @property
    def message_questions(self) -> Mapping[Category, NoulQuestion]:
        return MappingProxyType({key: item.message_question
                                 for key, item in self.categories.items()})

    @property
    def session_questions(self) -> Mapping[Category, NoulQuestion]:
        return MappingProxyType({key: item.session_question
                                 for key, item in self.categories.items()
                                 if item.session_question is not None})


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _keys(value: dict[str, Any], label: str, required: set[str], optional: set[str] = set()) -> None:
    if missing := required - value.keys():
        raise ValueError(f"{label} is missing {', '.join(sorted(missing))}")
    if extra := value.keys() - required - optional:
        raise ValueError(f"{label} has unknown keys: {', '.join(sorted(extra))}")


def _categories(value: Any, label: str) -> frozenset[Category]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a nonempty list of category IDs")
    parsed = [Category(item) for item in value]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} contains duplicate categories")
    return frozenset(parsed)


def _question(value: Any, label: str) -> NoulQuestion:
    raw = _object(value, label)
    _keys(raw, label, {"instructions", "true_criteria", "false_criteria"})
    if any(not isinstance(raw[key], str) or len(raw[key]) > 4_000
           for key in ("instructions", "true_criteria", "false_criteria")):
        raise ValueError(f"{label} fields must be strings of at most 4000 characters")
    return NoulQuestion(**raw)


def _bands(value: Any, label: str) -> dict[Stage, dict[Category, ThresholdBand]]:
    raw = _object(value, label)
    result: dict[Stage, dict[Category, ThresholdBand]] = {}
    for stage_id, categories in raw.items():
        stage = Stage(stage_id)
        if stage in result:
            raise ValueError(f"{label} contains duplicate stage")
        result[stage] = {}
        for category_id, band in _object(categories, f"{label}.{stage_id}").items():
            category = Category(category_id)
            spec = _object(band, f"{label}.{stage_id}.{category_id}")
            _keys(spec, f"{label}.{stage_id}.{category_id}",
                  {"review_at", "refuse_at"}, {"middle_action", "upper_action"})
            for key in ("review_at", "refuse_at"):
                if isinstance(spec[key], bool) or not isinstance(spec[key], (int, float)):
                    raise ValueError(f"{label}.{stage_id}.{category_id}.{key} must be numeric")
            result[stage][category] = ThresholdBand(
                review_at=float(spec["review_at"]),
                refuse_at=float(spec["refuse_at"]),
                middle_action=Action(spec.get("middle_action", "review")),
                upper_action=Action(spec.get("upper_action", "refuse")),
            )
    return result


def parse_policy_config(value: Any) -> PolicyConfiguration:
    """Validate all category IDs, questions, and threshold coverage before use."""
    raw = _object(value, "policy")
    _keys(raw, "policy", {"schema", "version", "mode", "required_categories",
                          "categories", "thresholds"},
          {"optional_categories", "session_thresholds", "limits", "note"})
    if raw["schema"] != _SCHEMA:
        raise ValueError("unsupported policy schema")
    if not isinstance(raw["version"], str) or not raw["version"].strip():
        raise ValueError("policy version must be nonempty text")
    required = _categories(raw["required_categories"], "required_categories")
    optional_raw = raw.get("optional_categories", [])
    if not isinstance(optional_raw, list):
        raise ValueError("optional_categories must be a list")
    optional = (_categories(optional_raw, "optional_categories")
                if optional_raw else frozenset())
    enabled = required | optional
    if required & optional:
        raise ValueError("required and optional categories overlap")
    definitions: dict[Category, CategoryDefinition] = {}
    for category_id, item in _object(raw["categories"], "categories").items():
        category = Category(category_id)
        spec = _object(item, f"categories.{category_id}")
        _keys(spec, f"categories.{category_id}",
              {"description", "message_question"}, {"session_question"})
        description = spec["description"]
        if not isinstance(description, str) or not description.strip() or len(description) > 1_000:
            raise ValueError(f"categories.{category_id}.description must be bounded text")
        definitions[category] = CategoryDefinition(
            description=description,
            message_question=_question(spec["message_question"],
                                       f"categories.{category_id}.message_question"),
            session_question=(_question(spec["session_question"],
                                        f"categories.{category_id}.session_question")
                              if "session_question" in spec else None),
        )
    if set(definitions) != enabled:
        raise ValueError("category definitions must match enabled categories")
    thresholds = _bands(raw["thresholds"], "thresholds")
    if set(thresholds) != set(Stage):
        raise ValueError("message thresholds must cover every stage")
    session_thresholds = _bands(raw.get("session_thresholds", {}), "session_thresholds")
    if session_thresholds and set(session_thresholds) != set(Stage):
        raise ValueError("session thresholds must cover every stage when enabled")
    if session_thresholds and any(item.session_question is None for item in definitions.values()):
        raise ValueError("session thresholds require session questions for every category")
    limits = _object(raw.get("limits", {}), "limits")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in limits.values()):
        raise ValueError("context limits must be integers")
    try:
        context_limits = ContextLimits(**limits)
    except TypeError as exc:
        raise ValueError("invalid context limits") from exc
    policy = PolicyBundle(
        version=raw["version"], mode=Mode(raw["mode"]),
        thresholds=thresholds, required_categories=required,
        optional_categories=optional, session_thresholds=session_thresholds,
        limits=context_limits,
    )
    return PolicyConfiguration(policy, MappingProxyType(definitions))


def load_policy_json(path: str | Path) -> PolicyConfiguration:
    """Read a bounded local JSON file controlled by the calling application."""
    return parse_policy_config(_read_config_json(path))


def _read_config_json(path: str | Path, max_bytes: int = _MAX_CONFIG_BYTES) -> dict:
    policy_path = Path(path)
    with policy_path.open("rb") as source:
        document = source.read(max_bytes + 1)
    if len(document) > max_bytes:
        raise ValueError("configuration JSON exceeds size limit")
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON policy key: {key}")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON numeric constant: {value}")

    return json.loads(
        document.decode("utf-8"), object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
