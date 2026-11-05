"""Provider-neutral criteria for a binary model judgment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryQuestion:
    instructions: str
    true_criteria: str
    false_criteria: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() or len(value) > 4_000
               for value in (self.instructions, self.true_criteria, self.false_criteria)):
            raise ValueError("question instructions and criteria must be nonempty bounded text")
