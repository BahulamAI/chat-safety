"""Validate context bounds before any model call; never silently clip content."""

from dataclasses import dataclass

from .models import SafetyInput


class ContextTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class ContextLimits:
    max_content_chars: int = 12_000
    max_recent_turns: int = 8
    max_recent_turn_chars: int = 3_000
    max_summary_chars: int = 2_000
    max_surface_chars: int = 128
    max_authorization_facts: int = 32
    max_authorization_key_chars: int = 128
    max_attachment_kinds: int = 8
    max_focus_categories: int = 16
    max_session_ref_chars: int = 128

    def __post_init__(self) -> None:
        if min(self.max_content_chars, self.max_recent_turns,
               self.max_recent_turn_chars, self.max_summary_chars,
               self.max_surface_chars, self.max_authorization_facts,
               self.max_authorization_key_chars, self.max_attachment_kinds,
               self.max_focus_categories, self.max_session_ref_chars) <= 0:
            raise ValueError("context limits must be positive")

    def validate(self, request: SafetyInput) -> None:
        if len(request.content) > self.max_content_chars:
            raise ContextTooLarge("content exceeds configured bound")
        if len(request.recent_turns) > self.max_recent_turns:
            raise ContextTooLarge("recent turn count exceeds configured bound")
        if any(len(turn.text) > self.max_recent_turn_chars for turn in request.recent_turns):
            raise ContextTooLarge("recent turn exceeds configured bound")
        if len(request.safety_summary) > self.max_summary_chars:
            raise ContextTooLarge("safety summary exceeds configured bound")
        if len(request.surface) > self.max_surface_chars:
            raise ContextTooLarge("surface label exceeds configured bound")
        if len(request.authorization_facts) > self.max_authorization_facts:
            raise ContextTooLarge("authorization fact count exceeds configured bound")
        if any(len(key) > self.max_authorization_key_chars
               for key in request.authorization_facts):
            raise ContextTooLarge("authorization fact key exceeds configured bound")
        if len(request.attachment_kinds) > self.max_attachment_kinds:
            raise ContextTooLarge("attachment count exceeds configured bound")
        if any(len(kind) > self.max_surface_chars for kind in request.attachment_kinds):
            raise ContextTooLarge("attachment label exceeds configured bound")
        if len(request.focus_categories) > self.max_focus_categories:
            raise ContextTooLarge("focus category count exceeds configured bound")
        if request.session_ref and len(request.session_ref) > self.max_session_ref_chars:
            raise ContextTooLarge("session reference exceeds configured bound")
