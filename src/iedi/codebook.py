from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import (
    CodebookEntry,
    InterpretationRequest,
    MatchEvidence,
    PragmaticRule,
    require_string_sequence,
)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s']+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class PersonaProfile:
    profile_id: str
    profile_version: str
    display_name: str
    dialect: str
    cultural_context: str
    entry_ids: tuple[str, ...]
    pragmatic_rules: tuple[PragmaticRule, ...]
    reviewed_by: tuple[str, ...]
    review_status: str = "approved"

    def __post_init__(self) -> None:
        required = (
            self.profile_id,
            self.profile_version,
            self.display_name,
            self.dialect,
            self.cultural_context,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("persona fields cannot be empty")
        if self.review_status not in {"approved", "pending", "rejected", "superseded"}:
            raise ValueError(f"unsupported persona review_status: {self.review_status}")
        if not self.reviewed_by:
            raise ValueError("persona must record at least one reviewer")

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self))).hexdigest()


class Codebook:
    """Validated, append-only case retrieval with first-class persona evidence."""

    def __init__(
        self,
        entries: Iterable[CodebookEntry],
        personas: Iterable[PersonaProfile] = (),
        *,
        dataset_version: str = "unversioned",
    ) -> None:
        self.entries = tuple(entries)
        self.personas = {profile.profile_id: profile for profile in personas}
        self.dataset_version = dataset_version
        if not self.entries:
            raise ValueError("codebook requires at least one entry")

        entry_ids: set[str] = set()
        for entry in self.entries:
            if entry.entry_id in entry_ids:
                raise ValueError(f"duplicate entry_id: {entry.entry_id}")
            entry_ids.add(entry.entry_id)

        for profile in self.personas.values():
            missing = set(profile.entry_ids) - entry_ids
            if missing:
                raise ValueError(
                    f"persona {profile.profile_id} references unknown entries: {sorted(missing)}"
                )
            if profile.review_status == "approved":
                unapproved = {
                    entry_id
                    for entry_id in profile.entry_ids
                    if self._entry_status(self.entries, entry_id) != "approved"
                }
                if unapproved:
                    raise ValueError(
                        f"approved persona {profile.profile_id} references unapproved entries: "
                        f"{sorted(unapproved)}"
                    )
            for rule in profile.pragmatic_rules:
                if rule.preferred_entry_id not in entry_ids:
                    raise ValueError(
                        f"rule {rule.rule_id} references unknown entry {rule.preferred_entry_id}"
                    )

        for entry in self.entries:
            unknown_personas = set(entry.persona_ids) - set(self.personas)
            if unknown_personas:
                raise ValueError(
                    f"entry {entry.entry_id} references unknown personas: {sorted(unknown_personas)}"
                )

        self._by_id = {entry.entry_id: entry for entry in self.entries}
        self._active_entries = tuple(
            entry for entry in self.entries if entry.review_status == "approved"
        )
        self._validate_patterns()

    @classmethod
    def from_json(cls, path: str | Path) -> "Codebook":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Codebook":
        if not isinstance(payload, Mapping):
            raise ValueError("codebook root must be an object")
        raw_entries = payload.get("entries")
        raw_personas = payload.get("personas", [])
        if not isinstance(raw_entries, list):
            raise ValueError("entries must be an array")
        if not isinstance(raw_personas, list):
            raise ValueError("personas must be an array")

        entries = tuple(_parse_entry(item) for item in raw_entries)
        personas = tuple(_parse_persona(item) for item in raw_personas)
        return cls(
            entries,
            personas,
            dataset_version=str(payload.get("dataset_version", "unversioned")),
        )

    @property
    def version_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(
                {
                    "dataset_version": self.dataset_version,
                    "entries": [asdict(entry) for entry in self.entries],
                    "personas": [asdict(profile) for profile in self.personas.values()],
                }
            )
        ).hexdigest()

    def active_personas(self, profile_ids: Iterable[str]) -> tuple[PersonaProfile, ...]:
        profiles: list[PersonaProfile] = []
        for profile_id in profile_ids:
            profile = self.personas.get(profile_id)
            if profile is None:
                raise ValueError(f"unknown persona: {profile_id}")
            if profile.review_status != "approved":
                raise ValueError(f"persona is not approved: {profile_id}")
            profiles.append(profile)
        return tuple(profiles)

    def get_entry(self, entry_id: str) -> CodebookEntry:
        try:
            return self._by_id[entry_id]
        except KeyError as exc:
            raise KeyError(f"unknown entry: {entry_id}") from exc

    def append_version(self, entry: CodebookEntry) -> "Codebook":
        """Return a new materialized view while retaining superseded history."""

        if entry.entry_id in self._by_id:
            raise ValueError(f"entry_id already exists: {entry.entry_id}")
        updated = list(self.entries)
        profiles = list(self.personas.values())
        if entry.supersedes_entry_id is not None:
            previous = self.get_entry(entry.supersedes_entry_id)
            if entry.version <= previous.version:
                raise ValueError("new entry version must exceed the superseded version")
            updated = [
                replace(item, review_status="superseded")
                if item.entry_id == previous.entry_id
                else item
                for item in updated
            ]
            profiles = [
                replace(
                    profile,
                    entry_ids=tuple(
                        entry.entry_id if entry_id == previous.entry_id else entry_id
                        for entry_id in profile.entry_ids
                    ),
                    pragmatic_rules=tuple(
                        replace(rule, preferred_entry_id=entry.entry_id)
                        if rule.preferred_entry_id == previous.entry_id
                        else rule
                        for rule in profile.pragmatic_rules
                    ),
                )
                for profile in profiles
            ]
        else:
            profiles = [
                replace(profile, entry_ids=(*profile.entry_ids, entry.entry_id))
                if profile.profile_id in entry.persona_ids
                else profile
                for profile in profiles
            ]
        updated.append(entry)
        return Codebook(
            updated,
            profiles,
            dataset_version=f"{self.dataset_version}+{entry.entry_id}:v{entry.version}",
        )

    def search(
        self,
        request: InterpretationRequest,
        *,
        top_k: int = 5,
        fuzzy_floor: float = 0.35,
    ) -> tuple[MatchEvidence, ...]:
        if top_k < 2:
            raise ValueError("top_k must be at least 2 so ambiguity can be measured")

        normalized_query = normalize_text(request.utterance)
        profiles = self.active_personas(request.active_persona_ids)
        persona_entry_ids = {
            entry_id for profile in profiles for entry_id in profile.entry_ids
        }
        rule_matches = self._matching_rules(request, profiles)
        preferred_by_entry: dict[str, list[str]] = {}
        for rule in rule_matches:
            preferred_by_entry.setdefault(rule.preferred_entry_id, []).append(rule.rule_id)

        ranked: list[MatchEvidence] = []
        for entry in self._active_entries:
            method, base_score = self._score_entry(normalized_query, entry)
            is_persona_entry = entry.entry_id in persona_entry_ids
            rules = tuple(preferred_by_entry.get(entry.entry_id, ()))

            if rules:
                method = "persona_rule"
                base_score = max(base_score, 1.0)
            elif preferred_by_entry and base_score >= 0.90:
                # A validated contextual rule resolves this surface form in favor of
                # another sense. Preserve the alternative, but do not leave a false tie.
                method = "contextually_deprioritized"
                base_score = 0.70
            elif is_persona_entry and base_score > 0:
                base_score = min(1.0, base_score + 0.03)
                method = f"persona_{method}"

            if base_score >= fuzzy_floor:
                ranked.append(
                    MatchEvidence(
                        entry=entry,
                        score=base_score,
                        method=method,
                        persona_priority=is_persona_entry,
                        rule_ids=rules,
                    )
                )

        ranked.sort(
            key=lambda evidence: (
                evidence.score,
                bool(evidence.rule_ids),
                evidence.persona_priority,
                evidence.entry.priority,
            ),
            reverse=True,
        )
        return tuple(ranked[:top_k])

    def render_persona_context(self, profile_ids: Iterable[str]) -> str:
        profiles = self.active_personas(profile_ids)
        rendered: list[dict[str, Any]] = []
        for profile in profiles:
            entries = [
                asdict(self._by_id[entry_id])
                for entry_id in profile.entry_ids
                if self._by_id[entry_id].review_status == "approved"
            ]
            rendered.append(
                {
                    "profile_id": profile.profile_id,
                    "profile_version": profile.profile_version,
                    "profile_hash": profile.content_hash,
                    "dialect": profile.dialect,
                    "cultural_context": profile.cultural_context,
                    "entries": entries,
                    "pragmatic_rules": [asdict(rule) for rule in profile.pragmatic_rules],
                }
            )
        return canonical_json(rendered).decode("utf-8")

    @staticmethod
    def _entry_status(entries: Iterable[CodebookEntry], entry_id: str) -> str:
        return next(entry.review_status for entry in entries if entry.entry_id == entry_id)

    def is_polysemous_surface(self, utterance: str) -> bool:
        query = normalize_text(utterance)
        concepts: set[tuple[str, str]] = set()
        for entry in self._active_entries:
            if any(normalize_text(form) == query for form in entry.all_surface_forms):
                concepts.add((entry.concept_id, entry.universal_gloss))
        return len(concepts) > 1

    def _score_entry(self, query: str, entry: CodebookEntry) -> tuple[str, float]:
        best_method = "fuzzy"
        best_score = 0.0
        for form in entry.all_surface_forms:
            candidate = normalize_text(form)
            if candidate == query:
                return "exact", 1.0
            if candidate and re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", query):
                if 0.94 > best_score:
                    best_method, best_score = "literal_phrase", 0.94
            fuzzy_score = SequenceMatcher(None, query, candidate).ratio()
            if fuzzy_score > best_score:
                best_method, best_score = "fuzzy", fuzzy_score

        for pattern in entry.syntax_patterns:
            if re.search(pattern, query, flags=re.IGNORECASE):
                if 0.98 > best_score:
                    best_method, best_score = "reviewed_regex", 0.98
        return best_method, best_score

    def _matching_rules(
        self,
        request: InterpretationRequest,
        profiles: Iterable[PersonaProfile],
    ) -> tuple[PragmaticRule, ...]:
        query = normalize_text(request.utterance)
        supplied_context = normalize_text(
            " ".join(
                value
                for value in (
                    request.supplied_context or "",
                    request.supplied_tone or "",
                    request.speaker_role or "",
                    *request.conversation_context,
                )
                if value
            )
        )
        matches: list[PragmaticRule] = []
        for profile in profiles:
            for rule in profile.pragmatic_rules:
                trigger = normalize_text(rule.trigger)
                if not re.search(rf"(?<!\w){re.escape(trigger)}(?!\w)", query):
                    continue
                if rule.speaker_role and normalize_text(rule.speaker_role) != normalize_text(
                    request.speaker_role or ""
                ):
                    continue
                if rule.tone and normalize_text(rule.tone) not in supplied_context:
                    continue
                if rule.context_condition and normalize_text(rule.context_condition) not in supplied_context:
                    continue
                matches.append(rule)
        matches.sort(key=lambda rule: rule.priority, reverse=True)
        return tuple(matches)

    def _validate_patterns(self) -> None:
        unsupported_operator = re.compile(r"(?<!\\)[()*+{}|]")
        for entry in self.entries:
            for pattern in entry.syntax_patterns:
                if len(pattern) > 256:
                    raise ValueError(f"regex too long for {entry.entry_id}")
                if unsupported_operator.search(pattern) or re.search(r"\\[1-9]", pattern):
                    raise ValueError(
                        f"unsafe or unsupported regex operator for {entry.entry_id}; "
                        "use literal surface forms for complex patterns"
                    )
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"invalid regex for {entry.entry_id}: {exc}") from exc


def _parse_entry(raw: Any) -> CodebookEntry:
    if not isinstance(raw, Mapping):
        raise ValueError("every codebook entry must be an object")

    aliases = {
        "text": raw.get("text", raw.get("Utterance")),
        "dialect": raw.get("dialect", raw.get("Dialect")),
        "universal_gloss": raw.get(
            "universal_gloss", raw.get("Clarification", raw.get("gloss"))
        ),
        "intent": raw.get("intent", raw.get("Intent")),
    }
    return CodebookEntry(
        entry_id=str(raw.get("entry_id", "")).strip(),
        concept_id=str(raw.get("concept_id", "")).strip(),
        text=str(aliases["text"] or "").strip(),
        dialect=str(aliases["dialect"] or "").strip(),
        universal_gloss=str(aliases["universal_gloss"] or "").strip(),
        intent=str(aliases["intent"] or "").strip(),
        sociolinguistic_tags=require_string_sequence(
            raw.get("sociolinguistic_tags", []), "sociolinguistic_tags"
        ),
        tone_categories=require_string_sequence(
            raw.get("tone_categories", raw.get("Tone_Category", [])),
            "tone_categories",
        ),
        linguistic_contexts=require_string_sequence(
            raw.get("linguistic_contexts", raw.get("Linguistic_Context", [])),
            "linguistic_contexts",
        ),
        pragmatic_analysis=str(
            raw.get("pragmatic_analysis", raw.get("Pragmatic_Analysis", ""))
        ).strip(),
        surface_forms=require_string_sequence(raw.get("surface_forms", []), "surface_forms"),
        syntax_patterns=require_string_sequence(
            raw.get("syntax_patterns", raw.get("Syntax_Pattern", [])),
            "syntax_patterns",
        ),
        examples=require_string_sequence(raw.get("examples", []), "examples"),
        counterexamples=require_string_sequence(
            raw.get("counterexamples", []), "counterexamples"
        ),
        speaker_roles=require_string_sequence(raw.get("speaker_roles", []), "speaker_roles"),
        persona_ids=require_string_sequence(raw.get("persona_ids", []), "persona_ids"),
        priority=int(raw.get("priority", 0)),
        source_type=str(raw.get("source_type", "human")),
        source_reference=str(raw.get("source_reference", "")),
        reviewed_by=require_string_sequence(raw.get("reviewed_by", []), "reviewed_by"),
        review_status=str(raw.get("review_status", "pending")),
        version=int(raw.get("version", 1)),
        supersedes_entry_id=raw.get("supersedes_entry_id"),
        audio_uri=raw.get("audio_uri"),
        audio_sha256=raw.get("audio_sha256"),
        created_at=str(raw.get("created_at") or "").strip() or CodebookEntry.__dataclass_fields__["created_at"].default_factory(),
    )


def _parse_persona(raw: Any) -> PersonaProfile:
    if not isinstance(raw, Mapping):
        raise ValueError("every persona must be an object")
    raw_rules = raw.get("pragmatic_rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError("pragmatic_rules must be an array of objects")
    rules: list[PragmaticRule] = []
    for item in raw_rules:
        if not isinstance(item, Mapping):
            raise ValueError("pragmatic_rules must contain objects, not strings")
        rules.append(
            PragmaticRule(
                rule_id=str(item.get("rule_id", "")),
                trigger=str(item.get("trigger", "")),
                preferred_entry_id=str(item.get("preferred_entry_id", "")),
                interpretation=str(item.get("interpretation", "")),
                tone=str(item["tone"]) if item.get("tone") is not None else None,
                speaker_role=(
                    str(item["speaker_role"]) if item.get("speaker_role") is not None else None
                ),
                context_condition=(
                    str(item["context_condition"])
                    if item.get("context_condition") is not None
                    else None
                ),
                priority=int(item.get("priority", 0)),
            )
        )
    return PersonaProfile(
        profile_id=str(raw.get("profile_id", "")),
        profile_version=str(raw.get("profile_version", "")),
        display_name=str(raw.get("display_name", "")),
        dialect=str(raw.get("dialect", "")),
        cultural_context=str(raw.get("cultural_context", "")),
        entry_ids=require_string_sequence(raw.get("entry_ids", []), "entry_ids"),
        pragmatic_rules=tuple(rules),
        reviewed_by=require_string_sequence(raw.get("reviewed_by", []), "reviewed_by"),
        review_status=str(raw.get("review_status", "approved")),
    )
