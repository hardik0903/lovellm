"""
display_agent.py
================
Smart Display Formatting Agent for LoveLLM.

Responsibilities:
  1. AMBIGUITY RESOLUTION  — understands what the user *really* wants,
     including follow-up continuations like "Now with GCP" after a prior
     comparison of AWS vs Azure.
  2. FORMAT DETECTION      — maps intent to the richest display format.
  3. PROMPT INJECTION      — tells Groq exactly what JSON schema to return
     so we never parse free-form prose into tables.
  4. TRANSFORMATION        — converts the Groq-structured JSON into a
     frontend-ready display object.

Zero LLM calls. Zero network calls. <5 ms per request.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# SECTION 1 — CONSTANTS
# ---------------------------------------------------------------------------

# Priority-ordered format detection rules.
# Each entry: (format_name, [trigger_phrases])
FORMAT_RULES: List[Tuple[str, List[str]]] = [
    ("comparison", [
        "compare", "vs ", "versus", "difference between", "differentiate",
        "contrast", "which is better", "similarities and differences",
        "how do .* differ", "what.*difference.*between",
    ]),
    ("pros_cons", [
        "pros and cons", "advantages and disadvantages",
        "benefits and drawbacks", "upsides and downsides",
        "good and bad", "should i use", "is it worth",
        "tradeoffs", "trade-offs",
    ]),
    ("steps", [
        "how to ", "steps to", "walk me through", "guide to",
        "tutorial", "procedure for", "instructions for",
        "how do i install", "how do i set up", "how do i create",
    ]),
    ("timeline", [
        "history of", "timeline", "chronology", "when did",
        "sequence of events", "evolution of", "over the years",
        "milestones of",
    ]),
    ("troubleshoot", [
        "error", "not working", "broken", "debug",
        "fix ", "why is.*failing", "crash", "exception",
        "troubleshoot", "502", "404", "500 error",
    ]),
    ("recommend", [
        "recommend", "suggest", "best option", "what should i",
        "which should i choose", "help me decide", "which one",
        "which is best for", "what.*choose",
    ]),
    ("stats", [
        "statistics", "how many", "what percentage", "rate of",
        "population of", "how much does", "what.*number",
        "data on", "metrics",
    ]),
    ("summary", [
        "summarize", "tldr", "key points", "main takeaways",
        "in brief", "give me the gist", "overview of",
        "sum up",
    ]),
    ("definition", [
        "what is ", "what are ", "define ", "explain ",
        "meaning of", "what does.*mean",
    ]),
    ("list_format", [
        "list ", "enumerate", "name the", "give me all",
        "types of", "kinds of", "examples of", "what are some",
    ]),
]

# Phrases that signal "this is a follow-up to the previous comparison".
CONTINUATION_SIGNALS: List[str] = [
    "now with", "add ", "include ", "what about", "also include",
    "and ", "now add", "extend", "plus ", "throw in",
    "alongside", "compared to all", "all three", "all of them",
]

# Signals that the answer is too short to warrant structured display.
MIN_WORDS_FOR_TABLE = 30


# ---------------------------------------------------------------------------
# SECTION 2 — AMBIGUITY RESOLVER
# ---------------------------------------------------------------------------

class AmbiguityResolver:
    """
    Detects and resolves query ambiguities BEFORE format detection.

    Handles:
      - Follow-up continuations  ("Now with GCP" after AWS vs Azure)
      - Implicit comparison       ("which is better?" with no entities given)
      - Entity expansion          ("compare them" after mentioning X and Y)
      - Typos in trigger words    ("comapre" → "compare")
    """

    # Common typos for high-value trigger words
    TYPO_MAP: Dict[str, str] = {
        "comapre": "compare",
        "copmare": "compare",
        "comprae": "compare",
        "compar": "compare",
        "comare": "compare",
        "differntiate": "differentiate",
        "differenciate": "differentiate",
        "sumamrize": "summarize",
        "sumarize": "summarize",
        "reccomend": "recommend",
        "recomend": "recommend",
        "explian": "explain",
        "exaplain": "explain",
        "procs": "pros",
        "prons": "pros",
    }

    def resolve(
        self,
        raw_query: str,
        conversation_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Returns a resolved query context dict with fields:
          - resolved_query: str          cleaned, typo-corrected query
          - is_continuation: bool        True if this extends a prior turn
          - continuation_type: str|None  "add_entity" | "clarify" | "reformat"
          - prior_entities: list         entities from the prior comparison turn
          - new_entities: list           entities extracted from this turn
          - all_entities: list           union of prior + new entities
          - format_hint: str|None        forced format if continuation implies it
        """
        cleaned = self._fix_typos(raw_query.strip())
        cleaned_lower = cleaned.lower()

        prior_comparison = self._find_prior_comparison(conversation_history)
        is_continuation = self._detect_continuation(cleaned_lower, prior_comparison)

        prior_entities: List[str] = []
        new_entities: List[str] = []

        if is_continuation and prior_comparison:
            prior_entities = prior_comparison.get("entities", [])
            new_entities = self._extract_new_entities(cleaned, prior_entities)
            continuation_type = self._classify_continuation(cleaned_lower, new_entities)
        else:
            continuation_type = None
            new_entities = self._extract_entities_from_query(cleaned)

        all_entities = self._deduplicate(prior_entities + new_entities)

        # If it's a continuation that adds entities, rewrite the query
        # to be self-contained so the retriever and generator understand it.
        if is_continuation and continuation_type == "add_entity" and all_entities:
            resolved_query = self._rewrite_continuation(
                cleaned, all_entities, prior_comparison
            )
        else:
            resolved_query = cleaned

        format_hint = None
        if is_continuation and prior_comparison:
            format_hint = prior_comparison.get("display_format")

        return {
            "resolved_query": resolved_query,
            "original_query": raw_query,
            "is_continuation": is_continuation,
            "continuation_type": continuation_type,
            "prior_entities": prior_entities,
            "new_entities": new_entities,
            "all_entities": all_entities,
            "format_hint": format_hint,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fix_typos(self, query: str) -> str:
        words = query.split()
        fixed = []
        for word in words:
            lower = word.lower().rstrip(".,!?")
            if lower in self.TYPO_MAP:
                fixed.append(self.TYPO_MAP[lower])
            else:
                fixed.append(word)
        return " ".join(fixed)

    def _find_prior_comparison(
        self, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Walk history backwards to find the most recent assistant turn
        that had a comparison / structured display format.
        """
        for turn in reversed(history):
            if turn.get("role") != "assistant":
                continue
            display = turn.get("display", {})
            if not display:
                continue
            fmt = display.get("type", "")
            if fmt in ("comparison_table", "pros_cons_table", "recommend_card"):
                return {
                    "display_format": fmt.replace("_table", "").replace("_card", ""),
                    "entities": display.get("entities", []),
                    "turn": turn,
                }
        return None

    def _detect_continuation(
        self, query_lower: str, prior_comparison: Optional[Dict]
    ) -> bool:
        if not prior_comparison:
            return False
        # Check if the query contains a continuation signal
        for signal in CONTINUATION_SIGNALS:
            if signal in query_lower:
                return True
        # Very short queries (< 5 words) after a comparison are almost
        # always continuations ("and GCP?", "what about Azure?")
        if len(query_lower.split()) <= 5:
            return True
        return False

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract compared entities from a fresh comparison query.
        Handles: "compare X and Y", "X vs Y", "X versus Y and Z",
                 "difference between X, Y and Z"
        """
        q = query.strip()

        # Pattern: X vs Y (vs Z ...)
        vs_match = re.split(r'\s+vs\.?\s+|\s+versus\s+', q, flags=re.IGNORECASE)
        if len(vs_match) > 1:
            entities = []
            for part in vs_match:
                # each part may be "X and Z" from "X vs Y and Z"
                sub = re.split(r'\s+and\s+', part.strip(), flags=re.IGNORECASE)
                entities.extend([s.strip().rstrip("?.,!") for s in sub if s.strip()])
            return [e for e in entities if len(e) > 1]

        # Pattern: "compare/difference between X and Y (and Z)"
        for pattern in [
            r"(?:compare|contrast|differentiate)\s+(.+)",
            r"difference between\s+(.+)",
            r"similarities.*between\s+(.+)",
        ]:
            m = re.search(pattern, q, re.IGNORECASE)
            if m:
                rest = m.group(1)
                parts = re.split(r',\s*|\s+and\s+', rest, flags=re.IGNORECASE)
                return [p.strip().rstrip("?.,!") for p in parts if p.strip() and len(p.strip()) > 1]

        return []

    def _extract_new_entities(
        self, query: str, prior_entities: List[str]
    ) -> List[str]:
        """
        From a continuation query, extract ONLY the NEW entities
        not already in the prior comparison.
        """
        # Look for proper nouns / known service names
        # Simple heuristic: capitalized words or known tech names
        candidates = re.findall(r'\b[A-Z][a-zA-Z0-9+#]*\b', query)
        prior_lower = {e.lower() for e in prior_entities}

        new = []
        for c in candidates:
            if c.lower() not in prior_lower and len(c) > 1:
                # Filter out sentence-start capitals and common words
                if c not in {"Now", "What", "Can", "How", "Also", "Add",
                             "Include", "And", "With", "The", "I"}:
                    new.append(c)
        return list(dict.fromkeys(new))  # deduplicate preserving order

    def _classify_continuation(
        self, query_lower: str, new_entities: List[str]
    ) -> str:
        if new_entities:
            return "add_entity"
        if any(w in query_lower for w in ["reformat", "as a table", "in table"]):
            return "reformat"
        return "clarify"

    def _rewrite_continuation(
        self,
        query: str,
        all_entities: List[str],
        prior_comparison: Dict,
    ) -> str:
        """
        Rewrites a vague continuation into a fully self-contained query.
        "Now with GCP" + [AWS, Azure] → "Compare AWS, Azure, and GCP"
        """
        if len(all_entities) == 2:
            entity_str = f"{all_entities[0]} and {all_entities[1]}"
        else:
            entity_str = ", ".join(all_entities[:-1]) + f", and {all_entities[-1]}"

        fmt = prior_comparison.get("display_format", "comparison")
        if fmt in ("comparison", "pros_cons"):
            return f"Compare {entity_str} in detail"
        return f"Compare {entity_str}"

    def _deduplicate(self, items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


# ---------------------------------------------------------------------------
# SECTION 3 — FORMAT DETECTOR
# ---------------------------------------------------------------------------

class FormatDetector:
    """
    Maps a resolved query to the best display format.
    Returns a (format_name, confidence, entities) tuple.
    """

    def detect(
        self,
        resolved_query: str,
        ambiguity_context: Dict[str, Any],
        answer_text: str = "",
    ) -> Tuple[str, float, List[str]]:
        """
        Three-stage detection:
          1. Forced format from continuation context
          2. Keyword matching on the query
          3. Answer-level structural inference (fallback)
        """
        q_lower = resolved_query.lower()
        entities = ambiguity_context.get("all_entities", [])

        # Stage 1 — continuation forces the same format
        if ambiguity_context.get("format_hint"):
            return ambiguity_context["format_hint"], 1.0, entities

        # Stage 2 — keyword matching
        for fmt, phrases in FORMAT_RULES:
            for phrase in phrases:
                if re.search(phrase, q_lower):
                    # Extract entities if comparison and none found yet
                    if fmt == "comparison" and not entities:
                        entities = AmbiguityResolver()._extract_entities_from_query(
                            resolved_query
                        )
                    confidence = 1.0 if len(phrase) > 6 else 0.8
                    return fmt, confidence, entities

        # Stage 3 — answer-level inference
        if answer_text:
            fmt = self._infer_from_answer(answer_text)
            if fmt:
                return fmt, 0.5, entities

        return "prose", 0.0, entities

    def _infer_from_answer(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(p in t for p in ["on the other hand", "whereas", "in contrast", "while both"]):
            return "comparison"
        if any(p in t for p in ["advantage", "disadvantage", "benefit", "drawback"]):
            return "pros_cons"
        if re.search(r'\b(step \d|first[,.]|second[,.]|third[,.]|finally)', t):
            return "steps"
        if re.search(r'\b(19|20)\d{2}\b.*\b(19|20)\d{2}\b', t):
            return "timeline"
        if any(p in t for p in ["error", "fix", "resolve", "solution"]):
            return "troubleshoot"
        return None


# ---------------------------------------------------------------------------
# SECTION 4 — PROMPT SCHEMA INJECTOR
# ---------------------------------------------------------------------------

class PromptSchemaInjector:
    """
    Injects a structured JSON schema instruction into the Groq prompt
    BEFORE the LLM call, so we get structured data back — not prose.
    """

    SCHEMAS: Dict[str, str] = {

        "comparison": """
Since this is a COMPARISON query, your JSON response MUST include a "display" field structured as:
"display": {{
  "type": "comparison_table",
  "entities": {entities},
  "features": [
    {{
      "name": "Feature name (e.g. Performance, Cost, Learning Curve)",
      {entity_cols}
    }}
  ],
  "verdict": "One sentence stating which is best and for what use case, or 'It depends on...' if contextual"
}}
Include 6-10 meaningful features. Each feature value should be 1-2 sentences max, not just a word.
""",

        "pros_cons": """
Since this is a PROS/CONS query, your JSON response MUST include a "display" field:
"display": {{
  "type": "pros_cons_table",
  "subject": "The thing being evaluated",
  "pros": ["Pro 1", "Pro 2", "Pro 3"],
  "cons": ["Con 1", "Con 2", "Con 3"],
  "neutral": ["Neutral point if any"],
  "verdict": "One-line recommendation"
}}
""",

        "steps": """
Since this is a HOW-TO query, your JSON response MUST include a "display" field:
"display": {{
  "type": "step_list",
  "goal": "What this guide achieves",
  "steps": [
    {{
      "number": 1,
      "title": "Short step title",
      "detail": "Full explanation of this step",
      "warning": "Optional — caution for this step, or null"
    }}
  ],
  "tip": "Optional overall tip or null"
}}
""",

        "timeline": """
Since this is a TIMELINE/HISTORY query, your JSON response MUST include a "display" field:
"display": {{
  "type": "timeline_table",
  "subject": "Topic of the timeline",
  "events": [
    {{
      "period": "Year or era (e.g. 1969, Early 1990s)",
      "event": "What happened",
      "significance": "Why it mattered (1 sentence)"
    }}
  ]
}}
Sort events chronologically oldest to newest.
""",

        "troubleshoot": """
Since this is a TROUBLESHOOTING query, your JSON response MUST include a "display" field:
"display": {{
  "type": "troubleshoot_table",
  "problem": "The stated problem",
  "causes": [
    {{
      "symptom": "What the user sees",
      "cause": "Root cause",
      "fix": "Exact fix steps",
      "code_fix": "Optional code snippet or null"
    }}
  ]
}}
""",

        "recommend": """
Since this is a RECOMMENDATION query, your JSON response MUST include a "display" field:
"display": {{
  "type": "recommend_card",
  "winner": "The recommended option",
  "reason": "One sentence why",
  "options": [
    {{
      "name": "Option name",
      "best_for": "Ideal use case",
      "avoid_if": "When NOT to use it",
      "score": 8
    }}
  ],
  "decision_factors": ["Factor 1", "Factor 2"]
}}
""",

        "stats": """
Since this is a STATISTICS/DATA query, your JSON response MUST include a "display" field:
"display": {{
  "type": "stats_table",
  "metrics": [
    {{
      "metric": "Metric name",
      "value": "The value with units",
      "context": "What this means",
      "source": "Source name"
    }}
  ]
}}
""",

        "summary": """
Since this is a SUMMARY query, your JSON response MUST include a "display" field:
"display": {{
  "type": "summary_block",
  "tldr": "One sentence summary",
  "key_points": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
  "conclusion": "One sentence conclusion"
}}
""",

        "definition": """
Since this is a DEFINITION query, your JSON response MUST include a "display" field:
"display": {{
  "type": "definition_block",
  "term": "The term being defined",
  "one_liner": "Single sentence definition",
  "analogy": "A plain-English analogy if helpful, or null",
  "details": "Longer explanation"
}}
""",

        "list_format": """
Since this is a LIST query, your JSON response MUST include a "display" field:
"display": {{
  "type": "list_block",
  "title": "What is being listed",
  "items": [
    {{
      "name": "Item name",
      "description": "Brief description"
    }}
  ]
}}
""",
    }

    def build_injection(
        self,
        fmt: str,
        entities: List[str],
    ) -> str:
        """
        Returns the schema injection string to append to the Groq prompt.
        Returns empty string for 'prose' format.
        """
        if fmt == "prose" or fmt not in self.SCHEMAS:
            return ""

        schema = self.SCHEMAS[fmt]

        if fmt == "comparison" and entities:
            entity_list = str(entities)
            entity_cols = ", ".join(
                f'"{e}": "Value for {e}"' for e in entities
            )
            schema = schema.format(entities=entity_list, entity_cols=entity_cols)
        else:
            # Remove any remaining format placeholders
            schema = re.sub(r'\{[^}]+\}', '...', schema)

        return schema.strip()


# ---------------------------------------------------------------------------
# SECTION 5 — DISPLAY TRANSFORMER
# ---------------------------------------------------------------------------

class DisplayTransformer:
    """
    Takes the Groq answer object (which should now contain a 'display' field
    from the structured prompt) and validates / normalises it.

    If Groq failed to return a structured display (it returned prose),
    we attempt a best-effort extraction. If that also fails, we gracefully
    fall back to prose — we NEVER crash.
    """

    def transform(
        self,
        answer_obj: Dict[str, Any],
        fmt: str,
        entities: List[str],
        confidence: float,
    ) -> Dict[str, Any]:
        """
        Returns the answer_obj enriched with a normalised 'display' field.
        """
        # Too short — never force structured display
        answer_text = answer_obj.get("answer", "")
        if len(answer_text.split()) < MIN_WORDS_FOR_TABLE and fmt != "definition":
            answer_obj["display"] = {"type": "prose", "content": answer_text}
            return answer_obj

        # If Groq already returned a display field, validate it
        existing_display = answer_obj.get("display")
        if existing_display and isinstance(existing_display, dict):
            validated = self._validate(existing_display, fmt, entities)
            if validated:
                answer_obj["display"] = validated
                return answer_obj

        # Groq didn't return structured display — attempt prose extraction
        extracted = self._extract_from_prose(answer_text, fmt, entities)
        answer_obj["display"] = extracted
        return answer_obj

    # ------------------------------------------------------------------
    # Validators — ensure required fields exist
    # ------------------------------------------------------------------

    def _validate(
        self, display: Dict, fmt: str, entities: List[str]
    ) -> Optional[Dict]:
        d_type = display.get("type", "")

        if d_type == "comparison_table":
            if not display.get("features") or not display.get("entities"):
                return None
            # Back-fill entities if missing
            if not display.get("entities") and entities:
                display["entities"] = entities
            return display

        if d_type == "pros_cons_table":
            if not display.get("pros") and not display.get("cons"):
                return None
            display.setdefault("pros", [])
            display.setdefault("cons", [])
            return display

        if d_type == "step_list":
            if not display.get("steps"):
                return None
            return display

        if d_type in (
            "timeline_table", "troubleshoot_table", "recommend_card",
            "stats_table", "summary_block", "definition_block", "list_block"
        ):
            return display if display else None

        return None

    # ------------------------------------------------------------------
    # Prose extraction fallback
    # ------------------------------------------------------------------

    def _extract_from_prose(
        self, text: str, fmt: str, entities: List[str]
    ) -> Dict[str, Any]:
        """
        Best-effort extraction when Groq returned prose instead of JSON.
        Returns a display dict or falls back to prose.
        """
        if fmt == "comparison" and entities:
            return self._prose_to_comparison(text, entities)
        if fmt == "pros_cons":
            return self._prose_to_pros_cons(text)
        if fmt == "steps":
            return self._prose_to_steps(text)
        if fmt == "summary":
            return self._prose_to_summary(text)

        # All other formats — return prose
        return {"type": "prose", "content": text}

    def _prose_to_comparison(self, text: str, entities: List[str]) -> Dict:
        """
        Heuristic: split text into sentences, assign each to an entity
        if the entity name appears in that sentence.
        Build feature rows from co-occurring sentences.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        feature_map: Dict[str, Dict[str, str]] = {}

        for sent in sentences:
            matched = [e for e in entities if e.lower() in sent.lower()]
            if len(matched) >= 2:
                # This sentence compares at least 2 entities directly
                key = sent[:40].rstrip(".,")
                feature_map[key] = {}
                for e in entities:
                    if e.lower() in sent.lower():
                        feature_map[key][e] = sent.strip()
                    else:
                        feature_map[key][e] = "—"

        if feature_map:
            features = [
                {"name": k, **v} for k, v in list(feature_map.items())[:8]
            ]
            return {
                "type": "comparison_table",
                "entities": entities,
                "features": features,
                "verdict": None,
                "_extracted_from_prose": True,
            }

        # Could not build a meaningful table
        return {"type": "prose", "content": text}

    def _prose_to_pros_cons(self, text: str) -> Dict:
        pros, cons = [], []
        for sent in re.split(r'(?<=[.!?])\s+', text):
            sl = sent.lower()
            if any(w in sl for w in ["advantage", "benefit", "pro ", "positive", "good"]):
                pros.append(sent.strip())
            elif any(w in sl for w in ["disadvantage", "drawback", "con ", "negative", "bad", "however"]):
                cons.append(sent.strip())

        if pros or cons:
            return {
                "type": "pros_cons_table",
                "subject": "Subject",
                "pros": pros[:6],
                "cons": cons[:6],
                "neutral": [],
                "verdict": None,
                "_extracted_from_prose": True,
            }
        return {"type": "prose", "content": text}

    def _prose_to_steps(self, text: str) -> Dict:
        # Look for numbered sentences or "First/Second/Finally" patterns
        steps = []
        for i, match in enumerate(
            re.finditer(
                r'(?:^|\n)\s*(?:\d+[\.\)]\s*|(?:first|second|third|fourth|fifth|finally)[,\s])',
                text, re.IGNORECASE | re.MULTILINE
            )
        ):
            start = match.start()
            end_match = re.search(r'(?:\n|$)', text[start + 1:])
            end = start + 1 + (end_match.start() if end_match else len(text))
            step_text = text[start:end].strip().lstrip("0123456789.) ")
            if step_text:
                steps.append({
                    "number": i + 1,
                    "title": step_text[:50],
                    "detail": step_text,
                    "warning": None,
                })

        if steps:
            return {
                "type": "step_list",
                "goal": "Complete the task",
                "steps": steps[:10],
                "tip": None,
                "_extracted_from_prose": True,
            }
        return {"type": "prose", "content": text}

    def _prose_to_summary(self, text: str) -> Dict:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return {
            "type": "summary_block",
            "tldr": sentences[0] if sentences else text[:100],
            "key_points": [s.strip() for s in sentences[1:6] if s.strip()],
            "conclusion": sentences[-1] if len(sentences) > 1 else "",
            "_extracted_from_prose": True,
        }


# ---------------------------------------------------------------------------
# SECTION 6 — MAIN AGENT
# ---------------------------------------------------------------------------

class DisplayFormattingAgent:
    """
    The single entry point. Wire this into pipeline.py.

    Usage:
        # In PipelineOrchestrator.__init__:
        self.display_agent = DisplayFormattingAgent()

        # After answer_verifier.verify(), before final yield:
        if mode != "direct_web":
            verified_obj = self.display_agent.process(
                answer_obj=verified_obj,
                display_context=display_context,
            )

        # In generator.py _build_prompt(), append:
        display_injection = self.display_agent.get_prompt_injection(display_context)
        prompt += "\\n\\n" + display_injection
    """

    def __init__(self):
        self.ambiguity_resolver = AmbiguityResolver()
        self.format_detector = FormatDetector()
        self.prompt_injector = PromptSchemaInjector()
        self.transformer = DisplayTransformer()

    # ------------------------------------------------------------------
    # Step A — called BEFORE the Groq call (in pipeline.py)
    # ------------------------------------------------------------------

    def resolve_and_detect(
        self,
        raw_query: str,
        conversation_history: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Resolves ambiguity and detects format.
        Call this BEFORE the Groq generation call.
        Returns a context dict to be passed to get_prompt_injection()
        and later to process().
        """
        if conversation_history is None:
            conversation_history = []

        ambiguity_ctx = self.ambiguity_resolver.resolve(
            raw_query, conversation_history
        )
        fmt, confidence, entities = self.format_detector.detect(
            ambiguity_ctx["resolved_query"],
            ambiguity_ctx,
        )

        return {
            "ambiguity": ambiguity_ctx,
            "format": fmt,
            "confidence": confidence,
            "entities": entities,
            "resolved_query": ambiguity_ctx["resolved_query"],
        }

    # ------------------------------------------------------------------
    # Step B — called to get the prompt injection string
    # ------------------------------------------------------------------

    def get_prompt_injection(self, display_context: Dict[str, Any]) -> str:
        """
        Returns the schema instruction to append to the Groq prompt.
        Pass the result of resolve_and_detect() here.
        """
        fmt = display_context.get("format", "prose")
        entities = display_context.get("entities", [])
        confidence = display_context.get("confidence", 0.0)

        # Only inject if we're confident enough
        if confidence < 0.5:
            return ""

        return self.prompt_injector.build_injection(fmt, entities)

    # ------------------------------------------------------------------
    # Step C — called AFTER the Groq call (in pipeline.py)
    # ------------------------------------------------------------------

    def process(
        self,
        answer_obj: Dict[str, Any],
        display_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transforms the answer object after Groq generation.
        Attaches the display field. Never crashes.
        Call this AFTER answer_verifier.verify().
        """
        fmt = display_context.get("format", "prose")
        entities = display_context.get("entities", [])
        confidence = display_context.get("confidence", 0.0)

        # Skip display formatting for direct prose cases
        if fmt == "prose" or confidence < 0.5:
            answer_obj.setdefault("display", {
                "type": "prose",
                "content": answer_obj.get("answer", "")
            })
            return answer_obj

        # Attach resolved query info for the frontend
        answer_obj["resolved_query"] = display_context.get("resolved_query")
        answer_obj["is_continuation"] = display_context["ambiguity"].get("is_continuation", False)

        try:
            return self.transformer.transform(answer_obj, fmt, entities, confidence)
        except Exception as e:
            # Never let the display agent crash the pipeline
            from logger import logger
            logger.error(f"[DisplayAgent] transform failed: {e}")
            answer_obj.setdefault("display", {
                "type": "prose",
                "content": answer_obj.get("answer", "")
            })
            return answer_obj
