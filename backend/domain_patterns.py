"""
FIX (#1): single source of truth for the legal/polarity and technical-domain
detection regexes that used to be duplicated byte-for-byte across
query_understanding.py (used at query-planning time) and retriever.py (used
at retrieval-routing time). The two copies had already drifted in variable
naming and were only kept content-identical by manual discipline -- there
was nothing preventing someone from patching one copy (as already happened
once, per the rfc_q1 fix referenced in retriever.py's comments) without
updating the other, which would make query-planning and retrieval routing
silently disagree about the same query's classification.

This does not fix the deeper "regex/keyword heuristics are inherently
incomplete" risk (a synonym not in any list still falls through to the
default path with no fallback signal) -- that is a more fundamental
architecture decision (heuristic vs. learned classifier) outside the scope
of a contained bug fix. What this DOES fix is the easy, concrete part: one
list to extend instead of two, and a documented place new domain vocabulary
(medical, financial, etc.) should be added when this system is pointed at a
new document domain.
"""
import re


# Unambiguous constitutional/legal-domain terms. Deliberately excludes bare
# "section" / "power to" / "court" / "treaty" / "statute" / "regulation" /
# "ordinance", which also occur routinely in technical specs (RFCs use
# "Section 3.2", API docs discuss "regulation" of traffic, etc). Including
# those unconditionally previously forced every RFC/technical-spec question
# that happened to mention "section" out of the fast simple-query path and
# into full multi-query expansion -- diluting precision on exact-term
# technical lookups (e.g. rfc_q1 dropped from 0.8 to 0.2 correctness).
LEGAL_OR_POLARITY_RE = re.compile(
    r"\b(article|amendment|clause|constitution|congress|senate|house of representatives|"
    r"jurisdiction|shall not|"
    r"branch of government|ratification|ratify|ratified|apportion|succession|"
    r"voting age|inferior federal courts)\b",
    re.IGNORECASE,
)

# Ambiguous terms that are legal signals ONLY when there's no competing
# technical-spec signal in the same query (see is_legal_or_polarity below).
LEGAL_AMBIGUOUS_RE = re.compile(
    r"\b(section|court|statute|regulation|ordinance|power to|treaty)\b",
    re.IGNORECASE,
)

TECHNICAL_DOMAIN_RE = re.compile(
    r"\b(rfc\s?\d*|http/?\d|protocol|header field|status code|specification|"
    r"syntax|grammar|implementation|client|server|request|response|"
    r"endpoint|api|payload|encoding)\b",
    re.IGNORECASE,
)


def is_legal_or_polarity(query_lower: str) -> bool:
    """Combines the unambiguous legal pattern with a guarded check on
    ambiguous terms (section/court/treaty/etc) that only count as a legal
    signal when there's no technical-spec vocabulary in the same query.

    `query_lower` should already be lowercased by the caller (both existing
    call sites did this before the refactor; kept as a caller responsibility
    rather than re-lowering here to avoid a behavior change in either file).
    """
    if LEGAL_OR_POLARITY_RE.search(query_lower):
        return True
    if LEGAL_AMBIGUOUS_RE.search(query_lower):
        return not TECHNICAL_DOMAIN_RE.search(query_lower)
    return False


NEGATION_RE = re.compile(
    r"\b(not|n't|never|no|none|without|except|excluding|isn't|doesn't|don't|didn't|cannot|can't|won't|wouldn't|shouldn't)\b",
    re.IGNORECASE,
)

# FIX (#1): the keyword list above only catches negation expressed through a
# closed set of explicit negation words. A query like "Congress lacks the
# authority to declare war" or "the President is barred from initiating
# legislation" expresses the same negative-polarity meaning through a verb
# choice instead of an explicit negation word, and previously fell through
# to the default (non-multi_hop) path with no signal that it should have
# been caught. This does not require enumerating every possible synonym --
# it targets the much smaller, more stable set of verb/preposition
# *patterns* English uses to express "X does not have the power/ability to
# do Y" without the word "not" itself. Still incomplete (no regex-based
# approach can be exhaustive), but meaningfully narrows the gap described
# above without the maintenance burden of a per-domain keyword list.
IMPLICIT_NEGATIVE_POLARITY_RE = re.compile(
    r"\b(lacks?|lacking|excludes?|excluded|"
    r"(?:is|are|was|were)\s+(?:barred|prohibited|restricted|excluded|exempt(?:ed)?|"
    r"forbidden|precluded|disqualified)\s+from\b|"
    r"(?:has|have|had)\s+no\s+(?:power|authority|right|ability)\s+to\b)",
    re.IGNORECASE,
)


def has_negation_signal(query_lower: str) -> bool:
    """True if the query expresses negation either explicitly (NEGATION_RE)
    or through an implicit negative-polarity verb pattern
    (IMPLICIT_NEGATIVE_POLARITY_RE). Callers that previously checked
    NEGATION_RE alone should prefer this for the broader coverage; NEGATION_RE
    remains exported separately since some callers want explicit-only
    matching (e.g. the generator's negation-prompt trigger, which is meant
    to catch the much narrower "answer must restate polarity" case).
    """
    return bool(NEGATION_RE.search(query_lower) or IMPLICIT_NEGATIVE_POLARITY_RE.search(query_lower))
