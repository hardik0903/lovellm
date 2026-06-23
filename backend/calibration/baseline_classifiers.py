"""
baseline_classifiers.py
------------------------
Two alternative routing classifiers used as baselines/comparisons against
the regex/heuristic detector ensemble (MasterRouter + ConfidenceCalibrator):

1. TfidfLogRegClassifier
   A trained multiclass classifier: TF-IDF features (word + char n-grams)
   feeding a multinomial logistic regression over the 8 routing labels
   (math, code, data, document, writing, research, knowledge, none).
   Trained on calibration/labeled_queries.json.

2. LLMRoutingClassifier
   A single-call Groq classifier: gives the LLM the query (+ session
   context flags) and the list of valid agent names, asks for the single
   best route as JSON. This is the "ask an LLM to classify" baseline that
   the regex detectors are implicitly being compared against.

Both expose a uniform interface:

    predict(query: str, context: dict | None) -> str   # one of AGENT_NAMES + "none"

so that calibration/benchmark_detectors.py can run all three approaches
(regex ensemble, TF-IDF+LogReg, LLM) over the same held-out test split and
report accuracy / per-class F1 / latency in one table.
"""

import json
import os
import time
from typing import Dict, List, Optional

AGENT_NAMES = ["math", "code", "data", "document", "writing", "research", "knowledge", "none"]


# --------------------------------------------------------------------------- #
# 1. TF-IDF + Logistic Regression                                              #
# --------------------------------------------------------------------------- #

class TfidfLogRegClassifier:
    """
    Trained multiclass classifier over the 8 routing labels.

    Features: TF-IDF over (a) word 1-2 grams and (b) character 3-5 grams,
    concatenated. Character n-grams help with code snippets, stacktraces,
    and math notation (e.g. "x^2", "TypeError:", "\\frac{") that word-level
    n-grams alone would miss.

    Model: multinomial logistic regression (sklearn LogisticRegression,
    solver='lbfgs', class_weight='balanced' to handle the mild label
    imbalance in labeled_queries.json).
    """

    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfidf_logreg_model.joblib")

    def __init__(self):
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.model = None
        self.classes_ = None

    def fit(self, queries: List[str], labels: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from scipy.sparse import hstack

        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True
        )
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=(3, 5), min_df=1, sublinear_tf=True, analyzer="char_wb", lowercase=True
        )

        X_word = self.word_vectorizer.fit_transform(queries)
        X_char = self.char_vectorizer.fit_transform(queries)
        X = hstack([X_word, X_char])

        self.model = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=2.0
        )
        self.model.fit(X, labels)
        self.classes_ = self.model.classes_

    def _vectorize(self, queries: List[str]):
        from scipy.sparse import hstack
        X_word = self.word_vectorizer.transform(queries)
        X_char = self.char_vectorizer.transform(queries)
        return hstack([X_word, X_char])

    def predict(self, query: str, context: Optional[Dict] = None) -> str:
        X = self._vectorize([query])
        return self.model.predict(X)[0]

    def predict_proba(self, query: str, context: Optional[Dict] = None) -> Dict[str, float]:
        X = self._vectorize([query])
        probs = self.model.predict_proba(X)[0]
        return {cls: float(p) for cls, p in zip(self.classes_, probs)}

    def save(self, path: str = None):
        import joblib
        path = path or self.MODEL_PATH
        joblib.dump(
            {
                "word_vectorizer": self.word_vectorizer,
                "char_vectorizer": self.char_vectorizer,
                "model": self.model,
                "classes_": self.classes_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str = None) -> "TfidfLogRegClassifier":
        import joblib
        path = path or cls.MODEL_PATH
        data = joblib.load(path)
        clf = cls()
        clf.word_vectorizer = data["word_vectorizer"]
        clf.char_vectorizer = data["char_vectorizer"]
        clf.model = data["model"]
        clf.classes_ = data["classes_"]
        return clf


# --------------------------------------------------------------------------- #
# 2. LLM-based classifier (Groq)                                               #
# --------------------------------------------------------------------------- #

class LLMRoutingClassifier:
    """
    Asks an LLM (via Groq) to pick the single best agent for a query.

    This is the "LLM-as-classifier" baseline: instead of regex heuristics
    or a trained TF-IDF model, an LLM reads the query (plus session context
    flags such as whether documents/data files are present) and returns the
    name of the agent that should handle it, or "none" if no specialist
    agent applies.

    Requires GROQ_API_KEY to be set. Uses the same small/fast model
    (llama-3.1-8b-instant) the project already uses for lightweight
    classification tasks (see knowledge_classifier.py), so the latency
    comparison in the benchmark is apples-to-apples with the rest of the
    system rather than an oversized model.
    """

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        import os as _os
        from groq import AsyncGroq

        api_key = _os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    async def predict_async(self, query: str, context: Optional[Dict] = None) -> str:
        context = context or {}
        prompt = f"""You are a query router for a multi-agent assistant. Given a user query and
session context, choose the SINGLE best agent to handle it.

Agents:
- math: arithmetic, algebra, calculus, statistics, equations
- code: writing, debugging, explaining, or refactoring source code
- data: analyzing an uploaded tabular data file (csv/xlsx) - only valid if has_data_file is true
- document: answering questions about an uploaded document/PDF - only valid if has_documents is true
- writing: drafting, editing, proofreading, or rewriting text (emails, essays, posts, letters)
- research: questions needing up-to-date / current-events information
- knowledge: general conceptual or factual questions answerable from static knowledge
- none: greetings, chit-chat, or anything not matching the above

Session context: has_documents={context.get('has_documents', False)}, has_data_file={context.get('has_data_file', False)}

Query: "{query}"

Respond with ONLY a JSON object: {{"agent": "<one of the agent names above>"}}
No other text."""

        response = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
            agent = parsed.get("agent", "none")
        except (json.JSONDecodeError, AttributeError):
            agent = "none"

        if agent not in AGENT_NAMES:
            agent = "none"
        return agent

    def predict(self, query: str, context: Optional[Dict] = None) -> str:
        """Synchronous wrapper for use in scripts."""
        import asyncio
        return asyncio.run(self.predict_async(query, context))