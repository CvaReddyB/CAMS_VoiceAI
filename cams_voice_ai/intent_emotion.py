"""Intent + emotion from one encoder pass (all-MiniLM-L6-v2) + prototype similarity."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

INTENT_PROTOTYPES: Dict[str, List[str]] = {
    "redemption_request": [
        "I want to redeem from my folio",
        "Withdraw money from my mutual fund",
        "Redemption payout please",
        "Sell units and transfer to bank",
    ],
    "account_statement": [
        "Send my account statement",
        "Email transaction history",
        "CAS statement for investments",
        "Statement to registered email",
    ],
    "compliance_query": [
        "KYC pending compliance",
        "FATCA blocked redemption",
        "AML hold regulatory",
        "PAN mismatch with KRA",
    ],
    "human_agent_request": [
        "I want to speak to a human",
        "Connect me to an agent",
        "Transfer to representative",
        "Talk to customer care executive",
    ],
    "session_end": [
        "Disconnect the call",
        "Hang up now",
        "Goodbye end call",
        "That is all thank you bye",
    ],
    "general_support": [
        "Hello how can you help",
        "What services do you offer",
        "Good morning",
    ],
}

EMOTION_PROTOTYPES: Dict[str, List[str]] = {
    "calm_positive": [
        "Thank you so much I appreciate it",
        "That sounds great I am happy",
        "Everything is fine no issues",
    ],
    "neutral": [
        "I need information about my account",
        "Can you help me with this request",
        "What is the status",
    ],
    "frustrated": [
        "This is ridiculous I am very angry",
        "I am fed up nothing works",
        "Terrible service I am upset",
    ],
    "anxious": [
        "I am worried about my money",
        "I am concerned something is wrong",
        "Nervous about this hold on account",
    ],
}


@dataclass(frozen=True)
class Signals:
    intent: str
    intent_confidence: float
    emotion: str
    emotion_confidence: float


class IntentEmotionEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._intent_mat: Optional[np.ndarray] = None
        self._intent_labels: Optional[List[str]] = None
        self._emotion_mat: Optional[np.ndarray] = None
        self._emotion_labels: Optional[List[str]] = None

    def warm(self) -> None:
        """Load SentenceTransformer and prototype embeddings (call once at startup)."""
        self._ensure()

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)

        it_texts: List[str] = []
        it_labels: List[str] = []
        for intent, phrases in INTENT_PROTOTYPES.items():
            for p in phrases:
                it_labels.append(intent)
                it_texts.append(p)
        emb_i = self._model.encode(
            it_texts, convert_to_numpy=True, normalize_embeddings=True
        )
        self._intent_mat = emb_i.astype(np.float32)
        self._intent_labels = it_labels

        em_texts: List[str] = []
        em_labels: List[str] = []
        for emo, phrases in EMOTION_PROTOTYPES.items():
            for p in phrases:
                em_labels.append(emo)
                em_texts.append(p)
        emb_e = self._model.encode(
            em_texts, convert_to_numpy=True, normalize_embeddings=True
        )
        self._emotion_mat = emb_e.astype(np.float32)
        self._emotion_labels = em_labels

    def classify(self, user_text: str, *, context: str = "") -> Signals:
        self._ensure()
        assert self._model is not None
        q = (user_text or "").strip()
        if not q:
            return Signals("general_support", 0.5, "neutral", 0.5)
        query = f"{context.strip()}\n{q}" if context.strip() else q
        qv = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[
            0
        ].astype(np.float32)

        intent, ic = self._best_label(qv, self._intent_mat, self._intent_labels)
        emotion, ec = self._best_label(qv, self._emotion_mat, self._emotion_labels)

        intent = _keyword_overrides(q, intent)
        if _disconnect_words(q):
            return Signals("session_end", 0.97, emotion, ec)

        return Signals(intent, ic, emotion, ec)

    @staticmethod
    def _best_label(
        qv: np.ndarray,
        mat: Optional[np.ndarray],
        labels: Optional[List[str]],
    ) -> Tuple[str, float]:
        assert mat is not None and labels is not None
        sims = mat @ qv
        order = np.argsort(sims)
        best_i = int(order[-1])
        best = float(sims[best_i])
        second = float(sims[order[-2]]) if len(sims) > 1 else 0.0
        margin = best - second
        conf = float(np.clip(0.52 + 0.4 * best + 0.15 * margin, 0.52, 0.95))
        return labels[best_i], conf


def _disconnect_words(text: str) -> bool:
    t = (text or "").lower()
    keys = (
        "disconnect",
        "hang up",
        "end the call",
        "end this call",
        "cut the call",
        "terminate the call",
    )
    return any(k in t for k in keys)


def _keyword_overrides(text: str, intent: str) -> str:
    t = (text or "").lower()
    if any(
        k in t
        for k in (
            "human",
            "agent",
            "representative",
            "executive",
            "person",
            "real person",
        )
    ):
        if "session_end" not in t:
            return "human_agent_request"
    return intent


def extract_last_four_mobile(transcript: str) -> Optional[str]:
    digits = "".join(re.findall(r"\d", transcript or ""))
    if len(digits) >= 4:
        return digits[-4:]
    m = re.search(r"\b(\d{4})\b", transcript or "")
    return m.group(1) if m else None


def extract_pan_last_four(transcript: str) -> Optional[str]:
    t = (transcript or "").upper().replace(" ", "")
    alnum = "".join(c for c in t if c.isalnum())
    if len(alnum) >= 4:
        tail = alnum[-4:]
        if tail.isdigit() or re.match(r"^[A-Z0-9]{4}$", tail):
            return tail
    return None


def normalize_dob_iso(transcript: str) -> Optional[str]:
    """Return YYYY-MM-DD if a clear date is found in transcript."""
    t = transcript or ""
    m = re.search(
        r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})",
        t,
    )
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return _iso(y, mo, d)
    m = re.search(
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})",
        t,
    )
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 1900 if y > 30 else 2000
        return _iso(y, mo, d)
    digits = "".join(re.findall(r"\d", t))
    if len(digits) == 8:
        d, mo, y = int(digits[:2]), int(digits[2:4]), int(digits[4:])
        if y < 100:
            y += 1900 if y > 30 else 2000
        return _iso(y, mo, d)
    spoken = _spoken_english_date_iso(t)
    if spoken:
        return spoken
    return None


_MONTH_WORDS: Dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "nov": 11,
    "dec": 12,
}

_DAY_WORDS: Dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "twenty-first": 21,
    "twenty-second": 22,
    "twenty-third": 23,
    "twenty-fourth": 24,
    "twenty-fifth": 25,
    "twenty-sixth": 26,
    "twenty-seventh": 27,
    "twenty-eighth": 28,
    "twenty-ninth": 29,
    "thirtieth": 30,
    "thirty-first": 31,
    "twenty first": 21,
    "twenty second": 22,
    "twenty third": 23,
    "twenty fourth": 24,
    "twenty fifth": 25,
    "twenty sixth": 26,
    "twenty seventh": 27,
    "twenty eighth": 28,
    "twenty ninth": 29,
}

_YEAR_ONES = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

_YEAR_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_YEAR_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _dob_tokens(text: str) -> List[str]:
    """Lowercase word tokens; normalize it's -> its so day words stay aligned."""
    t = (text or "").lower().replace("\u2019", "'")
    t = re.sub(r"\bit's\b", "its", t)
    return re.findall(r"[a-z0-9]+", t)


def _day_from_tokens(toks: List[str]) -> Optional[int]:
    if not toks:
        return None
    joined = " ".join(toks)
    if joined in _DAY_WORDS:
        return _DAY_WORDS[joined]
    if len(toks) == 1:
        w = toks[0]
        if w in _DAY_WORDS:
            return _DAY_WORDS[w]
        if w.isdigit():
            v = int(w)
            if 1 <= v <= 31:
                return v
        return None
    if len(toks) == 2 and toks[0] == "twenty" and toks[1] in _YEAR_ONES:
        v = 20 + _YEAR_ONES[toks[1]]
        if 1 <= v <= 31:
            return v
    if len(toks) == 2 and toks[0] == "thirty" and toks[1] == "first":
        return 31
    jh = f"{toks[0]}-{toks[1]}"
    if jh in _DAY_WORDS:
        return _DAY_WORDS[jh]
    return None


def _parse_year_sub00_99(toks: List[str]) -> Optional[int]:
    """0–99 for years like 1994 from 'ninety four' or 14 from 'fourteen'."""
    if not toks:
        return 0
    joined = " ".join(toks)
    if joined in _YEAR_TEENS:
        return _YEAR_TEENS[joined]
    if len(toks) == 2 and toks[0] in _YEAR_TENS and toks[1] in _YEAR_ONES:
        return _YEAR_TENS[toks[0]] + _YEAR_ONES[toks[1]]
    if len(toks) == 1 and toks[0] in _YEAR_TENS:
        return _YEAR_TENS[toks[0]]
    return None


def _year_from_word_span(toks: List[str]) -> Optional[int]:
    if not toks:
        return None
    s = " ".join(toks)
    m = re.search(r"\b(19\d{2}|20\d{2})\b", s)
    if m:
        return int(m.group(1))
    for i, w in enumerate(toks):
        if w == "nineteen" and i + 1 < len(toks) and toks[i + 1] == "hundred":
            sub = _parse_year_sub00_99(toks[i + 2 :])
            if sub is not None:
                return 1900 + sub
            return 1900
        if w == "nineteen":
            sub = _parse_year_sub00_99(toks[i + 1 :])
            if sub is not None:
                return 1900 + sub
        if w == "two" and i + 1 < len(toks) and toks[i + 1] == "thousand":
            rest = toks[i + 2 :]
            if rest and rest[0] == "and":
                rest = rest[1:]
            if not rest:
                return 2000
            sub = _parse_year_sub00_99(rest)
            if sub is not None:
                return 2000 + sub
    return None


def _skip_article(tokens: List[str], j: int, *, forward: bool) -> int:
    skip = {"the", "a", "my", "uh", "um"}
    if forward:
        while j < len(tokens) and tokens[j] in skip:
            j += 1
        return j
    while j >= 0 and tokens[j] in skip:
        j -= 1
    return j


def _day_before_month(tokens: List[str], month_i: int) -> Optional[int]:
    j = month_i - 1
    while j >= 0 and tokens[j] in {"the", "a", "my", "uh", "um"}:
        j -= 1
    if j < 0:
        return None
    if j >= 1 and tokens[j] == "of":
        return _day_from_tokens([tokens[j - 1]])
    if j >= 1 and tokens[j - 1] == "twenty":
        return _day_from_tokens([tokens[j - 1], tokens[j]])
    if j >= 1 and tokens[j - 1] == "thirty":
        return _day_from_tokens([tokens[j - 1], tokens[j]])
    return _day_from_tokens([tokens[j]])


def _spoken_english_date_iso(text: str) -> Optional[str]:
    """
    Parse ASR phrases like 'first August nineteen ninety four' or
    'August first nineteen ninety four' into YYYY-MM-DD.
    """
    tokens = _dob_tokens(text)
    for i, w in enumerate(tokens):
        if w not in _MONTH_WORDS:
            continue
        mo = _MONTH_WORDS[w]
        tail = tokens[i + 1 :]
        y_full = _year_from_word_span(tail)
        if y_full is not None:
            d = _day_before_month(tokens, i)
            if d is not None:
                got = _iso(y_full, mo, d)
                if got:
                    return got
        j = _skip_article(tokens, i + 1, forward=True)
        for n_take in (2, 1):
            if j + n_take > len(tokens):
                continue
            d = _day_from_tokens(tokens[j : j + n_take])
            if d is None:
                continue
            y = _year_from_word_span(tokens[j + n_take :])
            if y is not None:
                got = _iso(y, mo, d)
                if got:
                    return got
    return None


def _iso(y: int, mo: int, d: int) -> Optional[str]:
    if not (1 <= mo <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100):
        return None
    try:
        date(y, mo, d)
    except ValueError:
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def dob_matches_transcript(transcript: str, expected_iso: str) -> bool:
    """Loose match for ASR noise: ISO equality, or YYYYMMDD digit sequence in transcript."""
    got = normalize_dob_iso(transcript)
    exp = (expected_iso or "").strip()
    if got and exp and got == exp:
        return True
    digits = re.sub(r"\D", "", transcript or "")
    compact = exp.replace("-", "")
    if len(compact) == 8 and compact in digits:
        return True
    return False
