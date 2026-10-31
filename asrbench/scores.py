"""Punteggi per le prove: normalizzazione italiana, WER, CER. Solo stdlib."""
from __future__ import annotations

import re
import unicodedata


def norm_it(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.replace("'", " ").replace("’", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _edit(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[len(b)]


def wer(ref: str, hyp: str) -> float | None:
    r, h = norm_it(ref).split(), norm_it(hyp).split()
    if not r:
        return None
    return _edit(r, h) / len(r)


def cer(ref: str, hyp: str) -> float | None:
    r, h = norm_it(ref).replace(" ", ""), norm_it(hyp).replace(" ", "")
    if not r:
        return None
    return _edit(list(r), list(h)) / len(r)


def fmt(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "—"
