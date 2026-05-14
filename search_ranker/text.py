"""Text normalization utilities shared by retrieval components."""

from __future__ import annotations

import re
from typing import List

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase text and return deterministic alphanumeric tokens."""
    return TOKEN_PATTERN.findall(text.lower())

