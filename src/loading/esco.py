"""ESCO loaders: occupations and skills.

occupations_en.csv and skills_en.csv have the same shape -- conceptUri +
preferredLabel + altLabels/hiddenLabels (newline-packed alternate surface
forms) + a couple of type-specific columns -- so both loaders share one reader.
"""

import os
import re

import pandas as pd

from src.paths import ESCO_DIR

DEFAULT_OCCUPATIONS_PATH = os.path.join(ESCO_DIR, "occupations_en.csv")
DEFAULT_SKILLS_PATH = os.path.join(ESCO_DIR, "skills_en.csv")

_LABEL_COLS = ["altLabels", "hiddenLabels"]


def normalize(s: str) -> str:
    """Lowercase and strip punctuation, keeping 'c++', 'c#' and 'node.js' intact.

    The trailing strip matters: punctuation becomes a space, so a label ending
    in one ('social worker (child protection)') would otherwise keep a trailing
    space in its key and never match a query that normalizes without it.
    """
    s = s.lower().strip()
    s = re.sub(r"[^\w\s+#.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_concepts(path, extra_cols, dtype=None):
    """Read an ESCO CSV into (lookup, concepts).

    lookup:   {normalized label -> conceptUri}, every variant of every concept
    concepts: {conceptUri -> {canonical, variants, description, **extra_cols}}
    """
    df = pd.read_csv(path, dtype=dtype, usecols=[
        "conceptUri", "preferredLabel", "altLabels", "hiddenLabels",
        "description", *extra_cols.values(),
    ])

    lookup, concepts = {}, {}

    for row in df.itertuples(index=False):
        canonical = str(row.preferredLabel).strip()

        variants = {canonical}
        for col in _LABEL_COLS:
            raw = getattr(row, col)
            if isinstance(raw, str):
                variants.update(v.strip() for v in raw.split("\n") if v.strip())

        concepts[row.conceptUri] = {
            "canonical": canonical,
            "variants": sorted(variants),
            "description": row.description,
            **{name: getattr(row, col) for name, col in extra_cols.items()},
        }

        for v in variants:
            lookup.setdefault(normalize(v), row.conceptUri)  # first writer wins

    return lookup, concepts


def load_occupations(path=None):
    """Return (lookup, occupations). Each occupation carries its ISCO group."""
    # iscoGroup must stay a string: the armed-forces groups are zero-padded, and
    # read as an int '0110' becomes '110', whose first two digits are '11'
    # (chief executives) rather than '01' (commissioned armed forces officers).
    return _load_concepts(path or DEFAULT_OCCUPATIONS_PATH,
                          {"code": "code", "isco_group": "iscoGroup"},
                          dtype={"iscoGroup": str, "code": str})


def load_skills(path=None):
    """Return (lookup, skills)."""
    return _load_concepts(path or DEFAULT_SKILLS_PATH,
                          {"skill_type": "skillType",
                           "reuse_level": "reuseLevel"})
