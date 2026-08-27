"""Rule-based canonicalization against ESCO.

Every skill and job title mined in linkedin.py is a raw surface string:
'ms excel', 'Sr. Software Engineer II (Remote)'. This stage maps each one onto
its official ESCO preferred label by exact match on a normalized form, so later
stages work over a closed, controlled vocabulary.

A raw title almost never hits on the first try, so title_candidates() peels the
posting-specific noise off and tries progressively shorter forms.

Columns after run():
    skills       list[str]  one slot per input term, canonical label or None
    occupation   str|<NA>   ESCO canonical occupation, <NA> if the title missed
"""

import re

import pandas as pd

from src.loading import esco

_DEFAULT_SKILLS_PATH = esco.DEFAULT_SKILLS_PATH
_DEFAULT_OCCUPATIONS_PATH = esco.DEFAULT_OCCUPATIONS_PATH

# Posting noise that is never part of an ESCO occupation label.
_PAREN = re.compile(r"[\(\[\{].*?[\)\]\}]")
_SPLITTERS = re.compile(r"\s+-\s+|\s+\|\s+|\s+/\s+|\s+at\s+|\s*,\s*|\s+:\s+")
_SENIORITY = re.compile(
    r"^(sr|snr|senior|jr|junior|lead|principal|staff|entry[ -]?level|"
    r"assistant|associate|head of|deputy)\b[.\s]*", re.IGNORECASE)
_TRAILING_LEVEL = re.compile(r"\s+(i{1,3}|iv|v|\d)$", re.IGNORECASE)

# Employment type / work arrangement. Limited to tokens that can never
# themselves name an occupation: 'contract', 'permanent' and 'hiring' are
# deliberately absent, since 'Contract Manager' and 'Hiring Manager' are jobs.
_JOB_TYPE = re.compile(
    r"\b(ft|pt|f\s*/\s*t|p\s*/\s*t|full[\s-]?time|part[\s-]?time|fulltime|"
    r"parttime|prn|per\s+diem|temp|temporary|seasonal|"
    r"remote|onsite|on[\s-]site|hybrid|w2|1099|hourly|salaried)\b",
    re.IGNORECASE)

# Pure rank markers: they modify a role but are never the role, so they come off
# either end. 'head of' is excluded -- reducing 'Head of Marketing' to
# 'marketing' loses the part that makes it an occupation.
_RANK = re.compile(
    r"(sr|snr|senior|jr|junior|lead|principal|staff|entry[\s-]?level|"
    r"experienced|trainee|graduate|new\s+grad|mid[\s-]?level|deputy)",
    re.IGNORECASE)
_RANK_PREFIX = re.compile(r"^\s*" + _RANK.pattern + r"\b[.\s]*", re.IGNORECASE)
_RANK_SUFFIX = re.compile(r"[\s,-]+" + _RANK.pattern + r"\s*$", re.IGNORECASE)

# 'associate'/'assistant' modify a role as a prefix ('associate product
# manager') but ARE the role as a suffix ('sales associate'), so they only come
# off the front.
_ROLE_PREFIX = re.compile(r"^\s*(associate|assistant)\b[.\s]*", re.IGNORECASE)

# ESCO files bare role words as hiddenLabels of one arbitrary specific
# occupation, and the lookup is first-writer-wins: 'supervisor' resolves to oil
# refinery control room operator, 'driver' to hearse driver, 'teacher' to
# politics lecturer. A candidate that is only a generic role word is refused;
# the same word inside a longer phrase ('stage director') is still fine.
_GENERIC_ROLES = {
    "supervisor", "assistant", "technician", "director", "manager", "associate",
    "coordinator", "specialist", "consultant", "analyst", "lead", "engineer",
    "intern", "staff", "officer", "agent", "clerk", "operator", "worker",
    "professional", "executive", "representative", "administrator", "advisor",
    "adviser", "trainee", "apprentice", "expert", "generalist", "partner",
    "head", "chief", "president", "owner", "member", "employee", "contractor",
    "volunteer", "remote", "part time", "full time", "other",
    "driver", "teacher", "designer", "editor", "inspector", "installer",
    "planner", "general", "controller", "host",
}


def load_esco(skills_path=_DEFAULT_SKILLS_PATH,
              occupations_path=_DEFAULT_OCCUPATIONS_PATH):
    """Load both ESCO dictionaries once."""
    print("Loading ESCO dictionaries...")
    skill_lookup, skills = esco.load_skills(skills_path)
    occ_lookup, occupations = esco.load_occupations(occupations_path)
    print(f"  skills:      {len(skills):,} concepts / {len(skill_lookup):,} labels")
    print(f"  occupations: {len(occupations):,} concepts / {len(occ_lookup):,} labels")
    return skill_lookup, skills, occ_lookup, occupations


def canonicalize_skills_aligned(raw_skills, lookup, skills):
    """Map a posting's skill list onto ESCO, one output slot per input term.

    Returns a list the same length and order as raw_skills, holding the ESCO
    canonical label at each position and None where the term did not resolve.
    Deliberately does not deduplicate: a concept named twice comes back at both
    positions, so callers wanting concept counts must dedupe themselves.
    """
    out = []
    for term in raw_skills or []:
        uri = lookup.get(esco.normalize(str(term)))
        out.append(skills[uri]["canonical"] if uri is not None else None)
    return out


def clean_title(title: str) -> str:
    """Reduce a raw posting title to the occupation it names.

        'Sr. Software Engineer II (Remote)'  -> 'software engineer'
        'LEAD SALES ASSOCIATE-PT'            -> 'sales associate'
        'Associate Product Manager, Senior'  -> 'product manager'

    Returns '' when nothing survives, so callers must handle the empty string.
    """
    if not isinstance(title, str) or not title.strip():
        return ""

    text = title.lower()
    text = _PAREN.sub(" ", text)
    text = re.sub(r"[-/_|]+", " ", text)        # 'front-end' -> 'front end'
    text = _JOB_TYPE.sub(" ", text)
    text = re.sub(r"[^\w\s+#.]", " ", text)     # keep c++, c#, node.js intact
    text = re.sub(r"\s+", " ", text).strip()
    text = _TRAILING_LEVEL.sub("", text).strip()

    # Repeated, since 'senior lead engineer' carries two ranks. min_words is how
    # much has to survive: a rank may reduce a title to one word, but
    # 'associate'/'assistant' may not, because 'Assistant Manager' is its own
    # role and 'manager' is not a usable substitute for it.
    for pattern, min_words in ((_RANK_PREFIX, 1), (_RANK_SUFFIX, 1),
                               (_ROLE_PREFIX, 2)):
        while True:
            stripped = re.sub(r"\s+", " ", pattern.sub(" ", text)).strip()
            if not stripped or stripped == text or len(stripped.split()) < min_words:
                break
            text = stripped

    text = _TRAILING_LEVEL.sub("", text).strip()
    return re.sub(r"\s+", " ", text).strip()


def title_candidates(title):
    """Progressively stripped-down forms of a raw title, best guess first."""
    if not isinstance(title, str) or not title.strip():
        return []

    def peel(text):
        return _TRAILING_LEVEL.sub("", _SENIORITY.sub("", text)).strip()

    # Whole-title forms first: they carry the most information, so a hit there
    # is the most trustworthy one available.
    candidates = [title]
    stripped = _PAREN.sub(" ", title)
    if stripped != title:
        candidates.append(stripped)
    candidates.extend(peel(c) for c in list(candidates))
    candidates.append(clean_title(title))

    # Then the pieces. The occupation is not reliably the first segment
    # ('Remote - Data Analyst'), so try the longest first: more words means more
    # specific means a safer match.
    segments = [s.strip() for s in _SPLITTERS.split(stripped) if s and s.strip()]
    segments.sort(key=len, reverse=True)
    for segment in segments:
        candidates.append(segment)
        candidates.append(peel(segment))

    out, seen = [], set()
    for cand in candidates:
        key = esco.normalize(cand)
        if key and key not in _GENERIC_ROLES and key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def canonicalize_title(title, lookup, occupations, use_candidates=True):
    """Return the ESCO canonical occupation for a raw title, or None."""
    candidates = title_candidates(title) if use_candidates else [title]
    for cand in candidates:
        uri = lookup.get(esco.normalize(str(cand)))
        if uri is not None:
            return occupations[uri]["canonical"]
    return None


def canonicalize_frame(df, esco_data, use_candidates=True):
    """Canonicalize one dataframe; returns the new frame."""
    skill_lookup, skills, occ_lookup, occupations = esco_data
    new_skills, occupations_col = [], []

    for raw_skills, title in zip(df["skills"], df["title"]):
        raw_skills = raw_skills if isinstance(raw_skills, (list, tuple)) else []
        new_skills.append(
            canonicalize_skills_aligned(raw_skills, skill_lookup, skills))

        if isinstance(title, str) and title.strip():
            canonical = canonicalize_title(title, occ_lookup, occupations,
                                           use_candidates=use_candidates)
            occupations_col.append(canonical if canonical is not None else pd.NA)
        else:
            occupations_col.append(pd.NA)

    out = df.copy()
    out["skills"] = new_skills
    out["occupation"] = pd.array(occupations_col, dtype="string")
    return out


def run(corpus_df, skills_path=_DEFAULT_SKILLS_PATH,
        occupations_path=_DEFAULT_OCCUPATIONS_PATH, drop_unmatched_rows=False,
        use_candidates=True):
    """Canonicalize the corpus against ESCO; returns the enriched frame.

    With drop_unmatched_rows, rows whose title never mapped to an ESCO
    occupation are deleted rather than kept with occupation=<NA>.
    """
    esco_data = load_esco(skills_path, occupations_path)
    corpus_df = canonicalize_frame(corpus_df, esco_data, use_candidates)

    if drop_unmatched_rows:
        corpus_df = corpus_df.dropna(subset=["occupation"]).reset_index(drop=True)

    return corpus_df
