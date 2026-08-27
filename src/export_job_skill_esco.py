"""Flatten a finished pipeline run into one shareable CSV.

    python -m src.export_job_skill_esco

The pipeline writes cache/*.pkl, which nothing outside Python can open. This
writes data/job_skill_esco_name.csv: one row per posting whose title resolved
to an ESCO occupation and whose skills produced at least one ESCO skill,
carrying the occupation (label, URI, how it matched), its ISCO group at two
levels, and the canonical skill names alongside the original raw terms.

No matching happens here; it reads the run main.py already saved.
"""

import os
import sys

import pandas as pd

from src import pipeline
from src.loading import esco, linkedin
from src.paths import DATA_DIR, ESCO_DIR

# ==========================================================================
# CONFIGURATION -- edit these, then run the file
# ==========================================================================

# Which saved run to export, i.e. which cache/pipeline_output_{N}_a.pkl.
# None = the full 1.35M-row run.
SAMPLE_SIZE = None

# Minimum ESCO-matched skills a posting needs to be kept.
MIN_ESCO_SKILLS = 1

OUT_PATH = os.path.join(DATA_DIR, "job_skill_esco_name.csv")

# ==========================================================================

ISCO_PATH = os.path.join(ESCO_DIR, "ISCOGroups_en.csv")

COLUMNS = ["job_link", "job_title", "esco_occupation", "esco_uri",
           "title_match_method",
           "isco_submajor_code", "isco_submajor_label",
           "isco_group_code", "isco_group_label",
           "job_skills_original", "n_skills_original",
           "esco_skills", "n_skills_esco", "esco_skill_match_methods",
           "skills_unmatched"]

# Slot separator for the list-valued columns. Not a comma: 33 ESCO canonical
# labels contain one ("organise information, objects and resources"), so a
# comma-joined cell could not be split back into slots. No ESCO label and no raw
# skill term contains a pipe, so cell.split(" | ") round-trips exactly.
SEP = " | "


def join_list(value):
    """List-valued cell -> the SEP-separated string a CSV can hold.

    An unmatched slot is None and renders as an empty slot rather than the
    string "None", which would be indistinguishable from a skill named that.
    Keeping the empty slot is what preserves the index alignment between
    job_skills_original and esco_skills.
    """
    if isinstance(value, (list, tuple)):
        return SEP.join("" if v is None else str(v) for v in value)
    return ""


def list_len(value):
    return len(value) if isinstance(value, (list, tuple)) else 0


def n_matched(value):
    """How many slots actually resolved -- list_len counts the empty ones too."""
    if isinstance(value, (list, tuple)):
        return sum(v is not None for v in value)
    return 0


def isco_labels():
    """{ISCO code -> preferredLabel}, at every level, not just the 10 majors.

    dtype=str because the codes are zero-padded and would otherwise lose the
    leading zero.
    """
    df = pd.read_csv(ISCO_PATH, usecols=["code", "preferredLabel"], dtype=str)
    return dict(zip(df["code"].str.strip(), df["preferredLabel"]))


def original_skills(sample_size):
    """{job_id -> raw skill list} from the corpus as it was before matching.

    The pipeline overwrites `skills` in place with canonical labels, so the raw
    text of each term only survives in the pre-match corpus cache.
    """
    cached = linkedin.load_cached(sample_size)
    if cached is None:
        raise FileNotFoundError(
            f"No raw corpus cache at {linkedin.cache_path(sample_size)} -- it "
            f"holds the pre-match skill lists this export needs.")
    raw = cached["corpus"]
    return dict(zip(raw["job_id"], raw["skills"]))


def build(sample_size=SAMPLE_SIZE, min_esco_skills=MIN_ESCO_SKILLS):
    """Return the export frame, in COLUMNS order."""
    print(f"Loading pipeline output (sample_size={sample_size})...")
    corpus = pipeline.load_pipeline_output(sample_size)["corpus"]
    print(f"  {len(corpus):,} rows")

    n_total = len(corpus)
    has_occupation = corpus["esco_uri"].notna()
    n_skills_esco = corpus["skills"].map(n_matched)
    kept = corpus[has_occupation & (n_skills_esco >= min_esco_skills)].copy()
    print(f"  with an ESCO occupation:  {int(has_occupation.sum()):,}")
    print(f"  ... and >={min_esco_skills} ESCO skill(s):  {len(kept):,} / "
          f"{n_total:,} ({100 * len(kept) / n_total:.1f}%)")

    print("Attaching ISCO groups...")
    _lookup, occupations = esco.load_occupations()
    uri_to_group = {uri: entry["isco_group"] for uri, entry in occupations.items()}
    labels = isco_labels()

    group_code = kept["esco_uri"].map(uri_to_group).fillna("")
    submajor_code = group_code.str[:2]

    print("Recovering original skill lists...")
    raw_skills = kept["job_id"].map(original_skills(sample_size))

    out = pd.DataFrame({
        "job_link": kept["job_id"],
        "job_title": kept["title"],
        "esco_occupation": kept["esco_occupation"],
        "esco_uri": kept["esco_uri"],
        "title_match_method": kept["match_method"],
        "isco_submajor_code": submajor_code,
        "isco_submajor_label": submajor_code.map(labels).fillna(""),
        "isco_group_code": group_code,
        "isco_group_label": group_code.map(labels).fillna(""),
        "job_skills_original": raw_skills.map(join_list),
        "n_skills_original": raw_skills.map(list_len),
        "esco_skills": kept["skills"].map(join_list),
        "n_skills_esco": kept["skills"].map(n_matched),
        "esco_skill_match_methods": kept["skill_match_methods"].map(join_list),
        "skills_unmatched": kept["skills_dropped_terms"].map(join_list),
    })
    return out[COLUMNS].reset_index(drop=True)


def main():
    try:
        out = build()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH} "
          f"({os.path.getsize(OUT_PATH) / 1e6:.1f} MB, {len(out):,} rows)")

    print(f"\ndistinct ESCO occupations: {out['esco_occupation'].nunique():,}")
    print(f"distinct ISCO sub-majors:  {out['isco_submajor_code'].nunique():,}")
    print(f"skills per posting:        {out['n_skills_esco'].mean():.1f} ESCO "
          f"of {out['n_skills_original'].mean():.1f} original (mean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
