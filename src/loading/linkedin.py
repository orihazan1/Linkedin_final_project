"""Raw data loader.

Reads the LinkedIn job-posting dump off disk, normalizes it into the schema
below, and caches the result so later runs skip the parsing cost.

    data/Linkedin Jobs & Skills (2024)/
        linkedin_job_postings.csv    the row set: one job_link + job_title each
        job_skills.csv               the author-labelled skill list per job_link

    job_id      str        the posting's job_link; unique within the dump
    title       str        raw job title
    skills      list[str]  the posting's author-labelled skill terms

job_summary.csv (the third file in the download) is never read: it holds only
the free-text body, which nothing downstream uses, and it is ~5 GB.
"""

import os
import pickle

import pandas as pd

from src.paths import CACHE_DIR, DATA_DIR, LINKEDIN_DIR

FILES = {
    "linkedin_job_postings": os.path.join(LINKEDIN_DIR, "linkedin_job_postings.csv"),
    "job_skills": os.path.join(LINKEDIN_DIR, "job_skills.csv"),
}

# The dumps are not clean UTF-8; latin1 never raises and keeps the bytes.
ENCODING = "latin1"
CHUNKSIZE = 200_000

SCHEMA = ["job_id", "title", "skills"]


def _lookup_by_link(path, value_col, wanted_links, chunksize=CHUNKSIZE):
    """Scan a job_link-keyed CSV for the wanted rows.

    job_skills.csv is ~0.7 GB, so stream it and stop as soon as every wanted
    link has been found.
    """
    found = {}
    reader = pd.read_csv(path, usecols=["job_link", value_col],
                         chunksize=chunksize, encoding=ENCODING,
                         on_bad_lines="skip")
    for chunk in reader:
        hits = chunk[chunk["job_link"].isin(wanted_links)]
        for link, value in zip(hits["job_link"], hits[value_col]):
            found.setdefault(link, value)
        if len(found) >= len(wanted_links):
            break
    return found


def _split_skill_cell(cell):
    """'Cleaning, Sanitation, ...' -> ['cleaning', 'sanitation', ...]"""
    if not isinstance(cell, str):
        return []
    return [s.strip().lower() for s in cell.split(",") if s.strip()]


def load_postings(n_rows=5000):
    """The postings, joined with their skills. n_rows=None reads every row."""
    print(f"  reading {'all' if n_rows is None else f'{n_rows:,}'} postings")
    df = pd.read_csv(FILES["linkedin_job_postings"], nrows=n_rows,
                     usecols=["job_link", "job_title"],
                     encoding=ENCODING, on_bad_lines="skip")
    df = df.dropna(subset=["job_title"]).reset_index(drop=True)

    links = set(df["job_link"])
    print("  scanning job_skills.csv for matching rows...")
    skills_by_link = _lookup_by_link(FILES["job_skills"], "job_skills", links)
    print(f"  matched skills for {len(skills_by_link):,}/{len(links):,} postings")

    return pd.DataFrame({
        "job_id": df["job_link"],
        "title": df["job_title"],
        "skills": df["job_link"].map(skills_by_link).map(_split_skill_cell),
    })[SCHEMA]


def size_tag(sample_n):
    """Filename-friendly label for a sample size. None means 'every row'."""
    return "all" if sample_n is None else str(sample_n)


def cache_path(sample_n):
    """One cache file per sample size, so a quick 300-row test run cannot
    overwrite the result of an hour-long full run.

    The `_a` suffix marked this dump as source "a" when a second posting
    dataset was supported. It is kept only so existing caches stay readable.
    """
    return os.path.join(CACHE_DIR, f"corpus_{size_tag(sample_n)}_a.pkl")


def save_data(corpus_df, sample_n):
    """Pickle the loaded corpus -- it round-trips the list-valued `skills`
    column as-is, which Parquet would not without extra dependencies.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(sample_n)
    with open(path, "wb") as fh:
        pickle.dump({"corpus": corpus_df}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved cache -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def load_cached(sample_n):
    """Return the cached bundle, or None if this sample size hasn't been built."""
    path = cache_path(sample_n)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def check_files_exist():
    """Fail early rather than halfway through a 0.7 GB scan."""
    missing = [name for name, path in FILES.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing raw data files {missing} -- expected them in {DATA_DIR}. "
            f"See README.md for how to download them.")


def load_all(sample_n=5000, use_cache=True):
    """Load sample_n postings (None = every row) into one corpus DataFrame."""
    if use_cache:
        cached = load_cached(sample_n)
        if cached is not None:
            print(f"Loaded cached corpus for sample_n={sample_n} "
                  f"(delete {cache_path(sample_n)} to rebuild)")
            # SCHEMA doubles as a projection: older caches carry columns this
            # loader no longer produces.
            return cached["corpus"][SCHEMA]

    check_files_exist()

    print("Loading postings...")
    corpus_df = load_postings(n_rows=sample_n)
    print(f"\ncorpus={corpus_df.shape}")
    save_data(corpus_df, sample_n)
    return corpus_df
