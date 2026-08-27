"""The pipeline itself.

run_pipeline() loads the postings once and walks them through every matching
stage, returning one enriched corpus. Each stage builds on the one before it:

    1. LOAD          linkedin.load_all()                  raw postings
    2. RULE-BASED    rule_based_matching.run()            exact ESCO matches
    3. TITLES        semantic_matching.match_titles()     + embeddings, cross-encoder
    4. SKILLS        semantic_matching.match_skills()     + embeddings
    5. SAVE          cache/pipeline_output_{n}_a.pkl
"""

import os
import pickle

from src.loading import linkedin
from src.canonicalization import semantic_matching, rule_based_matching
from src.paths import CACHE_DIR


def pipeline_output_path(sample_size):
    """One output file per sample size, so a quick test run cannot overwrite
    the result of an hour-long full run.
    """
    return os.path.join(CACHE_DIR,
                        f"pipeline_output_{linkedin.size_tag(sample_size)}_a.pkl")


def load_pipeline_output(sample_size):
    """Restore a finished run in seconds instead of re-running every stage."""
    path = pipeline_output_path(sample_size)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved run at {path}. Run main.py with "
            f"SAMPLE_SIZE={sample_size} first.")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def merge_stage(corpus_df, stage_df, columns):
    """Fold a stage's new columns back into the corpus.

    Joined on job_id rather than the index: stages may drop rows or reset the
    index, and job_id is unique within the dump, so it survives that.
    """
    new = [c for c in columns if c in stage_df.columns]
    overlapping = [c for c in new if c in corpus_df.columns and c != "job_id"]
    return (corpus_df.drop(columns=overlapping)
                     .merge(stage_df[["job_id"] +
                                     [c for c in new if c != "job_id"]],
                            on="job_id", how="left"))


def print_pipeline_summary(result):
    """The end-of-run report."""
    corpus_df = result["corpus"]
    banner("SUMMARY")
    print(f"corpus:     {len(corpus_df):,} rows")
    print(f"columns:    {list(corpus_df.columns)}")

    if "occupation" in corpus_df:
        matched = int(corpus_df["occupation"].notna().sum())
        print(f"\nrows with an ESCO occupation: {matched:,} / {len(corpus_df):,} "
              f"({100 * matched / len(corpus_df):.1f}%)")

    title_df = result["title_df"]
    if title_df is not None:
        n_exact = int((title_df["match_method"] == "exact").sum())
        n_matched = int(title_df["esco_occupation"].notna().sum())
        print(f"title matching:   {n_matched:,} / {len(title_df):,} "
              f"matched  (exact-only baseline was {n_exact:,})")

    skill_meta = result["skill_meta"]
    if skill_meta is not None:
        counts = skill_meta["counts"]
        print(f"skill matching:   {sum(counts.values()):,} skill mentions kept "
              f"(exact {counts['exact']:,}, embedding {counts['embedding']:,}, "
              f"margin {counts['embedding_margin']:,}, ce {counts['embedding_ce']:,})")


def run_pipeline(sample_size=20000, use_cache=True,
                 drop_unmatched=False, run_title_embeddings=True,
                 run_skill_embeddings=True, skill_cross_encoder=False,
                 save_output=True, verbose=True):
    """Run every stage and return the enriched corpus.

    The postings are loaded once here and passed into each stage, so every
    stage sees the same rows.

    Returns a dict with:
        corpus      the enriched DataFrame (the main artifact)
        title_df    stage 3's output, or None if it was skipped
        title_meta  stage 3's metadata, or None
        skill_df    stage 4's output, or None if it was skipped
        skill_meta  stage 4's metadata, or None
    """
    # ---- 1. load, once -----------------------------------------------------
    banner("STAGE 1/4  LOADING DATA")
    raw_corpus_df = linkedin.load_all(sample_n=sample_size, use_cache=use_cache)
    print(f"loaded {len(raw_corpus_df):,} postings")

    # ---- 2. rule-based exact matching --------------------------------------
    banner("STAGE 2/4  RULE-BASED MATCHING (exact)")
    corpus_df = rule_based_matching.run(raw_corpus_df,
                                        drop_unmatched_rows=drop_unmatched)

    # ---- 3. semantic title matching ----------------------------------------
    # Deliberately raw_corpus_df, not corpus_df: the matcher recomputes its own
    # exact baseline and needs the untouched title column.
    title_df, title_meta = None, None
    if run_title_embeddings:
        banner("STAGE 3/4  TITLE MATCHING (embeddings + cross-encoder)")
        title_df, title_meta = semantic_matching.match_titles(
            corpus_df=raw_corpus_df, use_cross_encoder=True)
        if verbose:
            semantic_matching.print_summary(title_df, title_meta)

        corpus_df = merge_stage(corpus_df, title_df,
                                ["esco_occupation", "esco_uri", "match_method"])
        # `occupation` came from stage 2 (exact only); stage 3's esco_occupation
        # is a superset, so prefer it where it exists.
        if "esco_occupation" in corpus_df:
            corpus_df["occupation"] = (corpus_df["esco_occupation"]
                                       .fillna(corpus_df["occupation"]))

    # ---- 4. semantic skill matching ----------------------------------------
    # Also raw_corpus_df: match_skills needs the original raw skill terms, and
    # stage 2 already overwrote `skills` with canonical labels.
    skill_df, skill_meta = None, None
    if run_skill_embeddings:
        banner("STAGE 4/4  SKILL MATCHING (embeddings)")
        skill_df, skill_meta = semantic_matching.match_skills(
            corpus_df=raw_corpus_df, use_cross_encoder=skill_cross_encoder)
        if verbose:
            semantic_matching.print_skill_summary(skill_df, skill_meta)

        corpus_df = merge_stage(corpus_df, skill_df,
                                semantic_matching.SKILL_NEW_COLUMNS)

    result = {"corpus": corpus_df,
              "title_df": title_df, "title_meta": title_meta,
              "skill_df": skill_df, "skill_meta": skill_meta}

    # ---- 5. save -----------------------------------------------------------
    if save_output:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = pipeline_output_path(sample_size)

        # title_df/skill_df are left out: their new columns are already merged
        # into corpus_df, and they carry a full copy of the posting text. The
        # metas are the part that is NOT recoverable from corpus_df -- by_text
        # holds the per-text scores and margins that recalibration needs.
        payload = {"corpus": corpus_df,
                   "title_meta": title_meta, "skill_meta": skill_meta,
                   "config": {"sample_size": sample_size,
                              "drop_unmatched": drop_unmatched,
                              "title_embeddings": run_title_embeddings,
                              "skill_embeddings": run_skill_embeddings,
                              "skill_cross_encoder": skill_cross_encoder}}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSaved pipeline output -> {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")
        print(f"  restore with: pipeline.load_pipeline_output({sample_size})")

    return result
