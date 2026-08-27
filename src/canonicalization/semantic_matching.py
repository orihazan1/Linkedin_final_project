"""Embedding-based ESCO matching for job titles and skill terms.

rule_based_matching resolves roughly a quarter of the titles by exact string
match on a normalized form. The rest fail for reasons string matching cannot
fix: ESCO has no label 'registered nurse', though 'specialist nurse' means
nearly the same thing, and a skill written 'ms excel' is filed under a
different phrasing.

This module fills that gap with sentence embeddings, exact matches first. Each
unresolved text climbs an acceptance ladder and stops at the first rung it
clears:

    exact             kept as-is, score 1.0
    embedding         nearest ESCO label at or above SIMILARITY_THRESHOLD
    embedding_margin  below that, but decisively clear of the runner-up label
                      (score >= MARGIN_MIN_SCORE and best - runner-up >=
                      MARGIN_THRESHOLD) -- a different signal than simply
                      lowering the absolute threshold
    embedding_ce      still unmatched, but a cross-encoder scores one of its
                      top-k bi-encoder candidates >= CE_THRESHOLD

The cross-encoder rung exists because the bi-encoder pools each side into its
own vector and dots them, which is why it falls for shared-word domain drift:
'Employee Communications Manager' lands on 'telecommunications manager' at
0.90. A cross-encoder reads both sides as one sequence and weighs the whole
phrase, so it catches that. It is far too slow to run against all ~32k ESCO
labels, hence the top-k shortlist.

Every ESCO variant (preferredLabel + altLabels + hiddenLabels) is a match
target, so a text can land via any phrasing, and the winning variant is
recorded so a match can always be traced back. Bare generic role words are
excluded from the occupation index -- ESCO files 'supervisor' as a hiddenLabel
of 'oil refinery control room operator', and embeddings make that trap worse,
since every supervisor-ish title sits close to the bare word.
"""

import hashlib
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src.loading import esco, linkedin
from src.canonicalization import rule_based_matching
from src.paths import CACHE_DIR

DEFAULT_OCCUPATIONS_PATH = esco.DEFAULT_OCCUPATIONS_PATH
DEFAULT_SKILLS_PATH = esco.DEFAULT_SKILLS_PATH

# A GPU is used when one is available. On Colab this is the difference between
# a full run finishing and not; nothing else in the module needs to change.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BATCH_SIZE = 1024 if DEVICE.type == "cuda" else 256
CE_BATCH_SIZE = 256 if DEVICE.type == "cuda" else 64
CE_TOP_K = 5
SIM_CHUNK = 512           # query rows per similarity block
MAX_TOKENS = 64           # titles and skill terms are short
SAMPLE_N = 5000

# --------------------------------------------------------------------------
# Thresholds
#
# SIMILARITY_THRESHOLD was derived by sampling 25 matches per score band on
# this corpus: precision was ~80% at >=0.80 but only ~55% in 0.75-0.80, where
# 'hvac foreman' -> footwear production technician and 'lockbox specialist' ->
# locksmith start appearing. The rest were set by eye from the same kind of
# sampling and are cruder. print_calibration() re-derives all of them.
#
# The skill thresholds were seeded from the title ones. Skill terms are short
# noun phrases rather than full titles, so their score distribution has no
# particular reason to sit in the same place; treat them as provisional.
# --------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.90
MARGIN_MIN_SCORE = 0.85
MARGIN_THRESHOLD = 0.20
# A raw MS MARCO relevance score (sigmoid of the logit), not on the same scale
# as bi-encoder cosine -- it ranks query/passage relevance rather than measuring
# title/label equivalence, so SIMILARITY_THRESHOLD's value has no bearing here.
CE_THRESHOLD = 0.50

SKILL_SIMILARITY_THRESHOLD = 0.90
SKILL_MARGIN_MIN_SCORE = 0.85
SKILL_MARGIN_THRESHOLD = 0.20
SKILL_CE_THRESHOLD = 0.50

# Hand-judged precision per rung, from 30-title random samples via the
# print_*_sample functions. There is no ground truth to score against
# automatically. The embedding_margin figure was measured at looser settings
# than the ones above (min score 0.60, margin 0.02), so it understates the
# current rung; re-sample after changing any threshold.
PRECISION_ESTIMATE = {
    "exact": 1.00,             # controlled-vocabulary string match, not sampled
    "embedding": 0.67,         # 20/30
    "embedding_margin": 0.40,  # 12/30, at the looser settings noted above
    "embedding_ce": 0.60,      # 18/30
}

# Optional ESCO-side enrichment. A bare 2-6 word label often cannot separate two
# concepts sharing one word ('police inspector' vs a building-inspection
# occupation); appending part of the ESCO description gives the encoder
# something to work with. Enabling it changes what gets embedded, so the
# thresholds above would need re-deriving.
ESCO_DESC_MAX_CHARS = 200
ESCO_DESC_MAX_TOKENS = 128

EMB_CACHE_PATH = os.path.join(CACHE_DIR, "esco_occ_emb.npz")
EMB_DESC_CACHE_PATH = os.path.join(CACHE_DIR, "esco_occ_emb_desc.npz")
EMB_SKILL_CACHE_PATH = os.path.join(CACHE_DIR, "esco_skill_emb.npz")
EMB_SKILL_DESC_CACHE_PATH = os.path.join(CACHE_DIR, "esco_skill_emb_desc.npz")

# skill_match_methods runs parallel to the "skills" column, and both are
# index-aligned with the raw input skill list: slot i means "input term i
# resolved to this label via this method", or None/None when it did not.
# Per-term scores for recalibration live in meta["by_text"], not on the frame.
SKILL_NEW_COLUMNS = ["skills", "skill_match_methods", "skills_dropped_terms",
                     "n_skills_exact", "n_skills_embedding",
                     "n_skills_embedding_margin", "n_skills_embedding_ce"]


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------

def load_model(model_name=MODEL_NAME):
    """Tokenizer + bi-encoder on DEVICE, in eval mode."""
    print(f"Loading {model_name} on {DEVICE} (first run downloads ~90 MB)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return tokenizer, model


def load_cross_encoder(model_name=CE_MODEL_NAME):
    """Tokenizer + cross-encoder on DEVICE, in eval mode.

    Cross-encoder checkpoints on the HF hub are plain single-label
    classification heads, so AutoModelForSequenceClassification is enough and
    the sentence-transformers package is not needed.
    """
    print(f"Loading cross-encoder {model_name} on {DEVICE} "
          f"(first run downloads ~80 MB)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return tokenizer, model


def embed(texts, tokenizer, model, batch_size=BATCH_SIZE, label="",
          max_tokens=MAX_TOKENS):
    """Mean-pooled, L2-normalized sentence embeddings.

    Normalizing here makes cosine similarity a plain dot product later.
    max_tokens defaults to the short title/label budget; the ESCO index passes
    ESCO_DESC_MAX_TOKENS when description enrichment is on.
    """
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_tokens, return_tensors="pt")
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state

            # Pool over real tokens only -- padding must not dilute the vector.
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            out[start:start + len(batch)] = pooled.cpu().numpy()
            if label and (start // batch_size) % 20 == 0:
                print(f"    {label}: {min(start + batch_size, len(texts)):,}"
                      f"/{len(texts):,}")
    return out


def cross_encode(pairs, tokenizer, model, batch_size=CE_BATCH_SIZE, label=""):
    """Score (text_a, text_b) pairs jointly. Returns sigmoid(logit) per pair.

    Each pair is encoded as one sequence, so the model attends across both
    sides at once. That costs a forward pass per pair rather than per text,
    which is why this only ever runs on a short candidate list.
    """
    out = np.empty(len(pairs), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            encoded = tokenizer([a for a, _ in batch], [b for _, b in batch],
                                padding=True, truncation=True,
                                max_length=MAX_TOKENS, return_tensors="pt")
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            logits = model(**encoded).logits.squeeze(-1)
            out[start:start + len(batch)] = torch.sigmoid(logits).cpu().numpy()
            if label and (start // batch_size) % 20 == 0:
                print(f"    {label}: {min(start + batch_size, len(pairs)):,}"
                      f"/{len(pairs):,}")
    return out


# --------------------------------------------------------------------------
# ESCO target index
# --------------------------------------------------------------------------

def build_esco_index(concepts, skip_generic=False):
    """Flatten ESCO concepts into (labels, uris) to embed as match targets.

    skip_generic drops bare role words, which is an occupation-side quirk: they
    are hiddenLabels of one arbitrary specific occupation, so matching against
    them is noise. Skill labels have no documented equivalent trap, so every
    distinct normalized variant is kept there.
    """
    labels, uris, skipped = [], [], 0
    seen = set()

    for uri, entry in concepts.items():
        for variant in entry["variants"]:
            key = esco.normalize(variant)
            if not key or key in seen:
                continue
            if skip_generic and key in rule_based_matching._GENERIC_ROLES:
                skipped += 1
                continue
            seen.add(key)
            labels.append(variant)
            uris.append(uri)

    note = f" ({skipped} generic labels skipped)" if skip_generic else ""
    print(f"  ESCO index: {len(labels):,} labels over {len(set(uris)):,} "
          f"concepts{note}")
    return labels, uris


def build_esco_embed_text(labels, uris, concepts, max_chars=ESCO_DESC_MAX_CHARS):
    """Label + a snippet of the concept's ESCO description, for embedding.

    Same order and length as labels, so only what gets embedded changes --
    everything downstream still indexes into labels unchanged.
    """
    texts = []
    for label, uri in zip(labels, uris):
        desc = concepts[uri].get("description")
        texts.append(f"{label} - {desc[:max_chars]}"
                     if isinstance(desc, str) and desc else label)
    return texts


def _content_signature(texts):
    """Hash of the exact text embedded, so the cache goes stale when the
    content changes even if the label count does not -- which is what toggling
    description enrichment does.
    """
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def esco_embeddings(embed_text, tokenizer, model, cache_path=EMB_CACHE_PATH,
                    max_tokens=MAX_TOKENS):
    """Embed the ESCO index, reusing the cached matrix when it still applies."""
    signature = _content_signature(embed_text)
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        cached_sig = str(cached["signature"]) if "signature" in cached.files else None
        if (str(cached["model"]) == MODEL_NAME
                and int(cached["n_labels"]) == len(embed_text)
                and cached_sig == signature):
            print(f"  reusing cached ESCO embeddings ({cache_path})")
            return cached["emb"]
        print("  cached ESCO embeddings are stale, re-encoding")

    print(f"  encoding {len(embed_text):,} ESCO index entries...")
    emb = embed(embed_text, tokenizer, model, label="esco", max_tokens=max_tokens)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path, emb=emb, model=MODEL_NAME,
                        n_labels=len(embed_text), signature=signature)
    print(f"  saved -> {cache_path}")
    return emb


def esco_index(concepts, tokenizer, model, skip_generic, cache_path,
               desc_cache_path, use_description, desc_max_chars):
    """The match targets and their embedding matrix, description-enriched or not."""
    labels, uris = build_esco_index(concepts, skip_generic=skip_generic)
    if use_description:
        text = build_esco_embed_text(labels, uris, concepts, desc_max_chars)
        emb = esco_embeddings(text, tokenizer, model, desc_cache_path,
                              ESCO_DESC_MAX_TOKENS)
    else:
        emb = esco_embeddings(labels, tokenizer, model, cache_path)
    return labels, uris, emb


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

def top_k_matches(query_emb, esco_emb, k, chunk=SIM_CHUNK):
    """Top-k ESCO labels per query row. Returns (idx, score), both (n, k).

    One similarity block at a time -- the full matrix is hundreds of MB at
    corpus scale. Runs on DEVICE, which on a GPU turns the second-slowest stage
    of the pipeline into a handful of matmuls.
    """
    targets = torch.as_tensor(esco_emb).to(DEVICE)
    idx = np.empty((len(query_emb), k), dtype=np.int64)
    score = np.empty((len(query_emb), k), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(query_emb), chunk):
            block = torch.as_tensor(query_emb[start:start + chunk]).to(DEVICE)
            top_score, top_idx = (block @ targets.T).topk(k, dim=1)
            idx[start:start + chunk] = top_idx.cpu().numpy()
            score[start:start + chunk] = top_score.cpu().numpy()

    return idx, score


def bi_encoder_method(score, margin, threshold, margin_min_score,
                      margin_threshold):
    """The bi-encoder's verdict: 'embedding', 'embedding_margin', or None.

    Both matchers and the cross-encoder pool share this, so the ladder cannot
    drift between them -- anything this accepts is never reranked, and anything
    it rejects always gets its shot at the cross-encoder.
    """
    if score >= threshold:
        return "embedding"
    if score >= margin_min_score and margin >= margin_threshold:
        return "embedding_margin"
    return None


def rerank_with_cross_encoder(texts, query_emb, labels, uris, concepts,
                              esco_emb, tokenizer_ce, model_ce, top_k=CE_TOP_K):
    """Cross-encoder rerank of each text's top-k bi-encoder candidates.

    Returns (by_text_ce, elapsed_seconds), mapping text -> (canonical, uri,
    matched_label, ce_score, bi_encoder_score) for whichever candidate the
    cross-encoder liked best. Reuses the embeddings the caller already has, so
    only the cross-encoder pass is new work.
    """
    start_time = time.time()

    cand_idx, cand_score = top_k_matches(query_emb, esco_emb, top_k)
    pairs = [(text, labels[cand_idx[i, j]])
             for i, text in enumerate(texts) for j in range(top_k)]

    print(f"  cross-encoding {len(pairs):,} (text, candidate) pairs "
          f"({len(texts):,} texts x top-{top_k})...")
    ce_scores = cross_encode(pairs, tokenizer_ce, model_ce,
                             label="rerank").reshape(len(texts), top_k)

    by_text_ce = {}
    for i, text in enumerate(texts):
        j = int(ce_scores[i].argmax())
        uri = uris[cand_idx[i, j]]
        by_text_ce[text] = (concepts[uri]["canonical"], uri,
                            labels[cand_idx[i, j]],
                            float(ce_scores[i, j]), float(cand_score[i, j]))

    elapsed = time.time() - start_time
    print(f"  cross-encoder rerank done in {elapsed:.1f}s "
          f"({len(pairs) / max(elapsed, 1e-9):.0f} pairs/sec)")
    return by_text_ce, elapsed


def score_texts(texts, labels, uris, concepts, esco_emb, tokenizer, model,
                threshold, margin_min_score, margin_threshold,
                use_cross_encoder, ce_top_k, ce_model_name, kind):
    """Embed texts, find their best ESCO match, optionally cross-encode.

    Returns (by_text, by_text_ce, ce_elapsed). by_text maps each text to
    (canonical, uri, matched_label, score, margin); by_text_ce covers only the
    texts the bi-encoder left unresolved, since the ladder reads a CE score for
    nothing else.
    """
    print(f"  encoding {len(texts):,} {kind}...")
    query_emb = embed(texts, tokenizer, model, label=kind)

    print("  computing similarities...")
    idx, score = top_k_matches(query_emb, esco_emb, 2)
    margin = score[:, 0] - score[:, 1]

    by_text = {}
    for text, i, s, m in zip(texts, idx[:, 0], score[:, 0], margin):
        by_text[text] = (concepts[uris[i]]["canonical"], uris[i], labels[i],
                         float(s), float(m))

    by_text_ce, ce_elapsed = {}, None
    if use_cross_encoder:
        todo = [i for i, text in enumerate(texts)
                if bi_encoder_method(by_text[text][3], by_text[text][4],
                                     threshold, margin_min_score,
                                     margin_threshold) is None]
        print(f"  cross-encoder pool: {len(todo):,} of {len(texts):,} "
              f"{kind} still unresolved")
        if todo:
            tokenizer_ce, model_ce = load_cross_encoder(ce_model_name)
            by_text_ce, ce_elapsed = rerank_with_cross_encoder(
                [texts[i] for i in todo], query_emb[todo], labels, uris,
                concepts, esco_emb, tokenizer_ce, model_ce, top_k=ce_top_k)

    return by_text, by_text_ce, ce_elapsed


# LinkedIn titles carry an employer tail and a requisition code -- 'registered
# nurse at beth israel lahey', 'electrical engineer 23 01066'. Both are dead
# weight in an embedding: the encoder averages them in and drags the title away
# from its occupation.
_COMPANY_TAIL = re.compile(r"\s+at\s+.*$", re.IGNORECASE)
_REQ_CODE = re.compile(r"[\s\-#]*\b\d[\w\d]*\s*$")


def prepare_text(title, use_clean_title=True):
    """What actually gets embedded: the cleaned title, raw as fallback.

    Company-tail and requisition-code stripping always apply. clean_title is
    separable because it was tuned against exact string matching, and a
    transformer already looks past 'Sr.'/'FT'/parentheses to some degree --
    use_clean_title=False measures how much it is really buying here.
    """
    if not isinstance(title, str):
        return ""

    text = _COMPANY_TAIL.sub("", title).strip()
    for _ in range(2):                       # 'engineer 23 01066' has two codes
        text = _REQ_CODE.sub("", text).strip()
    if not text:
        text = title

    if not use_clean_title:
        return text.strip()

    cleaned = rule_based_matching.clean_title(text)
    return cleaned if cleaned else text.strip()


def match_titles(corpus_df=None, sample_n=SAMPLE_N,
                    threshold=SIMILARITY_THRESHOLD,
                    margin_min_score=MARGIN_MIN_SCORE,
                    margin_threshold=MARGIN_THRESHOLD,
                    use_cross_encoder=False, ce_top_k=CE_TOP_K,
                    ce_threshold=CE_THRESHOLD, ce_model_name=CE_MODEL_NAME,
                    use_esco_description=False,
                    esco_desc_max_chars=ESCO_DESC_MAX_CHARS,
                    model_name=MODEL_NAME, use_clean_title=True):
    """Match job titles against ESCO occupations. Returns (df, meta).

    corpus_df, if passed, is used as-is rather than loading a fresh copy, so
    the whole pipeline analyses one consistent dataset.
    """
    if corpus_df is None:
        corpus_df = linkedin.load_all(sample_n=sample_n)
    df = corpus_df.copy()
    print(f"\n{len(df):,} rows, {df['title'].nunique():,} distinct titles")

    print("Loading ESCO occupations...")
    occ_lookup, occupations = esco.load_occupations(DEFAULT_OCCUPATIONS_PATH)

    # ---- 1. exact baseline -------------------------------------------------
    print("Exact matching...")
    exact = [rule_based_matching.canonicalize_title(t, occ_lookup, occupations)
             for t in df["title"]]
    df["esco_occupation"] = pd.array(exact, dtype="string")
    df["match_method"] = pd.array(["exact" if e else None for e in exact],
                                  dtype="string")
    df["esco_score"] = [1.0 if e else np.nan for e in exact]
    df["esco_matched_label"] = df["esco_occupation"]
    n_exact = int(df["esco_occupation"].notna().sum())
    print(f"  exact matches: {n_exact:,} / {len(df):,} "
          f"({100 * n_exact / len(df):.1f}%)")

    # ---- 2. embed only what exact matching missed --------------------------
    unmatched = df["esco_occupation"].isna()
    # The generic-word guard applies to the title side too: a title that reduces
    # to a bare 'manager' names no occupation, but the encoder will still place
    # it next to 'business manager' at 0.84.
    todo_titles = sorted({
        text for text in (prepare_text(t, use_clean_title)
                          for t in df.loc[unmatched, "title"])
        if text and esco.normalize(text) not in rule_based_matching._GENERIC_ROLES
    })
    print(f"  unmatched rows: {int(unmatched.sum()):,} "
          f"({len(todo_titles):,} distinct cleaned titles to encode)")

    tokenizer, model = load_model(model_name)
    labels, uris, esco_emb = esco_index(
        occupations, tokenizer, model, True, EMB_CACHE_PATH,
        EMB_DESC_CACHE_PATH, use_esco_description, esco_desc_max_chars)

    by_text, by_text_ce, ce_elapsed = score_texts(
        todo_titles, labels, uris, occupations, esco_emb, tokenizer, model,
        threshold, margin_min_score, margin_threshold, use_cross_encoder,
        ce_top_k, ce_model_name, "titles")

    # ---- 3. walk the ladder per row ----------------------------------------
    # pd.NA is not usable in a boolean test, so the nullable string columns are
    # unpacked to plain lists with None.
    occupation_col = [o if pd.notna(o) else None
                      for o in df["esco_occupation"].tolist()]
    uri_col = [occ_lookup.get(esco.normalize(str(o))) if o else None
               for o in occupation_col]
    matched_col = [m if pd.notna(m) else None
                   for m in df["esco_matched_label"].tolist()]
    score_col = df["esco_score"].tolist()
    method_col = [m if pd.notna(m) else None
                  for m in df["match_method"].tolist()]
    guess_col = [None] * len(df)
    guess_score_col = [np.nan] * len(df)
    guess_margin_col = [np.nan] * len(df)
    guess_ce_score_col = [np.nan] * len(df)

    for i, (is_unmatched, title) in enumerate(zip(unmatched, df["title"])):
        if not is_unmatched:
            continue
        text = prepare_text(title, use_clean_title)
        hit = by_text.get(text)
        if hit is None:
            continue
        canonical, uri, label, score, margin_val = hit

        # Always record the best guess, so the threshold can be re-tuned later
        # without paying for another encode.
        guess_col[i] = canonical
        guess_score_col[i] = score
        guess_margin_col[i] = margin_val

        method = bi_encoder_method(score, margin_val, threshold,
                                   margin_min_score, margin_threshold)
        if method is None and use_cross_encoder:
            ce_hit = by_text_ce.get(text)
            if ce_hit is not None:
                ce_canonical, ce_uri, ce_label, ce_score, _bi = ce_hit
                guess_ce_score_col[i] = ce_score
                if ce_score >= ce_threshold:
                    canonical, uri, label, score = (ce_canonical, ce_uri,
                                                    ce_label, ce_score)
                    method = "embedding_ce"

        if method is not None:
            occupation_col[i] = canonical
            uri_col[i] = uri
            matched_col[i] = label
            score_col[i] = score
            method_col[i] = method

    df["esco_occupation"] = pd.array(occupation_col, dtype="string")
    df["esco_uri"] = pd.array(uri_col, dtype="string")
    df["esco_matched_label"] = pd.array(matched_col, dtype="string")
    df["esco_score"] = score_col
    df["match_method"] = pd.array(method_col, dtype="string")
    df["esco_best_guess"] = pd.array(guess_col, dtype="string")
    df["esco_best_score"] = guess_score_col
    df["esco_best_margin"] = guess_margin_col
    df["esco_ce_score"] = guess_ce_score_col

    meta = {"n_exact": n_exact, "threshold": threshold,
            "margin_min_score": margin_min_score,
            "margin_threshold": margin_threshold,
            "by_text": by_text, "n_rows": len(df),
            "use_clean_title": use_clean_title,
            "use_cross_encoder": use_cross_encoder, "ce_threshold": ce_threshold,
            "by_text_ce": by_text_ce, "ce_elapsed": ce_elapsed,
            "use_esco_description": use_esco_description}
    return df, meta


def match_skills(corpus_df=None, sample_n=SAMPLE_N,
                 threshold=SKILL_SIMILARITY_THRESHOLD,
                 margin_min_score=SKILL_MARGIN_MIN_SCORE,
                 margin_threshold=SKILL_MARGIN_THRESHOLD,
                 use_cross_encoder=False, ce_top_k=CE_TOP_K,
                 ce_threshold=SKILL_CE_THRESHOLD, ce_model_name=CE_MODEL_NAME,
                 use_esco_description=False,
                 esco_desc_max_chars=ESCO_DESC_MAX_CHARS,
                 model_name=MODEL_NAME, skills_path=DEFAULT_SKILLS_PATH):
    """Match raw skill terms against ESCO skills. Returns (df, meta).

    corpus_df must carry its ORIGINAL raw "skills" column as linkedin.load_all
    produces it -- not rule_based_matching's output, whose "skills" column has
    already been overwritten with canonical labels, losing the raw term text
    this needs to embed.

    The output "skills" column is index-aligned with the input one: slot i
    holds the canonical label for input term i, or None if it did not resolve.
    The raw list is not deduplicated first, so a concept named twice in one
    posting occupies both slots -- meaning the counts are of raw MENTIONS, not
    of distinct concepts per row.
    """
    if corpus_df is None:
        corpus_df = linkedin.load_all(sample_n=sample_n)
    df = corpus_df.copy()
    print(f"\n{len(df):,} rows")

    print("Loading ESCO skills...")
    skill_lookup, skills = esco.load_skills(skills_path)

    # ---- 1. exact baseline, per row ----------------------------------------
    # unresolved_pos carries the slot index of every term still to try, so
    # step 4 can put a match back where it came from instead of appending.
    raw_col, aligned_col, method_lists, unresolved_pos = [], [], [], []
    for raw_skills in df["skills"]:
        raw = list(raw_skills) if isinstance(raw_skills, (list, tuple)) else []
        mapped = rule_based_matching.canonicalize_skills_aligned(
            raw, skill_lookup, skills)
        raw_col.append(raw)
        aligned_col.append(mapped)
        method_lists.append(["exact" if c is not None else None for c in mapped])
        unresolved_pos.append([j for j, c in enumerate(mapped) if c is None])
    n_exact_terms = sum(m.count("exact") for m in method_lists)
    print(f"  exact-matched skill mentions: {n_exact_terms:,}")

    # ---- 2. dedupe dropped terms across all rows, embed each once ----------
    term_to_rows = {}
    for i, positions in enumerate(unresolved_pos):
        for j in positions:
            text = str(raw_col[i][j]).strip()
            if text:
                term_to_rows.setdefault(text, []).append(i)
    todo_terms = sorted(term_to_rows)
    print(f"  distinct dropped terms to encode: {len(todo_terms):,} "
          f"(from {sum(len(v) for v in term_to_rows.values()):,} row occurrences)")

    tokenizer, model = load_model(model_name)
    labels, uris, esco_emb = esco_index(
        skills, tokenizer, model, False, EMB_SKILL_CACHE_PATH,
        EMB_SKILL_DESC_CACHE_PATH, use_esco_description, esco_desc_max_chars)

    by_text, by_text_ce, ce_elapsed = score_texts(
        todo_terms, labels, uris, skills, esco_emb, tokenizer, model,
        threshold, margin_min_score, margin_threshold, use_cross_encoder,
        ce_top_k, ce_model_name, "skill terms")

    # ---- 3. walk the ladder per distinct term ------------------------------
    term_method = {}   # text -> (method|None, canonical|None, score|None)
    for text in todo_terms:
        canonical, _uri, _label, score, margin_val = by_text[text]
        method = bi_encoder_method(score, margin_val, threshold,
                                   margin_min_score, margin_threshold)
        if method is not None:
            term_method[text] = (method, canonical, score)
        elif use_cross_encoder and text in by_text_ce:
            ce_canonical, _u, _l, ce_score, _bi = by_text_ce[text]
            term_method[text] = (("embedding_ce", ce_canonical, ce_score)
                                 if ce_score >= ce_threshold
                                 else (None, None, None))
        else:
            term_method[text] = (None, None, None)

    # ---- 4. write each match back into its own slot ------------------------
    counts = {"exact": n_exact_terms, "embedding": 0,
              "embedding_margin": 0, "embedding_ce": 0}
    dropped_terms_col = []
    for i in range(len(df)):
        leftover = []
        for j in unresolved_pos[i]:
            term = raw_col[i][j]
            text = str(term).strip()
            method, canonical, _score = (term_method.get(text, (None, None, None))
                                         if text else (None, None, None))
            if method is None:
                leftover.append(term)
                continue
            aligned_col[i][j] = canonical
            method_lists[i][j] = method
            counts[method] += 1
        dropped_terms_col.append(leftover)

    df["skills"] = aligned_col
    df["skill_match_methods"] = method_lists
    df["skills_dropped_terms"] = dropped_terms_col
    df["n_skills_exact"] = [m.count("exact") for m in method_lists]
    df["n_skills_embedding"] = [m.count("embedding") for m in method_lists]
    df["n_skills_embedding_margin"] = [m.count("embedding_margin") for m in method_lists]
    df["n_skills_embedding_ce"] = [m.count("embedding_ce") for m in method_lists]

    meta = {"n_exact_terms": n_exact_terms, "threshold": threshold,
            "margin_min_score": margin_min_score,
            "margin_threshold": margin_threshold,
            "by_text": by_text, "by_text_ce": by_text_ce,
            "term_to_rows": term_to_rows, "term_method": term_method,
            "counts": counts, "n_rows": len(df),
            "use_cross_encoder": use_cross_encoder, "ce_threshold": ce_threshold,
            "ce_elapsed": ce_elapsed, "use_esco_description": use_esco_description}
    return df, meta


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_summary(df, meta):
    """Per-rung title match counts, plus the occupations each rung added."""
    n = len(df)
    counts = {m: int((df["match_method"] == m).sum())
              for m in ("exact", "embedding", "embedding_margin", "embedding_ce")}
    n_matched = sum(counts.values())

    print("\n" + "=" * 72)
    clean_note = "" if meta.get("use_clean_title", True) else "  [clean_title OFF]"
    desc_note = "  [+ESCO description]" if meta.get("use_esco_description") else ""
    print(f"TITLES -> ESCO  (threshold {meta['threshold']}, "
          f"margin >={meta.get('margin_threshold', MARGIN_THRESHOLD)} "
          f"@ score>={meta.get('margin_min_score', MARGIN_MIN_SCORE)})"
          f"{clean_note}{desc_note}")
    print("=" * 72)
    for method, count in counts.items():
        if method == "embedding_ce" and not meta.get("use_cross_encoder"):
            continue
        print(f"  {method:<17}{count:>6,}  ({100 * count / n:5.1f}%)")
    print(f"  {'unmatched':<17}{n - n_matched:>6,}  "
          f"({100 * (n - n_matched) / n:5.1f}%)")
    print(f"  TOTAL matched {n_matched:,} / {n:,} ({100 * n_matched / n:.1f}%)  "
          f"-- exact-only baseline was {100 * counts['exact'] / n:.1f}%")
    print(f"  distinct occupations: {df['esco_occupation'].nunique():,}")
    if meta.get("ce_elapsed") is not None:
        print(f"  cross-encoder rerank wall-clock: {meta['ce_elapsed']:.1f}s")

    if n_matched:
        blended = sum(counts[m] * PRECISION_ESTIMATE[m]
                      for m in counts) / n_matched
        print(f"  estimated precision: {100 * blended:.1f}%  "
              f"(PRECISION_ESTIMATE weighted by live counts -- hand-judged "
              f"samples, not ground truth)")

    for method, note in (("embedding", ""),
                         ("embedding_margin", " (confident but sub-threshold)"),
                         ("embedding_ce", " (cross-encoder confirmed)")):
        rows = df[df["match_method"] == method]
        if len(rows):
            print(f"\n  top 20 occupations added by {method}{note}:")
            for occ, c in rows["esco_occupation"].value_counts().head(20).items():
                print(f"    {c:>4}  {occ}")


def print_ce_sample(df, n=30, seed=42):
    """Random sample of embedding_ce title matches, for judging precision by
    eye. This is how CE_THRESHOLD gets calibrated rather than guessed.
    """
    ce = df[df["match_method"] == "embedding_ce"]
    if not len(ce):
        print("\n(no embedding_ce matches to sample)")
        return

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ce), size=min(n, len(ce)), replace=False)
    print("\n" + "=" * 72)
    print(f"RANDOM SAMPLE -- embedding_ce  ({len(ce):,} total rows)")
    print("=" * 72)
    for _, row in ce.iloc[idx].iterrows():
        print(f"  ce {row['esco_score']:.3f}  "
              f"{str(row['title'])[:44]:46} -> {row['esco_occupation']}"
              f"  [via {row['esco_matched_label']!r}]")


def print_unmatched(df, threshold, n=40):
    """Titles that did not make it in, closest misses first.

    Two reasons a title lands here, and both are shown: scored but below
    threshold (nudging the threshold down would catch it), or never scored at
    all (filtered out, e.g. it reduced to a bare generic role word).
    """
    unmatched = df[df["match_method"].isna()]
    if not len(unmatched):
        print("\nEverything matched.")
        return

    scored = unmatched[unmatched["esco_best_guess"].notna()].sort_values(
        "esco_best_score", ascending=False)
    unscored = unmatched[unmatched["esco_best_guess"].isna()]

    print("\n" + "-" * 72)
    print(f"UNMATCHED  {len(unmatched):,} / {len(df):,} rows (threshold {threshold})")
    print("-" * 72)
    print(f"  scored but below threshold: {len(scored):,}")
    print(f"  never scored at all:        {len(unscored):,}")

    if len(scored):
        print(f"\n  closest {min(n, len(scored))} misses:")
        for row in scored.head(n).itertuples(index=False):
            print(f"    {row.esco_best_score:.3f}  {row.title[:44]:46} "
                  f"-> {row.esco_best_guess}")

    if len(unscored):
        print(f"\n  sample of {min(n, len(unscored))} titles with no guess at all:")
        for title in unscored["title"].head(n):
            print(f"    {str(title)[:70]}")


def print_calibration(meta, bands=(0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
                      per_band=8, seed=0,
                      margin_candidates=((0.60, 0.10), (0.65, 0.10),
                                         (0.70, 0.15), (0.70, 0.20))):
    """Score and margin distributions plus sampled pairs, so a threshold is
    chosen on evidence. Works on either matcher's meta.

    margin_candidates is a grid of (min_score, margin_threshold) pairs for the
    embedding_margin rung, reported with the sample pairs each would newly
    accept.
    """
    by_text = meta["by_text"]
    if not by_text:
        print("\n(nothing to calibrate -- every text matched exactly)")
        return

    scores = np.array([v[3] for v in by_text.values()])
    margins = np.array([v[4] for v in by_text.values()])
    texts = list(by_text.keys())

    print("\n" + "=" * 72)
    print(f"CALIBRATION -- {len(scores):,} distinct unmatched texts scored")
    print("=" * 72)
    print("  score deciles:")
    for q in range(0, 101, 10):
        print(f"    p{q:<3} {np.percentile(scores, q):.3f}")

    print("\n  texts matched at each threshold:")
    for t in bands:
        k = int((scores >= t).sum())
        print(f"    >= {t:.2f}   {k:>6,}  ({100 * k / len(scores):5.1f}% of unmatched)")

    rng = np.random.default_rng(seed)
    print("\n  sample pairs per score band (judge precision by eye):")
    edges = list(bands) + [1.01]
    for lo, hi in zip(edges, edges[1:]):
        idx = np.where((scores >= lo) & (scores < hi))[0]
        if not len(idx):
            continue
        print(f"\n  --- {lo:.2f} to {hi:.2f}  ({len(idx):,} texts) ---")
        for i in rng.choice(idx, size=min(per_band, len(idx)), replace=False):
            canonical, _uri, label, score = by_text[texts[i]][:4]
            via = "" if label.lower() == canonical.lower() else f"  [via {label!r}]"
            print(f"    {score:.3f}  {texts[i][:38]:40} -> {canonical}{via}")

    sub = scores < meta.get("threshold", max(bands))

    print("\n" + "-" * 72)
    print("  margin deciles (best - runner-up), sub-threshold texts only:")
    print("-" * 72)
    if sub.any():
        for q in range(0, 101, 10):
            print(f"    p{q:<3} {np.percentile(margins[sub], q):.3f}")

    print("\n  embedding_margin candidate grid -- how many sub-threshold texts "
          "each (min_score, margin) pair would newly accept:")
    for min_score, margin_thr in margin_candidates:
        idx = np.where(sub & (scores >= min_score) & (margins >= margin_thr))[0]
        print(f"\n  --- score>={min_score:.2f} & margin>={margin_thr:.2f}  "
              f"({len(idx):,} texts) ---")
        for i in rng.choice(idx, size=min(per_band, len(idx)), replace=False):
            canonical, _uri, label, score, margin_val = by_text[texts[i]]
            via = "" if label.lower() == canonical.lower() else f"  [via {label!r}]"
            print(f"    score {score:.3f}  margin {margin_val:.3f}  "
                  f"{texts[i][:30]:32} -> {canonical}{via}")


def print_skill_summary(df, meta):
    """Per-rung skill counts, from meta["counts"] since skills are list-valued.

    These are raw mention counts: a posting naming one concept twice counts it
    twice.
    """
    counts = meta["counts"]
    n_matched = sum(counts.values())

    print("\n" + "=" * 72)
    desc_note = "  [+ESCO description]" if meta.get("use_esco_description") else ""
    print(f"SKILLS -> ESCO  (threshold {meta['threshold']}, "
          f"margin >={meta['margin_threshold']} @ score>={meta['margin_min_score']})"
          f"{desc_note}")
    print("=" * 72)
    for method in ("exact", "embedding", "embedding_margin", "embedding_ce"):
        pct = f"({100 * counts[method] / n_matched:5.1f}%)" if n_matched else ""
        print(f"  {method:<17}{counts[method]:>6,}  {pct}")
    print(f"  TOTAL kept mentions: {n_matched:,}  "
          f"-- exact-only baseline was {counts['exact']:,}")

    kept = pd.Series([s for skills in df["skills"] for s in skills if s is not None])
    print(f"  distinct canonical skills kept: {kept.nunique():,}")
    if meta.get("ce_elapsed") is not None:
        print(f"  cross-encoder rerank wall-clock: {meta['ce_elapsed']:.1f}s")

    for method in ("embedding", "embedding_margin", "embedding_ce"):
        added = pd.Series([s for skills, methods in
                           zip(df["skills"], df["skill_match_methods"])
                           for s, m in zip(skills, methods)
                           if m == method and s is not None])
        if len(added):
            print(f"\n  top 20 skills added by {method}:")
            for skill, c in added.value_counts().head(20).items():
                print(f"    {c:>4}  {skill}")


def print_skill_ce_sample(meta, n=30, seed=42):
    """Random sample of embedding_ce skill matches, at term level."""
    ce_terms = [t for t, (method, _c, _s) in meta["term_method"].items()
                if method == "embedding_ce"]
    if not ce_terms:
        print("\n(no embedding_ce skill matches to sample)")
        return

    rng = np.random.default_rng(seed)
    print("\n" + "=" * 72)
    print(f"RANDOM SAMPLE -- embedding_ce skills  ({len(ce_terms):,} total terms)")
    print("=" * 72)
    for text in rng.choice(ce_terms, size=min(n, len(ce_terms)), replace=False):
        canonical, _uri, label, ce_score, _bi = meta["by_text_ce"][text]
        via = "" if label.lower() == canonical.lower() else f"  [via {label!r}]"
        print(f"  ce {ce_score:.3f}  {text[:44]:46} -> {canonical}{via}")


def print_skill_unmatched(meta, n=40):
    """Dropped skill terms that never matched, closest misses first.

    Simpler than print_unmatched: every dropped term gets a bi-encoder score,
    since there is no generic-word filter on the skill index.
    """
    unmatched = [t for t, (method, _c, _s) in meta["term_method"].items()
                 if method is None]
    if not unmatched:
        print("\nEverything matched.")
        return

    unmatched.sort(key=lambda t: meta["by_text"][t][3], reverse=True)
    print("\n" + "-" * 72)
    print(f"UNMATCHED SKILL TERMS  {len(unmatched):,} / {len(meta['by_text']):,} "
          f"distinct dropped terms (threshold {meta['threshold']})")
    print("-" * 72)
    for text in unmatched[:n]:
        canonical, _uri, _label, score, _margin = meta["by_text"][text]
        print(f"    {score:.3f}  {text[:44]:46} -> {canonical}")
