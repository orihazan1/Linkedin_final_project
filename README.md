# Job postings → ESCO

Maps a dump of 1.35M LinkedIn job postings onto the [ESCO](https://esco.ec.europa.eu/)
classification: each raw job title becomes an ESCO occupation, and each raw
skill term becomes an ESCO skill. The output is one CSV of postings described
entirely in a closed, controlled vocabulary.

Raw postings say `Sr. Software Engineer II (Remote)` and `ms excel`. ESCO says
`software engineer` and `use spreadsheets software`. Bridging that gap is the
whole problem, and exact string matching only gets about a quarter of the way.

## How it works

The pipeline runs four stages, each building on the last:

| Stage | Module | What it does |
|---|---|---|
| 1. Load | `src/loading/linkedin.py` | Reads the dump, joins postings to their skill lists, caches the result |
| 2. Rule-based | `src/canonicalization/rule_based_matching.py` | Exact match on a normalized form, after peeling seniority/employment-type noise off each title |
| 3. Titles | `src/canonicalization/semantic_matching.py` | Sentence embeddings + a cross-encoder rerank for what stage 2 missed |
| 4. Skills | `src/canonicalization/semantic_matching.py` | The same ladder over distinct unmatched skill terms |

Stages 3 and 4 climb an acceptance ladder and stop at the first rung a text
clears. Every match records which rung it came from, in `match_method` /
`skill_match_methods`:

- `exact` — matched a normalized ESCO label outright.
- `embedding` — nearest ESCO label at or above `SIMILARITY_THRESHOLD` (0.90).
- `embedding_margin` — below that, but decisively clear of the runner-up label.
  A wide margin means the model picked one label confidently, which is a
  different signal than simply lowering the absolute threshold.
- `embedding_ce` — still unmatched, but a cross-encoder scored one of its top-5
  bi-encoder candidates above `CE_THRESHOLD`. This rung catches shared-word
  domain drift: the bi-encoder puts *Employee Communications Manager* next to
  *telecommunications manager* at 0.90, because it pools each side into its own
  vector; a cross-encoder reads both as one sequence and rejects it.

Thresholds live at the top of `semantic_matching.py`. Only
`SIMILARITY_THRESHOLD` was derived from real per-band precision sampling; the
rest were set by eye and are cruder. `print_calibration()` re-derives any of
them: it prints the score and margin distributions plus sampled pairs to judge
by hand, since there is no ground-truth labelling to score against.

## Getting the data

Neither input ships with this repo — together they are about 1.1 GB — and
**both are behind a free signup**.

**1. LinkedIn postings (Kaggle account required).** Create a free account at
[kaggle.com](https://www.kaggle.com/), then go to *Settings → API → Create New
Token* to download `kaggle.json`. Place it at `~/.kaggle/kaggle.json` and:

```bash
pip install kaggle
kaggle datasets download -d asaniczka/1-3m-linkedin-jobs-and-skills-2024 \
  -p "data/Linkedin Jobs & Skills (2024)" --unzip
```

Only `linkedin_job_postings.csv` and `job_skills.csv` are read.
`job_summary.csv` is ignored — it holds the free-text body, which nothing uses,
and it is roughly 5 GB.

**2. ESCO classification (ESCO portal account required).** These are not on
Kaggle. Register at
[esco.ec.europa.eu](https://esco.ec.europa.eu/en/use-esco/download), download
**ESCO v1.2.1, classification, English, CSV**, and unzip it into `data/` so the
folder keeps its published name:

```
data/ESCO dataset - v1.2.1 - classification - en - csv/
    occupations_en.csv
    skills_en.csv
    ISCOGroups_en.csv
```

## Running it locally

```bash
pip install -r requirements.txt
python -m src.main                    # runs the pipeline, writes cache/
python -m src.export_job_skill_esco   # flattens the run into a CSV
```

Edit the configuration block at the top of each file first. `SAMPLE_SIZE`
matters most: start at a few thousand rows to confirm the setup works, then set
it to `None` for the full dump. Each sample size caches to its own file, so a
quick test run cannot overwrite a long one.

A full CPU-only run takes hours; the cross-encoder rerank dominates. With a GPU
it is far quicker — the code uses one automatically when `torch.cuda.is_available()`.

## Running it on Colab

Open `notebooks/pipeline.ipynb` in Colab and set the runtime to a GPU
(*Runtime → Change runtime type → T4 GPU*). The notebook walks through
installing dependencies, fetching both datasets, running the pipeline and
exporting the CSV, and it prints which device it resolved before starting
anything expensive.

Set `PREP_CACHE_DIR` to a Google Drive path (there is a cell for it) if you want
the caches to survive a runtime restart. The ESCO embedding matrices take
minutes to rebuild and are the main thing worth keeping.

## Output

`python -m src.export_job_skill_esco` writes
`data/job_skill_esco_name.csv`: one row per posting whose title resolved to an
ESCO occupation and which produced at least one ESCO skill.

Alongside the occupation it carries the ISCO group at two levels (the 4-digit
group and its 2-digit sub-major), the canonical ESCO skill names, and the
original raw skill terms.

The two skill columns are **index-aligned**: slot *i* of `esco_skills` is what
slot *i* of `job_skills_original` resolved to, left empty when that term matched
nothing, so both cells always hold the same number of slots. Slots are
separated by ` | `, not by commas — 33 ESCO labels contain a comma, so a
comma-joined cell could not be split back apart.
