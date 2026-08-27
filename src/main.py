"""Pipeline entry point.

    python -m src.main

This file is a control panel: edit the configuration block below and run it.
The stages themselves live in pipeline.run_pipeline(); see src/pipeline.py for
what they are and why they run in that order.
"""

import sys

from src import pipeline

# ==========================================================================
# CONFIGURATION -- edit these, then run the file
# ==========================================================================

# Rows to pull from the dump. None = every row (1,348,510).
SAMPLE_SIZE = 100

# False re-reads the raw CSVs instead of reusing cache/.
USE_CACHE = False

# True deletes rows whose title never mapped to an ESCO occupation, instead of
# keeping them with <NA>.
DROP_UNMATCHED = False

# Stage 3: embedding-based title matching. Downloads a model on first run.
RUN_TITLE_EMBEDDINGS = True

# Stage 4: embedding-based skill matching. Separate switch because it touches
# every distinct unmatched skill term and is a much bigger job.
RUN_SKILL_EMBEDDINGS = True

# Cross-encoder rerank for stage 4. Roughly an order of magnitude slower than
# the title rerank.
SKILL_CROSS_ENCODER = True

# Write the result to cache/pipeline_output_{SAMPLE_SIZE}_a.pkl, so a later
# analysis can restore a finished run with
# pipeline.load_pipeline_output(SAMPLE_SIZE) instead of re-running every stage.
SAVE_OUTPUT = False

# ==========================================================================


def main():
    try:
        result = pipeline.run_pipeline(
            sample_size=SAMPLE_SIZE,
            use_cache=USE_CACHE,
            drop_unmatched=DROP_UNMATCHED,
            run_title_embeddings=RUN_TITLE_EMBEDDINGS,
            run_skill_embeddings=RUN_SKILL_EMBEDDINGS,
            skill_cross_encoder=SKILL_CROSS_ENCODER,
            save_output=SAVE_OUTPUT,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    pipeline.print_pipeline_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
