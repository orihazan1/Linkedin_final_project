"""Where the data and the caches live.

Every other module imports these instead of deriving paths from its own
__file__, so the layout is described once. Both roots can be redirected with an
environment variable, which is how a Colab session points the cache at mounted
Drive (the ESCO embedding matrices take minutes to rebuild and are worth
keeping across restarts).
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("PREP_DATA_DIR", os.path.join(REPO_ROOT, "data"))
CACHE_DIR = os.environ.get("PREP_CACHE_DIR", os.path.join(REPO_ROOT, "cache"))

ESCO_DIR = os.path.join(DATA_DIR,
                        "ESCO dataset - v1.2.1 - classification - en - csv")
LINKEDIN_DIR = os.path.join(DATA_DIR, "Linkedin Jobs & Skills (2024)")
