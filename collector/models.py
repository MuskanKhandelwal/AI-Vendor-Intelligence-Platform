"""Model IDs used by the collectors, in one place.

Providers retire models. Groq removed llama-3.3-70b-versatile with no notice,
which broke the news and EDGAR collectors at once and meant editing three
files that each hardcoded the same string. A model ID is a dependency, and
unlike a pip package it cannot be pinned, so the least it can do is live in
one place.

Overridable by env var so a model can be swapped without a code change.
"""

import os

# Classification and summarisation in the collectors: high volume, low stakes.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# gpt-oss models emit reasoning tokens before the answer; "low" roughly halves
# them with no measured loss of classification quality.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
