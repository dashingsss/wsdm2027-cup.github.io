"""Four standalone baselines with a shared planning interface.

No Retrieval uses only a common reminder. Random, BM25 and Dense Static
select one historical reference per task by default. BM25 and Dense encode
the goal and initial observation; guidance includes historical action returns.
Each method builds its index once per batch and resets task-local state.
"""
