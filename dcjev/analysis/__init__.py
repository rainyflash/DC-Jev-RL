import gzip
import json
from pathlib import Path


def read_log(run_dir):
    """The run's log.jsonl records; the repository stores the logs gzipped (log.jsonl.gz)."""
    p = Path(run_dir) / "log.jsonl"
    f = open(p) if p.exists() else gzip.open(p.with_suffix(".jsonl.gz"), "rt")
    with f:
        return [json.loads(l) for l in f]
