#!/bin/bash
# On the Colab session, from /content/work/lp: probe.py extract, then fit. Writes 'started' and, at the end,
# 'finished' with the exit code; logs in extract.log and fit.log. LIMIT=<n> runs a smoke test into out_smoke/.
set -uo pipefail
. /content/work/env.sh
export PYTHONPATH=/content/work/code/DC-Jev-RL
cd /content/work/lp
OUT=out; ARGS=""
if [ -n "${LIMIT:-}" ]; then OUT=out_smoke; ARGS="--limit $LIMIT"; fi
rm -f $OUT.finished; date -u +%FT%TZ > $OUT.started
$PY probe.py extract --data data --math /content/work/data/math_train.jsonl --out $OUT $ARGS > $OUT.extract.log 2>&1 \
  && $PY probe.py fit --out $OUT > $OUT.fit.log 2>&1
echo "exit $? $(date -u +%FT%TZ)" > $OUT.finished
