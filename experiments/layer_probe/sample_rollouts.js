// The policy's own completions for the 'study+policy' probes of probe.py: a uniform random sample of 500 training
// completions from steps 100-199 of each of four GRPO runs whose policies repeat after the answer. Seed 2 of each
// condition, because the repeat probe's GRPO completions (the repetition test) come from other seeds (and rlvr/s3).
//   node experiments/layer_probe/sample_rollouts.js [<repo> [<out.jsonl.gz>]]   (default: in/rollout_sample.jsonl.gz, not committed)
'use strict';
const fs = require('fs'), path = require('path'), zlib = require('zlib'), readline = require('readline');
const repo = process.argv[2] || path.join(__dirname, '..', '..');
const outFile = process.argv[3] || path.join(__dirname, 'in', 'rollout_sample.jsonl.gz');
const RUNS = [['part2_llm', 'kev_only', 2], ['part2_llm', 'rlvr_kev_gate', 2], ['part3b_labels', 'sub_kev', 2], ['part3b_labels', 'sub_kev_recal', 2]];
const PER_RUN = 500, STEPS = [100, 199];

function rng(seed) { let s = seed; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; }
function repeatsAfterAnswer(t, minLen = 20) {   // dcjev/analysis/part2.py repeats_after_answer
  const i = t.indexOf('\\boxed');
  if (i < 0) return false;
  const seen = new Set(t.slice(0, i).split('\n').map(l => l.trim()).filter(l => [...l].length >= minLen));
  return t.slice(i).split('\n').some(l => { const s = l.trim(); return [...s].length >= minLen && seen.has(s); });
}

(async () => {
  const out = [];
  for (const [part, cond, sd] of RUNS) {
    const run = `${cond}/s${sd}`, pool = [];
    const rl = readline.createInterface({ input: fs.createReadStream(path.join(repo, 'experiments', part, 'runs', cond, 's' + sd, 'rollouts.jsonl.gz')).pipe(zlib.createGunzip()), crlfDelay: Infinity });
    let n = -1;
    for await (const l of rl) {
      if (!l) continue;
      n++;
      const r = JSON.parse(l);
      if (r.step < STEPS[0] || r.step > STEPS[1]) continue;
      pool.push({ run, n, step: r.step, id: r.id, k: r.k, v: r.v, kev: r.kev, rep: repeatsAfterAnswer(r.text), ans: r.text.includes('\\boxed'), text: r.text });
    }
    const rnd = rng(1000 + out.length);   // partial Fisher-Yates: the first PER_RUN of a seeded shuffle
    for (let i = 0; i < PER_RUN; i++) { const j = i + Math.floor(rnd() * (pool.length - i)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
    const s = pool.slice(0, PER_RUN).sort((a, b) => a.n - b.n);
    out.push(...s);
    const share = f => (100 * s.filter(f).length / s.length).toFixed(0) + '%';
    console.log(`${run}: ${pool.length} completions in steps ${STEPS.join('-')}, sampled ${s.length}: right ${share(x => x.v)}, repeat ${share(x => x.rep)}, wrong and repeating ${share(x => x.rep && !x.v)}, no \\boxed ${share(x => !x.ans)}`);
  }
  fs.mkdirSync(path.dirname(outFile), { recursive: true });
  fs.writeFileSync(outFile, zlib.gzipSync(out.map(r => JSON.stringify(r)).join('\n') + '\n'));
  console.log(`${out.length} completions -> ${path.relative(process.cwd(), outFile).split(path.sep).join('/')}`);
})();
