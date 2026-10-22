// The layer-wise probes of probe.py against Kev-4B and the AR judge:
//   1. ranking on the Part II judge study (cross-fitted probe scores): AUROC pooled and within problem, with a paired
//      bootstrap over problems of the within-problem AUROC against Kev-4B's 'correct' and 'quality' and the AR judge;
//   2. the repeat probe's interventions (end3 and twice added to the judge study's completions, scored by the fold model
//      that did not see the problem; repetition cut from GRPO final completions, scored by the model fitted on all the
//      training data): the score with repetition minus without it, by verdict, in units of the signal's own right-wrong
//      gap on the judge study (2b: the gap among its completions with a \boxed answer), with a bootstrap over problems
//      (and against Kev-4B's 'correct' on the same pairs);
//   3. the probes fitted on the judge study alone, on the policy's own training completions (sample_rollouts.js):
//      AUROC, and repeating against other completions with a \boxed answer and the same verdict (unpaired: the
//      completions differ in problem), next to Kev-4B's 'correct' on the same completions.
// Kev-4B and the AR judge: experiments/part2_llm/judge_study/scores.jsonl.gz for the ranking, and the repeat probe's
// scores.jsonl.gz for the interventions, as in judge_combos/combos.js.
//   node experiments/layer_probe/analyze.js [<results dir> [<repo>]]   (needs in/rollout_sample.jsonl.gz: run sample_rollouts.js first)
'use strict';
const fs = require('fs'), path = require('path'), zlib = require('zlib');
const dir = process.argv[2] || path.join(__dirname, 'results');
const repo = process.argv[3] || path.join(__dirname, '..', '..');
const readGz = f => zlib.gunzipSync(fs.readFileSync(f)).toString('utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
function groupBy(a, f) { const m = new Map(); for (const x of a) { const k = f(x); if (!m.has(k)) m.set(k, []); m.get(k).push(x); } return m; }
function ranks(a) {
  const idx = a.map((_, i) => i).sort((i, j) => a[i] - a[j]), r = new Array(a.length);
  for (let i = 0; i < idx.length;) {
    let j = i; while (j + 1 < idx.length && a[idx[j + 1]] === a[idx[i]]) j++;
    for (let t = i; t <= j; t++) r[idx[t]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}
function auroc(s, y) {   // as dcjev.metrics: ties count one half
  let n1 = 0; for (const v of y) n1 += v;
  const n0 = y.length - n1; if (!n1 || !n0) return null;
  const r = ranks(s); let sr = 0;
  for (let i = 0; i < y.length; i++) if (y[i]) sr += r[i];
  return (sr - n1 * (n1 + 1) / 2) / (n1 * n0);
}
function rng(seed) { let s = seed; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; }
const qt = (a, p) => { const b = [...a].sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(p * b.length))]; };
const ci = a => [qt(a, 0.025), qt(a, 0.975)];
const B = 1000;

const rows = readGz(path.join(dir, 'scores.jsonl.gz'));
const fit = JSON.parse(fs.readFileSync(path.join(dir, 'fit.json'), 'utf8'));
const extract = JSON.parse(fs.readFileSync(path.join(dir, 'extract.json'), 'utf8'));
const PROBES = Object.keys(fit.probes);
const REFS = {
  kev_c: k => k.kev.correct, kev_q: k => k.kev.quality, 'kev_mean(c,g,q)': k => (k.kev.correct + k.kev.genuine + k.kev.quality) / 3,
  ar: k => typeof k.ar === 'number' ? k.ar : k.ar.correct,
};
const SIGS = [...PROBES, ...Object.keys(REFS)];
const isProbe = s => !(s in REFS);
const js = new Map(readGz(path.join(repo, 'experiments/part2_llm/judge_study/scores.jsonl.gz')).filter(r => r.probe === 'orig').map(r => [r.i, r]));
const kf = r => [r.kind, r.edit, r.source, r.n].join('|');
const rp = new Map(readGz(path.join(repo, 'experiments/part2_llm/repeat_probe/scores.jsonl.gz')).map(r => [kf(r), r]));
const res = { extract, probes: PROBES, lam: Object.fromEntries(PROBES.map(p => [p, fit.probes[p]])) };

// ---- 1. ranking on the judge study ---------------------------------------------------------------------------------
const study = rows.filter(r => r.group === 'probe' && r.kind === 'add' && r.edit === 'orig');
const Y = study.map(r => r.v);
const S = Object.fromEntries(SIGS.map(s => [s, study.map(r => isProbe(s) ? r.p[s] : REFS[s](js.get(r.n)))]));
const P = [...groupBy(study.map((r, i) => ({ r, i })), x => x.r.set + '|' + x.r.id).values()].map(g => g.map(x => x.i));
const mixed = P.filter(m => { const k = m.reduce((s, i) => s + Y[i], 0); return k > 0 && k < m.length; });
const perProblem = Object.fromEntries(SIGS.map(s => [s, mixed.map(m => auroc(m.map(i => S[s][i]), m.map(i => Y[i])))]));
const gapOf = s => mean(S[s].filter((_, i) => Y[i])) - mean(S[s].filter((_, i) => !Y[i]));
res.ranking = Object.fromEntries(SIGS.map(s => [s, { auroc: auroc(S[s], Y), auroc_within: mean(perProblem[s]), gap: gapOf(s) }]));
{
  const VS = ['kev_c', 'kev_q', 'ar'], rnd = rng(2024), mixedIdx = new Map(mixed.map((m, j) => [m, j]));
  const boot = Object.fromEntries(SIGS.map(s => [s, Object.fromEntries(VS.map(v => [v, []]))]));
  for (let b = 0; b < B; b++) {
    const ws = Array.from({ length: P.length }, () => P[Math.floor(rnd() * P.length)]).filter(m => mixedIdx.has(m)).map(m => mixedIdx.get(m));
    const w = Object.fromEntries(SIGS.map(s => [s, mean(ws.map(j => perProblem[s][j]))]));
    for (const s of SIGS) for (const v of VS) boot[s][v].push(w[s] - w[v]);
  }
  for (const s of SIGS) for (const v of VS) res.ranking[s][`d_within_vs_${v}_ci`] = ci(boot[s][v]);
}
// the same among the completions that state a \boxed answer: a judge may rank partly by whether there is an answer
{
  const text = readGz(path.join(repo, 'experiments/part2_llm/judge_study/completions.jsonl.gz')).map(c => c.text);
  const A = study.map(r => text[r.n].includes('\\boxed'));
  const idx = study.map((_, i) => i).filter(i => A[i]);
  const Pa = P.map(m => m.filter(i => A[i])).filter(m => { const k = m.reduce((s, i) => s + Y[i], 0); return k > 0 && k < m.length; });
  res.answered = { completions: idx.length, wrong_without_answer: study.filter((r, i) => !r.v && !A[i]).length / study.filter(r => !r.v).length,
                   right_without_answer: study.filter((r, i) => r.v && !A[i]).length, problems_mixed: Pa.length };
  for (const s of SIGS) {
    const G = S[s].map((x, i) => A[i] ? x : -1);   // gated: a completion without an answer ranks below every other
    res.ranking[s].answered = {
      auroc: auroc(idx.map(i => S[s][i]), idx.map(i => Y[i])), auroc_within: mean(Pa.map(m => auroc(m.map(i => S[s][i]), m.map(i => Y[i])))),
      gap: mean(idx.filter(i => Y[i]).map(i => S[s][i])) - mean(idx.filter(i => !Y[i]).map(i => S[s][i])),
      gated_auroc: auroc(G, Y), gated_auroc_within: mean(mixed.map(m => auroc(m.map(i => G[i]), m.map(i => Y[i])))),
    };
  }
}

// ---- 2. the repeat probe's interventions -----------------------------------------------------------------------------
const byKey = new Map(rows.filter(r => r.group === 'probe').map(r => [kf(r), r]));
const val = (s, r) => isProbe(s) ? r.p[s] : REFS[s](rp.get(kf(r)));
function premium(pairs, sign) {   // pairs [orig row, edited row]; the score with repetition minus without it
  const out = {}, byProblem = groupBy(pairs, p => p[0].id), probs = [...byProblem.keys()];
  for (const v of [0, 1]) {
    const sel = pairs.filter(p => p[0].v === v); if (!sel.length) continue;
    const lab = v ? 'right' : 'wrong', o = {};
    for (const s of SIGS) {
      const d = sel.map(([a, e]) => sign * (val(s, e) - val(s, a)));
      o[s] = { n: d.length, premium: mean(d), per_gap: mean(d) / res.ranking[s].gap, frac_up: d.filter(x => x > 0).length / d.length,
               per_answered_gap: mean(d) / res.ranking[s].answered.gap };
    }
    // bootstrap over problems: per_gap of every signal and its difference from Kev-4B's 'correct'
    const rnd = rng(7 + v), bs = Object.fromEntries(SIGS.map(s => [s, { pg: [], dc: [] }]));
    const selBy = groupBy(sel, p => p[0].id), keys = [...selBy.keys()];
    for (let b = 0; b < B; b++) {
      const pick = Array.from({ length: keys.length }, () => selBy.get(keys[Math.floor(rnd() * keys.length)])).flat();
      const m = Object.fromEntries(SIGS.map(s => [s, mean(pick.map(([a, e]) => sign * (val(s, e) - val(s, a)))) / res.ranking[s].gap]));
      for (const s of SIGS) { bs[s].pg.push(m[s]); bs[s].dc.push(m[s] - m.kev_c); }
    }
    for (const s of SIGS) { o[s].per_gap_ci = ci(bs[s].pg); if (s !== 'kev_c') o[s].d_per_gap_vs_kev_c_ci = ci(bs[s].dc); }
    out[lab] = o;
  }
  out.problems = probs.length;
  return out;
}
const pairsOf = edit => rows.filter(r => r.group === 'probe' && r.edit === edit).map(r => [byKey.get(kf({ ...r, edit: 'orig' })), r]).filter(p => p[0]);
res.repetition = { add_end3: premium(pairsOf('end3'), 1), add_twice: premium(pairsOf('twice'), 1), remove_cut: premium(pairsOf('cut'), -1) };
res.repetition.remove_cut_by_source = {};
for (const [src, ps] of groupBy(pairsOf('cut'), p => p[0].source)) {
  res.repetition.remove_cut_by_source[src] = {};
  for (const v of [0, 1]) {
    const sel = ps.filter(p => p[0].v === v);
    const d = s => mean(sel.map(([a, e]) => val(s, a) - val(s, e)));
    res.repetition.remove_cut_by_source[src][v ? 'right' : 'wrong'] = Object.fromEntries(SIGS.map(s => [s, { n: sel.length, per_gap: d(s) / res.ranking[s].gap,
                                                                                                                 per_answered_gap: d(s) / res.ranking[s].answered.gap }]));
  }
}

// ---- 3. the policy's own training completions -------------------------------------------------------------------------
if (!fs.existsSync(path.join(__dirname, 'in', 'rollout_sample.jsonl.gz'))) throw new Error('in/rollout_sample.jsonl.gz is missing: run sample_rollouts.js first');
const sample = new Map(readGz(path.join(__dirname, 'in', 'rollout_sample.jsonl.gz')).map(r => [r.run + '|' + r.n, r]));
const pol = rows.filter(r => r.group === 'policy').map(r => ({ ...r, ...sample.get(r.source + '|' + r.n), p: r.p }));
const POLS = [...PROBES.filter(p => p.endsWith('_study')), 'kev_c'];
const pv = (s, r) => s === 'kev_c' ? r.kev.correct : r.p[s];
res.policy = { n: pol.length, right: mean(pol.map(r => r.v)), repeating: mean(pol.map(r => +r.rep)), signals: {} };
for (const s of POLS) {
  const ans = pol.filter(r => r.ans), y = ans.map(r => r.v);
  const g = mean(ans.filter(r => r.v).map(r => pv(s, r))) - mean(ans.filter(r => !r.v).map(r => pv(s, r)));
  const o = { auroc: auroc(pol.map(r => pv(s, r)), pol.map(r => r.v)), auroc_answered: auroc(ans.map(r => pv(s, r)), y), gap_answered: g };
  for (const v of [0, 1]) {
    const R = ans.filter(r => r.v === v && r.rep), N = ans.filter(r => r.v === v && !r.rep);
    o[v ? 'right' : 'wrong'] = { rep: R.length, other: N.length, premium: mean(R.map(r => pv(s, r))) - mean(N.map(r => pv(s, r))), per_gap: (mean(R.map(r => pv(s, r))) - mean(N.map(r => pv(s, r)))) / g };
  }
  res.policy.signals[s] = o;
}
fs.writeFileSync(path.join(dir, 'analysis.json'), JSON.stringify(res, null, 1));

// ---- printed summary --------------------------------------------------------------------------------------------------
const f = (x, d = 3) => x === null || x === undefined || Number.isNaN(x) ? '-' : x.toFixed(d);
const sg = x => (x >= 0 ? '+' : '') + f(x);
const cis = a => `[${f(a[0])}, ${f(a[1])}]`;
console.log(`${extract.texts} texts, ${extract.tokens} tokens, ${f(extract.seconds / 60, 1)} min on ${extract.gpu}; policy right answers equal to the dataset's: ${f(extract.policy_right_boxed_equals_answer)}`);
console.log('\n1. judge study (4000 completions; within = mean over ' + mixed.length + ' problems with both verdicts)');
console.log('signal                    AUROC  within  gap    d_within vs kev_c [95%]     vs kev_q [95%]              vs ar [95%]');
for (const s of SIGS) {
  const r = res.ranking[s], d = v => s === v ? '—'.padEnd(26) : `${sg(r.auroc_within - res.ranking[v].auroc_within)} ${cis(r[`d_within_vs_${v}_ci`])}`.padEnd(26);
  console.log(`${s.padEnd(24)}  ${f(r.auroc)}  ${f(r.auroc_within)}  ${f(r.gap)}  ${d('kev_c')}  ${d('kev_q')}  ${d('ar')}`);
}
console.log(`\n1b. only completions with a \\boxed answer (${res.answered.completions}; ${f(100 * res.answered.wrong_without_answer, 0)}% of the wrong ones have none; within over ${res.answered.problems_mixed} problems)`);
console.log(`   (right completions without a \\boxed answer: ${res.answered.right_without_answer}); gated = all 4000 with those without an answer ranked last`);
for (const s of SIGS) {
  const a = res.ranking[s].answered, c = res.repetition.remove_cut;
  console.log(`${s.padEnd(24)}  AUROC ${f(a.auroc)}  within ${f(a.auroc_within)}  gap ${f(a.gap)} | gated AUROC ${f(a.gated_auroc)} within ${f(a.gated_auroc_within)}`
    + ` | GRPO cut / answered gap: wrong ${f(c.wrong[s].premium / a.gap, 2)} right ${f(c.right[s].premium / a.gap, 2)}`);
}
console.log('\n2. repetition premium / own right-wrong gap: wrong | right   [95% over problems]');
for (const [lab, R] of Object.entries(res.repetition)) {
  if (lab === 'remove_cut_by_source') continue;
  console.log(`  ${lab}: pairs wrong ${R.wrong ? R.wrong.kev_c.n : 0}, right ${R.right.kev_c.n}`);
  for (const s of SIGS) console.log(`    ${s.padEnd(24)} ${R.wrong ? `${f(R.wrong[s].per_gap, 2)} ${cis(R.wrong[s].per_gap_ci)}`.padEnd(24) : '-'.padEnd(24)} | ${f(R.right[s].per_gap, 2)} ${cis(R.right[s].per_gap_ci)}`
    + (s !== 'kev_c' && R.wrong ? `   minus kev_c (wrong) ${cis(R.wrong[s].d_per_gap_vs_kev_c_ci)}` : ''));
}
console.log('  remove_cut by source, wrong (n): ' + Object.entries(res.repetition.remove_cut_by_source).map(([src, o]) => `${src} (${o.wrong ? o.wrong.kev_c.n : 0})`).join(', '));
for (const s of SIGS) console.log(`    ${s.padEnd(24)} ` + Object.values(res.repetition.remove_cut_by_source).map(o => f(o.wrong ? o.wrong[s].per_gap : null, 2).padStart(6)).join(' '));
console.log('\n2b. the same per right-wrong gap among the completions with a \\boxed answer (1b): end3 wrong right | twice wrong right | cut wrong right | cut by source, wrong (as above)');
for (const s of SIGS) {
  const R = res.repetition, g = x => f(x, 2).padStart(6), w = o => o.wrong ? o.wrong[s].per_answered_gap : null;
  console.log(`    ${s.padEnd(24)} ${g(w(R.add_end3))} ${g(R.add_end3.right[s].per_answered_gap)} | ${g(w(R.add_twice))} ${g(R.add_twice.right[s].per_answered_gap)} | `
    + `${g(w(R.remove_cut))} ${g(R.remove_cut.right[s].per_answered_gap)} | ` + Object.values(R.remove_cut_by_source).map(o => g(w(o))).join(' '));
}
console.log(`\n3. policy completions (${res.policy.n}, ${f(100 * res.policy.right, 0)}% right, ${f(100 * res.policy.repeating, 0)}% repeating), probes fitted on the judge study only`);
console.log('signal                    AUROC  answered  | repeating minus other answered, / gap: wrong (n rep/other) | right');
for (const [s, o] of Object.entries(res.policy.signals))
  console.log(`${s.padEnd(24)}  ${f(o.auroc)}  ${f(o.auroc_answered)}     | ${f(o.wrong.per_gap, 2)} (${o.wrong.rep}/${o.wrong.other}) | ${f(o.right.per_gap, 2)} (${o.right.rep}/${o.right.other})`);
console.log(`\nwritten to ${path.relative(process.cwd(), path.join(dir, 'analysis.json')).split(path.sep).join('/')}`);
