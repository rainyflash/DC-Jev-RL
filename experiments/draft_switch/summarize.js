// Summaries of results/runs.json (written by analyze.js) for the report: the rise of each switching run, the selection
// on drafting during it against a permutation null, how well selection predicts the change in the draft share, early
// features of the switching runs against the others, subjects and difficulty at the rise, and the final policies'
// drafts against their other completions within a problem.
//
//   node experiments/draft_switch/summarize.js [<results dir>]
//
// By default the results/ next to this file; run analyze.js first, which writes runs.json there.
'use strict';
const fs = require('fs'), path = require('path');
const dir = process.argv[2] || path.join(__dirname, 'results');
const R = JSON.parse(fs.readFileSync(path.join(dir, 'runs.json'), 'utf8'));
const out = {};

const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
const sum = a => a.reduce((s, x) => s + x, 0);
const logit = (p, n) => { const k = Math.round(p * n), q = (k + 0.5) / (n + 1); return Math.log(q / (1 - q)); };
const pct = (x, d = 0) => x === null || x === undefined ? '-' : (100 * x).toFixed(d) + '%';
const f = (x, d = 2) => x === null || x === undefined ? '-' : x.toFixed(d);
function pearson(x, y) {
  const mx = mean(x), my = mean(y);
  let a = 0, b = 0, c = 0;
  for (let i = 0; i < x.length; i++) { a += (x[i] - mx) * (y[i] - my); b += (x[i] - mx) ** 2; c += (y[i] - my) ** 2; }
  return { r: a / Math.sqrt(b * c), slope: a / b, intercept: my - (a / b) * mx };
}
const SW = R.filter(r => r.style === 'draft'), OT = R.filter(r => r.style !== 'draft');

// 1. The rise: first steps after the trough at which the trailing 5-step share reaches 20% and 50%, the largest fall
// after reaching 50%, and the selection on drafting over the 20 steps up to the 50% crossing.
out.rise = [];
for (const r of SW) {
  const s = r.series, p = s.map(x => x.draft);
  const m5 = p.map((_, t) => t < 4 ? null : mean(p.slice(t - 4, t + 1)));
  const tr = r.takeoff.trough.step;
  const after = lv => { for (let t = tr; t < s.length; t++) if (m5[t] !== null && m5[t] >= lv) return t; return null; };
  const t20 = after(0.2), t50 = after(0.5);
  let maxFall = 0;
  if (t50 !== null) { let peak = 0; for (let t = t50; t < s.length; t++) { peak = Math.max(peak, m5[t]); maxFall = Math.max(maxFall, peak - m5[t]); } }
  const end = t50 !== null ? t50 : s.length - 1, lo = Math.max(0, end - 19);
  const seg = s.slice(lo, end + 1);
  const S = sum(seg.map(x => x.S_draft)), sd = Math.sqrt(sum(seg.map(x => x.S_draft_null_sd ** 2)));
  const mixed = sum(seg.map(x => x.mixed)), dr = mixed ? sum(seg.map(x => (x.d_right || 0) * x.mixed)) / mixed : null;
  out.rise.push({
    run: r.run, trough_step: tr, trough_share: r.takeoff.trough.share, step_20: t20, step_50: t50,
    steps_20_to_50: t20 !== null && t50 !== null ? t50 - t20 : null, max_fall_after_50: maxFall,
    share_before: mean(p.slice(lo, lo + 5)), share_at_end: mean(p.slice(end - 4, end + 1)),
    rise_steps: [lo, end], S_sum: S, S_null_sd: sd, z: S / sd, d_right: dr, mixed_groups: mixed,
  });
}

// 2. Does selection predict the change? Over all 10-step windows of all runs: the change in the logit of the draft share
// from one window to the next against the summed selection differential in the first window.
const pts = [];
for (const r of R) {
  const s = r.series;
  for (let w = 0; w + 20 <= s.length; w += 10) {
    const a = s.slice(w, w + 10), b = s.slice(w + 10, w + 20);
    const pa = mean(a.map(x => x.draft)), pb = mean(b.map(x => x.draft)), n = 10 * a[0].n;
    pts.push({ run: r.run, style: r.style, w, S: sum(a.map(x => x.S_draft)), sd: Math.sqrt(sum(a.map(x => x.S_draft_null_sd ** 2))), dl: logit(pb, n) - logit(pa, n), pa, pb });
  }
}
const fit = (P) => pearson(P.map(x => x.S), P.map(x => x.dl));
out.response = {
  all: { n: pts.length, ...fit(pts) },
  early: { n: pts.filter(x => x.w < 50).length, ...fit(pts.filter(x => x.w < 50)) },
  late_others: { n: pts.filter(x => x.w >= 50 && x.style !== 'draft').length, ...fit(pts.filter(x => x.w >= 50 && x.style !== 'draft')) },
};
const b = out.response.early.slope, a0 = out.response.early.intercept;
out.rise_windows = pts.filter(x => x.style === 'draft' && x.dl > 0.7).map(x => ({ run: x.run, w: x.w, share: [x.pa, x.pb], dlogit: x.dl, S: x.S, z: x.S / x.sd, predicted_dlogit: a0 + b * x.S }));
// the largest change per window in the runs that never switched, for scale
out.others_max_dlogit = Math.max(...pts.filter(x => x.style !== 'draft' && x.w >= 50).map(x => x.dl));

// 3. Early features: switching runs against the others, steps 0-49 and 50-99, with an exact permutation test of the
// difference in means (all C(26,6) ways to pick the 6), and against the other seeds of the same condition.
const FEATS = {
  draft: s => mean(s.map(x => x.draft)), S_draft: s => sum(s.map(x => x.S_draft)), d_right: s => {
    const m = sum(s.map(x => x.mixed)); return m ? sum(s.map(x => (x.d_right || 0) * x.mixed)) / m : null;
  },
  lat_nondraft: s => mean(s.map(x => x.lat_nondraft)), fm100: s => mean(s.map(x => x.fm100)), right: s => mean(s.map(x => x.right)),
  trunc: s => mean(s.map(x => x.trunc)), tokens: s => mean(s.map(x => x.tokens)), groups_all_right: s => mean(s.map(x => x.groups_all_right)),
  grad_norm: s => mean(s.map(x => x.grad_norm)), adv_scale: s => mean(s.map(x => x.adv_scale)),
};
function combos(n, k) {   // all k-subsets of 0..n-1 as bitmasks
  const res = [];
  const rec = (start, left, mask) => { if (!left) { res.push(mask); return; } for (let i = start; i <= n - left; i++) rec(i + 1, left - 1, mask | (1 << i)); };
  rec(0, k, 0); return res;
}
const subsets = combos(R.length, SW.length);
const isSw = R.map(r => r.style === 'draft');
out.early = [];
for (const [lo, hi] of [[0, 49], [50, 99]]) {
  for (const [name, fn] of Object.entries(FEATS)) {
    const v = R.map(r => fn(r.series.slice(lo, hi + 1)));
    if (v.some(x => x === null || Number.isNaN(x))) continue;
    const tot = sum(v), k = SW.length, n = R.length;
    const diffOf = mask => { let s1 = 0; for (let i = 0; i < n; i++) if (mask & (1 << i)) s1 += v[i]; return s1 / k - (tot - s1) / (n - k); };
    let obsMask = 0; isSw.forEach((x, i) => { if (x) obsMask |= 1 << i; });
    const obs = diffOf(obsMask);
    let ge = 0; for (const m of subsets) if (Math.abs(diffOf(m)) >= Math.abs(obs) - 1e-12) ge++;
    // matched: each switching run against the mean of the non-switching seeds of its condition
    const matched = SW.map(r => {
      const mates = OT.filter(o => o.cond === r.cond);
      if (!mates.length) return null;
      return v[R.indexOf(r)] - mean(mates.map(o => v[R.indexOf(o)]));
    }).filter(x => x !== null);
    out.early.push({ steps: [lo, hi], feature: name, switching_mean: mean(v.filter((_, i) => isSw[i])), others_mean: mean(v.filter((_, i) => !isSw[i])), diff: obs, p_perm: ge / subsets.length, matched_diffs: matched });
  }
}

// 4. Subjects, and difficulty, in the 25-step window in which each switching run first reaches 20% (and the window
// before). A problem's difficulty is the accuracy of the same-seed runs that did not switch on the same problem at the
// same step (mates as in 7 below), so it does not depend on the switching run's own drafts.
const BINS = [[0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.01]];
const matesOf = r => R.filter(z => z.seed === r.seed && z.style !== 'draft' && !['kev_only', 'tree_kev'].includes(z.cond));
out.onset_composition = SW.map(r => {
  const t20 = out.rise.find(x => x.run === r.run).step_20;
  const i = Math.floor(t20 / 25), c = r.composition[i], prev = r.composition[i - 1];
  const M = matesOf(r).map(m => new Map(m.problems.map(([s, id, v]) => [s + '|' + id, v])));
  const byBin = (lo, hi) => BINS.map(([a, b]) => {
    const d = r.problems.filter(([s, id]) => s >= lo && s <= hi).map(([s, id, v, dr]) => {
      const o = M.map(m => m.get(s + '|' + id)); return o.every(y => y !== undefined) ? [mean(o), dr] : null;
    }).filter(x => x && x[0] >= a && x[0] < b);
    return { bin: [a, Math.min(b, 1)], problems: d.length, draft: mean(d.map(x => x[1])) };
  });
  return {
    run: r.run, window: c.steps, subj: Object.fromEntries(Object.entries(c.subj).map(([k, v]) => [k, [prev.subj[k] ? prev.subj[k].draft : null, v.draft, v.n]])),
    difficulty_before: byBin(prev.steps[0], prev.steps[1]), difficulty: byBin(c.steps[0], c.steps[1]),
  };
});

// 5. The final policies: drafts against the other completions of the same problem on the sampled sets
// (AIME24, AIME25, AMC23; 16 completions per problem), pooled over runs, with a bootstrap over problems.
function bootstrap(W, B = 10000) {
  let s = 12345; const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
  const m = [];
  for (let k = 0; k < B; k++) { let t = 0; for (let i = 0; i < W.length; i++) t += W[Math.floor(rnd() * W.length)].diff; m.push(t / W.length); }
  m.sort((x, y) => x - y); return [m[Math.floor(0.025 * B)], m[Math.floor(0.975 * B)]];
}
const SETS = ['aime24', 'aime25', 'amc23'];
out.final_within = {};
for (const [k, rs] of [['switching', SW], ['others', OT]]) {
  const W = rs.flatMap(r => SETS.flatMap(s => r.final_split[s].within));
  const acc = mean(rs.map(r => mean(SETS.map(s => r.final_split[s].acc))));
  out.final_within[k] = { runs: rs.length, problems: W.length, diff: mean(W.map(x => x.diff)), ci: bootstrap(W), mean_acc_sampled_sets: acc };
}

// 6. Over the rise windows of all six runs pooled: drafts minus others in the verifier's verdict within a group, with a
// bootstrap over groups (stratified by run), and the summed selection against its null.
function bootstrapStrata(strata, B = 10000) {   // strata: arrays of values; resample within each, mean of all
  let s = 777; const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
  const n = sum(strata.map(x => x.length)), m = [];
  for (let k = 0; k < B; k++) { let t = 0; for (const x of strata) for (let i = 0; i < x.length; i++) t += x[Math.floor(rnd() * x.length)]; m.push(t / n); }
  m.sort((x, y) => x - y); return [m[Math.floor(0.025 * B)], m[Math.floor(0.975 * B)]];
}
{
  const strata = out.rise.map(x => R.find(r => r.run === x.run).series.slice(x.rise_steps[0], x.rise_steps[1] + 1).flatMap(s => s.d_right_groups));
  const S = sum(out.rise.map(x => x.S_sum)), sd = Math.sqrt(sum(out.rise.map(x => x.S_null_sd ** 2)));
  out.rise_pooled = { groups: sum(strata.map(x => x.length)), d_right: sum(strata.map(sum)) / sum(strata.map(x => x.length)), ci: bootstrapStrata(strata), S_sum: S, S_null_sd: sd, z: S / sd };
}

// 7. Training accuracy on the same problems: each switching run minus the mean of the same-seed runs that did not
// switch (same problems in the same order), 40 steps before the rise, during it and 40 steps after; bootstrap over
// problems. Mates exclude kev_only (its policies degrade) and tree_kev (its roots are chosen to be right more often).
// tree_kev/s1 switched at the very end and has no "after".
out.paired_training = [];
for (const x of out.rise) {
  const r = R.find(z => z.run === x.run);
  if (r.cond === 'tree_kev') continue;
  const mates = R.filter(z => z.seed === r.seed && z.style !== 'draft' && !['kev_only', 'tree_kev'].includes(z.cond));
  const acc = run => new Map(run.problems.map(([s, id, v]) => [s + '|' + id, v]));
  const mine = acc(r), M = mates.map(acc);
  const seg = (a, b) => {
    const d = [];
    for (const [k, v] of mine) { const s = +k.split('|')[0]; if (s < a || s > b) continue; const o = M.map(m => m.get(k)); if (o.every(y => y !== undefined)) d.push(v - mean(o)); }
    return { steps: [a, b], problems: d.length, diff: mean(d), ci: bootstrapStrata([d]) };
  };
  const lo = x.rise_steps[0], t50 = x.step_50;
  out.paired_training.push({ run: x.run, mates: mates.map(m => m.run), before: seg(Math.max(0, lo - 40), lo - 1), rise: seg(lo, t50), after: seg(t50 + 1, Math.min(199, t50 + 40)) });
}

// 8. Greedy MATH-500 and val (periodic evaluations every 50 steps): the change over the interval that contains the 50%
// crossing, against the same-seed runs that did not switch (without kev_only) over the same interval, and against the
// switching run's own other intervals after step 50.
out.periodic = [];
for (const x of out.rise) {
  const r = R.find(z => z.run === x.run), a = Math.floor(x.step_50 / 50) * 50, b = a + 50;
  const mates = R.filter(z => z.seed === r.seed && z.style !== 'draft' && z.cond !== 'kev_only');
  const d = (run, set, s0 = a, s1 = b) => run.periodic_evals[s1][set] - run.periodic_evals[s0][set];
  const others = [50, 100, 150].filter(s => s !== a);
  out.periodic.push({
    run: x.run, interval: [a, b], math500: d(r, 'math500'), math500_mates: mean(mates.map(m => d(m, 'math500'))),
    val: d(r, 'val'), val_mates: mean(mates.map(m => d(m, 'val'))), math500_own_other_intervals: mean(others.map(s => d(r, 'math500', s, s + 50))),
  });
}
{
  const ch = [];
  for (const r of OT.filter(z => z.cond !== 'kev_only')) for (const s of [50, 100, 150]) ch.push(r.periodic_evals[s + 50].math500 - r.periodic_evals[s].math500);
  out.periodic_others = { intervals: ch.length, max: Math.max(...ch), share_at_least_3pts: ch.filter(c => c >= 0.03 - 1e-9).length / ch.length, mean: mean(ch) };
}

fs.writeFileSync(path.join(dir, 'summary.json'), JSON.stringify(out, null, 1));

// printed tables
console.log('1. Rise of the draft share in the six switching runs (training completions, trailing 5-step share)');
for (const x of out.rise) console.log(`  ${x.run.padEnd(18)} trough ${pct(x.trough_share)} at ${x.trough_step}; 20% at ${x.step_20}, 50% at ${x.step_50} (${x.steps_20_to_50} steps); largest fall after 50%: ${pct(x.max_fall_after_50)}; steps ${x.rise_steps.join('-')}: share ${pct(x.share_before)} -> ${pct(x.share_at_end)}, sum S ${f(x.S_sum, 3)} (null sd ${f(x.S_null_sd, 3)}, z ${f(x.z, 1)}), d_right ${f(100 * x.d_right, 1)} pts over ${x.mixed_groups} groups`);
console.log('\n2. Change in logit(draft share) to the next 10-step window against summed selection S in the window');
for (const [k, v] of Object.entries(out.response)) console.log(`  ${k.padEnd(12)} n=${v.n} r=${f(v.r)} slope=${f(v.slope, 1)} intercept=${f(v.intercept, 3)}`);
console.log('  windows of the switching runs with a rise of more than 0.7 in logit:');
for (const x of out.rise_windows) console.log(`    ${x.run.padEnd(18)} steps ${x.w}-${x.w + 9}: ${pct(x.share[0])} -> ${pct(x.share[1])}, dlogit ${f(x.dlogit)}, S ${f(x.S, 3)} (z ${f(x.z, 1)}), predicted by the early fit ${f(x.predicted_dlogit)}`);
console.log(`  largest rise in logit after step 50 in a run that did not switch: ${f(out.others_max_dlogit)}`);
console.log('\n3. Early features, switching (6) against other (20) runs; exact permutation p; switching minus same-condition seeds');
for (const x of out.early) console.log(`  steps ${x.steps.join('-').padEnd(6)} ${x.feature.padEnd(17)} ${f(x.switching_mean, 3).padStart(8)} vs ${f(x.others_mean, 3).padStart(8)}  p=${f(x.p_perm, 3)}  matched: ${x.matched_diffs.map(d => f(d, 3)).join(' ')}`);
console.log('\n4. Draft share by subject in the window of the 20% crossing (previous window -> this window)');
for (const x of out.onset_composition) {
  console.log(`  ${x.run.padEnd(18)} ${x.window.join('-')}: ` + Object.entries(x.subj).map(([k, v]) => `${k.slice(0, 8)} ${pct(v[0])}->${pct(v[1])}`).join(', '));
  console.log(`  ${''.padEnd(18)} by mates' accuracy: ` + x.difficulty.map((d, j) => `[${d.bin.map(b => f(b, 2)).join(',')}) ${pct(x.difficulty_before[j].draft)}->${pct(d.draft)} (${d.problems} problems)`).join(', '));
}
console.log('\n5. Final policies, sampled sets: drafts minus other completions of the same problem');
for (const [k, v] of Object.entries(out.final_within)) console.log(`  ${k.padEnd(10)} ${v.runs} runs, ${v.problems} problems with both: ${f(100 * v.diff, 1)} pts [${v.ci.map(c => f(100 * c, 1)).join(', ')}]; mean accuracy on these sets ${pct(v.mean_acc_sampled_sets, 1)}`);
const P = out.rise_pooled;
console.log(`\n6. Rise windows pooled: drafts minus others within a group ${f(100 * P.d_right, 1)} pts [${P.ci.map(c => f(100 * c, 1)).join(', ')}] over ${P.groups} groups; summed S ${f(P.S_sum, 3)} (null sd ${f(P.S_null_sd, 3)}, z ${f(P.z, 1)})`);
console.log('\n7. Training accuracy on the same problems, switching run minus same-seed non-switching runs, points [95% CI]');
for (const x of out.paired_training) console.log(`  ${x.run.padEnd(18)} (${x.mates.length} mates) ` + ['before', 'rise', 'after'].map(k => `${k} ${x[k].steps.join('-')}: ${f(100 * x[k].diff, 1)} [${x[k].ci.map(c => f(100 * c, 1)).join(', ')}]`).join(' | '));
console.log('\n8. Greedy evaluations over the 50-step interval containing the 50% crossing (points)');
for (const x of out.periodic) console.log(`  ${x.run.padEnd(18)} ${x.interval.join('-')}: MATH-500 ${f(100 * x.math500, 1)} (same-seed mates ${f(100 * x.math500_mates, 1)}; own other intervals ${f(100 * x.math500_own_other_intervals, 1)}), val ${f(100 * x.val, 1)} (mates ${f(100 * x.val_mates, 1)})`);
const po = out.periodic_others;
console.log(`  non-switching runs, ${po.intervals} intervals after step 50: mean ${f(100 * po.mean, 1)}, max ${f(100 * po.max, 1)}, share >= 3 pts ${pct(po.share_at_least_3pts)}`);
