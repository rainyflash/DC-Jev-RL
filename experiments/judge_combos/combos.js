// Combinations of Kev-4B's three answers (c = 'correct', g = 'genuine', q = 'quality', each in [0, 1]) as a judge
// score: how well each ranks right above wrong solutions, and how much each pays for repetition and other edits that
// leave the final answer unchanged. Offline, from data already in the repository:
//   - experiments/part2_llm/judge_study: 4000 base-policy completions (1000 problems x 4) with the verifier's verdict,
//     scored as written and in four edited copies (answer_only, padded, asserted, altered);
//   - experiments/part2_llm/repeat_probe: the same completions with repetition added (end3, twice), and repeating final
//     completions of GRPO runs as written and cut after their first answer;
//   - training completions of the GRPO runs: within a group and verdict, repeating completions (repeats_after_answer)
//     against the others. The headline pools the 11 runs whose reward used Kev-4B (every completion scored on all three
//     questions); the other runs logged Kev-4B for half their completions, which shows whether the premium needs a
//     policy trained against Kev-4B. rlvr_ar_add also logged the AR judge that was in its reward;
//   - Part III-B: the same comparison in the fill-in reward of the unlabelled completions (released, refitted or
//     scratch head).
// Premiums are divided by the combination's own gap between right and wrong solutions on the same data, so that
// combinations on different scales can be compared ("how many right-wrong gaps does the edit buy").
//
//   node experiments/judge_combos/combos.js [<repo> [<out.json>]]   (default: results/results.json)
'use strict';
const fs = require('fs'), path = require('path'), zlib = require('zlib'), readline = require('readline');
const repo = process.argv[2] || path.join(__dirname, '..', '..');
const outFile = process.argv[3] || path.join(__dirname, 'results', 'results.json');
fs.mkdirSync(path.dirname(outFile), { recursive: true });

const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
const lines = f => readline.createInterface({ input: fs.createReadStream(f).pipe(zlib.createGunzip()), crlfDelay: Infinity });
async function readJsonl(f) { const o = []; for await (const l of lines(f)) if (l) o.push(JSON.parse(l)); return o; }
function groupBy(a, f) { const m = new Map(); for (const x of a) { const k = f(x); if (!m.has(k)) m.set(k, []); m.get(k).push(x); } return m; }

// --- the combinations -------------------------------------------------------------------------------------------
const SETS = { c: ['c'], g: ['g'], q: ['q'], cg: ['c', 'g'], cq: ['c', 'q'], gq: ['g', 'q'], cgq: ['c', 'g', 'q'] };
const OPS = { mean: xs => mean(xs), min: xs => Math.min(...xs), prod: xs => xs.reduce((a, b) => a * b, 1) };
const COMBOS = {};
for (const [k, s] of Object.entries(SETS)) {
  if (s.length === 1) COMBOS[k] = x => x[s[0]];
  else for (const [o, f] of Object.entries(OPS)) COMBOS[`${o}(${s.join(',')})`] = x => f(s.map(z => x[z]));
}
const NAMES = Object.keys(COMBOS);
const SIG = [...NAMES, 'ar_correct'];   // reference: the AR judge (Qwen3.5-4B asked 'correct', P(yes))
const kevOf = k => ({ c: k.correct, g: k.genuine, q: k.quality });

// --- AUROC as dcjev.metrics: ties count one half; within-problem = mean over the problems holding both verdicts ---
function ranks(a) {
  const idx = a.map((_, i) => i).sort((i, j) => a[i] - a[j]), r = new Array(a.length);
  for (let i = 0; i < idx.length;) {
    let j = i; while (j + 1 < idx.length && a[idx[j + 1]] === a[idx[i]]) j++;
    for (let t = i; t <= j; t++) r[idx[t]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}
function auroc(s, y) {
  let n1 = 0; for (const v of y) n1 += v;
  const n0 = y.length - n1; if (!n1 || !n0) return null;
  const r = ranks(s); let sr = 0;
  for (let i = 0; i < y.length; i++) if (y[i]) sr += r[i];
  return (sr - n1 * (n1 + 1) / 2) / (n1 * n0);
}
function pearson(a, b) {
  const ma = mean(a), mb = mean(b); let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i] - ma, z = b[i] - mb; ab += x * z; aa += x * x; bb += z * z; }
  return ab / Math.sqrt(aa * bb);
}
const spearman = (a, b) => pearson(ranks(a), ranks(b));
function rng(seed) { let s = seed; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; }
const q = (a, p) => { const b = [...a].sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(p * b.length))]; };
const ci = a => [q(a, 0.025), q(a, 0.975)];

function repeatsAfterAnswer(t, minLen = 20) {   // dcjev/analysis/part2.py repeats_after_answer
  const i = t.indexOf('\\boxed');
  if (i < 0) return false;
  const seen = new Set(t.slice(0, i).split('\n').map(l => l.trim()).filter(l => [...l].length >= minLen));
  return t.slice(i).split('\n').some(l => { const s = l.trim(); return [...s].length >= minLen && seen.has(s); });
}

(async () => {
  const res = { combos: NAMES };
  // ---- 1. judge study: ranking -----------------------------------------------------------------------------------
  const JS = path.join(repo, 'experiments', 'part2_llm', 'judge_study');
  const comp = await readJsonl(path.join(JS, 'completions.jsonl.gz'));
  const scores = await readJsonl(path.join(JS, 'scores.jsonl.gz'));
  const probe = {};
  for (const r of scores) (probe[r.probe] = probe[r.probe] || new Map()).set(r.i, r);
  const items = comp.map((c, i) => ({ pid: c.set + '|' + c.id, v: c.v, x: kevOf(probe.orig.get(i).kev), ar: probe.orig.get(i).ar.correct }));
  const sig = (name, it) => name === 'ar_correct' ? it.ar : COMBOS[name](it.x);
  const S = Object.fromEntries(SIG.map(n => [n, items.map(it => sig(n, it))]));
  const Y = items.map(it => it.v);
  const P = [...groupBy(items.map((it, i) => ({ ...it, i })), it => it.pid).values()].map(g => g.map(it => it.i));
  const mixed = P.filter(m => { const k = m.reduce((s, i) => s + Y[i], 0); return k > 0 && k < m.length; });
  const perProblem = Object.fromEntries(SIG.map(n => [n, mixed.map(m => auroc(m.map(i => S[n][i]), m.map(i => Y[i])))]));
  const gap = n => mean(items.filter(it => it.v).map(it => sig(n, it))) - mean(items.filter(it => !it.v).map(it => sig(n, it)));
  res.ranking = {};
  for (const n of SIG) res.ranking[n] = { auroc: auroc(S[n], Y), auroc_within: mean(perProblem[n]), gap: gap(n) };
  // paired bootstrap over problems: each combination minus 'c', and minus 'q' (the best single answer)
  const B = 1000, rnd = rng(2024), REF = ['c', 'q'];
  const boot = Object.fromEntries(SIG.map(n => [n, Object.fromEntries(REF.map(r => [r, { a: [], w: [] }]))]));
  const mixedIdx = new Map(mixed.map((m, j) => [m, j]));
  for (let b = 0; b < B; b++) {
    const pick = Array.from({ length: P.length }, () => P[Math.floor(rnd() * P.length)]);
    const idx = pick.flat(), y = idx.map(i => Y[i]);
    const ws = pick.filter(m => mixedIdx.has(m)).map(m => mixedIdx.get(m));
    const val = Object.fromEntries(SIG.map(n => [n, { a: auroc(idx.map(i => S[n][i]), y), w: mean(ws.map(j => perProblem[n][j])) }]));
    for (const n of SIG) for (const r of REF) { boot[n][r].a.push(val[n].a - val[r].a); boot[n][r].w.push(val[n].w - val[r].w); }
  }
  for (const n of SIG) for (const r of REF) {
    res.ranking[n][`d_auroc_vs_${r}_ci`] = ci(boot[n][r].a);
    res.ranking[n][`d_within_vs_${r}_ci`] = ci(boot[n][r].w);
  }
  // how alike the answers are: rank correlation over all completions, among the right and among the wrong ones (once
  // the verdict is known, what is left is each answer's error, and combining answers can only cancel errors that differ)
  res.correlation = {};
  const I1 = Y.map((y, i) => y ? i : -1).filter(i => i >= 0), I0 = Y.map((y, i) => y ? -1 : i).filter(i => i >= 0);
  for (const [a, b] of [['c', 'g'], ['c', 'q'], ['g', 'q'], ['c', 'ar_correct'], ['g', 'ar_correct'], ['q', 'ar_correct']]) {
    const on = I => spearman(I.map(i => S[a][i]), I.map(i => S[b][i]));
    res.correlation[`${a}~${b}`] = { all: spearman(S[a], S[b]), right: on(I1), wrong: on(I0) };
  }

  // ---- 2. judge study: edits that keep or change the final answer -------------------------------------------------
  res.edits = {};
  for (const p of ['answer_only', 'padded', 'asserted', 'altered']) {
    res.edits[p] = {};
    for (const n of SIG) {
      const d = { 0: [], 1: [] }, up = { 0: 0, 1: 0 };
      for (const [i, e] of probe[p]) {
        const o = items[i], x = n === 'ar_correct' ? e.ar.correct - o.ar : COMBOS[n](kevOf(e.kev)) - COMBOS[n](o.x);
        d[o.v].push(x); up[o.v] += x > 0;
      }
      const g = res.ranking[n].gap;
      res.edits[p][n] = Object.fromEntries([0, 1].filter(v => d[v].length).map(v => [v ? 'right' : 'wrong', { n: d[v].length, shift: mean(d[v]), per_gap: mean(d[v]) / g, frac_up: up[v] / d[v].length }]));
    }
  }

  // ---- 3. repeat probe: repetition added to base completions, or removed from GRPO completions --------------------
  const rp = await readJsonl(path.join(repo, 'experiments', 'part2_llm', 'repeat_probe', 'scores.jsonl.gz'));
  const key = r => [r.kind, r.source, r.n].join('|');
  const origs = new Map(rp.filter(r => r.edit === 'orig').map(r => [key(r), r]));
  const pairsOf = edit => rp.filter(r => r.edit === edit).map(r => [origs.get(key(r)), r]);
  const probePremium = (pairs, sign) => Object.fromEntries(SIG.map(n => {
    const d = { 0: [], 1: [] }, up = { 0: 0, 1: 0 };
    for (const [o, e] of pairs) {
      const so = n === 'ar_correct' ? o.ar : COMBOS[n](kevOf(o.kev)), se = n === 'ar_correct' ? e.ar : COMBOS[n](kevOf(e.kev));
      const x = sign * (se - so);   // the score with repetition minus without it
      d[o.v].push(x); up[o.v] += x > 0;
    }
    const g = res.ranking[n].gap;
    return [n, Object.fromEntries([0, 1].filter(v => d[v].length).map(v => [v ? 'right' : 'wrong', { n: d[v].length, premium: mean(d[v]), per_gap: mean(d[v]) / g, frac_up: up[v] / d[v].length }]))];
  }));
  res.repeat_probe = {};
  for (const [label, edit, sign] of [['add_end3', 'end3', 1], ['add_twice', 'twice', 1], ['remove_cut', 'cut', -1]]) res.repeat_probe[label] = probePremium(pairsOf(edit), sign);
  // the cut edit by the run whose final completions they are (Kev-4B in its reward or not)
  res.repeat_probe.remove_cut_by_source = {};
  for (const [src, ps] of groupBy(pairsOf('cut'), p => p[0].source)) res.repeat_probe.remove_cut_by_source[src] = probePremium(ps, -1);

  // ---- 4. training completions -----------------------------------------------------------------------------------
  // Every run logs Kev-4B's three answers: for all completions when Kev-4B is in the reward, otherwise for 64 of the
  // 128 completions of each step (grpo.py --kev_log_n). rlvr_ar_add also logs the AR judge for all completions. The
  // family says how Kev-4B entered a run's training; 'training' pools the family kev_reward. Per (run, step, problem)
  // group and signal: the right-wrong gap, and by verdict the repeating minus the other completions, summed per
  // cluster of 25 steps of one run, which is what the bootstrap resamples. A repeating completion has a \boxed answer
  // by definition, while some of the others have none (cut off, or never answered), and every judge scores those low;
  // population 'answered' therefore compares only completions with a \boxed answer, gap included.
  const RUNS = [
    ['kev_reward', 'part2_llm', 'kev_only', [1, 2]], ['kev_reward', 'part2_llm', 'rlvr_kev_gate', [1, 2]],
    ['kev_reward', 'part2_llm', 'rlvr_kev_add', [1, 2, 3]], ['kev_reward', 'part3b_labels', 'sub_kev', [1, 2]],
    ['kev_reward', 'part3b_labels', 'sub_kev_recal', [1, 2]],
    ['kev_features', 'part3b_labels', 'sub_probe', [1, 2]],   // fill-in from a head trained from scratch on Kev-4B's features
    ['kev_roots', 'part3c_tree', 'tree_kev', [1, 2, 3]],      // Kev-4B's on_track answer picks the roots; the reward is v
    ['ar_reward', 'part2_llm', 'rlvr_ar_add', [1, 2]],
    ['no_kev', 'part2_llm', 'rlvr', [1, 2, 3]], ['no_kev', 'part3b_labels', 'rlvr_sub', [1, 2]],
    ['no_kev', 'part3c_tree', 'tree_random', [1, 2, 3]],
  ];
  const NT = SIG.length, BLOCK = 25, F = () => new Float64Array(NT), POPS = ['all', 'answered'];
  const clusters = [];
  res.repeat_share = {};
  for (const [fam, part, cond, seeds] of RUNS) for (const sd of seeds) {
    const run = `${cond}/s${sd}`, groups = new Map(), cl = new Map(), sh = { n: 0, rep: 0, nLate: 0, repLate: 0, w: 0, wNoAns: 0 };
    for await (const l of lines(path.join(repo, 'experiments', part, 'runs', cond, 's' + sd, 'rollouts.jsonl.gz'))) {
      if (!l) continue;
      const r = JSON.parse(l), s = F().fill(NaN);
      if (r.kev) { const x = kevOf(r.kev); NAMES.forEach((n, j) => { s[j] = COMBOS[n](x); }); }
      if (typeof r.ar === 'number') s[NT - 1] = r.ar;
      const it = { v: r.v, rep: repeatsAfterAnswer(r.text), ans: r.text.includes('\\boxed'), s };
      sh.n++; sh.rep += it.rep; if (r.step >= 150) { sh.nLate++; sh.repLate += it.rep; }
      if (!it.v) { sh.w++; sh.wNoAns += !it.ans; }
      const k = r.step + '|' + r.id; if (!groups.has(k)) groups.set(k, { step: r.step, m: [] }); groups.get(k).m.push(it);
    }
    res.repeat_share[run] = { all: sh.rep / sh.n, steps_150_199: sh.repLate / sh.nLate, wrong_without_answer: sh.wNoAns / sh.w };
    for (const { step, m: all } of groups.values()) for (const pop of POPS) {
      const m = pop === 'all' ? all : all.filter(x => x.ans), b = pop + '|' + Math.floor(step / BLOCK);
      if (!cl.has(b)) cl.set(b, { pop, fam, cond, run, sg: F(), ng: F(), sp: [F(), F()], np: [F(), F()], pw: [F(), F()], pn: [F(), F()] });
      const c = cl.get(b);
      for (let j = 0; j < NT; j++) {
        const sc = m.filter(x => !Number.isNaN(x.s[j]));
        const R1 = sc.filter(x => x.v === 1), R0 = sc.filter(x => x.v === 0);
        if (R1.length && R0.length) { c.sg[j] += mean(R1.map(x => x.s[j])) - mean(R0.map(x => x.s[j])); c.ng[j]++; }
        for (const v of [0, 1]) {
          const Rr = sc.filter(x => x.v === v && x.rep), Nr = sc.filter(x => x.v === v && !x.rep);
          if (!Rr.length || !Nr.length) continue;
          c.sp[v][j] += mean(Rr.map(x => x.s[j])) - mean(Nr.map(x => x.s[j])); c.np[v][j]++;
          for (const a of Rr) for (const z of Nr) { c.pw[v][j] += a.s[j] > z.s[j] ? 1 : a.s[j] === z.s[j] ? 0.5 : 0; c.pn[v][j]++; }
        }
      }
    }
    clusters.push(...cl.values());
  }
  // premium per right-wrong gap over a set of clusters, with a bootstrap over the clusters (and the difference from 'c')
  const ratio = (cs, j, v) => {
    let a = 0, na = 0, g = 0, ng = 0;
    for (const c of cs) { a += c.sp[v][j]; na += c.np[v][j]; g += c.sg[j]; ng += c.ng[j]; }
    return na && ng ? (a / na) / (g / ng) : null;
  };
  function trainSummary(cs) {
    const out = {}, sum = f => cs.reduce((s, c) => s + f(c), 0), rb = rng(7), jc = SIG.indexOf('c');
    const bs = Array.from({ length: B }, () => Array.from({ length: cs.length }, () => cs[Math.floor(rb() * cs.length)]));
    SIG.forEach((n, j) => {
      const ng = sum(c => c.ng[j]); if (!ng) return;
      const o = { gap_within_group: sum(c => c.sg[j]) / ng, groups_with_both_verdicts: ng };
      for (const v of [0, 1]) {
        const np = sum(c => c.np[v][j]); if (!np) continue;
        const pg = bs.map(b => ratio(b, j, v)).filter(x => x !== null);
        const dc = n === 'c' ? [] : bs.map(b => { const x = ratio(b, j, v), y = ratio(b, jc, v); return x === null || y === null ? null : x - y; }).filter(x => x !== null);
        o[v ? 'right' : 'wrong'] = { groups: np, premium: sum(c => c.sp[v][j]) / np, per_gap: ratio(cs, j, v), per_gap_ci: ci(pg), p_rep_above: sum(c => c.pw[v][j]) / sum(c => c.pn[v][j]), ...(dc.length ? { d_per_gap_vs_c_ci: ci(dc) } : {}) };
      }
      out[n] = o;
    });
    return out;
  }
  res.training = trainSummary(clusters.filter(c => c.pop === 'all' && c.fam === 'kev_reward'));
  res.training_answered = trainSummary(clusters.filter(c => c.pop === 'answered' && c.fam === 'kev_reward'));
  res.training_by_family = {}; res.training_by_condition = {};
  for (const pop of POPS) {
    res.training_by_family[pop] = {}; res.training_by_condition[pop] = {};
    for (const fam of new Set(RUNS.map(r => r[0]))) res.training_by_family[pop][fam] = trainSummary(clusters.filter(c => c.pop === pop && c.fam === fam));
    for (const [, , cond] of RUNS) res.training_by_condition[pop][cond] = trainSummary(clusters.filter(c => c.pop === pop && c.cond === cond));
  }

  // ---- 5. Part III-B: the fill-in reward of the unlabelled completions ---------------------------------------------
  // Within a group and verdict, the repeating minus the other unlabelled completions, in the reward itself (the head's
  // P(correct): the released head in sub_kev, refitted online in sub_kev_recal, trained from scratch in sub_probe);
  // populations as in section 4. The bootstrap is over groups and ignores that the groups of one run are related.
  res.fill_in = {};
  for (const cond of ['sub_kev', 'sub_kev_recal', 'sub_probe']) {
    const W = Object.fromEntries(POPS.map(p => [p, [0, 1, 2, 3].map(() => ({ 0: [], 1: [] }))]));
    for (const sd of [1, 2]) {
      const groups = new Map();
      for await (const l of lines(path.join(repo, 'experiments', 'part3b_labels', 'runs', cond, 's' + sd, 'rollouts.jsonl.gz'))) {
        if (!l) continue;
        const r = JSON.parse(l);
        if (r.labelled !== false || !r.used) continue;
        const k = r.step + '|' + r.id; if (!groups.has(k)) groups.set(k, []);
        groups.get(k).push({ step: r.step, v: r.v, r: r.r, rep: repeatsAfterAnswer(r.text), ans: r.text.includes('\\boxed') });
      }
      for (const g of groups.values()) for (const pop of POPS) {
        const m = pop === 'all' ? g : g.filter(x => x.ans), w = W[pop][Math.floor(g[0].step / 50)];
        for (const v of [0, 1]) {
          const R = m.filter(x => x.v === v && x.rep), N = m.filter(x => x.v === v && !x.rep);
          if (R.length && N.length) w[v].push(mean(R.map(x => x.r)) - mean(N.map(x => x.r)));
        }
      }
    }
    const out = res.fill_in[cond] = {};
    for (const pop of POPS) {
      out[pop] = { by_50_steps: W[pop].map(w => ({ wrong: mean(w[0]), wrong_groups: w[0].length, right: mean(w[1]), right_groups: w[1].length })) };
      for (const v of [0, 1]) {
        const d = [...W[pop][2][v], ...W[pop][3][v]], rb = rng(99);
        const bm = Array.from({ length: 2000 }, () => { let t = 0; for (let i = 0; i < d.length; i++) t += d[Math.floor(rb() * d.length)]; return t / d.length; });
        out[pop][`steps_100_199_${v ? 'right' : 'wrong'}`] = { premium: mean(d), ci: ci(bm), groups: d.length };
      }
    }
  }
  fs.writeFileSync(outFile, JSON.stringify(res, null, 1));

  // ---- printed summary --------------------------------------------------------------------------------------------
  const f = (x, d = 3) => x === null || x === undefined ? '-' : x.toFixed(d);
  const sgn = x => (x >= 0 ? '+' : '') + f(x);
  console.log('combination      AUROC  within  d_within vs c [95%]        d_within vs q [95%]        | repetition premium / right-wrong gap:  training wrong  right | answered wrong right | probe twice wrong right | GRPO cut wrong right | asserted wrong');
  for (const n of SIG) {
    const r = res.ranking[n], t = res.training[n], ta = res.training_answered[n], tw = res.repeat_probe.add_twice[n], rc = res.repeat_probe.remove_cut[n], as = res.edits.asserted[n];
    const d = ref => n === ref ? '        —' : `${sgn(r.auroc_within - res.ranking[ref].auroc_within)} [${f(r[`d_within_vs_${ref}_ci`][0])}, ${f(r[`d_within_vs_${ref}_ci`][1])}]`;
    const two = o => o ? f(o.wrong.per_gap, 2).padStart(6) + ' ' + f(o.right.per_gap, 2).padStart(6) : '     -      -';
    console.log(`${n.padEnd(15)} ${f(r.auroc)}  ${f(r.auroc_within)}  ${d('c').padEnd(26)} ${d('q').padEnd(26)} |  ` +
      `${two(t)} | ${two(ta)} | ${f(tw.wrong.per_gap, 2).padStart(6)} ${f(tw.right.per_gap, 2).padStart(6)} | ${f(rc.wrong ? rc.wrong.per_gap : null, 2).padStart(6)} ${f(rc.right.per_gap, 2).padStart(6)} | ${f(as.wrong.per_gap, 2).padStart(6)}`);
  }
  console.log('\nrank correlation of the answers on the judge study: all | among right | among wrong');
  for (const [k, c] of Object.entries(res.correlation)) console.log(`  ${k.padEnd(14)} ${f(c.all, 2)} | ${f(c.right, 2)} | ${f(c.wrong, 2)}`);
  console.log('\ntraining completions: premium for repeating / within-group right-wrong gap [95%, bootstrap over 25-step clusters] (groups)');
  const SHOW = ['c', 'g', 'q', 'mean(c,g,q)', 'prod(c,g,q)', 'ar_correct'];
  const cell = (o, v) => o && o[v] ? `${f(o[v].per_gap, 2)} [${f(o[v].per_gap_ci[0], 2)}, ${f(o[v].per_gap_ci[1], 2)}] (${o[v].groups})` : '-';
  const runsOf = pred => RUNS.filter(pred).flatMap(([, , cond, seeds]) => seeds.map(s => `${cond}/s${s}`));
  const block = (label, t, runs) => {
    const sh = k => (100 * mean(runs.map(r => res.repeat_share[r][k]))).toFixed(1);
    console.log(`  ${label}: ${runs.length} runs, ${sh('steps_150_199')}% of completions repeat in steps 150-199, ${sh('wrong_without_answer')}% of wrong ones have no \\boxed`);
    for (const n of SHOW) if (t[n]) console.log(`    ${n.padEnd(12)} gap ${f(t[n].gap_within_group)}  wrong ${cell(t[n], 'wrong').padEnd(30)} right ${cell(t[n], 'right')}`);
  };
  for (const pop of POPS) {
    console.log(`  == population ${pop} ==`);
    for (const fam of Object.keys(res.training_by_family[pop])) block(fam, res.training_by_family[pop][fam], runsOf(r => r[0] === fam));
  }
  console.log('  -- answered, by condition --');
  for (const [, , cond] of RUNS) block(cond, res.training_by_condition.answered[cond], runsOf(r => r[2] === cond));
  console.log('\nGRPO cut probe by source run: premium / gap, wrong (n) | right (n)');
  for (const [src, o] of Object.entries(res.repeat_probe.remove_cut_by_source))
    console.log(`  ${src.padEnd(20)} ` + ['c', 'g', 'q', 'ar_correct'].map(n => `${n} ${o[n].wrong ? f(o[n].wrong.per_gap, 2) : '-'} (${o[n].wrong ? o[n].wrong.n : 0}) | ${f(o[n].right.per_gap, 2)} (${o[n].right.n})`).join('   '));
  const pts = x => x === null ? '-' : (100 * x).toFixed(1);
  console.log('\nPart III-B fill-in reward of unlabelled completions, repeating minus others (points): steps 0-49 | 50-99 | 100-149 | 150-199 || 100-199 [95%] (groups)');
  for (const [cond, o] of Object.entries(res.fill_in)) for (const pop of POPS) for (const lab of ['wrong', 'right']) {
    const s = o[pop][`steps_100_199_${lab}`];
    console.log(`  ${cond.padEnd(14)} ${pop.padEnd(9)} ${lab.padEnd(6)} ${o[pop].by_50_steps.map(w => pts(w[lab])).join(' | ')} || ${pts(s.premium)} [${pts(s.ci[0])}, ${pts(s.ci[1])}] (${s.groups})`);
  }
  console.log(`\nwritten to ${path.relative(process.cwd(), outFile).split(path.sep).join('/')}`);
})();
