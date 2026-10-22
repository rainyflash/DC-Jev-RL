// Step-by-step analysis of the switch to drafting first in the 26 GRPO runs of DC-Jev-RL, from their training
// completions (experiments/<part>/runs/<cond>/s<seed>/rollouts.jsonl.gz: every completion of every step, 16 groups of 8).
//
//   node experiments/draft_switch/analyze.js [<repo> [<outdir>]]
//
// By default <repo> is the repository this file is in and <outdir> is results/ next to it.
//
// The draft measure is the papers' (dcjev/analysis/part2.py, informal_opening): no LaTeX delimiter ($, \( or \[) and no
// markdown heading or bold (### or **) in the first 300 characters. Before anything else the script recomputes the
// papers' 50-step windows (draft_windows in experiments/<part>/results/summary.json) and stops if they differ.
//
// Output: runs.json (per run: style, step series, 10- and 25-step windows, takeoff, opening types, subjects, the
// final policy's completions by opening, per-problem accuracy; about 9 MB, not committed), per_step.csv (one row per
// run and step), validation.json.
'use strict';
const fs = require('fs'), path = require('path'), zlib = require('zlib'), readline = require('readline');

const MARKS = ['$', '\\(', '\\[', '###', '**'];
const OPEN = 300;                      // part2.informal_opening: opening window in characters
const SWITCH = 0.5, REPEAT = 0.05;     // paper_numbers.style_mode
const PARTS = [['part2_llm', 'II'], ['part3b_labels', 'III-B'], ['part3c_tree', 'III-C']];
const FM_CAP = 1000;                   // first-marker position is capped here (no marker at all counts as FM_CAP)

// Python slices and lengths count code points, JS strings count UTF-16 units; they differ only around astral characters.
const SURR = /[\uD800-\uDFFF]/;
const head = (t, n) => SURR.test(t.slice(0, 2 * n)) ? Array.from(t).slice(0, n).join('') : t.slice(0, n);
const cpLen = t => SURR.test(t) ? Array.from(t).length : t.length;
const informal = (t, n = OPEN) => { const h = head(t, n); return !MARKS.some(m => h.includes(m)); };

function firstMark(t) {   // character index of the first marker, at most FM_CAP
  let i = Infinity;
  for (const m of MARKS) { const j = t.indexOf(m); if (j >= 0 && j < i) i = j; }
  if (i === Infinity) return FM_CAP;
  const pre = t.slice(0, i);
  return Math.min(FM_CAP, SURR.test(pre) ? Array.from(pre).length : i);
}

function density(t) {     // markers per 1000 characters of the whole completion
  let n = 0;
  for (const m of MARKS) n += t.split(m).length - 1;
  const L = cpLen(t);
  return L ? 1000 * n / L : 0;
}

const openingKey = t => t.trim().split(/\s+/).slice(0, 3).join(' ').toLowerCase();

// "We need probability that ...", "We need count of ...": an opening that drops the article or "to" after "we need".
const NEED_OK = new Set(['to', 'the', 'a', 'an', 'this', 'that', 'these', 'those', 'our', 'its', 'their', 'some', 'both', 'all', 'each', 'every', 'one', 'two']);
function telegraphic(t) {
  const m = /^\s*(?:<think>\s*)?we need (\S+)/i.exec(t.slice(0, 80));
  return m ? !NEED_OK.has(m[1].toLowerCase().replace(/[^a-z]/g, '')) : false;
}
const thinkTag = t => t.includes('<think>') || t.includes('</think>');

const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
function median(a) {
  if (!a.length) return null;
  const b = [...a].sort((x, y) => x - y), n = b.length;
  return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2;
}
function groupBy(a, f) {
  const m = new Map();
  for (const x of a) { const k = f(x); if (!m.has(k)) m.set(k, []); m.get(k).push(x); }
  return m;
}
const lines = file => readline.createInterface({
  input: file.endsWith('.gz') ? fs.createReadStream(file).pipe(zlib.createGunzip()) : fs.createReadStream(file),
  crlfDelay: Infinity,
});

async function readRollouts(file) {
  const out = [];
  for await (const l of lines(file)) {
    if (!l) continue;
    const r = JSON.parse(l), t = r.text;
    out.push({
      step: r.step, id: r.id, subj: r.id.split('-').slice(2, -1).join('-'), d: informal(t) ? 1 : 0, fm: firstMark(t),
      lat: density(t), key: openingKey(t), think: thinkTag(t) ? 1 : 0, tele: telegraphic(t) ? 1 : 0,
      v: r.v, adv: r.adv, used: r.used, ntok: r.n_tok, trunc: r.finish === 'length' ? 1 : 0,
      kev: r.kev ? r.kev.correct : null, root: r.root, open: t.slice(0, 240),
    });
  }
  return out;
}

async function readLog(dir) {
  const gz = path.join(dir, 'log.jsonl.gz'), f = fs.existsSync(gz) ? gz : path.join(dir, 'log.jsonl');
  const steps = {}, evals = {};
  for await (const l of lines(f)) {
    if (!l) continue;
    const r = JSON.parse(l);
    if (r.kind === 'step') steps[r.step] = r;
    if (r.kind === 'eval') evals[r.step] = Object.fromEntries(Object.entries(r).filter(([, v]) => v && typeof v === 'object' && 'acc' in v).map(([k, v]) => [k, v.acc]));
  }
  return { steps, evals };
}

// Statistics of a set of training completions (one step or a window of steps). Groups are (step, problem).
// S_x is the selection differential of trait x: the within-group covariance of x with the advantage over the
// completions used in the update, sum of A_k (x_k - mean of x in the group) divided by their number. To first order a
// policy-gradient step raises the mean of x when S_x > 0. d_right / d_adv: over the groups holding both kinds, the
// mean of drafts minus others in verdict / advantage, as part2.draft_windows computes them.
function stats(rs, keepGroups = false) {
  const nd = rs.filter(r => !r.d), dr = rs.filter(r => r.d);
  const o = {
    n: rs.length, draft: mean(rs.map(r => r.d)),
    fm100: mean(rs.map(r => r.fm >= 100 ? 1 : 0)), fm200: mean(rs.map(r => r.fm >= 200 ? 1 : 0)),
    fm_med_nondraft: median(nd.map(r => r.fm)), fm_mean_nondraft: mean(nd.map(r => r.fm)),
    lat: mean(rs.map(r => r.lat)), lat_nondraft: mean(nd.map(r => r.lat)), lat_draft: mean(dr.map(r => r.lat)),
    right: mean(rs.map(r => r.v)), right_draft: mean(dr.map(r => r.v)), right_nondraft: mean(nd.map(r => r.v)),
    tokens: mean(rs.map(r => r.ntok)), tokens_draft: mean(dr.map(r => r.ntok)), tokens_nondraft: mean(nd.map(r => r.ntok)),
    trunc: mean(rs.map(r => r.trunc)), trunc_draft: mean(dr.map(r => r.trunc)), trunc_nondraft: mean(nd.map(r => r.trunc)),
    think: mean(rs.map(r => r.think)), think_draft: mean(dr.map(r => r.think)), tele: mean(rs.map(r => r.tele)),
    tele_of_drafts: mean(dr.map(r => r.tele)),
  };
  // Sd_var: the variance of the numerator of S_draft when the openings are permuted within each group (no association
  // with the advantage): m(n-m)/(n(n-1)) times the sum of squared centred advantages, summed over groups.
  let nu = 0, Sd = 0, Sl = 0, Sf = 0, Sl_nd = 0, nu_nd = 0, Sd_var = 0;
  const dv = [], da = [];
  for (const g of groupBy(rs, r => r.step + '|' + r.id).values()) {
    const u = g.filter(r => r.used);
    if (u.length) {
      const md = mean(u.map(r => r.d)), ml = mean(u.map(r => r.lat)), mf = mean(u.map(r => Math.log1p(r.fm)));
      for (const r of u) { Sd += r.adv * (r.d - md); Sl += r.adv * (r.lat - ml); Sf += r.adv * (Math.log1p(r.fm) - mf); nu++; }
      const n = u.length, m = u.filter(r => r.d).length, am = mean(u.map(r => r.adv));
      if (n > 1) Sd_var += m * (n - m) / (n * (n - 1)) * u.reduce((s, r) => s + (r.adv - am) ** 2, 0);
      const un = u.filter(r => !r.d);   // the same for the LaTeX density among the formal completions only
      if (un.length > 1) {
        const mn = mean(un.map(r => r.lat)), an = mean(un.map(r => r.adv));
        for (const r of un) { Sl_nd += (r.adv - an) * (r.lat - mn); nu_nd++; }
      }
    }
    const D = g.filter(r => r.d), F = g.filter(r => !r.d);
    if (D.length && F.length) {
      dv.push(mean(D.map(r => r.v)) - mean(F.map(r => r.v)));
      const Du = D.filter(r => r.used), Fu = F.filter(r => r.used);
      if (Du.length && Fu.length) da.push(mean(Du.map(r => r.adv)) - mean(Fu.map(r => r.adv)));
    }
  }
  Object.assign(o, {
    used: nu, S_draft: nu ? Sd / nu : null, S_draft_null_sd: nu ? Math.sqrt(Sd_var) / nu : null, S_lat: nu ? Sl / nu : null, S_logfm: nu ? Sf / nu : null,
    S_lat_nondraft: nu_nd ? Sl_nd / nu_nd : null, mixed: dv.length, d_right: mean(dv), d_adv: mean(da), d_adv_groups: da.length,
  });
  if (keepGroups) o.d_right_groups = dv;   // per mixed group, for intervals over groups
  return o;
}

function windows(byStep, width) {
  const out = [];
  for (let s = 0; s < 200; s += width) {
    const rs = [];
    for (let t = s; t < s + width; t++) if (byStep.has(t)) rs.push(...byStep.get(t));
    if (rs.length) out.push({ steps: [s, s + width - 1], ...stats(rs) });
  }
  return out;
}

// The share of drafts by subject and by difficulty (verifier accuracy of the group's other, non-draft completions),
// and the commonest opening words of the drafts, per window.
function composition(byStep, width) {
  const out = [];
  for (let s = 0; s < 200; s += width) {
    const rs = [];
    for (let t = s; t < s + width; t++) if (byStep.has(t)) rs.push(...byStep.get(t));
    if (!rs.length) continue;
    const subj = {};
    for (const [k, g] of groupBy(rs, r => r.subj)) subj[k] = { n: g.length, draft: mean(g.map(r => r.d)) };
    const diff = { '0': [], '(0,0.5)': [], '[0.5,1)': [], '1': [] };
    for (const g of groupBy(rs, r => r.step + '|' + r.id).values()) {
      const F = g.filter(r => !r.d);
      if (!F.length) continue;
      const a = mean(F.map(r => r.v));
      const b = a === 0 ? '0' : a < 0.5 ? '(0,0.5)' : a < 1 ? '[0.5,1)' : '1';
      diff[b].push(...g.map(r => r.d));
    }
    const keys = {};
    for (const r of rs) if (r.d) keys[r.key] = (keys[r.key] || 0) + 1;
    const top = Object.entries(keys).sort((x, y) => y[1] - x[1]).slice(0, 8);
    out.push({
      steps: [s, s + width - 1], n: rs.length, drafts: rs.filter(r => r.d).length, subj,
      difficulty: Object.fromEntries(Object.entries(diff).map(([k, v]) => [k, { n: v.length, draft: mean(v) }])), top_draft_openings: top,
    });
  }
  return out;
}

// Takeoff of a run's training draft share: the trailing 5-step mean, the first step at which it reaches each level,
// and the minimum of the centred 11-step mean before it first reaches 0.2.
function takeoff(series) {
  const p = series.map(x => x.draft), m5 = p.map((_, t) => t < 4 ? null : mean(p.slice(t - 4, t + 1)));
  const first = lv => { const t = m5.findIndex(x => x !== null && x >= lv); return t < 0 ? null : t; };
  const at = { 0.1: first(0.1), 0.2: first(0.2), 0.3: first(0.3), 0.5: first(0.5) };
  const end = at[0.2] === null ? p.length : at[0.2];
  let trough = null;
  for (let t = 5; t < Math.max(6, end - 5); t++) {
    const m = mean(p.slice(t - 5, t + 6));
    if (trough === null || m < trough.share) trough = { step: series[t].step, share: m };
  }
  const lv = Object.fromEntries(Object.entries(at).map(([k, t]) => [k, t === null ? null : series[t].step]));
  return { first_step_at: lv, trough, final_window_share: mean(p.slice(-10)) };
}

// The final policy's evaluation completions (final_completions.jsonl.gz) split by opening. Greedy sets have one
// completion per problem; the sampled sets (16 per problem) also give, over the problems with both kinds, the mean of
// drafts minus others in accuracy within a problem.
async function finalSplit(file) {
  const sets = {};
  for await (const l of lines(file)) {
    if (!l) continue;
    const r = JSON.parse(l);
    (sets[r.set] = sets[r.set] || []).push({ id: r.id, v: r.v, d: informal(r.text) ? 1 : 0, think: thinkTag(r.text) ? 1 : 0, tele: telegraphic(r.text) ? 1 : 0, ntok: r.n_tok });
  }
  const out = {};
  for (const [s, rs] of Object.entries(sets)) {
    const D = rs.filter(r => r.d), F = rs.filter(r => !r.d), within = [];
    for (const [id, g] of groupBy(rs, r => r.id)) {
      const gd = g.filter(r => r.d), gf = g.filter(r => !r.d);
      if (gd.length && gf.length) within.push({ id, n_draft: gd.length, n_other: gf.length, diff: mean(gd.map(r => r.v)) - mean(gf.map(r => r.v)) });
    }
    out[s] = {
      n: rs.length, problems: new Set(rs.map(r => r.id)).size, acc: mean(rs.map(r => r.v)), draft: mean(rs.map(r => r.d)),
      think: mean(rs.map(r => r.think)), tele: mean(rs.map(r => r.tele)), acc_draft: mean(D.map(r => r.v)), acc_other: mean(F.map(r => r.v)),
      tokens_draft: mean(D.map(r => r.ntok)), tokens_other: mean(F.map(r => r.ntok)), within,
    };
  }
  return out;
}

function styleMode(st) {
  if ((st.draft_opening || 0) >= SWITCH) return 'draft';
  return (st.repeated_lines || 0) >= REPEAT ? 'repeat' : 'plain';
}

async function main() {
  const repo = process.argv[2] || path.join(__dirname, '..', '..'), outdir = process.argv[3] || path.join(__dirname, 'results');
  fs.mkdirSync(outdir, { recursive: true });
  const runs = [], validation = [];
  for (const [part, label] of PARTS) {
    const summary = JSON.parse(fs.readFileSync(path.join(repo, 'experiments', part, 'results', 'summary.json'), 'utf8')).conditions;
    const root = path.join(repo, 'experiments', part, 'runs');
    for (const cond of fs.readdirSync(root).sort()) {
      for (const sd of fs.readdirSync(path.join(root, cond)).sort()) {
        const dir = path.join(root, cond, sd), f = path.join(dir, 'rollouts.jsonl.gz');
        if (!fs.existsSync(f)) continue;
        const seed = sd.slice(1), e = summary[cond];
        const st = e.math500_style_per_seed[seed];
        const t0 = Date.now();
        const recs = await readRollouts(f), { steps: log, evals } = await readLog(dir);
        const byStep = groupBy(recs, r => r.step);
        // reproduce the papers' 50-step windows before using anything else
        const w50 = windows(byStep, 50), ref = e.draft_windows[seed];
        let maxdiff = 0;
        ref.forEach((w, i) => {
          const m = w50[i];
          maxdiff = Math.max(maxdiff, Math.abs(m.draft - w.draft));
          if (w.verdict) { maxdiff = Math.max(maxdiff, Math.abs(m.d_right - w.verdict.diff)); if (m.mixed !== w.verdict.groups) maxdiff = Infinity; }
          if (w.advantage) maxdiff = Math.max(maxdiff, Math.abs(m.d_adv - w.advantage.diff));
        });
        validation.push({ run: `${cond}/s${seed}`, part: label, max_abs_diff: maxdiff });
        if (!(maxdiff < 1e-9)) throw new Error(`${cond}/s${seed}: the 50-step windows differ from summary.json (${maxdiff})`);
        const series = [...byStep.keys()].sort((a, b) => a - b).map(s => {
          const L = log[s] || {};
          return {
            step: s, ...stats(byStep.get(s), true), acc: L.acc, groups_mixed: L.groups_mixed, groups_all_right: L.groups_all_right,
            groups_all_wrong: L.groups_all_wrong, grad_norm: L.grad_norm, adv_scale: L.adv_scale,
          };
        });
        // a few draft openings at the start and where the share first reaches 0.2 (for reading)
        const tk = takeoff(series), pick = (lo, hi) => recs.filter(r => r.d && r.step >= lo && r.step <= hi).slice(0, 4).map(r => ({ step: r.step, v: r.v, open: r.open }));
        const examples = { start: pick(0, 9) };
        if (tk.first_step_at[0.2] !== null) examples.takeoff = pick(Math.max(0, tk.first_step_at[0.2] - 10), tk.first_step_at[0.2]);
        runs.push({
          run: `${cond}/s${seed}`, part: label, cond, seed: +seed, style: styleMode(st), final_math500: st,
          eval_draft_by_step: e.math500_draft_by_step[seed], final_split: await finalSplit(path.join(dir, 'final_completions.jsonl.gz')),
          takeoff: tk, series, periodic_evals: evals,
          // the verifier's accuracy and the share of drafts on each problem at each step; runs with the same seed get the
          // same problems in the same order
          problems: [...groupBy(recs, r => r.step + '|' + r.id).values()].map(g => [g[0].step, g[0].id, mean(g.map(r => r.v)), mean(g.map(r => r.d))]),
          w10: windows(byStep, 10), w25: windows(byStep, 25), composition: composition(byStep, 25), examples,
        });
        console.log(`${label} ${cond}/s${seed}: ${recs.length} completions, style ${styleMode(st)}, windows match (max diff ${maxdiff.toExponential(1)}), ${((Date.now() - t0) / 1000).toFixed(1)}s`);
      }
    }
  }
  // Part II's rlvr runs also appear in III-B's and III-C's summaries; each run is read once, from its own folder.
  fs.writeFileSync(path.join(outdir, 'runs.json'), JSON.stringify(runs));
  fs.writeFileSync(path.join(outdir, 'validation.json'), JSON.stringify(validation, null, 1));
  const cols = ['draft', 'fm100', 'fm200', 'fm_med_nondraft', 'lat', 'lat_nondraft', 'right', 'right_draft', 'right_nondraft',
    'tokens', 'trunc', 'think', 'tele', 'S_draft', 'S_lat', 'S_logfm', 'S_lat_nondraft', 'mixed', 'd_right', 'd_adv', 'acc', 'groups_mixed',
    'groups_all_right', 'groups_all_wrong', 'grad_norm', 'adv_scale'];
  const rows = ['run,part,style,step,' + cols.join(',')];
  for (const r of runs) for (const s of r.series) rows.push([r.run, r.part, r.style, s.step, ...cols.map(c => s[c] === null || s[c] === undefined ? '' : +(+s[c]).toPrecision(6))].join(','));
  fs.writeFileSync(path.join(outdir, 'per_step.csv'), rows.join('\n') + '\n');
  console.log(`${runs.length} runs written to ${outdir}`);
}

main().catch(e => { console.error(e); process.exit(1); });
