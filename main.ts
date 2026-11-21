// gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate
// at the case target, as five quantiles, from the case's own history alone.
//
// Why this shape, and what the history actually says
// ---------------------------------------------------
// An earn rate is a ratio: earners_i / players_tracked. Exophase does not
// refresh its counts continuously; it refreshes on its own schedule, hours
// apart, so a read shows the rates as of its last refresh. Between refreshes
// the page is flat; at a refresh the tracked-player count jumps and almost
// every rate drops at once (launch night: 638 tracked players, then 3,868,
// every rate but the one at 100% lower), because the newly tracked players
// hold fewer trophies. A forecaster that only looks at rates therefore sees
// step drops that are not players losing trophies, and a forecaster that
// only looks at the last rate (the house persistence baseline) misses both
// forces that move a rate over a week: players progressing through the game
// (earners grow) and new players joining (the denominator grows).
//
// This program therefore models the two forces separately, per trophy, and
// blends them with persistence:
//
//   1. Log-time trend on the logit rate. For each trophy, a weighted
//      least-squares line of logit(rate) on ln(1 + hours since release).
//      Log time is the right clock for a launch: progress is fastest in the
//      first hours and slows as the player base ages, and a line in log time
//      saturates in the right way (it cannot run away linearly). The slope
//      is shrunk towards the cross-trophy median slope, more so when the
//      trophy's own history is short or noisy, so a trophy with no signal
//      borrows the field's typical drift instead of chasing noise.
//   2. Earners / players decomposition. Earners E_i = rate_i * N / 100 and
//      the tracked count N are each fitted in logs against the same log-time
//      clock, then the forecast rate is E(target) / N(target). This is
//      refresh-proof in a way the rate trend is not: at a refresh E_i is
//      (nearly) unchanged while N jumps, so the two series are smooth where
//      the rate series steps. It also handles dilution directly: a trophy
//      whose earners grow slower than the player base falls, even while
//      more players are earning it every day.
//   3. Persistence. The last read's rate, which anchors both extrapolations
//      when the history is flat, short, or contradictory.
//
// The median is a logit-space blend of the three (0.20 persistence, 0.45
// log-time trend, 0.35 decomposition; renormalised when one is missing),
// with the trend and decomposition forecasts clipped to 3.5 logit units
// from the last rate so no extrapolation can run away, and pulled further
// towards persistence when the two trend models disagree by more than
// 2 logit units.
//
// The spread is a predictive standard deviation on the logit scale, built
// from three honest sources of error and combined in quadrature:
//   - a refresh/dilution shock floor (0.90 logit): the next refresh can
//     move every rate at once, as launch night showed, when the tracked
//     count jumped 6x and every rate but the one at 100% fell; even a
//     short horizon is not safe from one such refresh. The floor was
//     0.65 in the first submitted version; the server's practice score
//     for that version (loss 1.07, 90% coverage 0.073) showed the bands
//     far too narrow when a refresh the history could not see moved
//     every rate by ~1.8 logit at once, so the floor is raised to 0.90
//     and the total is floored at 1.0: no trophy's rate a week out is
//     known to better than about 1 logit;
//   - a random-walk term, 0.5 * sqrt(horizon in days), the house baseline's
//     own spread rate, which grows with the horizon;
//   - the trend fit's prediction error for that trophy (residual scatter
//     times the leverage of extrapolating to the target), so trophies whose
//     history is volatile or whose target is far outside the observed
//     log-time range get wider bands.
// The total is floored at 1.0 and capped at 2.2 logit. Trophies that have
// never been earned (rate 0.00 throughout) get at least 1.3, because their
// lift-off, if it comes, is a jump from zero; the one trophy pinned at
// 100% throughout gets 0.45, because every refresh so far has left it at
// 100%. Quantiles are the normal quantiles of that logit distribution,
// which is the pinball-optimal shape for a roughly normal logit error.
//
// Everything is deterministic: no clock, no randomness, no network, no
// state between runs. The same case file always produces the same bytes.
// Standard library only; runs on Node 22.
//
// Usage (as the grader runs it): node /bot/main.ts, reading /input/*.in
// and writing /output/<id>.out. Locally:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number | null;
  trophies_earned: number | null;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes?: string;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const clamp = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));

const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

interface Fit {
  a: number;
  b: number;
  /** Residual standard deviation (weighted), 0 when the fit is exact. */
  s: number;
  /** Weighted means of x and y: the line is anchored here, so shrinking
      the slope pivots the line around this point instead of shifting it. */
  xbar: number;
  ybar: number;
  sxx: number;
  n: number;
}

/** Weighted least squares of y on x. Returns null when it cannot be fit. */
function wls(
  xs: number[],
  ys: number[],
  ws: number[],
): Fit | null {
  const n = xs.length;
  if (n < 2) return null;
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  if (sw === 0) return null;
  const xbar = sx / sw;
  const ybar = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += ws[i] * (xs[i] - xbar) * (xs[i] - xbar);
    sxy += ws[i] * (xs[i] - xbar) * (ys[i] - ybar);
  }
  if (sxx <= 1e-12) return null;
  const b = sxy / sxx;
  const a = ybar - b * xbar;
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    sse += ws[i] * r * r;
  }
  // Weighted residual sd: the weighted mean squared residual, corrected
  // by n / (n - 2) for the two fitted parameters. With 2 points the fit
  // is exact, so s = 0.
  const s = n > 2 ? Math.sqrt((sse / sw) * (n / (n - 2))) : 0;
  return { a, b, s, xbar, ybar, sxx, n };
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case ${c.id} asks for levels ${JSON.stringify(c.levels)}`);
  }
  if (c.history.length === 0) throw new Error(`case ${c.id} has no history`);

  // Release clock: hours since the game's release day began (UTC). Log time
  // is the clock for a launch: the first hours move fastest.
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const t0 = Number.isFinite(releaseMs)
    ? releaseMs
    : Date.parse(c.history[0].slot);
  const hoursSinceRelease = (iso: string): number =>
    Math.max(0, (Date.parse(iso) - t0) / 3600000);
  const xOf = (iso: string): number => Math.log1p(hoursSinceRelease(iso));
  const xTarget = xOf(c.target);
  const horizonDays = c.horizon_minutes / 1440;

  // Compress the history to its change points: between refreshes the page
  // is byte-identical, and counting every flat read would overweight the
  // stale stretches. Keep a read when any rate or the tracked count moved,
  // plus always the last read.
  const distinct: Read[] = [];
  for (const r of c.history) {
    const prev = distinct[distinct.length - 1];
    if (!prev) {
      distinct.push(r);
      continue;
    }
    let changed = r.players_tracked !== prev.players_tracked;
    if (!changed) {
      for (let i = 0; i < c.trophies.length; i++) {
        if (r.rates[i] !== prev.rates[i]) {
          changed = true;
          break;
        }
      }
    }
    if (changed) distinct.push(r);
  }
  const lastRead = c.history[c.history.length - 1];
  if (distinct[distinct.length - 1] !== lastRead) distinct.push(lastRead);

  const tLast = Date.parse(lastRead.slot);
  // Recency weight: half-life 24h, so launch night's flat page (which is
  // largely a pre-refresh artefact) does not outweigh the last two days.
  const weightOf = (r: Read): number =>
    Math.pow(0.5, Math.max(0, tLast - Date.parse(r.slot)) / (24 * 3600000));

  // First pass: per-trophy trend fits, to get the cross-trophy medians
  // that shrinkage borrows from.
  interface TrophyFit {
    yLast: number;
    rLast: number;
    fitY: Fit | null;
    fitE: Fit | null;
    eLast: number;
    n: number;
  }
  const fits: TrophyFit[] = [];
  const slopesY: number[] = [];
  const slopesE: number[] = [];
  const resids: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    const xsE: number[] = [];
    const ysE: number[] = [];
    const wsE: number[] = [];
    let eLast = 0;
    for (const r of distinct) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined) continue;
      const x = xOf(r.slot);
      const w = weightOf(r);
      xs.push(x);
      ys.push(logit(rate));
      ws.push(w);
      if (r.players_tracked !== null && r.players_tracked > 0) {
        const e = (rate / 100) * r.players_tracked;
        if (e > 0) {
          xsE.push(x);
          ysE.push(Math.log(e + 0.5));
          wsE.push(w);
        }
        if (r === lastRead) eLast = e;
      }
    }
    const fitY = wls(xs, ys, ws);
    const fitE = wls(xsE, ysE, wsE);
    const rLast = lastRead.rates[i] ?? 0;
    if (fitY) {
      slopesY.push(fitY.b);
      resids.push(fitY.s);
    }
    if (fitE) slopesE.push(fitE.b);
    fits.push({ yLast: logit(rLast), rLast, fitY, fitE, eLast, n: xs.length });
  }

  // Tracked-players trend: log N on log time, over the distinct reads.
  const xsN: number[] = [];
  const ysN: number[] = [];
  const wsN: number[] = [];
  for (const r of distinct) {
    if (r.players_tracked !== null && r.players_tracked > 0) {
      xsN.push(xOf(r.slot));
      ysN.push(Math.log(r.players_tracked));
      wsN.push(weightOf(r));
    }
  }
  const fitN = wls(xsN, ysN, wsN);
  const nLast = lastRead.players_tracked ?? 0;
  const logNTarget =
    fitN && nLast > 0
      ? clamp(fitN.a + fitN.b * xTarget, Math.log(nLast), Math.log(nLast * 10))
      : nLast > 0
        ? Math.log(nLast)
        : 0;
  const nTarget = Math.exp(logNTarget);

  const slopeYMed = median(slopesY);
  const slopeEMed = median(slopesE);
  const residMed = median(resids);

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const t = c.trophies[i];
    const f = fits[i];
    const yLast = f.yLast;

    // --- Model 1: log-time trend on the logit rate, slope shrunk. ---
    let mTrend: number | null = null;
    let predVar = 0;
    if (f.fitY && f.n >= 3 && f.fitY.sxx > 0.05) {
      // Shrinkage: the slope keeps a share sxx / (sxx + 0.35) of itself
      // and borrows the rest from the field's median slope. The shrunk
      // line is anchored at the fit's weighted mean point (xbar, ybar):
      // pivoting there, never shifting the line by keeping the old
      // intercept with a new slope.
      const keep = f.fitY.sxx / (f.fitY.sxx + 0.35);
      const b = keep * f.fitY.b + (1 - keep) * slopeYMed;
      const bClipped = clamp(b, -3, 3);
      mTrend = clamp(
        f.fitY.ybar + bClipped * (xTarget - f.fitY.xbar),
        yLast - 3.5,
        yLast + 3.5,
      );
      // Prediction variance: residual scatter, leveraged by how far the
      // target sits outside the observed log-time range.
      const sRes = Number.isFinite(f.fitY.s) ? f.fitY.s : residMed;
      const lev =
        1 + ((xTarget - f.fitY.xbar) * (xTarget - f.fitY.xbar)) / f.fitY.sxx;
      predVar = sRes * sRes * lev;
    }

    // --- Model 2: earners / players decomposition. ---
    let mDecomp: number | null = null;
    if (f.fitE && f.eLast > 0.5 && nTarget > 0) {
      const keepE = f.fitE.sxx / (f.fitE.sxx + 0.35);
      const bE = clamp(
        keepE * f.fitE.b + (1 - keepE) * slopeEMed,
        -0.5,
        3,
      );
      // Anchored at the fit's weighted mean point, as in model 1.
      const logE = f.fitE.ybar + bE * (xTarget - f.fitE.xbar);
      const eTarget = Math.max(0, Math.exp(logE) - 0.5);
      const rateC = (eTarget / nTarget) * 100;
      mDecomp = clamp(logit(rateC), yLast - 3.5, yLast + 3.5);
    }

    // --- Median: blend in logit space. ---
    let m: number;
    const pinned100 = f.rLast >= 99.99;
    const neverEarned = f.rLast <= 0.009 && f.eLast <= 0.5;
    if (pinned100) {
      // Every refresh so far has left it at 100%: trust the pin.
      m = logit(99.99);
    } else if (neverEarned) {
      // No one has earned it yet. The platinum and golds (100% completion)
      // take hundreds of hours, so they stay at the floor; a bronze grind
      // trophy may just lift off, so hedge a touch above it.
      m = t.tier === "platinum" || t.tier === "gold" ? logit(0.01) : logit(0.02);
    } else if (mTrend === null && mDecomp === null) {
      m = yLast;
    } else if (mTrend !== null && mDecomp !== null) {
      if (Math.abs(mTrend - mDecomp) > 2) {
        // The two trend models disagree badly: trust persistence more.
        m = 0.4 * yLast + 0.3 * mTrend + 0.3 * mDecomp;
      } else {
        m = 0.2 * yLast + 0.45 * mTrend + 0.35 * mDecomp;
      }
    } else if (mTrend !== null) {
      m = 0.3 * yLast + 0.7 * mTrend;
    } else {
      m = 0.3 * yLast + 0.7 * (mDecomp as number);
    }
    m = clamp(m, logit(0.01), logit(99.99));

    // --- Spread: refresh shock + random walk + trend prediction error. ---
    const rw = 0.5 * Math.sqrt(horizonDays);
    let sigma = Math.sqrt(0.9 * 0.9 + rw * rw + Math.min(predVar, 1.44));
    sigma = clamp(sigma, 1.0, 2.2);
    if (neverEarned) sigma = Math.max(sigma, 1.3);
    if (pinned100) sigma = 0.45;

    const q = Z.map((z) => round4(clamp(invLogit(m + z * sigma), 0, 100)));
    // Rounding can only equalise neighbours, never invert them.
    for (let j = 1; j < q.length; j++) {
      if (q[j] < q[j - 1]) q[j] = q[j - 1];
    }
    rows.push({ id: t.id, q });
  }

  return { trophies: rows };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith(".in"))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as Case;
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
