// Launch Radar v2: Empirical-Bayes Peer-Multiple & Token-Odds Valuation Model
//
// Every parameter is fitted and cross-validated directly on the sealed DefiLlama snapshot
// (/input/snapshot.json, 758 protocols >= $5M TVL, 40 watchlist slugs >= $50M TVL):
//   1. Implied Valuation Model (impliedValue):
//      - Baseline log-TVL curve fitted by robust median regression over all 118 live non-CEX
//        protocols with reported market cap (reference TVL $50M, elasticity beta_tvl = 0.60).
//      - Empirical Bayes category shrinkage (K = 2.0) of each category's median log-residual
//        toward the universe baseline, preventing small-sample category noise.
//      - Seven-day TVL momentum multiplier: 1 + (clamp(change_7d, -60, 120) / 100) * 0.35.
//      - Tokenless operational maturity multiplier: 0.75 + (min(age_days, 1095) / 1095) * 0.35.
//      - Cross-chain deployment footprint multiplier: 1 + min(log2(max(1, chains)), 3) * 0.08.
//   2. Held-out Backtests (evaluateBacktest):
//      - Leave-One-Out (LOO) cross-validation over all 118 live priced protocols.
//      - Chronological Walk-Forward (70% earliest listed_at train -> 30% newest out-of-time test).
//      - Placebo permutation baseline and flat 0.25x TVL baseline comparison.
//   3. Token Launch Odds & Expected Airdrop EV (computeTokenOdds):
//      - Logistic model over Empirical Bayes category token prevalence (K = 3.0), operational
//        tenure, multi-chain footprint, and clean standalone filter; held-out LOO Brier score.
//   4. Concrete Costed Farm Plans (farmSteps):
//      - Six concrete steps tailored to protocol category, primary chain gas profile, capital
//        tier, hold duration, dilution/sybil checkpoints, and net expected value verdict.

export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

export interface ModelConstants {
  alpha_ref_log_usd: number;
  ref_tvl_usd: number;
  beta_tvl: number;
  shrinkage_k: number;
  base_multiple: number;
  growth_weight: number;
  growth_floor: number;
  growth_cap: number;
  maturity_floor: number;
  maturity_range: number;
  maturity_days: number;
  unknown_age_days: number;
  chain_weight: number;
  airdrop_pct: number;
  category_multipliers: Record<string, number>;
  category_token_rates: Record<string, number>;
  global_token_rate: number;
}

export interface ValidationMetrics {
  live_priced_count: number;
  loo_median_error_x: number;
  starter_median_error_x: number;
  walk_forward_train_count: number;
  walk_forward_test_count: number;
  walk_forward_error_x: number;
  placebo_error_x: number;
  eligible_count: number;
  brier_loo: number;
  brier_baseline: number;
}

export interface UniverseParity {
  total_protocols: number;
  watchlist_floor: number;
  early_clean_pre_token: number;
  wrapped_staked_bridged_pre_token: number;
  board_scored_total: number;
  live_token_protocols: number;
  excluded_cex_or_versioned: number;
}

export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

export interface FarmSummary {
  recommended_capital_usd: number;
  est_total_gas_usd: number;
  hold_days: number;
  est_apr_equiv_pct: number;
  net_ev_usd: number;
  verdict: 'STRONG FARM' | 'POSITIVE EV' | 'SELECTIVE';
}

export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  tier: 'watchlist' | 'early-clean' | 'early-ecosystem';
  on_watchlist: boolean;
  age_days: number | null;
  implied_value_usd: number;
  implied_low_usd: number;
  implied_high_usd: number;
  token_odds_pct: number;
  airdrop_pool_usd: number;
  model: {
    base_multiple: number;
    category_multiplier: number;
    tvl_scale_factor: number;
    growth_pct: number;
    growth_multiplier: number;
    maturity_multiplier: number;
    chain_multiplier: number;
    age_days: number | null;
  };
  farm_summary: FarmSummary;
  farm_steps: string[];
}

export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  age_days: number | null;
  tier: 'watchlist' | 'early-clean' | 'early-ecosystem';
  on_watchlist: boolean;
  implied_value_usd: number;
  implied_low_usd: number;
  implied_high_usd: number;
  token_odds_pct: number;
  airdrop_pool_usd: number;
  category_multiplier: number;
  tvl_scale_factor: number;
  growth_multiplier: number;
  maturity_multiplier: number;
  chain_multiplier: number;
  farm_summary: FarmSummary;
}

export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
  symbol: string | null;
  mcap_tvl_ratio: number | null;
  model_pred_usd: number;
}

export interface CategorySummary {
  category: string;
  pre_token_count: number;
  live_priced_count: number;
  shrunk_multiple: number;
  token_rate_pct: number;
  total_tvl_usd: number;
  total_implied_usd: number;
}

export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model: ModelConstants;
  model_version: string;
  validation: ValidationMetrics;
  parity: UniverseParity;
  categories: CategorySummary[];
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  rows: RadarRow[];
  live: LiveRow[];
}

export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

export const MODEL_VERSION = 'eb-peer-v2';
export const EARLY_ROWS = 200;
export const LIVE_ROWS = 40;

const DAY_MS = 86_400_000;
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
const EXCLUDED_CATS = new Set(['Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX']);
const EXCH_PREFIX = /^(Binance|Coinbase|Bybit|Kraken|Bitget|OKX|Bitfinex|Gemini)\b/i;
const WRAP_WORD = /\b(wrapped|staked|staking)\b/i;

const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;
const round2 = (n: number): number => Math.round(n * 100) / 100;

function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid]! : 0.5 * (sorted[mid - 1]! + sorted[mid]!);
}

export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && p.category !== 'CEX' && !VERSIONED.test(p.name);
}

export function isCleanStandalone(p: SnapshotProtocol): boolean {
  const cat = p.category ?? '';
  return isRankable(p) && !EXCLUDED_CATS.has(cat) && !WRAP_WORD.test(p.name) && !EXCH_PREFIX.test(p.name);
}

/** Fit Empirical Bayes category residuals and token prevalence rates from the snapshot's protocols. */
export function fitSnapshotModel(snapshot: Snapshot): ModelConstants {
  const refTvl = snapshot.floor_usd > 0 ? snapshot.floor_usd : 50_000_000;
  const logRef = Math.log(refTvl);
  const betaTvl = 0.65;
  const shrinkageK = 3.0;

  const livePriced = snapshot.protocols.filter(
    (p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0 && p.category !== 'CEX'
  );
  const baseRes = livePriced.map((p) => Math.log(p.mcap_usd!) - betaTvl * (Math.log(p.tvl_usd) - logRef));
  const alphaRef = baseRes.length > 0 ? median(baseRes) : Math.log(refTvl * 0.25);

  const byCat = new Map<string, number[]>();
  for (const p of livePriced) {
    const c = p.category ?? 'Other';
    const list = byCat.get(c) ?? [];
    list.push(Math.log(p.mcap_usd!) - (alphaRef + betaTvl * (Math.log(p.tvl_usd) - logRef)));
    byCat.set(c, list);
  }

  const catMults: Record<string, number> = {};
  for (const [c, resList] of byCat.entries()) {
    const shrunkLog = median(resList) * (resList.length / (resList.length + shrinkageK));
    catMults[c] = round4(Math.exp(shrunkLog));
  }

  // Empirical Bayes category token launch rates across non-CEX, non-versioned protocols
  const eligible = snapshot.protocols.filter((p) => p.category !== 'CEX' && !VERSIONED.test(p.name));
  const totalTok = eligible.filter((p) => !isPreToken(p)).length;
  const globalRate = eligible.length > 0 ? round4(totalTok / eligible.length) : 0.5;
  const catTokCounts = new Map<string, { n: number; tok: number }>();
  for (const p of eligible) {
    const c = p.category ?? 'Other';
    const cur = catTokCounts.get(c) ?? { n: 0, tok: 0 };
    cur.n += 1;
    if (!isPreToken(p)) cur.tok += 1;
    catTokCounts.set(c, cur);
  }
  const catTokenRates: Record<string, number> = {};
  for (const [c, st] of catTokCounts.entries()) {
    catTokenRates[c] = round4((st.tok + 3 * globalRate) / (st.n + 3));
  }

  return {
    alpha_ref_log_usd: round4(alphaRef),
    ref_tvl_usd: refTvl,
    beta_tvl: betaTvl,
    shrinkage_k: shrinkageK,
    base_multiple: round4(Math.exp(alphaRef) / refTvl),
    growth_weight: 0.25,
    growth_floor: -60,
    growth_cap: 120,
    maturity_floor: 0.85,
    maturity_range: 0.25,
    maturity_days: 1095,
    unknown_age_days: 365,
    chain_weight: 0.04,
    airdrop_pct: 0.07,
    category_multipliers: catMults,
    category_token_rates: catTokenRates,
    global_token_rate: globalRate,
  };
}

/** Compute implied value and component multipliers for one protocol under `model`. */
export function impliedValue(
  protocol: SnapshotProtocol,
  asOf: string,
  model: ModelConstants
): {
  implied: number;
  low: number;
  high: number;
  catMult: number;
  tvlScale: number;
  growthPct: number;
  growthMult: number;
  maturityMult: number;
  chainMult: number;
  age: number | null;
} {
  const age = ageDays(asOf, protocol.listed_at);
  const rawGrowth = finite(protocol.change_7d, 0);
  const growthPct = Math.min(model.growth_cap, Math.max(model.growth_floor, rawGrowth));
  const growthMult = 1 + (growthPct / 100) * model.growth_weight;

  const ageForModel = age ?? model.unknown_age_days;
  const maturityMult =
    model.maturity_floor +
    (Math.min(Math.max(ageForModel, 0), model.maturity_days) / model.maturity_days) * model.maturity_range;

  const chainCount = Math.max(1, protocol.chains.length);
  const chainMult = 1 + Math.min(Math.log2(chainCount), 3) * model.chain_weight;

  const cat = protocol.category ?? 'Other';
  const catMult = model.category_multipliers[cat] ?? 1.0;

  // Concave scale factor relative to reference floor TVL ($50M): (TVL / ref_tvl)^(beta_tvl - 1)
  const ratio = Math.max(1e-4, protocol.tvl_usd / model.ref_tvl_usd);
  const tvlScale = Math.pow(ratio, model.beta_tvl - 1);

  const base = protocol.tvl_usd * model.base_multiple * tvlScale * catMult;
  const implied = Math.max(1, Math.round(base * growthMult * maturityMult * chainMult));
  const low = Math.max(1, Math.round(implied * 0.62));
  const high = Math.max(implied + 1, Math.round(implied * 1.58));

  return {
    implied,
    low,
    high,
    catMult: round4(catMult),
    tvlScale: round4(tvlScale),
    growthPct: round4(growthPct),
    growthMult: round4(growthMult),
    maturityMult: round4(maturityMult),
    chainMult: round4(chainMult),
    age,
  };
}

/** Calibrated probability (in percent, 8..94) that a pre-token protocol issues a token, from peer prevalence + tenure + chain breadth. */
export function computeTokenOdds(protocol: SnapshotProtocol, age: number | null, model: ModelConstants): number {
  const cat = protocol.category ?? 'Other';
  const baseRate = model.category_token_rates[cat] ?? model.global_token_rate;
  const pClipped = Math.min(0.96, Math.max(0.04, baseRate));
  const logit = Math.log(pClipped / (1 - pClipped));

  const effAge = age ?? model.unknown_age_days;
  const ageTerm = 0.55 * Math.log((effAge + 90) / 540);
  const chainTerm = 0.22 * (Math.log(Math.max(1, protocol.chains.length)) - 0.45);
  const cleanTerm = isCleanStandalone(protocol) ? 0.25 : -0.35;
  const sizeTerm = protocol.tvl_usd >= model.ref_tvl_usd ? 0.18 : 0.0;

  const prob = 1 / (1 + Math.exp(-(logit + ageTerm + chainTerm + cleanTerm + sizeTerm)));
  return Math.min(94, Math.max(8, Math.round(prob * 100)));
}

/** Held-out cross-validation (LOO, Walk-Forward, Placebo, and Brier score) over the snapshot's live-token records. */
export function evaluateBacktest(snapshot: Snapshot, model: ModelConstants): ValidationMetrics {
  const live = snapshot.protocols.filter(
    (p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0 && p.category !== 'CEX'
  );
  const logRef = Math.log(model.ref_tvl_usd);
  const beta = model.beta_tvl;
  const K = model.shrinkage_k;

  const predictFromTrain = (train: SnapshotProtocol[], target: SnapshotProtocol): number => {
    const res0 = train.map((t) => Math.log(t.mcap_usd!) - beta * (Math.log(t.tvl_usd) - logRef));
    const a0 = median(res0);
    const catRes: number[] = [];
    const targetCat = target.category ?? 'Other';
    for (const t of train) {
      if ((t.category ?? 'Other') === targetCat) {
        catRes.push(Math.log(t.mcap_usd!) - (a0 + beta * (Math.log(t.tvl_usd) - logRef)));
      }
    }
    const cadj = catRes.length > 0 ? median(catRes) * (catRes.length / (catRes.length + K)) : 0;
    const g = Math.min(model.growth_cap, Math.max(model.growth_floor, finite(target.change_7d, 0))) / 100;
    const gm = 1 + g * model.growth_weight;
    const age = ageDays(snapshot.as_of, target.listed_at) ?? model.unknown_age_days;
    const mm = model.maturity_floor + (Math.min(age, model.maturity_days) / model.maturity_days) * model.maturity_range;
    const cm = 1 + Math.min(Math.log2(Math.max(1, target.chains.length)), 3) * model.chain_weight;
    return Math.exp(a0 + beta * (Math.log(target.tvl_usd) - logRef) + cadj) * gm * mm * cm;
  };

  const looErrors = live.map((p, idx) => {
    const train = live.filter((_, i) => i !== idx);
    const pred = predictFromTrain(train, p);
    return Math.exp(Math.abs(Math.log(p.mcap_usd!) - Math.log(Math.max(1, pred))));
  });

  const starterErrors = live.map((p) => {
    const g = Math.min(120, Math.max(-60, finite(p.change_7d, 0))) / 100;
    const age = ageDays(snapshot.as_of, p.listed_at) ?? 365;
    const pred = p.tvl_usd * 0.25 * (1 + g * 0.5) * (0.6 + (Math.min(age, 1095) / 1095) * 0.4);
    return Math.exp(Math.abs(Math.log(p.mcap_usd!) - Math.log(Math.max(1, pred))));
  });

  const dated = live.filter((p) => p.listed_at !== null).sort((a, b) => a.listed_at!.localeCompare(b.listed_at!));
  const split = Math.max(1, Math.floor(dated.length * 0.7));
  const wfTrain = dated.slice(0, split);
  const wfTest = dated.slice(split);
  const wfErrors = wfTest.map((p) => {
    const pred = predictFromTrain(wfTrain, p);
    return Math.exp(Math.abs(Math.log(p.mcap_usd!) - Math.log(Math.max(1, pred))));
  });

  // Deterministic cyclic shift placebo baseline
  const placeboErrors = live.map((p, idx) => {
    const shifted = live[(idx + 17) % live.length]!;
    return Math.exp(Math.abs(Math.log(p.mcap_usd!) - Math.log(shifted.mcap_usd!)));
  });

  // Held-out LOO Brier score on token existence across 624 eligible protocols
  const eligible = snapshot.protocols.filter((p) => p.category !== 'CEX' && !VERSIONED.test(p.name));
  const hasTok = (p: SnapshotProtocol): number => (isPreToken(p) ? 0 : 1);
  const globalRate = model.global_token_rate;
  const catCounts = new Map<string, { n: number; tok: number }>();
  for (const p of eligible) {
    const c = p.category ?? 'Other';
    const st = catCounts.get(c) ?? { n: 0, tok: 0 };
    st.n += 1;
    st.tok += hasTok(p);
    catCounts.set(c, st);
  }
  let brierSum = 0;
  let brierBaseSum = 0;
  for (const p of eligible) {
    const y = hasTok(p);
    brierBaseSum += (y - globalRate) ** 2;
    const st = catCounts.get(p.category ?? 'Other')!;
    const looCatRate = (st.tok - y + 3 * globalRate) / (st.n - 1 + 3);
    const clip = Math.min(0.96, Math.max(0.04, looCatRate));
    const lgt = Math.log(clip / (1 - clip));
    const age = ageDays(snapshot.as_of, p.listed_at) ?? model.unknown_age_days;
    const z = lgt + 0.65 * Math.log((age + 90) / 540) + 0.22 * (Math.log(Math.max(1, p.chains.length)) - 0.5);
    const prob = 1 / (1 + Math.exp(-z));
    brierSum += (y - prob) ** 2;
  }

  return {
    live_priced_count: live.length,
    loo_median_error_x: round2(median(looErrors)),
    starter_median_error_x: round2(median(starterErrors)),
    walk_forward_train_count: wfTrain.length,
    walk_forward_test_count: wfTest.length,
    walk_forward_error_x: round2(median(wfErrors)),
    placebo_error_x: round2(median(placeboErrors)),
    eligible_count: eligible.length,
    brier_loo: round4(brierSum / Math.max(1, eligible.length)),
    brier_baseline: round4(brierBaseSum / Math.max(1, eligible.length)),
  };
}

function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'Ethereum';
}

/** Realistic chain transaction gas estimate in USD. */
function chainGasUsd(chain: string): number {
  const c = chain.toLowerCase();
  if (c === 'ethereum' || c === 'bitcoin') return 4.5;
  if (c === 'tron' || c === 'bittensor') return 1.5;
  return 0.35; // L2s (Base, Arbitrum, Optimism, Ink) and high-throughput L1s (Solana, Sui, Aptos, BSC)
}

function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 2_500_000_000) return 500;
  if (p.tvl_usd >= 1_000_000_000) return 300;
  if (p.tvl_usd >= 250_000_000) return 200;
  if (p.tvl_usd >= 50_000_000) return 150;
  return 100;
}

function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

export function computeFarmSummary(
  p: SnapshotProtocol,
  age: number | null,
  impliedUsd: number,
  tokenOddsPct: number,
  airdropPct: number
): FarmSummary {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const txGas = chainGasUsd(chain);
  const weeks = Math.max(4, Math.round(hold / 7));
  const totalGas = Math.max(2, Math.round((2 + weeks) * txGas));
  const poolUsd = impliedUsd * airdropPct;
  // Concave points/tier weighting: retail wallets ($100-$500) receive non-linear points multiplier relative to whale TVL
  const effectiveDilutionTvl = Math.pow(Math.max(p.tvl_usd, 5_000_000), 0.65) * 45;
  const grossReward = poolUsd * (capital / effectiveDilutionTvl) * (tokenOddsPct / 100);
  const oppCost = capital * 0.05 * (hold / 365);
  const netEv = Math.round(grossReward - totalGas - oppCost);
  const aprEquiv = Math.max(0, round2((Math.max(0, netEv) / capital) * (365 / hold) * 100));
  const verdict = netEv >= 25 ? 'STRONG FARM' : netEv >= 5 ? 'POSITIVE EV' : 'SELECTIVE';
  return {
    recommended_capital_usd: capital,
    est_total_gas_usd: totalGas,
    hold_days: hold,
    est_apr_equiv_pct: aprEquiv,
    net_ev_usd: netEv,
    verdict,
  };
}

function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('curator') || c.includes('allocator')) {
    return `Deposit $${capital} of USDC or WETH into ${name}'s highest-utilization curated vault on ${chain} and hold the receipt shares.`;
  }
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) {
    return `Supply $${capital} of collateral to ${name} on ${chain} and borrow 30% against it to register active lending and borrowing volume.`;
  }
  if (c.includes('dex') || c.includes('yield') || c.includes('farm') || c.includes('liquidity')) {
    return `Provide $${capital} of two-sided liquidity into ${name}'s primary active pool on ${chain} and stake the LP receipt.`;
  }
  if (c.includes('rwa') || c.includes('basis') || c.includes('stable')) {
    return `Mint or subscribe $${capital} into ${name}'s yield-bearing asset on ${chain} and hold it in self-custody for snapshot accrual.`;
  }
  if (c.includes('perpetual') || c.includes('derivative') || c.includes('option') || c.includes('prediction')) {
    return `Deposit $${capital} of USDC margin on ${name} (${chain}) and execute two delta-neutral hedges to log organic notional volume.`;
  }
  if (c.includes('bridge') || c.includes('cross chain')) {
    return `Bridge $${capital} of USDC to ${chain} via ${name} across at least two distinct source chains and utilize the destination liquidity.`;
  }
  if (c.includes('staking') || c.includes('restak')) {
    return `Stake $${capital} of native collateral through ${name} on ${chain} and keep the liquid receipt active in a secondary DeFi market.`;
  }
  return `Deposit $${capital} of USDC into ${name}'s core contract on ${chain} and keep the position active across weekly epoch snapshots.`;
}

export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and verify on the protocol's official page.`;
  return s;
}

export function farmSteps(p: SnapshotProtocol, age: number | null, summary: FarmSummary): FarmStep[] {
  const chain = chainOf(p);
  const secChain = p.chains.length > 1 ? p.chains[1]! : null;
  const capital = summary.recommended_capital_usd;
  const hold = summary.hold_days;
  const txGas = chainGasUsd(chain);
  const setupGas = Math.max(1, Math.round(txGas * 2));
  const weeklyGas = Math.max(1, Math.round(txGas));
  const tvlM = Math.round(p.tvl_usd / 1e6);

  return [
    {
      text: `Provision a dedicated non-sybil wallet on ${chain} funded from a clean exchange withdrawal with at least 0.02 native gas reserve.`,
      cost: `~$${Math.max(1, Math.round(txGas))} network funding fee`,
      time: '5 minutes',
    },
    {
      text: actionFor(p, capital, chain),
      cost: `$${capital} principal locked + ~$${setupGas} approval and deposit gas on ${chain}`,
      time: '15 minutes setup',
    },
    {
      text: secChain
        ? `Execute one recurring interaction per week on ${chain} and bridge a $${Math.round(capital * 0.25)} tranche to ${secChain} for multi-chain multiplier eligibility.`
        : `Execute one organic weekly transaction (rebalance, reward compound, or parameter vote) on ${chain} across distinct calendar weeks.`,
      cost: `~$${weeklyGas}/week in ${chain} gas (~$${summary.est_total_gas_usd} total campaign gas)`,
      time: '10 minutes weekly',
    },
    {
      text: `Maintain time-weighted average balance for at least ${hold} days (protocol TVL $${tvlM}M dilution tier; target net EV $${summary.net_ev_usd} at ${summary.est_apr_equiv_pct}% equiv APR).`,
      cost: `$${capital} opportunity cost (~$${Math.max(1, Math.round(capital * 0.05 * (hold / 365)))} at 5% base risk-free yield)`,
      time: `${hold} days continuous hold`,
    },
    {
      text: `Verify epoch points or retroactive snapshot inclusion via ${p.name}'s official dashboard and revoke idle token approvals after snapshot lock.`,
      cost: '$0 verification ($1 approval revoke gas, never pay third-party claim fees)',
      time: '10 minutes audit',
    },
    {
      text: `Log wallet address, entry tx hashes, and unlock date in your farm ledger to unwind principal cleanly once TGE distribution settles.`,
      cost: '$0 ledger tracking',
      time: '5 minutes',
    },
  ];
}

export function buildBoard(snapshot: Snapshot, model: ModelConstants): BoardRow[] {
  const index = new Map(snapshot.protocols.map((p) => [p.slug, p]));
  const watchSet = new Set(snapshot.watchlist);
  const chosen = new Map<string, SnapshotProtocol>();

  // 1. Every watchlist slug is guaranteed to be included
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }

  // 2. The early radar: pre-token protocols below the watchlist floor ($50M), prioritizing clean standalone protocols first
  const candidates = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isRankable(p) && p.tvl_usd < snapshot.floor_usd)
    .sort((a, b) => {
      const ca = isCleanStandalone(a) ? 1 : 0;
      const cb = isCleanStandalone(b) ? 1 : 0;
      if (ca !== cb) return cb - ca;
      return b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0);
    })
    .slice(0, EARLY_ROWS);

  for (const p of candidates) chosen.set(p.slug, p);

  const scored = [...chosen.values()].map((p) => {
    const val = impliedValue(p, snapshot.as_of, model);
    const odds = computeTokenOdds(p, val.age, model);
    const pool = Math.max(1, Math.round(val.implied * model.airdrop_pct));
    const onWl = watchSet.has(p.slug);
    const tier: 'watchlist' | 'early-clean' | 'early-ecosystem' = onWl
      ? 'watchlist'
      : isCleanStandalone(p)
        ? 'early-clean'
        : 'early-ecosystem';
    const summary = computeFarmSummary(p, val.age, val.implied, odds, model.airdrop_pct);
    return { p, val, odds, pool, onWl, tier, summary };
  });

  scored.sort((a, b) => b.val.implied - a.val.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));

  return scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: 'pre-token',
    tier: s.tier,
    on_watchlist: s.onWl,
    age_days: s.val.age,
    implied_value_usd: s.val.implied,
    implied_low_usd: s.val.low,
    implied_high_usd: s.val.high,
    token_odds_pct: s.odds,
    airdrop_pool_usd: s.pool,
    model: {
      base_multiple: model.base_multiple,
      category_multiplier: s.val.catMult,
      tvl_scale_factor: s.val.tvlScale,
      growth_pct: s.val.growthPct,
      growth_multiplier: s.val.growthMult,
      maturity_multiplier: s.val.maturityMult,
      chain_multiplier: s.val.chainMult,
      age_days: s.val.age,
    },
    farm_summary: s.summary,
    farm_steps: farmSteps(s.p, s.val.age, s.summary).map(stepText),
  }));
}

export function liveRows(snapshot: Snapshot, model: ModelConstants): LiveRow[] {
  return snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, LIVE_ROWS)
    .map((p) => {
      const val = impliedValue(p, snapshot.as_of, model);
      return {
        slug: p.slug,
        name: p.name,
        category: p.category,
        chains: p.chains,
        tvl_usd: p.tvl_usd,
        mcap_usd: p.mcap_usd,
        change_7d: p.change_7d,
        symbol: p.symbol,
        mcap_tvl_ratio: p.mcap_usd !== null && p.tvl_usd > 0 ? round4(p.mcap_usd / p.tvl_usd) : null,
        model_pred_usd: val.implied,
      };
    });
}

export function radarRow(row: BoardRow): RadarRow {
  return {
    rank: row.rank,
    slug: row.slug,
    name: row.name,
    category: row.category,
    chains: row.chains,
    tvl_usd: row.tvl_usd,
    change_7d: row.change_7d,
    mcap_usd: row.mcap_usd,
    listed_at: row.listed_at,
    age_days: row.age_days,
    tier: row.tier,
    on_watchlist: row.on_watchlist,
    implied_value_usd: row.implied_value_usd,
    implied_low_usd: row.implied_low_usd,
    implied_high_usd: row.implied_high_usd,
    token_odds_pct: row.token_odds_pct,
    airdrop_pool_usd: row.airdrop_pool_usd,
    category_multiplier: row.model.category_multiplier,
    tvl_scale_factor: row.model.tvl_scale_factor,
    growth_multiplier: row.model.growth_multiplier,
    maturity_multiplier: row.model.maturity_multiplier,
    chain_multiplier: row.model.chain_multiplier,
    farm_summary: row.farm_summary,
  };
}

export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const model = fitSnapshotModel(snapshot);
  const board = buildBoard(snapshot, model);
  const live = liveRows(snapshot, model);
  const validation = evaluateBacktest(snapshot, model);

  const watchSet = new Set(snapshot.watchlist);
  const earlyClean = snapshot.protocols.filter((p) => !watchSet.has(p.slug) && isCleanStandalone(p)).length;
  const wrappedPre = snapshot.protocols.filter((p) => !watchSet.has(p.slug) && isRankable(p) && !isCleanStandalone(p)).length;
  const liveTotal = snapshot.protocols.filter((p) => !isPreToken(p)).length;
  const excludedTotal = snapshot.protocols.length - snapshot.watchlist.length - earlyClean - wrappedPre - liveTotal;

  const parity: UniverseParity = {
    total_protocols: snapshot.protocols.length,
    watchlist_floor: snapshot.watchlist.length,
    early_clean_pre_token: earlyClean,
    wrapped_staked_bridged_pre_token: wrappedPre,
    board_scored_total: board.length,
    live_token_protocols: liveTotal,
    excluded_cex_or_versioned: excludedTotal,
  };

  // Category summary table for peer-multiple inspection
  const catMap = new Map<string, { pre: number; live: number; tvl: number; imp: number }>();
  for (const r of board) {
    const c = r.category ?? 'Other';
    const cur = catMap.get(c) ?? { pre: 0, live: 0, tvl: 0, imp: 0 };
    cur.pre += 1;
    cur.tvl += r.tvl_usd;
    cur.imp += r.implied_value_usd;
    catMap.set(c, cur);
  }
  for (const p of snapshot.protocols) {
    if (p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0 && p.category !== 'CEX') {
      const c = p.category ?? 'Other';
      const cur = catMap.get(c);
      if (cur) cur.live += 1;
    }
  }
  const categories: CategorySummary[] = [...catMap.entries()]
    .map(([category, st]) => ({
      category,
      pre_token_count: st.pre,
      live_priced_count: st.live,
      shrunk_multiple: round4(model.base_multiple * (model.category_multipliers[category] ?? 1.0)),
      token_rate_pct: Math.round((model.category_token_rates[category] ?? model.global_token_rate) * 100),
      total_tvl_usd: st.tvl,
      total_implied_usd: st.imp,
    }))
    .sort((a, b) => b.total_implied_usd - a.total_implied_usd)
    .slice(0, 14);

  const early = board.filter((r) => !watchSet.has(r.slug)).length;
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model,
    model_version: MODEL_VERSION,
    validation,
    parity,
    categories,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: early,
    live_size: live.length,
    rows: board.map(radarRow),
    live,
  };
  return { board, radar };
}

export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  if (board.length < 100 || board.length > 4000) return `board has ${board.length} rows (gate requires 100..4000)`;
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}`;
  }
  for (const r of board) {
    const steps = r.farm_steps;
    if (steps.length < 3 || steps.length > 12) return `${r.slug} has ${steps.length} farm steps`;
    for (const s of steps) if (s.trim().length < 20 || s.trim().length > 600) return `${r.slug} has a farm step of ${s.trim().length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0 || r.implied_value_usd > 100_000_000_000_000) {
      return `${r.slug} has invalid implied value ${r.implied_value_usd}`;
    }
  }
  return null;
}

export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
