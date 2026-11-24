# token-radar

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/quantran2k/token-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/quantran2k/token-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

`token-radar` (`eb-peer-v2`) is a pre-token launch valuation and farming research desk built on the sealed DefiLlama protocol snapshot (`758` protocols $\ge$ $5M TVL, `40` watchlist protocols $\ge$ $50M TVL). It fits an Empirical-Bayes peer-multiple curve and logistic token-issuance odds model over all `119` live non-CEX benchmark protocols (held-out LOO median error `x3.66` vs flat starter `x5.24`, chronological walk-forward `x3.27`, token odds LOO Brier `0.2232`), ranks `226` pre-token protocols, and generates concrete 6-step costed farming plans with chain-specific gas, dilution, and net expected value verdicts.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input ../token-radar-01-kit/input/snapshot.json
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=p2zspbr7&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
