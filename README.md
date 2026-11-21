# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arbitojeep/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arbitojeep/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one: for every PlayStation 5 trophy, five quantiles of its Exophase earn rate at the case target, from the case's own history of Exophase reads. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## How it works

An earn rate is a ratio, earners / players tracked, and Exophase refreshes its counts on its own schedule, hours apart, so the history is flat between refreshes and steps at them. The forecaster in `main.ts` models the two forces that move a rate separately, per trophy, and blends them with persistence:

1. **Log-time trend on the logit rate** — weighted least squares of logit(rate) on ln(1 + hours since release), slope shrunk towards the cross-trophy median and anchored at the fit's weighted mean point.
2. **Earners / players decomposition** — earners and the tracked count are each fitted in logs against the same log-time clock, then the forecast rate is earners(target) / players(target). This is refresh-proof: at a refresh the earners are nearly unchanged while the denominator jumps.
3. **Persistence** — the last read's rate, anchoring both extrapolations.

The spread is a predictive standard deviation on the logit scale from a refresh/dilution shock floor, a random-walk term growing with the horizon, and each trophy's trend-fit prediction error. See the comments at the top of `main.ts` for the full reasoning.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network, deterministic: the same case always produces the same forecast.

---

<sub>Entered on [make.fun](https://make.fun/?ref=arbit&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
