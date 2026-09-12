# accounting

What a position earned and what it cost, in dollars per day.

**One component, called at three levels.** Only the position handed in differs:

| Level | Position handed in | Rounded | Buffered |
|---|---|---|---|
| Rule (stage 3) | one rule's forecast, sized as if it traded alone | no | no |
| Subsystem (stage 6) | the combined forecast for an instrument | yes | yes |
| Portfolio (stage 9) | subsystem position × instrument weight × IDM | yes | yes |

Everything after that is identical, so this code never learns which level it is
serving: no `level` argument, no branching. **Accounting measures a position; it
never decides one.** Whatever chooses the position lives upstream.

## Files

| File | Job |
|---|---|
| `timing.py` | The one place in the codebase that shifts anything |
| `accounting.py` | `Contract`, the cost model, and `account()` |
| `rule_level.py` | Stage 3: a *caller*, applying the component to each rule |

`timing.py` is small and still separate. Both the cost code and the P&L code
need it, so folding it into either would make the other import it and create a
circular import. Keeping it a leaf module also makes "exactly one file shifts
anything" a claim you can check by grep.

## Timing: the whole point of the rebuild

A forecast labelled Monday comes from Monday's close, so it does not exist until
Monday is over. A price change labelled Monday happened *before* that forecast
existed. Multiplying the two pays you for the day whose information you used to
decide — and nothing errors, because pandas aligns both by date quite happily.

Measured across the eight instrument/rule pairs here:

| Convention | Mean Sharpe |
|---|---|
| No shift — pairing a forecast with its own day's move | **1.164** |
| `TRADE_LAG_DAYS = 0` — trade at the signal's own close | 0.307 |
| `TRADE_LAG_DAYS = 1` — trade at the next close | 0.308 |

Nearly 4× inflation, none of it real. And the realistic convention is free:
0.307 against 0.308.

```
close t              close t+1            close t+2
  |                    |                    |
  decide, using        trade here, at       this day's price change is
  everything up        this close           the first the new position
  to this close                             earns
```

```python
held   = decision.shift(trade_lag + 1)   # pair with price change
traded = decision.shift(trade_lag)       # its changes are the trades
```

The two shifts have different reasons, which is why the code says
`trade_lag + 1` rather than 2. One is an execution assumption that could change;
the other is structural, since a price change labelled *t* covers *t−1 → t*.

`account()` takes the **decision** and applies the lag itself, so a caller
cannot express a different convention. Costs come from the same call, which is
why they cannot drift from the P&L — the old build priced trades on one day and
booked the matching P&L on another, with nothing to reveal it.

## Costs, in dollars

No Sharpe ratios anywhere in the pipeline. Gross is contracts × price move ×
`currency_per_point`; costs are dollars; net is the difference. Ratios belong to
reporting and to the optimiser, which derive them from the dollars.

pysystemtrade expresses rule-level costs in Sharpe units because it runs
rule-level P&L on an arbitrary capital of 100, where a dollar cost would be
meaningless. We use real capital, which works because net P&L is exactly linear
in capital — doubling capital doubles every number and changes no ratio.

**Per trade**, on every contract that changes hands:

| Instrument | Spread | Commission | Total |
|---|---|---|---|
| CORN | $8.00 | $2.92 | $10.92 |
| SOFR | $7.50 | $1.50 | $9.00 |
| SP500_micro | $0.95 | $0.57 | $1.52 |
| US10 | $8.00 | $1.51 | $9.51 |

The spread dominates — 73–84% of the total for three of the four. Commission is
the largest of three structures, never their sum: per contract, a percentage of
traded value, or a flat charge per order.

**Rolls are a tax on holding.** A contract expires; to keep the position you
close it and open the next, paying on the whole position whether or not the
forecast changed. For the *slow* rules that is about 40% of all costs, because a
slow rule barely trades:

```
 instrument         rule  trade_cost_per_yr  roll_cost_per_yr  roll_share
       CORN   ewmac_8_32            $17,766            $1,193        6.3%
       SOFR ewmac_32_128             $6,109            $4,154       40.5%
       US10 ewmac_32_128             $2,731            $1,808       39.8%
```

Ignoring them would make slow rules look systematically cheaper right before
stage 4 chooses between fast and slow.

Roll dates are **synthesised** by spacing `rolls_per_year` evenly through each
year, and charged against the position actually held that day — so a position
that happens to be flat pays nothing. Smearing the annual cost across every day
would charge for rolling a position you do not hold.

### Why not the real roll calendars

`roll_calendars_csv` exists and was rejected, which is worth recording so it
does not read as an oversight:

- **SOFR's file covers only 2020-03 to 2021-09**, with rolls as little as 12
  days apart on contracts expiring one to three years later — the fingerprint of
  a data back-fill, not a trading record. It implies 7.43 rolls a year against a
  configured 4.
- **Every calendar stops two to three years before the prices do** (2020–2022
  against 2023-08), so recent years would be charged nothing.

The price data independently confirms the configured frequency. Contract changes
in `multiple_prices` run at 1.00, 4.18, 3.98 and 4.00 per year for CORN, SOFR,
SP500_micro and US10, against a configured 1, 4, 4, 4.

## Assumptions, and which way each errs

- **Costs are present-day costs applied to forty years of history.** Trading in
  the 1970s and 1980s cost considerably more — wider spreads, floor execution —
  so **early-period net results are optimistic**, increasingly so further back.

  pysystemtrade scales historical costs by volatility, assuming spreads widen
  with volatility. We do not. Because these markets were *quieter* in the past,
  that adjustment would *lower* historical costs and raise net Sharpe — CORN's
  fast rule from 0.266 to 0.324, SP500_micro's from −0.140 to −0.065 — on an
  assumption this data cannot verify. Declining an unverifiable assumption that
  flatters results is the more conservative of the two available choices, though
  neither corrects the underlying optimism.

- **Cost is strictly proportional to contracts traded**, which is what makes net
  P&L linear in capital. A flat per-order commission would break it, being
  charged once regardless of size; `account()` refuses a non-zero one rather
  than silently distorting comparisons between rules. All four instruments have
  `commission_per_trade = 0`.

- **Positions are not rounded here.** Rounding is a step function and would also
  break linearity. It belongs at the levels where orders are real.

- **The adjusted price is never used as a money amount.** It goes negative, so a
  percentage commission needs the current contract's real price; `account()`
  demands it rather than quietly charging a percentage of −36.4.

## Using it

```python
from refactory_2026.accounting.accounting import account_for
from refactory_2026.accounting.rule_level import rule_accounts, rule_net_returns

account_for("US10", decision)     # gross, costs, net for any position series
rule_accounts("US10")             # per rule, for one instrument
rule_net_returns()                # net dollars per rule per instrument: stage 4's input
```

`account()` itself takes primitives and a `Contract`, not a config row, so it can
be tested with nothing on disk and cannot break when a column in
`instruments.py` is renamed — which has already happened twice.
