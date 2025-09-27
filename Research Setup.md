# Ultimate Research Setup

## 1. Research Overview

---

### Core Research Question

**Does adding machine learning principles such as IRM regularization help hedging strategies maintain risk control when tested under unseen volatility or skew regimes?**

- **Unseen volatility and skew regimes:** Level shifts from low to high volatility environments
- **Maintain risk control:** Measured by CVaR-95 of hedging P&L, net of transaction costs

This research investigates whether IRM can create more robust hedging strategies by:

1. Identifying invariant features (normalized Greeks, time-to-maturity, inventory)
2. Ignoring spurious features (recent realized volatility)
3. Training across diverse volatility environments
4. Testing on extreme out-of-distribution crisis scenarios

H1: Policies trained with IRM regularisation achieve significantly lower CVaR-95 hedging error in out-of-distribution regimes than policies trained under ERM or DRO.

H2: IRM-trained policies show smaller degradation when volatility, skew, or liquidity regimes shift, compared to ERM-trained policies.

H3: Applying IRM on risk-estimator heads (not raw actions) yields better robustness–performance trade-offs than applying it directly to policy outputs.

H4 (falsification test): If invariance does not exist in financial features, IRM regularisation will either underperform ERM or force spurious invariances, degrading OOD hedging.

### The Gap

While regime-based robustness has been explored with robust optimization and adversarial training, very few works apply IRM explicitly to finance. In practice, hedging models often appear robust in-sample but fail when volatility clusters change regime (2007–09 GFC, 2018 Volmageddon, 2020 COVID crash). *This project asks whether IRM can extract “causal-like” invariances from Greeks rather than regime-specific features like realized volatility.*

### Practical Impact

Financial institutions lose billions due to model failure during regime changes. A more robust hedging approach could have real economic value.

### Why IRM?

Financial markets have spurious patterns that work until they don't—like how tech stocks were 'defensive' until COVID hit. IRM is the only method that specifically identifies which patterns are causal versus spurious. While robust optimization prepares you for bad scenarios, IRM ensures you're using the right features to be robust. It's the difference between building a bunker randomly versus building it where the storm will actually hit.

Beyond vol-level shifts, regime diversity can also include **jump risk** (modeled with Poisson jumps in price dynamics) and **liquidity stress** (proxied by widening bid–ask spreads in crisis). If time permits, these may be introduced as additional “stress environments” to test IRM’s extrapolation beyond volatility-only perturbations.

## 2. Volatility Regime Classification

| **Volatility** | **Bands** | **Real Word Example** | **Observed data** | **Interpretation** |
| --- | --- | --- | --- | --- |
| Low | <10% | Mid-2000s calm bull market, e.g. 2004–2006. | Realized S&P 500 volatility averaged ~8–10% annualized, with VIX often under 12. | Stable growth, low uncertainty. A “boring” and calm  market environment. |
| Medium | 10-25% | 2013–2017 post-GFC recovery. | Realized volatility typically 12–20% annualized, VIX bouncing around 15–20. | Moderate daily swings, option prices carry normal skew. Considered the baseline regime. |
| High | 25-40% | Late-2018 selloff (Volmageddon) | Realized vol reached 25–35%, with VIX spiking into the 30–40 range. | Stress markets but not real collapse |
| Crisis | >40% | **COVID-19 crash, March 2020.** | VIX averaged ~58 (≈ 3.6% daily σ), peaking above 80; realized vol on the S&P 500 exceeded 60% annualized. | True crisis regime; hedger trained on calmer data typically blow up |

## 3. Experimental Design

---

According to the Invariant Risk Minimization research, IRM requires sufficiently diverse training environments to identify invariances. The current vol-skew pairs are baseline for the training environment. We will evaluate whether adding environments with jump dynamics (Merton model) or mild liquidity frictions improve IRM stability.

**Models to compare**

- **ERM-base** (your current deep hedger).
- **ERM-reg** (same as ERM-base + strong regularization):
    
    weight decay = 1e-3, dropout = 0.3 on hidden layers, label smoothing = 0.05.
    
- **GroupDRO**, **V-REx** (as planned).
- **IRM-v1** (your main method) with λ grid and your adaptive schedule.

**What “winning” means (pre-commit criteria)**

- **Primary:** IRM (or any method) must **improve Crisis CVaR-95** vs ERM-base by **≥ 10%** and be **statistically significant** (non-overlapping 95% CIs across 30 seeds).
- **Secondary guards:** turnover not more than **+20%** vs ERM-base; mean P&L not **≤ −10%** of ERM-base.
- **If ERM-reg ties IRM on CVaR within CI** and beats it on simplicity (fewer hyperparams), record as “no advantage of IRM over strong regularization” — still a publishable negative result.

**Narrative hedge (what you’ll say if IRM underperforms)**

- “IRM did not beat ERM-reg under volatility-only shifts; however, under added **jump** and **liquidity** environment diversity, IRM narrowed the tail-loss gap. This suggests the **value of invariance grows with environment heterogeneity**, aligning with IRM theory. Future work: richer environments and nonlinear invariance penalties.”

### 3.1 Training Environments

### Environment Structure

1. **Training (In-Distribution):** Low and Medium volatility (0-25%)
2. **Validation & Hyperparameter Tuning:** High volatility (25-40%)
3. **Test (Out-of-Distribution):** Crisis volatility (>40%)

### Training Environment Grid

Four training environment variations by sampling pairs (σ, skew): 1) Low σ & low skew; 2) Low σ & high skew; 3) Medium σ & low skew; 4) Medium σ & high skew.

### 3.2 Data Generation

Synthetic environments will be complemented with real market datasets. This include:

- **Options & Futures:** SPY options and S&P E-mini futures from 2016–2023 (CBOE & CME data vendors).
- **Crisis Windows:** 2018 Volmageddon, March 2020 COVID crash, and 2022 inflation shock.
- Preprocessing: filter daily close prices, interpolate missing vols, compute realized σ and skew.

This ensures synthetic Heston stress tests have a **real-data anchor** for external validity.

### Models

- **Low/Medium environments:** Geometric Brownian Motion (GBM)
    - Appropriate for calm and low-volatility markets
- **High/Crisis environments:** Heston model
    - Parameters: θ = 0.25, vol-of-vol σ_v = 0.6
    - Naturally encodes volatility clustering for realistic crisis modeling

### Sampling Rules

- **Volatility sampling:** Uniform random within each band
    - Low: σ ~ Uniform[0.08, 0.12]
    - Medium: σ ~ Uniform[0.15, 0.25]
- **Rationale:** IRM relies on diversity within environments to identify spurious vs. invariant features

### Data Requirements

- 10,000 paths per environment
- Report for each environment:
    - Sample mean
    - Sample kurtosis
    - Tail 1% quantile
    - ACF(1) of squared returns
    - Realized volatility histogram
- Tune Heston parameters to match empirical COVID-19 moments

### 3.3 Market Realism Parameters

*Real-data preprocessing:* We restrict to SPY options with open interest ≥ 500 and daily volume ≥ 50; quotes must have a non-zero bid and bid–ask spread ≤ 5% of mid (or ≤ $0.15). P&L is marked at mid; a conservative variant applies 25% of spread as slippage per side. We add slippage = 10% of spread during crisis days. Days failing filters are skipped, and coverage statistics are reported.

- **Instrument:** SPY European-style synthetic call (cash-settled in sim).
- **Moneyness:** ATM at trade time (strike = spot).
- **Maturity:** **60 calendar days** constant-maturity (roll strikes daily to stay ATM-60D).
- **Episode length:** keep your 60 trading days.
- **Rebalancing:** once per day (EOD) for base runs; add an hourly variant only in a robustness appendix.
- **Position sizing:** notional = $1 per episode for clean comparability.
- **Greeks/pricing:** Black–Scholes with σ_imp from the VRP rule you already defined.

### Transaction Costs

- **Low/Medium environments:**
    - Linear cost: 5 bps (0.0005)
    - Quadratic cost: 0
- **High/Crisis environments:**
    - Linear cost: 20 bps (0.002)
    - Quadratic cost: 0.1

### Option Pricing

- Implied volatility: σ_imp = σ_realized + VRP
- Variance Risk Premium (VRP) by regime:
    - Low/Medium: VRP = 0.05
    - High: VRP = 0.15
    - Crisis: VRP = 0.25
- Options priced with σ_imp, but P&L realizes under physical dynamics

## 4. Hypothesis & Feature Engineering

---

Greeks (Δ, Γ, Θ) are derived from no-arbitrage pricing rules and hold across volatility regimes (structural invariants). In contrast, realized volatility estimates are regime-sensitive and nonstationary. *This theoretical grounding justifies treating Greeks as invariant candidates and realized σ as spurious r(S).*

### Core Hypothesis

The optimal policy depends on features Φ(S) that remain invariant across volatility environments, while nuisance features r(S) are spurious.

### Invariant Features Φ(S)

```markdown
Φ(S) = [
    delta / notional,
    gamma / notional,
    time_to_maturity_days / 252,
    inventory / notional
]
```

All features standardized using training environment statistics only.

### Spurious Feature r(S)

- Recent realized volatility (trailing 20-day estimate)
- Changes across environments and should not drive hedging decisions

## 5. Training Protocol

Because IRM is sensitive to the regularization parameter λ, *we will experiment with adaptive λ scheduling*:

- Monitor ERM loss vs. invariance penalty trade-off.
- Adjust λ upward until validation CVaR improves but training loss does not collapse.
- Use Pareto-frontier analysis to balance predictive power and invariance.

### 5.1 Training Phases

1. **ERM Pretrain:** 20,000 gradient steps, batch size 128
2. **IRM Ramp:** Linear ramp λ from 0→λ_target over 10,000 steps
3. **Full Training:** Continue for 150,000 total updates

### 5.2 Optimization Settings

- **Optimizer:** Adam
- **Learning rate:** 1e-4
- **Gradient clipping:** 1.0
- **Weight decay:** 1e-6
- **Mini-batches:** Environment-balanced (equal samples per environment)

### 5.3 Hyperparameter Tuning

- **λ grid:** {0, 0.01, 0.1, 1, 10}
- **Tune on:** High environment validation set
- **Monitor:** Per-environment loss curves and gradient penalty to avoid collapse

## 6. Evaluation Framework

---

For baselines, we will implement:

- **GroupDRO:** Using PyTorch implementations with environment-weighting schedules.
- **V-REx:** Following the implementation details in the original paper.
- **Delta/Vega hedging:** Closed-form Greeks and replication to benchmark against traditional finance practice.

This ensures comparisons are reproducible and fair, avoiding reviewer concerns about implementation bias.

### 6.1 Baselines

- Delta hedging
- Delta + Vega hedging
- ERM (standard deep hedging)
- GroupDRO
- V-REx

### 6.2 Metrics

- **Primary:** CVaR-95 of P&L
- **Secondary:**
    - Mean P&L
    - Portfolio turnover
    - Maximum drawdown
    - Sharpe ratio

### 6.3 Test Protocol

- Episode length: 60 trading days
- Test episodes: 1,000 per seed on Crisis environment
- Seeds: 30 different seeds
- Report: Mean and 95% CI for all metrics

## 7. Analysis & Visualization

---

### Required Plots

1. **Training curves:** Per-environment loss and gradient penalty over time
2. **Conditional P&L distributions:** P&L | Φ-bins (e.g., deciles of delta_norm) across train/val/test
3. **Feature sensitivity:** Rank features by occlusion/perturbation; show IRM reduces sensitivity to r(S)
4. **QQ-plot:** P&L tails comparison (ERM vs IRM) on Crisis
5. **λ-sweep:** CVaR-95 (OOD) vs λ with confidence intervals

### Diagnostic Checks

- Verify gradient penalty doesn't cause collapse
- Confirm features Φ(S) remain predictive across environments
- Test that r(S) sensitivity decreases with IRM

## 8. Implementation & Reproducibility

### Technical Stack

- **Logging:** Weights & Biases (W&B)
- **Configuration:** Hydra configs
- **Version control:** Git with commit hash tracking

### Reproducibility Archive

- `requirements.txt` with pinned versions
- Hydra configuration files
- `seed_list.txt` with 30 seeds
- `make reproduce` command for evaluation with seed 0
- Commit hash: [to be filled on release]

- **Section 3.1 (Training Environments):** “We fix an ATM, 60-day constant-maturity call; daily re-hedging; notional $1. We roll strike daily to remain ATM-60D.”
- **Section 3.2 (Data Generation → Real data):** paste the *Real-data preprocessing* paragraph above.
- **Section 5 (Training Protocol):** add “ERM-reg” config (weight decay 1e-3, dropout 0.3, label smoothing 0.05).
- **Section 6 (Evaluation):** add the pre-commit thresholds for CVaR, turnover, mean P&L.
- **Section 7 (Analysis):** add a **coverage table** and a **spread-sensitivity plot** (CVaR vs allowed spread cap).
- **Turnover diagnostics:** set the environment variable `HIRM_DEBUG_PROBE=1` before launching training to print the first-episode probe (price, previous position, raw NN output, final position, trade, cost) for the first 20 steps only. Leave it unset for clean runs.
- **Trade spike guard:** `train.max_trade_warning` (default 50.0) logs a warning if any single trade magnitude exceeds the threshold. This is meant to catch the “seed 1” churn bug early.
- **Baseline automation:** run `scripts/run_baseline.py --steps 20000 --seeds 0-4` to retrain + evaluate the ERM baseline. The script writes `baseline_summary.csv` and (optionally) keeps per-seed evaluation artifacts under `runs_eval/` when `--keep-eval` is supplied.
