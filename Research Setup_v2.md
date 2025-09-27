# Ultimate Research Setup

## 1. Research Overview

---

### Core Research Question

**Does adding machine learning principles such as IRM regularization help hedging strategies maintain risk control when tested under unseen volatility or skew regimes?**

* **Unseen volatility and skew regimes:** Level shifts from low to high volatility environments
* **Maintain risk control:** Measured by CVaR-95 of hedging P&L, net of transaction costs

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

| **Volatility** | **Bands** | **Real Word Example**                       | **Observed data**                                                                                         | **Interpretation**                                                                      |
| -------------- | --------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Low            | <10%      | Mid-2000s calm bull market, e.g. 2004–2006. | Realized S&P 500 volatility averaged ~8–10% annualized, with VIX often under 12.                          | Stable growth, low uncertainty. A “boring” and calm  market environment.                |
| Medium         | 10-25%    | 2013–2017 post-GFC recovery.                | Realized volatility typically 12–20% annualized, VIX bouncing around 15–20.                               | Moderate daily swings, option prices carry normal skew. Considered the baseline regime. |
| High           | 25-40%    | Late-2018 selloff (Volmageddon)             | Realized vol reached 25–35%, with VIX spiking into the 30–40 range.                                       | Stress markets but not real collapse                                                    |
| Crisis         | >40%      | **COVID-19 crash, March 2020.**             | VIX averaged ~58 (≈ 3.6% daily σ), peaking above 80; realized vol on the S&P 500 exceeded 60% annualized. | True crisis regime; hedger trained on calmer data typically blow up                     |

## 3. Experimental Design

---

According to the Invariant Risk Minimization research, IRM requires sufficiently diverse training environments to identify invariances. The current vol-skew pairs are baseline for the training environment. We will evaluate whether adding environments with jump dynamics (Merton model) or mild liquidity frictions improve IRM stability.

### Models to Compare

* **ERM-base** (deep hedger baseline).
* **ERM-reg** (ERM-base + strong regularization):

  * weight decay = 1e-3, dropout = 0.3 on hidden layers, label smoothing = 0.05.
* **GroupDRO**
* **V-REx**
* **IRM-v1** (main method) with λ grid and adaptive schedule.

### Pre-Commit Success Criteria

* **Primary:** IRM (or any method) must **improve Crisis CVaR-95** vs ERM-base by **≥ 10%** and be **statistically significant** (non-overlapping 95% CIs across 30 seeds).
* **Secondary guards:** turnover not more than **+20%** vs ERM-base; mean P&L not **≤ −10%** of ERM-base.
* **If ERM-reg ties IRM on CVaR within CI** and beats it on simplicity, record as “no advantage of IRM over strong regularization.”

### Narrative Hedge

If IRM underperforms: *“IRM did not beat ERM-reg under volatility-only shifts; however, under added **jump** and **liquidity** environment diversity, IRM narrowed the tail-loss gap. This suggests the **value of invariance grows with environment heterogeneity**, aligning with IRM theory. Future work: richer environments and nonlinear invariance penalties.”*

### 3.1 Training & Evaluation Plan (Two-Run Compact Design)

* **Run 1 (Training & Validation):**

  * Train on Synthetic Low + Medium volatility and Real pre-COVID (2017–2019), split into volatility terciles.
  * Validate on Synthetic High volatility for λ selection and early stopping.
  * Models trained: ERM head (average risk) and IRM head (risk + invariance penalty).
  * Crisis data is not used in training/validation to prevent leakage.

* **Run 2 (Evaluation):**

  * Evaluate saved checkpoints from Run 1 across unseen regimes:

    1. **Synthetic → Synthetic Crisis** (internal validity)
    2. **Real pre-COVID → Real COVID (2020)** (external validity)
    3. **Synthetic → Real COVID** (cross-generalization)
  * **Robustness probes (eval-only, no retraining):**

    * Cost stress: re-run Real COVID with +50% slippage/fees.
    * Spurious feature mask: re-run Synthetic Crisis with realized vol feature removed. IRM should degrade less than ERM.

---

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

* Recent realized volatility (trailing 20-day estimate)
* Changes across environments and should not drive hedging decisions

## 5. Training Protocol

---

Because IRM is sensitive to the regularization parameter λ, *we will experiment with adaptive λ scheduling*:

* Monitor ERM loss vs. invariance penalty trade-off.
* Adjust λ upward until validation CVaR improves but training loss does not collapse.
* Use Pareto-frontier analysis to balance predictive power and invariance.

### 5.1 Training Phases

1. **ERM Pretrain:** 20,000 gradient steps, batch size 128
2. **IRM Ramp:** Linear ramp λ from 0→λ_target over 10,000 steps
3. **Full Training:** Continue for 150,000 total updates

### 5.2 Optimization Settings

* **Optimizer:** Adam
* **Learning rate:** 1e-4
* **Gradient clipping:** 1.0
* **Weight decay:** 1e-6
* **Mini-batches:** Environment-balanced (equal samples per environment)

### 5.3 Hyperparameter Tuning

* **λ grid:** {0, 0.01, 0.1, 1, 10}
* **Tune on:** High environment validation set
* **Monitor:** Per-environment loss curves and gradient penalty to avoid collapse

## 6. Evaluation Framework

---

### Baselines

* Delta hedging
* Delta + Vega hedging
* ERM (standard deep hedging)
* GroupDRO
* V-REx

### Metrics

* **Primary:** CVaR-95 of P&L
* **Secondary:** Mean P&L, Portfolio turnover, Maximum drawdown, Sharpe ratio

### Test Protocol

* Episode length: 60 trading days
* Test episodes: 1,000 per seed on Crisis environment
* Seeds: 30 different seeds
* Report: Mean and 95% CI for all metrics

## 7. Analysis & Visualization

---

### Required Plots

1. Training curves (per-environment loss and gradient penalty)
2. Conditional P&L distributions
3. Feature sensitivity (IRM vs ERM on spurious vol)
4. QQ-plot of P&L tails (ERM vs IRM on Crisis)
5. λ-sweep (CVaR-95 OOD vs λ)

### Diagnostics

* Verify gradient penalty doesn't cause collapse
* Confirm invariant features remain predictive across environments
* Test that r(S) sensitivity decreases with IRM

## 8. Implementation & Reproducibility

---

### Technical Stack

* **Logging:** Weights & Biases (W&B)
* **Configuration:** Hydra configs
* **Version control:** Git with commit hash tracking

### Reproducibility Archive

* `requirements.txt` with pinned versions
* Hydra configuration files
* `seed_list.txt` with 30 seeds
* `make reproduce` command for evaluation with seed 0
* Commit hash: [to be filled on release]

### Extra Notes

* Section 3.1 (Training Environments): “We fix an ATM, 60-day constant-maturity call; daily re-hedging; notional $1. We roll strike daily to remain ATM-60D.”
* Section 3.2 (Data Generation → Real data): include the *Real-data preprocessing* paragraph.
* Section 5 (Training Protocol): add “ERM-reg” config.
* Section 6 (Evaluation): include pre-commit thresholds for CVaR, turnover, mean P&L.
* Section 7 (Analysis): add coverage table and spread-sensitivity plot.
* Turnover diagnostics: enable `HIRM_DEBUG_PROBE=1` to log structured probes.
* Trade spike guard: `train.max_trade_warning_factor` logs max-trade metrics.
* Baselines packaging: run `scripts/run_baseline.py` to refresh ERM baseline; outputs land in `outputs/_baseline_erm_v1/summary.csv`.
