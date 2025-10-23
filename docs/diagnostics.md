# Diagnostic Metrics (Track 4)

This document summarises the diagnostic metrics implemented for Track 4. The
metrics reproduce the equations from Section 4.3 of the paper and provide the
inputs required by the downstream visualisation tasks (Track 5).

## Invariance Diagnostics

### Invariance Spectrum Index (ISI)

ISI aggregates three stability components across probes:

- **C1 – Global Stability** uses the variance of per-environment risks
  \(R_e\):
  \[
  C_1 = 1 - \min\left(1, \frac{\widehat{\operatorname{Var}}_e[R_e]}{\tau_R + \epsilon}\right).
  \]
  Variance is computed after averaging risk within each environment.

- **C2 – Mechanistic Stability** evaluates the cosine alignment of head
  gradients.  For every environment pair \((e, e')\) we take the 10% trimmed
  mean of the recorded cosines and normalise to \([0, 1]\):
  \[
  C_2 = \frac{1}{Z} \sum_{e < e'} \frac{1 + \operatorname{cos}(\nabla_\psi R_e,\nabla_\psi R_{e'})}{2}.
  \]

- **C3 – Structural Stability** uses feature dispersion statistics recorded for
  each probe:
  \[
  C_3 = 1 - \min\left(1, \frac{\operatorname{Disp}(\mathbb{E}[z\cdot \ell])}{\tau_C + \epsilon}\right).
  \]

Each component is aggregated across probes using a symmetric 10% trimmed mean.
ISI is the weighted sum \(\sum_i \alpha_i \widetilde{C_i}\).  The weights are
configurable (default: equal weights).

### Invariance Gap (IG)

The invariance gap is evaluated over the test environments:
\[
\mathrm{IG} = \max_{e, e' \in E_\text{test}} |R_e - R_{e'}|.
\]
The implementation optionally reports the normalised gap \(\mathrm{IG}_\text{norm} =
\mathrm{IG} / (\tau_{\mathrm{IG}} + \epsilon)\).

## Robustness Diagnostics

### Worst-Case Generalisation (WG)

WG is computed via the Rockafellar–Uryasev CVaR objective across environment
risks:
\[
\mathrm{WG}_\alpha = \inf_\tau \left[\tau + \frac{1}{\alpha} \mathbb{E}_e[(R_e - \tau)_+]\right].
\]
The optimum is obtained by evaluating the objective at every observed risk
level.  The tail parameter \(\alpha\) defaults to 0.25.

### Volatility Ratio (VR)

The volatility ratio for a rolling risk series \(R_t\) is
\(\mathrm{VR} = \operatorname{Std}[R_t] / (\mathbb{E}[R_t] + \epsilon)\).

## Efficiency Diagnostics

### Expected Return–Risk Efficiency (ER)

ER uses the mean PnL and the lower-tail CVaR at the 5% level:
\[
\mathrm{ER} = \frac{\mathbb{E}[\text{PnL}_t]}{\text{CVaR}_{0.05}(\text{PnL}_t) + \epsilon}.
\]
The CVaR is defined as the average of the worst 5% returns.

### Turnover Ratio (TR)

TR compares the mean \(\ell_2\) turnover to the average position magnitude:
\[
\mathrm{TR} = \frac{\mathbb{E}[\lVert \Delta a_t \rVert_2]}{\mathbb{E}[\lVert a_t \rVert_2] + \epsilon}.
\]

## Aggregation Outputs

Running `python tools/scripts/aggregate_diagnostics.py --run_dir <path>` produces the
following tables inside `<run_dir>/tables/`:

- `diagnostics_summary.csv`: one row per seed × regime × split with ISI, IG,
  robustness, and efficiency metrics.
- `invariance_diagnostics.csv`: aggregate and per-probe breakdown of C1, C2,
  C3, and ISI together with the invariance gap.
- `robustness_diagnostics.csv`: WG and VR diagnostics along with auxiliary
  statistics (tail parameter, mean, standard deviation).
- `efficiency_diagnostics.csv`: ER, TR, and the intermediate turnover/position
  statistics.
- `capital_efficiency_frontier.csv`: subset of summary columns (`mean_pnl`,
  `cvar95`, `ER`, `TR`) used by Track 5 to trace capital-efficiency frontiers.

All tables are sorted lexicographically by `(seed, regime_name, split)` to ensure
stable downstream consumption.
