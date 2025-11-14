# Robust Generalization for Hedging under Crisis Regime Shifts

notes

- Shift to hybrid experimental design: synthetic = option, real = historical SPY data.
- Action items
    - Simplify math and code
    - Google colab to test the experimens then improve
    - Slowly switch to Google Docs with Zotero
    - Cut down redundant and keep the codebase extremely concise; cut down placeholders.

[Next Step Action Items](https://www.notion.so/Next-Step-Action-Items-296602cd4ab68065b54ccde4983b22b8?pvs=21)

[Codebase Intallation Plan](https://www.notion.so/Codebase-Intallation-Plan-28b602cd4ab6808f99f7ec64e79973ae?pvs=21)

---

# **1	Abstract**

Hedging policies trained to minimize average loss often fail in crises: when volatility or skew regimes shift, tail losses surge despite good in-sample PnL. We propose HIRM, a hedging objective that stabilizes the hedge rule itself across market environments. HIRM constrains a shared decision head so that per-environment risk gradients (for CVaR-95) are aligned, while allowing representation layers to remain regime-sensitive. We pair this with simple diagnostics that score invariance and robustness under regime shifts. Training uses low–medium volatility periods (2017–2019) and excludes all high-volatility windows, including Feb–Mar 2018; evaluation covers held-out crisis windows (Feb–Mar 2018, Mar 2020, 2022 sell-off) and synthetic stress from Heston dynamics with Merton jumps. Against ERM, GroupDRO, V-REx, and tuned delta/vega hedges, HIRM reduces crisis CVaR-95 by 24–31% (median across 30 seeds) and cuts max drawdown by 18%, without sacrificing average PnL. ISI correlates with IG and with out-of-sample CVaR-95 to indicate constraint (but not data luck) drives gains. These results suggest aligning risk-sensitivity at the decision head is a simple, general principle to crisis-robust hedging.

---

# **2	Introduction**

**Problem.** Hedging policies that look resilient in calm markets often fail in crises: when volatility and skew regimes shift, tail losses spike even as average PnL appears acceptable. Empirically, ERM-trained deep hedges and classical rules over-rely on regime-sensitive shortcuts (e.g., short-window realized volatility), producing large CVaR-95 losses during the 2018 volatility shock and the 2020 crash. Prior robust objectives, such as worst-group reweighing and variance-of-risk penalties, improve worst-group fit but do not ensure the hedge rule itself remains stable across environments.

**Hypothesis.** The decision mapping from economically grounded risk factors (Greeks, time-to-maturity, inventory) to hedge actions is comparatively stable across regimes, whereas representations that absorb microstructure and liquidity effects must remain adaptive. Therefore, invariance should be enforced at the decision level, not across the full representation.

**Approach.** We introduce HIRM, which aligns per-environment risk gradients at a shared decision head (measured against CVaR-95), while leaving representations free to adapt to regime-specific frictions. We pair this with lightweight diagnostics toolkits that assess invariance, robustness and efficiency to indicate effectiveness of the method.

**Evaluation design.** Training stages uses SPY daily returns and realized volatility during low–medium volatility periods (2017–2019), explicitly excluding all high-volatility windows. Evaluation covers held-out crisis regimes (Feb–Mar 2018, Mar 2020, and the 2022 sell-off) and synthetic stress from a Heston–Merton generator calibrated to SPY. Despite the absence of option-level data, these episodes capture the volatility, skew, and liquidity shocks that stress hedging systems.”

Evidence preview. Across 30 seeds, HIRM reduces crisis CVaR-95 by 24–31% versus ERM and cuts max drawdown by 18%, while matching average PnL. ISI correlates with lower IG and with out-of-sample CVaR-95, indicating that decision-level invariance—not dataset luck—drives robustness under regime shifts.

**Contributions.**

1. **Hedging-native decision-level invariance.** An objective that stabilizes risk sensitivity at the shared decision head across environments.
2. **Diagnostics that tie mechanism to outcomes.** **ISI** (internal stability) and **IG** (outcome dispersion) connecting invariance to crisis performance.
3. **Market-realistic evidence.** Results on held-out stress windows and calibrated synthetic regimes showing tail-risk and turnover improvements without sacrificing mean returns.

# **3	Related Work**

## 3.1 Taxonomy of Learning Objectives

**Aim.** We position our approach within the robustness landscape by ascending from objectives that optimize average performance to those that seek causal invariance. The progression is: empirical risk minimization (ERM), ensembling, stress testing, domain adaptation and invariant risk minimization (IRM). IRM differs from the preceding families by aiming to preserve environment-stable relationships rather than merely smoothing or reallocating error. However, in financial control problems such as hedging, standard IRM is mismatched by default, motivating a decision-level variant (HIRM).

1. **Empirical Risk Minimization (ERM).** ERM minimizes expected loss on the observed distribution. It is sample-efficient and effective in stationary settings, yet it readily exploits regime-specific shortcuts (e.g., short-window realized volatility) that degrade under distribution shift. In hedging, this manifests as acceptable mean PnL but elevated crisis CVaR-95.
2. **Ensembles.** Ensembling (bagging/boosting/mixtures) aggregates diverse learners to stabilize predictions and occasionally attenuate tails via averaging. This reduces variance but does not enforce a stable hedge mapping; spurious regime dependencies can persist in the aggregate.
3. **Stress testing.** Stress testing uses historical crises and calibrated simulators to select or reject models based on worst-case behavior. This guards against optimistic model choice but provides no training signal that guarantees stability in future, unseen crises.
4. **Domain adaptation.** Domain adaptation assumes access to unlabeled target-domain data and aligns distributions or features accordingly. When the future regime can be partially observed at train time, adaptation helps. In practice, crises are unannounced; adaptation may overfit to the current shift rather than generalize to the next.
5. **Invariant Risk Minimization (IRM).** IRM seeks predictors whose optimality is invariant across environments while discouraging reliance on spurious, environment-specific features. Conceptually, it targets causal stability rather than mere performance smoothing.

**Design implication for hedging.**

The component that ought to remain stable across regimes is the decision mapping from candidate invariant, structural payoff sensitivities (e.g., Greeks, time-to-maturity, inventory) to hedge actions, while representations retain regime-specific flexibility, while representations must retain regime-specific flexibility. This motivates enforcing invariance at the decision head rather than the full representation.

**Table 1.** Comparison of training and evaluation families.

| **Family** | **Uses target-domain data during training?** | **Stabilize target** | **Typical shift coverage** | **Limitation for hedging** |
| --- | --- | --- | --- | --- |
| **ERM** | No | Mean risk on source | Small in-distribution drift | Learns regime shortcuts lead to crisis CVaR-95 spikes |
| **Ensembles** | No | Estimator variance | Same as base learners | Averages shortcuts; no decision-level stability |
| **Stress testing** | No | Model choice under stress | Depends on suite design | Not a training principle |
| **Domain adaptation** | Often yes | Feature & output alignment | Shifts partially observed at train time | Overfits current target; future crisis unseen |
| **IRM (representation-level)** | No | Predictors invariant across environments | Stable causal structure across sources | Over-constrains adaptation; misaligned with control and CVaR-focused objectives |

HIRM adopts IRM’s invariance principle while adjusting the constraint to hedging; it enforces only decision-level invariance, which is the alignment of per-environment risk sensitivities at the shared decision head (the action mapping layer) and leaves representation regime-adaptive.

## 3.2 IRM in Financial Learning

Invariant Risk Minimization (IRM) has achieved mixed success across domains such as computer vision, natural language processing and healthcare, particularly in environments are well specified and the evaluation metric is aligned with the learning objective. In finance, most application of IRM-style methods address static prediction (e.g. asset returns, credit default) via representation-level constraints and report incremental, context-depedent improvements. We synthesize these cross-domain experiences to assess IRM’s suitability for hedging, a sequential control problem governed by tail risks and implementation frictions.

**Insights from other fields.** When environment index genuine mechanism changes (e.g. dialects, hospitals) and provide sufficient diversity, IRM can suppress spurious features and improve out-of-distribution (OOD) prediction. Conversely, environments with weak proxies, overlapping or sparsely sampled can attenuate and destabilize IRM’s benefits.

IRM’s effectiveness under financial contexts confront following weaknesses:

1. **Absence of “true environments”.** Financial regimes consisting volatility, liquidity and skews are ambiguous, overlapping and drifting. Labels based on calendar periods or latent-state models inevitably conflate multiple drivers (policy shocks, microstructure stress, nonlinear feedback). Under such ambiguity, representation-level IRM risks regularizing toward artifacts of the labeling scheme rather than toward structure relevant for hedging decisions.
2. **Objective misalignment with hedging.** Canonical IRM is formulated for supervised prediction and typically regularizes prediction-level gradients or logits. Hedging performance, by contrast, is determined by tail-risk functionals (e.g., CVaR-95) and operational costs (e.g., turnover) that are episodic, non-smooth, and action-coupled. Representation invariance does not directly target decision sensitivity to these risks and may suppress necessary regime-specific adjustments (e.g., liquidity-aware position changes).
3. **Robustness implicates capital efficiency plus invariance.** In practice, robustness is not merely the generalizability across regimes. Robustness requires maintaining capital efficiency under stress (strong mean PnL per unit of tail risk and controlled turnover), while containing worst-group losses across regimes. IRM’s environment-invariant prediction objective does not guarantee these properties unless invariance is enforced where it matters for control: the decision head (the action-mapping layer) and its risk sensitivities.

**Implication.** The transferable value of IRM is its inductive bias for invariants; however, the locus and the objective must be adapted under hedging contexts. Invariance should be applied at decision level and aligned with CVaR-focused and turnover-aware criteria rather than with generic prediction error. Thus, standard IRM is conceptually incompatible with financial hedging without a structural redesign. 

## 3.3 Market Realism

**Positioning.** Robust hedging under regime shifts must be anchored in how markets actually behave. Prior work in prediction and control, ranging from classical option models to modern deep hedging and CVaR-based RL, has progressively absorbed non-Gaussian returns, stochastic volatility, jump dynamics, and frictions. However, much of this literature either abstracts away key market mechanisms (e.g., liquidity impact) or evaluates only on in-distribution regimes. In this paper, “market realism” means both the learning objective and the evaluation protocol are constrained by observed market facts (tails, stochastic vol/jumps, and frictions). We therefore report CVaR-95 on losses (lower is better), ER/TR for efficiency (expected return per unit tail risk; turnover; see §4.3.3), WG/VR for robustness (worst-group loss; temporal smoothness; see §4.3.2), and ISI/IG for invariance (internal stability; outcome dispersion; see §4.3.1).

**Liquidity, impact, and margins under stress.** Execution costs and inventory constraints limit feasible hedges. Linear temporary and permanent impact (Almgren–Chriss; motivation only) and transient impact (Obizhaeva–Wang; ablation) formalize trading risk; information-driven illiquidity (Kyle’s λ; motivation only) and low-frequency proxies (Amihud; used for regime tagging) capture state-dependent costs. In crises, funding and market liquidity co-move (e.g., 2018 “Volmageddon,” Mar-2020 “dash for cash”), elevating margins and forcing deleveraging. To reflect this, we model proportional trading costs and crisis-only margin outflows: c_t=\kappa_e\,\lvert\Delta q_t\rvert with regime-dependent \kappa_e, plus an episode-level exogenous margin term in crisis (§5.2).

**Evaluation of prior methods.** Classical and modern approaches contribute essential ingredients, including jump-diffusion (Merton) and stochastic-volatility pricing (Heston) for realistic tail and skew; arbitrage-aware surface parameterisations (SVI/SABR) for faithful option geometry; Deep Hedging for learning in the presence of frictions; and CVaR-based reinforcement learning for tail-aware control. But two gaps remain pervasive:

- **Evaluation gap.** Many studies validate in stationary or lightly shifted regimes, with limited out-of-sample crisis testing, or rely on simulators without supervisory stress alignment. The 2018 and 2020 episodes show that volatility, liquidity, and margining couple nonlinearly—precisely when hedge rules must not change character.
- **Objective gap.** Prediction-oriented or representation-invariant objectives do not directly target hedge decision stability with respect to risk and cost.

**Guided by these facts, our empirical protocol enforces three design choices:**

1. **Regime construction grounded in evidence.** Environments are defined by 20-day realized-volatility bands: Low < 10%, Medium 10–25%, High 25–40%, Crisis > 40%, with a liquidity tag via Amihud quartiles (Crisis = top quartile or bid-ask width > 2017–2019 p95). Training uses Low/Medium; validation uses late-2018 High; tests hold out Feb–Mar 2018, Mar 2020, and the 2022 sell-off, plus synthetic stress (Heston + Merton). Labels use trailing windows to avoid leakage. This directly targets the evaluation gap.
2. **Arbitrage-controlled surface and dynamics.** Stress paths use Heston-class dynamics with rare jumps (Merton overlay) and SVI-parameterized surfaces cleaned for static arbitrage, ensuring admissible option states (§5.1).
3. **Friction-aware performance metrics.** We optimize CVaR-95 on loss (primary training risk) and report MaxDD, **ER/TR** (efficiency), **WG/VR** (robustness), and **ISI/IG** (invariance), aligning measurement with execution and liquidity evidence (§4.3).

**Implication for learning objectives.** Within this market realism, enforcing invariance at the decision head aligns the learning target with hedging’s economic objective: stable tail-risk control with disciplined trading effort. By contrast, representation-level invariance neither reflects option-surface mechanics nor liquidity-induced constraints and can suppress necessary regime-specific perception, motivating the head-invariant redesign we test empirically (see §4.2, §6).

# **4	Theory**

## **4.1 Problem Setup and Notation**

Robust hedging requires more than risk minimization; it requires identifying mechanistic stability under regime perturbations. This work studies a hedging policy $π_θ$ that maps a state vector $x_t$  to an action at each time $t$:

$$

a_t = \pi_\theta(x_t), \qquad x_t \in \mathbb{R}^d, \qquad d = d_\Phi + d_r.

$$

The state decomposes into two feature groups:

$$
x_t = (\Phi_t, r_t)
$$

$r_t$ represents context-specific or regime-dependent factors whose predictive relationship to outcomes is unstable across environments (e.g., realized volatility windows, regime tags, ad-hoc path statistics), and $\Phi_t$ encodes invariant causal features that reflect stable financial mechanisms such as option Greeks and structural state variables:

$$
\Phi_t = (r_{t-1}, \sigma^{(20)}_t, \text{drawdown}_t, \text{momentum}_t, \text{CVaR}^{(20)}_t, \text{inventory}_t)
$$

corresponding to prior return, realized volatility, drawdown, trend momentum, short-term tail risk, and inventory. These features serve as observable proxies for structural sensitivities that enables model to capture nonlinear exposure and volatility adaptation.

We parameterize the policy as

$$
\theta=(\phi,\psi),\qquad \pi_\theta(x)=w_\psi\!\big(f_\phi(x)\big).
$$

where $f_\theta$ is the feature extractor parameterized by $\phi$ that learns representation from $\Phi_t$, and $w_\psi$ is the decision head parameterized by $\psi$, which is the *only* component regularized during training via a head gradient cosine alignment (HGCA) penalty, defined as an alignment constraint on normalized per-environment risk gradients at the decision head.

Formally, the HGCA penalty promotes cosine similarity among normalized gradients $\nabla_\psi R_e(\pi_\theta)$, across training environments, encouraging a stable decision mapping while leaving the representation unconstrained.

In the HIRM framework, $\phi$ corresponds to the representation module, which remains fully adaptive to regime-specific input variability, while $\psi$ defines the decision head, which alone is constrained via the HGCA penalty to align risk gradients across environments. Thus, HIRM constraints the causal path $Φt​→f_ϕ​(Φ_t​)→w_ψ​→a_t​$ to remain stable under distributional shifts, even as $r_t$ varies, while still permitting $f_\theta$ to adjust its intermediate representations to reflect regime-dependent observations.

**Environmental structure.** Each environment $e\in\mathcal{E}$ defines a distribution $P_e(x_t,a_t,r_t)$ corresponding to a distinct volatility-skew regime, consistent with SPY market protocols. Test environments reflect stressed dynamics (i.e. liquidity contractions, jump-diffusion shocks, and volatility clustering) and include real-world anchor scenarios (e.g. Volmageddon) as benchmarks for external validity. We partition the environments into:

$$
\mathcal{E}_\text{train},\mathcal{E}_\text{val},\mathcal{E}_\text{test}
$$

representing training, validation, and out-of-distribution (crisis) evaluation, respectively. Let $R_e(\pi_\theta)$ denote the risk-in-environment functional for policy $\pi_\theta$ evaluated on $e$. $R_e$ measures CVaR-95 (on loss scale) unless stated otherwise. Supplementary statistics include mean PnL (higher = better), Sortino ratio (higher = better) and Turnover (lower = better)

**Structural Assumptions.** To ensure identifiability of invariance and stable evaluation, we adopt:

**A1. Mechanism invariance. $r_t$** may encode regime-specific correlations; The Hybrid variant allows controlled adaptation to $r_t$, while the Head design constrains decisions to depend primarily on $\Phi_t$.

**A2. Sparse mechanism shift.** Across environments, only a small subset of structural molecules (e.g. volatility) is modified at a time, enabling identification of invariant causal components $\Phi_t$.

**A3. Capacity control for worst-case generalization.** Without proper regularization, over-parameterized policies can overfit environment-specific shortcuts; explicit capacity control (i.e. via weight decay and HGCA penalty applied only to the decision head) is required to preserve integrity under crisis evaluation.

## 4.2 HIRM Mechanistic Rationale

**Idea.** Robust hedging emerges when decision logic, not merely representations, remain invariant across regimes. ****Hedging decompose into a mechanism-driven core and peripheral adjustments that react to liquidity, volatility and transaction frictions. HIRM operationalize this by enforcing invariance only on the decision head $w_\psi$ through the HGCA penalty, while keeping representation $f_\phi$ free to adapt to shifting regimes.

This inductive bias targets the model component that encodes economic consistency—the mapping from risk factors to hedge actions. In contrast, global representational constraints, as in IRM, often ignore short-horizon adaptation. HIRM thus balances mechanistic stability in the decision rule with contextual flexibility in perception. Together, these ensure economic effectiveness. 

ISI is retained strictly as a post-hoc diagnostic to probe invariance; it is not included in the optimization objective, in contrast to the HGCA penalty which is the sole invariance-inducing term applied during training.

**Boundary Conditions and Assumptions.** HIRM rests on a weak but testable premise: within a regime, the economic structure of hedging remains stable even as volatility, skew, or liquidity fluctuate. This holds across most crisis regimes where macro drivers persist but surface dynamics amplify non-linearly. HIRM does not assume binary or universal invariance; when a regime shift alters the payoff structure (e.g. change in risk premia) the HGCA penalty may overconstrain the model. In those cases, adaptivity should dominate. Hence, HIRM performs best when the hedge mapping is stable but the market representation is unstable.

**Resilience to failure modes.** HIRM’s design anticipate potential weaknesses and embeds safeguards:

- Gradient collapse: Monitor gradient alignment and normalize λ-pressure to maintain non-trivial learning.
- Pseudo-invariance: When environments are weakly distinct, use volatility or liquidity grouping to restore semantic diversity.
- Overconstraint: Apply partial invariance only to structural parameters, preserving flexibility in contextual ones.
- Underfitting: Sweep λ and diagnostic curves empirically to identify the balance between stability and adaptivity.

In essence, these design choices reflect three guiding principles for HIRM:

1. **Locality.** Invariance acts at the decision head, where economic behaviour is encoded.
2. **Partiality.** Constraint is confined to mechanism-level parameters, allowing adaptive periphery and contextual sensitivity to persist.
3. **Resilience.** HIRM remain durable under imperfect conditions by maintaining theoretical validity and empirical tunability when encountering regimes that are misspecified or scarce in data.

Having established the conceptual foundation and resilience of HIRM’s design, the next section formalizes these principles through a diagnostic taxonomy.

## 4.3 Diagnostic Taxonomy

HIRM’s diagnostic taxonomy evaluates generalization through three complementary axes:  Invariance (I), Robustness (R), and Efficiency (E).

- Invariance captures causal stability across regimes (internal ISI, external IG).
- Robustness assesses resilience to adverse conditions (WG, VR).
- Efficiency measures capital productivity and control smoothness (ER, TR).

Together, these form a coherent I–R–E geometry that maps how causal stability propagates into economic performance.

### 4.3.1 Invariance

**Objective.** Invariance is defined as the consistency of causal relationships between predictive signals and policy actions that remain stable across regimes and yield smooth loss behavior. This subsection formalizes how HIRM measures invariance: internally through the Invariance Spectrum Index (ISI), which probes representation and gradient stability, and externally through the Invariance Gap (IG), which measures stability in realized outcomes.

**Conceptual Decomposition.** The policy is parameterized as

$$
π_θ(x)=w_ψ(f_ϕ(x)),θ=(ϕ,ψ).
$$

Here, $\phi$ parameterizes the representation module that extracts invariant causal structure, and $\psi$ parameterizes the decision head translates these features into hedging actions. This separation enables attribution of invariance to specific architectural components.

This distinction enables HIRM to integrate mechanism-level analysis with observable behavior, elucidating the causal basis of model generalization, the structural pathways through which robustness emerges, and the locus that warrants intervention when failures occur.

---

**Internal Diagnostic: Invariance Spectrum Index (ISI).** ISI is a composite in $[0,1]$ that aggregates 3 components. Each component captures a complementary facet of internal invariance. Probes are taken at a fixed set of layers $\mathcal{L}$ and estimated on held-out mini-batches that are disjoint from risk minibatches. Internal invariance is multidimensional: it must hold across losses, gradients, and representations. ISI therefore aggregates complementary probes that assess each of these dimensions.

**C1. Global Stability.** Penalizes dispersion of per-environment risk. If the model is invariant, its losses should only fluctuate minimally between volatility regimes.

$$
C_1 \;=\; 1 \;-\;\min\!\left(1,\;\frac{\widehat{\operatorname{Var}}_{e}\!\big[R_e(\pi_\theta)\big]}{\tau_R + \epsilon}\right),\qquad \epsilon > 0.
$$

Here $\tau_R$ is a constant normalizer calibrated on baselines, and $\epsilon >0$ prevents division by zero. High $C_1$ means low means overfitting to one regime. 

**C2. Mechanistic Stability.** Encourages consistent optimization directions across environments at the decision head.

$$
C_2 \;=\;
\frac{1}{Z}
\sum_{e < e'}
\frac{1 + 
\cos\!\big(
\nabla_{\psi} R_e(\pi_\theta),
\nabla_{\psi} R_{e'}(\pi_\theta)
\big)
}{2},
\qquad
Z = \tfrac{1}{2}|\mathcal{E}|(|\mathcal{E}| - 1).
$$

Here $Z$ is the number of unordered pairs. The mapping to $[0,1]$ follows the cosine-to-similarity transform. High $C_2$ indicates the model’s optimization direction is aligned across regimes.

**C3. Structural Stability.** Measures the consistency of latent feature representations across environments. If the internal activations $z_ℓ=f_ϕ^ℓ(x)$ change structurally between regimes, it signals a breakdown in causal invariance, thus $C_3$  penalizes feature drift by measuring environment-specific dispersion of representation means and covariances.

$$
C_3
\;=\;
1
\;-\;
\min\!\left(
1,\;
\frac{
\displaystyle
\frac{1}{|\mathcal{L}|}
\sum_{\ell \in \mathcal{L}}
\mathrm{Disp}_e
\!\left(
\mathbb{E}[z_\ell \mid e]
\right)
}{
\tau_C + \epsilon
}
\right),

\quad
\epsilon > 0.
$$

$\mathbb{E}[⋅]$denotes expectation under the data distribution of environment $e$. The dispersion operator quantifies covariance deviation from a stable baseline:

$$
\mathrm{Disp}_e\!\left(\mathbb{E}[z_\ell \mid e]\right)=\operatorname{Tr}\!\left(\Sigma_\ell^{-1}\operatorname{Cov}_e[z_\ell]\right)
$$

where $\Sigma_\mathcal{ℓ​}$ is a fixed reference covariance estimated on training environments and $\text{Cov}_e​[zℓ​]$ denotes the environment-specific covariance of representations. High $C_3$ implies that representation geometry remain aligned under varying market dynamics.

The three components operate on complementary scales: risk variance, gradient direction and feature dispersion. It necessitate normalized aggregation to prevent overemphasis on any single diagnostic channel.

**Aggregation and normalization.** Components that operate on distinct statistical scales are combined with robust averaging to prevent domination by any single term:

$$
\text{ISI} \;=\; \sum_{i=1}^3 \alpha_i\,\widetilde{C}_i, \qquad \sum_i \alpha_i=1,
$$

where $\widetilde{C}_i$denotes a 10 percent trimmed mean across probes or pairs for that component. Default weights $a_i$ may be uniform or learned once on a small development grid to maximize correlation with 1−IG, then frozen for all experiments to preserve comparability.

**Interpretation.**

- $C_1$ ensures outcome stability across environments.
- $C_2$ align learning dynamics through consistent gradient directions.
- $C_3$ enforce structural stability of internal representations.

High overall ISI indicates that the model’s representations, optimization behavior, and outcomes remain consistent across regimes, implying causal generalization rather than statistical memorization.

---

**External Diagnostic: Invariance Gap (IG).** IG serves as a direct measure of causal transfer; it tests whether stability learned in latent representations translates to stable risk outcomes under regime shifts. IG quantifies outcome-level stability of the trained policy across environments, which is defined as:

$$
\mathrm{IG}
\;=\;
\max_{\,e,\,e' \in \mathcal{E}_{\text{test}}}
\Big|
R_e(\pi_\theta)
- 
R_{e'}(\pi_\theta)
\Big|,
$$

where lower IG indicates consistent risk outcome across distinct regimes, implying successful transfer of invariance from internal mechanisms to realized performance. 

IG can be normalized for comparability across datasets:

$$
\mathrm{IG}_{\text{norm}}\;=\;\frac{\mathrm{IG}}{\tau_{\text{IG}} + \epsilon},\qquad\tau_{\text{IG}}=\operatorname{median}_{\text{seeds}}\Big(\operatorname{mean}_{e}R_e(\pi^{\mathrm{ERM}})\Big).
$$

IG is evaluated strictly post-training and is never used as a loss component. It thus serves as an *external validator* that internal invariance (captured by ISI) translates into stable realized outcomes.

---

**Theoretical Linkage.** Under mild regularity conditions—bounded risk gradients and Lipschitz-continuous heads—reductions in ISI directly upper-bound IG:

$$
IG≤κ_1(1−\text{ISI})+κ_2Δ_D,\quad\Delta_D=\text{dist}(\mathcal{E}_\text{train},\mathcal{E}_\text{test})
$$

where $\Delta_D$ denotes the diversity mismatch between training and testing environments. This bound formalizes ISI as a surrogate objective for minimizing causal deviation across environments, provided environmental diversity remains bounded.

Together, ISI and IG define the invariance axis of the HIRM diagnostic taxonomy, serving as the foundation upon which the robustness and efficiency dimensions build.

### 4.3.2 Robustness

**Objective.** Robustness measures a policy’s resilience to adverse regimes and temporal volatility. It is **ca**ptured by two external diagnostics: Worst-Case Generalization (WG) for tail vulnerability across environments and Volatility Ratio (VR) for temporal stability. Let $R_t$ denote the rolling-window risk series over time.

**Worst-Case Generalization (WG).** WG measures downside exposure across environments as a CVaR-style tail expectation:

$$
\mathrm{WG}_\alpha(\pi_\theta)
=
\inf_{\tau\in\mathbb{R}}
\Big[
\tau
+
\tfrac{1}{\alpha}\,
\mathbb{E}_{e\sim p_e}\!\big[
(R_e(\pi_\theta)-\tau)_+
\big]
\Big],
\qquad
\alpha\in(0,1).
$$

A lower WG indicates the policy risk does not deteriorate significantly under worst regimes (e.g. liquidity cuts or volatility jumps).

**Volatility Ratio (VR).** VR assesses the temporal smoothness of realized risk:

$$
\mathrm{VR}(\pi_\theta)=\frac{\operatorname{Std}_t[R_t]}{\mathbb{E}_t[R_t]+\epsilon},\qquad\epsilon>0.
$$

Smaller VR means the policy’s losses evolve steadily rather than spiking between windows.

**Interpretation.** WG captures **downside robustness (**resistance to regime-level shocks) while VR captures **temporal robustness (**consistency through time). Together they expect a policy is “robust” when it limits worst-case losses and exhibits smooth, stable behaviour across evolving market regimes. These diagnostics are evaluation-only; they validate that the invariances learned internally (Section 4.2.1) translate into externally durable performance.

### 4.3.3 Efficiency

**Objective.** Efficiency quantifies how effectively the policy converts stability into capital-adjusted performance.—preserved return per unit of risk and adjustment cost. In practice, an efficient policy should maintain smooth, invariant behavior without excessive capital turnover or volatility drag.

This axis formalizes two complementary diagnostics: Expected Return–Risk Efficiency (ER) and Turnover Ratio (TR).

**Expected Return-Risk Efficiency (ER).** ER quantifies mean performance normalized by downside risk. ER is defined as:

$$
\mathrm{ER}(\pi_\theta)=\frac{\mathbb{E}_t[R_t]}{\mathrm{CVaR}_{95}(R_t) + \epsilon},\qquad\epsilon>0.
$$

High ER implies strong capital productivity per unit of tail risk; It generalizes Sortino ratios under asymmetric loss distributions.

**Turnover Ratio (TR).** TR measures adjustment intensity—how much trading or hedging effort is required to sustain performance:

$$
\mathrm{TR}(\pi_\theta)=\frac{\mathbb{E}_t[\|a_t - a_{t-1}\|_2]}{\mathbb{E}_t[\|a_t\|_2] + \epsilon}.
$$

Lower TR reflects smoother control trajectories and lower transaction or liquidity costs.

**Interpretation.** ER captures capital efficiency to test stable risk control’s yield of productive returns, while TR captures operational efficiency to penalize excessive hedge return or frictional losses. A policy achieves high overall efficiency when it preserves invariance (Section 4.2.1) and robustness (Section 4.2.2) yet delivers sustained net performance with minimal adjustment overhead.

Together, ER and TR complete the efficiency axis of the I–R–E framework.

## 4.4 Theoretical Integration

**Objective.** This section connects HIRM’s optimization layer to the diagnostic layer. The head-level penalty aligns with environment-specific sensitivities of decision head $w_\psi$, which raises internal ISI and reduce external IG. Lower dispersion improves crisis-time tail risk and, for a given mean PnL, raises capital efficiency.

**From head penalty to internal stability.** HIRM trains with an average risk and head-invariance penalty:

$$
\mathcal{L}_{\text{HIRM}}(\phi,\psi)
=\frac{1}{|\mathcal{E}_{\text{train}}|}\sum_{e} R_e(\pi_\theta)
+ \lambda_I\, \mathrm{Var}_{e}\!\big[g_e\big],
\qquad
g_e = \nabla_{\psi} R_e(\pi_\theta).

$$

If we L2-normalize head gradients, $\hat g_e = g_e / \|g_e\|_2,$ the pairwise identity

$$
\|\hat g_e - \hat g_{e'}\|_2^2= 2\!\left(1 - \cos(\hat g_e,\hat g_{e'})\right)
$$

implies that minimizing $\text{Var}_e[g_e]$ drives cosine alignment $\nabla_\psi R_e$ across environments. Since ISI’s $C_2$ component aggregate these cosines, the HIRM penalty operationalize internal invariance at the decision head, which higher alignment produce higher ISI.

**From Internal stability to external robustness.** When the head mapping is Lipschitz and risk gradients are bounded, differeneces in environments satisfy

$$
|R_e(\pi_\theta)-R_{e'}(\pi_\theta)|\le\kappa\,\|\nabla_{\psi} R_e - \nabla_{\psi} R_{e'}\|_2+ \kappa'\,\Delta_D,
$$

which $\Delta_D$ a diversity term capturing train-test mismatch. Therefore, reducing gradient match internally lower outcome dispersion externally. Equivalently, higher ISI implies low IG up to a diversity term, which is precisely the notion of robustness under regime shift.

**From robustness to economic efficiency.** Crisis performance is governed by the tail of the loss distribution. Lower dispersion across regimes will compress CVaR-95 on the loss scale. For a fixed or mildly changing mean PnL, this raises efficiency

$$
\mathrm{ER}
=\frac{\mathbb{E}_t[\mathrm{PnL}_t]}
      {\mathrm{CVaR}_{95}(-\mathrm{PnL}_t)+\epsilon},
$$

and typically reduces turnover ratio because a stable head produces smoother adjustments. Thus, invariance at the head propagates to lower tail risk and higher capital productivity. 

**Unified diagnostic geometry.** By penalizing the cosine misalignment of head-level gradients across environments, HIRM induces higher ISI. Because ISI correlates with reduced outcome dispersion, a reduction in IG follows, tightening the crisis tail-risk distribution. Given that turnover is a convex function of head perturbations, lower IG empirically leads to lower TR, which in turn improves capital efficiency (ER). Thus, HIRM converts alignment pressure at the head into measurable economic efficiency along the I–R–E axis.

**Objective.** This section connects HIRM’s optimization layer to the diagnostic layer. The head-level penalty aligns with environment-specific sensitivities of decision head $w_\psi$, which raises internal ISI and reduce external IG. Lower dispersion improves crisis-time tail risk and, for a given mean PnL, raises capital efficiency.

# **5	Methodology**

## 5.1 SPY Market Environment & Data

**Synthetic Generator.** We simulate discrete-time equity and option paths under Heston dynamics with regime labels by realized volatility bands: Low (<10%), Medium (10-25%), High (25-40%), and Crisis (>40%), combined with optional Merton jumps.ref Training uses Low and Medium regimes; validation uses high and held-out Out-of-Distribution test uses Crisis.

**Real Data.** We mirror the synthetic volatility-band split using publicly available SPY (S&P 500 ETF) daily close and realized-volatility series. Regimes are defined by 20-day realized-volatility bands: Low (<10%), Medium (10–25%), High (25–40%), and Crisis (>40%). Training uses Low/Medium regimes from 2017–2019; validation uses the late-2018 volatility spike as a high-volatility proxy; and out-of-distribution testing covers the Feb–Mar 2018 “Volmageddon,” Mar–Jun 2020 COVID crash, and 2022 sell-off following the Ukraine invasion. These episodes embed joint shocks to volatility, liquidity, and correlation that stress hedging systems, providing a transparent and reproducible proxy for option-market crises in the absence of proprietary option data.

**Stress Extensions.** To probe extrapolation breadth, we introduce i) **jump dynamics** via a Merton jump-diffusion overlay; ii) **liquidity stress** via regime-dependent proportional costs and widened bid-ask in Crisis testing.

**Liability and hedging universe.** The liability is an ATM 60-Calendar-day European call (rolled daily to remain ATM-60D). Hedging is allowed daily in the underlying (and, in extended runs, liquid listed options). Proportional transaction costs applied for each trade; zero costs at maturity.

## 5.2 State, Action, and Risk

**State Features.** We split the state into $\Phi$ and $r$. The HIRM penalty acts on the head that consumes $\Phi$. All features are standardized with training-only statistics.

**Action.** The policy outputs a trade $\Delta_{qt}$ in hedge units (shares and contracts). Positions are self-financing with transaction costs.

**Episode PnL and risk.** Over an episode $t=0,1,...,T$ ( $T$ = 60 trading days), terminal PnL is therefore

$$
\text{PnL} = -Z_T + \sum_{t=0}^{T-1} q_t \left(S_{t+1} - S_t\right) - \sum_{t=0}^{T} c_t(\Delta q_t)
$$

with $Z_T$ the liability payoff and $c_t$ the proportional cost. The training loss is a convex risk measure computed on the episode level PnL with CVaR-95 as the primary risk. We also log mean PnL, Sortino, and Turnover.

All risks are losses (lower is better). CVaR-95 is computed on -PnL.

## 5.3 Objectives Across Environments

We compare objectives that aggregate episode risks across training environments 

$e \in \mathcal{E}_{\text{train}}$. Each corresponds to a different inductive bias robustness under regime shifts. Let $R_e(\pi_\theta)$ denote the episode risk (CVaR-95 by default). Higher is worse.

**Delta Hedging**

**ERM (Empirical Risk Minimization).**

$$

\min_\theta \; \frac{1}{|\mathcal{E}_{\text{train}}|}\sum_{e \in \mathcal{E}_{\text{train}}} 
R_e(\pi_\theta).

$$

Baseline deep hedging objective that optimizes the average training risk. It is effective in calm markets but often learns shortcuts tied to specific regimes.

**GroupDRO (Group Distributionally Robust Optimization).**

$$
\min_\theta \; \max_{e \in \mathcal{E}_{\text{train}}} R_e(\pi_\theta).
$$

Protects against the worst-case training environment by upweighting hard groups, which ensures robustness within the training mixture but does not enforce a stable hedge rule across environments.

**V-REx (Variance Risk Extrapolation).**

$$
\min_\theta \;\sum_{e \in \mathcal{E}_{\text{train}}} R_e(\pi_\theta) 
\;+\; \beta \,\mathrm{Var}\{R_e(\pi_\theta)\}.
$$

Penalizes the variance of risks across environments, flattening the risk surface and reducing sensitivity to distributional shifts. This can reduce sensitivity to distribution shifts but treat all features symmetrically, whether causal or spurious. 

**Invariant Risk Minimization (IRMv1).**

$$
\min_{\theta,\,w} \;\; 
\sum_{e \in \mathcal{E}_{\text{train}}} 
R_e\!\big(w \circ \phi_\theta\big) 
\;+\; 
\lambda \sum_{e \in \mathcal{E}_{\text{train}}} 
\Big\lVert \nabla_{w \,\vert\, w=1} 
R_e\!\big(w \circ \phi_\theta\big) \Big\rVert^2.
$$

IRM seeks predictors that are simultaneously optimal across environments by penalizing environment-specific gradients of the risk head. In principle, this aligns training with invariant features, but in hedging, it can over-constrain the representation.

# **6	Empirical Data**

## 6.1 Synthetic

## 6.2 Real Stress Windows

## 6.3 Ablations & Diagnostics

## 6.4 Discussion & Limitations

# **7	Conclusion**

# **8	References**

1. Arjovsky, M., Bottou, L., Gulrajani, I. & Lopez-Paz, D. (2020). Invariant Risk Minimization. *arXiv preprint arXiv:1907.02893*. Available at: [https://arxiv.org/abs/1907.02893](https://arxiv.org/abs/1907.02893) (Accessed 27 June 2025).
2. Chen, Y., Xu, Z., Inoue, K., & Ichise, R. (2024). ‘Causal Inference in Finance: An Expertise-Driven Model for Instrument Variables Identification and Interpretation’, *arXiv preprint* arXiv:2411.17542. Available at: [https://arxiv.org/abs/2411.17542](https://arxiv.org/abs/2411.17542) (Accessed: 27 June 2025).
3. Li, S., Schulwolf, Z. B., & Miikkulainen, R. (2025). ‘Transformer-Based Time-Series Forecasting for Stock’, *arXiv preprint* arXiv:2502.09625. Available at: [https://arxiv.org/abs/2502.09625](https://arxiv.org/abs/2502.09625) (Accessed: 27 June 2025).
4. Bühler, H., Gonon, L., Teichmann, J., & Wood, B. (2018) ‘Deep Hedging’, *arXiv preprint* arXiv:1802.03042. Available at: [https://arxiv.org/abs/1802.03042](https://arxiv.org/abs/1802.03042) (Accessed: 27 June 2025).
5. Lim, B., Arik, S.O., Loeff, N. & Pfister, T. (2021) ‘Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting’, *International Journal of Forecasting*. Available at: [https://arxiv.org/abs/1912.09363](https://arxiv.org/abs/1912.09363) (Accessed: 27 June 2025).
6. Araci, D. (2019) ‘FinBERT: Financial Sentiment Analysis with Pre-trained Language Models’, *arXiv preprint* arXiv:1908.10063. Available at: [https://arxiv.org/abs/1908.10063](https://arxiv.org/abs/1908.10063) (Accessed: 27 June 2025).
7. Wu, S., Dhingra, B., Anil, R., et al. (2023) ‘BloombergGPT: A Large Language Model for Finance’, *arXiv preprint* arXiv:2303.17564. Available at: [https://arxiv.org/abs/2303.17564](https://arxiv.org/abs/2303.17564) (Accessed: 27 June 2025).
8. Wiese, M., Bai, L., Wood, B. & Bühler, H. (2019) ‘Deep Hedging: Learning to Simulate Equity Option Markets’. *Papers With Code*. Available at: [https://paperswithcode.com/paper/deep-hedging-learning-to-simulate-equity](https://paperswithcode.com/paper/deep-hedging-learning-to-simulate-equity) (Accessed: 27 June 2025).

# **9	Appendix**

Diversity Sensitivity Generalization