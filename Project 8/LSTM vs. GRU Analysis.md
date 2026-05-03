# LSTM vs. GRU Analysis & Architecture Exploration

We will vary:
- **Input window length $S$** (e.g., $7 \rightarrow 14 \rightarrow 30$): how many past days we feed in.
- **Depth** (1 vs 2 recurrent layers) and **dropout** (e.g., $0.0$ vs $0.2$).
- **Cell type:** **LSTM** vs **GRU** (both with gating, different parameterization).

Our question: **Which combination gives the best generalization under our constraints?**


**⚖️ What “better” means (our success criteria):**
We’ll call a configuration “better” if it achieves, on **validation** and then **test**:

- **Lower error in °C**:
  $$
  \text{MAE} = \frac{1}{N}\sum_{i=1}^{N}\lvert y_i - \hat y_i\rvert,
  \qquad
  \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat y_i)^2}.
  $$
  (We also look at per-horizon metrics $\text{MAE}@k$, $\text{RMSE}@k$ when we roll out.)
- **Flatter error vs horizon** for short multi-step rollouts (less **compounding** as $k$ grows).
- **Healthier training curves**: train and val loss both go down without widening gaps (less overfitting).
- **Reasonable runtime** on CPU (we keep epochs modest and models compact).

When two configs are statistically similar, we break ties by **stability** (flatter MAE@k) and **simplicity** (fewer params).

**⏱️ Constraints (so we don’t melt CPUs)**
- **CPU-only** execution.
- **Short runs** (about 15–20 epochs per config).
- **Modest hidden sizes** (e.g., $64$) and batch-first tensors to match our previous notebooks.
- **Small, focused grids** (we avoid combinatorial explosions).

### Experiment A — Sequence Length Sensitivity ($S\in\{7,14,30\}$)

* Hypothesis: larger $S$ may encode **seasonal lead-ups** and reduce **phase lag**, but can also **raise variance** and slow learning.
* Keep **one cell type fixed** (e.g., **GRU**, 1 layer, hidden=64), vary **only $S$**.

**Code:**

* Loop over $S\in\{7,14,30\}$: rebuild windows/loaders; train **short** (e.g., 15–20 epochs).
* Track **train/val MSE** in scaled space, then compute **val MAE/RMSE** in °C.

**Outputs:**

* Table: $S$ vs val/test MAE & RMSE in °C.
* Plot: **val-loss vs epoch** per $S$ (overlay).

**Reflection:** Does larger $S$ flatten error at $k=1$? Any sign of **overfitting** (train ↓ while val ↔/↑)?

> Validation loss vs epoch for different sequence lengths S
![Validation loss vs epoch for different sequence lengths S](<Validation loass vs epoch for different sequence lengths S.png>)

- **Curves (val MSE):**  
  - If $S=30$ descends **slower** or **plateaus** higher, that suggests the model struggles to exploit the longer context under our compute budget.  
  - If $S=14$ or $S=30$ sits **below** $S=7$ consistently, we’ve gained from extra context.

**💬 Reflection — Does larger $S$ flatten error at $k=1$?**
- **Yes:** likely the model is using **seasonal lead-ups**, reducing **phase lag** and improving turning points.  
- **No / Mixed:** with small data and short training, larger $S$ can **raise variance** and be **harder to optimize**; try a **few more epochs**, a **slightly larger hidden size**, or keep $S$ modest and add **seasonal features** (e.g., day-of-year $\sin/\cos$) in NB05.

> If we see **train MSE** falling while **val MSE** stays flat or rises, that’s a sign of **overfitting** (or insufficient regularization). In Experiment B we’ll test **depth** and **dropout** to address this.

### Experiment B: Depth & Dropout (LSTM vs GRU × 1 vs 2 layers × dropout)

**🧭 Goal**
Now that we’ve chosen a good sequence length $S$ (from Experiment A), we compare **architectures** under a small, CPU-friendly grid:

- **Cell:** GRU vs LSTM  
- **Depth:** $1$ vs $2$ recurrent layers  
- **Dropout:** $0.0$ vs $0.2$ (applies only when layers $>1$ in PyTorch)  
- **Fixed:** hidden size $=64$, batch-first $(B,S,1)$, single-step target, brief training (≈15–20 epochs)

**Why stack?**

Deeper encoders may capture **hierarchical temporal structure**: a first layer learns short jitter; a second layer summarizes **slower drift**. But adding depth increases parameters and can **overfit** faster.

**💡 Why dropout?**
Dropout reduces **co-adaptation** between hidden units, often improving **generalization**. Too much dropout can **wash out** subtle temporal cues (especially with small data).

> Configuring the top 3 validation loss curves

![Top 3 configs - validation loss curves](<Top 3 configs - validation loss curves.png>)

**💬 Reflection — LSTM vs GRU after controlling depth/dropout**
- If **LSTM** wins here, we can argue its **cell state** helped carry information over the selected $S$.  
- If **GRU** wins (common on smaller datasets), we can argue it’s **lighter** (fewer parameters), easier to optimize, and sufficiently expressive for this task.  
- **Early overfit** pattern: train MSE ↓ while val MSE ↔/↑ (especially with 2 layers, no dropout). In that case, we prefer the **simpler** config or add a touch of **dropout**.

> Our selection criterion is **Val\_RMSE\_C** (tie-break by Val\_MAE\_C). We’ll carry the winner forward to the next section for a quick **generalization** check and a short **rollout** comparison.

---
### Summary & Key Takeaways

**Effect of longer sequences (S):**
- Increasing the window (e.g., 7 → 14 → 30 days) gives us more context and can reduce near-term error—especially around slow trends and seasonal phases.
- Diminishing returns kick in: longer S increases compute, risk of over-smoothing, and exposure to distribution shift. Choose S with **walk-forward validation** rather than intuition.

**Importance of reproducibility:**
- Fix seeds and deterministic ops so that our comparisons are fair across runs.
- Log core config (S, H, hidden sizes, learning rate, epochs, scaler fit range, split dates, seed) with each result.

**Controlled variable testing (one change at a time):**
- When we compare models or settings, keep everything else fixed (same split, scaler, optimizer, epochs, scheduler).
- Recommended ablations:  
  1) **Sequence length S** (e.g., 7/14/30)  
  2) **Model family** (RNN vs GRU vs LSTM) with the **same** hidden size  
  3) **Horizon H** (per-horizon MAE/RMSE curves)

**Use multiple evaluation metrics:**
- Report **MAE**/**RMSE** in °C (interpretability), plus a **percentage error** (e.g., **sMAPE**).  
- For multi-step forecasts, include **per-horizon** metrics (k = 1…H) and an overall average.  
- Include simple baselines (Persistence, **MA(7)**) so improvements are meaningful.

**Best practices for data splitting (no leakage):**
- Split by **time**, not randomly. Fit scalers on **train only**; impute **within each split**.  
- Prefer **walk-forward (rolling origin) validation** when we tune S/H or compare families.  
- Consider a **seasonal-naïve** baseline (e.g., \( \hat{y}_t = y_{t-365} \)) when annual effects are relevant.

**Bottom line:** We balance context length (S) with runtime and robustness, keep experiments reproducible and controlled, evaluate with multiple metrics and per-horizon views, and always guard against temporal leakage.
