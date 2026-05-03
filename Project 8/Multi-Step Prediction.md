## Multi-Step Prediction 

We will compare two families:

- **Recursive (autoregressive)** — reuse our **NB02 single-step** model $f$ repeatedly  

  Pros: easy reuse. Cons: **exposure bias / compounding error**.

- **Direct (multi-output)** — train a model $g$ that outputs all $H$ steps at once  

  Pros: avoids step-by-step drift. Cons: harder target, bigger head.

**Summary:**

> **Recursive**</br>
Input $(B, S, 1)$ --$f$--> $\hat{y}_{t+1}$  
slide + append $\hat{y}_{t+1}$ → $(B, S, 1)$ --$f$--> $\hat{y}_{t+2}$ → ... → $\hat{y}_{t+H}$  
Stack $H$ scalars → $(B, H)$

>**Direct**</br>
Input $(B, S, 1)$ --$g$--> $[ \hat{y}_{t+1}, \hat{y}_{t+2}, \dots, \hat{y}_{t+H} ]$ → $(B, H)$

> In both cases, **inputs are batch-first** $(B,S,1)$. The **direct** model outputs $(B,H)$.  
> For **recursive**, we generate $H$ one-step predictions and then stack them to $(B,H)$.


**Builder we need next: make $(\text{window} \to \text{H-vector})$ pairs**

To train/evaluate either strategy fairly, we construct **supervised pairs** where each input window of length \(S\) is paired with the **next \(H\) ground-truth values**.

### Baselines for $H$-step Forecasting (yardsticks first)

**Why we start with baselines**

Before we compare **Recursive** and **Direct** models, we need **yardsticks** that tell us whether our fancy models actually add value.

We’ll use two simple, strong baselines (computed on the **test** windows):

1) **Persistence (last value)**  
   $$\hat T_{t+k} = T_t \quad \text{for } k=1,\dots,H.$$
   Intuition: tomorrow looks like **today**; the next week is today **copied forward**.

2) **7-day mean (last $S$ values)**  
   $$
   \hat T_{t+k} \;=\; \frac{1}{S}\sum_{j=0}^{S-1} T_{t-j} \quad \text{for } k=1,\dots,H,
   $$
   with our default $S=7$. Intuition: use the **recent weekly average** as a stable forecast.

> These are often **surprisingly competitive** for short horizons, especially on smooth series with strong autocorrelation. If our models can’t beat these, we should **rethink window size, features, or training**.

We’ll compute **MAE** and **RMSE** **by horizon** $k=1,\dots,H$ in **°C**, and plot **MAE vs horizon** as a quick visual.

#### Strategy A: <u>Recursive</u> Rollout (reuse NB02 single-step)

**Idea (one-step model → $H$-step forecast)**
We reuse our **NB02** single-step forecasters $f$ (RNN / LSTM / GRU).  
Starting from an input window $X_t=[T_{t-S+1},\dots,T_t]$, we:

1. Predict the **next day**: $\hat T_{t+1}=f(X_t)$.  
2. **Append** $\hat T_{t+1}$ to the window, **drop** the oldest value → new window.  
3. Predict $\hat T_{t+2}$, append, drop … **repeat** until $\hat T_{t+H}$.

This is called **recursive** (or **autoregressive**) rollout.

**Why in *scaled* space?**  
The model was trained on scaled inputs (MinMax to $[0,1]$). We therefore **roll forward in scaled space** and only **inverse-transform** to °C for reporting/plots. This keeps the numeric range stable.

$(B, S, 1) \xrightarrow{f} \hat{y}_{t+1}$  
slide & append $\hat{y}_{t+1} \Rightarrow (B, S, 1) \xrightarrow{f} \hat{y}_{t+2} \Rightarrow \dots \Rightarrow \hat{y}_{t+H}$  
stack $\{ \hat{y} \}$ → $(B, H)$

**⚠️ What to watch for (compounding + lag)**
Every time we feed our **own prediction** back in, any small miss can **propagate**.  
Two typical artifacts we’ll look for in plots:

- **Drift**: forecasts gradually shift away from the true series as $k$ grows.  
- **Phase lag**: peaks/troughs are predicted **late** relative to the true series.

We’ll compute **MAE/RMSE by horizon** and compare against our **baselines** (persistence, mean-7).

#### Strategy B: <u>Direct</u> Multi-Output Head (seq → $\mathbf{H}$)

**🧭 Idea**
So far we rolled a **one-step** model forward $H$ times (recursive). Now we predict the **entire horizon at once**:

$$
X_t = [T_{t-S+1}, \dots, T_t] \;\longrightarrow\;
\hat{\mathbf{y}}_t = 
\big[\hat T_{t+1},\, \hat T_{t+2},\, \dots,\, \hat T_{t+H}\big] \in \mathbb{R}^{H}.
$$

We reuse the same **encoder** (e.g., GRU). From the **last hidden state** \(h_{\text{last}}\) we send one **Linear** head to length $H$:

$$
h_{\text{last}} \in \mathbb{R}^{d}
\;\xrightarrow{\;\;W\in\mathbb{R}^{H\times d}\;\;}\;
\hat{\mathbf{y}} \in \mathbb{R}^{H}.
$$

**✅ Why try Direct?**
- **Pros:** avoids **compounding error** (we don’t feed predictions back), learns **horizon-specific** weights.  
- **Cons:** the target is **harder** (predict many steps jointly), can **smooth** peaks, and needs enough capacity/regularization.

We will:
1) Build a **GRU-Direct** with `batch_first=True` to accept $(B,S,1)$ and output $(B,H)$ in **scaled** space.  
2) Train briefly (CPU-friendly).  
3) Evaluate **MAE/RMSE by horizon** in °C and compare to baselines (and, if available, our recursive curves).

> Shapes we’ll keep in mind: **input** $(B,S,1)$ → **output** $(B,H)$.