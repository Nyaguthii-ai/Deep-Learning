## Framing Time Series as Supervised Learning
Instead of predicting a **class label**, we want to predict the **next numeric value** in the sequence.

- **Project 8:**  
  $X = [T_1, T_2, \dots, T_n] \;\;\longrightarrow\;\; y = T_{n+1}$  

**Key Idea**
- We convert a time series into a **supervised learning dataset** using **sliding windows**.  
- Each input $X$ is a fixed-length sequence of past observations.  
- The target $y$ is simply the **next observation**.  

👉 This is called **lag-based framing**.

**Examples**
- The first few rows show exactly how this framing works:
  - **Example 1:**  
    $X_0 = [20.7, 17.9, 18.8, 14.6, 15.8, 15.8, 15.8]$  
    $y_0 = 17.4$  
    → “If the last 7 days looked like this, predict 17.4 tomorrow.”

  - **Example 2:**  
    $X_1 = [17.9, 18.8, 14.6, 15.8, 15.8, 15.8, 17.4]$  
    $y_1 = 21.8$  
    → “If the last 7 days looked like this, predict 21.8 tomorrow.”

- Notice how the **window slides forward by 1 day** each time.  
- Because Long Short Term Memory networks(LSTMs) cannot “understand” raw time series directly. They need the series **segmented into input-output pairs** for supervised learning. 

📌 This supervised framing is the **bridge** that allows us to apply deep learning models to time series forecasting.

In Project 7 we sometimes used **random splits** because activities were i.i.d. at the sequence level.  
For **time series**, random splits cause **data leakage**.

**Preparing for Windows (no leakage across the boundary)**

There’s another subtle but critical point:  
When we create **sliding windows** (e.g., 7 past days → next day), we must build them **separately** for train and test.  

- If we created windows on the full dataset first, some windows would **cross the boundary**.  
- That would let information from test sneak into train.  
- Even though it seems small, this type of leakage can make results misleading.

👉 To avoid this, we first split the series into train/test and *then* apply our `create_sequences` function separately to each part.

### **5. Scaling & Normalization**

**Why Scaling Matters**

When we train a neural network, the raw values we feed into it have a direct effect on how well the model learns.  
If the inputs are very large or very small, the network’s **weights and gradients** can behave unpredictably.  
For example, if we trained on raw temperatures in degrees Celsius (values around 10–30) alongside another variable in the hundreds (say rainfall in millimeters), the larger numbers would dominate the learning process.  

This domination leads to two problems:
1. The model pays more attention to features with bigger numeric ranges, even if they are not actually more important.  
2. During training, the **gradients** that update the model’s weights can either shrink too quickly (vanishing gradients) or grow uncontrollably (exploding gradients). Both make learning unstable.

Scaling helps prevent these issues.  
By transforming all inputs into a **common range**, we give every feature an equal chance to influence the model, and we keep the gradients at healthy magnitudes.  
This leads to **faster and more stable training**.

**Avoiding Data Leakage**

It is extremely important that we **fit the scaler only on the training set**.  
If we let the scaler see the full dataset, it would learn the maximum and minimum temperatures of the **entire period**.  

After applying MinMax scaling, all of these values are compressed into the range $[0, 1]$.  

This doesn’t change the *pattern* of the data (the ups and downs, the seasonality), but it does change the *scale*.  
By doing so, the model sees inputs that are numerically well-behaved and easier to learn from.
