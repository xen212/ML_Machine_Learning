# ML_Machine_Learning
## 🎯 Project Overview & Objective

A public agency aims to flag households with an annual income exceeding $50,000 from census survey data to target social outreach and benefit programs[cite: 8, 12].

**Central Question:** Does a neural network outperform a well-built classical model on this tabular dataset, and is the added complexity worthwhile when decisions affect real people[cite: 8, 12]?

---

## 📊 Dataset Description

* **Dataset:** Adult / Census Income dataset (OpenML, version 2)[cite: 8, 12].
* **Dimensions:** 48,842 observations across 14 tabular features[cite: 12].
  * **6 Numeric Features:** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`[cite: 12].
  * **8 Categorical Features:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`[cite: 12].
* **Target Variable:** Binary income bracket (`<=50K` $\rightarrow$ `0`, `>50K` $\rightarrow$ `1`)[cite: 8, 12].
* **Class Imbalance:** Only ~23.9% of samples belong to the positive higher-income class (`>50K`), while ~76.1% belong to the majority class (`<=50K`)[cite: 12].

---

## ⚙️ Preprocessing Pipeline

To eliminate data leakage, all transformations are managed through Scikit-Learn's `ColumnTransformer`[cite: 8, 12]:
1. **Train/Test Split:** Stratified 80/20 train-test split (`test_size=0.2, random_state=42, stratify=y`) yielding 39,073 training instances and 9,769 test instances[cite: 8, 12].
2. **Numeric Pipeline:**
   * Missing values imputed using the median (`SimpleImputer(strategy='median')`)[cite: 8, 12].
   * Centered and scaled to unit variance (`StandardScaler()`)[cite: 8, 12].
3. **Categorical Pipeline:**
   * Missing values imputed using the most frequent category (`SimpleImputer(strategy='most_frequent')`)[cite: 8, 12].
   * Converted to binary indicators using one-hot encoding (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`)[cite: 8, 12].
4. **Feature Output:** The preprocessed matrix expands to **105 dense numeric features**[cite: 12].

---

## 🧠 Model Architectures

### 1. Classical Baseline: Logistic Regression
* Unified into a single Scikit-Learn `Pipeline` with the `ColumnTransformer` preprocessor[cite: 8, 12].
* Settings: `max_iter=1000`, `random_state=42`[cite: 8, 12].
* Fits 106 trainable parameters (105 weights + 1 bias), providing transparent, auditable feature weights[cite: 8, 12].

### 2. Deep Learning Model: Dense Neural Network (Keras)
* **Input Layer:** 105 features[cite: 12].
* **Hidden Layer 1:** 32 units, ReLU activation, $L_2$ weight regularization ($10^{-4}$)[cite: 12].
* **Dropout Layer 1:** Rate of 0.4 to prevent co-adaptation[cite: 12].
* **Hidden Layer 2:** 16 units, ReLU activation, $L_2$ weight regularization ($10^{-4}$)[cite: 12].
* **Dropout Layer 2:** Rate of 0.4[cite: 12].
* **Output Layer:** 1 unit with a sigmoid activation producing class probabilities $P(y=1\vert{}\mathbf{x}) \in [0, 1]$[cite: 8, 12].
* **Training Settings:** Compiled with `adam` optimizer and `binary_crossentropy` loss[cite: 8, 12]. Trained with batch size 128, a 10% validation split, and `EarlyStopping` monitoring `val_loss` (patience = 3, `restore_best_weights=True`)[cite: 8, 12].

---

## 📈 Results & Evaluation

Due to the class imbalance, the **F1-score** on the positive minority class (`>50K`) serves as the primary metric, while **Accuracy** provides supporting context[cite: 8, 12].

### Test Set Performance Comparison

| Metric | Logistic Regression | Neural Network | Absolute Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **F1-Score (Primary)**[cite: 12] | **0.6562**[cite: 12] | **0.6712**[cite: 12] | **+1.50%**[cite: 12] |
| **Accuracy (Supporting)**[cite: 12] | **0.8524 (85.24%)**[cite: 12] | **0.8571 (85.71%)**[cite: 12] | **+0.47%**[cite: 12] |
| **5-Fold CV F1 (Train)**[cite: 12] | **0.6576 ± 0.0098**[cite: 12] | — | — |

### Confusion Matrix Breakdown
* **False Negatives (Missed Eligible Households):** The neural network decreased False Negatives from 964 to 912 (boosting Recall from 59% to 61%)[cite: 12].
* **False Positives (Misallocated Outreach):** Remained almost unchanged (478 vs. 484), maintaining Precision at ~75% across both classifiers[cite: 12].

---

## ⚖️ Ethical Considerations

Using demographic survey variables (such as sex, race, and native country) to predict income carries a substantial risk of codifying and entrenching historical socio-economic disparities into automated decisions[cite: 12]. In a targeted benefit or outreach program, false negatives exclude deserving households from critical awareness and support[cite: 12]. Therefore, **Logistic Regression** is ethically preferable because its linear coefficients can be audited directly by regulators and the public for disparate impact, unlike an opaque neural network[cite: 12].

---

## 📌 Key Findings & Conclusion

1. **Marginal Performance Difference:** The neural network achieved a slightly higher F1-score (+1.50%) primarily because its hidden ReLU layers captured non-linear interactions (e.g., education combined with hours worked)[cite: 12].
2. **Strong Linear Baseline:** On a medium-sized tabular dataset (~39k training rows) with one-hot encoded categories, the decision boundary is largely linearly separable, allowing logistic regression to capture most predictive signal[cite: 12].
3. **Complexity Trade-Off:** In public administration, a **1.5% gain in F1-score does not justify the added engineering overhead, stochastic training variability, and loss of interpretability**[cite: 12]. Logistic Regression represents the more practical, trustworthy, and deployable solution[cite: 12].

---

## 🚀 How to Run

### Requirements
* Python 3.10+[cite: 8, 12]
* Scikit-Learn[cite: 8, 12]
* Keras / TensorFlow[cite: 8, 12]
* Pandas, NumPy, Matplotlib[cite: 8, 12]

### Execution in Google Colab
1. Open `Final_Project_Option2_Tabular_NN_vs_Classical_Xen.ipynb` in Google Colab[cite: 12].
2. Select **Runtime** $\rightarrow$ **Run all** to sequentially execute the data loading, preprocessing, model training, and evaluation steps[cite: 12].
