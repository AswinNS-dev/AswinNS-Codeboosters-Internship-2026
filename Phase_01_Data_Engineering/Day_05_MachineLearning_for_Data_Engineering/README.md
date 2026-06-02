# Day 5 — Student Reference Guide
## Machine Learning for Data Engineers | Phase 1 Finale
### Codeboosters Tech — Data Engineering + GenAI Internship

---

## The Full Circle — 5-Day Pipeline

```
Day 1  student_performance.csv → Loaded with Pandas, explored
Day 2  → Stored in SQLite, queried with SQL
Day 3  → ETL pipeline, cleaned and validated
Day 4  → Processed with PySpark at scale
Day 5  → Feature Engineering → ML Model → Predictions  ← TODAY
```

---

## Machine Learning Concepts

| Term | Definition | Example |
|------|-----------|---------|
| **Machine Learning** | Computer learns patterns from examples | Learn from 24 students to predict new ones |
| **Supervised Learning** | Learn from labelled data (input + answer) | 30 students with known programming scores |
| **Regression** | Predict a number | Predict programming_score |
| **Classification** | Predict a category | Predict pass/fail |
| **Training data** | Examples the model learns from | 80% = 24 students |
| **Test data** | Unseen data to measure accuracy | 20% = 6 students |
| **Overfitting** | Memorizes training data, fails on new | High train R², low test R² |
| **Feature** | An input column used for prediction | math_score, attendance_percentage |
| **Target** | The column we want to predict | programming_score |

---

## Feature Engineering — Required Before ML

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('student_performance.csv')
df_ml = df.copy()   # Always copy — never modify original

# ── Step 1: Encode text columns ──
le_gender = LabelEncoder()
df_ml['gender_encoded'] = le_gender.fit_transform(df_ml['gender'])
# Female→0, Male→1

le_dept = LabelEncoder()
df_ml['department_encoded'] = le_dept.fit_transform(df_ml['department'])
# Civil→0, Computer Science→1, Electronics→2, Mechanical→3

# ── Step 2: Select features and target ──
feature_cols = ['math_score', 'science_score', 'english_score',
                'attendance_percentage', 'gender_encoded', 'department_encoded']

X = df_ml[feature_cols]           # Feature matrix (30, 6)
y = df_ml['programming_score']    # Target vector  (30,)

# ── Step 3: Train/test split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# test_size=0.2  → 20% test (6 students), 80% train (24 students)
# random_state=42 → same split every run

# ── Step 4: Scale features ──
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit AND transform
X_test_scaled  = scaler.transform(X_test)        # transform ONLY — no fit!
# CRITICAL: NEVER fit_transform on test data (data leakage)
```

---

## Training Three Models

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Model 1: Linear Regression ──
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)     # TRAIN
lr_pred = lr.predict(X_test_scaled) # PREDICT

# ── Model 2: Decision Tree ──
dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)
dt_pred = dt.predict(X_test_scaled)

# ── Model 3: Random Forest ──
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

# ── Evaluate all three ──
for name, pred in [("Linear Reg", lr_pred), ("Decision Tree", dt_pred), ("Random Forest", rf_pred)]:
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    print(f"{name:<15}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
```

---

## Evaluation Metrics

| Metric | Meaning | Formula | Good Value |
|--------|---------|---------|------------|
| **MAE** | Average absolute error | Mean of \|actual − predicted\| | < 10 marks |
| **RMSE** | Error with large-mistake penalty | √Mean of (actual − predicted)² | < 12 marks |
| **R²** | How much variation is explained | 1 − (model error / total variation) | > 0.70 |

**R² interpretation:**
- R² = 1.00 → Perfect predictions
- R² = 0.85 → Model explains 85% of score variation
- R² = 0.00 → Model is no better than predicting the mean
- R² < 0.00 → Model is worse than predicting the mean (something is wrong)

---

## Predicting for New Students

```python
# Define new student (not in training data)
new_student = pd.DataFrame({
    'math_score':            [88],
    'science_score':         [82],
    'english_score':         [75],
    'attendance_percentage': [92],
    'gender_encoded':        [1],    # 1 = Male
    'department_encoded':    [1],    # 1 = Computer Science
})

# Scale using training scaler
new_scaled = scaler.transform(new_student[feature_cols])

# Predict
predicted_score = rf.predict(new_scaled)[0]
print(f"Predicted programming score: {predicted_score:.1f}")
```

---

## Model Comparison Summary

| Model | Strengths | Weaknesses | Use When |
|-------|-----------|-----------|----------|
| **Linear Regression** | Simple, fast, interpretable | Cannot capture non-linear patterns | Baseline comparison |
| **Decision Tree** | Captures non-linear patterns, explainable | Overfits without max_depth | Need interpretable rules |
| **Random Forest** | Best accuracy, resistant to overfitting | Slower, less interpretable | Accuracy is priority |

**Random Forest almost always wins** because averaging 100 trees cancels out individual errors (ensemble learning).

---

## Feature Importance

After training Random Forest, inspect which features matter most:

```python
feat_imp = dict(zip(feature_cols, rf.feature_importances_))
for feat, imp in sorted(feat_imp.items(), key=lambda x: -x[1]):
    print(f"  {feat:<28}: {imp:.3f}")
# feature_importances_ values sum to 1.0
# Higher = more important for predictions
```

---

## Scikit-learn Workflow Summary

```
1. Load data           pd.read_csv()
2. Copy                df.copy()
3. Encode text         LabelEncoder().fit_transform()
4. Select features     X = df[feature_cols]
5. Define target       y = df['target_col']
6. Split               train_test_split(X, y, test_size=0.2)
7. Scale               StandardScaler().fit_transform(X_train)
                       scaler.transform(X_test)     ← no fit!
8. Train               model.fit(X_train, y_train)
9. Predict             model.predict(X_test)
10. Evaluate           r2_score(y_test, predictions)
11. New prediction     model.predict(scaler.transform(new_data))
```

---

## Common Errors — Day 5

| Error | Cause | Fix |
|-------|--------|-----|
| `ValueError: could not convert string to float` | Text column not encoded | Apply `LabelEncoder` to `gender` and `department` |
| `ValueError: Input contains NaN` | Missing values in features | `df.fillna(df.median(numeric_only=True))` |
| `NotFittedError: call fit before predict` | `predict()` called before `fit()` | Always `model.fit(X_train, y_train)` first |
| `R² is negative` | Wrong feature selection or encoding error | Check `df.dtypes` — all features must be numeric |
| `DataConversionWarning` | `y` is 2D instead of 1D | Use `df['col']` not `df[['col']]` for target |

---

## Practice Questions

1. What is the difference between supervised and unsupervised ML? Give one real-world example of each.
2. Why can't you feed raw `student_performance.csv` directly into Scikit-learn?
3. What does `train_test_split()` do and why is it necessary?
4. A model has R²=0.92 on training and R²=0.41 on test. What is this problem called?
5. Write code to train a Random Forest and print its R² score.
6. Which Day 1–4 step is most critical for ML model quality and why?

---

## Phase 1 Complete Checklist

- ✅ `LabelEncoder` applied to gender and department
- ✅ `StandardScaler` applied correctly (fit on train, transform on test only)
- ✅ All 3 models trained and compared
- ✅ `ml_results_dashboard.png` saved (3 panels)
- ✅ Predicted score printed for a new hypothetical student
- ✅ Phase 1 Final Mini Project completed
- ✅ `student_analytics_system.png` saved
- ✅ Notebook uploaded to GitHub
- ✅ Commit: `Add Day 5: ML Pipeline + Phase 1 Final Project`

---

## Phase 2 Preview (Days 6–10)

| Day | Topic |
|-----|-------|
| Day 6 | GenAI + Prompt Engineering (Groq API) |
| Day 7 | Embeddings + Semantic Search (ChromaDB) |
| Day 8 | RAG Systems — PDF chatbot |
| Day 9 | AI Agents + Text-to-SQL |
| Day 10 | Capstone Project |

**Get ready:** Register for a free Groq API key at `console.groq.com` before Day 6.

---
*Codeboosters Tech | Data Engineering + GenAI Internship | Day 5 of 10 | Phase 1 Finale*✅
