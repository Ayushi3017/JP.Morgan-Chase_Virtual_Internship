## Projects

### 1. Natural Gas Price Estimation & Storage Simulation

* **Goal:** Predict monthly natural gas prices and optimize storage injections.
* **Features:**

  * Dynamic price estimation via interpolation
  * Injection volume simulation respecting storage limits
  * Cost calculation per injection

**Example:**

```python
price = estimate_price("2025-01-15")
print(price)
```

---

### 2. Loan Default Prediction & Expected Loss

* **Goal:** Predict borrower **Probability of Default (PD)** and compute **Expected Loss (EL)**.
* **Features:**

  * Uses borrower attributes: income, debt, loan amount, FICO score, employment years
  * Logistic Regression for PD prediction
  * EL calculation: `EL = PD × Exposure × (1 - Recovery Rate)`

**Example:**

```python
loan_example = {
    'credit_lines_outstanding': 5,
    'loan_amt_outstanding': 20000,
    'total_debt_outstanding': 5000,
    'income': 60000,
    'years_employed': 3,
    'fico_score': 710
}
el = expected_loss(loan_example)
print(el)
```

---

### 3. FICO Score Quantization & Credit Rating Buckets

* **Goal:** Map FICO scores into discrete credit rating buckets for risk assessment.
* **Features:**

  * Configurable number of rating buckets
  * **MSE-based** and **likelihood-based** quantization
  * Lower rating = better creditworthiness
  * Generalizable to future datasets

**Example:**

```python
rating = map_fico_to_rating(720, bin_edges, 5)
print(rating)
```

---

## Technologies

Python 3 | Pandas | NumPy | scikit-learn | Optional: XGBoost, RandomForest

---

## Key Learnings

* Time series interpolation & storage simulation
* Predictive modeling of loan defaults and expected losses
* Data-driven FICO score quantization for credit ratings
* Modular, generalizable functions for real-world financial datasets

