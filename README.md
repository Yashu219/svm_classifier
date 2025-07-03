# Support Vector Machines (SVM) - Breast Cancer Dataset

This task demonstrates the implementation of Support Vector Machines for binary classification using the Breast Cancer Dataset.

---

## Objective

- Apply SVM with both **linear** and **RBF kernels**
- Tune hyperparameters `C` and `gamma` using **GridSearchCV**
- Evaluate the model using accuracy, classification report, confusion matrix
- Perform **5-fold cross-validation**

---

## Dataset

Used built-in `load_breast_cancer` dataset from `sklearn.datasets`.

- 30 features related to breast tissue properties
- 2 classes: **Malignant (0)** and **Benign (1)**

---

## Tools Used

- Python
- Scikit-learn
- Pandas
- Matplotlib & Seaborn

---

## Results

| Model           | Accuracy | Best Params            |
|----------------|----------|------------------------|
| Linear Kernel   | 95.61%   | `C=1` (default)         |
| RBF Kernel      | **97.36%** | `C=1`, `gamma='scale'`  |
| Cross-Validation (RBF) | ~97.36% | Very consistent across folds |

**Best Parameters Found by GridSearchCV:**
```python
{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
