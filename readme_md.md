# Student Performance Prediction — Machine Learning Case Study

An end-to-end machine learning case study predicting secondary school student academic outcomes using both **Regression** (exact final grade `G3` out of 20) and **Classification** (`Pass` vs. `Fail` where `G3 >= 10` is a Pass). The project also features an interactive command-line prediction tool allowing users to input student attributes and compare predictions across four trained models.

---

## 📊 Dataset Overview

The analysis combines two datasets from the UCI Machine Learning Repository covering student achievement in two distinct subjects:
* **`student-mat.csv`** — Mathematics course (`395` records, `33` attributes)
* **`student-por.csv`** — Portuguese language course (`649` records, `33` attributes)
* **Combined Dataset** — `1,044` total student records split into **80% training (`835` samples)** and **20% testing (`209` samples)** (`random_state=42`).

### Selected Features (12 Core Predictors)
To keep the model interpretable and practical for interactive input, 12 key demographic, behavioral, and academic features were selected:

| Feature | Description | Scale / Range |
| :--- | :--- | :--- |
| `age` | Student's age | `15` – `22` |
| `Medu` | Mother's education level | `0` (none) to `4` (higher education) |
| `Fedu` | Father's education level | `0` (none) to `4` (higher education) |
| `studytime` | Weekly study time | `1` (<2h), `2` (2–5h), `3` (5–10h), `4` (>10h) |
| `failures` | Number of past class failures | `0` – `3` |
| `absences` | Number of school absences | `0` – `93` |
| `goout` | Going out with friends | `1` (very low) to `5` (very high) |
| `Dalc` | Workday alcohol consumption | `1` (very low) to `5` (very high) |
| `Walc` | Weekend alcohol consumption | `1` (very low) to `5` (very high) |
| `health` | Current health status | `1` (very bad) to `5` (very good) |
| `G1` | First period grade | `0` – `20` |
| `G2` | Second period grade | `0` – `20` |

---

## ⚙️ Methodology & Pipeline

1. **Data Integration & Target Engineering:** Merged Math and Portuguese records and created a binary target `pass` (`1` if `G3 >= 10`, else `0`).
2. **Preprocessing:** Applied `StandardScaler` to normalize features for Linear Regression and Logistic Regression, while passing unscaled features to tree-based Random Forest models.
3. **Model Training:** Trained four supervised learning algorithms:
   * **Linear Regression**
   * **Random Forest Regressor** (`n_estimators=150`, `max_depth=12`, `random_state=42`)
   * **Logistic Regression** (`max_iter=1000`, `random_state=42`)
   * **Random Forest Classifier** (`n_estimators=150`, `max_depth=12`, `random_state=42`)
4. **Interactive CLI Predictor:** Built a user-input pipeline with boundary validation and default fallback values to test individual student profiles across all four models in real time.

---

## 📈 Model Performance Summary

### 1. Regression Results (Predicting Exact Final Grade `G3`)
| Algorithm | $R^2$ Score | Mean Absolute Error (MAE) | Status |
| :--- | :---: | :---: | :--- |
| **Random Forest Regressor** | **0.801** | **0.965** | **Best Regression Model** |
| Linear Regression | 0.801 | 0.978 | Baseline |

* **Key Takeaway:** **Random Forest Regressor** achieved the lowest Mean Absolute Error (`0.965` grade points) while explaining `80.1%` of the variance in final grades.

### 2. Classification Results (Predicting `Pass` vs. `Fail`)
| Algorithm | Accuracy | ROC-AUC Score | Status |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | **87.1%** | **0.963** | **Best Classification Model** |
| Random Forest Classifier | 87.1% | 0.952 | Runner-up |

* **Key Takeaway:** Both classifiers reached `87.1%` test accuracy, with **Logistic Regression** selected as the top classifier due to its superior **ROC-AUC score (`0.963`)**, demonstrating strong class separation between passing and failing students.

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/SURYA1117/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Install Required Libraries
Ensure Python 3.8+ is installed, then install the dependencies:
```bash
pip install pandas numpy scikit-learn jupyter
```

### 3. Verify Project Structure
Place both CSV datasets in the root folder alongside your notebook:
```text
student-performance-prediction/
├── student-mat.csv
├── student-por.csv
├── student_performance_case_study.ipynb
└── README.md
```

### 4. Run the Notebook
Launch Jupyter Notebook or open the `.ipynb` file in VS Code / Google Colab and run all cells sequentially to train the models, view evaluation metrics, and launch the interactive student grade predictor.