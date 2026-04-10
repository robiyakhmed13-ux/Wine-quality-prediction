# 🍷 Wine Quality Prediction

A machine learning project that predicts whether a red wine is **good quality or not** using a **Random Forest Classifier**, based on its physicochemical properties.

---

## 📌 Project Overview

Wine quality is traditionally assessed by human experts — a slow and subjective process. This project trains a Random Forest model on measurable chemical properties to automatically classify a wine as good (score ≥ 7) or not good (score < 7).

| Item | Detail |
|------|--------|
| **Algorithm** | Random Forest Classifier |
| **Task** | Binary Classification |
| **Dataset** | [Red Wine Quality – Kaggle / UCI](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) |
| **Target** | `quality` → Good (1, score ≥ 7) / Not Good (0, score < 7) |

---

## 📂 Project Structure

```
wine_quality_prediction/
│
├── wine_quality_prediction.ipynb   # Jupyter Notebook (full walkthrough)
├── wine_quality_prediction.py      # Clean Python script
├── requirements.txt                # Dependencies
├── winequality-red.csv             # Dataset (download from Kaggle)
├── eda_plots.png                   # Generated EDA visualizations
├── correlation_heatmap.png         # Generated correlation heatmap
├── feature_importances.png         # Generated feature importance chart
├── confusion_matrix.png            # Generated confusion matrix
└── README.md
```

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `fixed acidity` | Most acids involved with wine (tartaric acid) |
| `volatile acidity` | Amount of acetic acid — too high leads to vinegar taste |
| `citric acid` | Adds freshness and flavor |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Amount of salt in the wine |
| `free sulfur dioxide` | Free form of SO₂ — prevents microbial growth |
| `total sulfur dioxide` | Total SO₂ concentration |
| `density` | Density of wine (close to water depending on alcohol/sugar) |
| `pH` | Acidity level (0 = very acidic, 14 = very basic) |
| `sulphates` | Wine additive that contributes to SO₂ levels |
| `alcohol` | Percent alcohol content |
| `quality` | ✅ **Target** — Score 0–10, binarized to Good (≥7) / Not Good (<7) |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/wine-quality-prediction.git
cd wine-quality-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `winequality-red.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) and place it in the project root.

### 4. Run the script
```bash
python wine_quality_prediction.py
```

Or open the notebook:
```bash
jupyter notebook wine_quality_prediction.ipynb
```

---

## 🔄 Pipeline

```
Raw CSV Data
    │
    ▼
Exploratory Data Analysis (quality dist, barplots, heatmap)
    │
    ▼
Label Binarization (quality ≥ 7 → Good, else Not Good)
    │
    ▼
Train / Test Split (80% / 20%, stratified)
    │
    ▼
Random Forest Training (100 estimators)
    │
    ▼
Accuracy Evaluation + Classification Report + Feature Importances
    │
    ▼
Single-instance Prediction
```

---

## 📈 Results

| Split | Accuracy |
|-------|----------|
| Training | ~100% (ensemble overfits training data) |
| Test | ~93% |

> The high training accuracy is expected for Random Forest. Test accuracy reflects true generalization.

---

## 🔑 Key Findings (EDA)

- **Alcohol** content shows the strongest positive correlation with quality
- **Volatile acidity** negatively impacts quality — wines rated higher tend to have lower volatile acidity
- **Citric acid** shows a positive relationship with quality scores

---

## 🔮 Sample Prediction

```python
# Input format:
# (fixed acidity, volatile acidity, citric acid, residual sugar,
#  chlorides, free sulfur dioxide, total sulfur dioxide, density,
#  pH, sulphates, alcohol)

sample_input = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)
result = predict_wine_quality(model, sample_input)
# Output: 🍷 This wine is predicted to be GOOD quality (score ≥ 7).
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** — data manipulation
- **numpy** — numerical operations
- **scikit-learn** — Random Forest, train/test split, metrics
- **seaborn / matplotlib** — data visualization

---

## 🚀 Future Improvements

- [ ] Try multi-class classification (predict exact quality score 3–8)
- [ ] Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`
- [ ] Add cross-validation for more robust evaluation
- [ ] Extend to white wine dataset and compare models
- [ ] Deploy as a Streamlit web app for live wine quality checks

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🙋 Author

**[Your Name]**  
[GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile)
