# 🎯 Recruitment Offer Acceptance Predictor

> > **AI-Powered Machine Learning System for Job Offer Acceptance Prediction with 95% Accuracy**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25-brightgreen.svg)

---

## 📋 Table of Content

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

A Machine Learning–based prediction system that supports HR and Recruitment teams to:
- ✅ Predict the likelihood of candidates accepting a job offer
- 📊 Analyze key factors influencing candidates’ decisions
- 🔮 Perform what-if analysis to optimize recruitment strategies
- 💡 Provide data-driven strategic recommendations

### Key Metrics
- **Model Accuracy:** 95%
- **Training Data:** 5,000 recruitment records
- **Features:** 13 engineered features
- **Classes:** 3 (Likely Reject, Uncertain, Likely Accept)

---

## ✨ Features

### 1. Accurate Predictions
- 95% accuracy rate
- Multi-class classification (Reject, Uncertain, Accept)
- Probability distribution for each class

### 2. Interactive UI
- Modern, professional design
- Real-time predictions
- Interactive charts using Plotly

### 3. What-If Analysis
- Faster hiring scenario
- Better package scenario
- Referral source scenario
- Side-by-side comparison

### 4. Detailed Insights
- Contributing factors explanation
- Feature engineering display
- Strategic recommendations
- Risk assessment

### 5. Key Metrics Display
- Applicant Pressure Index
- Cost Efficiency Daily
- Difficulty Index (log-scaled)
- Acceptance Pressure Features

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone Repository
```bash
git clone trimedian-app
cd recruitment-predictor
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Model Artifacts
**Ensure the `model_artifacts/` folder contains 4 files:**
```
model_artifacts/
├── recruitment_model.joblib      # Trained XGBoost model
├── label_encoders.joblib          # Categorical encoders
├── scaler.joblib                  # StandardScaler (CRITICAL!)
└── metadata.joblib                # Feature names & mappings
```

---

## 🚀 Usage

### Running Locally

```bash
streamlit run app.py
```

The app will open in the browser at `http://localhost:8501`

### Using the App

1. **Input Parameters** (Sidebar):
   - Department
   - Job Title
   - Recruitment Source
   - Number of Applicants
   - Time to Hire (days)
   - Cost per Hire ($)

2. **Click "Predict"**

3. **View Results:**
   - Prediction outcome
   - Confidence level
   - Probability distribution
   - Contributing factors
   - What-if scenarios
   - Strategic recommendations

---

## 📁 Project Structure

```
recruitment-predictor/
│
├── app.py
│
├── Stage_3.ipynb       
│
├── model_saving_fixed.py            # Helper functions for model saving
│
├── model_artifacts/                 # Model artifacts directory
│   ├── recruitment_model.joblib     # Trained model
│   ├── label_encoders.joblib        # Encoders
│   ├── scaler.joblib                # StandardScaler (NEW!)
│   └── metadata.joblib              # Metadata
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
```

---

## 🔬 Technical Details

### Model Architecture
- **Algorithm:** XGBoost (Gradient Boosted Trees)
- **Type:** Multi-class Classification
- **Classes:** 3 (Likely Reject, Uncertain, Likely Accept)
- **Hyperparameters:** Tuned via RandomizedSearchCV

### Feature Engineering Pipeline

#### Raw Features:
- department
- job_title
- source
- num_applicants
- time_to_hire_days
- cost_per_hire

#### Engineered Features:
1. **applicant_pressure_index** = num_applicants / time_to_hire_days
2. **cost_efficiency_daily** = cost_per_hire / time_to_hire_days
3. **cost_per_applicant** = cost_per_hire / num_applicants
4. **hire_days_per_applicant** = time_to_hire_days / num_applicants
5. **difficulty_index** = num_applicants × cost_per_hire × time_to_hire_days
6. **difficulty_index_log** = log(difficulty_index)
7. **acceptance_cost_pressure** = cost_per_hire × (1 - offer_acceptance_rate)
8. **acceptance_time_pressure** = time_to_hire_days × (1 - offer_acceptance_rate)

#### Categorical Features:
- **time_to_hire_category:** Fast (<30), Medium (30-60), Slow (>60)
- **cost_bucket:** Low (<3000), Medium (3000-6000), High (>6000)

### Preprocessing Steps:
1. Feature engineering (8 new features)
2. Categorical encoding (Label Encoding for dept, job, source)
3. Categorical mapping (time_category, cost_bucket)
4. **StandardScaler** (8 numerical features)
   
### Model Training:
- Train/Test Split: 80/20
- SMOTE for class balancing
- RandomizedSearchCV for hyperparameter tuning
- Cross-validation: 5-fold

---

## 🌐 Deployment

### Option 1: Local Deployment
```bash
streamlit run app.py
```

### Option 2: Streamlit Cloud

1. Push code ke GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

**Important:** Pastikan `model_artifacts/` folder ter-push ke GitHub!

### Option 3: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t recruitment-predictor .
docker run -p 8501:8501 recruitment-predictor
```

---

## 🐛 Troubleshooting

### Issue 1: Model Files Not Found
```
Error: [Errno 2] No such file or directory: 'model_artifacts/recruitment_model.joblib'
```

**Solution:** Pastikan Anda menjalankan notebook untuk generate model artifacts terlebih dahulu.

---

### Issue 2: Predictions Don't Match Notebook
```
Notebook prediction: Class 2
Streamlit prediction: Class 0
```

**Solution:** 
1. ✅ Gunakan `app.py`
2. ✅ Pastikan `scaler.joblib` ada dan di-load
3. ✅ Verifikasi formula acceptance_cost_pressure dan acceptance_time_pressure

---

### Issue 3: Scaler Not Found
```
Error: No such file: 'model_artifacts/scaler.joblib'
```

**Solution:** 
1. Buka notebook `_Model_Stage_3_FIXED.ipynb`
2. Jalankan cell untuk save scaler (setelah Cell 95)
3. Re-run model saving cell

---

### Issue 4: Wrong Predictions
```
All predictions are "Likely Reject" or model seems biased
```

**Solution:**
- Check if scaler is being applied: `X[num_cols_to_scale] = scaler.transform(X[num_cols_to_scale])`
- Verify formulas match exactly with training notebook
- Print intermediate feature values to debug

---

## 📊 Model Performance

### Confusion Matrix (Test Set)
```
                  Predicted
Actual      Reject  Uncertain  Accept
Reject        458        12       8
Uncertain      15       201      14
Accept          7         8      277

Accuracy: 95%
```

### Per-Class Metrics
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Likely Reject | 0.95 | 0.96 | 0.95 |
| Uncertain | 0.91 | 0.87 | 0.89 |
| Likely Accept | 0.93 | 0.95 | 0.94 |

### Feature Importance (Top 5)
1. acceptance_time_pressure (0.187)
2. acceptance_cost_pressure (0.156)
3. difficulty_index_log (0.142)
4. num_applicants (0.121)
5. source (0.098)

---

## 🎓 Lessons Learned

### 1. Preprocessing Consistency
Training dan inference HARUS menggunakan preprocessing yang IDENTIK:
- ✅ Same formulas
- ✅ Same scaling
- ✅ Same encoding
- ✅ Same feature order

### 2. Save Everything
Jangan hanya save model! Save juga:
- ✅ Encoders
- ✅ Scalers
- ✅ Feature names
- ✅ Metadata

### 3. Always Verify
Before deployment:
- ✅ Test loading artifacts
- ✅ Compare predictions
- ✅ Validate probabilities

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

## 👤 Author

**Final Project Bootcamp Data Science**

- Notebook: `_Model_Stage_3.ipynb`
- Model: XGBoost with 95% accuracy
- Framework: Streamlit + Plotly

---

## 🙏 Acknowledgments

- Scikit-learn for preprocessing utilities
- XGBoost for powerful gradient boosting
- Streamlit for beautiful web framework
- Plotly for interactive visualizations

---

**Remember:** Machine Learning in production is 90% about getting the data pipeline right! 🚀

**Status:** ✅ Production Ready  
**Version:** 2.0 (Fixed)  
**Date:** January 21, 2026
