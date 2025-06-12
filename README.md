# 🌾 Sri Lanka Rice Analytics Dashboard

A rice yield prediction and analytics system for Sri Lanka, analyzing agricultural data from 2004-2023 across all districts and seasons.

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

3. **Open in browser**: `http://localhost:8501`

## ✨ Features

- **Analytics**: Compare yield and production across districts and seasons
- **Predictions**: ML-powered yield forecasting with cultivation inputs
- **Insights**: Automated analysis of trends and patterns  
- **Visualizations**: Interactive charts and data exploration

## 📊 Data

- **Coverage**: 20 years (2004-2023), 25+ districts, Maha & Yala seasons
- **Records**: 1000+ seasonal district records
- **Features**: Cultivation areas, harvest data, yield metrics, production

## 🔬 Models

- **Random Forest**: Yield prediction (R² > 0.7)
- **Gradient Boosting**: Production prediction (R² > 0.8)
- **Features**: 13 engineered features including efficiency ratios

## 📁 Structure

```
├── dashboard.py          # Main Streamlit app
├── data_processor.py     # Data processing
├── ml_model.py          # ML models
├── data/                # Raw CSV files
└── requirements.txt     # Dependencies
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML**: Scikit-learn
- **Data**: Pandas, NumPy
- **Viz**: Plotly, Seaborn

---

Built for Sri Lankan agriculture 🇱🇰 