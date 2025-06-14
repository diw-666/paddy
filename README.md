# 🌾 Sri Lanka Rice Analytics Dashboard

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-View%20Here-blue)](https://diw-666-paddy-dashboard-kgigph.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Interactive rice analytics platform for Sri Lanka (2004-2023) with ML-powered predictions and insights.

## 🎯 Quick Links

- [📊 Live Dashboard](https://diw-666-paddy-dashboard-kgigph.streamlit.app/) - Explore data interactively
- [📈 Model Performance](#model-performance) - View prediction metrics
- [🔧 Local Setup](#local-setup) - Run locally

## 📊 Model Performance

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Yield Prediction | 357.50 | 461.55 | 0.804 |
| Production Prediction | 11,163 | 24,289 | 0.934 |

## 🔧 Local Setup

```bash
# Clone & install
git clone https://github.com/yourusername/paddy.git
cd paddy
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard.py
```

## 📦 Key Features

- **📈 Analytics**: District-wise yield & production analysis
- **🤖 ML Predictions**: Real-time yield forecasting
- **📊 Visualizations**: Interactive charts & trends
- **📋 Data**: 1000+ records across 36 districts (2004-2023)

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML**: Scikit-learn
- **Data**: Pandas, NumPy
- **Viz**: Plotly, Seaborn

---

<div align="center">
  <sub>Built with ❤️ by Yasiru Vithana</sub>
</div> 