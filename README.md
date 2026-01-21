# 🎯 Dynamic Pricing Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-green.svg)](https://scikit-learn.org/)

An AI-powered dynamic pricing system that uses **Machine Learning** and **Reinforcement Learning** to optimize e-commerce product prices in real-time, maximizing profit while maintaining market competitiveness.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

---

## 🌟 Overview
Note: The dataset is synthetically generated to simulate realistic e-commerce pricing scenarios.

Traditional e-commerce platforms rely on **static pricing strategies** that fail to adapt to:
- ⚡ Rapid competitor price changes
- 📈 Demand fluctuations (festivals, weekends)
- 📦 Inventory levels
- 👥 Customer segments

This project implements a **Dynamic Pricing Recommendation Engine** that:
- 🤖 Predicts customer demand using ML regression models
- 🧠 Learns optimal pricing strategies using Q-Learning (Reinforcement Learning)
- 💰 Maximizes profit while staying competitive
- ⚡ Provides real-time price recommendations

### Key Highlights
- **85% prediction accuracy** (R² score) for demand forecasting
- **16% average profit improvement** over static pricing
- **307% profit boost** in competitive scenarios
- **Jupyter Notebook-based** - All code in one comprehensive notebook

---

## ✨ Features

### 🤖 Machine Learning
- **Multiple Regression Models**: Linear Regression, Decision Tree, Random Forest, Gradient Boosting
- **Automated Model Selection**: Evaluates and selects best-performing model
- **Feature Engineering**: Price elasticity, discount percentages, rolling averages
- **High Accuracy**: R² score of 0.85 with ±6.5 units MAE

### 🧠 Reinforcement Learning
- **Q-Learning Algorithm**: Custom implementation with 30 states × 5 actions
- **Smart Actions**: Decrease 10%, Decrease 5%, Keep Same, Increase 5%, Increase 10%
- **Adaptive Learning**: Epsilon-greedy exploration with decay (1,000 episodes)
- **Profit Optimization**: Reward function based on (Price - Cost) × Demand

### 📊 Analytics & Insights
- **Price Elasticity Analysis**: Understand demand sensitivity to price changes
- **Seasonal Pattern Detection**: Festival and weekend demand spikes
- **Competitor Analysis**: Real-time price competitiveness tracking
- **Static vs Dynamic Comparison**: Clear profit improvement metrics

---

## 🎥 Demo

### Example Results

**Test Case: Competitive Scenario**
```
Input:
  Current Price: ₹2,500
  Competitor Price: ₹2,200
  Market: Weekday, Non-festival, High inventory

Output:
  ✅ Recommended Price: ₹2,375 (Decrease 5%)
  📈 Demand: 4 → 17 units (+325%)
  💰 Profit: ₹3,754 → ₹15,305 (+307%)
```

---

## 🏗️ Architecture

### System Flow
```
Input Data → Feature Engineering → ML Model + RL Agent → Recommendation Engine → Output
```

### Components

1. **Data Layer**: Product details, market conditions, time features
2. **Feature Engineering**: Price difference, discount %, profit margin, rolling averages
3. **ML Module**: Random Forest regressor for demand prediction
4. **RL Module**: Q-Learning agent for price optimization
5. **Recommendation Engine**: Combines ML + RL for optimal pricing

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- pip package manager

### Setup Instructions

1. **Clone the repository** (or download ZIP)
```bash
   git clone https://github.com/yourusername/dynamic-pricing-engine.git
   cd dynamic-pricing-engine
```

2. **Install dependencies**
```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

   Or use requirements file:
```bash
   pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
   jupyter notebook
```

4. **Open and run**
   - Open `Dynamic_Pricing_Engine.ipynb`
   - Run all cells sequentially (Shift + Enter)
   - Wait for training to complete (~15 minutes total)

---

## 💻 Usage

### Running the Project

1. **Open Jupyter Notebook**
```bash
   jupyter notebook
```

2. **Open the main notebook**
   - Navigate to `Dynamic_Pricing_Engine.ipynb`
   - This notebook contains all code in sequential order

3. **Run the notebook**
   - Execute cells sequentially from top to bottom
   - **Step 1:** Data Generation (~1 min)
   - **Step 2:** Data Exploration & Visualization (~1 min)
   - **Step 3:** Feature Engineering (~30 sec)
   - **Step 4:** ML Model Training (~2 min)
   - **Step 5:** Price Elasticity Analysis (~30 sec)
   - **Step 6:** RL Agent Training (~8-10 min for 1,000 episodes)
   - **Step 7:** Price Recommendation Engine (~30 sec)
   - **Step 8:** Results & Evaluation (~1 min)

4. **View results**
   - All visualizations are displayed inline
   - Models are automatically saved to `models/` folder
   - Processed data saved to `data/` folder

---

## 📁 Project Structure
```
Dynamic_Pricing_Engine/
│
├── data/                          # Generated datasets
│   ├── pricing_data.csv           # Raw synthetic dataset
│   └── pricing_data_processed.csv # Processed dataset with features
│
├── models/                        # Saved trained models
│   ├── best_demand_model.pkl      # Best ML model (Random Forest)
│   ├── rl_agent.pkl               # Trained Q-Learning agent
│   ├── pricing_environment.pkl    # RL environment
│   └── encoders.pkl               # Label encoders
│
├── app.py
│
├── templates
│   ├──index.html
│   ├──home.html
│
│
├── static
│   ├──styles.css
|
├── dynamic_pricing_engine.ipynb   # 📓 Main Jupyter notebook (ALL CODE)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

### Notebook Organization

**Dynamic_Pricing_Engine.ipynb** contains 8 major sections:

1. **Data Generation** - Synthetic dataset creation (2,000 transactions)
2. **Data Exploration** - Visualization and statistical analysis
3. **Feature Engineering** - Price elasticity, encoding, derived features
4. **ML Model Training** - Demand prediction models comparison
5. **Price Elasticity Analysis** - Understanding price sensitivity
6. **RL Agent Training** - Q-Learning implementation (1,000 episodes)
7. **Price Recommendation** - Combining ML + RL for optimal pricing
8. **Evaluation** - Static vs Dynamic pricing comparison with results

---

## 📊 Results

### Model Performance

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | 0.76 | 8.2 | 11.4 |
| Decision Tree | 0.82 | 7.1 | 9.8 |
| **Random Forest** | **0.85** ✅ | **6.5** | **8.9** |
| Gradient Boosting | 0.84 | 6.8 | 9.1 |

### Business Impact

| Metric | Static Pricing | Dynamic Pricing | Improvement |
|--------|---------------|-----------------|-------------|
| **Average Profit** | ₹82,000 | ₹95,000 | **+16%** |
| **Competitive Scenario** | ₹3,754 | ₹15,305 | **+307%** |

### Key Insights

📈 **Demand Patterns**
- Weekend demand: **+20%** vs weekdays
- Festival periods: **+40-50%** demand spike
- November-December peak: **+35%** average

💰 **Price Elasticity**
- Electronics: High sensitivity (elasticity < -1)
- Accessories: Moderate sensitivity (-1 < elasticity < 0)
- 5% price drop → 300%+ demand increase (competitive markets)

🎯 **Optimal Actions**
- Competitive pressure → Decrease 5-10%
- High demand + Low inventory → Increase 5-10%
- Balanced market → Keep same or slight adjustment

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Jupyter Notebook** - Interactive development environment
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - Machine learning models and preprocessing

### Machine Learning
- **Random Forest Regressor** - Demand prediction (best performer)
- **Gradient Boosting** - Alternative high-accuracy model
- **Label Encoding** - Categorical feature transformation

### Reinforcement Learning
- **Q-Learning** - Custom implementation from scratch
- **Epsilon-Greedy Strategy** - Exploration-exploitation balance
- **Reward Shaping** - Profit-based optimization

### Data Visualization
- **Matplotlib** - Statistical plots and charts
- **Seaborn** - Advanced data visualization

---

## 🔮 Future Enhancements

### Short-term Improvements
- [ ] **Flask Web Application**: Basic demo interface implemented
- [ ] **Deep Q-Learning (DQN)**: Neural network-based RL
- [ ] **Extended Training**: Increase to 5,000+ episodes
- [ ] **Business Rule Overrides**: Add constraints for peak demand scenarios

### Medium-term Goals
- [ ] **Real-time Data Integration**: Connect to competitor pricing APIs
- [ ] **Multi-product Optimization**: Portfolio-level pricing strategies
- [ ] **Customer Lifetime Value**: Factor in long-term relationships
- [ ] **Modular Code Structure**: Refactor into production-ready components

### Long-term Vision
- [ ] **Time-series Forecasting**: LSTM/Prophet for demand prediction
- [ ] **Multi-agent RL**: Competitive pricing simulation
- [ ] **Personalized Pricing**: Customer segment-specific recommendations
- [ ] **Cloud Deployment**: AWS/Azure with auto-scaling

---

## 👤 Contact

**Soham**
- GitHub: https://github.com/Soham-global
- LinkedIn: https://www.linkedin.com/in/sohamkalsi/
- Project: Dynamic Pricing Recommendation Engine
- Internship Project - January 2026

---

## 🙏 Acknowledgments

- **Assigned By**: Maziya Iffat
- **Inspiration**: Real-world dynamic pricing systems (Amazon, Uber, Airlines)
- **Dataset**: Synthetically generated to simulate e-commerce patterns
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib communities

---

## 📚 References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
3. Ferreira, K. J., et al. (2016). "Dynamic Pricing with an Unknown Demand Model"
4. Chen, M., & Chen, Z. (2015). "Recent developments in dynamic pricing research"

---

<div align="center">

### ⭐ Dynamic Pricing Recommendation Engine

**Built with ❤️ using Python, Machine Learning & Reinforcement Learning**

*Internship Project - January 2026*

</div>