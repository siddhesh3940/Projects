# Retail Analytics Web App
**Data Warehousing & Mining Project**

A comprehensive web-based application for retail data analysis using machine learning and data mining techniques.

## Features

### 📁 Data Upload & Preprocessing
- CSV file upload with validation
- Missing value handling (median for numeric, mode for categorical)
- Outlier detection and treatment using IQR method
- Categorical encoding with Label Encoder
- Summary statistics and data quality reports

### 🔍 Exploratory Data Analysis (EDA)
- Distribution plots and histograms
- Correlation heatmaps for numeric variables
- Box plots for outlier visualization
- Bar charts and pie charts for categorical analysis
- Interactive column selection

### 🌳 Decision Tree Classification
- Customer purchase behavior classification
- Feature selection interface
- Decision tree visualization (max depth 3 for clarity)
- Accuracy metrics and classification reports
- Automatic encoding of categorical variables

### 🎯 K-Means Clustering
- Customer/product segmentation
- Elbow method for optimal cluster selection
- Silhouette score evaluation
- Interactive cluster visualization
- Standardized feature scaling
- Cluster analysis and interpretation

### 🔗 Association Rule Mining (Apriori)
- Frequent itemset discovery
- Association rule generation
- Support, confidence, and lift metrics
- Interactive parameter tuning
- Rule visualization with scatter plots
- Market basket analysis

## Installation

1. Install Python 3.8+
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Upload a CSV file or use the provided sample data

4. Navigate through different analysis sections using the sidebar

## Data Format

Your CSV should include:
- **Numeric columns**: For clustering and classification features
- **Categorical columns**: For association rule mining
- **Transaction ID**: Customer/order identifier
- **Item/Product column**: For market basket analysis

## Sample Data

The app includes `sample_retail_data.csv` with:
- Customer transactions
- Product categories
- Purchase quantities and prices
- Customer demographics
- Temporal data

## Tech Stack

- **Backend**: Python, Streamlit
- **ML Libraries**: Scikit-learn, MLxtend
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Processing**: Pandas, NumPy

## Key Algorithms

1. **Decision Tree**: Customer classification with pruning
2. **K-Means**: Unsupervised customer segmentation
3. **Apriori**: Frequent pattern mining for recommendations

## Performance Features

- Efficient data preprocessing pipeline
- Interactive parameter tuning
- Real-time visualization updates
- Memory-optimized processing
- Error handling and validation

## Deployment

Deploy to Streamlit Cloud:
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy with one click

Alternative: Deploy to Heroku with `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```