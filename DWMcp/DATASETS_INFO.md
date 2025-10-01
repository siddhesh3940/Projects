# 📊 Retail Analytics Datasets

## Available Datasets for Analysis

### 1. 🛍️ **large_retail_dataset.csv** (Primary Dataset)
**Size:** 10,000 transactions | **Customers:** 2,000 | **Products:** 800
- **Date Range:** 2022-01-01 to 2024-03-31
- **Categories:** 10 major categories with 60+ subcategories
- **Features:** 24 columns including customer demographics, product details, pricing, and business metrics

**Key Columns:**
- `TransactionID`, `CustomerID`, `ProductID`
- `Category`, `Subcategory`, `Brand`, `ProductName`
- `Quantity`, `UnitPrice`, `TotalAmount`, `Discount`
- `CustomerAge`, `CustomerGender`, `CustomerCity`, `CustomerSegment`
- `PaymentMethod`, `Channel`, `Season`, `Revenue`, `Profit`

**Perfect for:** All analysis types - Classification, Clustering, Association Rules, EDA

---

### 2. 🛒 **ecommerce_dataset.csv** (E-commerce Focus)
**Size:** 50 orders with detailed e-commerce metrics
- **Features:** Product ratings, review counts, return status
- **Shipping methods and customer segments**
- **Premium brands and electronics focus**

**Key Columns:**
- `OrderID`, `ProductRating`, `ReviewCount`, `IsReturned`
- `ShippingMethod`, `CustomerSegment`

**Perfect for:** Customer satisfaction analysis, return prediction, rating analysis

---

### 3. 🏪 **supermarket_dataset.csv** (Retail Store Focus)
**Size:** 50 transactions with store-specific details
- **Features:** Store locations, cashier IDs, aisle information
- **Member card usage and discount tracking**
- **Physical store operations data**

**Key Columns:**
- `InvoiceNo`, `StockCode`, `Aisle`, `Store_Location`
- `Cashier_ID`, `Member_Card`, `Discount_Applied`

**Perfect for:** Store operations analysis, inventory management, location-based insights

---

### 4. 📋 **comprehensive_retail_data.csv** (Sample Dataset)
**Size:** 50 transactions - Clean sample for testing
- **Simple structure for quick testing**
- **All major retail categories represented**

---

## 🎯 **Recommended Analysis Workflows**

### **For Decision Tree Classification:**
- **Dataset:** `large_retail_dataset.csv`
- **Target:** Create repurchase prediction (customers with >1 transaction)
- **Features:** Age, Gender, City, Category, TotalAmount, Season
- **Expected Accuracy:** 75-85%

### **For K-Means Clustering:**
- **Dataset:** `large_retail_dataset.csv`
- **Features:** Customer aggregated metrics (Total_Spend, Avg_Order_Value, Purchase_Frequency)
- **Optimal Clusters:** 4-6 customer segments
- **Expected Silhouette Score:** 0.4-0.7

### **For Association Rule Mining:**
- **Dataset:** `large_retail_dataset.csv` or `supermarket_dataset.csv`
- **Transaction ID:** `CustomerID`
- **Items:** `Category` or `ProductName`
- **Expected Rules:** 20-50 strong association rules

### **For Advanced EDA:**
- **Dataset:** `large_retail_dataset.csv`
- **Visualizations:** Sales trends, seasonal patterns, customer segments
- **Business Insights:** Revenue drivers, profit margins, customer behavior

---

## 🚀 **Quick Start Guide**

1. **Load any dataset in the web app**
2. **Start with EDA** to understand data patterns
3. **Apply preprocessing** with recommended settings:
   - Missing values: Median for numeric, Mode for categorical
   - Encoding: Label Encoder
   - Normalization: Yes
4. **Run all algorithms** to get comprehensive insights

---

## 📈 **Expected Business Insights**

### **Customer Segmentation (K-Means):**
- **Premium Customers:** High spend, low frequency
- **Regular Customers:** Medium spend, medium frequency  
- **Budget Customers:** Low spend, high frequency
- **VIP Customers:** High spend, high frequency

### **Product Associations (Apriori):**
- **Electronics + Accessories:** High confidence rules
- **Seasonal patterns:** Winter clothing, Summer sports
- **Cross-category:** Home & Kitchen items

### **Purchase Prediction (Decision Tree):**
- **Key factors:** Customer age, previous spend, category preference
- **Seasonal influence:** Holiday shopping patterns
- **Geographic patterns:** City-based preferences

---

## 💡 **Pro Tips for Analysis**

1. **Start with the large dataset** for comprehensive analysis
2. **Use customer aggregation** for better clustering results
3. **Filter by date ranges** for seasonal analysis
4. **Combine multiple datasets** for cross-validation
5. **Focus on business metrics** (Revenue, Profit) for actionable insights

---

**🎯 Ready to analyze? Upload any dataset and start exploring!**