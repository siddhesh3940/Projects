import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, precision_score, recall_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Retail Analytics Web App", layout="wide")

def preprocess_data(df, missing_strategy='median', encoding_type='label', normalize=True):
    """Comprehensive preprocessing with multiple options"""
    df_processed = df.copy()
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    st.subheader("🔧 Preprocessing Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        missing_strategy = st.selectbox("Missing Value Strategy", 
                                      ['drop', 'mean', 'median', 'mode'])
    with col2:
        encoding_type = st.selectbox("Categorical Encoding", 
                                   ['label', 'onehot'])
    with col3:
        normalize = st.checkbox("Normalize Features", value=True)
    
    # Handle missing values
    if missing_strategy == 'drop':
        df_processed = df_processed.dropna()
    else:
        for col in numeric_cols:
            if missing_strategy == 'mean':
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
            elif missing_strategy == 'median':
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        for col in categorical_cols:
            df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown', inplace=True)
    
    # Handle outliers using IQR method
    for col in numeric_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
    
    # Encode categorical variables
    if encoding_type == 'label':
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
        st.session_state['label_encoders'] = le_dict
    
    # Normalize numerical features
    if normalize and len(numeric_cols) > 0:
        scaler = MinMaxScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
        st.session_state['scaler'] = scaler
    
    return df_processed

def show_summary_stats(df):
    """Display summary statistics"""
    st.subheader("📊 Dataset Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())
    
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes)
        st.write("**Numeric Summary:**")
        st.write(df.describe())

def perform_eda(df):
    """Comprehensive Exploratory Data Analysis"""
    st.header("🔍 Exploratory Data Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Sales Distribution Histogram
    if 'Price' in df.columns or 'Quantity' in df.columns:
        st.subheader("📊 Sales Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Price' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                df['Price'].hist(bins=30, ax=ax, color='skyblue', alpha=0.7)
                ax.set_title('Price Distribution', fontsize=14)
                ax.set_xlabel('Price')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
        
        with col2:
            if 'Quantity' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                df['Quantity'].hist(bins=20, ax=ax, color='lightgreen', alpha=0.7)
                ax.set_title('Quantity Distribution', fontsize=14)
                ax.set_xlabel('Quantity')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
    
    # Correlation Heatmap
    if len(numeric_cols) > 1:
        st.subheader("🔥 Correlation Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation Heatmap', fontsize=16)
        st.pyplot(fig)
    
    # Top N Products by Sales
    if 'Product' in df.columns and 'Price' in df.columns:
        st.subheader("🏆 Top Products by Sales")
        n_products = st.slider("Number of top products", 5, 20, 10)
        
        if 'Quantity' in df.columns:
            df['Total_Sales'] = df['Price'] * df['Quantity']
            top_products = df.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False).head(n_products)
        else:
            top_products = df.groupby('Product')['Price'].sum().sort_values(ascending=False).head(n_products)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_products.plot(kind='bar', ax=ax, color='coral')
            ax.set_title(f'Top {n_products} Products by Sales', fontsize=14)
            ax.set_xlabel('Products')
            ax.set_ylabel('Sales Value')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            # Plotly interactive chart
            fig = px.pie(values=top_products.values, names=top_products.index, 
                        title=f'Top {n_products} Products Sales Distribution')
            st.plotly_chart(fig)
    
    # Category Analysis
    if 'Category' in df.columns:
        st.subheader("📈 Category Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = df['Category'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            category_counts.plot(kind='bar', ax=ax, color='lightblue')
            ax.set_title('Products by Category', fontsize=14)
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            if 'Price' in df.columns:
                avg_price_by_category = df.groupby('Category')['Price'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 6))
                avg_price_by_category.plot(kind='bar', ax=ax, color='gold')
                ax.set_title('Average Price by Category', fontsize=14)
                ax.set_xlabel('Category')
                ax.set_ylabel('Average Price')
                plt.xticks(rotation=45)
                st.pyplot(fig)
    
    # Additional visualizations for any numeric columns
    if len(numeric_cols) > 0:
        st.subheader("📊 Distribution Analysis")
        selected_col = st.selectbox("Select column for detailed analysis", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            df[selected_col].hist(bins=30, ax=ax, alpha=0.7, color='purple')
            ax.axvline(df[selected_col].mean(), color='red', linestyle='--', label=f'Mean: {df[selected_col].mean():.2f}')
            ax.axvline(df[selected_col].median(), color='green', linestyle='--', label=f'Median: {df[selected_col].median():.2f}')
            ax.set_title(f'Distribution of {selected_col}')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            df.boxplot(column=selected_col, ax=ax)
            ax.set_title(f'Box Plot of {selected_col}')
            st.pyplot(fig)

def decision_tree_classification(df):
    """Enhanced Decision Tree Classification"""
    st.header("🌳 Decision Tree Classification")
    
    # Create binary target for repurchase prediction
    st.subheader("🎯 Target Definition")
    
    if 'CustomerID' in df.columns:
        # Create repurchase target
        customer_purchases = df.groupby('CustomerID').size()
        repurchase_customers = customer_purchases[customer_purchases > 1].index
        df['Will_Repurchase'] = df['CustomerID'].isin(repurchase_customers).astype(int)
        st.info("Created 'Will_Repurchase' target based on customer purchase frequency")
    
    # Select target and features
    target_col = st.selectbox("Select target column", df.columns)
    feature_cols = st.multiselect("Select feature columns", 
                                 [col for col in df.columns if col != target_col])
    
    if not feature_cols:
        st.warning("Please select feature columns")
        return
    
    try:
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Encode categorical variables
        le_dict = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
        
        # Encode target if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Train-test split
        test_size = st.slider("Test size ratio", 0.1, 0.5, 0.3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Model parameters
        max_depth = st.slider("Max depth", 3, 10, 5)
        min_samples_split = st.slider("Min samples split", 2, 20, 2)
        
        # Train model
        dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        dt.fit(X_train, y_train)
        
        # Predictions
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Performance")
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("Precision", f"{precision:.3f}")
            st.metric("Recall", f"{recall:.3f}")
            
            st.write("**Classification Report:**")
            st.text(classification_report(y_test, y_pred))
            
            # Feature importance
            st.write("**Feature Importance:**")
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': dt.feature_importances_
            }).sort_values('Importance', ascending=False)
            st.write(importance_df)
        
        with col2:
            st.subheader("🌳 Decision Tree Visualization")
            fig, ax = plt.subplots(figsize=(15, 10))
            plot_tree(dt, feature_names=feature_cols, max_depth=3, filled=True, 
                     rounded=True, fontsize=10, ax=ax)
            ax.set_title('Decision Tree Structure', fontsize=16)
            st.pyplot(fig)
            
            # Feature importance plot
            fig, ax = plt.subplots(figsize=(8, 6))
            importance_df.plot(x='Feature', y='Importance', kind='bar', ax=ax, color='lightcoral')
            ax.set_title('Feature Importance')
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error in classification: {str(e)}")

def kmeans_clustering(df):
    """Enhanced K-Means Clustering"""
    st.header("🎯 K-Means Clustering")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for clustering")
        return
    
    # Feature engineering for customer segmentation
    if 'CustomerID' in df.columns and 'Price' in df.columns:
        st.subheader("🔧 Customer Segmentation Features")
        customer_features = df.groupby('CustomerID').agg({
            'Price': ['sum', 'mean', 'count'],
            'Quantity': 'sum' if 'Quantity' in df.columns else 'count'
        }).round(2)
        
        customer_features.columns = ['Total_Spend', 'Avg_Order_Value', 'Purchase_Frequency', 'Total_Quantity']
        customer_features = customer_features.reset_index()
        
        st.write("**Generated Customer Features:**")
        st.write(customer_features.head())
        
        # Use customer features for clustering
        feature_cols = ['Total_Spend', 'Avg_Order_Value', 'Purchase_Frequency']
        X = customer_features[feature_cols].copy()
    else:
        # Select features manually
        feature_cols = st.multiselect("Select features for clustering", numeric_cols)
        if len(feature_cols) < 2:
            st.warning("Please select at least 2 features")
            return
        X = df[feature_cols].copy()
    
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Elbow method
            st.subheader("📈 Elbow Method")
            max_k = min(11, len(X))
            inertias = []
            silhouette_scores = []
            k_range = range(2, max_k)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, clusters))
            
            # Elbow curve
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia (WCSS)')
            ax.set_title('Elbow Method for Optimal k')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Silhouette scores
            st.subheader("📊 Silhouette Analysis")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Silhouette Score')
            ax.set_title('Silhouette Score vs Number of Clusters')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Select optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        st.info(f"Suggested optimal k based on silhouette score: {optimal_k}")
        
        k = st.slider("Select number of clusters", 2, max_k-1, optimal_k)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        sil_score = silhouette_score(X_scaled, clusters)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Clustering Results")
            st.metric("Silhouette Score", f"{sil_score:.3f}")
            st.metric("Number of Clusters", k)
            st.metric("Inertia (WCSS)", f"{kmeans.inertia_:.2f}")
            
            # Cluster sizes
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            st.write("**Cluster Sizes:**")
            for i, count in cluster_counts.items():
                st.write(f"Cluster {i}: {count} points ({count/len(clusters)*100:.1f}%)")
        
        with col2:
            # Scatter plot visualization
            st.subheader("🔍 Cluster Visualization")
            if len(feature_cols) >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, 
                                   cmap='viridis', alpha=0.7, s=50)
                
                # Plot centroids
                centroids = kmeans.cluster_centers_
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', 
                          s=300, linewidths=3, label='Centroids')
                
                ax.set_xlabel(f'{feature_cols[0]} (Standardized)')
                ax.set_ylabel(f'{feature_cols[1]} (Standardized)')
                ax.set_title('K-Means Clustering Results')
                ax.legend()
                plt.colorbar(scatter, label='Cluster')
                st.pyplot(fig)
        
        # Cluster analysis
        st.subheader("📋 Cluster Analysis")
        X_with_clusters = X.copy()
        X_with_clusters['Cluster'] = clusters
        
        cluster_summary = X_with_clusters.groupby('Cluster').agg(['mean', 'std']).round(2)
        st.write("**Cluster Statistics:**")
        st.write(cluster_summary)
        
        # 3D visualization if more than 2 features
        if len(feature_cols) >= 3:
            st.subheader("🌐 3D Cluster Visualization")
            fig = px.scatter_3d(x=X_scaled[:, 0], y=X_scaled[:, 1], z=X_scaled[:, 2],
                              color=clusters, title='3D Cluster Visualization',
                              labels={'x': feature_cols[0], 'y': feature_cols[1], 'z': feature_cols[2]})
            st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error in clustering: {str(e)}")

def apriori_analysis(df):
    """Enhanced Apriori Association Rule Mining with Network Visualization"""
    st.header("🔗 Association Rule Mining (Apriori)")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        st.warning("Need categorical columns for association rule mining")
        return
    
    # Select transaction and item columns
    transaction_col = st.selectbox("Select transaction/customer ID column", df.columns)
    item_col = st.selectbox("Select item/product column", categorical_cols)
    
    if transaction_col == item_col:
        st.warning("Transaction and item columns should be different")
        return
    
    try:
        # Create market basket format
        st.subheader("🛒 Market Basket Analysis")
        
        # Show transaction summary
        transaction_summary = df.groupby(transaction_col)[item_col].count().describe()
        st.write("**Transaction Summary:**")
        st.write(transaction_summary)
        
        # Create transaction format
        transactions = df.groupby(transaction_col)[item_col].apply(list).tolist()
        
        # Encode transactions
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1)
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5)
        with col3:
            min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.2)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            st.warning("No frequent itemsets found. Try lowering the minimum support.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Frequent Itemsets")
            frequent_itemsets_sorted = frequent_itemsets.sort_values('support', ascending=False)
            st.write(frequent_itemsets_sorted.head(15))
            
            # Frequent itemsets visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            itemset_lengths = frequent_itemsets['itemsets'].apply(len)
            itemset_counts = itemset_lengths.value_counts().sort_index()
            itemset_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Frequent Itemsets by Length')
            ax.set_xlabel('Itemset Length')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        
        with col2:
            # Generate association rules
            if len(frequent_itemsets) > 1:
                rules = association_rules(frequent_itemsets, metric="confidence", 
                                        min_threshold=min_confidence)
                
                # Filter by lift
                rules = rules[rules['lift'] >= min_lift]
                
                if not rules.empty:
                    st.subheader("📋 Association Rules")
                    rules_display = rules[['antecedents', 'consequents', 'support', 
                                         'confidence', 'lift']].sort_values('lift', ascending=False)
                    st.write(rules_display.head(10))
                    
                    # Rules metrics visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(rules['support'], rules['confidence'], 
                                       c=rules['lift'], s=rules['lift']*20, 
                                       cmap='viridis', alpha=0.7)
                    ax.set_xlabel('Support')
                    ax.set_ylabel('Confidence')
                    ax.set_title('Association Rules: Support vs Confidence (Size = Lift)')
                    plt.colorbar(scatter, label='Lift')
                    st.pyplot(fig)
                else:
                    st.warning("No association rules found with current thresholds.")
        
        # Network Graph Visualization
        if 'rules' in locals() and not rules.empty:
            st.subheader("🕸️ Association Rules Network Graph")
            
            # Create network graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            for idx, rule in rules.head(20).iterrows():  # Limit to top 20 rules for clarity
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                
                G.add_node(antecedents, node_type='antecedent')
                G.add_node(consequents, node_type='consequent')
                G.add_edge(antecedents, consequents, 
                          weight=rule['confidence'], 
                          lift=rule['lift'],
                          support=rule['support'])
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(15, 10))
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes
            antecedent_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'antecedent']
            consequent_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'consequent']
            
            nx.draw_networkx_nodes(G, pos, nodelist=antecedent_nodes, 
                                 node_color='lightblue', node_size=1000, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=consequent_nodes, 
                                 node_color='lightcoral', node_size=1000, ax=ax)
            
            # Draw edges with varying thickness based on confidence
            edges = G.edges(data=True)
            edge_weights = [edge[2]['weight'] * 5 for edge in edges]  # Scale for visibility
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, 
                                 edge_color='gray', arrows=True, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
            
            ax.set_title('Association Rules Network\n(Blue: Antecedents, Red: Consequents, Edge thickness: Confidence)', 
                        fontsize=14)
            ax.axis('off')
            st.pyplot(fig)
            
            # Top rules summary
            st.subheader("🏆 Top Association Rules Summary")
            top_rules = rules.nlargest(5, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            
            for idx, rule in top_rules.iterrows():
                antecedents_str = ', '.join(list(rule['antecedents']))
                consequents_str = ', '.join(list(rule['consequents']))
                
                st.write(f"**Rule {idx+1}:** {antecedents_str} → {consequents_str}")
                st.write(f"- Support: {rule['support']:.3f} | Confidence: {rule['confidence']:.3f} | Lift: {rule['lift']:.3f}")
                st.write("---")
        
    except Exception as e:
        st.error(f"Error in association rule mining: {str(e)}")

def main():
    st.title("🛍️ Retail Analytics Web App")
    st.markdown("**Data Warehousing & Mining Project**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose Analysis", 
                               ["Upload & Preprocessing", "EDA", "Classification", 
                                "Clustering", "Association Rules"])
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"File uploaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            if page == "Upload & Preprocessing":
                st.header("📁 Dataset Upload & Preprocessing")
                
                # Show raw data
                st.subheader("📋 Raw Data Preview")
                st.write(df.head())
                
                # Show summary stats
                show_summary_stats(df)
                
                # Preprocessing
                if st.button("🔧 Apply Preprocessing"):
                    df_processed = preprocess_data(df.copy())
                    st.success("✅ Preprocessing completed!")
                    
                    # Show before/after comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Before Preprocessing")
                        st.write(f"Shape: {df.shape}")
                        st.write(f"Missing values: {df.isnull().sum().sum()}")
                    
                    with col2:
                        st.subheader("After Preprocessing")
                        st.write(f"Shape: {df_processed.shape}")
                        st.write(f"Missing values: {df_processed.isnull().sum().sum()}")
                    
                    st.subheader("🔍 Processed Data Preview")
                    st.write(df_processed.head())
                    
                    # Update session state
                    st.session_state['processed_df'] = df_processed
            
            # Use processed data if available
            if 'processed_df' in st.session_state:
                df = st.session_state['processed_df']
            
            if page == "EDA":
                perform_eda(df)
            elif page == "Classification":
                decision_tree_classification(df)
            elif page == "Clustering":
                kmeans_clustering(df)
            elif page == "Association Rules":
                apriori_analysis(df)
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Sample data info
        st.markdown("""
        ### Expected Data Format:
        - **CSV file** with headers
        - **Numeric columns** for clustering and classification
        - **Categorical columns** for association rule mining
        - **Transaction/Customer ID** column for market basket analysis
        
        ### Features:
        - 📊 **Data Preprocessing**: Handle missing values, outliers, encoding
        - 🔍 **EDA**: Distribution plots, correlations, categorical analysis
        - 🌳 **Decision Tree**: Classification with accuracy metrics
        - 🎯 **K-Means**: Customer/product clustering with elbow method
        - 🔗 **Apriori**: Frequent itemsets and association rules
        """)

if __name__ == "__main__":
    main()