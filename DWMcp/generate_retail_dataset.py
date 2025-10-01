import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_large_retail_dataset(n_customers=5000, n_products=500, n_transactions=50000):
    """Generate a comprehensive retail dataset"""
    
    # Product categories and subcategories
    categories = {
        'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Headphones', 'Cameras', 'Gaming'],
        'Clothing': ['Mens Wear', 'Womens Wear', 'Kids Wear', 'Shoes', 'Accessories'],
        'Home & Garden': ['Furniture', 'Kitchen', 'Decor', 'Tools', 'Appliances'],
        'Books & Media': ['Fiction', 'Non-Fiction', 'Educational', 'Movies', 'Music'],
        'Sports & Outdoors': ['Fitness', 'Outdoor Gear', 'Sports Equipment', 'Activewear'],
        'Health & Beauty': ['Skincare', 'Makeup', 'Health Supplements', 'Personal Care'],
        'Food & Beverages': ['Snacks', 'Beverages', 'Organic', 'Frozen', 'Dairy']
    }
    
    # Generate products
    products = []
    product_id = 1
    
    for category, subcategories in categories.items():
        for subcategory in subcategories:
            for i in range(n_products // len([item for sublist in categories.values() for item in sublist])):
                base_price = np.random.uniform(10, 500)
                if category == 'Electronics':
                    base_price = np.random.uniform(50, 2000)
                elif category == 'Clothing':
                    base_price = np.random.uniform(15, 300)
                
                products.append({
                    'ProductID': f'P{product_id:05d}',
                    'ProductName': f'{subcategory} Item {i+1}',
                    'Category': category,
                    'Subcategory': subcategory,
                    'BasePrice': round(base_price, 2),
                    'Cost': round(base_price * 0.6, 2),
                    'Brand': f'Brand_{random.choice(["A", "B", "C", "D", "E"])}'
                })
                product_id += 1
    
    products_df = pd.DataFrame(products[:n_products])
    
    # Generate customers
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
              'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville']
    
    customers = []
    for i in range(n_customers):
        age = np.random.normal(40, 15)
        age = max(18, min(80, int(age)))
        
        customers.append({
            'CustomerID': f'C{i+1:05d}',
            'Age': age,
            'Gender': random.choice(['Male', 'Female']),
            'City': random.choice(cities),
            'State': random.choice(['NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']),
            'CustomerSegment': random.choice(['Premium', 'Regular', 'Budget']),
            'RegistrationDate': datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1095))
        })
    
    customers_df = pd.DataFrame(customers)
    
    # Generate transactions
    transactions = []
    transaction_id = 1
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    for _ in range(n_transactions):
        customer = customers_df.sample(1).iloc[0]
        product = products_df.sample(1).iloc[0]
        
        # Transaction date with seasonal patterns
        transaction_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        
        # Quantity based on product category
        if product['Category'] == 'Electronics':
            quantity = random.choices([1, 2], weights=[0.9, 0.1])[0]
        elif product['Category'] == 'Food & Beverages':
            quantity = random.choices([1, 2, 3, 4, 5], weights=[0.3, 0.3, 0.2, 0.1, 0.1])[0]
        else:
            quantity = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
        
        # Price with discounts and variations
        discount = random.uniform(0, 0.3) if random.random() < 0.2 else 0
        unit_price = product['BasePrice'] * (1 - discount)
        
        # Payment method
        payment_method = random.choices(
            ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet'],
            weights=[0.4, 0.3, 0.2, 0.1]
        )[0]
        
        # Channel
        channel = random.choices(['Online', 'In-Store'], weights=[0.6, 0.4])[0]
        
        transactions.append({
            'TransactionID': f'T{transaction_id:06d}',
            'CustomerID': customer['CustomerID'],
            'ProductID': product['ProductID'],
            'ProductName': product['ProductName'],
            'Category': product['Category'],
            'Subcategory': product['Subcategory'],
            'Brand': product['Brand'],
            'Quantity': quantity,
            'UnitPrice': round(unit_price, 2),
            'TotalAmount': round(unit_price * quantity, 2),
            'Discount': round(discount * 100, 1),
            'TransactionDate': transaction_date,
            'PaymentMethod': payment_method,
            'Channel': channel,
            'CustomerAge': customer['Age'],
            'CustomerGender': customer['Gender'],
            'CustomerCity': customer['City'],
            'CustomerSegment': customer['CustomerSegment']
        })
        transaction_id += 1
    
    transactions_df = pd.DataFrame(transactions)
    
    # Add derived features
    transactions_df['Year'] = transactions_df['TransactionDate'].dt.year
    transactions_df['Month'] = transactions_df['TransactionDate'].dt.month
    transactions_df['DayOfWeek'] = transactions_df['TransactionDate'].dt.day_name()
    transactions_df['Season'] = transactions_df['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    return transactions_df, customers_df, products_df

if __name__ == "__main__":
    print("Generating comprehensive retail dataset...")
    
    # Generate dataset
    transactions, customers, products = generate_large_retail_dataset(
        n_customers=5000, 
        n_products=500, 
        n_transactions=50000
    )
    
    # Save datasets
    transactions.to_csv('comprehensive_retail_data.csv', index=False)
    customers.to_csv('customers_data.csv', index=False)
    products.to_csv('products_data.csv', index=False)
    
    print(f"✅ Generated datasets:")
    print(f"📊 Transactions: {len(transactions):,} records")
    print(f"👥 Customers: {len(customers):,} records") 
    print(f"🛍️ Products: {len(products):,} records")
    print(f"📅 Date range: {transactions['TransactionDate'].min()} to {transactions['TransactionDate'].max()}")
    print(f"💰 Total sales: ${transactions['TotalAmount'].sum():,.2f}")
    
    # Dataset summary
    print("\n📈 Dataset Summary:")
    print(f"Categories: {transactions['Category'].nunique()}")
    print(f"Brands: {transactions['Brand'].nunique()}")
    print(f"Cities: {transactions['CustomerCity'].nunique()}")
    print(f"Avg transaction value: ${transactions['TotalAmount'].mean():.2f}")
    print(f"Top category: {transactions['Category'].value_counts().index[0]}")