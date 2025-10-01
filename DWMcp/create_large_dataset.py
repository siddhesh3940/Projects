import csv
import random
from datetime import datetime, timedelta

# Configuration
NUM_TRANSACTIONS = 10000
NUM_CUSTOMERS = 2000
NUM_PRODUCTS = 800

# Data pools
categories = {
    'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Headphones', 'Cameras', 'Gaming', 'Smart Home', 'Wearables'],
    'Clothing': ['Mens Wear', 'Womens Wear', 'Kids Wear', 'Shoes', 'Accessories', 'Sportswear', 'Formal Wear'],
    'Home & Garden': ['Furniture', 'Kitchen', 'Decor', 'Tools', 'Appliances', 'Bedding', 'Storage'],
    'Books & Media': ['Fiction', 'Non-Fiction', 'Educational', 'Movies', 'Music', 'Games', 'Magazines'],
    'Sports & Outdoors': ['Fitness', 'Outdoor Gear', 'Sports Equipment', 'Activewear', 'Camping', 'Water Sports'],
    'Health & Beauty': ['Skincare', 'Makeup', 'Health Supplements', 'Personal Care', 'Hair Care', 'Fragrances'],
    'Food & Beverages': ['Snacks', 'Beverages', 'Organic', 'Frozen', 'Dairy', 'Bakery', 'International'],
    'Automotive': ['Car Parts', 'Tools', 'Accessories', 'Maintenance', 'Electronics'],
    'Toys & Games': ['Educational Toys', 'Action Figures', 'Board Games', 'Outdoor Toys', 'Electronic Toys'],
    'Office Supplies': ['Stationery', 'Electronics', 'Furniture', 'Storage', 'Printing']
}

brands = ['Apple', 'Samsung', 'Nike', 'Adidas', 'Sony', 'LG', 'HP', 'Dell', 'Canon', 'Nikon', 
          'Zara', 'H&M', 'Uniqlo', 'IKEA', 'Amazon Basics', 'Generic Brand', 'Premium Co', 'Value Plus']

cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
          'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
          'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle']

payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet', 'Buy Now Pay Later']
channels = ['Online', 'In-Store', 'Mobile App']
segments = ['Premium', 'Regular', 'Budget', 'VIP']

def generate_price(category):
    """Generate realistic prices based on category"""
    price_ranges = {
        'Electronics': (50, 2500),
        'Clothing': (15, 400),
        'Home & Garden': (20, 1200),
        'Books & Media': (5, 150),
        'Sports & Outdoors': (25, 800),
        'Health & Beauty': (8, 200),
        'Food & Beverages': (2, 50),
        'Automotive': (15, 500),
        'Toys & Games': (10, 300),
        'Office Supplies': (5, 400)
    }
    min_price, max_price = price_ranges.get(category, (10, 100))
    return round(random.uniform(min_price, max_price), 2)

def generate_quantity(category):
    """Generate realistic quantities based on category"""
    if category in ['Electronics', 'Home & Garden', 'Automotive']:
        return random.choices([1, 2], weights=[0.85, 0.15])[0]
    elif category in ['Food & Beverages', 'Health & Beauty']:
        return random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]
    else:
        return random.choices([1, 2, 3], weights=[0.7, 0.25, 0.05])[0]

# Generate dataset
print("Generating large retail dataset...")

with open('large_retail_dataset.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Header
    header = ['TransactionID', 'CustomerID', 'ProductID', 'ProductName', 'Category', 'Subcategory', 
              'Brand', 'Quantity', 'UnitPrice', 'TotalAmount', 'Discount', 'TransactionDate', 
              'PaymentMethod', 'Channel', 'CustomerAge', 'CustomerGender', 'CustomerCity', 
              'CustomerSegment', 'Year', 'Month', 'DayOfWeek', 'Season', 'Revenue', 'Profit']
    writer.writerow(header)
    
    # Generate transactions
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    for i in range(1, NUM_TRANSACTIONS + 1):
        # Transaction details
        transaction_id = f'T{i:06d}'
        customer_id = f'C{random.randint(1, NUM_CUSTOMERS):05d}'
        product_id = f'P{random.randint(1, NUM_PRODUCTS):05d}'
        
        # Product details
        category = random.choice(list(categories.keys()))
        subcategory = random.choice(categories[category])
        brand = random.choice(brands)
        product_name = f'{subcategory} {brand} Model {random.randint(1, 50)}'
        
        # Pricing and quantity
        unit_price = generate_price(category)
        quantity = generate_quantity(category)
        discount_pct = random.choices([0, 5, 10, 15, 20, 25, 30], 
                                    weights=[0.6, 0.15, 0.1, 0.08, 0.04, 0.02, 0.01])[0]
        discount_amount = unit_price * (discount_pct / 100)
        final_price = unit_price - discount_amount
        total_amount = final_price * quantity
        
        # Date and time
        transaction_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        year = transaction_date.year
        month = transaction_date.month
        day_of_week = transaction_date.strftime('%A')
        
        # Season
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        season = season_map[month]
        
        # Customer details
        age = random.randint(18, 75)
        gender = random.choice(['Male', 'Female'])
        city = random.choice(cities)
        segment = random.choice(segments)
        
        # Transaction details
        payment_method = random.choice(payment_methods)
        channel = random.choice(channels)
        
        # Business metrics
        cost_ratio = random.uniform(0.4, 0.7)  # Cost is 40-70% of selling price
        cost = unit_price * cost_ratio
        profit = (final_price - cost) * quantity
        revenue = total_amount
        
        # Write row
        row = [transaction_id, customer_id, product_id, product_name, category, subcategory,
               brand, quantity, round(final_price, 2), round(total_amount, 2), discount_pct,
               transaction_date.strftime('%Y-%m-%d'), payment_method, channel, age, gender,
               city, segment, year, month, day_of_week, season, round(revenue, 2), round(profit, 2)]
        
        writer.writerow(row)
        
        if i % 1000 == 0:
            print(f"Generated {i:,} transactions...")

print(f"✅ Successfully generated {NUM_TRANSACTIONS:,} transactions!")
print(f"📊 Dataset saved as 'large_retail_dataset.csv'")
print(f"👥 {NUM_CUSTOMERS:,} unique customers")
print(f"🛍️ {NUM_PRODUCTS:,} unique products")
print(f"📅 Date range: 2022-01-01 to 2024-03-31")
print(f"🏪 {len(categories)} categories with {sum(len(v) for v in categories.values())} subcategories")