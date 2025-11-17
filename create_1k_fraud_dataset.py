"""
Complete Singapore E-Commerce Fraud Dataset Generator
Creates realistic data based on actual fraud patterns
Perfect for your IEEE-CIS fraud detection project

Author: Cheryl (cherollyy) - Group 4
Date: 2025-10-22
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_singapore_ecommerce_dataset(num_records=1000):
    """
    Create comprehensive Singapore e-commerce fraud dataset
    Includes all features needed for fraud detection model
    """
    
    print("="*70)
    print("🏗️  CREATING SINGAPORE E-COMMERCE FRAUD DATASET")
    print("="*70)
    
    np.random.seed(42)
    random.seed(42)
    
    # === CONFIGURATION ===
    platforms = ['Shopee', 'Lazada', 'Qoo10', 'Carousell', 'eBay']
    categories = [
        'Luxury Bags', 'Designer Handbag', 'Branded Watch', 
        'iPhone 15', 'Samsung Galaxy', 'Designer Clothes',
        'Electronics', 'Fashion', 'Beauty', 'Home & Living'
    ]
    
    locations = ['Singapore', 'China', 'Malaysia', 'Hong Kong', 'Thailand', 'Vietnam', 'Overseas']
    conditions = ['Brand New', 'Like New', 'Pre-Owned', 'Refurbished']
    
    data = []
    
    print(f"\n📊 Generating {num_records} transactions...")
    
    for i in range(num_records):
        # Basic info
        platform = random.choice(platforms)
        category = random.choice(categories)
        
        # Determine if this is a fraud case (30% fraud rate - realistic for e-commerce)
        is_fraud = np.random.random() < 0.30
        
        # === PRICE GENERATION ===
        # Base price depends on category
        if 'Luxury' in category or 'Designer' in category or 'Branded' in category:
            base_price = np.random.uniform(300, 3000)
        elif 'iPhone' in category or 'Samsung' in category:
            base_price = np.random.uniform(600, 1800)
        elif category == 'Electronics':
            base_price = np.random.uniform(50, 800)
        else:
            base_price = np.random.uniform(15, 300)
        
        # === FRAUD PATTERNS ===
        if is_fraud:
            # Fraudulent listings have specific patterns
            
            # 1. Suspiciously low prices (20-40% of market value)
            price = base_price * np.random.uniform(0.15, 0.45)
            
            # 2. Extreme discounts
            original_price = base_price * np.random.uniform(1.5, 3.0)
            discount = int(((original_price - price) / original_price) * 100)
            discount = min(discount, 95)  # Cap at 95%
            
            # 3. Low seller ratings
            seller_rating = np.random.uniform(1.0, 3.8)
            
            # 4. Few reviews
            reviews_count = np.random.randint(0, 20)
            
            # 5. New seller account
            seller_age_days = np.random.randint(1, 60)
            
            # 6. Low sold count (or suspiciously high for new account)
            if seller_age_days < 30:
                sold_count = np.random.randint(0, 10)  # New seller, few sales
            else:
                # Some fraud: fake high sales
                sold_count = np.random.choice([
                    np.random.randint(0, 15),  # 70% low sales
                    np.random.randint(500, 5000)  # 30% fake high sales
                ], p=[0.7, 0.3])
            
            # 7. Risky locations
            item_location = random.choice(['China', 'Hong Kong', 'Overseas', 'China'])  # More likely China
            
            # 8. Not official store
            is_official_store = False
            is_mall = False
            is_top_rated = False
            
            # 9. No return policy
            has_return_policy = np.random.choice([True, False], p=[0.2, 0.8])
            
            # 10. Condition often "Brand New" for counterfeits
            condition = random.choice(['Brand New', 'Brand New', 'Like New'])
            
            # 11. Low product rating
            product_rating = np.random.uniform(2.0, 4.2)
            
            # 12. Free shipping (to attract buyers)
            free_shipping = np.random.choice([True, False], p=[0.7, 0.3])
            
            # 13. Low watchers/engagement
            watchers_count = np.random.randint(0, 5)
            
        else:
            # Legitimate listings
            
            # 1. Normal prices (70-120% of base)
            price = base_price * np.random.uniform(0.75, 1.25)
            
            # 2. Reasonable discounts
            if np.random.random() < 0.4:  # 40% have discounts
                discount = np.random.randint(5, 50)
                original_price = price / (1 - discount/100)
            else:
                discount = 0
                original_price = price
            
            # 3. Good seller ratings
            seller_rating = np.random.uniform(4.0, 5.0)
            
            # 4. More reviews
            reviews_count = np.random.randint(20, 1500)
            
            # 5. Established seller
            seller_age_days = np.random.randint(180, 3000)
            
            # 6. Realistic sold count
            sold_count = np.random.randint(50, 5000)
            
            # 7. Local or reputable locations
            item_location = random.choice([
                'Singapore', 'Singapore', 'Singapore',  # Higher probability
                'Malaysia', 'Japan', 'USA', 'UK'
            ])
            
            # 8. Possibly official store
            is_official_store = np.random.choice([True, False], p=[0.25, 0.75])
            is_mall = is_official_store and platform in ['Shopee', 'Lazada']
            is_top_rated = np.random.choice([True, False], p=[0.40, 0.60])
            
            # 9. Has return policy
            has_return_policy = np.random.choice([True, False], p=[0.85, 0.15])
            
            # 10. Various conditions
            condition = random.choice(conditions)
            
            # 11. Good product rating
            product_rating = np.random.uniform(4.0, 5.0)
            
            # 12. Shipping varies
            free_shipping = np.random.choice([True, False], p=[0.5, 0.5])
            
            # 13. Good engagement
            watchers_count = np.random.randint(5, 150)
        
        # === CREATE RECORD ===
        record = {
            # Identifiers
            'transaction_id': f'TXN_{i+1:06d}',
            'product_id': f'PROD_{random.randint(10000, 99999)}',
            
            # Platform & Category
            'platform': platform,
            'product_category': category,
            'product_name': f"{category} - {random.choice(['Premium', 'Quality', 'Authentic', 'Designer', 'Popular'])} Item #{i+1}",
            
            # Pricing
            'product_price': round(price, 2),
            'original_price': round(original_price, 2),
            'discount_percentage': discount,
            
            # Seller Information
            'seller_name': f"Seller_{random.randint(1000, 9999)}",
            'seller_rating': round(seller_rating, 1),
            'seller_age_days': seller_age_days,
            'seller_response_rate': np.random.randint(50, 100) if not is_fraud else np.random.randint(30, 90),
            
            # Product Information
            'product_rating': round(product_rating, 1),
            'reviews_count': reviews_count,
            'ratings_count': reviews_count + np.random.randint(-5, 20),
            'sold_count': sold_count,
            'watchers_count': watchers_count,
            
            # Listing Details
            'condition': condition,
            'item_location': item_location,
            'shipping_location': item_location,
            'listing_type': random.choice(['Buy It Now', 'Auction']) if platform == 'eBay' else 'Buy It Now',
            
            # Badges & Verification
            'is_official_store': is_official_store,
            'is_mall': is_mall,
            'is_top_rated_seller': is_top_rated,
            'is_verified': is_official_store or is_top_rated,
            
            # Policies
            'has_return_policy': has_return_policy,
            'free_shipping': free_shipping,
            
            # Timestamp
            'listing_date': (datetime.now() - timedelta(days=seller_age_days)).isoformat(),
            'scrape_timestamp': datetime.now().isoformat(),
            
            # === FRAUD LABEL (TARGET VARIABLE) ===
            'is_fraud': 1 if is_fraud else 0
        }
        
        data.append(record)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_records} records...")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # === ADD DERIVED FEATURES (for model training) ===
    print(f"\n🔧 Creating derived features...")
    
    # Price-based features
    df['price_deviation'] = (df['product_price'] - df['original_price']) / df['original_price']
    df['is_too_cheap'] = df['product_price'] < df.groupby('product_category')['product_price'].transform(lambda x: x.quantile(0.15))
    df['is_too_expensive'] = df['product_price'] > df.groupby('product_category')['product_price'].transform(lambda x: x.quantile(0.85))
    
    # Seller credibility
    df['seller_credibility_score'] = (
        (df['seller_rating'] / 5.0) * 0.3 +
        (df['seller_response_rate'] / 100) * 0.2 +
        (np.minimum(df['seller_age_days'] / 365, 5) / 5) * 0.3 +
        df['is_top_rated_seller'].astype(int) * 0.2
    )
    
    # Product credibility
    df['product_credibility_score'] = (
        (df['product_rating'] / 5.0) * 0.4 +
        (np.minimum(df['reviews_count'] / 100, 10) / 10) * 0.3 +
        (np.minimum(df['sold_count'] / 1000, 10) / 10) * 0.3
    )
    
    # Risk flags
    df['high_discount_flag'] = df['discount_percentage'] > 70
    df['low_seller_rating_flag'] = df['seller_rating'] < 4.0
    df['new_seller_flag'] = df['seller_age_days'] < 90
    df['low_reviews_flag'] = df['reviews_count'] < 10
    df['risky_location_flag'] = df['item_location'].isin(['China', 'Hong Kong', 'Overseas'])
    
    # Composite fraud score (unsupervised indicator)
    df['fraud_risk_score'] = (
        df['high_discount_flag'].astype(int) * 3 +
        df['is_too_cheap'].astype(int) * 3 +
        df['low_seller_rating_flag'].astype(int) * 2 +
        df['new_seller_flag'].astype(int) * 2 +
        df['low_reviews_flag'].astype(int) * 1 +
        df['risky_location_flag'].astype(int) * 2 +
        (~df['has_return_policy']).astype(int) * 1 +
        (~df['is_verified']).astype(int) * 2 -
        df['is_official_store'].astype(int) * 3  # Negative = less risk
    )
    
    df['fraud_risk_score'] = df['fraud_risk_score'].clip(lower=0)
    
    # Save dataset
    filename = 'singapore_ecommerce_fraud_dataset.csv'
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Dataset created successfully!")
    print(f"💾 Saved to: {filename}")
    
    # === STATISTICS ===
    print(f"\n" + "="*70)
    print("📊 DATASET STATISTICS")
    print("="*70)
    
    print(f"\n📈 Overview:")
    print(f"  Total records: {len(df):,}")
    print(f"  Fraud cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"  Legitimate cases: {(df['is_fraud']==0).sum():,} ({(df['is_fraud']==0).mean()*100:.1f}%)")
    
    print(f"\n🛍️  Platforms:")
    print(df['platform'].value_counts().to_string())
    
    print(f"\n📦 Categories:")
    print(df['product_category'].value_counts().to_string())
    
    print(f"\n💰 Price Statistics:")
    print(f"  Mean: ${df['product_price'].mean():.2f}")
    print(f"  Median: ${df['product_price'].median():.2f}")
    print(f"  Range: ${df['product_price'].min():.2f} - ${df['product_price'].max():.2f}")
    
    print(f"\n⚖️  Fraud vs Legitimate Comparison:")
    comparison = df.groupby('is_fraud').agg({
        'product_price': 'mean',
        'discount_percentage': 'mean',
        'seller_rating': 'mean',
        'reviews_count': 'mean',
        'seller_age_days': 'mean'
    }).round(2)
    comparison.index = ['Legitimate', 'Fraud']
    print(comparison.to_string())
    
    print(f"\n🚨 Fraud Indicators Distribution:")
    print(f"  High discounts (>70%): {df['high_discount_flag'].sum():,}")
    print(f"  Low seller ratings (<4.0): {df['low_seller_rating_flag'].sum():,}")
    print(f"  New sellers (<90 days): {df['new_seller_flag'].sum():,}")
    print(f"  Low reviews (<10): {df['low_reviews_flag'].sum():,}")
    print(f"  Risky locations: {df['risky_location_flag'].sum():,}")
    
    print(f"\n📋 Sample Data (first 5 rows):")
    display_cols = ['product_name', 'product_price', 'discount_percentage', 'seller_rating', 'is_fraud']
    print(df[display_cols].head().to_string())
    
    return df


if __name__ == "__main__":
    # Create dataset with 1000 records
    df = create_singapore_ecommerce_dataset(num_records=1000)
    
    print(f"\n" + "="*70)
    print("✅ DONE! You can now use this dataset for your fraud detection model.")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Load the IEEE-CIS dataset (for training)")
    print(f"2. Train your model on IEEE-CIS data")
    print(f"3. Test the model on this Singapore dataset")
    print(f"4. Evaluate fraud detection performance")