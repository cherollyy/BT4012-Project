"""
Shopee Singapore Scraper - OPTIMIZED FOR SPEED
Faster extraction with timeout protection
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
import pandas as pd
import time
import random
import re
from datetime import datetime
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ShopeeScraper:
    def __init__(self, headless=False):
        """Initialize the Shopee scraper"""
        options = uc.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')
        
        # Speed optimizations
        options.add_argument('--disable-images')  # Don't load images
        options.add_argument('--blink-settings=imagesEnabled=false')
        
        logger.info("Starting Chrome browser...")
        self.driver = uc.Chrome(options=options)
        self.driver.set_page_load_timeout(30)
        self.data = []
        
    def search_products(self, search_term, max_products=20):
        """Search for products on Shopee Singapore"""
        logger.info(f"Searching for: {search_term}")
        
        search_url = f"https://shopee.sg/search?keyword={search_term.replace(' ', '%20')}"
        self.driver.get(search_url)
        
        logger.info("Waiting for page to load...")
        time.sleep(8)
        
        # Quick JSON extraction first (fastest method)
        products_data = self._extract_from_json_fast(search_term, max_products)
        
        if not products_data:
            # Fallback to page text extraction (much faster than element iteration)
            products_data = self._extract_from_page_text(search_term, max_products)
        
        if products_data:
            logger.info(f"✅ Successfully extracted {len(products_data)} products")
            self.data.extend(products_data)
        else:
            logger.error("❌ Extraction failed")
            # Save debug info
            self.driver.save_screenshot(f"failed_{search_term.replace(' ', '_')}.png")
        
        return self.data
    
    def _extract_from_json_fast(self, search_term, max_products):
        """Fast JSON extraction from page source"""
        products = []
        
        try:
            logger.info("Extracting from embedded JSON data...")
            page_source = self.driver.page_source
            
            # Look for the main data structure
            # Shopee typically uses window.__INITIAL_STATE__ or similar
            patterns = [
                r'<script>window\.__INITIAL_STATE__=(.+?)</script>',
                r'"items":(\[{.+?}\])',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, page_source, re.DOTALL)
                
                if match:
                    try:
                        json_str = match.group(1)
                        
                        # Clean up the JSON string
                        if json_str.endswith(';'):
                            json_str = json_str[:-1]
                        
                        data = json.loads(json_str)
                        logger.info(f"Found JSON data structure")
                        
                        # Navigate to items
                        items = self._find_items_in_json(data)
                        
                        if items:
                            logger.info(f"Found {len(items)} items in JSON")
                            
                            for idx, item in enumerate(items[:max_products]):
                                product = self._parse_shopee_item(item, search_term, idx)
                                if product:
                                    products.append(product)
                                    logger.info(f"  ✓ {idx+1}. {product['product_name'][:50]} - ${product['product_price']}")
                            
                            return products
                    
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.debug(f"JSON parse error: {e}")
                        continue
        
        except Exception as e:
            logger.warning(f"JSON extraction failed: {e}")
        
        return products
    
    def _find_items_in_json(self, data, max_depth=5, current_depth=0):
        """Recursively find items array in nested JSON"""
        if current_depth > max_depth:
            return None
        
        if isinstance(data, dict):
            # Check common paths
            if 'items' in data:
                items = data['items']
                if isinstance(items, list) and len(items) > 0:
                    # Verify it looks like product data
                    if isinstance(items[0], dict) and any(k in items[0] for k in ['name', 'price', 'itemid']):
                        return items
            
            # Recursive search
            for key, value in data.items():
                result = self._find_items_in_json(value, max_depth, current_depth + 1)
                if result:
                    return result
        
        elif isinstance(data, list) and len(data) > 0:
            # Check if this list contains product-like objects
            if isinstance(data[0], dict) and any(k in data[0] for k in ['name', 'price', 'itemid']):
                return data
        
        return None
    
    def _parse_shopee_item(self, item, search_term, index):
        """Parse Shopee product item from JSON"""
        try:
            product = {
                'search_term': search_term,
                'scrape_timestamp': datetime.now().isoformat(),
                'platform': 'shopee_sg',
                'product_position': index + 1
            }
            
            # Product name
            product['product_name'] = item.get('name') or item.get('title') or f"Product {index+1}"
            
            # Price (Shopee stores in smallest unit, usually cents * 1000)
            price_raw = item.get('price') or item.get('price_min') or item.get('price_max') or 0
            product['product_price'] = float(price_raw) / 100000  # Convert to SGD
            
            # Original price for discount calculation
            price_max = item.get('price_max_before_discount') or item.get('price_before_discount')
            if price_max:
                product['original_price'] = float(price_max) / 100000
                if product['original_price'] > product['product_price']:
                    product['discount_percentage'] = int(((product['original_price'] - product['product_price']) / product['original_price']) * 100)
                else:
                    product['discount_percentage'] = 0
            else:
                product['original_price'] = None
                product['discount_percentage'] = item.get('raw_discount', 0)
            
            # Rating
            rating_data = item.get('item_rating') or {}
            product['product_rating'] = float(rating_data.get('rating_star', 0))
            product['ratings_count'] = int(rating_data.get('rating_count', [0])[0] if isinstance(rating_data.get('rating_count'), list) else rating_data.get('rating_count', 0))
            
            # Sold count
            product['sold_count'] = int(item.get('historical_sold') or item.get('sold', 0))
            
            # URL
            itemid = item.get('itemid')
            shopid = item.get('shopid')
            if itemid and shopid:
                product['product_url'] = f"https://shopee.sg/product/{shopid}/{itemid}"
            else:
                product['product_url'] = None
            
            # Shop/Seller info
            product['seller_name'] = item.get('shop_name') or item.get('shopname')
            product['seller_location'] = item.get('shop_location') or item.get('item_location')
            product['shipping_location'] = product['seller_location']
            
            # Badges
            product['is_mall'] = item.get('is_cc_installment_payment_eligible', False) or item.get('is_mart', False)
            product['is_official_store'] = item.get('is_official_shop', False) or item.get('shopee_verified', False)
            product['is_preferred'] = item.get('is_preferred_plus_seller', False)
            product['free_shipping'] = item.get('show_free_shipping', False)
            
            # Additional fields
            product['brand'] = item.get('brand')
            product['category'] = None  # Would need category mapping
            product['stock'] = item.get('stock')
            
            # Fraud indicators
            product['suspicious_discount'] = product['discount_percentage'] > 70
            product['low_price_indicator'] = product['product_price'] < 5
            product['low_review_count'] = product['ratings_count'] < 10
            product['reviews_count'] = product['ratings_count']
            
            # Seller placeholders
            product['seller_rating'] = None
            product['seller_response_rate'] = None
            product['seller_join_date'] = None
            product['seller_followers'] = None
            product['seller_product_count'] = None
            product['has_return_policy'] = False
            product['return_policy'] = None
            product['likes_count'] = item.get('liked_count', 0)
            
            return product
            
        except Exception as e:
            logger.warning(f"Error parsing item: {e}")
            return None
    
    def _extract_from_page_text(self, search_term, max_products):
        """Fast extraction from page body text"""
        products = []
        
        try:
            logger.info("Extracting from page text...")
            
            # Get body text (much faster than iterating elements)
            body = self.driver.find_element(By.TAG_NAME, 'body')
            page_text = body.text
            
            logger.info(f"Page text length: {len(page_text)} characters")
            
            # Find all prices
            price_pattern = r'\$(\d+\.?\d*)'
            prices = re.findall(price_pattern, page_text)
            logger.info(f"Found {len(prices)} prices")
            
            # Find all sold counts
            sold_pattern = r'([\d,]+\.?[\dkK]*)\s+[Ss]old'
            solds = re.findall(sold_pattern, page_text)
            logger.info(f"Found {len(solds)} sold counts")
            
            # Create products from found data
            count = min(len(prices), len(solds), max_products)
            
            for i in range(count):
                try:
                    # Parse sold count
                    sold_str = solds[i].replace(',', '').lower()
                    if 'k' in sold_str:
                        sold_count = int(float(sold_str.replace('k', '')) * 1000)
                    else:
                        sold_count = int(sold_str) if sold_str.replace('.', '').isdigit() else 0
                    
                    product = {
                        'search_term': search_term,
                        'scrape_timestamp': datetime.now().isoformat(),
                        'platform': 'shopee_sg',
                        'product_position': i + 1,
                        'product_name': f"{search_term} - Product {i+1}",
                        'product_price': float(prices[i]),
                        'sold_count': sold_count,
                        'product_rating': None,
                        'ratings_count': 0,
                        'reviews_count': 0,
                        'discount_percentage': 0,
                        'original_price': None,
                        'product_url': None,
                        'seller_name': None,
                        'seller_location': None,
                        'shipping_location': None,
                        'is_mall': False,
                        'is_official_store': False,
                        'is_preferred': False,
                        'free_shipping': False,
                        'brand': None,
                        'category': None,
                        'stock': None,
                        'suspicious_discount': False,
                        'low_price_indicator': float(prices[i]) < 5,
                        'low_review_count': True,
                        'seller_rating': None,
                        'seller_response_rate': None,
                        'seller_join_date': None,
                        'seller_followers': None,
                        'seller_product_count': None,
                        'has_return_policy': False,
                        'return_policy': None,
                        'likes_count': 0
                    }
                    
                    products.append(product)
                    logger.info(f"  ✓ Product {i+1}: ${product['product_price']} - {product['sold_count']} sold")
                
                except Exception as e:
                    logger.debug(f"Error creating product {i}: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"Page text extraction failed: {e}")
        
        return products
    
    def save_to_csv(self, filename='shopee_data.csv'):
        """Save scraped data to CSV"""
        if not self.data:
            logger.warning("No data to save")
            return None
        
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"✅ Data saved to {filename} ({len(df)} products)")
        
        return df
    
    def get_dataframe(self):
        """Return data as pandas DataFrame"""
        return pd.DataFrame(self.data)
    
    def close(self):
        """Close the browser"""
        try:
            self.driver.quit()
            logger.info("Browser closed")
        except:
            pass


# MAIN EXECUTION
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SHOPEE SINGAPORE SCRAPER - OPTIMIZED VERSION")  
    print("Group 4: Fraud Detection Project")
    print("="*70)
    
    scraper = ShopeeScraper(headless=False)
    
    try:
        # Search terms for fraud detection
        search_terms = [
            "luxury bags",
            "designer handbag",
            "branded watch",
            "iphone 15",
            "samsung galaxy",
            "designer clothes"
        ]
        
        # Start with one term for testing
        for term in search_terms[:1]:  # Remove [:1] to scrape all terms
            logger.info(f"\n{'='*70}")
            scraper.search_products(term, max_products=20)
            time.sleep(random.uniform(5, 8))
        
        # Get results
        df = scraper.get_dataframe()
        
        if len(df) > 0:
            scraper.save_to_csv('shopee_singapore_data.csv')
            
            print("\n" + "="*70)
            print("✅ SCRAPING SUCCESSFUL!")
            print("="*70)
            print(f"\nTotal products: {len(df)}")
            
            print(f"\n📊 Sample Data:")
            print(df[['product_name', 'product_price', 'sold_count', 'product_rating']].head(10).to_string())
            
            print(f"\n📈 Statistics:")
            print(f"  Avg price: ${df['product_price'].mean():.2f}")
            print(f"  Price range: ${df['product_price'].min():.2f} - ${df['product_price'].max():.2f}")
            if df['product_rating'].notna().any():
                print(f"  Avg rating: {df['product_rating'].mean():.2f}")
            print(f"  Total sold: {df['sold_count'].sum():,}")
            
            print(f"\n🚨 Fraud Indicators:")
            print(f"  Suspicious discounts (>70%): {df['suspicious_discount'].sum()}")
            print(f"  Low prices (<$5): {df['low_price_indicator'].sum()}")
            print(f"  Low reviews (<10): {df['low_review_count'].sum()}")
            print(f"  Mall sellers: {df['is_mall'].sum()}")
            print(f"  Official stores: {df['is_official_store'].sum()}")
            
            print(f"\n💾 Data saved to: shopee_singapore_data.csv")
            
        else:
            print("\n❌ No products scraped")
            print("Check debug screenshots for details")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔒 Closing browser...")
        scraper.close()
        print("Done!")