import json

class ProductDatabase:
    def __init__(self):
        self.products = self.load_products()

    def load_products(self):
        # Sample African beauty brands database
        return {
            "skincare": {
                "cleansers": [
                    {"name": "Shea Moisture African Black Soap", "price": "$8.99", "skin_type": ["acne_prone", "combination"], "link": "https://example.com/1"},
                    {"name": "Nokware Gentle Cleanser", "price": "$12.50", "skin_type": ["normal", "sensitive"], "link": "https://example.com/2"}
                ],
                "moisturizers": [
                    {"name": "Palmer's Cocoa Butter", "price": "$6.99", "skin_type": ["dry", "normal"], "link": "https://example.com/3"},
                    {"name": "Nubian Heritage Lotion", "price": "$9.99", "skin_type": ["all"], "link": "https://example.com/4"}
                ]
            },
            "hair": {
                "shampoos": [
                    {"name": "Carol's Daughter Sulfate-Free Shampoo", "price": "$11.99", "hair_type": ["4a", "4b", "4c"], "link": "https://example.com/5"},
                    {"name": "Cantu Cleansing Shampoo", "price": "$4.99", "hair_type": ["3a", "3b", "3c"], "link": "https://example.com/6"}
                ]
            }
        }

    def get_recommendations(self, skin_condition, hair_type):
        recommendations = []
        # Get skincare products
        for category, products in self.products["skincare"].items():
            for product in products:
                if skin_condition in product["skin_type"] or "all" in product["skin_type"]:
                    recommendations.append({
                        "category": f"skincare_{category}",
                        "product": product
                    })
        # Get hair products
        for category, products in self.products["hair"].items():
            for product in products:
                if hair_type in product["hair_type"]:
                    recommendations.append({
                        "category": f"hair_{category}",
                        "product": product
                    })
        return recommendations[:6]  # Limit to 6 recommendations

def handle_product_request(skin_condition, hair_type):
    product_db = ProductDatabase()
    recommendations = product_db.get_recommendations(skin_condition, hair_type)
    response = "\U0001F6CD️ *Recommended Products for You:*\n"
    for i, rec in enumerate(recommendations, 1):
        product = rec["product"]
        response += f"{i}. **{product['name']}**\n"
        response += f"   \U0001F4B0 {product['price']}\n"
        response += f"   \U0001F517 {product['link']}\n\n"
    response += "\U0001F4A1 *Tip: Always patch test new products!*"
    return response
