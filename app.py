from flask import Flask, request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import os
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
import logging
from datetime import datetime
from utils.product_database import handle_product_request
from utils.user_session import UserSession
from googletrans import Translator

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER')

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('beauty_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Session and translation support
session_manager = UserSession()
translator = Translator()

# --- Enhanced Unified Message Handler ---
def enhanced_message_handler(incoming_msg, sender, media_url=None):
    user_data = session_manager.get_user_data(sender)
    user_lang = 'en'
    try:
        detected = translator.detect(incoming_msg)
        if detected.confidence > 0.8:
            user_lang = detected.lang
    except Exception:
        pass

    # Handle image/selfie
    if media_url:
        analysis = process_beauty_image(media_url, sender)
        session_manager.save_user_data(sender, {
            "last_analysis": analysis,
            "analysis_date": datetime.now().isoformat(),
            "face_shape": analysis.get("face_shape"),
            "skin_condition": analysis.get("skin_condition"),
            "hair_type": analysis.get("hair_type", "unknown")
        })
        response = format_response(analysis)
        if user_lang != 'en':
            response = translator.translate(response, dest=user_lang).text
        return response

    # Handle product recommendations
    if "products" in incoming_msg.lower():
        if user_data.get("last_analysis"):
            response = handle_product_request(
                user_data["last_analysis"].get("skin_condition", {}).get("condition", "normal"),
                user_data["last_analysis"].get("hair_type", "unknown")
            )
        else:
            response = "Please send me a selfie first so I can recommend the right products for you!"
        if user_lang != 'en':
            response = translator.translate(response, dest=user_lang).text
        return response

    # Handle salon booking (placeholder)
    if "book" in incoming_msg.lower():
        # You can implement a real handler here
        response = "Salon booking is coming soon!"
        if user_lang != 'en':
            response = translator.translate(response, dest=user_lang).text
        return response

    # Default/help
    response = handle_text_message(incoming_msg, sender)
    if user_lang != 'en':
        response = translator.translate(response, dest=user_lang).text
    return response

@app.route('/webhook', methods=['POST'])
def whatsapp_webhook():
    incoming_msg = request.values.get('Body', '').strip()
    sender = request.values.get('From')
    media_url = request.values.get('MediaUrl0')
    response = MessagingResponse()
    try:
        reply = enhanced_message_handler(incoming_msg, sender, media_url)
        response.message(reply)
        logger.info(f"Handled message from {sender}")
    except Exception as e:
        logger.error(f"Error handling message from {sender}: {str(e)}")
        response.message("Sorry, something went wrong. Please try again later.")
    return str(response)

def handle_text_message(message, sender):
    responses = {
        'hello': "\ud83d\udc4b Welcome to your AI Beauty Advisor! Send me a selfie and I'll recommend the perfect hairstyle and skincare routine for you!",
        'help': "\ud83d\udcf1 Here's what I can do:\n\u2022 Send a selfie for hair & skin analysis\n\u2022 Get personalized product recommendations\n\u2022 Book salon appointments\n\u2022 Ask beauty questions",
        'menu': "\ud83c\udfaf *Main Menu*:\n1\ufe0f\u20e3 Hair Analysis\n2\ufe0f\u20e3 Skin Analysis\n3\ufe0f\u20e3 Product Recommendations\n4\ufe0f\u20e3 Book Salon\n5\ufe0f\u20e3 Beauty Tips"
    }
    return responses.get(message, "I didn't understand that. Type 'help' for available options!")

def process_beauty_image(media_url, sender):
    try:
        # Step 1: Download the image from WhatsApp
        response = requests.get(media_url)
        image = Image.open(BytesIO(response.content))
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Step 3: Perform analyses using AI stubs
        face_shape = ai_predict_face_shape(cv_image)
        skin_analysis = ai_predict_skin_tone(cv_image)
        hair_analysis = ai_predict_hair_texture(cv_image)

        # Step 4: Generate recommendations
        recommendations = generate_recommendations(face_shape, skin_analysis, hair_analysis)

        # Step 5: Format the response
        return format_response(recommendations)
    except Exception as e:
        return f"\u274C Sorry, I couldn't process your image. Please try again with a clear selfie!"

# === Advanced AI Model Stubs ===
# Example: Load your models here (update paths as needed)
# face_shape_model = tf.keras.models.load_model('models/face_shape_model.h5')
# skin_tone_model = tf.keras.models.load_model('models/skin_tone_model.h5')
# hair_texture_model = tf.keras.models.load_model('models/hair_texture_model.h5')

# Example inference function for face shape (replace with real model logic)
def ai_predict_face_shape(image):
    # Preprocess image as needed for your model
    # prediction = face_shape_model.predict(preprocessed_image)
    # return decode_prediction(prediction)
    return predict_face_shape_from_array(image)  # use the trained model for prediction

# Example inference function for skin tone (replace with real model logic)
def ai_predict_skin_tone(image):
    # Preprocess image as needed for your model
    # prediction = skin_tone_model.predict(preprocessed_image)
    # return decode_prediction(prediction)
    return analyze_skin_condition(image)  # fallback for now

# Example inference function for hair texture (replace with real model logic)
def ai_predict_hair_texture(image):
    # Preprocess image as needed for your model
    # prediction = hair_texture_model.predict(preprocessed_image)
    # return decode_prediction(prediction)
    return analyze_hair_texture(image)  # fallback for now

def predict_face_shape_from_array(image_array):
    model = tf.keras.models.load_model('models/face_shape_model.h5')
    img = cv2.resize(image_array, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    FACE_SHAPES = ['oval', 'round', 'square']
    return FACE_SHAPES[predicted_class], confidence

def analyze_face_shape(image):
    face_shape, confidence = predict_face_shape_from_array(image)
    return face_shape

def analyze_skin_condition(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_area = cv2.bitwise_and(image, image, mask=skin_mask)
    gray_skin = cv2.cvtColor(skin_area, cv2.COLOR_BGR2GRAY)
    texture_variance = np.var(gray_skin)
    if texture_variance > 1000:
        return {"condition": "acne_prone", "dryness": "moderate"}
    elif texture_variance > 500:
        return {"condition": "combination", "dryness": "low"}
    else:
        return {"condition": "normal", "dryness": "balanced"}

def analyze_hair_texture(image):
    # Placeholder for now (will enhance later with ML models)
    return "4C"  # Default for Afro-textured hair

def generate_recommendations(face_shape, skin_analysis, hair_analysis):
    hairstyle_recommendations = {
        "round": ["Box braids", "High top fade", "Twist out", "Bantu knots"],
        "oval": ["Afro", "Cornrows", "Locs", "Finger waves"],
        "square": ["Soft curls", "Side-swept bangs", "Layered cut", "Protective styles"]
    }
    skincare_recommendations = {
        "acne_prone": ["Gentle cleanser with salicylic acid", "Non-comedogenic moisturizer", "SPF 30+ sunscreen"],
        "combination": ["Balancing cleanser", "Lightweight moisturizer", "Oil-free products"],
        "normal": ["Hydrating cleanser", "Daily moisturizer", "Vitamin C serum"]
    }
    return {
        "hairstyles": hairstyle_recommendations.get(face_shape, ["Consult with a stylist"]),
        "skincare": skincare_recommendations.get(skin_analysis.get("condition", "normal"), ["Basic routine"]),
        "face_shape": face_shape,
        "skin_condition": skin_analysis
    }

def format_response(recommendations):
    response = f"""✨ *Your Beauty Analysis Results* ✨\n👤 **Face Shape:** {recommendations['face_shape'].title()}\n💇‍♀️ **Recommended Hairstyles:**\n"""
    for i, style in enumerate(recommendations['hairstyles'][:3], 1):
        response += f"{i}. {style}\n"
    response += f"""\n🧴 **Skincare Routine:**\n"""
    for i, product in enumerate(recommendations['skincare'][:3], 1):
        response += f"{i}. {product}\n"
    response += """\n🛍️ *Reply with 'products' for shopping links*\n📅 *Reply with 'book' to find nearby salons*\n❓ *Reply with 'tips' for styling advice*"""
    return response

print("Stripe Key:", os.getenv('STRIPE_SECRET_KEY'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
