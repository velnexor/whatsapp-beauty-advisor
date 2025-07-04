# WhatsApp AI Beauty Advisor

A full-featured WhatsApp bot that provides personalized beauty advice, product recommendations, and salon booking for African users. Built with Flask, Twilio, TensorFlow, OpenCV, Redis, Google Maps, and more. Supports multi-language, user session management, and scalable deployment.

---

## Features
- **WhatsApp Integration**: Chatbot accessible via WhatsApp using Twilio API.
- **AI-Powered Image Analysis**: Detects face shape, skin condition, and hair texture from selfies using deep learning and OpenCV.
- **Personalized Recommendations**: Suggests skincare and haircare products based on user analysis.
- **Salon Booking**: Finds and books appointments at nearby salons using Google Places API.
- **User Session Management**: Remembers user data and preferences with Redis.
- **Multi-language Support**: Responds in English, Swahili, Yoruba, French, Arabic, Amharic, and more.
- **Performance Monitoring**: Logs key metrics and errors for production reliability.
- **Webcam Scanning**: Local script for face shape detection using a webcam.

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo-url.git whatsapp-beauty-advisor
cd whatsapp-beauty-advisor
```

### 2. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
DATABASE_URL=postgresql://user:password@localhost/beautybot
REDIS_URL=redis://localhost:6379
SECRET_KEY=your_secret_key
```

### 4. Start Redis (for session management)
```bash
sudo service redis-server start
```

### 5. Train the Face Shape Model (Optional)
Add images to `dataset/train/<class>/` and `dataset/validation/<class>/` folders, then run:
```bash
python train_face_shape_model.py
```

### 6. Run the Bot (Development)
```bash
python app.py
```

### 7. Expose the Server (for Twilio Webhook)
- Use Codespaces port forwarding or [ngrok](https://ngrok.com/):
```bash
ngrok http 5000
```
- Set the webhook URL in Twilio to `https://<your-ngrok-or-codespace-url>/webhook`

### 8. Test the Bot
- Send "hello", "help", or a selfie to your WhatsApp number.
- Reply with "products" for recommendations or "book" for salon booking.

---

## Project Structure
```
whatsapp-beauty-advisor/
├── app.py                  # Main Flask app and WhatsApp webhook
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── models/                 # Trained ML models (e.g., face_shape_model.h5)
├── dataset/                # Training/validation images for ML
├── utils/
│   ├── product_database.py # Product recommendation logic
│   ├── user_session.py     # Redis session management
│   └── download_sample_images.py # (Optional) Download sample images
├── scan_face_webcam.py     # Local webcam face shape scanner
├── train_face_shape_model.py # Model training script
├── README.md               # Project documentation
```

---

## API & Code Documentation

### app.py
- **/webhook**: Main endpoint for WhatsApp messages (POST). Handles text, images, product, and booking requests.
- **process_beauty_image(media_url, sender)**: Downloads and analyzes user selfies.
- **enhanced_message_handler(incoming_msg, sender, media_url)**: Unified handler for all user flows.

### utils/product_database.py
- **ProductDatabase**: Loads and manages product data.
- **handle_product_request(skin_condition, hair_type)**: Returns product recommendations as a formatted string.

### utils/user_session.py
- **UserSession**: Saves and retrieves user data from Redis.

### scan_face_webcam.py
- Local script for face shape prediction using a webcam and the trained model.

### train_face_shape_model.py
- Script to train a MobileNetV3-based face shape classifier. Place images in `dataset/` as described in the script.

---

## Deployment
See the deployment section in this README or your project plan for:
- Cloud server setup (Ubuntu, Digital Ocean, AWS, etc.)
- Gunicorn + Nginx configuration
- Environment variable setup
- SSL/HTTPS with Certbot
- Scaling and monitoring

---

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License
[MIT](LICENSE)

---

## Acknowledgments
- African beauty brands and open datasets
- Twilio, Google Cloud, OpenCV, TensorFlow, Redis, and the open-source community