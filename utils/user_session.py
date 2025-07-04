import redis
import json
from datetime import datetime, timedelta

class UserSession:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    def save_user_data(self, phone_number, data):
        key = f"user:{phone_number}"
        self.redis_client.setex(key, timedelta(hours=24), json.dumps(data))

    def get_user_data(self, phone_number):
        key = f"user:{phone_number}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else {}

    def update_user_preferences(self, phone_number, preferences):
        user_data = self.get_user_data(phone_number)
        user_data.update(preferences)
        self.save_user_data(phone_number, user_data)
