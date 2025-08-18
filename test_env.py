from dotenv import load_dotenv
import os
load_dotenv()
print("OPENAI_API_KEY:", os.getenv('OPENAI_API_KEY'))
print("SECRET_KEY:", os.getenv('SECRET_KEY'))
print("DATABASE_URL:", os.getenv('DATABASE_URL'))