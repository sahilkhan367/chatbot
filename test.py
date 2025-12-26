from dotenv import load_dotenv
import os

load_dotenv()   # loads .env from current directory

TOKEN = os.getenv("HF_TOKEN")
print(TOKEN)  