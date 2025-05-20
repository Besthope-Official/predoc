from dotenv import load_dotenv
import os

load_dotenv()
env = os.getenv('ENV', 'dev')
