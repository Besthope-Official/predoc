from dotenv import load_dotenv
import os

load_dotenv()

VALID_ENVIRONMENTS = ['dev', 'test', 'prod']
env = os.getenv('ENV', 'dev')

if env not in VALID_ENVIRONMENTS:
    raise ValueError(
        f"Invalid environment: {env}. Must be one of {VALID_ENVIRONMENTS}")
