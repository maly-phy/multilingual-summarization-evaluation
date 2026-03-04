import json
from groq import Groq
import sys, os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.model_handler import ModelHandler

load_dotenv()


def read_json_criteria(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def initialize_model(
    max_tokens=1000,
):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model_name = "llama-3.1-8b-instant"
    model_init = ModelHandler(
        client=client,
        model_name=model_name,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return model_init
