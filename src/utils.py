import os
import re, ast
import gc
import torch
import glob
import pandas as pd
from groq import Groq
from src.model_handler import ModelHandler
from src.local_model import LocalModel
from dotenv import load_dotenv
import nltk

load_dotenv()


def extract_content_between_tags(text, tag):
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        print(f"Tag <{tag}> not found in the text.\n")
        return text


def free_memory(device):
    torch.cuda.empty_cache()
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"memory reserved: {memory_reserved} GB on {device}\n\n")


def merge_data_files(data_dir, language):
    files = glob.glob(f"{data_dir}/{language}/*.csv")

    meetings_df = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, encoding="utf-8")
        filename = os.path.basename(file)
        base_name = os.path.splitext(filename)[0]
        meeting_type = base_name.split("_")[0]
        df["Meeting_Type"] = meeting_type
        meetings_df = pd.concat([meetings_df, df], ignore_index=True)

    meetings_df.reset_index(drop=True, inplace=True)
    return meetings_df


def initialize_model(
    task="meeting_challenges_assess",
    meeting_language="English",
    from_local_model=False,
    max_tokens=1000,
):
    if from_local_model:
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        model_init = LocalModel(
            model_name=model_name,
            max_new_tokens=max_tokens,
        )
    else:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model_name = "llama-3.1-8b-instant"
        model_init = ModelHandler(
            client=client, model_name=model_name, max_tokens=max_tokens
        )

    save_dir = os.path.join("evaluation", f"{meeting_language}", f"{task}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name.split('/')[-1]}_{task}.csv")

    return model_init, save_path


def text_chunker(text, chunk_size=20):
    words = text.split()
    all_chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        all_chunks.append(chunk)
    results = "\n\n".join(all_chunks)
    return results


def truncate_text(text):
    sents = nltk.sent_tokenize(text, language="german")
    ending_puncts = (".", "!", "?")
    filtered = [sent for sent in sents if sent.strip().endswith(ending_puncts)]
    truncated_text = " ".join(filtered)
    return truncated_text


def save_summaries(task, language, file_path):
    df = pd.read_csv(file_path)
    df["Meeting"] = df["Meeting"].apply(text_chunker)

    with open(f"evaluation/{language}/{task}/{task}.txt", "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            f.write(f"\n\n***Meeting idx: {idx + 2}***\n\n")
            f.write(f"Meeting Transcript:\n{row['Meeting']}\n\n\n")
