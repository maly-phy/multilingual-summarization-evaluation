import os
import re, ast
import gc
import torch
import glob
import pandas as pd


def extract_content_between_tags(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        clean_text = text.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
        return clean_text.strip()


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
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        base_name = os.path.splitext(filename)[0]
        meeting_type = base_name.split("_")[0]
        df["Meeting_Type"] = meeting_type
        meetings_df = pd.concat([meetings_df, df], ignore_index=True)

    meetings_df.reset_index(drop=True, inplace=True)
    return meetings_df


if __name__ == "__main__":
    eng_meetings_df = merge_data_files("data/fame_dataset", "English")
    row = eng_meetings_df.iloc[0]
    print(row["Meeting_Plan"], "\n")
