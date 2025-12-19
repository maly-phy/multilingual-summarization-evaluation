import pandas as pd
from nltk import sent_tokenize
from groq import Groq
from dotenv import load_dotenv
import os, json

# from sentence_transformers import SentenceTransformer
import re

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
extra_sent1 = (
    "Here's the list of single facts extracted from the meeting transcript:\n\n"
)
extra_sent2 = "Here's the breakdown of the meeting transcript into single facts:\n\n"


def atomic_facts_from_meeting(df, language):
    system_prompt = "You are an expert text analyzer."
    results = {}
    for idx, row in df[:30].iterrows():
        meeting = row["Meeting"]
        user_prompt = (
            f"Please breakdown the following meeting transcript in {language} into single facts:\n\n{meeting}\n\n"
            f"Please consider all factual information presented in the transcript.\n"
            f"Do not repeat the same fact.\n"
            f"Provide the facts as a numbered list."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            max_tokens=3000,
            temperature=0.0,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=None,
            n=1,
        )
        output = response.choices[0].message.content
        results[idx] = output.replace(extra_sent1, "").replace(extra_sent2, "").strip()

    save_dir = "outputs/atomic_facts"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/atomic_facts.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Atomic facts saved to {save_dir}/atomic_facts.json\n")
    return results


def generate_factual_summary(df, language):
    system_prompt = "You are an expert meeting summarizer."
    with open("outputs/atomic_facts/atomic_facts.json", "r", encoding="utf-8") as f:
        atomic_facts = json.load(f)

    results = []
    for idx, row in df[:30].iterrows():
        meeting = row["Meeting"]
        facts = atomic_facts.get(str(idx), "")
        facts = facts.replace(extra_sent2, "").strip()
        user_prompt = (
            f"You will be given a meeting transcript and atomic facts extracted from it in {language} language. Please follow these steps carefully to generate a summary in {language}:\n\n"
            f"- First read the meeting transcript thoroughly to get familiar with the topic.\n"
            f"- Then review the list of atomic facts to identify the key points discussed in the meeting.\n"
            f"Now, generate a concise and accurate summary that captures the main ideas and important details from both the transcript and the atomic facts.\n\n"
            f"Meeting transcript:\n{meeting}\n\n"
            f"Atomic Facts:\n{facts}\n\n"
            f"Please output just the summary text (complete; without bullet points) within <summary> </summary> tags and no additional text. The summary must be **strictly under 300 words**."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            max_tokens=512,
            temperature=0.0,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=None,
            n=1,
        )
        output = response.choices[0].message.content.strip()
        summary = output.replace("<summary>", "").replace("</summary>", "").strip()
        results.append(
            {
                "model_factual_summary": summary,
                "model_summary": row["model_summary"],
                "ref_summary": row["ref_summary"],
            }
        )

    output_df = pd.DataFrame(results)
    save_dir = "outputs/atomic_facts"
    output_df.to_csv(f"{save_dir}/fact_based_summaries.csv", index=False)
    print(f"Fact-based summaries saved to {save_dir}/fact_based_summaries.csv\n")
    return output_df


def review_and_correct_summary(df, language):
    system_prompt = "You are an expert summary corrector."

    with open("outputs/atomic_facts/atomic_facts.json", "r", encoding="utf-8") as f:
        atomic_facts = json.load(f)

    results = []
    for idx, row in df[:30].iterrows():
        ref_summary = row["ref_summary"]
        facts = atomic_facts.get(str(idx), "")
        user_prompt = (
            f"You will be given a summary and atomic facts extracted from the original knowledge source of the summary in {language} language. Please follow these steps carefully to review and correct the summary based on the provided facts:\n\n"
            f"- First read the summary thoroughly to get familiar with its content.\n"
            f"- Then review the list of atomic facts to identify any discrepancies or inaccuracies in the summary.\n"
            f"Now, read the summary again and check for any factual inaccuracies or discrepancies based on the provided atomic facts following these guidelines:\n\n"
            f"- If you find any information in the summary that contradicts the atomic facts, please correct it based on the facts.\n"
            f"- If you find information in the summary that is not supported by any of the atomic facts, please remove them from the summary.\n"
            f"- Do not correct the summary based on your own knowledge, always rely on the provided atomic facts.\n\n"
            f"Original Summary:\n{ref_summary}\n\n"
            f"Atomic Facts:\n{facts}\n\n"
            f"Please output just the corrected summary text (complete; without bullet points) within <corrected_summary> </corrected_summary> tags and no additional text. The corrected summary must be **strictly under 300 words**."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            max_tokens=512,
            temperature=0.0,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=None,
            n=1,
        )
        output = response.choices[0].message.content.strip()
        corrected_summary = (
            output.replace("<corrected_summary>", "")
            .replace("</corrected_summary>", "")
            .strip()
        )
        results.append(
            {
                "corrected_summary": corrected_summary,
                "model_factual_summary": row["model_factual_summary"],
                "model_summary": row["model_summary"],
                "ref_summary": ref_summary,
            }
        )

    output_df = pd.DataFrame(results)
    save_dir = "outputs/atomic_facts"
    output_df.to_csv(f"{save_dir}/corrected_summary.csv", index=False)
    print(f"Corrected summaries saved to {save_dir}/corrected_summary.csv")

    return output_df


# def break_and_review_summary(df):
#     encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     all_sents = []
#     for idx, row in df[:5].iterrows():
#         ref_summary = row["ref_summary"]
#         sents = sent_tokenize(ref_summary)
#         all_sents.append(sents)
#         for i, sents in enumerate(all_sents):
#             pass


def convert_facts_into_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        atomic_facts = json.load(f)
    all_facts = []
    for idx, fact in atomic_facts.items():
        facts = fact.split("\n")
        facts = [
            sent
            for sent in facts
            if not sent.startswith("Note") or sent.startswith("The list")
        ]
        facts = [
            re.sub(r"^\d+\.\s", "", sent).strip() for sent in facts if sent.strip()
        ]
        fact_text = " ".join(facts)
        all_facts.append(fact_text)

    return all_facts


if __name__ == "__main__":
    language = "German"
    task = "summary_eval"
    src_path = f"evaluation/{language}/{task}/llama-3.1-8b-instant_{task}_0_29.csv"
    target_path = f"evaluation/{language}/atomic_facts/corrected_summary.csv"
    facts_path = f"evaluation/{language}/atomic_facts/atomic_facts.json"
    # facts = convert_facts_into_text(facts_path)
    df = pd.read_csv(src_path)
    print(df["Meeting"].iloc[29], "\n")
    # atomic_facts = atomic_facts_from_meeting(df, language)
    # factual_summary_df = generate_factual_summary(df)
    # corr_summ = review_and_correct_summary(df)
    # all_facts = convert_text_into_sents()
    # print(len(all_facts), "\n")
    # print(all_facts[:2], "\n")
