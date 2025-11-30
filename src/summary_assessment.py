import pandas as pd
import os
from groq import Groq
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch
from dotenv import load_dotenv
from utils import extract_content_between_tags, free_memory, merge_data_files
from model_handler import ModelHandler

load_dotenv()

eng_meetings_df = merge_data_files("data/fame_dataset", "English")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
device = "cuda" if torch.cuda.is_available() else "cpu"
meeting_language = "English"
num_words = "250"
model_name = "llama-3.1-8b-instant"
task = "summary_assess"
save_dir = os.path.join("evaluation", f"{meeting_language}", f"{task}")
os.makedirs(save_dir, exist_ok=True)
start_from = 78
save_path = os.path.join(save_dir, f"{model_name}_summary_evaluations_{start_from}.csv")


def evaluate_summaries(candidate_summaries, reference_text, language="English"):
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True
    )
    results = {}
    for model_name, candidate_text in candidate_summaries.items():
        rouge_scores = rouge_scorer_obj.score(reference_text, candidate_text)
        results["rouge1_f1"] = rouge_scores["rouge1"].fmeasure
        results["rouge2_f1"] = rouge_scores["rouge2"].fmeasure
        results["rougeL_f1"] = rouge_scores["rougeLsum"].fmeasure

    candidate_texts = list(candidate_summaries.values())
    reference_texts = [reference_text] * len(candidate_texts)

    if language == "English":
        P, R, F1 = bert_score(
            cands=candidate_texts,
            refs=reference_texts,
            model_type="bert-base-uncased",
            lang="en",
            verbose=False,
            idf=False,
            batch_size=4,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        for idx, model_name in enumerate(candidate_summaries.keys()):
            results["bert_p"] = P[idx].item()
            results["bert_r"] = R[idx].item()
            results["bert_f1"] = F1[idx].item()

    return results


def main():
    output_df = pd.DataFrame()
    for idx, row in eng_meetings_df[start_from:].iterrows():
        print(f"Processing meeting {idx} / {len(eng_meetings_df)}\n")
        text_to_summarize = row["Meeting"]
        reference_summary = row["Summary"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Summarize the following meeting transcript (in {meeting_language}){text_to_summarize}.\n Output just the summary text (complete; without bullet points) (use {meeting_language} ) within <summary> </summary> tags and no additional text. The summary must be **strictly under {num_words} words**.",
            },
        ]
        response = ModelHandler.call_model_with_retry(
            client=client,
            messages=messages,
            model=model_name,
            max_tokens=310,
            max_attempts=2,
            base_delay=1.5,
        )
        if response is None:
            print(f"Failed to get response for meeting {idx}. breaking...\n")
            break

        full_model_summary = response.choices[0].message.content

        candidate_summary = extract_content_between_tags(full_model_summary, "summary")
        summary_for_this_sample = {model_name: candidate_summary}

        eval_results = evaluate_summaries(
            summary_for_this_sample, reference_summary, meeting_language
        )

        output_df = pd.concat(
            [
                output_df,
                pd.DataFrame(
                    {
                        "model_summary": [[candidate_summary]],
                        **{k: [v] for k, v in eval_results.items()},
                        "Meeting": [[text_to_summarize]],
                        "ref_summary": [[reference_summary]],
                    }
                ),
            ],
            axis=0,
            ignore_index=True,
        )

        output_df = output_df.reset_index(drop=True)
        output_df.to_csv(save_path, header=True, index=False)
        print(f"Evaluation results saved to {save_path}\n")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
