import os
import pandas as pd
from rouge_score import rouge_scorer
from utils import merge_data_files
from preprocess_corpus import Cleaner


def measure_rouge_groundedness(article_text, meeting_text):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(article_text, meeting_text)
    results = {}
    for rouge_type, score_obj in scores.items():
        results[rouge_type] = {
            "precision": score_obj.precision,
            "recall": score_obj.recall,
            "f1": score_obj.fmeasure,
        }
    return results


def compute_meeting_groundedness(meeting_df, save_path):
    all_results = []
    for idx, row in meeting_df.iterrows():
        article_text = row["Article"]
        meeting_text = row["Meeting"]
        scores = measure_rouge_groundedness(article_text, meeting_text)
        all_results.append(
            {
                "Article": article_text,
                "Meeting": meeting_text,
                ## ROUGE-1
                "rouge1_precision": scores["rouge1"]["precision"],
                "rouge1_recall": scores["rouge1"]["recall"],
                "rouge1_f1": scores["rouge1"]["f1"],
                ## ROUGE-2
                "rouge2_precision": scores["rouge2"]["precision"],
                "rouge2_recall": scores["rouge2"]["recall"],
                "rouge2_f1": scores["rouge2"]["f1"],
                ## ROUGE-L
                "rougel_precision": scores["rougeL"]["precision"],
                "rougel_recall": scores["rougeL"]["recall"],
                "rougel_f1": scores["rougeL"]["f1"],
            }
        )
        if idx % 50 == 0:
            print(f"Processed {idx} / {len(meeting_df)} meetings...")
    results_df = pd.DataFrame(all_results)
    results_df.reset_index(drop=True, inplace=True)
    results_df.to_csv(save_path, index=False)
    print(f"Meeting groundedness results saved to {save_path}")

    return results_df


if __name__ == "__main__":
    task = "meeting_groundedness_eval"
    meeting_language = "German"
    save_dir = os.path.join("evaluation", f"{meeting_language}", f"{task}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"groundedness_eval.csv")

    meetings_df = merge_data_files("data/fame_dataset", meeting_language)
    groundedness_results_df = compute_meeting_groundedness(meetings_df, save_path)
