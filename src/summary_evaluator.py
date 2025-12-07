import pandas as pd
import os
from groq import Groq
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch
from dotenv import load_dotenv
from utils import extract_content_between_tags, merge_data_files, initialize_model
import nltk

load_dotenv()


class SummaryEvaluator:
    def __init__(self, df, task, language, from_local_model, max_tokens):
        self.df = df
        self.task = task
        self.language = language
        self.from_local_model = from_local_model
        self.max_tokens = max_tokens

    def evaluate_summaries(self, candidate_summaries, reference_text):
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

        if self.language == "English":
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
        elif self.language == "German":
            P, R, F1 = bert_score(
                cands=candidate_texts,
                refs=reference_texts,
                model_type="google-bert/bert-base-german-cased",
                lang="de",
                verbose=False,
                idf=False,
                batch_size=4,
                device="cuda" if torch.cuda.is_available() else "cpu",
                num_layers=12,
            )

        for idx, model_name in enumerate(candidate_summaries.keys()):
            results["bert_p"] = P[idx].item()
            results["bert_r"] = R[idx].item()
            results["bert_f1"] = F1[idx].item()

        return results

    def process_meeting_summaries(self):
        system_prompt = "You are a helpful assistant."
        df = self.df[:30]
        start_idx = df.index[0]
        end_idx = df.index[-1]

        model_init, save_path = initialize_model(
            self.task, self.language, self.from_local_model, self.max_tokens
        )
        root_filename = os.path.basename(save_path).replace(".csv", "")
        root_filename += f"_{start_idx}_{end_idx}.csv"
        dir_name = os.path.dirname(save_path)
        save_path = os.path.join(dir_name, root_filename)
        print(f"save_path: {save_path}")

        all_results = []
        for idx, row in df.iterrows():
            text_to_summarize = row["Meeting"]
            reference_summary = row["Summary"]
            user_prompt = f"Summarize the following meeting transcript (in {self.language}){text_to_summarize}.\n Output just the summary text (complete; without bullet points) (use {self.language} ) within <summary> </summary> tags and no additional text. The summary must be **strictly under 250 words**."

            response = model_init.call_model(system_prompt, user_prompt)

            if response is None:
                print(f"Failed to get response for meeting {idx}. breaking...\n")
                break

            candidate_summary = extract_content_between_tags(response, "summary")
            summary_for_this_sample = {model_name: candidate_summary}

            eval_results = self.evaluate_summaries(
                summary_for_this_sample, reference_summary
            )
            all_results.append(
                {
                    "model_summary": candidate_summary,
                    **eval_results,
                    "Meeting": text_to_summarize,
                    "ref_summary": reference_summary,
                }
            )
            if idx % 10 == 0:
                print(f"Processing {idx} / {len(df)}\n")

        output_df = pd.DataFrame(all_results)
        output_df.to_csv(save_path, index=False)
        print(f"Evaluation results saved to {save_path}\n")

        return output_df


if __name__ == "__main__":
    language = "German"
    max_tokens = 310
    model_name = "llama-3.1-8b-instant"
    task = "summary_eval"
    from_local_model = False

    meetings_df = merge_data_files("data/fame_dataset", language)
    summary_evaluator = SummaryEvaluator(
        meetings_df, task, language, from_local_model, max_tokens
    )
    eval_df = summary_evaluator.process_meeting_summaries()
