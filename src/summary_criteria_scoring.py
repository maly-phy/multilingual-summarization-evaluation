import pandas as pd
import os, sys
from groq import Groq
from dotenv import load_dotenv
from utils import extract_content_between_tags, initialize_model

load_dotenv()


class SummaryScorer:
    def __init__(self, df, task, language, from_local_model, max_tokens):
        self.df = df
        self.task = task
        self.language = language
        self.from_local_model = from_local_model
        self.max_tokens = max_tokens
        self.criteria = {
            "Naturalness": f"How natural the conversation flows, like native {self.language} speakers (1-5)"
        }

    def summary_criteria_eval(self, model_init, model_summary):
        criteria_eval = {}
        for criterion, description in self.criteria.items():
            system_prompt = (
                f"You are an expert linguist evaluating meeting summaries (in {self.language}). "
                f"Evaluate the following meeting summary thoroughly for **{criterion}**: {description}. \n"
                "- **Rating 1**: Highlights minimal or absent behaviour for each criterion.\n"
                "- **Rating 5**: Showcases strong, explicit demonstration of the behaviour.\n"
                f"Provide your step-by-step reasoning in only 1-2 sentences in {self.language}, a confidence score (0-100%), and a final score as a decimal number between 1.0 and 5.0, demonstrating the degree to which the chosen criterion is satisfied in {self.language} context. "
                "Format your response using XML tags: "
                "<reasoning>detailed step-by-step analysis</reasoning> "
                "<confidence>your confidence percentage</confidence> "
                "<score>decimal number between 1.0 and 5.0</score> "
                "You must NOT return any reasoning text with either the confidence score or the final score."
            )
            user_prompt = f"Please evaluate this meeting summary in {self.language} for {criterion}:\n\n{model_summary}"

            response = model_init.call_model(system_prompt, user_prompt)
            if not response:
                print(f"No response for criterion {criterion}, breaking...\n")
                break

            criteria_eval[criterion] = {
                "base_reasoning": extract_content_between_tags(response, "reasoning"),
                "base_confidence": extract_content_between_tags(response, "confidence"),
                "base_score": extract_content_between_tags(response, "score"),
            }

        eval_results = {
            "model_summary": model_summary,
            **{
                f"{criterion}_{key}": value
                for criterion, res in criteria_eval.items()
                for key, value in res.items()
            },
        }

        return eval_results

    def process_summary_scoring(self):
        df = self.df[:30]
        start_idx = df.index[0]
        end_idx = df.index[-1]
        model_init, save_path = initialize_model(
            self.task, self.language, self.from_local_model, self.max_tokens
        )

        root_filename = os.path.basename(save_path).replace(".csv", "")
        root_filename += f"_with_naturalness_{start_idx}_{end_idx}.csv"
        dir_name = os.path.dirname(save_path)
        save_path = os.path.join(dir_name, root_filename)
        print(f"save_path: {save_path}")

        all_results = []
        for idx, row in df.iterrows():
            model_summary = row["model_summary"]
            eval_results = self.summary_criteria_eval(model_init, model_summary)

            if not eval_results:
                print(f"No evaluation results for idx {idx}, continuing...\n")
                continue

            if idx % 4 == 0:
                print(f"\nProcessing {idx}/{len(df)}\n")
                for key, results in eval_results.items():
                    if key == "model_summary":
                        continue
                    print(f"{key}: {results}\n")

            all_results.append(eval_results)

        output_df = pd.DataFrame(all_results)
        output_df.to_csv(save_path, index=False)
        print(f"Evaluation results saved to {save_path}\n")


if __name__ == "__main__":
    language = "German"
    task = "summary_eval"
    max_tokens = 256
    from_local_model = False
    df_path = f"evaluation/{language}/{task}/llama-3.1-8b-instant_{task}_0_29.csv"
    df = pd.read_csv(df_path)

    summary_scorer = SummaryScorer(
        df,
        task,
        language,
        from_local_model,
        max_tokens,
    )
    summary_scorer.process_summary_scoring()
