import pandas as pd
import os
from utils import initialize_model, read_json_criteria


class Refiner:
    def __init__(self, feedback_df, max_tokens, language, criteria_path):
        self.feedback_df = feedback_df
        self.max_tokens = max_tokens
        self.language = language
        self.criteria = read_json_criteria(criteria_path)

    def refine_summary(self, model_init, model_summary, meeting_transcript, idx):
        update_refined_summary = {}
        system_prompt = "You are an experienced linguist and expert in refining meeting summaries to achieve the best quality.\n"
        for criterion, description in self.criteria.items():
            user_prompt = (
                "You will be given a summary for a meeting transcript, a defined error type, along with a review that includes error instances present in the summary for the considered error type, and feedback suggestions to correct the error instances to obtain a high quality summary.\n"
                "Your task is to refine the summary for the considered error type, based on the provided review to end up with the best version of the summary.\n"
                "Please make sure you read and understand the following instructions carefully that guide you through the task.\n"
                "1. Please read the following definition of the error type which will help you understand the task:\n"
                f"{criterion}: {description['definition']}\n"
                "2. Read the meeting transcript carefully and identify the main topics and key points discussed.\n"
                "3. Read the original summary carefully.\n"
                "4. Consider the error instances, and understand the suggested changes.\n"
                "5. Refine the summary based on the feedback provided for each instance.\n"
                "6. At the end, make sure that your ultimate refined summary is coherent, readable and contextually accurate according to the original meeting transcript.\n\n"
                "Now, you should perform the task, given the following inputs:\n"
                f"Meeting transcript: {meeting_transcript}\n"
                f"Summary: {model_summary if criterion == 'Redundancy' else update_refined_summary['Refined summary']}\n"
                f"List of error instances with feedback: {self.feedback_df.iloc[idx][criterion]}\n\n"
                "Please return only the refined summary strictly in **valid JSON format**, using **double quotes** for keys and values without any extra preambles, explanations, or text outside the JSON structure. Make sure to return your answer strictly in the following format:\n"
                "{\n"
                '  "Refined summary": "<your refined summary>"\n'
                "}"
            )
            response = model_init.call_model(system_prompt, user_prompt)
            update_refined_summary["Refined summary"] = response["Refined summary"]
        return update_refined_summary

    def iter_refine_summary(self):
        model_init = initialize_model(max_tokens=self.max_tokens)
        for idx, row in self.feedback_df.iterrows():
            model_summary = row["model_summary"]
            meeting_transcript = row["meeting_transcript"]
            refined_summary = self.refine_summary(
                model_init, model_summary, meeting_transcript, idx
            )
            self.feedback_df.at[idx, "refined_summary"] = refined_summary[
                "Refined summary"
            ]

            if not refined_summary["Refined summary"]:
                print(f"Failed to get a valid response for index: {idx}")
                continue

            if idx % 5 == 0:
                print(f"Processing {idx} rows so far ...")

        save_dir = f"multiagent_summary/evaluation/{self.language}/error_based/refined_summary.csv"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        self.feedback_df.to_csv(save_dir, index=False)
        print(f"Refined summary saved to {save_dir}")


if __name__ == "__main__":
    criteria_path = "multiagent_summary/error_types/error_types_eng.json"
    language = "English"
    feedback_df_path = (
        f"multiagent_summary/evaluation/{language}/error_based/feedback.csv"
    )
    feedback_df = pd.read_csv(feedback_df_path)
    max_tokens = 3000
    # summary_refiner = Refiner(feedback_df, max_tokens, language, criteria_path)
    # summary_refiner.iter_refine_summary()

    from utils import text_chunker

    out_df = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/error_based/refined_summary.csv"
    )
    out_file = f"multiagent_summary/outputs/{language}/refined_samples.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        for idx, row in out_df[:3].iterrows():
            f.write(f"Refined Summary {idx}:\n")
            f.write(f"{text_chunker(row['refined_summary'])}\n\n")
