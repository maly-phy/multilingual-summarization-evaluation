import pandas as pd
import os, sys
from utils import initialize_model, read_json_criteria
import ast


class Refiner:
    def __init__(self, feedback_df, max_tokens, language, criteria_path):
        self.feedback_df = feedback_df
        self.max_tokens = max_tokens
        self.language = language
        self.criteria = read_json_criteria(criteria_path)

    def refiner_prompt(
        self,
        model_init,
        criterion,
        description,
        model_summary,
        meeting_transcript,
        update_refined_summary,
        i,
        row,
    ):
        system_prompt = "You are an experienced linguist and expert in refining meeting summaries to achieve the best quality.\n"
        user_prompt = (
            "You will be given a summary for a meeting transcript, a defined error type, along with a feedback that includes suggestions to correct the errors present in the summary for the considered error type to end up with a high quality summary.\n"
            "Your task is to refine the summary for the considered error type, based on the provided feedback to end up with the best version of the summary.\n"
            "Please make sure you read and understand the following instructions carefully that guide you through the task.\n"
            "1. Please read the following definition of the error type which will help you understand the task:\n"
            f"{criterion}: {description['definition']}\n"
            "2. Read the meeting transcript carefully and identify the main topics and key points discussed. Please keep the transcript open while performing the task, and refer to it whenever needed.\n"
            "3. Read the original summary carefully.\n"
            "4. Consider the feedback and understand the suggested changes.\n"
            "Please consider only the suggestions of the mid-to-high severity instances (3-5 severity score) whose corrections can improve the summary's quality significantly.\n"
            "5. Refine the summary based on the feedback provided.\n"
            "You can look up the original meeting transcript to pick content or information that suffices the feedback, if needed.\n"
            "Please keep the length of the refined summary within the length of the original summary (under 250 words).\n"
            "6. At the end, make sure that your ultimate refined summary is coherent, readable, contextually accurate, and maintains the main topics and key points discussed in the meeting transcript.\n\n"
            "Now, you should perform the task, given the following inputs:\n"
            f"Meeting transcript: {meeting_transcript}\n"
            f"Summary: {model_summary if criterion == 'Redundancy' else update_refined_summary[i-1][f'refined_summary_{i-1}']}\n"
            f"Feedback: {row[criterion]}\n\n"
            "Please return only the refined summary without any extra preambles, explanations, or text:\n"
            "<your refined summary>"
        )
        response = model_init.call_model(system_prompt, user_prompt)
        return response

    def refine_summary(self, model_init, model_summary, meeting_transcript, row):
        update_refined_summary = []
        for i, (criterion, description) in enumerate(self.criteria.items()):
            response = self.refiner_prompt(
                model_init,
                criterion,
                description,
                model_summary,
                meeting_transcript,
                update_refined_summary,
                i,
                row,
            )
            update_refined_summary.append({f"refined_summary_{i}": response})

        return update_refined_summary

    def process_refine_summary(self):
        model_init = initialize_model(max_tokens=self.max_tokens)
        for idx, row in self.feedback_df.iterrows():
            model_summary = row["model_summary"]
            meeting_transcript = row["meeting_transcript"]
            refined_summary = self.refine_summary(
                model_init, model_summary, meeting_transcript, row
            )
            if idx > 3:
                break

            print("refined summ length:", len(refined_summary), "\n")
            for i in range(len(refined_summary)):
                self.feedback_df.at[idx, f"refined_summary_{i}"] = refined_summary[i][
                    f"refined_summary_{i}"
                ]
        save_dir = f"multiagent_summary/evaluation/{self.language}/error_based/sample_refined_summaries.csv"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        self.feedback_df.to_csv(save_dir, index=False)
        print(f"Refined summary saved to {save_dir}")


if __name__ == "__main__":
    criteria_path = "multiagent_summary/error_types/error_types_eng.json"
    language = "English"
    feedback_df_path = f"multiagent_summary/evaluation/{language}/error_based/feedback_severe_error.csv"
    feedback_df = pd.read_csv(feedback_df_path)
    max_tokens = 3000
    # summary_refiner = Refiner(feedback_df, max_tokens, language, criteria_path)
    # summary_refiner.process_refine_summary()

    from utils import text_chunker

    out_df = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/error_based/sample_refined_summaries.csv"
    )
    out_file = f"multiagent_summary/outputs/{language}/refined_samples.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        for idx, row in out_df[:2].iterrows():
            f.write(f"*** Starting Summary {idx} ***\n\n")
            f.write(f"Original Model Summary:\n")
            f.write(f"{text_chunker(row['model_summary'])}\n\n")
            for i, criterion in enumerate(read_json_criteria(criteria_path).keys()):
                f.write(f"{criterion} {i}:\n")
                f.write(f"{row[criterion]}\n\n")
                f.write(f"Refined Summary {i}:\n")
                f.write(f"{text_chunker(row[f'refined_summary_{i}'])}\n\n")
