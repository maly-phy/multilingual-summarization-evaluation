import pandas as pd
import os
from utils import read_json_criteria, initialize_model


class FeedbackSystem:
    def __init__(self, error_df, language, max_tokens, criteria_path):
        self.error_df = error_df
        self.language = language
        self.max_tokens = max_tokens
        self.criteria = read_json_criteria(criteria_path)

    def get_feedback(self, model_init, model_summary, meeting_transcript, idx):
        system_prompt = "You are an expert in analyzing the linguistic errors in meeting summaries. You help with providing feedback to fix the errors and improve the quality of the summaries."
        collect_feedback = {}
        for criterion, description in self.criteria.items():
            user_prompt = (
                "You will be given a summary for a meeting transcript, a defined error type, a list of potential error instances extracted from the summary for the considered error type, accompanied with their severity scores.\n"
                "Your task is to provide feedback and suggestions on how to correct the error instances found in the summary for the considered error type. What changes would you propose to reduce the error's impact on the quality of the summary?\n"
                "Please make sure you read and understand the following instructions carefully that guide you through the task. Following is the error type you should consider:\n"
                f"{criterion}: {description['definition']}.\n\n"
                "Instructions:\n"
                "1. Read the meeting transcript carefully and identify the main topics and key points discussed.\n"
                "2. Read the summary carefully.\n"
                "3. Consider the list of observed error instances, the chain-of-thought reasoning of why they are considered errors, and the severity of the error instances which ranges from 1 (low severity) to 5 (high severity).\n"
                "5. Provide your suggestions or changes that should be made to correct each error instance to end up with no or less impact of the error type on the quality of the summary.\n"
                "Please pay more attention to the mid-to-high severe errors (3-5 severity score) that worsen the quality of the summary.\n"
                "6. Write down a short reasoning explaining why you consider these changes are effective towards improving the summary.\n"
                "7. Additionally, provide a confidence score for your suggestions certainty, ranging from 0 (totally unsure) to 10 (totally sure).\n"
                "Tip: Consider the whole input, i.e., the meeting transcript, the summary, and the error instances provided in the user's prompt to make a good decision that humans will agree on.\n\n"
                "Now, you should perform the task, given the following inputs:\n"
                f"Meeting transcript: {meeting_transcript}\n"
                f"Summary: {model_summary}\n"
                f"Error instances: {self.error_df.iloc[idx][criterion]}\n"
                "Please ensure that your answer is provided strictly in **valid JSON format**, using **double quotes** for keys and values, without any extra preambles, explanations, or text outside the JSON structure. Make sure to return your answer strictly in the following format:\n"
                "[{\n"
                '  "instance": "<error instance>",\n'
                '  "severity_score": "<1-5>",\n'
                '  "feedback": "<your suggestions to correct the error instances for a high quality summary>",\n'
                '  "reasoning": "<chain-of-thought reasoning>",\n'
                '  "confidence_score": "<0-10>"\n'
                "},\n"
                "{<same for instance 2},...{<same for instance n>}]}"
            )
            response = model_init.call_model(system_prompt, user_prompt)
            if not response:
                print(f"Failed to get a valid response for criterion: {criterion}")
                break
            collect_feedback[criterion] = response

        return collect_feedback

    def process_feedback(self):
        model_init = initialize_model(max_tokens=self.max_tokens)
        results = []
        for idx, row in self.error_df.iterrows():
            model_summary = row["model_summary"]
            meeting_transcript = row["meeting_transcript"]
            feedback = self.get_feedback(
                model_init, model_summary, meeting_transcript, idx
            )
            if not feedback:
                print(f"Failed to get feedback for row {idx}, skipping...")
                continue
            if idx % 5 == 0:
                print(f"Processed {idx} rows so far...")

            results.append(
                {
                    "model_summary": model_summary,
                    "meeting_transcript": meeting_transcript,
                    **{
                        f"{criterion}": feedback[criterion]
                        for criterion in self.criteria.keys()
                    },
                }
            )

        output_df = pd.DataFrame(results)
        save_dir = f"multiagent_summary/evaluation/{self.language}/error_based/feedback_severe_error.csv"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        output_df.to_csv(save_dir, index=False)
        print(f"Feedback results saved to {save_dir}")


if __name__ == "__main__":
    criteria_path = "multiagent_summary/error_types/error_types_eng.json"
    language = "English"
    error_df_path = (
        f"multiagent_summary/evaluation/{language}/error_based/error_severity.csv"
    )
    error_df = pd.read_csv(error_df_path)
    max_tokens = 3000
    feedback = FeedbackSystem(error_df, language, max_tokens, criteria_path)
    feedback.process_feedback()
