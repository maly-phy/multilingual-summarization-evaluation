from utils import read_json_criteria
import pandas as pd
import os
from utils import initialize_model


class SeverityScorer:
    def __init__(self, df, language, max_tokens, criteria_path):
        self.df = df
        self.language = language
        self.max_tokens = max_tokens
        self.criteria = read_json_criteria(criteria_path)

    def init_severity_eval(self, model_init, model_summary, meeting_transcript):
        system_prompt = "You are an experienced linguist and expert in identifying the error types present in meeting summaries and evaluating their severity."
        error_severity_eval = {}
        for criterion, description in self.criteria.items():
            user_prompt = (
                "You will be given a summary for a meeting transcript, a defined error type and two examples of the error.\n"
                "Your tasks:\n"
                "- Determine if the summary contains the error type.\n"
                "- If the error exists, then find the instances, phrases or words in the summary that correspond to the defined error type.\n"
                "- Rate the severity of the potential error instances.\n"
                "Please make sure you read and understand the following instructions carefully that guide you through the tasks. Following is the error type you should consider:\n"
                f"{criterion}: {description['definition']}.\n\n"
                "Instructions:\n"
                "1. Read the meeting transcript carefully and identify the main topics and key points discussed.\n"
                "2. Read the summary and compare if it contains instances of the described error type.\n"
                "3. Rate the summary based on the existence of the error type with yes when at least one instance of the error type is observed or no if the summary does not exhibit the error type.\n"
                "4. If the error type exists, then write down every instance that is part of this error type, and specify its location (beginning, middle, or end of the summary).\n"
                "- Please focus only on error instances that are really an issue and impact the quality of the summary badly.\n"
                "- Please consider only the previously defined error type and no other kind of errors in the summary, if found.\n"
                "5. For every instance, write down a short reasoning thinking step-by-step of why this instance could be an error.\n"
                "6. Rate the severity of the potential error instances, ranging from 1 (low severity) to 5 (high severity). Also provide a confidence score for your certainty, ranging from 0 (totally unsure) to 10 (totally sure).\n"
                "Tip: Consider the whole input, i.e., the meeting transcript and the summary, provided in the user's prompt to make a good decision that humans will agree on.\n"
                "Below are two examples demonstrating the different severity levels of the previously described error type. Please learn from these examples the error pattern and how the rating works.\n"
                f"Low severity example: {description['example']['low_severity']}\n"
                f"High severity example: {description['example']['high_severity']}\n\n"
                f"You should now perform the error search on the following summary: {model_summary}\n"
                f"The original meeting transcript for look up: {meeting_transcript}\n"
                "Please ensure that each instance is provided strictly in **valid JSON format**, using **double quotes** for keys and values, and no additional text outside the JSON structure. Return your answer only in the following format:\n"
                "[{\n"
                '  "error_exists": <yes or no depending on you decision>,\n'
                '  "instance": "<text passage, sentence or words from summary, if error exists else "">",\n'
                '  "location": "<beginning, middle or end of the summary if error exists else "">",\n'
                '  "reasoning": "<chain-of-thought reasoning if error exists else "">",\n'
                '  "severity_score": "<1-5 if error exists else "">",\n'
                '  "confidence_score": "<0-10 if error exists else "">"\n'
                "},\n"
                "{<same for instance 2>},...{<same for instance n>}]\n"
                "Make sure the output is strictly valid JSON, with no preambles, extra explanations, or text outside the JSON structure."
            )
            response = model_init.call_model(system_prompt, user_prompt)
            if not response:
                print(f"Failed to get a valid response for criterion: {criterion}")
                break
            error_severity_eval[criterion] = response

        return error_severity_eval

    def process_error_severity(self):
        model_init = initialize_model(max_tokens=self.max_tokens)
        results = []
        for idx, row in self.df.iterrows():
            model_summary = row["model_factual_summary"]
            meeting_transcript = row["Meeting"]
            severity_eval = self.init_severity_eval(
                model_init, model_summary, meeting_transcript
            )
            if not severity_eval:
                print(f"Failed to get severity evaluation for row {idx}, skipping...")
                continue
            if idx % 5 == 0:
                print(f"Processed {idx} rows so far...")

            results.append(
                {
                    "model_summary": model_summary,
                    "meeting_transcript": meeting_transcript,
                    **{
                        f"{criterion}": severity_eval[criterion]
                        for criterion in self.criteria.keys()
                    },
                }
            )

        output_df = pd.DataFrame(results)
        save_dir = f"multiagent_summary/evaluation/{self.language}/less_strict_error/error_severity.csv"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        output_df.to_csv(save_dir, index=False)
        print(f"Severity results saved to {save_dir}")


if __name__ == "__main__":
    import json

    criteria_path = "multiagent_summary/error_types/error_types_eng.json"
    language = "English"
    df_path = f"evaluation/{language}/atomic_facts/corrected_summary.csv"
    df = pd.read_csv(df_path)
    max_tokens = 3000
    # init_severity = SeverityScorer(df, language, max_tokens, criteria_path)
    # init_severity.process_error_severity()

    out_df = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/less_strict_error/error_severity.csv"
    )
    with open(f"multiagent_summary/outputs/{language}/error_severity.json", "w") as f:
        f.write(json.dumps(out_df.iloc[1]["Omission"], indent=4))
