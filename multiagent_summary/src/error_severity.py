from utils import read_json_criteria
import pandas as pd
import os
from utils import initialize_model


class SeverityScorer:
    def __init__(self, df, language, max_tokens, criteria_path, exclude_criteria=None):
        self.df = df
        self.language = language
        self.max_tokens = max_tokens
        self.criteria = read_json_criteria(criteria_path)
        self.exclude_criteria = exclude_criteria

    def severity_prompt(
        self, model_init, criterion, description, model_summary, meeting_transcript
    ):
        system_prompt = "You are an experienced linguist and expert in identifying the error types present in meeting summaries and evaluating their severity."
        user_prompt = (
            "You will be given a summary for a meeting transcript, a defined error type and two examples of the error.\n"
            "Your tasks:\n"
            "- Search the summary to find the instances, phrases or words that are part of the defined error type.\n"
            "- Rate the severity of the potential error instances.\n"
            "Please make sure you read and understand the following instructions carefully that guide you through the tasks. Following is the error type you should consider:\n"
            f"{criterion}: {description['definition']}.\n\n"
            "Instructions:\n"
            "1. Read the meeting transcript carefully and identify the main topics and key points discussed. Please keep the transcript open while performing the tasks, and refer to it whenever needed.\n"
            "2. Read the summary and check if it contains instances of the described error type. Refer to the meeting transcript for sanity checking the error instances and to clarify any doubts, if necessary.\n"
            "3. Write down every instance that is part of this error type, and specify its location (beginning, middle, or end of the summary).\n"
            "Please consider only the previously defined error type and no other kind of errors in the summary, if found.\n"
            "5. Write down a short reasoning thinking step-by-step of why this instance could be an error.\n"
            "6. Rate the severity of the instances that already show the error type, ranging from 1 (low severity) to 5 (high severity). Also provide a confidence score for your certainty, ranging from 0 (totally unsure) to 10 (totally sure).\n"
            "Please do not be too harsh in your rating, unless the instance is clearly problematic and impacts the quality of the summary badly, otherwise be a friendly reviewer.\n"
            "Tip: Consider the whole input, i.e., the meeting transcript and the summary, provided in the user's prompt to make a good decision that humans will agree on.\n"
            "Below are two examples demonstrating the different severity levels of the previously described error type. Please learn from these examples the error pattern and how the rating works.\n"
            f"Low severity example: {description['example']['low_severity']}\n"
            f"High severity example: {description['example']['high_severity']}\n\n"
            f"Now, you should perform the tasks, given the following inputs:\n"
            f"Meeting transcript: {meeting_transcript}\n"
            f"Summary: {model_summary}\n"
            "Please ensure that each instance is provided strictly in **valid JSON format**, using **double quotes** for keys and values, without extra preambles, explanations, or text outside the JSON structure. Return your answer only in the following format:\n"
            "[{\n"
            '  "instance": "<text passage, sentence or words from summary>",\n'
            '  "location": "<beginning, middle or end of the summary>",\n'
            '  "reasoning": "<chain-of-thought reasoning>",\n'
            '  "severity_score": "<1-5>",\n'
            '  "confidence_score": "<0-10>"\n'
            "},\n"
            "{<same for instance 2>},...{<same for instance n>}]"
        )
        response = model_init.call_model(system_prompt, user_prompt)
        return response

    def init_severity_eval(self, model_init, model_summary, meeting_transcript):
        error_severity_eval = {}
        for criterion, description in self.criteria.items():
            if self.exclude_criteria and criterion in self.exclude_criteria:
                continue
            response = self.severity_prompt(
                model_init, criterion, description, model_summary, meeting_transcript
            )
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

            if idx > 3:
                break
            results.append(
                {
                    "model_summary": model_summary,
                    **{
                        f"{criterion}": severity_eval[criterion]
                        for criterion in self.criteria.keys()
                    },
                }
            )

        output_df = pd.DataFrame(results)
        save_dir = f"multiagent_summary/evaluation/{self.language}/test_samples/error_severity.csv"
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
    exclude_criteria = None
    init_severity = SeverityScorer(
        df, language, max_tokens, criteria_path, exclude_criteria
    )
    init_severity.process_error_severity()

    from utils import text_chunker

    out_df = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/test_samples/error_severity.csv"
    )
    out_file = f"multiagent_summary/outputs/{language}/severity_samples2.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        for idx, row in out_df.iterrows():
            f.write(f"*** Starting Summary {idx} ***\n\n")
            f.write(f"{text_chunker(row['model_summary'])}\n")
            for i, criterion in enumerate(read_json_criteria(criteria_path).keys()):
                f.write(f"{criterion} {i}:\n")
                f.write(f"{row[criterion]}\n\n")
