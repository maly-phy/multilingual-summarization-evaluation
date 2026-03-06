from utils import read_json_criteria
import pandas as pd
import os
from utils import initialize_model


class SeverityImpactScorer:
    def __init__(self, language, max_tokens, criteria_path, exclude_criteria=None):
        self.language = language
        self.max_tokens = max_tokens
        self.criteria = read_json_criteria(criteria_path)
        self.exclude_criteria = exclude_criteria

    def severity_impact_prompt(
        self, model_init, criterion, description, model_summary, meeting_transcript, row
    ):
        system_prompt = "You are an experienced linguist and expert in evaluating the impact of linguistic errors on the quality of meeting summaries."
        user_prompt = (
            "You will be given a summary for a meeting transcript, a defined error type, and a list of potential error instances extracted from the summary for the considered error type.\n"
            "You task is to rate the quality of the summary considering the actual error instances and their severity.\n"
            "Please make sure you read and understand the following instructions carefully that guide you through the task. Following is the error type you should consider:\n"
            f"{criterion}: {description['definition']}.\n\n"
            "Instructions:\n"
            "1. Read the meeting transcript carefully and identify the main topics and key points discussed.\n"
            "2. Read the summary carefully."
            "3. Read the observed error instances, the reasoning (why they are considered errors), and their severity scores (ranging from 1 for slightly severe error to 5 for extreme severity).\n"
            "Tip: You do not have to agree with these severity scores, so please critically evaluate them when rating the summary.\n"
            "4. Rate the error's impact on the quality of summary with a score ranging from 0 (no impact at all) to 5 (a very high impact) regarding this error type.\n"
            "- Please do not be too harsh in your rating, unless the errors are really an issue and impact the quality of the summary badly.\n"
            "- Please judge the summary considering only the given error type and no other kind of errors in the summary, if found.\n"
            "5. Also, provide a short reasoning explaining why you rated the summary as you did.\n"
            "6. Additionally, provide a confidence score for your rating certainty, ranging from 0 (totally unsure) to 10 (totally sure).\n"
            "Tip: Consider the whole input, i.e., the meeting transcript, the summary, and the error instances with their severity scores provided in the user's prompt to make a good decision that humans will agree on.\n\n"
            "Now, you should perform the task given the following inputs:\n"
            f"Meeting transcript: {meeting_transcript}\n"
            f"Summary: {model_summary}\n"
            f"Potential Error instances: {row[criterion]}\n"
            "Please ensure that your answer is provided strictly in **valid JSON format**, using **double quotes** for keys and values, without any extra preambles, explanations, or text outside the JSON structure. Make sure to return your answer strictly in the following format:\n"
            "{\n"
            '  "impact_score": "<0-5>",\n'
            '  "reasoning": "<your reasoning>",\n'
            '  "confidence_score": "<0-10>"\n'
            "}"
        )
        response = model_init.call_model(system_prompt, user_prompt)
        return response

    def severity_impact(self, model_init, model_summary, meeting_transcript, row):
        summary_quality = {}
        for criterion, description in self.criteria.items():
            if self.exclude_criteria and criterion in self.exclude_criteria:
                continue
            response = self.severity_impact_prompt(
                model_init,
                criterion,
                description,
                model_summary,
                meeting_transcript,
                row,
            )
            summary_quality[criterion] = response
        return summary_quality

    def process_severity_impact(self, severity_df):
        model_init = initialize_model(max_tokens=self.max_tokens)
        results = []
        for idx, row in severity_df.iterrows():
            model_summary = row["model_summary"]
            meeting_transcript = row["meeting_transcript"]
            impact_eval = self.severity_impact(
                model_init, model_summary, meeting_transcript, row
            )
            if not impact_eval:
                print(f"Failed to get impact evaluation for row {idx}, skipping...")
                continue
            if idx % 5 == 0:
                print(f"Processed {idx} rows so far...")

            results.append(
                {
                    "model_summary": model_summary,
                    "meeting_transcript": meeting_transcript,
                    **{
                        f"{criterion}": impact_eval[criterion]
                        for criterion in self.criteria.keys()
                    },
                }
            )

        output_df = pd.DataFrame(results)
        save_dir = f"multiagent_summary/evaluation/{self.language}/error_based/severity_impact.csv"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        output_df.to_csv(save_dir, index=False)
        print(f"Impact results saved to {save_dir}")


if __name__ == "__main__":
    import json

    criteria_path = "multiagent_summary/error_types/error_types_eng.json"
    language = "English"
    severity_df_path = (
        f"multiagent_summary/evaluation/{language}/error_based/error_severity.csv"
    )
    severity_df = pd.read_csv(severity_df_path)
    max_tokens = 3000
    init_severity = SeverityImpactScorer(language, max_tokens, criteria_path)
    init_severity.process_severity_impact(severity_df)

    out_df = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/error_based/error_severity.csv"
    )
    for i in range(len(list(out_df.columns))):
        if i == 2:
            print(out_df.columns[i])
