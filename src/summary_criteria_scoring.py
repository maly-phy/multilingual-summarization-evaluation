import pandas as pd
import os, sys, json
from groq import Groq
from dotenv import load_dotenv
from utils import extract_content_between_tags, initialize_model

load_dotenv()


class SummaryScorer:
    def __init__(self, df, task, language, from_local_model, max_tokens, use_facts):
        self.df = df
        self.task = task
        self.language = language
        self.from_local_model = from_local_model
        self.max_tokens = max_tokens
        self.use_facts = use_facts
        self.criteria = {
            # "Naturalness": f"How natural the conversation flows, like native {self.language} speakers (1-5)",
            "Linguistic": f"How well the summary adheres to the linguistic norms and the syntactic (grammar and structure) rules of {self.language} (1-5)",
            # "Factuality": f"How accurately and precisely the summary adheres to information mentioned in the meeting transcript for {self.language} (1-5)"
        }

    def summary_criteria_eval(
        self, model_init, model_summary, meeting_transcript=None, atomic_facts=None
    ):
        criteria_eval = {}
        for criterion, description in self.criteria.items():
            system_prompt = (
                f"You are an expert evaluating meeting summaries (in {self.language}). "
                f"Evaluate the following meeting summary thoroughly for **{criterion}**: {description}.\n"
                "- **Rating 1**: Highlights minimal or absent behaviour for each criterion.\n"
                "- **Rating 5**: Showcases strong, explicit demonstration of the behaviour.\n"
                f"Provide your step-by-step reasoning in only 1-2 sentences in {self.language}, a confidence score (0-100%), and a final score as a decimal number between 1.0 and 5.0, demonstrating the degree to which the chosen criterion is satisfied in {self.language} context. "
                "Format your response using XML tags: "
                "<reasoning>detailed step-by-step analysis</reasoning> "
                "<confidence>your confidence percentage</confidence> "
                "<score>decimal number between 1.0 and 5.0</score> "
                "You must NOT return any reasoning text with either the confidence score or the final score."
            )
            user_prompt = f"Please evaluate this meeting summary in {self.language} for {criterion}:\n\nMeeting summary:\n{model_summary}"

            if criterion == "Factuality":
                system_prompt += (
                    "Please follow these steps carefully to assess the factuality of the summary:\n\n"
                    "- First read the given summary thoroughly to get familiar with its content.\n"
                    "- Then read the given meeting transcript to identify the key points mentioned.\n"
                    "Now, check the summary against the meeting transcript to determine how accurately and precisely the summary adheres to the facts mentioned without being hallucinated (summary content not mentioned in the meeting transcript).\n"
                )
                user_prompt += f"\n\nMeeting transcript:\n{meeting_transcript}"
                if self.use_facts:
                    system_prompt = system_prompt.replace(
                        "meeting transcript", "atomic facts"
                    ).replace("Meeting transcript", "Atomic facts")
                    user_prompt += f"\n\nAtomic facts:\n{atomic_facts}"

            response = model_init.call_model(system_prompt, user_prompt)
            if not response:
                print(f"No response for criterion {criterion}, breaking...\n")
                break

            criteria_eval[criterion] = {
                "reasoning": extract_content_between_tags(response, "reasoning"),
                "confidence": extract_content_between_tags(response, "confidence"),
                "score": extract_content_between_tags(response, "score"),
            }

        eval_results = {
            "corrected_summary": model_summary,
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

        # root_filename = os.path.basename(save_path).replace(".csv", "")
        # root_filename = f"ref_summary_acc_factuality_{start_idx}_{end_idx}.csv"
        # dir_name = os.path.dirname(save_path)
        # save_path = os.path.join(dir_name, root_filename)
        dir_name = "summary_criteria_eval"
        save_path = f"evaluation/{self.language}/{dir_name}/model_summary_acc_linguistic_0_29.csv"
        print(f"save_path: {save_path}")

        if self.use_facts:
            with open(
                f"evaluation/{self.language}/atomic_facts/atomic_facts.json",
                "r",
                encoding="utf-8",
            ) as f:
                atomic_facts = json.load(f)

        all_results = []
        for idx, row in df.iterrows():
            model_summary = row["corrected_summary"]
            meeting_transcript = row["Meeting"] if not self.use_facts else None
            facts = atomic_facts.get(str(idx), "") if self.use_facts else None
            eval_results = self.summary_criteria_eval(
                model_init, model_summary, meeting_transcript, facts
            )

            if not eval_results:
                print(f"No evaluation results for idx {idx}, continuing...\n")
                continue

            if idx % 4 == 0:
                print(f"\nProcessing {idx}/{len(df)}\n")
                for key, results in eval_results.items():
                    if key == "corrected_summary":
                        continue
                    print(f"{key}: {results}\n")

            all_results.append(eval_results)

        output_df = pd.DataFrame(all_results)
        output_df.to_csv(save_path, index=False)
        print(f"Evaluation results saved to {save_path}\n")


if __name__ == "__main__":
    language = "German"
    task = "atomic_facts"
    max_tokens = 256
    from_local_model = False
    use_facts = False
    df_path = f"evaluation/{language}/{task}/corrected_summary.csv"
    df = pd.read_csv(df_path)

    summary_scorer = SummaryScorer(
        df, task, language, from_local_model, max_tokens, use_facts
    )
    summary_scorer.process_summary_scoring()
