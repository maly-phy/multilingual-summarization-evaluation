import pandas as pd
import os
import time
from utils import (
    merge_data_files,
    extract_content_between_tags,
    initialize_model,
)


class MeetingEvaluator:
    def __init__(self, df, task, language, from_local_model, max_tokens):
        self.df = df
        self.task = task
        self.language = language
        self.from_local_model = from_local_model
        self.max_tokens = max_tokens
        self.criteria = {
            "Naturalness": f"How natural the conversation flows, like native {self.language} speakers (1-5)",
            "Coherence": f"How well the conversation maintains logical flow and connection (1-5)",
            "Interesting": "How engaging and content-rich the conversation is (1-5)",
            "Consistency": "How consistent each speaker's contributions are (1-5)",
        }

    def basic_llm_evaluator(self, model_init, meeting_transcript):
        basic_evaluation = {}
        for criterion, description in self.criteria.items():
            system_prompt = (
                f"You are an expert conversation analyst evaluating meeting transcripts (in {self.language}). "
                f"Evaluate the following meeting transcript thoroughly for **{criterion}**: {description}. \n"
                "- **Rating 1**: Highlights minimal or absent behaviour for each criterion.\n"
                "- **Rating 5**: Showcases strong, explicit demonstration of the behaviour.\n"
                f"Provide your step-by-step reasoning in only 1-2 sentences in {self.language}, a confidence score (0-100%), and a final score as a decimal number between 1.0 and 5.0, demonstrating the degree to which the chosen criterion is satisfied in {self.language} context. "
                "Format your response using XML tags: "
                "<reasoning>detailed step-by-step analysis</reasoning> "
                "<confidence>your confidence percentage</confidence> "
                "<score>decimal number between 1.0 and 5.0</score> "
                "You must NOT return any reasoning text with either the confidence score or the final score."
            )

            user_prompt = f"Please evaluate this meeting transcript in {self.language} for {criterion}:\n\n{meeting_transcript}"

            response = model_init.call_model(system_prompt, user_prompt)
            if not response:
                print(f"No response for criterion {criterion}, breaking...\n")
                break

            basic_evaluation[criterion] = {
                "base_reasoning": extract_content_between_tags(response, "reasoning"),
                "base_confidence": extract_content_between_tags(response, "confidence"),
                "base_score": extract_content_between_tags(response, "score"),
            }

        eval_results = {
            "Meeting": meeting_transcript,
            **{
                f"{criterion}_{key}": value
                for criterion, res in basic_evaluation.items()
                for key, value in res.items()
            },
        }

        return eval_results

    def process_meeting_evaluation(self):
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
        start_loop = time.time()
        for idx, row in df[start_idx:end_idx].iterrows():
            meeting_transcript = row["Meeting"]
            title = row["Title"]
            basic_meeting_eval = self.basic_llm_evaluator(
                model_init, meeting_transcript
            )

            if not basic_meeting_eval:
                print(f"No evaluation results for idx {idx}, continuing...\n")
                continue

            if idx % 4 == 0:
                print(f"Processing {idx} / {len(df)}\n")
                for key, results in basic_meeting_eval.items():
                    if key == "Meeting" or key == "Title":
                        continue
                    print(f"{key}: {results}\n")

            basic_meeting_eval["Title"] = title
            all_results.append(basic_meeting_eval)

        print(f"Loop time: {(time.time() - start_loop)/60:.2f} minutes\n")

        output_df = pd.DataFrame(all_results)
        output_df.to_csv(save_path, index=False)
        print(f"Evaluation results saved to {save_path}")


if __name__ == "__main__":
    language = "German"
    task = "basic_meeting_eval"
    max_tokens = 512
    from_local_model = False

    meetings_df = merge_data_files("data/fame_dataset", language)
    meeting_evaluator = MeetingEvaluator(
        meetings_df,
        task,
        language,
        from_local_model,
        max_tokens,
    )
    meeting_evaluator.process_meeting_evaluation()
