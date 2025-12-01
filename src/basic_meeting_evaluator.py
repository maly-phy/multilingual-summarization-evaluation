import pandas as pd
import os
import time
from utils import (
    merge_data_files,
    extract_content_between_tags,
    initialize_model,
)


class MeetingEvaluator:
    def __init__(self, input_df, task, meeting_language, from_local_model):
        self.input_df = input_df
        self.task = task
        self.meeting_language = meeting_language
        self.from_local_model = from_local_model
        self.criteria = {
            "Naturalness": "How natural the conversation flows, like native English speakers (1-5)",
            "Coherence": "How well the conversation maintains logical flow and connection (1-5)",
            "Interesting": "How engaging and content-rich the conversation is (1-5)",
            "Consistency": "How consistent each speaker's contributions are (1-5)",
        }

    def basic_llm_evaluator(self, model_init, meeting_transcript):
        basic_evaluation = {}
        for criterion, description in self.criteria.items():
            system_prompt = (
                f"You are an expert conversation analyst evaluating meeting transcripts. "
                f"Evaluate the following meeting transcript thoroughly for **{criterion}**: {description}. \n"
                "- **Rating 1**: Highlights minimal or absent behaviour for each criterion.\n"
                "- **Rating 5**: Showcases strong, explicit demonstration of the behaviour.\n"
                "Provide your step-by-step reasoning, a confidence score (0-100%), and a final score as a decimal number between 1.0 and 5.0, demonstrating the degree to which the chosen criterion is satisfied. "
                "Format your response using XML tags: "
                "<reasoning>detailed step-by-step analysis</reasoning> "
                "<confidence_score>your confidence percentage</confidence_score> "
                "<score>decimal number between 1.0 and 5.0</score> "
                "You must NOT return any reasoning text with either the confidence score or the final score."
            )

            user_prompt = f"Please evaluate this meeting transcript for {criterion}:\n\n{meeting_transcript}"

            response = model_init.call_model(system_prompt, user_prompt)
            if not response:
                print(f"No response for criterion {criterion}, breaking...\n")
                break

            basic_evaluation[criterion] = {
                "base_reasoning": extract_content_between_tags(response, "reasoning"),
                "base_confidence": extract_content_between_tags(
                    response, "confidence_score"
                ),
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
        for criterion, results in basic_evaluation.items():
            print(
                f"{criterion}: {results['base_score']}, Confidence: {results['base_confidence']}\n"
            )

        return eval_results

    def process_meeting_evaluation(self):
        input_df = self.input_df[:20]
        start_idx = input_df.index[0]
        end_idx = input_df.index[-1]
        model_init, save_path = initialize_model(
            self.task, self.meeting_language, self.from_local_model
        )

        root_filename = os.path.basename(save_path).replace(".csv", "")
        root_filename += f"_{start_idx}_{end_idx}.csv"
        dir_name = os.path.dirname(save_path)
        save_path = os.path.join(dir_name, root_filename)
        print(f"save_path: {save_path}")

        all_results = []
        start_loop = time.time()
        for idx, row in input_df[start_idx:end_idx].iterrows():
            print(f"Processing meeting {idx} / {len(input_df)}\n")
            meeting_transcript = row["Meeting"]
            title = row["Title"]
            basic_meeting_eval = self.basic_llm_evaluator(
                model_init, meeting_transcript
            )

            if not basic_meeting_eval:
                print(f"No evaluation results for idx {idx}, continuing...\n")
                continue

            basic_meeting_eval["Title"] = title
            all_results.append(basic_meeting_eval)

        print(f"Loop time: {(time.time() - start_loop)/60:.2f} minutes\n")

        output_df = pd.DataFrame(all_results)
        output_df.to_csv(save_path, index=False)
        print(f"Evaluation results saved to {save_path}")


if __name__ == "__main__":
    eng_meetings_df = merge_data_files("data/fame_dataset", "English")
    meeting_evaluator = MeetingEvaluator(
        input_df=eng_meetings_df,
        task="basic_meeting_eval",
        meeting_language="English",
        from_local_model=False,
    )
    meeting_evaluator.process_meeting_evaluation()
