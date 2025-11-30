import os
import pandas as pd
from groq import Groq
from utils import merge_data_files, extract_content_between_tags
from model_handler import ModelHandler
from local_model import LocalModel
from dotenv import load_dotenv

load_dotenv()


class MeetingChallengesEvaluator:
    def __init__(self, from_local_model=True):
        self.from_local_model = from_local_model
        self.challenges = {
            "Spoken language": {
                "definition": (
                    "The extent to which the transcript exhibits spoken-language features-"
                    "such as colloquialisms, jargon, false starts, or filler words-that make it "
                    "harder to parse or summarize."
                ),
                "instructions": (
                    "1. Are there noticeable filler words, false starts, or informal expressions?\n"
                    "2. Does domain-specific jargon disrupt straightforward summarization?\n"
                    "3. How challenging are these elements for generating a coherent summary?\n"
                ),
            },
            "Speaker dynamics": {
                "definition": (
                    "The challenge of correctly identifying and distinguishing between multiple speakers, "
                    "tracking who said what, and maintaining awareness of speaker roles if relevant."
                ),
                "instructions": (
                    "1. Is it difficult to keep track of speaker identities or roles?\n"
                    "2. How significantly do these dynamics affect clarity for summarization?\n"
                ),
            },
            "Coreference": {
                "definition": (
                    "The difficulty in resolving references (e.g., who or what a pronoun refers to) or clarifying "
                    "references to previous actions or decisions, so the summary remains coherent."
                ),
                "instructions": (
                    "1. Are references (e.g., pronouns like “he” or “that”) ambiguous?\n"
                    "2. Do unclear references to earlier points or events appear?\n"
                    "3. How difficult is it to track these references for summary generation?\n"
                ),
            },
            "Discourse structure": {
                "definition": (
                    "The complexity of following the meeting’s high-level flow-especially if it changes topics "
                    "or has multiple phases (planning, debate, decision)."
                ),
                "instructions": (
                    "1. Does the transcript jump between topics or phases without clear transitions?\n"
                    "2. Are meeting phases or topical shifts difficult to delineate?\n"
                    "3. How challenging is it to maintain an overview for the summary?\n"
                ),
            },
            "Contexual turn-taking": {
                "definition": (
                    "The challenge of interpreting local context as speakers take turn, including interruptions, "
                    "redundancies, and how each turn depends on previous utterances."
                ),
                "instructions": (
                    "1. Do abrupt speaker turns or interjections complicate continuity?\n"
                    "2. Are important points lost or repeated inconsistently?\n"
                    "3. How difficult is it to integrate these nuances into a coherent summary?\n"
                ),
            },
            "Implicit context": {
                "definition": (
                    "The reliance on unspoken or assumed knowledge, such as organizational history, known issues, "
                    "or prior decisions, only vaguely referenced in the meeting."
                ),
                "instructions": (
                    "1. Do participants refer to hidden topics or internal knowledge without explaining?\n"
                    "2. Is there essential background or context missing?\n"
                    "3. How much does summarization rely on understanding this hidden context?\n"
                ),
            },
            "Low information density": {
                "definition": (
                    "Segments where salient info is sparse, repeated, or only occasionally surfaced-making it "
                    "hard to find and isolate key points in a sea of less relevant content."
                ),
                "instructions": (
                    "1. Are there long stretches with minimal new information?\n"
                    "2. Is meaningful content buried under trivial or repetitive remarks?\n"
                    "3. How challenging is it to isolate crucial points for the summary?\n"
                ),
            },
        }

    def initialize_model(self):
        if self.from_local_model:
            model_name = "mistralai/Mistral-7B-Instruct-v0.3"
            model_init = LocalModel(
                model_name=model_name,
                max_new_tokens=1000,
            )
        else:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            model_name = "llama-3.1-8b-instant"
            model_init = ModelHandler(
                client=client, model_name=model_name, max_tokens=1000
            )

        meeting_language = "English"
        task = "meeting_challenges_assess"
        save_dir = os.path.join("evaluation", f"{meeting_language}", f"{task}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{model_name.split('/')[-1]}_meeting_challenges_eval.csv"
        )

        return model_init, save_path

    def evaluate_meeting_challenges(self, meeting_transcript):
        results = {}
        model_init, save_path = self.initialize_model()

        for dimension, info in self.challenges.items():
            definition = info["definition"]
            instructions = info["instructions"]

            system_prompt = (
                f"You are a meeting challenge evaluator focusing on the dimension: {dimension}.\n\n"
                f"Definition: {definition}\n"
                f"Instructions:\n{instructions}\n"
                "You must:\n"
                "- Provide a step-by-step reasoning about how the challenge is present (or not) in the transcript.\n"
                "- Give a confidence score (0-100%).\n"
                "- Provide a final numeric rating (0 to 5), following the scoring guide:\n"
                "  0: Not observed.\n"
                "  1-2: Mild presence.\n"
                "  3-4: Noticeable presence complicating summarization.\n"
                "  5: Severe presence making it very difficult to summarize.\n\n"
                "Format your final response in the following tags:\n"
                "<reasoning>...</reasoning>\n"
                "<confidence>...</confidence>\n"
                "<score>...</score>\n\n"
                "Do NOT include any text outside these tags."
            )

            user_prompt = (
                f"Meeting Transcript:\n\n{meeting_transcript}\n\n"
                "Please identify how challenging this dimension is for summarization."
            )

            response = model_init.call_model(system_prompt, user_prompt)

            if not response:
                print(f"No response for dimension {dimension}, breaking...\n")
                break

            reasoning = extract_content_between_tags(response, "reasoning")
            confidence = extract_content_between_tags(response, "confidence_score")
            score = extract_content_between_tags(response, "score")

            results[dimension] = {
                "reasoning": reasoning,
                "confidence": confidence,
                "score": score,
            }
        return results, save_path


def process_meeting_challenges(input_df):
    challenge_evaluator = MeetingChallengesEvaluator()
    output_df = pd.DataFrame()

    for idx, row in input_df[:1].iterrows():
        print(f"Processing meeting {idx} / {len(input_df)}\n")
        title = row["Title"]
        meeting_transcript = row["Meeting"]
        challenge_scores, save_path = challenge_evaluator.evaluate_meeting_challenges(
            meeting_transcript
        )

        row_results = {"title": title, "meeting_transcript": [meeting_transcript]}
        for dimension, outcomes in challenge_scores.items():
            dim_key = dimension.replace(" ", "_")
            row_results[f"{dim_key}_Score"] = outcomes["score"]
            row_results[f"{dim_key}_Confidence"] = outcomes["confidence"]
            row_results[f"{dim_key}_Reasoning"] = outcomes["reasoning"]

        results_to_save = {k: [v] for k, v in row_results.items()}
        output_df = pd.concat(
            [
                output_df,
                pd.DataFrame({**results_to_save}),
            ],
            axis=0,
            ignore_index=True,
        )
        output_df = output_df.reset_index(drop=True)
        output_df.to_csv(save_path, header=True, index=False)
        print(f"Challenge evaluation results saved to {save_path}\n")

    return output_df


if __name__ == "__main__":
    eng_meetings_df = merge_data_files("data/fame_dataset", "English")
    challenges_results_df = process_meeting_challenges(eng_meetings_df)
