import os
import pandas as pd
from utils import merge_data_files, extract_content_between_tags, initialize_model
import time


class MeetingChallengesEvaluator:
    def __init__(self, input_df, task, meeting_language, from_local_model):
        self.input_df = input_df
        self.task = task
        self.meeting_language = meeting_language
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

    def evaluate_meeting_challenges(self, model_init, meeting_transcript):
        results = {}

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
                "Do NOT include any text outside these tags.\n"
                "**Strictly Important:**\n"
                "- You MUST NOT include any reasoning text with either the confidence score or the final score for any dimension."
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

        for dimension, outcome in results.items():
            print(
                f"{dimension}: {outcome['score']}, Confidence: {outcome['confidence']}\n"
            )

        return results

    def process_meeting_challenges(self):
        input_df = self.input_df[:2]
        start_idx = input_df.index[0]
        end_idx = input_df.index[-1]
        model_init, save_path = initialize_model(
            self.task, self.meeting_language, self.from_local_model
        )
        root_filename = os.path.basename(save_path).replace(".csv", "")
        root_filename += f"_{start_idx}_{end_idx}_clean.csv"
        dir_name = os.path.dirname(save_path)
        save_path = os.path.join(dir_name, root_filename)
        print(f"save_path: {save_path}")

        all_results = []
        start_loop = time.time()
        for idx, row in input_df[start_idx:end_idx].iterrows():
            print(f"Processing meeting {idx} / {len(input_df)}\n")
            title = row["Title"]
            meeting_transcript = row["Meeting"]
            challenge_scores = self.evaluate_meeting_challenges(
                model_init, meeting_transcript
            )
            if not challenge_scores:
                print(f"No challenge scores for idx {idx}, skipping...\n")
                continue

            row_results = {"title": title, "meeting_transcript": meeting_transcript}
            for dimension, outcomes in challenge_scores.items():
                dim_key = dimension.replace(" ", "_")
                row_results[f"{dim_key}_Score"] = outcomes["score"]
                row_results[f"{dim_key}_Confidence"] = outcomes["confidence"]
                row_results[f"{dim_key}_Reasoning"] = outcomes["reasoning"]

            all_results.append(row_results)

        print(f"Loop time: {(time.time() - start_loop)/60:.2f} minutes\n")
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(save_path, header=True, index=False)
        print(f"Challenge evaluation results saved to {save_path}\n")


if __name__ == "__main__":
    eng_meetings_df = merge_data_files("data/fame_dataset", "English")
    meeting_evaluator = MeetingChallengesEvaluator(
        task="meeting_challenges_assess",
        meeting_language="English",
        from_local_model=False,
        input_df=eng_meetings_df,
    )
    meeting_evaluator.process_meeting_challenges()
