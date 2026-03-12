import pandas as pd
import os
from utils import initialize_model
import ast
from utils import read_json_criteria
import time


class MultiQualityJudge:
    def __init__(
        self, df, max_tokens, language, criteria_path, exclude_criteria, rounds
    ):
        self.df = df
        self.max_tokens = max_tokens
        self.language = language
        self.exclude_criteria = exclude_criteria
        self.criteria = read_json_criteria(criteria_path)
        self.rounds = rounds
        self.out_df = pd.DataFrame()

    def multi_quality_prompt(self, model_init, meeting_transcript, refined_summaries):
        system_prompt = "You are an experienced linguist and expert in evaluating the quality of meeting summaries based on pre-defined criteria.\n"
        user_prompt = (
            "You will be given multiple summaries for a meeting transcript, and criteria that help you judge the quality of these summaries.\n"
            "Your task is to evaluate the quality of each summary based on the provided criteria. How much each summary suffices and entails the criteria?\n"
            "Please make sure you read and understand the following instructions carefully that guide you through the task.\n"
            "1. Please consider the following judgement criteria:\n"
            "- The summary should not contain any content-wise redundant information, that does not aid the understanding or contextualization.\n"
            "- The summary should be coherent, maintain logical flow, relevance, and clarity within a sentence and across sentences.\n"
            "- The summary should use appropriate language with correct and grammatical use. Language should not be ambiguous or unclear.\n"
            "- The summary should not omit relevant content. Neither should content be completely absent nor relevant details be missing.\n"
            "2. Read the meeting transcript carefully and identify the main topics and key points discussed. Please keep the transcript open while performing the task, and refer to it whenever needed.\n"
            "3. Read the summaries carefully, and observe how each of them addresses the criteria provided.\n"
            "4. Rate the quality of summaries based on the criteria provided, in a range from 1 (poor quality) to 10 (excellent quality).\n"
            "5. Write a short explanation of why you rated the summaries as you did. Therefore, use chain-of-thought and think step-by-step.\n"
            "6. Additionally, provide a confidence score for your rating certainty, in a range from 0 (totally unsure) to 10 (totally sure).\n"
            "Now, you should perform the task, given the following inputs:\n"
            f"Meeting transcript: {meeting_transcript}\n"
            f"Summary 1: {refined_summaries[0]}\n"
            f"Summary 2: {refined_summaries[1]}\n"
            f"Summary 3: {refined_summaries[2]}\n"
            f"Summary 4: {refined_summaries[3]}\n"
            "Please return your answer strictly in a **valid JSON format**, using **double quotes** for keys and values, without extra preambles, explanations, or text outside the JSON structure. Make sure to return your answer strictly in the following format:\n"
            "[{\n"
            '  "llm_quality_score": "<1-10>",\n'
            '  "reasoning": "<chain-of-thought reasoning>",\n'
            '  "confidence": "<0-10>"\n'
            "},\n"
            "{\n"
            '  "llm_quality_score": "<1-10>",\n'
            '  "reasoning": "<chain-of-thought reasoning>",\n'
            '  "confidence": "<0-10>"\n'
            "},\n"
            "{\n"
            '  "llm_quality_score": "<1-10>",\n'
            '  "reasoning": "<chain-of-thought reasoning>",\n'
            '  "confidence": "<0-10>"\n'
            "},\n"
            "{\n"
            '  "llm_quality_score": "<1-10>",\n'
            '  "reasoning": "<chain-of-thought reasoning>",\n'
            '  "confidence": "<0-10>"\n'
            "}]"
        )
        response = model_init.call_model(system_prompt, user_prompt)
        return response

    def collect_refined_summaries(self, idx):
        refined_summaries = []
        for criterion in self.criteria:
            if self.exclude_criteria and criterion in self.exclude_criteria:
                continue
            refined_summary = self.df.iloc[idx][f"refined_{criterion}"]
            refined_summaries.append(refined_summary)
        return refined_summaries

    def process_multi_quality(self):
        model_init = initialize_model(max_tokens=self.max_tokens)
        for j in range(self.rounds):
            print(f"*** Starting iteration {j}/{self.rounds} ***\n")
            start_round_time = time.time()
            for idx, row in self.df.iterrows():
                print(f"Processing row {idx}/{len(self.df)}\n")
                meeting_transcript = row["meeting_transcript"]
                refined_summaries = self.collect_refined_summaries(idx)
                llm_quality_score = self.multi_quality_prompt(
                    model_init, meeting_transcript, refined_summaries
                )
                llm_quality_score = ast.literal_eval(llm_quality_score)
                for i, criterion in enumerate(self.criteria):
                    if self.exclude_criteria and criterion in self.exclude_criteria:
                        continue
                    print(
                        f"Processing criterion {i}/{len(self.criteria)} | summary {idx}/{len(self.df)} | iteration {j}\n"
                    )
                    self.out_df.at[f"{j}_{idx}", f"refined_{criterion}_quality"] = [
                        llm_quality_score[i]
                    ]

                avg_quality_score = sum(
                    float(llm_quality_score[i]["llm_quality_score"])
                    for i in range(len(llm_quality_score))
                ) / len(llm_quality_score)
                self.out_df.at[f"{j}_{idx}", "refined_llm_quality"] = avg_quality_score

            print(
                f"Round {j} completed in {(time.time() - start_round_time) / 60} minutes\n\n"
            )
        save_dir = f"multiagent_summary/evaluation/{self.language}/agent_loop/refined_llm_quality.csv"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        self.out_df.to_csv(save_dir, index=True)
        print(f"Quality results saved to {save_dir}")


if __name__ == "__main__":
    language = "English"
    criteria_path = "multiagent_summary/error_types/error_types_eng.json"
    df_path = f"multiagent_summary/evaluation/{language}/agent_loop/agent_iter.csv"
    df = pd.read_csv(df_path)
    max_tokens = 3000
    exclude_criteria = ["Hallucination", "Structure", "Irrelevance"]
    rounds = 5
    quality_judge = MultiQualityJudge(
        df, max_tokens, language, criteria_path, exclude_criteria, rounds
    )
    quality_judge.process_multi_quality()
