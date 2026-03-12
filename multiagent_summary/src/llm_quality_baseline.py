import pandas as pd
import os
from utils import initialize_model
import ast


class LLMQualityJudge:
    def __init__(self, df, max_tokens, language):
        self.df = df
        self.max_tokens = max_tokens
        self.language = language

    def llm_quality_prompt(
        self, model_init, meeting_transcript, model_summary, ref_summary
    ):
        system_prompt = "You are an experienced linguist and expert in evaluating the quality of meeting summaries based on pre-defined criteria.\n"
        user_prompt = (
            "You will be given two summaries for a meeting transcript, and criteria that help you judge the quality of these summaries.\n"
            "Your task is to evaluate the quality of each summary based on the provided criteria. How much each summary suffices and entails the criteria?\n"
            "Please make sure you read and understand the following instructions carefully that guide you through the task.\n"
            "1. Please consider the following judgement criteria:\n"
            "- The summary should not contain any content-wise redundant information, that does not aid the understanding or contextualization.\n"
            "- The summary should be coherent, maintain logical flow, relevance, and clarity within a sentence and across sentences.\n"
            "- The summary should use appropriate language with correct and grammatical use. Language should not be ambiguous or unclear.\n"
            "- The summary should not omit relevant content. Neither should content be completely absent nor relevant details be missing.\n"
            "- The summary should not contain hallucinated content. This includes the addition of new information that is not present in the original meeting transcript as well as changing details.\n"
            "- The summary should maintain the logical and temporal structure and misplace topics or events.\n"
            "- The summary should not contain irrelevant information, but focus on what is important.\n"
            "2. Read the meeting transcript carefully and identify the main topics and key points discussed. Please keep the transcript open while performing the task, and refer to it whenever needed.\n"
            "3. Read the summaries carefully, and observe how each of them addresses the criteria provided.\n"
            "4. Rate the quality of summaries based on the criteria provided, in a range from 1 (poor quality) to 10 (excellent quality).\n"
            "5. Write a short explanation of why you rated the summaries as you did. Therefore, use chain-of-thought and think step-by-step.\n"
            "6. Additionally, provide a confidence score for your rating certainty, in a range from 0 (totally unsure) to 10 (totally sure).\n"
            "Now, you should perform the task, given the following inputs:\n"
            f"Meeting transcript: {meeting_transcript}\n"
            f"Summary 1: {model_summary}\n"
            f"Summary 2: {ref_summary}\n"
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
            "}]"
        )
        response = model_init.call_model(system_prompt, user_prompt)
        return response

    def process_llm_quality(self):
        model_init = initialize_model(max_tokens=self.max_tokens)
        results = []
        for idx, row in self.df.iterrows():
            print(f"Processing row {idx}/{len(self.df)}\n")
            model_summary = row["model_factual_summary"]
            ref_summary = row["corrected_summary"]
            meeting_transcript = row["Meeting"]
            llm_quality_score = self.llm_quality_prompt(
                model_init, meeting_transcript, model_summary, ref_summary
            )
            llm_quality_score = ast.literal_eval(llm_quality_score)
            results.append(
                {
                    "model_summary_quality": [llm_quality_score[0]],
                    "ref_summary_quality": [llm_quality_score[1]],
                }
            )

        out_df = pd.DataFrame(results)
        save_dir = f"multiagent_summary/evaluation/{self.language}/agent_loop/quality_baseline2.csv"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        out_df.to_csv(save_dir, index=False)
        print(f"Quality results saved to {save_dir}")


if __name__ == "__main__":
    language = "English"
    df_path = f"evaluation/{language}/atomic_facts/corrected_summary.csv"
    df = pd.read_csv(df_path)
    max_tokens = 3000
    quality_judge = LLMQualityJudge(df, max_tokens, language)
    quality_judge.process_llm_quality()
