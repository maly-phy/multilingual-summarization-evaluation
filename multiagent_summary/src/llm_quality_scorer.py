import pandas as pd
import os


class LLMQualityJudge:
    def __init__(self, refined_df, max_tokens, language):
        self.refined_df = refined_df
        self.max_tokens = max_tokens
        self.language = language
        self.system_prompt = "You are an experienced linguist and expert in evaluating the quality of meeting summaries based on pre-defined criteria.\n"

        def single_summary_prompt(self, idx):
            user_prompt = (
                "You will be given a summary for a meeting transcript, and criteria that help you judge the quality of summary.\n"
                "Your task is to evaluate the quality of the summary based on the provided criteria.\n"
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
                "3. Read the summary carefully, and observe how it addresses the criteria provided.\n"
                "4. Rate the quality of summary based on the criteria provided, in a range from 1 (poor quality) to 10 (excellent quality).\n"
                "5. Write a short explanation of why you rated the summary as you did. Therefore, use chain-of-thought and think step-by-step.\n"
                "6. Additionally, provide a confidence score for your rating certainty, in a range from 0 (totally unsure) to 10 (totally sure).\n"
                "Now, you should perform the task, given the following inputs:\n"
                f"Meeting transcript: {self.refined_df.iloc[idx]['meeting_transcript']}\n"
                f"Summary: {self.refined_df.iloc[idx]['summary']}\n"
                "Please return your answer strictly in a **valid JSON format**, using **double quotes** for keys and values, without extra preambles, explanations, or text outside the JSON structure. Please use the following format:\n"
                "{\n"
                '  "llm_quality_score": "<1-10>",\n'
                '  "reasoning": "<chain-of-thought reasoning>",\n'
                '  "confidence": "<0-10>"\n'
                "}"
            )

        def multiple_summary_prompt(self, idx):
            user_prompt = (
                "You will be given multiple summaries for a meeting transcript, and criteria that help you judge the quality of these summaries.\n"
                "Your task is to evaluate the quality of each summary based on the provided criteria. How much the summary suffice and entail the criteria?\n"
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
                f"Meeting transcript: {self.refined_df.iloc[idx]['meeting_transcript']}\n"
                f"Summary 1: {self.refined_df.iloc[idx]['summary_1']}\n"
                f"Summary 2: {self.refined_df.iloc[idx]['summary_2']}\n"
                f"Summary 3: {self.refined_df.iloc[idx]['summary_3']}\n"
                "Please return your answer strictly in a **valid JSON format**, using **double quotes** for keys and values, without extra preambles, explanations, or text outside the JSON structure. Please use the following format:\n"
                "[{\n"
                '  "llm_quality_score": "<1-10>",\n'
                '  "reasoning": "<chain-of-thought reasoning>",\n'
                '  "confidence": "<0-10>"\n'
                "}\n"
                "{<same for summary 2>},{<same for summary 3>}]"
            )
