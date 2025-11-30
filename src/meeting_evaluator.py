import sys, os
import re
from groq import Groq
from dotenv import load_dotenv
from utils import extract_content_between_tags, free_memory
import random, time
from model_handler import ModelHandler


def basic_llm_evaluator(meeting_transcript, client, model_name, max_tokens=1000):
    criteria = {
        "Naturalness": "How natural the conversation flows, like native English speakers (1-5)",
        "Coherence": "How well the conversation maintains logical flow and connection (1-5)",
        "Interesting": "How engaging and content-rich the conversation is (1-5)",
        "Consistency": "How consistent each speaker's contributions are (1-5)",
    }
    results = {}
    for criterion, description in criteria.items():
        system_prompt = (
            f"You are an expert conversation analyst evaluating meeting transcripts. "
            f"Evaluate the following meeting transcript thoroughly for **{criterion}**: {description}. \n"
            "- **Rating 1**: Highlights minimal or absent behaviour for each criterion.\n"
            "- **Rating 5**: Showcases strong, explicit demonstration of the behaviour.\n"
            "Provide your step-by-step reasoning, a confidence score (0-100%), and a final score as a decimal number between 1.0 and 5.0, demonstrating the degree to which the chosen criterion is satisfied. "
            "Format your response using XML tags: "
            "<reasoning>detailed step-by-step analysis</reasoning> "
            "<confidence_score>your confidence percentage</confidence_score> "
            "<score>decimal number between 1.0 and 5.0</score>"
        )

        user_prompt = f"Please evaluate this meeting transcript for {criterion}:\n\n{meeting_transcript}"

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = ModelHandler.call_model_with_retry(
            client=client,
            messages=message,
            model=model_name,
            max_tokens=max_tokens,
            max_attempts=6,
            base_delay=3,
        )

        response_text = response.choices[0].message.content.strip()
        print("response:\n", response_text, "\n")

        results[criterion] = {
            "base_reasoning": extract_content_between_tags(response_text, "reasoning"),
            "base_confidence": extract_content_between_tags(
                response_text, "confidence_score"
            ),
            "base_score": extract_content_between_tags(response_text, "score"),
        }

    return results
