import pandas as pd
import os, ast
from groq import Groq
import torch
from dotenv import load_dotenv
from utils import free_memory, merge_data_files
from meeting_evaluator import basic_llm_evaluator

load_dotenv()

eng_meetings_df = merge_data_files("data/fame_dataset", "English")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model_name = "llama-3.1-8b-instant"
meeting_language = "English"
device = "cuda" if torch.cuda.is_available() else "cpu"
task = "basic_meeting_assess"
save_dir = os.path.join("evaluation", f"{meeting_language}", f"{task}")
os.makedirs(save_dir, exist_ok=True)
start_from = 8
save_path = os.path.join(
    save_dir, f"{model_name}_basic_llm_meeting_eval_{start_from}.csv"
)


def main():

    output_df = pd.DataFrame()
    for idx, row in eng_meetings_df[start_from:].iterrows():
        print(f"Evaluating meeting {idx} / {len(eng_meetings_df)}\n")
        meeting_transcript = row["Meeting"]
        basic_meeting_eval = basic_llm_evaluator(meeting_transcript, client, model_name)

        if not basic_meeting_eval:
            print(f"No evaluation results for meeting {idx}, breaking...\n")
            break

        result_dict = {}
        for criterion, results in basic_meeting_eval.items():
            for key, value in results.items():
                result_dict[f"{criterion}_{key}"] = value

        output_df = pd.concat(
            [
                output_df,
                pd.DataFrame(
                    {
                        "meeting_transcript": [[meeting_transcript]],
                        **{k: [v] for k, v in result_dict.items()},
                    }
                ),
            ],
            axis=0,
            ignore_index=True,
        )

        output_df = output_df.reset_index(drop=True)
        output_df.to_csv(save_path, header=True, index=False)
        print(f"Evaluation results saved to {save_path}")

    print("Evaluation complete.")


if __name__ == "__main__":
    # main()
    df = pd.read_csv(save_path)
    df["meeting_transcript"] = df["meeting_transcript"].apply(ast.literal_eval).str[0]
    print(df.shape, "\n")
    print(df.head(), "\n")
