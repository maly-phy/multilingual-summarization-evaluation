import pandas as pd
import os, sys
import ast


class SummaryQualityComparator:
    def __init__(self, df_agent_iter, df_llm_refined, df_baseline, language, rounds):
        self.df_agent_iter = df_agent_iter
        self.df_llm_refined = df_llm_refined
        self.df_baseline = df_baseline
        self.language = language
        self.rounds = rounds

    def extract_round_scores(self):
        results = []
        for i, _ in self.df_agent_iter.iterrows():
            idx = i.split("_")[1]
            quality_scores = 0
            refined_llm = 0
            for j in range(self.rounds):
                summary_quality = float(
                    self.df_agent_iter.at[f"{j}_{idx}", "summary_quality"]
                )
                refined_llm_quality = float(
                    self.df_llm_refined.at[f"{j}_{idx}", "refined_llm_quality"]
                )
                quality_scores += summary_quality
                refined_llm += refined_llm_quality
            results.append(
                {
                    "avg_summary_quality_over_rounds": quality_scores / self.rounds,
                    "avg_refined_llm_quality_over_rounds": refined_llm / self.rounds,
                }
            )

        for _, row in self.df_baseline.iterrows():
            model_summary_quality = ast.literal_eval(row["model_summary_quality"])
            reference_summary_quality = ast.literal_eval(row["ref_summary_quality"])
            baseline_per_summary = (
                float(model_summary_quality[0]["llm_quality_score"])
                + float(reference_summary_quality[0]["llm_quality_score"])
            ) / 2
            results.append({"avg_quality_baseline": baseline_per_summary})

        out_df = pd.DataFrame(results)
        save_dir = f"multiagent_summary/evaluation/{self.language}/agent_loop/avg_quality_scores.csv"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        out_df.to_csv(save_dir, index=False)
        print(f"Quality results saved to {save_dir}")


def save_avg_scores(results_dir, save_dir):
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    results_df = pd.read_csv(results_dir)
    avg_summary_quality = results_df["avg_summary_quality_over_rounds"].mean()
    avg_refined_llm_quality = results_df["avg_refined_llm_quality_over_rounds"].mean()
    avg_baseline = results_df["avg_quality_baseline"].mean()
    with open(save_dir, "w") as f:
        f.write(f"Average Summary Quality (error based): {avg_summary_quality}\n\n")
        f.write(f"Average Refined Quality (LLM judge): {avg_refined_llm_quality}\n\n")
        f.write(
            f"Average Baseline Quality (reference and model summaries): {avg_baseline}\n\n"
        )


if __name__ == "__main__":
    language = "English"
    rounds = 5
    df_agent_iter = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/agent_loop/agent_iter.csv"
    )
    df_llm_refined = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/agent_loop/refined_llm_quality.csv"
    )
    df_agent_iter.set_index("Unnamed: 0", inplace=True)
    df_llm_refined.set_index("Unnamed: 0", inplace=True)

    df_baseline = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/agent_loop/quality_baseline.csv"
    )

    comparator = SummaryQualityComparator(
        df_agent_iter, df_llm_refined, df_baseline, language, rounds
    )
    comparator.extract_round_scores()

    save_dir = (
        f"multiagent_summary/evaluation/{language}/agent_loop/avg_quality_scores.txt"
    )
    results_dir = (
        f"multiagent_summary/evaluation/{language}/agent_loop/avg_quality_scores.csv"
    )
    save_avg_scores(results_dir, save_dir)
