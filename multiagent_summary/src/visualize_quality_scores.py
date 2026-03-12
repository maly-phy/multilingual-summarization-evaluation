import pandas as pd
import os, sys, ast
import matplotlib.pyplot as plt
import numpy as np


class QualityVisualizer:
    def __init__(self, df, rounds, language, baseline_df=None):
        self.df = df
        self.rounds = rounds
        self.language = language
        self.baseline_df = baseline_df

    def extract_scores(self):
        all_quality_scores = []
        for j in range(self.rounds):
            quality_scores = [
                self.df.at[f"{j}_{i}", "summary_quality"] for i in range(30)
            ]
            all_quality_scores.append({f"round_{j}": quality_scores})
        return all_quality_scores

    def visualize_scores(self):
        all_quality_scores = self.extract_scores()
        avg_scores_per_round = [
            np.mean(all_quality_scores[j][f"round_{j}"]) for j in range(self.rounds)
        ]
        plt.figure(figsize=(12, 6))
        plt.plot(
            range(1, self.rounds + 1),
            avg_scores_per_round,
            marker="o",
            label="Multi-Agent Summary",
            color="blue",
        )

        if self.baseline_df is not None:
            baseline_scores = []
            for i, row in self.baseline_df.iterrows():
                model_summary_quality = ast.literal_eval(row["model_summary_quality"])
                reference_summary_quality = ast.literal_eval(row["ref_summary_quality"])
                baseline_per_summary = (
                    float(model_summary_quality[0]["llm_quality_score"])
                    + float(reference_summary_quality[0]["llm_quality_score"])
                ) / 2
                baseline_scores.append(baseline_per_summary)

            avg_baseline_score = np.mean(baseline_scores)
            plt.plot(
                range(1, self.rounds + 1),
                [avg_baseline_score] * self.rounds,
                linestyle="--",
                label="Baseline",
                color="red",
            )

        plt.xlabel("Rounds")
        plt.ylabel("AVG summary quality / round")
        plt.title("Summary Quality Scores Over Rounds")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        save_dir = f"multiagent_summary/evaluation/{self.language}/visuals"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"quality_per_round2.png"))
        plt.show()


if __name__ == "__main__":
    rounds = 5
    language = "English"
    df = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/agent_loop/agent_iter.csv"
    )
    df_baseline = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/agent_loop/quality_baseline.csv"
    )
    df.set_index("Unnamed: 0", inplace=True)
    visualizer = QualityVisualizer(df, rounds, language, df_baseline)
    visualizer.visualize_scores()
