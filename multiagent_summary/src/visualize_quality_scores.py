import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import numpy as np


class QualityVisualizer:
    def __init__(self, df, rounds, language):
        self.df = df
        self.rounds = rounds
        self.language = language

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
        plt.plot(range(1, self.rounds + 1), avg_scores_per_round)
        plt.xlabel("Rounds")
        plt.ylabel("AVG summary quality / round")
        plt.title("Summary Quality Scores Over Rounds")

        plt.grid()
        plt.tight_layout()
        save_dir = f"multiagent_summary/evaluation/{self.language}/visuals"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"quality_per_round.png"))
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("multiagent_summary/evaluation/English/agent_loop/agent_iter.csv")
    df.set_index("Unnamed: 0", inplace=True)
    rounds = 5
    language = "English"
    visualizer = QualityVisualizer(df, rounds, language)
    visualizer.visualize_scores()
