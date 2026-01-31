import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.append(os.getcwd())
init_mean_ger = [
    0.28,
    0.05,
    0.17,
    0.63,
    0.87,
    0.52,
    0.39,
    0.33,
    0.19,
    0.39,
    0.073,
    0.01,
    -5.1,
    0.31,
    1.0,
    0.25,
    0.1,
    0.05,
    0.29,
    -5.3,
]
init_std_ger = [
    0.04,
    0.02,
    0.03,
    0.03,
    0.11,
    0.07,
    0.03,
    0.06,
    0.02,
    0.03,
    0.03,
    0.02,
    0.4,
    0.04,
    0.17,
    0.06,
    0.07,
    0.02,
    0.06,
    0.5,
]

regen_mean_ger = [
    0.57,
    0.38,
    0.48,
    0.77,
    0.87,
    0.46,
    0.71,
    0.42,
    0.46,
    0.63,
    0.40,
    0.34,
    -3.7,
    0.41,
    0.79,
    0.6,
    0.38,
    0.29,
    0.30,
    -3.7,
]

regen_std_ger = [
    0.12,
    0.16,
    0.14,
    0.06,
    0.12,
    0.06,
    0.09,
    0.1,
    0.12,
    0.09,
    0.16,
    0.16,
    0.49,
    0.07,
    0.29,
    0.17,
    0.17,
    0.09,
    0.07,
    0.82,
]

init_mean_criteria_eng = [4.3, 4.0, 3.2]
init_std_criteria_eng = [0.13, 0.49, 1.3]
regen_mean_criteria_eng = [4.5, 3.9, 4.0]
regen_std_criteria_eng = [0.17, 0.48, 0.59]

corr_coef_eng = [-0.15, -0.2, -0.13]
corr_p_eng = [0.43, 0.3, 0.44]

metrics = [
    "ROUGE-1 F1",
    "ROUGE-2 F1",
    "ROUGE-L F1",
    "BERT F1",
    "LAR",
    "LENS",
    "BLANC help",
    "Bleurt",
    "Meteor",
    "Chrf",
    "Sacrebleu",
    "Bleu",
    "LongDocFact",
    "Questeval",
    "Estime alarms",
    "Estime soft",
    "Estime coherence",
    "BLANC tune",
    "Perplexity",
    "BART",
]
criteria = ["Linguistic", "Naturalness", "Factuality"]
correlations = ["Spearman", "Pearson", "Kendall"]

corr_coef_ger = [0.22, 0.24, 0.21]
corr_p_ger = [0.26, 0.21, 0.26]

init_mean_criteria_ger = [4.3, 3.9, 3.5]
init_std_criteria_ger = [0.2, 0.33, 0.59]
regen_mean_criteria_ger = [4.1, 3.6, 3.9]
regen_std_criteria_ger = [0.27, 0.64, 0.46]

init_mean_eng = [
    0.37,
    0.08,
    0.22,
    0.60,
    0.81,
    0.585,
    0.13,
    0.41,
    0.27,
    0.45,
    0.08,
    0.04,
    -2.45,
    0.41,
    0.87,
    0.64,
    0.23,
    0.08,
    0.21,
    -3.17,
]
init_std_eng = [
    0.05,
    0.08,
    0.03,
    0.03,
    0.15,
    0.061,
    0.02,
    0.05,
    0.05,
    0.04,
    0.04,
    0.03,
    0.22,
    0.05,
    0.15,
    0.05,
    0.12,
    0.02,
    0.04,
    0.3,
]

regen_mean_eng = [
    0.54,
    0.28,
    0.44,
    0.69,
    0.85,
    0.582,
    0.31,
    0.42,
    0.41,
    0.57,
    0.28,
    0.23,
    -1.7,
    0.52,
    0.5,
    0.81,
    0.36,
    0.23,
    0.18,
    -2.6,
]
regen_std_eng = [
    0.09,
    0.11,
    0.11,
    0.05,
    0.17,
    0.068,
    0.07,
    0.06,
    0.08,
    0.07,
    0.1,
    0.11,
    0.4,
    0.08,
    0.2,
    0.08,
    0.2,
    0.07,
    0.06,
    0.43,
]


def visualize_eval_results(
    metrics,
    init_mean,
    init_std,
    regen_mean,
    regen_std,
    lang,
    start,
    stop,
    png_name,
    title,
):
    x = np.arange(start, stop, 1)
    metric_names = [metrics[i] for i in range(start, stop, 1)]
    width = 0.35
    pos_ticks = np.arange(0.0, 1.2, 0.2)
    neg_ticks = np.arange(-5.9, 0.0, 1.5)

    if min([init_mean[i] for i in range(start, stop, 1)]) < 0.0:
        fig, (ax_neg, ax_pos) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 3]}, figsize=(12, 6)
        )
        ax_neg.bar(
            x - width / 2,
            [init_mean[i] for i in range(start, stop, 1)],
            width,
            yerr=[init_std[i] for i in range(start, stop, 1)],
            label=f"Initial ({lang})",
            capsize=5,
        )
        ax_neg.bar(
            x + width / 2,
            [regen_mean[i] for i in range(start, stop, 1)],
            width,
            yerr=[regen_std[i] for i in range(start, stop, 1)],
            label=f"Regenerated ({lang})",
            capsize=5,
        )

        ax_pos.bar(
            x - width / 2,
            [init_mean[i] for i in range(start, stop, 1)],
            width,
            yerr=[init_std[i] for i in range(start, stop, 1)],
            label=f"Initial ({lang})",
            capsize=5,
        )
        ax_pos.bar(
            x + width / 2,
            [regen_mean[i] for i in range(start, stop, 1)],
            width,
            yerr=[regen_std[i] for i in range(start, stop, 1)],
            label=f"Regenerated ({lang})",
            capsize=5,
        )
        ax_neg.set_ylim(-5.9, 0.0)
        ax_neg.set_yticks(neg_ticks)
        ax_neg.axhline(0, color="black", linewidth=0.8)

        ax_pos.set_ylim(0.0, 1.2)
        ax_pos.set_yticks(pos_ticks)
        ax_pos.axhline(0, color="black", linewidth=0.8)
        ax_pos.set_xticks(x, metric_names, rotation=40)
        ax_pos.set_ylabel("Scores")
        ax_pos.legend()
        ax_pos.set_title(title)

    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(
            x - width / 2,
            [init_mean[i] for i in range(start, stop, 1)],
            width,
            yerr=[init_std[i] for i in range(start, stop, 1)],
            label=f"Initial ({lang})",
            capsize=5,
        )
        ax.bar(
            x + width / 2,
            [regen_mean[i] for i in range(start, stop, 1)],
            width,
            yerr=[regen_std[i] for i in range(start, stop, 1)],
            label=f"Regenerated ({lang})",
            capsize=5,
        )

        ax.set_ylabel("Scores")
        ax.set_xticks(x, metric_names, rotation=40)
        ax.legend()
        ax.set_title(title)

    fig.tight_layout()
    save_dir = "evaluation_scores/visualize"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{png_name}.png"))


def visualize_corr_results(png_name):
    x = np.arange(len(correlations))
    width = 0.35
    plt.figure(figsize=(8, 6))

    bars_eng = plt.bar(x - width / 2, corr_coef_eng, width, label="ENG")
    bars_ger = plt.bar(x + width / 2, corr_coef_ger, width, label="GER")

    # plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, correlations)
    plt.ylabel("Correlation Coefficient")
    plt.title("Correlation for the naturalness criterion")
    plt.legend()
    plt.tight_layout()

    for bar, p in zip(bars_eng, corr_p_eng):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"p={p:.2f}",
            ha="center",
            va="bottom",
        )

    for bar, p in zip(bars_ger, corr_p_ger):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"p={p:.2f}",
            ha="center",
            va="bottom",
        )

    save_dir = "evaluation_scores/visualize"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{png_name}.png"))


if __name__ == "__main__":
    visualize_corr_results("corr_naturalness")
