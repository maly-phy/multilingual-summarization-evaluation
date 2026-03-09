import pandas as pd
import os, sys
import pandas as pd
import time
from utils import read_json_criteria, initialize_model
from error_severity import SeverityScorer
from severity_impact import SeverityImpactScorer
from quality_score import SummQualityScorer
from feedback import FeedbackSystem
from refiner import Refiner


class MultiagentSummaryIterator:
    def __init__(self, df, language, max_tokens, criteria_path, exclude_criteria=None):
        self.df = df
        self.language = language
        self.max_tokens = max_tokens
        self.criteria_path = criteria_path
        self.exclude_criteria = exclude_criteria
        self.model_init = initialize_model(max_tokens=self.max_tokens)
        self.criteria = read_json_criteria(self.criteria_path)
        self.out_df = pd.DataFrame()
        self.severity_scorer = SeverityScorer(
            self.df,
            self.language,
            self.max_tokens,
            self.criteria_path,
            self.exclude_criteria,
        )
        self.impact_scorer = SeverityImpactScorer(
            self.language, self.max_tokens, self.criteria_path, self.exclude_criteria
        )
        self.quality_scorer = SummQualityScorer(self.language, self.exclude_criteria)
        self.feedback_system = FeedbackSystem(
            self.language, self.max_tokens, self.criteria_path, self.exclude_criteria
        )
        self.refine_system = Refiner(self.max_tokens, self.language, self.criteria_path)

    def agent_iter(self, rounds):
        for j in range(rounds):
            print(f"*** Starting iteration {j}/{rounds} ***\n")
            start_round_time = time.time()
            summary_quality_scores = []
            for idx, row in self.df.iterrows():
                model_summary = row["model_factual_summary"]
                meeting_transcript = row["Meeting"]
                self.out_df.at[f"{j}_{idx}", "model_summary"] = model_summary
                self.out_df.at[f"{j}_{idx}", "meeting_transcript"] = meeting_transcript
                nums, denos = 0.0, 0.0
                update_refined_summary = []
                for i, (criterion, description) in enumerate(self.criteria.items()):
                    if self.exclude_criteria and criterion in self.exclude_criteria:
                        continue
                    print(
                        f"Processing criterion {i}/{len(self.criteria)} | summary {idx}/{len(self.df)} | iteration {j}\n"
                    )

                    summary_to_process = model_summary
                    for k in range(len(update_refined_summary)):
                        if (
                            i == 0
                            and j > 0
                            and f"iter_{j-1}_criterion_{len(self.criteria) - 1}_refined_{idx}"
                            in update_refined_summary[k]
                        ):
                            summary_to_process = update_refined_summary[k][
                                f"iter_{j-1}_criterion_{len(self.criteria) - 1}_refined_{idx}"
                            ]
                        elif (
                            f"iter_{j}_criterion_{i-1}_refined_{idx}"
                            in update_refined_summary[k]
                        ):
                            summary_to_process = update_refined_summary[k][
                                f"iter_{j}_criterion_{i-1}_refined_{idx}"
                            ]

                    severity_response = self.severity_scorer.severity_prompt(
                        self.model_init,
                        criterion,
                        description,
                        summary_to_process,
                        meeting_transcript,
                    )
                    feedback_response = self.feedback_system.feedback_prompt(
                        model_init=self.model_init,
                        criterion=None,
                        description=description,
                        model_summary=summary_to_process,
                        meeting_transcript=meeting_transcript,
                        row=severity_response,
                    )
                    refiner_response = self.refine_system.refiner_prompt(
                        model_init=self.model_init,
                        criterion=None,
                        description=description,
                        model_summary=summary_to_process,
                        meeting_transcript=meeting_transcript,
                        update_refined_summary=update_refined_summary,
                        row=feedback_response,
                        i=None,
                    )

                    update_refined_summary.append(
                        {f"iter_{j}_criterion_{i}_refined_{idx}": refiner_response}
                    )
                    self.out_df.at[f"{j}_{idx}", f"severity_{criterion}"] = (
                        severity_response
                    )
                    self.out_df.at[f"{j}_{idx}", f"feedback_{criterion}"] = (
                        feedback_response
                    )
                    self.out_df.at[f"{j}_{idx}", f"refined_{criterion}"] = (
                        refiner_response
                    )

                    severity_impact = self.impact_scorer.severity_impact_prompt(
                        model_init=self.model_init,
                        criterion=None,
                        description=description,
                        model_summary=summary_to_process,
                        meeting_transcript=meeting_transcript,
                        row=severity_response,
                    )

                    numerator, denominator = self.quality_scorer.impact_scores(
                        row=severity_impact,
                        criteria=None,
                        weight=self.quality_scorer.weight_importances[criterion],
                    )
                    nums += numerator
                    denos += denominator

                overall_impact, summary_quality = (
                    self.quality_scorer.calculate_overall_quality(nums, denos)
                )
                self.out_df.at[f"{j}_{idx}", "overall_impact"] = overall_impact
                self.out_df.at[f"{j}_{idx}", "summary_quality"] = summary_quality
                summary_quality_scores.append(
                    {f"iter_{j}_summary_{idx}": summary_quality}
                )
                nums, denos = 0.0, 0.0

            print(
                f"Round {j} completed in {(time.time() - start_round_time) / 60} minutes\n\n"
            )
        save_dir = (
            f"multiagent_summary/evaluation/{self.language}/agent_loop/agent_iter.csv"
        )
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        self.out_df.to_csv(save_dir, index=True)
        print(f"Agent Loop results saved to {save_dir}\n")


if __name__ == "__main__":
    criteria_path = "multiagent_summary/error_types/error_types_eng.json"
    language = "English"
    df_path = f"evaluation/{language}/atomic_facts/corrected_summary.csv"
    df = pd.read_csv(df_path)
    max_tokens = 3000
    exclude_criteria = ["Hallucination", "Structure", "Irrelevance"]
    rounds = 5
    multiagent_iterator = MultiagentSummaryIterator(
        df, language, max_tokens, criteria_path, exclude_criteria
    )
    start_time = time.time()
    multiagent_iterator.agent_iter(rounds)
    print(f"Full agent loops completed in {(time.time() - start_time) / 60} minutes")

    # from utils import text_chunker
    # out_df = pd.read_csv(
    #     f"multiagent_summary/evaluation/{language}/test_samples/agent_loop.csv"
    # )
    # out_file = f"multiagent_summary/outputs/{language}/agent_loop_samples.txt"
    # os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # with open(out_file, "w") as f:
    #     for idx, row in out_df.iterrows():
    #         f.write(f"*** Starting Summary {idx} ***\n\n")
    #         for i, criterion in enumerate(read_json_criteria(criteria_path).keys()):
    #             if criterion in exclude_criteria:
    #                 continue
    #             f.write(f"{criterion}\n\n")
    #             f.write(f"{row[f'feedback_{criterion}']}\n\n")
    #             f.write(f"{text_chunker(row[f'refined_{criterion}'])}\n\n")

    #         f.write(f"{row['summary_quality']}\n\n")
