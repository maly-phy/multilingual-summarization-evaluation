import pandas as pd
import os
import pandas as pd
from utils import read_json_criteria, initialize_model
from error_severity import SeverityScorer
from severity_impact import SeverityImpactScorer
from quality_score import SummQualityScorer
from feedback import FeedbackSystem


class MultiagentSummaryIterator:
    def __init__(self, df, language, max_tokens, criteria_path, exclude_criteria=None):
        self.df = df
        self.language = language
        self.max_tokens = max_tokens
        self.criteria_path = criteria_path
        self.exclude_criteria = exclude_criteria
        self.model_init = initialize_model(max_tokens=self.max_tokens)
        self.criteria = read_json_criteria(self.criteria_path)
        self.impact_scorer = SeverityImpactScorer(
            self.language, self.max_tokens, self.criteria_path, self.exclude_criteria
        )
        self.severity_scorer = SeverityScorer(
            self.language,
            self.max_tokens,
            self.criteria_path,
            self.exclude_criteria,
        )
        self.quality_scorer = SummQualityScorer(self.language, self.exclude_criteria)
        self.feedback_system = FeedbackSystem(
            self.language, self.max_tokens, self.criteria_path, self.exclude_criteria
        )

    def agent_iter(self):
        for i in range(5):
            for idx, row in self.df.iterrows():
                model_summary = row["model_factual_summary"]
                meeting_transcript = row["Meeting"]
                severity_eval = self.severity_scorer.init_severity_eval(
                    self.model_init, model_summary, meeting_transcript
                )
                severity_df = pd.DataFrame(
                    [
                        {
                            **{
                                f"{criterion}": severity_eval[criterion]
                                for criterion in self.criteria.keys()
                            }
                        }
                    ]
                )
                severity_impact = self.impact_scorer.severity_impact(
                    self.model_init, model_summary, meeting_transcript, severity_df, idx
                )
                severity_impact_df = pd.DataFrame(
                    [
                        {
                            **{
                                f"{criterion}": severity_impact[criterion]
                                for criterion in self.criteria.keys()
                            }
                        }
                    ]
                )

                overall_impact, summary_quality = (
                    self.quality_scorer.weighted_severity_impact(
                        severity_impact_df, idx
                    )
                )
                feedback = self.feedback_system.get_feedback(
                    self.model_init, model_summary, meeting_transcript, severity_df, idx
                )
                feedback_df = pd.DataFrame(
                    [
                        {
                            **{
                                f"{criterion}": feedback[criterion]
                                for criterion in self.criteria.keys()
                            }
                        }
                    ]
                )
