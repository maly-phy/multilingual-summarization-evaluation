from utils import extract_content_between_tags
import pandas as pd
import os


def clean_meeting_challenges_eval(file_path):
    df = pd.read_csv(file_path)
    df.reset_index(drop=True, inplace=True)
    df["Spoken_language_Score"] = (
        df["Spoken_language_Score"]
        .apply(lambda x: extract_content_between_tags(x, "score"))
        .astype(float)
    )
    df["Spoken_language_Confidence"] = (
        df["Spoken_language_Confidence"]
        .apply(lambda x: extract_content_between_tags(x, "confidence").replace("%", ""))
        .astype(float)
    )
    df["Speaker_dynamics_Score"] = (
        df["Speaker_dynamics_Confidence"]
        .apply(lambda x: extract_content_between_tags(x, "score"))
        .astype(float)
    )
    df["Speaker_dynamics_Confidence"] = (
        df["Speaker_dynamics_Confidence"]
        .apply(lambda x: extract_content_between_tags(x, "confidence").replace("%", ""))
        .astype(float)
    )
    df["Coreference_Confidence"] = (
        df["Coreference_Confidence"]
        .apply(lambda x: extract_content_between_tags(x, "confidence").replace("%", ""))
        .astype(float)
    )
    df["Discourse_structure_Score"][1:2] = (
        df["Discourse_structure_Score"][1:2]
        .apply(lambda x: x.split("\n\n")[-1].strip())
        .astype(float)
    )
    df["Discourse_structure_Confidence"] = (
        df["Discourse_structure_Confidence"]
        .apply(lambda x: extract_content_between_tags(x, "confidence").replace("%", ""))
        .astype(float)
    )
    df["Contextual_turn-taking_Confidence"] = (
        df["Contextual_turn-taking_Confidence"]
        .apply(lambda x: extract_content_between_tags(x, "confidence").replace("%", ""))
        .astype(float)
    )
    df["Implicit_context_Confidence"] = (
        df["Implicit_context_Confidence"]
        .apply(lambda x: extract_content_between_tags(x, "confidence").replace("%", ""))
        .astype(float)
    )
    df["Low_information_density_Confidence"] = (
        df["Low_information_density_Confidence"]
        .apply(lambda x: extract_content_between_tags(x, "confidence").replace("%", ""))
        .astype(float)
    )
    return df


def clean_basic_meeting_eval(file_path):
    df = pd.read_csv(file_path)
    df.reset_index(drop=True, inplace=True)
    for col in df.columns.tolist():
        if "confidence" in col.lower():
            df[col] = df[col].apply(lambda x: x.replace("%", "")).astype(float)

    return df
