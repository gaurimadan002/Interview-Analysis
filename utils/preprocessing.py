import pandas as pd

def engineer_features(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["question_length"] = df["question"].apply(lambda x: len(str(x).split()))

    summary = df.groupby("session").agg({
        "start_time": "first",
        "end_time": "first",
        "question": "count",
        "topic": pd.Series.nunique,
        "panel": pd.Series.nunique
    }).reset_index()

    summary.columns = ["session", "start_time", "end_time", "num_questions", "unique_topics", "panel_count"]
    summary["duration_mins"] = (summary["end_time"] - summary["start_time"]).dt.total_seconds() / 60

    avg_len = df.groupby("session")["question_length"].mean().reset_index(name="avg_question_length")
    summary = summary.merge(avg_len, on="session")

    return summary
