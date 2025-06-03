import numpy as np
import pandas as pd

def simulate_labels(summary):
    np.random.seed(42)
    verdicts = ["Yes", "WeakYes", "No"]
    summary["verdict"] = np.random.choice(verdicts, size=len(summary))
    summary["target"] = summary["verdict"].apply(lambda x: 0 if x == "No" else 1)
    summary["stage_outcome"] = np.random.choice([0, 1, 2, 3], size=len(summary), p=[0.3, 0.2, 0.2, 0.3])
    return summary

def simulate_panelist_strictness(df):
    panel_strictness = df.groupby("panel").size().reset_index(name="interview_count")
    panel_strictness["strictness"] = np.clip(np.random.normal(0.6, 0.2, len(panel_strictness)), 0, 1)
    return panel_strictness

def simulate_candidate_behavior(summary):
    np.random.seed(99)
    summary["confidence_score"] = np.clip(np.random.normal(loc=0.7, scale=0.15, size=len(summary)), 0, 1)
    summary["answer_speed"] = np.clip(np.random.normal(loc=30, scale=10, size=len(summary)), 5, 90)
    summary["correctness_prob"] = np.clip(np.random.normal(loc=0.65, scale=0.2, size=len(summary)), 0, 1)
    return summary
