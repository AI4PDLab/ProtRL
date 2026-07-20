from datasets import Dataset

def create_ProtRL_dataset(df, prompt, sequence_col="sequence", activity_col="activity"):
    """
    Create Dataset obj with correct format for ProtRL
    """
    rows = []
    for idx, entry in df.iterrows():
        row = entry.to_dict()

        row["prompt"] = prompt
        row["completion"] = row.pop(sequence_col)
        row["reward"] = row.pop(activity_col)

        rows.append(row)

    return Dataset.from_list(rows)
