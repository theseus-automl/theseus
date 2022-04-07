import pandas as pd


def check_balance(
    texts: pd.Series,
    labels: pd.Series,
) -> bool:
    df = pd.DataFrame(
        {
            'texts': texts,
            'labels': labels,
        },
    ).reset_index()
    lengths = set()

    for label in df['labels'].unique():
        lengths.add(len(df[df['labels'] == label]))

    return len(lengths) == 1
