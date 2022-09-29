import numpy as np


def sample_df(df, n_samples, id_column="sample_id"):
    ids = df[[id_column]].to_numpy().reshape(-1)
    return np.random.choice(ids, size=n_samples, replace=False)


def get_samples(df, target_concept, seed, id_column="sample_id", distribution=None):
    np.random.seed(seed)
    samples_by_value = dict()

    # Creates a balanced distribution, according to the number of samples of the class with the lower number of samples
    if distribution is None or len(distribution) == 0:
        count_df = df[[id_column, target_concept]].groupby(target_concept).count()
        min_n = count_df[[id_column]].values.min()
        for v in df[target_concept].unique():
            samples_by_value[v] = min_n

    # If it is only specified the values of the "positive" samples (it is a binary distribution)
    elif len(distribution) == 1:

        target_value = list(distribution.keys())[0]
        pos_df = df[df[target_concept] == target_value]
        neg_df = df[df[target_concept] != target_value]

        # If the number of samples is not provided, use the most possible (while keeping the dataset balanced)
        if distribution[target_value] is None:
            n_samples = min(len(pos_df), len(neg_df))

        # Else use the provided number of positive samples (while keeping the dataset balanced)
        else:
            n_samples = min(distribution[target_value], len(neg_df))
        idxs = [sample_df(pos_df, n_samples, id_column), sample_df(neg_df, n_samples, id_column)]
        idxs = [idx for target_idxs in idxs for idx in target_idxs]
        return df[df[id_column].isin(idxs)]

    # Returns a samples according to the given distribution
    else:
        samples_by_value = distribution

    idxs = []
    for k, v in samples_by_value.items():
        target_df = df[df[target_concept] == k]

        # If no number of samples is specified, use all available
        if v is None:
            idxs.append(sample_df(target_df, len(df[df[target_concept] == k]), id_column))
        else:
            idxs.append(sample_df(target_df, v, id_column))

    idxs = [idx for target_idxs in idxs for idx in target_idxs]
    return df[df[id_column].isin(idxs)]


"""if __name__ == "__main__":
    FEATURES_IDX = range(1, 15)
    import pandas as pd

    df = pd.read_csv("dataset/boston_with_label_simplified.csv")
    df = get_samples(df, "target", 0, distribution={8: None})
    print(df.columns)"""
