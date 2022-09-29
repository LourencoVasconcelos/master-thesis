import string
import math


def encode_fun(idx, symbols, word_size):
    base = len(symbols)
    word = [symbols[0] for i in range(word_size)]
    i = 0
    while idx != 0:
        temp = idx % base
        word[i] = symbols[temp]
        i = i + 1
        idx = idx // base
    word.reverse()
    return "".join(word)


def decode_fun(word, symbols, original_columns):
    idx = 0
    for i in range(len(word)):
        pwr = len(word)-1-i
        idx += len(symbols)**pwr*symbols.index(word[i])
    return original_columns[idx]


def encode_concepts(concepts, keep=None):
    idx = []
    if keep is not None:
        idx = [concepts.index(c) for c in keep]
        idx.sort()
        new_concepts = [c for c in concepts if c not in keep]
    else:
        new_concepts = concepts

    if len(new_concepts) != 0:
        letters = string.ascii_lowercase
        k = max(math.ceil(math.log(len(new_concepts), len(letters))), 1)
        new_concepts = [encode_fun(i, letters, k) for i, _ in enumerate(new_concepts)]

    for i in idx:
        new_concepts.insert(i, concepts[i])
    return new_concepts


if __name__ == "__main__":
    FEATURES_IDX = range(1, 15)
    import pandas as pd

    df = pd.read_csv("dataset/boston_with_label_binary.csv")
