import numpy as np
import pandas as pd
import nltk
from multiprocessing import Pool, cpu_count
from setup import CreateArrays

setup = CreateArrays()
tool = setup.tool
spell = setup.spell

CONTENT_TAGS = {
    "NN", "NNS", "NNP", "NNPS",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "JJ", "JJR", "JJS",
    "RB", "RBR", "RBS",
}

DERIVATIONAL_SUFFIXES = {
    "tion", "sion", "ment", "ness", "ity", "ance", "ism", "ship", "hood",
    "ive", "ous", "able", "al", "ical", "ize", "ise", "ify", "ate",
    "ary", "ory", "ant", "ent", "ery", "ist"
}

def compute(text: str, total_words: int) -> np.ndarray:
    sentences = nltk.sent_tokenize(text)
    sent_lengths = [len(nltk.word_tokenize(sent)) for sent in sentences]
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    num_sentences = len(sentences)
    avg_sent_len = total_words / num_sentences
    std_sent_len = float(np.std(sent_lengths))

    cleaned_tokens = [w.replace('.', '').replace(',', '') for w in tokens]
    total_chars = sum(len(w) for w in cleaned_tokens)
    chars_per_word = total_chars / total_words

    content_words = sum(1 for _, tag in tagged if tag in CONTENT_TAGS)
    content_ratio_val = content_words / total_words

    total_suffixes = sum(
        1 for w in tokens
        if len(w) > 5 and any(w.lower().endswith(suffix) for suffix in DERIVATIONAL_SUFFIXES)
    )
    deriv_ratio_val = total_suffixes / total_words

    matches = tool.check(text)
    error_ratio_val = len(matches) / total_words

    misspelled = spell.unknown(tokens)
    spelling_error_ratio = len(misspelled) / total_words

    return np.array([
        num_sentences,
        avg_sent_len,
        std_sent_len,
        chars_per_word,
        content_ratio_val,
        deriv_ratio_val,
        error_ratio_val,
        spelling_error_ratio
    ], dtype=np.float32)

def compute_row(args):
    text, wc = args
    return compute(text, wc)

def stratified_train_test_split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8):
    y = y.ravel()
    unique_classes = np.unique(y)

    train_indices = []
    test_indices = []

    for c in unique_classes:
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)
        n_train_c = int(train_ratio * len(class_indices))
        train_indices.append(class_indices[:n_train_c])
        test_indices.append(class_indices[n_train_c:])

    train_indices = np.concatenate(train_indices)
    test_indices = np.concatenate(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]

if __name__ == "__main__":
    # just in case
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger_eng")
    # load data
    df = pd.read_csv("english_exam_database.csv")

    texts = df["text_corrected"].to_numpy()
    wordcounts = df["wordcount"].to_numpy()
    pairs = list(zip(texts, wordcounts))

    cpu_num = cpu_count()
    print("Using " + str(cpu_count) + " processes.")

    with Pool(processes=cpu_num) as pool:
        feature_rows = pool.map(compute_row, pairs, chunksize=100)

    text_features = np.vstack(feature_rows)
    print("Text features:", text_features.shape)

    features = df[["wordcount", "mtld"]].to_numpy(dtype=np.float32)
    print("Base features:", features.shape)

    X = np.concatenate([features, text_features], axis=1)
    print("Combined features:", X.shape)

    y = df[["cefr_numeric"]].to_numpy(dtype=np.float32)

    X_train, y_train, X_test, y_test = stratified_train_test_split(X, y, train_ratio=0.8)

    # standardization
    feature_means = X_train.mean(axis=0)
    feature_stds = X_train.std(axis=0)
    feature_stds[feature_stds == 0] = 1.0

    X_train_std = (X_train - feature_means) / feature_stds
    X_test_std = (X_test - feature_means) / feature_stds

    y_train_classifier = y_train.astype(int) - 1
    y_test_classifier = y_test.astype(int) - 1

    np.savez(
        "cefr_data.npz",
        X_train_std=X_train_std,
        y_train=y_train_classifier,
        X_test_std=X_test_std,
        y_test=y_test_classifier,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    print("Saved to cefr_data.npz")