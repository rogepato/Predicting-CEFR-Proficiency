import numpy as np
import nltk
import language_tool_python
from spellchecker import SpellChecker

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

tool = language_tool_python.LanguageTool("en-US")
spell = SpellChecker(language="en")

def wordcount(text: str) -> int:
    return len(nltk.word_tokenize(text))

# ************Code Implementation for McCarthy & Jarvis (2010)'s MTLD************
def _mtld_sequence(tokens, threshold=0.72):
    """
    Compute MTLD for a single direction (forward OR reverse), McCarthy & Jarvis (2010).
    """
    n_tokens = len(tokens)
    if n_tokens == 0:
        return 0.0

    factor_count = 0.0
    types_in_factor = set()
    tokens_in_factor = 0

    for tok in tokens:
        tokens_in_factor += 1
        types_in_factor.add(tok)
        ttr = len(types_in_factor) / tokens_in_factor

        # When TTR falls to or below threshold, we have a full factor
        if ttr <= threshold:
            factor_count += 1.0
            types_in_factor.clear()
            tokens_in_factor = 0

    # Handle remainder (partial factor) if there are leftover tokens
    if tokens_in_factor > 0 and len(types_in_factor) > 0:
        ttr_rem = len(types_in_factor) / tokens_in_factor

        # If ttr_rem == 1.0, this contributes 0 of a factor
        if ttr_rem < 1.0:
            partial_factor = (1.0 - ttr_rem) / (1.0 - threshold)
            factor_count += partial_factor

    # If no factor was completed, define MTLD as text length
    if factor_count == 0:
        return float(n_tokens)

    return n_tokens / factor_count

def mtld(text, threshold=0.72):
    raw_tokens = nltk.word_tokenize(text)
    tokens = [
        tok.lower()
        for tok in raw_tokens
        if any(ch.isalnum() for ch in tok)
    ]

    if not tokens:
        return 0.0

    mtld_forward = _mtld_sequence(tokens, threshold=threshold)
    mtld_reverse = _mtld_sequence(list(reversed(tokens)), threshold=threshold)

    return (mtld_forward + mtld_reverse) / 2.0
# ************Code Implementation for McCarthy & Jarvis (2010)'s MTLD************

def compute_text_features(text: str, total_words: int) -> np.ndarray:
    if total_words <= 0:
        return np.zeros(8, dtype=np.float32)

    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:
        sentences = [text]

    sent_lengths = [len(nltk.word_tokenize(sent)) for sent in sentences]
    tokens = nltk.word_tokenize(text)

    num_sentences = len(sentences)
    avg_sent_len = total_words / num_sentences
    std_sent_len = float(np.std(sent_lengths))

    cleaned_tokens = [w.replace(".", "").replace(",", "") for w in tokens]
    total_chars = sum(len(w) for w in cleaned_tokens)
    chars_per_word = total_chars / total_words

    tagged = nltk.pos_tag(tokens)
    content_words = sum(1 for _, tag in tagged if tag in CONTENT_TAGS)
    content_ratio_val = content_words / total_words

    total_suffixes = sum(
        1
        for w in tokens
        if len(w) > 5 and any(w.lower().endswith(suffix) for suffix in DERIVATIONAL_SUFFIXES)
    )
    deriv_ratio_val = total_suffixes / total_words

    matches = tool.check(text)
    error_ratio_val = len(matches) / total_words

    misspelled = spell.unknown(tokens)
    spelling_error_ratio = len(misspelled) / total_words

    return np.array(
        [
            num_sentences,
            avg_sent_len,
            std_sent_len,
            chars_per_word,
            content_ratio_val,
            deriv_ratio_val,
            error_ratio_val,
            spelling_error_ratio,
        ],
        dtype=np.float32,
    )

def text_to_feature_vector(text: str) -> np.ndarray:
    """
    Full 10-dim feature vector:
    [wordcount, mtld, 8 text features...]
    """
    wc = wordcount(text)
    mtld_val = mtld(text)

    text_feats = compute_text_features(text, wc)  # shape (8,)
    vector = np.concatenate(
        [np.array([wc, mtld_val], dtype=np.float32), text_feats],
        axis=0,
    )

    return vector