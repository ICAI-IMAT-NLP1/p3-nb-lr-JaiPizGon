from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    examples = []
    with open(infile, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                parts = line.strip().split("\t")
                label = int(parts[-1])  # The label is the last element after splitting
                sentence = "\t".join(
                    parts[:-1]
                )  # The sentence is everything before the last element
                tokenized_sent = tokenize(sentence)
                examples.append(SentimentExample(tokenized_sent, label))
    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # Flatten the list of all words in all examples
    all_words = [word for example in examples for word in example.words]

    # Count the frequency of each word and build the vocabulary
    vocab_count = Counter(all_words)
    vocab = {word: index for index, word in enumerate(vocab_count)}

    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    bow = torch.zeros(len(vocab), dtype=torch.float32)

    for word in text:
        if word in vocab:
            index = vocab[word]
            bow[index] = 1 if binary else bow[index] + 1

    return bow
