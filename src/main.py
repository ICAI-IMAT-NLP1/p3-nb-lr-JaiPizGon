from data_processing import read_sentiment_examples, build_vocab, bag_of_words
from naive_bayes import NaiveBayes
from logistic_regression import LogisticRegression
from utils import SentimentExample, evaluate_classification, get_word_weight
from typing import List, Dict
import torch


def main():
    # Load training data
    train_examples: List[SentimentExample] = read_sentiment_examples("data/train.txt")
    print("Building vocabulary...")
    vocab: Dict[str, int] = build_vocab(train_examples)

    # Prepare features and labels for the models
    print("Preparing training BoW...")
    train_features: torch.Tensor = torch.stack(
        [bag_of_words(ex.words, vocab) for ex in train_examples]
    )
    train_labels: torch.Tensor = torch.tensor(
        [ex.label for ex in train_examples], dtype=torch.float32
    )

    print("Training Naive Bayes model...")
    # Train Naive Bayes model
    nb_model = NaiveBayes()
    nb_model.fit(train_features, train_labels)

    # Train Logistic Regression model
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(train_features, train_labels, learning_rate=0.15, epochs=1500)

    # Load test data
    test_examples: List[SentimentExample] = read_sentiment_examples("data/test.txt")
    print("Preparing test BoW...")
    test_features: torch.Tensor = torch.stack(
        [bag_of_words(ex.words, vocab) for ex in test_examples]
    )
    test_labels: torch.Tensor = torch.tensor(
        [ex.label for ex in test_examples], dtype=torch.float32
    )

    # Evaluate Naive Bayes model
    nb_predictions: List[int] = [nb_model.predict(ex) for ex in test_features]
    nb_metrics: Dict[str, float] = evaluate_classification(
        torch.tensor(nb_predictions), test_labels
    )
    print("Naive Bayes Metrics:", nb_metrics)

    # Evaluate Logistic Regression model
    lr_predictions: List[int] = lr_model.predict(test_features)
    lr_metrics: Dict[str, float] = evaluate_classification(lr_predictions, test_labels)
    print("Logistic Regression Metrics:", lr_metrics)

    print("Analyzing words and their weights:\n")

    for word in [
        "charming",
        "great",
        "awesome",
        "best",
        "disgusting",
        "stupid",
        "pathetic",
        "worst",
    ]:
        print("Word:", word)
        print("Weight associated to word:", get_word_weight(lr_model, vocab, word))


if __name__ == "__main__":
    main()
