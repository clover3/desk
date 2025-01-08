import fire
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rule_gen.open_ai_mod.knn import SentenceEncoder
from rule_gen.open_ai_mod.train_proto import get_data_arguments
from rule_gen.reddit.colbert.train_common import apply_tokenize
from rule_gen.reddit.proto.train_proto_reddit import apply_tokenize
from rule_gen.reddit.proto.train_proto_reddit import get_tokenize_formatter
from rule_gen.reddit.train_common import ClfDatasetLoader, get_datasets_from_dataset_arg


def main(debug=False):
    # Initialize encoder and load datasets
    encoder = SentenceEncoder()
    dataset_builder = ClfDatasetLoader()
    dataset_args = get_data_arguments(debug)
    train_dataset, eval_dataset = get_datasets_from_dataset_arg(dataset_builder, dataset_args)

    # Tokenize datasets
    tokenize_format = get_tokenize_formatter(encoder.tokenizer, dataset_args.max_length)
    tokenized_train, tokenized_eval = apply_tokenize(
        train_dataset, eval_dataset, tokenize_format)

    # Encode and get labels for both train and eval sets
    X_train, y_train = encoder.encode_and_label(tokenized_train)
    X_eval, y_eval = encoder.encode_and_label(tokenized_eval)

    # Create and train the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1))
    ])

    # Fit the pipeline on training data
    pipeline.fit(X_train, y_train)

    # Make predictions on evaluation set
    y_pred = pipeline.predict(X_eval)

    # Calculate and print evaluation metrics
    accuracy = accuracy_score(y_eval, y_pred)
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")

    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_eval, y_pred))


if __name__ == "__main__":
    fire.Fire(main)
