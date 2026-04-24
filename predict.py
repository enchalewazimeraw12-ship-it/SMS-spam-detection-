import argparse
import joblib
import os

PIPELINE_FILE = os.path.join('models', 'spam_pipeline.pkl')


def load_pipeline():
    if not os.path.exists(PIPELINE_FILE):
        raise FileNotFoundError(
            "Model pipeline not found. Run 'python train_model.py' first."
        )
    return joblib.load(PIPELINE_FILE)


def main():
    parser = argparse.ArgumentParser(description='Classify an SMS message as spam or ham.')
    parser.add_argument('message', nargs='+', help='The SMS message text to classify.')
    args = parser.parse_args()

    message = ' '.join(args.message)
    pipeline = load_pipeline()

    prediction = pipeline.predict([message])[0]
    probability = pipeline.predict_proba([message])[0]
    spam_score = probability[1]
    ham_score = probability[0]

    print(f"Message: {message}")
    print(f"Prediction: {'SPAM' if prediction == 1 else 'NOT SPAM'}")
    print(f"Spam probability: {spam_score:.2%}")
    print(f"Ham probability: {ham_score:.2%}")


if __name__ == '__main__':
    main()
