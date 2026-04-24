import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

# Load your trained pipeline
MODEL_DIR = "models"
PIPELINE_FILE = os.path.join(MODEL_DIR, "spam_pipeline.pkl")

def load_pipeline():
    if not os.path.exists(PIPELINE_FILE):
        raise FileNotFoundError("Model not found. Run train_model.py first.")
    return joblib.load(PIPELINE_FILE)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot a confusion matrix using seaborn heatmap
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham (Predicted)', 'Spam (Predicted)'],
                yticklabels=['Ham (Actual)', 'Spam (Actual)'])

    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    # Add text annotations with percentages
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    # Add summary text
    plt.figtext(0.02, 0.02, f'Accuracy: {(tp+tn)/total:.1%}\n'
                            f'Precision: {tp/(tp+fp):.1%}\n'
                            f'Recall: {tp/(tp+fn):.1%}',
               fontsize=10, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    return plt

def evaluate_with_visualization():
    """
    Load test data and create confusion matrix visualization
    """
    # Load dataset
    df = pd.read_csv('SMSSpamCollection.csv')
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['message']
    y = df['label']

    # Split data (same as training)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load model and predict
    pipeline = load_pipeline()
    y_pred = pipeline.predict(X_test)

    # Create and show confusion matrix
    plt = plot_confusion_matrix(y_test, y_pred, "Spam Detection Confusion Matrix")
    # plt.show()  # Commented out for headless environment

    print("Confusion matrix saved as 'confusion_matrix.png'")
    print("Open the file to view the visualization.")

if __name__ == '__main__':
    evaluate_with_visualization()