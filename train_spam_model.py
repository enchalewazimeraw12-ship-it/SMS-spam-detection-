import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# Load the dataset
df = pd.read_csv('SMSSpamCollection.csv')

# Assuming the columns are 'label' and 'message'
# Map labels to binary: ham=0, spam=1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Prepare data
X = df['message']
y = df['label']

# Create TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Create LogisticRegression model
model = LogisticRegression(random_state=42)

# Create pipeline (optional, but useful)
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', model)
])

# Train the pipeline
pipeline.fit(X, y)

# Save the pipeline as spam_pipeline.pkl (to match the app)
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(pipeline, 'models/spam_pipeline.pkl')

# If you want separate files as requested:
# Save vectorizer
joblib.dump(vectorizer, 'models/vectorizer.joblib')

# Save model (but note: model needs vectorizer for prediction)
# For separate, you'd need to transform X first
X_transformed = vectorizer.fit_transform(X)
model.fit(X_transformed, y)
joblib.dump(model, 'models/spam_model.joblib')

print("Model training complete. Saved as models/spam_pipeline.pkl, models/vectorizer.joblib, and models/spam_model.joblib")