# SMS Spam Detector

This project is a working SMS spam detector using Python, scikit-learn, and Streamlit.

## Project Structure

- `app.py`: Streamlit app for live spam classification.
- `train_model.py`: Trains the spam detection pipeline from local dataset files.
- `predict.py`: Command-line tool to classify a single SMS message.
- `text_utils.py`: Shared text preprocessing utilities.
- `requirements.txt`: Pin dependencies for the project.
- `README.md`: Project documentation.
- `run_train.bat`: Windows helper script to install dependencies and train the model.
- `run_app.bat`: Windows helper script to start the Streamlit app.
- `data/`: Recommended folder for dataset files.
- `models/`: Saved trained pipeline and model files.

## Setup Instructions

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Place the dataset file locally in one of these locations:

- `data/SMSSpamCollection`
- `SMSSpamCollection`

3. Train the model:

```bash
python train_model.py
```

4. Run the Streamlit app:

```bash
python -m streamlit run app.py
```

5. Or use the Windows helper scripts:

```bat
run_train.bat
run_app.bat
```

## Dataset Handling

- The project prefers dataset files in `data/`.
- If `data/SMSSpamCollection.csv` exists, it is loaded directly.
- If only `data/SMSSpamCollection` exists, it is converted automatically to `data/SMSSpamCollection.csv`.
- Root dataset files are still supported for backwards compatibility.

## Prediction CLI

Classify a single SMS message from the command line:

```bash
python predict.py "Your SMS message here"
```

## Dependencies

The project depends on:

- `numpy`
- `pandas`
- `scikit-learn`
- `nltk`
- `streamlit`
- `joblib`

## Notes

- This project uses only local dataset files and does not download data during training.
- The trained pipeline is saved to `models/spam_pipeline.pkl`.
- The Streamlit app uses the saved pipeline and shows confidence for each prediction.
