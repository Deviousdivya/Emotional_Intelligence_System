# Emotional Intelligence System

A machine learning project that reads a short reflection alongside daily context, estimates an emotional state and intensity, and offers a practical next step. The repository now includes a local website for trying the complete workflow.

## Run the website

The web app uses Python 3.9+ and only the standard library, so no packages need to be installed. From this directory, run:

```bash
python app.py
```

Then open [http://127.0.0.1:8001](http://127.0.0.1:8001). The model trains from `train_data.csv` when the server starts. If that port is busy, set `PORT` to an available local port before running the app.

## What the website does

- Takes a journal reflection and context such as sleep, energy, stress, time of day, and session ambience.
- Predicts an emotional state and an intensity from 1 to 5 with a confidence score.
- Flags low-confidence predictions and shows the other state probabilities.
- Applies the project's decision rules to suggest an action and when to try it.
- Shows the training set's actual state distribution and check-in history stored locally in the browser.

The web app trains a compact hybrid Naive Bayes model on the included labeled data using Python's standard library. It uses token frequencies for text, categorical likelihoods for context, and Gaussian likelihoods for numeric features. This makes the demo straightforward to run without package installation. The original `Model.ipynb` contains the project's separate scikit-learn/XGBoost experiments.

Journal text is sent only to the local server at `127.0.0.1` for inference. The server does not store it or include it in request logs. Browser history stores only the predicted state, intensity, confidence, and timestamp; it never stores journal text.

## Project files

- `app.py` — local HTTP server and JSON API.
- `emotion_model.py` — trains the hybrid model and applies the action decision layer.
- `templates/index.html` — accessible check-in, insights, and model explainer pages.
- `static/` — responsive styles and browser interactions.
- `train_data.csv` — labeled reflections used to train the website model.
- `test_data.csv`, `predictions.csv`, `Model.ipynb` — project inference data and experiments.

## Model output

Each analysis returns `predicted_state`, `predicted_intensity`, `confidence`, `uncertain_flag`, `what_to_do`, `when_to_do`, and `support_message`, along with class probabilities for the interface.

The system is a reflective tool, not a clinical assessment. Its predictions can be wrong, especially for short or ambiguous reflections.
