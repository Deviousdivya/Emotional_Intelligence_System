<div align="center">

# 🧠 Emotional Intelligence System
### Reflective Journal Analysis, Emotion & Intensity Prediction, and Confidence-Aware Action Guidance

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB6420?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Turn a short journal reflection plus daily context into an emotional state, an intensity score, and one practical next step, and say so honestly when the model is unsure.</b>
</p>

<p align="center">
  🌐 <b>Live demo:</b> <a href="https://emotional-intelligence-system.onrender.com">emotional-intelligence-system.onrender.com</a>
  <br />
  <sub>Hosted on Render's free tier, so the first visit after a quiet period can take about a minute to wake up.</sub>
</p>

[Key Features](#-key-features) • [System Architecture](#-system-architecture) • [Machine Learning Pipeline](#-machine-learning-pipeline) • [Decision Engine](#-action-decision-engine) • [Quick Start](#-quick-start-guide) • [API Reference](#-api-reference)

---

</div>

## 📌 Overview

People rarely describe how they feel in a single clean sentence. A journal entry such as *"I tried to relax, yet my thoughts kept racing"* carries signal in its wording, but also in how little they slept, how much energy they have and what time of day it is.

**Emotional Intelligence System** reads a short reflection **together with** that daily context, estimates an **emotional state** and an **intensity from 1 to 5**, attaches a **confidence score**, and passes the result through a deterministic **decision layer** that suggests *what* to do and *when* to do it. When confidence is low, the system flags the prediction as uncertain instead of pretending to know.

```
        ┌─────────────────────────────────────────────────────────────┐
        │                      User check-in                          │
        │  Journal text + sleep, energy, stress, time of day, ambience │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────┐
                │          Input validation & cleaning       │
                │  length limits · clamped ranges · defaults  │
                └─────────────────────┬────────────────────┘
                                      │
                                      ▼
        ┌──────────────────────────────────────────────────────────┐
        │             Hybrid Naive Bayes (two models)              │
        ├───────────────────────────┬──────────────────────────────┤
        │  Emotional state (6 class) │   Intensity (1–5, 5 class)   │
        │  token frequencies (text)  │   categorical likelihoods    │
        │  + Gaussian (numeric)      │   + Gaussian (numeric)       │
        └─────────────┬─────────────┴──────────────┬───────────────┘
                      │                            │
                      ▼                            ▼
              state probabilities          intensity probabilities
                      │                            │
                      └────────────┬───────────────┘
                                   ▼
                ┌──────────────────────────────────────────┐
                │          Action decision engine            │
                │   state × intensity × energy × time of day │
                ├──────────────────────┬───────────────────┤
                │ what_to_do / when_to_do │ confidence < 0.6 →  │
                │ + support message       │ uncertain_flag      │
                └──────────────────────┴───────────────────┘
```

---

## ✨ Key Features

- **🧩 Multimodal-style input**: combines free text with sleep hours, energy, stress, session length, time of day, previous-day mood, ambience type, a facial-expression hint and a self-rated reflection quality.
- **🎯 Two predictions at once**: a 6-class **emotional state** (*calm, focused, neutral, restless, mixed, overwhelmed*) and a 5-level **intensity**.
- **📊 Calibrated honesty**: every result includes a confidence score, the full probability breakdown for all states and intensities, and an `uncertain_flag` when the top state is below 60%.
- **🧭 Deterministic action layer**: transparent rules turn the prediction into a concrete next step (for example box breathing now, grounding within 15 minutes, rest, deep work) with a suggested timing.
- **🔒 Privacy-first by design**: journal text is processed in memory only. The server neither stores it nor writes it to request logs, and the browser history keeps only the state, intensity, confidence and timestamp.
- **🪶 Zero third-party runtime dependencies**: the web app runs on Python's standard library, with its own compact Naive Bayes implementation, so it starts instantly and deploys anywhere.
- **🌿 Accessible, responsive interface**: check-in, insights and a model explainer, with a calm design, keyboard support and a mobile layout.
- **🔬 Research notebook included**: `Model.ipynb` holds the original TF-IDF + XGBoost experiments that the web model was derived from.

---

## 🏗️ System Architecture

```
Emotional_Intelligence_System/
├── app.py                 # Standard-library HTTP server + JSON API
├── emotion_model.py       # HybridNaiveBayes, EmotionModel and the decision engine
├── templates/
│   └── index.html         # Check-in, insights and "how it works" pages
├── static/
│   ├── css/styles.css     # Responsive, accessible styles
│   └── js/app.js          # Browser interactions and local check-in history
├── train_data.csv         # 1,200 labelled reflections used to train the web model
├── test_data.csv          # 120 unlabelled inputs for batch inference
├── predictions.csv        # Example batch predictions from the research model
├── Model.ipynb            # TF-IDF + XGBoost experiments and analysis
├── Edge.md                # Edge / mobile deployment plan and trade-offs
├── render.yaml            # One-file Render deployment
├── requirements.txt       # (no packages: standard library only)
└── LICENSE                # Apache-2.0
```

---

## 📊 Machine Learning Pipeline

### Training data
`train_data.csv` contains **1,200 labelled reflections**, roughly balanced across six emotional states and five intensity levels.

| State | Samples | State | Samples |
| :--- | ---: | :--- | ---: |
| calm | 216 | focused | 193 |
| restless | 209 | mixed | 191 |
| neutral | 201 | overwhelmed | 190 |

### Features

| Type | Features |
| :--- | :--- |
| **Text** | `journal_text`, tokenised into lowercase word tokens |
| **Categorical** | `ambience_type`, `time_of_day`, `previous_day_mood`, `face_emotion_hint`, `reflection_quality` |
| **Numeric** | `sleep_hours`, `energy_level`, `stress_level`, `duration_min`, plus derived `text_len` and `word_count` |

### Model
The website uses a **hybrid Naive Bayes classifier** written with the standard library only. It multiplies three kinds of evidence: token frequencies for the text, categorical likelihoods for the context fields and Gaussian likelihoods for the numeric fields. Two instances are trained at start-up, one for `emotional_state` and one for `intensity`.

The notebook explores a heavier approach (TF-IDF text vectors with XGBoost). The simpler model powers the live demo because it needs no installation, starts in well under a second and keeps inference latency very low. See [`Edge.md`](Edge.md) for the full trade-off discussion, including a path to on-device deployment.

### Robustness
- **Missing or out-of-range values** fall back to sensible defaults and are clamped (for example sleep 0–24 h, energy and stress 1–5).
- **Very short or vague text** still produces a prediction because the context features carry signal.
- **Low confidence** (top-state probability below 0.6) raises `uncertain_flag` so the interface can soften its wording.
- **Input limits**: reflections are capped at 5,000 characters and request bodies at 128 KB.

---

## 🧭 Action Decision Engine

A prediction becomes advice through explicit, auditable rules (`EmotionModel._recommendation`):

| Condition | Action → timing |
| :--- | :--- |
| Intensity ≥ 4 and *overwhelmed* / *stressed* | `box_breathing` → now |
| Intensity ≥ 4 and *restless* | `grounding` → within 15 min |
| Intensity ≥ 4 and *focused* | `deep_work` → now |
| Intensity ≥ 4 and *calm* | `deep_work` → within 15 min |
| Intensity ≥ 4 and *mixed* | `journaling` → now |
| Energy ≤ 2 (otherwise) | `rest` → now |
| *neutral* | `light_planning` → later today |
| *restless* | `movement` → within 15 min |
| *calm* / *focused* | `deep_work` → later today / now |
| Night time | `sound_therapy` → tonight |
| Anything else | `light_planning` → later today |

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.9+** (3.11 recommended). No packages need to be installed.

### Step 1: Clone the repository

```bash
git clone https://github.com/Deviousdivya/Emotional_Intelligence_System.git
cd Emotional_Intelligence_System
```

### Step 2: Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:8001**. The model trains from `train_data.csv` when the server starts. If the port is busy, set `PORT` to a free one:

```bash
# macOS / Linux
PORT=8080 python app.py

# Windows (PowerShell)
$env:PORT = 8080; python app.py
```

### Step 3 (optional): Explore the research notebook

```bash
pip install jupyter scikit-learn xgboost pandas
jupyter notebook Model.ipynb
```

### ☁️ Deploy on Render
The repository includes [`render.yaml`](render.yaml). Create a new **Web Service** from the repo and Render will run `python app.py`, binding to the `PORT` it provides. No environment variables are required.

---

## 🔌 API Reference

### `GET /api/overview`
Returns what the model was trained on.

```json
{
  "training_samples": 1200,
  "state_counts": { "calm": 216, "focused": 193, "mixed": 191, "neutral": 201, "overwhelmed": 190, "restless": 209 },
  "states": ["calm", "focused", "mixed", "neutral", "overwhelmed", "restless"],
  "features": ["journal text", "sleep", "energy", "stress", "session context"]
}
```

### `POST /api/analyze`
Analyses one check-in. Only `journal_text` is required; every other field has a default.

**Request**

```json
{
  "journal_text": "My thoughts keep racing and I cannot switch off, even though I tried to relax with the forest sounds.",
  "ambience_type": "forest",
  "time_of_day": "evening",
  "previous_day_mood": "calm",
  "face_emotion_hint": "tired_face",
  "reflection_quality": "vague",
  "sleep_hours": 6,
  "energy_level": 2,
  "stress_level": 4,
  "duration_min": 35
}
```

**Response** (real output from the model)

```json
{
  "predicted_state": "restless",
  "predicted_intensity": 4,
  "confidence": 0.9939,
  "uncertain_flag": false,
  "what_to_do": "grounding",
  "when_to_do": "within_15_min",
  "support_message": "It sounds like restless is showing up strongly. Take a moment for grounding.",
  "state_scores": { "restless": 99.4, "focused": 0.3, "calm": 0.2, "overwhelmed": 0.0, "neutral": 0.0, "mixed": 0.0 },
  "intensity_scores": { "1": 5.5, "2": 12.0, "3": 25.5, "4": 54.8, "5": 2.2 }
}
```

Errors return `400` with `{ "error": "..." }` (empty text, text over 5,000 characters, unreadable body).

---

## 🛠️ Technology Stack

| Domain | Technologies |
| :--- | :--- |
| **Backend** | Python standard library (`http.server`, `json`, `csv`, `urllib`) |
| **Machine Learning** | Custom hybrid Naive Bayes (text + categorical + Gaussian); research notebook with scikit-learn, TF-IDF and XGBoost |
| **Frontend** | Semantic HTML, responsive CSS, vanilla JavaScript, `localStorage` for private history |
| **Deployment** | Render (free web service) via `render.yaml` |

---

## ⚠️ Limitations & Responsible Use

This is a **reflective tool, not a clinical assessment**. It is not a diagnostic device and must not replace professional help. Predictions can be wrong, especially for short or ambiguous reflections, and the training data is a small, curated set. If you are struggling, please reach out to a qualified professional or a local helpline.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'feat: Add AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a **Pull Request**.

---

## 📄 License

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.
