"""Small, dependency-free hybrid Naive Bayes model for the local web app."""

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import re


TEXT_FIELD = "journal_text"
CATEGORY_FEATURES = (
    "ambience_type",
    "time_of_day",
    "previous_day_mood",
    "face_emotion_hint",
    "reflection_quality",
)
NUMERIC_FEATURES = (
    "sleep_hours",
    "energy_level",
    "stress_level",
    "duration_min",
    "text_len",
    "word_count",
)
TOKEN_PATTERN = re.compile(r"[a-z']+")


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text or "").lower())


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


class HybridNaiveBayes:
    """Multinomial text + categorical + Gaussian numeric Naive Bayes."""

    def __init__(self, rows, target, alpha=0.8):
        self.target = target
        self.alpha = alpha
        self.class_counts = Counter()
        self.text_counts = defaultdict(Counter)
        self.text_totals = Counter()
        self.category_counts = {
            feature: defaultdict(Counter) for feature in CATEGORY_FEATURES
        }
        self.category_values = {
            feature: set() for feature in CATEGORY_FEATURES
        }
        self.numeric_values = {
            feature: defaultdict(list) for feature in NUMERIC_FEATURES
        }
        self.vocabulary = set()

        for row in rows:
            label = str(row.get(target, "")).strip()
            if not label:
                continue
            self.class_counts[label] += 1

            tokens = tokenize(row.get(TEXT_FIELD, ""))
            self.text_counts[label].update(tokens)
            self.text_totals[label] += len(tokens)
            self.vocabulary.update(tokens)

            for feature in CATEGORY_FEATURES:
                value = str(row.get(feature, "")).strip().lower()
                if value:
                    self.category_counts[feature][label][value] += 1
                    self.category_values[feature].add(value)

            text = str(row.get(TEXT_FIELD, "") or "")
            values = {
                "text_len": len(text),
                "word_count": len(tokenize(text)),
            }
            for feature in NUMERIC_FEATURES[:4]:
                values[feature] = number(row.get(feature))
            for feature, value in values.items():
                if value is not None:
                    self.numeric_values[feature][label].append(value)

        self.classes = sorted(self.class_counts)
        self.total_rows = sum(self.class_counts.values())
        self.numeric_stats = {}
        for feature, by_class in self.numeric_values.items():
            self.numeric_stats[feature] = {}
            for label in self.classes:
                values = by_class[label]
                if not values:
                    self.numeric_stats[feature][label] = (0.0, 1.0)
                    continue
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                self.numeric_stats[feature][label] = (mean, max(variance, 0.12))

    def predict_proba(self, row):
        scores = {}
        vocab_size = max(1, len(self.vocabulary) + 1)
        token_list = tokenize(row.get(TEXT_FIELD, ""))
        token_counts = Counter(token_list)

        for label in self.classes:
            class_total = self.class_counts[label]
            score = math.log(class_total / self.total_rows)

            # A modest text weight leaves room for the contextual signals.
            token_denominator = self.text_totals[label] + self.alpha * vocab_size
            for token, count in token_counts.items():
                token_probability = (
                    self.text_counts[label][token] + self.alpha
                ) / token_denominator
                score += 0.65 * count * math.log(token_probability)

            for feature in CATEGORY_FEATURES:
                value = str(row.get(feature, "")).strip().lower()
                if not value:
                    continue
                possible_values = max(1, len(self.category_values[feature]) + 1)
                denominator = class_total + self.alpha * possible_values
                count = self.category_counts[feature][label][value]
                probability = (count + self.alpha) / denominator
                score += 0.8 * math.log(probability)

            for feature in NUMERIC_FEATURES:
                value = number(row.get(feature))
                if value is None:
                    continue
                mean, variance = self.numeric_stats[feature][label]
                log_likelihood = -0.5 * (
                    math.log(2 * math.pi * variance) + ((value - mean) ** 2 / variance)
                )
                score += 0.18 * log_likelihood

            scores[label] = score

        peak = max(scores.values())
        exponentials = {label: math.exp(value - peak) for label, value in scores.items()}
        total = sum(exponentials.values()) or 1.0
        return {label: value / total for label, value in exponentials.items()}

    def predict(self, row):
        probabilities = self.predict_proba(row)
        label = max(probabilities, key=probabilities.get)
        return label, probabilities


class EmotionModel:
    def __init__(self, csv_path):
        with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError("train_data.csv does not contain any training rows")

        self.state_model = HybridNaiveBayes(self.rows, "emotional_state")
        self.intensity_model = HybridNaiveBayes(self.rows, "intensity")
        counts = Counter(row.get("emotional_state", "") for row in self.rows)
        self.state_counts = dict(sorted(counts.items()))

    @staticmethod
    def _prepare_input(payload):
        raw_text = str(payload.get("journal_text", "")).strip()
        if not raw_text:
            raise ValueError("Write a few words about how you are feeling first.")
        if len(raw_text) > 5000:
            raise ValueError("Please keep your reflection under 5,000 characters.")

        row = {
            "journal_text": raw_text,
            "ambience_type": str(payload.get("ambience_type", "none"))[:80],
            "time_of_day": str(payload.get("time_of_day", "afternoon"))[:40],
            "previous_day_mood": str(payload.get("previous_day_mood", "unknown"))[:60],
            "face_emotion_hint": str(payload.get("face_emotion_hint", "unknown"))[:60],
            "reflection_quality": str(payload.get("reflection_quality", "clear"))[:40],
        }
        defaults = {
            "sleep_hours": (7, 0, 24),
            "energy_level": (3, 1, 5),
            "stress_level": (3, 1, 5),
            "duration_min": (10, 0, 240),
        }
        for feature, (default, low, high) in defaults.items():
            value = number(payload.get(feature))
            if value is None:
                value = default
            row[feature] = min(high, max(low, value))
        row["text_len"] = len(raw_text)
        row["word_count"] = len(tokenize(raw_text))
        return row

    @staticmethod
    def _recommendation(state, intensity, energy, time_of_day):
        if intensity >= 4:
            if state in ("overwhelmed", "stressed"):
                return "box_breathing", "now"
            elif state == "restless":
                return "grounding", "within_15_min"
            elif state == "focused":
                return "deep_work", "now"
            elif state == "calm":
                return "deep_work", "within_15_min"
            elif state == "mixed":
                return "journaling", "now"

        if energy <= 2:
            return "rest", "now"
        if state == "neutral":
            return "light_planning", "later_today"
        if state == "restless":
            return "movement", "within_15_min"
        if state == "calm":
            return "deep_work", "later_today"
        if state == "focused":
            return "deep_work", "now"
        if time_of_day == "night":
            return "sound_therapy", "tonight"
        return "light_planning", "later_today"

    def analyze(self, payload):
        row = self._prepare_input(payload)
        state, state_probabilities = self.state_model.predict(row)
        intensity_label, intensity_probabilities = self.intensity_model.predict(row)
        intensity = min(5, max(1, int(float(intensity_label))))
        confidence = state_probabilities[state]
        action, timing = self._recommendation(
            state,
            intensity,
            row["energy_level"],
            row["time_of_day"],
        )

        if intensity >= 4:
            support = f"It sounds like {state} is showing up strongly. Take a moment for {action.replace('_', ' ')}."
        elif intensity >= 3:
            support = f"You may be feeling {state}. A small step like {action.replace('_', ' ')} could help."
        else:
            support = "You seem to have a little steadiness right now. Keep the next step gentle and manageable."

        return {
            "predicted_state": state,
            "predicted_intensity": intensity,
            "confidence": round(confidence, 4),
            "uncertain_flag": confidence < 0.6,
            "what_to_do": action,
            "when_to_do": timing,
            "support_message": support,
            "state_scores": {
                name: round(probability * 100, 1)
                for name, probability in sorted(
                    state_probabilities.items(), key=lambda pair: pair[1], reverse=True
                )
            },
            "intensity_scores": {
                str(name): round(probability * 100, 1)
                for name, probability in sorted(
                    intensity_probabilities.items(), key=lambda pair: int(pair[0])
                )
            },
        }
