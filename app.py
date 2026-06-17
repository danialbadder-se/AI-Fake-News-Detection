"""
Fake News Detection Website — VeriCleri
Main Flask Application
"""
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import json
import os
from datetime import datetime, timedelta
import re
from pathlib import Path
import csv
import html
import urllib.parse
import urllib.request
import ssl
import xml.etree.ElementTree as ET
from threading import Lock

# Load .env file (SECRET_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests as http_requests
    from bs4 import BeautifulSoup
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

# Optional rate-limiting
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_AVAILABLE = True
except ImportError:
    LIMITER_AVAILABLE = False

# Optional background scheduler (auto-retraining)
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from utils.dataset_ingest import ingest_external_datasets

app = Flask(__name__)
# Secret key loaded from .env — never hardcode in production
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24).hex())
app.config['TEMPLATES_AUTO_RELOAD'] = True
ADMIN_API_KEY = os.getenv('ADMIN_API_KEY', '').strip()

# ── Rate Limiter ────────────────────────────────────────────────────────────
if LIMITER_AVAILABLE:
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[],
        storage_uri='memory://',
    )
    def _rl(limit_str):
        return limiter.limit(limit_str)
else:
    def _rl(limit_str):
        def decorator(f):
            return f
        return decorator


def _has_admin_key():
    if not ADMIN_API_KEY:
        return True
    header_key = request.headers.get('X-Admin-Key', '').strip()
    auth_header = request.headers.get('Authorization', '').strip()
    if auth_header.lower().startswith('bearer '):
        header_key = header_key or auth_header[7:].strip()
    return header_key == ADMIN_API_KEY


def _require_admin():
    if _has_admin_key():
        return None
    return jsonify({'success': False, 'message': 'Admin key required'}), 401

# Maximum text length accepted by /analyze (prevents abuse / memory spikes)
MAX_TEXT_LENGTH = 15_000  # ~3 000 words

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'data' / 'sample_data.csv'
USER_DATASET_PATH = BASE_DIR / 'data' / 'user_training_data.csv'
TRAINING_DIR = BASE_DIR / 'data' / 'training'
RAW_DATASET_DIR = BASE_DIR / 'data' / 'raw_datasets'
LATEST_REAL_DATASET_PATH = TRAINING_DIR / 'latest_reliable_news.csv'
MODEL_DIR = BASE_DIR / 'ml_model' / 'sample_models'
MODEL_PATH = MODEL_DIR / 'text_authenticity_model.joblib'
FEEDBACK_PATH = BASE_DIR / 'data' / 'feedback.json'
STATS_PATH    = BASE_DIR / 'data' / 'stats.json'


def _load_stats():
    if STATS_PATH.exists():
        try:
            with open(STATS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'total_analyses': 0, 'last_analysis': None}


def _increment_analysis():
    stats = _load_stats()
    stats['total_analyses'] = stats.get('total_analyses', 0) + 1
    stats['last_analysis'] = datetime.now().isoformat()
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats, f)


class TextAuthenticityModel:
    def __init__(self):
        self.pipeline = None
        self.word_pipeline = None
        self.char_pipeline = None
        self.metrics = {}
        self.dataset_size = 0
        self.last_error = None

    def load_or_train(self):
        if self._load_model():
            return True
        return self.train_model(force=True)

    def _load_model(self):
        try:
            if not MODEL_PATH.exists():
                return False

            payload = joblib.load(MODEL_PATH)
            self.word_pipeline = payload.get('word_pipeline')
            self.char_pipeline = payload.get('char_pipeline')

            # Backward compatibility with older single-pipeline format
            if self.word_pipeline is None and payload.get('pipeline') is not None:
                self.word_pipeline = payload.get('pipeline')

            self.pipeline = {
                'word': self.word_pipeline,
                'char': self.char_pipeline,
            }
            self.metrics = payload.get('metrics', {})
            self.dataset_size = payload.get('dataset_size', 0)

            if self.word_pipeline is None and self.char_pipeline is None:
                return False

            return True
        except Exception as error:
            self.last_error = f"Model load failed: {str(error)}"
            return False

    def train_model(self, force=False):
        try:
            dataset = self._load_dataset()
            required_columns = {'text', 'is_fake'}
            if not required_columns.issubset(set(dataset.columns)):
                self.last_error = "Dataset must contain 'text' and 'is_fake' columns"
                return False

            dataset = dataset[['text', 'is_fake']].dropna().copy()
            dataset['text'] = dataset['text'].astype(str)
            dataset['is_fake'] = dataset['is_fake'].astype(int)
            dataset = dataset[dataset['text'].str.len() >= 10]
            dataset['text'] = dataset['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
            dataset = dataset.drop_duplicates(subset=['text', 'is_fake'])

            if len(dataset) < 8:
                fallback_dataset = pd.DataFrame(self._fallback_training_samples())
                dataset = pd.concat([dataset, fallback_dataset], ignore_index=True)

            if len(dataset) < 40:
                synthetic_dataset = pd.DataFrame(self._synthetic_training_samples())
                max_synthetic_rows = min(max(len(dataset) * 2, 12), 30)
                synthetic_dataset = synthetic_dataset.sample(
                    n=min(max_synthetic_rows, len(synthetic_dataset)),
                    random_state=42,
                )
                dataset = pd.concat([dataset, synthetic_dataset], ignore_index=True)
                dataset = dataset.drop_duplicates(subset=['text', 'is_fake'])

            if len(dataset) < 8:
                self.last_error = 'Dataset is too small to train a model (minimum 8 usable samples)'
                return False

            label_counts = dataset['is_fake'].value_counts()
            if len(label_counts) < 2:
                self.last_error = 'Dataset must contain both fake and real samples'
                return False

            # Guardrail: if classes are too imbalanced, augment minority using synthetic samples
            real_count = int((dataset['is_fake'] == 0).sum())
            fake_count = int((dataset['is_fake'] == 1).sum())
            majority_count = max(real_count, fake_count)
            minority_label = 0 if real_count < fake_count else 1
            minority_count = min(real_count, fake_count)

            if majority_count > 0 and (minority_count / majority_count) < 0.35:
                synthetic_rows = pd.DataFrame(self._synthetic_training_samples())
                synthetic_rows = synthetic_rows[synthetic_rows['is_fake'] == minority_label]
                needed = max(int((0.4 * majority_count) - minority_count), 0)
                if needed > 0 and not synthetic_rows.empty:
                    synthetic_rows = synthetic_rows.sample(
                        n=min(needed, len(synthetic_rows)),
                        random_state=42,
                        replace=False,
                    )
                    dataset = pd.concat([dataset, synthetic_rows], ignore_index=True)
                    dataset = dataset.drop_duplicates(subset=['text', 'is_fake'])

                real_count = int((dataset['is_fake'] == 0).sum())
                fake_count = int((dataset['is_fake'] == 1).sum())

            # Balance classes (cap majority class to 2x minority for stable training)
            minimum_class_count = int(label_counts.min())
            maximum_per_class = max(minimum_class_count * 2, minimum_class_count)
            balanced_parts = []
            for class_value in sorted(dataset['is_fake'].unique()):
                class_subset = dataset[dataset['is_fake'] == class_value]
                if len(class_subset) > maximum_per_class:
                    class_subset = class_subset.sample(n=maximum_per_class, random_state=42)
                balanced_parts.append(class_subset)
            dataset = pd.concat(balanced_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

            x_values = dataset['text'].tolist()
            y_values = dataset['is_fake'].tolist()

            # Keep training responsive on large datasets
            max_training_rows = 6000
            if len(dataset) > max_training_rows:
                sampled = dataset.sample(n=max_training_rows, random_state=42)
                x_values = sampled['text'].tolist()
                y_values = sampled['is_fake'].tolist()

            split_size = 0.25 if len(dataset) >= 20 else 0.2
            stratify_values = y_values if min(label_counts.values) >= 2 else None

            try:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_values,
                    y_values,
                    test_size=split_size,
                    random_state=42,
                    stratify=stratify_values,
                )
            except ValueError:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_values,
                    y_values,
                    test_size=split_size,
                    random_state=42,
                    stratify=None,
                )

            word_pipeline = Pipeline([
                (
                    'tfidf',
                    TfidfVectorizer(
                        stop_words='english',
                        ngram_range=(1, 3),
                        min_df=1,
                        max_df=0.98,
                        sublinear_tf=True,
                    ),
                ),
                (
                    'classifier',
                    LogisticRegression(
                        max_iter=2000,
                        class_weight='balanced',
                        random_state=42,
                    ),
                ),
            ])

            char_pipeline = Pipeline([
                (
                    'tfidf',
                    TfidfVectorizer(
                        analyzer='char_wb',
                        ngram_range=(3, 5),
                        min_df=1,
                        max_df=1.0,
                        sublinear_tf=True,
                    ),
                ),
                (
                    'classifier',
                    LogisticRegression(
                        max_iter=2500,
                        class_weight='balanced',
                        random_state=42,
                    ),
                ),
            ])

            min_class_count = min(pd.Series(y_train).value_counts().min(), 5)
            use_grid_search = min_class_count >= 2 and len(x_train) >= 10 and len(x_train) <= 2500
            if use_grid_search:
                grid = GridSearchCV(
                    estimator=word_pipeline,
                    param_grid={
                        'classifier__C': [0.5, 1.0, 2.0, 4.0],
                        'classifier__solver': ['liblinear'],
                    },
                    cv=min_class_count,
                    scoring='f1_weighted',
                    n_jobs=1,
                )
                grid.fit(x_train, y_train)
                word_pipeline = grid.best_estimator_
            else:
                word_pipeline.fit(x_train, y_train)

            char_pipeline.fit(x_train, y_train)

            word_probabilities = word_pipeline.predict_proba(x_test)
            char_probabilities = char_pipeline.predict_proba(x_test)
            combined_probabilities = (word_probabilities + char_probabilities) / 2.0
            fake_probabilities = combined_probabilities[:, 1]
            base_predictions = (fake_probabilities >= 0.5).astype(int)

            thresholds = self._calibrate_thresholds(y_test, fake_probabilities)
            threshold_predictions = []
            for probability in fake_probabilities:
                if probability >= thresholds['fake_threshold']:
                    threshold_predictions.append(1)
                elif probability <= thresholds['real_threshold']:
                    threshold_predictions.append(0)
                else:
                    threshold_predictions.append(-1)

            decided_indices = [index for index, value in enumerate(threshold_predictions) if value != -1]
            decided_accuracy = None
            decided_f1 = None
            coverage = 0.0
            class_precision = {'real': None, 'fake': None}
            class_recall = {'real': None, 'fake': None}
            confusion = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}

            if decided_indices:
                decided_true = [y_test[index] for index in decided_indices]
                decided_pred = [threshold_predictions[index] for index in decided_indices]
                coverage = len(decided_indices) / max(len(y_test), 1)
                decided_accuracy = round(float(accuracy_score(decided_true, decided_pred) * 100), 2)
                decided_f1 = round(float(f1_score(decided_true, decided_pred, average='weighted', zero_division=0) * 100), 2)

                precision_values, recall_values, _, _ = precision_recall_fscore_support(
                    decided_true,
                    decided_pred,
                    labels=[0, 1],
                    average=None,
                    zero_division=0,
                )
                class_precision = {
                    'real': round(float(precision_values[0] * 100), 2),
                    'fake': round(float(precision_values[1] * 100), 2),
                }
                class_recall = {
                    'real': round(float(recall_values[0] * 100), 2),
                    'fake': round(float(recall_values[1] * 100), 2),
                }

                matrix = confusion_matrix(decided_true, decided_pred, labels=[0, 1])
                confusion = {
                    'tn': int(matrix[0, 0]),
                    'fp': int(matrix[0, 1]),
                    'fn': int(matrix[1, 0]),
                    'tp': int(matrix[1, 1]),
                }

            try:
                auc_score = round(float(roc_auc_score(y_test, combined_probabilities[:, 1]) * 100), 2)
            except Exception:
                auc_score = None

            self.metrics = {
                'accuracy': round(float(accuracy_score(y_test, base_predictions) * 100), 2),
                'f1_score': round(float(f1_score(y_test, base_predictions, average='weighted') * 100), 2),
                'roc_auc': auc_score,
                'trained_at': datetime.now().isoformat(),
                'train_samples': len(x_train),
                'test_samples': len(x_test),
                'dataset_samples': len(dataset),
                'decision_thresholds': {
                    'real_threshold': round(float(thresholds['real_threshold']), 3),
                    'fake_threshold': round(float(thresholds['fake_threshold']), 3),
                },
                'holdout_calibration': {
                    'coverage_percent': round(float(coverage * 100), 2),
                    'decided_accuracy': decided_accuracy,
                    'decided_f1_score': decided_f1,
                    'class_precision': class_precision,
                    'class_recall': class_recall,
                    'confusion_matrix': confusion,
                },
                'class_distribution': {
                    'real': int((dataset['is_fake'] == 0).sum()),
                    'fake': int((dataset['is_fake'] == 1).sum()),
                },
                'class_balance_ratio': round(
                    min(
                        int((dataset['is_fake'] == 0).sum()),
                        int((dataset['is_fake'] == 1).sum())
                    ) / max(
                        int((dataset['is_fake'] == 0).sum()),
                        int((dataset['is_fake'] == 1).sum()),
                        1,
                    ),
                    3,
                ),
            }

            self.word_pipeline = word_pipeline
            self.char_pipeline = char_pipeline
            self.pipeline = {
                'word': self.word_pipeline,
                'char': self.char_pipeline,
            }
            self.dataset_size = len(dataset)

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    'word_pipeline': self.word_pipeline,
                    'char_pipeline': self.char_pipeline,
                    'metrics': self.metrics,
                    'dataset_size': self.dataset_size,
                },
                MODEL_PATH,
            )

            self.last_error = None
            return True
        except Exception as error:
            self.last_error = f"Model training failed: {str(error)}"
            return False

    def _calibrate_thresholds(self, y_true, fake_probabilities):
        real_threshold_candidates = [value / 100 for value in range(30, 50)]
        fake_threshold_candidates = [value / 100 for value in range(51, 71)]
        best_score = float('-inf')
        best_thresholds = {
            'real_threshold': 0.45,
            'fake_threshold': 0.55,
        }

        for real_threshold in real_threshold_candidates:
            for fake_threshold in fake_threshold_candidates:
                if fake_threshold - real_threshold < 0.06:
                    continue

                predictions = []
                for probability in fake_probabilities:
                    if probability >= fake_threshold:
                        predictions.append(1)
                    elif probability <= real_threshold:
                        predictions.append(0)
                    else:
                        predictions.append(-1)

                decided_indices = [index for index, value in enumerate(predictions) if value != -1]
                if not decided_indices:
                    continue

                decided_true = [y_true[index] for index in decided_indices]
                decided_pred = [predictions[index] for index in decided_indices]

                coverage = len(decided_indices) / max(len(y_true), 1)
                decided_accuracy = accuracy_score(decided_true, decided_pred)
                decided_f1 = f1_score(decided_true, decided_pred, average='weighted', zero_division=0)

                precision_values, recall_values, _, _ = precision_recall_fscore_support(
                    decided_true,
                    decided_pred,
                    labels=[0, 1],
                    average=None,
                    zero_division=0,
                )
                real_recall = float(recall_values[0])
                fake_recall = float(recall_values[1])

                score = (decided_accuracy * 0.45) + (decided_f1 * 0.35) + (coverage * 0.20)
                if coverage < 0.55:
                    score -= (0.55 - coverage) * 0.5
                if min(real_recall, fake_recall) < 0.50:
                    score -= 0.08

                if score > best_score:
                    best_score = score
                    best_thresholds = {
                        'real_threshold': real_threshold,
                        'fake_threshold': fake_threshold,
                    }

        return best_thresholds

    def _load_dataset(self):
        dataframes = []

        candidate_files = [DATASET_PATH, USER_DATASET_PATH, LATEST_REAL_DATASET_PATH]
        if TRAINING_DIR.exists():
            candidate_files.extend(sorted(TRAINING_DIR.glob('*.csv')))

        for candidate_path in candidate_files:
            if not candidate_path.exists():
                continue
            frame = self._read_csv_flexible(candidate_path)
            if not frame.empty:
                dataframes.append(frame)

        if not dataframes:
            return pd.DataFrame(columns=['text', 'is_fake'])

        merged = pd.concat(dataframes, ignore_index=True)
        return merged

    def _read_csv_flexible(self, path):
        try:
            frame = pd.read_csv(path, engine='python', on_bad_lines='skip')
            if {'text', 'is_fake'}.issubset(set(frame.columns)):
                return frame[['text', 'is_fake']]
        except Exception:
            pass

        rows = []
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore', newline='') as data_file:
                reader = csv.DictReader(data_file)
                for row in reader:
                    if row and row.get('text') is not None and row.get('is_fake') is not None:
                        rows.append({'text': row.get('text'), 'is_fake': row.get('is_fake')})
        except Exception:
            return pd.DataFrame(columns=['text', 'is_fake'])

        return pd.DataFrame(rows)

    def add_training_sample(self, text, is_fake):
        try:
            USER_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
            file_exists = USER_DATASET_PATH.exists()

            with open(USER_DATASET_PATH, 'a', encoding='utf-8', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['text', 'is_fake'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({'text': text.strip(), 'is_fake': int(is_fake)})

            return True
        except Exception as error:
            self.last_error = f"Failed to store training sample: {str(error)}"
            return False

    def _synthetic_training_samples(self):
        real_topics = [
            'inflation slowed in the last quarter',
            'the treatment reduced hospital admissions',
            'the climate report updated emissions estimates',
            'economic growth remained stable in q4',
        ]
        fake_topics = [
            'vaccines cause autism in all children',
            '5g towers are causing covid symptoms',
            'the moon landing footage was faked',
            'a single herb cures all diseases instantly',
        ]

        rows = []
        for topic in real_topics:
            rows.append({'text': f'According to an official report, experts verified that {topic}.', 'is_fake': 0})
            rows.append({'text': f'A peer-reviewed study published in Nature found that {topic}.', 'is_fake': 0})

        for topic in fake_topics:
            rows.append({'text': f'SHOCKING: {topic}. Share this now before they delete it!', 'is_fake': 1})
            rows.append({'text': f'Secret leak proves that {topic}. Mainstream media is hiding this!', 'is_fake': 1})

        return rows

    def _fallback_training_samples(self):
        return [
            {'text': 'According to a peer-reviewed study in Nature, researchers found improved outcomes after controlled trials.', 'is_fake': 0},
            {'text': 'Official government data shows unemployment declined for the third consecutive quarter.', 'is_fake': 0},
            {'text': 'Reuters reported that independent agencies confirmed the earthquake magnitude and impact region.', 'is_fake': 0},
            {'text': 'The health ministry said in a statement that phase-3 safety checks were completed successfully.', 'is_fake': 0},
            {'text': 'SHOCKING truth they do not want you to know! Share this immediately before it gets deleted!', 'is_fake': 1},
            {'text': 'Secret cure for all diseases discovered but hidden by government and media.', 'is_fake': 1},
            {'text': 'Anonymous insiders confirm moon landing footage was staged in a studio.', 'is_fake': 1},
            {'text': 'Urgent warning! Mainstream media is lying and this proves the cover-up.', 'is_fake': 1},
        ]

    def predict(self, text):
        if self.word_pipeline is None and self.char_pipeline is None:
            return None

        try:
            probabilities_list = []
            if self.word_pipeline is not None:
                probabilities_list.append(self.word_pipeline.predict_proba([text])[0])
            if self.char_pipeline is not None:
                probabilities_list.append(self.char_pipeline.predict_proba([text])[0])

            if not probabilities_list:
                return None

            probabilities = sum(probabilities_list) / len(probabilities_list)
            fake_probability = float(probabilities[1] * 100)
            real_probability = float(probabilities[0] * 100)

            thresholds = self.metrics.get('decision_thresholds', {})
            real_threshold = float(thresholds.get('real_threshold', 0.45))
            fake_threshold = float(thresholds.get('fake_threshold', 0.55))

            fake_probability_raw = float(probabilities[1])
            if fake_probability_raw >= fake_threshold:
                prediction = 'FAKE'
                margin = (fake_probability_raw - fake_threshold) / max(1.0 - fake_threshold, 1e-6)
                confidence = 60 + (margin * 38)
            elif fake_probability_raw <= real_threshold:
                prediction = 'REAL'
                margin = (real_threshold - fake_probability_raw) / max(real_threshold, 1e-6)
                confidence = 60 + (margin * 38)
            else:
                prediction = 'UNCERTAIN'
                midpoint = (real_threshold + fake_threshold) / 2.0
                half_band = max((fake_threshold - real_threshold) / 2.0, 1e-6)
                distance = abs(fake_probability_raw - midpoint)
                confidence = 50 + max(0.0, (1 - (distance / half_band)) * 12)

            return {
                'prediction': prediction,
                'confidence': round(min(confidence, 99), 2),
                'real_percentage': round(real_probability, 2),
                'fake_percentage': round(fake_probability, 2),
                'thresholds': {
                    'real_threshold': round(real_threshold * 100, 2),
                    'fake_threshold': round(fake_threshold * 100, 2),
                },
                'model': 'ensemble_word_char_logreg',
            }
        except Exception as error:
            self.last_error = f"Prediction failed: {str(error)}"
            return None

    def status(self):
        return {
            'model_loaded': (self.word_pipeline is not None or self.char_pipeline is not None),
            'dataset_path': str(DATASET_PATH),
            'model_path': str(MODEL_PATH),
            'dataset_size': self.dataset_size,
            'metrics': self.metrics,
            'last_error': self.last_error,
        }


def refresh_latest_real_news_dataset(max_items=250):
    """Fetch latest headlines from reliable RSS sources and store as REAL samples."""
    feeds = [
        # Reuters
        'https://feeds.reuters.com/reuters/topNews',
        'https://feeds.reuters.com/reuters/worldNews',
        'https://feeds.reuters.com/reuters/businessNews',
        # BBC
        'https://feeds.bbci.co.uk/news/rss.xml',
        'https://feeds.bbci.co.uk/news/world/rss.xml',
        # CNN
        'http://rss.cnn.com/rss/edition.rss',
        'http://rss.cnn.com/rss/edition_world.rss',
        # NYT
        'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
        'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
        # AP / Guardian / NPR / Al Jazeera
        'https://feeds.apnews.com/apnews/topnews',
        'https://www.theguardian.com/world/rss',
        'https://feeds.npr.org/1001/rss.xml',
        'https://www.aljazeera.com/xml/rss/all.xml',
    ]

    rows = []
    seen_texts = set()
    source_counts = {}
    per_source_cap = max(15, max_items // 6)

    for feed_url in feeds:
        try:
            request = urllib.request.Request(
                feed_url,
                headers={"User-Agent": "Mozilla/5.0 (TruthDetect Latest News Refresh)"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = response.read()

            root = ET.fromstring(payload)
            items = root.findall('.//item')
            for item in items:
                title = (item.findtext('title') or '').strip()
                description = (item.findtext('description') or '').strip()
                pub_date = (item.findtext('pubDate') or '').strip()

                if not title:
                    continue

                combined_text = f"{title}. {description}".strip()
                combined_text = re.sub(r'<[^>]+>', ' ', combined_text)
                combined_text = html.unescape(combined_text)
                combined_text = re.sub(r'\s+', ' ', combined_text).strip()

                if len(combined_text) < 40:
                    continue

                normalized = combined_text.lower()
                if normalized in seen_texts:
                    continue

                source_host = urllib.parse.urlparse(feed_url).netloc.replace('www.', '')
                current_source_count = source_counts.get(source_host, 0)
                if current_source_count >= per_source_cap:
                    continue

                seen_texts.add(normalized)
                source_counts[source_host] = current_source_count + 1
                rows.append({
                    'text': combined_text,
                    'is_fake': 0,
                    'source_file': 'latest_reliable_rss',
                    'source_name': source_host,
                    'published': pub_date,
                    'fetched_at': datetime.now().isoformat(),
                })

                if len(rows) >= max_items:
                    break
        except Exception:
            continue

        if len(rows) >= max_items:
            break

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    if not rows:
        return {
            'success': False,
            'rows_added': 0,
            'file': str(LATEST_REAL_DATASET_PATH),
            'message': 'Could not fetch latest reliable RSS data',
        }

    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=['text'])
    frame.to_csv(LATEST_REAL_DATASET_PATH, index=False)

    return {
        'success': True,
        'rows_added': int(len(frame)),
        'file': str(LATEST_REAL_DATASET_PATH),
        'sources_used': source_counts,
        'message': 'Latest reliable news dataset refreshed',
    }


ml_model = TextAuthenticityModel()
ml_model.load_or_train()


# ─────────────────────────────────────────────────────
# AUTO-RETRAINING SCHEDULER
# Fetches latest verified headlines every 6 h and retrains the ML model
# automatically — no manual work required.
# ─────────────────────────────────────────────────────
_scheduler_status = {
    'enabled': False,
    'last_run': None,
    'last_result': None,
    'interval_hours': 6,
    'next_run': None,
}
_bg_scheduler = None


def _auto_retrain_job():
    """Fetch latest real-news RSS data and retrain the model. Runs automatically."""
    global _scheduler_status
    try:
        refresh_report = refresh_latest_real_news_dataset(max_items=200)
        retrained = False
        if refresh_report.get('success'):
            retrained = ml_model.train_model(force=True)
        _scheduler_status['last_run'] = datetime.now().isoformat()
        _scheduler_status['last_result'] = {
            'success': refresh_report.get('success', False),
            'rows_added': refresh_report.get('rows_added', 0),
            'retrained': retrained,
            'sources': list(refresh_report.get('sources_used', {}).keys()),
        }
    except Exception as err:
        _scheduler_status['last_run'] = datetime.now().isoformat()
        _scheduler_status['last_result'] = {'success': False, 'error': str(err)}

    # Update next_run estimate
    if _bg_scheduler is not None:
        try:
            job = _bg_scheduler.get_job('auto_retrain')
            if job and job.next_run_time:
                _scheduler_status['next_run'] = job.next_run_time.isoformat()
        except Exception:
            pass


if SCHEDULER_AVAILABLE:
    try:
        _bg_scheduler = BackgroundScheduler(daemon=True)
        # First run: 2 minutes after startup so it doesn't block boot
        _bg_scheduler.add_job(
            _auto_retrain_job,
            'interval',
            hours=6,
            id='auto_retrain',
            next_run_time=datetime.now() + timedelta(minutes=2),
        )
        _bg_scheduler.start()
        _scheduler_status['enabled'] = True
        _scheduler_status['next_run'] = (datetime.now() + timedelta(minutes=2)).isoformat()
    except Exception:
        _scheduler_status['enabled'] = False


def extract_search_query(text, max_terms=10):
    """Create a compact search query from input news text."""
    words = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text.lower())
    stopwords = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'were', 'have', 'has',
        'into', 'about', 'after', 'before', 'will', 'would', 'could', 'should', 'they',
        'them', 'their', 'there', 'been', 'said', 'according', 'report', 'news', 'also',
        'more', 'than', 'what', 'when', 'where', 'which', 'while', 'because', 'over',
        'under', 'into', 'just', 'your', 'you', 'are', 'not', 'but', 'our', 'out', 'all'
    }

    filtered = []
    seen = set()
    for word in words:
        if word in stopwords or len(word) < 4:
            continue
        if word in seen:
            continue
        seen.add(word)
        filtered.append(word)
        if len(filtered) >= max_terms:
            break

    if not filtered:
        filtered = words[:max_terms]

    return " ".join(filtered)


def lookup_circulation_sources(text, max_results=5):
    """Lookup related links where similar news is circulating using Google News RSS."""
    if not text or len(text.strip()) < 20:
        return []

    try:
        query = extract_search_query(text)
        if not query:
            return []

        rss_url = (
            "https://news.google.com/rss/search?q="
            f"{urllib.parse.quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        )

        request = urllib.request.Request(
            rss_url,
            headers={"User-Agent": "Mozilla/5.0 (TruthDetect Source Finder)"},
        )

        ssl_context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=8, context=ssl_context) as response:
            payload = response.read()

        root = ET.fromstring(payload)
        items = root.findall('.//item')

        sources = []
        seen_links = set()
        for item in items:
            title = (item.findtext('title') or '').strip()
            link = (item.findtext('link') or '').strip()
            pub_date = (item.findtext('pubDate') or '').strip()
            source_node = item.find('source')
            source_name = source_node.text.strip() if source_node is not None and source_node.text else 'Unknown Source'

            if not link or link in seen_links:
                continue

            seen_links.add(link)
            sources.append({
                'title': title[:200] if title else 'Related news item',
                'link': link,
                'source': source_name,
                'published': pub_date,
            })

            if len(sources) >= max_results:
                break

        return sources
    except Exception:
        return []

# Load trusted sources
def load_trusted_sources():
    try:
        with open('data/sources.json', 'r') as f:
            return json.load(f)
    except:
        return {
            "trusted_sources": {
                "reuters.com": {"score": 95, "type": "news_agency"},
                "apnews.com": {"score": 94, "type": "news_agency"},
                "bbc.com": {"score": 92, "type": "public_broadcaster"},
                "nytimes.com": {"score": 90, "type": "newspaper"},
                "cnn.com": {"score": 85, "type": "news_network"},
                "aljazeera.com": {"score": 87, "type": "news_network"}
            }
        }

# Load categories
def load_categories():
    return {
        "clickbait": "Sensational headlines designed to attract clicks",
        "disinformation": "False information spread deliberately to deceive",
        "hoaxes": "Deliberate fabrications presented as truth",
        "junk_news": "Poor quality, misleading, or fabricated content",
        "misinformation": "False information spread without malicious intent",
        "propaganda": "Biased information promoting a political agenda",
        "satire": "Humorous exaggeration not meant to be taken literally"
    }

# Text analysis function
def analyze_text(text):
    """Analyze news text for authenticity"""
    if not text or len(text.strip()) < 10:
        return {"error": "Text too short"}
    
    text_lower = text.lower()
    word_count = len(text.split())
    short_text_mode = word_count < 80
    
    # Fake news indicators
    fake_indicators = [
        'breaking', 'shocking', 'secret', 'cover-up', 'exposed',
        'they don\'t want you to know', 'share this', 'viral',
        'urgent', 'alert', 'warning'
    ]
    
    # Real news indicators
    real_indicators = [
        'according to', 'study shows', 'research indicates',
        'official report', 'said in a statement', 'according to data',
        'peer-reviewed', 'journal article', 'university study'
    ]
    
    # Count indicators
    fake_count = sum(1 for word in fake_indicators if word in text_lower)
    real_count = sum(1 for word in real_indicators if word in text_lower)

    sensational_terms = [
        'you won\'t believe', 'miracle', 'leaked', 'exclusive', 'hidden truth',
        'mainstream media', 'wake up', 'must share', 'exposed', 'secret cure'
    ]
    misinformation_phrases = [
        'vaccine causes autism',
        '5g technology causes covid',
        'moon landing was faked',
        'cure all diseases',
        'secret cure',
        'government cover-up',
        'mainstream media is hiding',
        'before they delete',
        'share this now',
        'anonymous insiders confirm',
    ]
    misinformation_patterns = [
        r'\bvaccines?\b.{0,50}\bautism\b',
        r'\b5g\b.{0,50}\bcovid\b',
        r'\bmoon\s+landing\b.{0,50}\bfak(?:e|ed)\b',
        r'\bcure\b.{0,50}\ball\s+diseases\b',
        r'\bmiracle\b.{0,50}\bcure\b',
        r'\bmainstream\s+media\b.{0,50}\bhiding\b',
        r'\bbefore\s+they\s+delete\b',
        r'\bshare\s+this\s+now\b',
    ]
    
    # Detection for unsubstantiated death/extreme claims (common fake news pattern)
    # These are claims that require citations but often appear without context
    unsubstantiated_claim_patterns = [
        r'\b(?:trump|biden|obama|clinton|president|celebrity|famous)\b.{0,40}\b(?:dead|died|killed|assassinated|murdered)\b',
        r'\b(?:dead|died|killed|assassinated)\b.{0,40}\b(?:trump|biden|obama|clinton|president|celebrity)\b',
        r'\b(?:world\s+leader|prime\s+minister|king|queen)\b.{0,40}\b(?:dead|died|killed|assassinated)\b',
        r'\b(?:war|nuclear|attack)\b.{0,30}\b(?:declared|started|launched)\b',
        r'\b(?:martial\s+law|state\s+of\s+emergency)\b.{0,40}\b(?:declared|announced)\b',
    ]
    citation_terms = [
        'according to', 'reported by', 'official report', 'study', 'research',
        'data', 'journal', 'published', 'confirmed', 'statement'
    ]
    trusted_source_terms = [
        'reuters',
        'associated press',
        'ap news',
        'bbc',
        'the guardian',
        'new york times',
        'official government report',
        'ministry',
        'federal reserve',
        'who',
        'cdc',
    ]
    institutional_terms = [
        'government', 'ministry', 'department', 'agency', 'council', 'parliament',
        'senate', 'committee', 'budget', 'policy', 'official', 'report', 'statement',
        'approved', 'review', 'auditor', 'court', 'mayor'
    ]

    sensational_count = sum(1 for term in sensational_terms if term in text_lower)
    misinformation_count = sum(1 for term in misinformation_phrases if term in text_lower)
    misinformation_count += sum(1 for pattern in misinformation_patterns if re.search(pattern, text_lower, re.IGNORECASE))
    
    # Check for unsubstantiated death/extreme claims (strong fake news indicator when no citation)
    unsubstantiated_claim_count = sum(1 for pattern in unsubstantiated_claim_patterns if re.search(pattern, text_lower, re.IGNORECASE))
    
    citation_count = sum(1 for term in citation_terms if term in text_lower)
    trusted_source_count = sum(1 for term in trusted_source_terms if term in text_lower)
    institutional_count = sum(1 for term in institutional_terms if term in text_lower)
    exclamation_count = text.count('!')
    question_count = text.count('?')
    uppercase_word_count = len(re.findall(r'\b[A-Z]{3,}\b', text))
    numeric_token_count = len(re.findall(r'\b\d+(?:\.\d+)?\b', text))
    quote_count = text.count('"')
    
    # Rule-based linguistic scoring
    fake_signal = (
        fake_count * 14
        + sensational_count * 10
        + misinformation_count * 22
        + exclamation_count * 2
        + uppercase_word_count * 3
        + (8 if question_count >= 3 else 0)
        + (4 if word_count < 20 else 0)
    )
    
    # Heavily penalize unsubstantiated death/extreme claims without citations
    # These are extremely likely to be fake news when stated without source
    if unsubstantiated_claim_count > 0:
        if citation_count == 0 and trusted_source_count == 0:
            # No source cited for extreme claim = very likely fake
            fake_signal += unsubstantiated_claim_count * 35
            # Extra penalty for very short claims (< 20 words) with no context
            if word_count < 20:
                fake_signal += 20
        elif citation_count > 0 or trusted_source_count > 0:
            # Has citation but still flagged - moderate suspicion
            fake_signal += unsubstantiated_claim_count * 8

    if sensational_count >= 2 and exclamation_count >= 2:
        fake_signal += 10

    real_signal = (
        real_count * 14
        + citation_count * 8
        + trusted_source_count * 12
        + min(numeric_token_count, 12) * 1.8
        + min(quote_count, 8) * 1.2
        + (10 if 80 <= word_count <= 1400 else 0)
    )

    if 20 <= word_count < 80 and sensational_count == 0 and misinformation_count == 0:
        real_signal += 6

    # Avoid over-penalizing neutral/plain language
    if fake_count == 0 and sensational_count == 0 and exclamation_count == 0 and uppercase_word_count == 0:
        fake_signal *= 0.7
        real_signal += 8

    # Strong formal-source cue handling
    if trusted_source_count >= 1 and misinformation_count == 0 and sensational_count == 0:
        real_signal += 18
        fake_signal *= 0.65

    # Neutral fallback when very few signals are present
    if fake_signal < 8 and real_signal < 8:
        heuristic_fake = 45.0
        heuristic_real = 55.0
    else:
        total_signal = max(fake_signal + real_signal, 1.0)
        heuristic_fake = (fake_signal / total_signal) * 100
        heuristic_real = (real_signal / total_signal) * 100

    trusted_report_context = (
        trusted_source_count >= 1
        and citation_count >= 1
        and misinformation_count == 0
        and sensational_count == 0
        and exclamation_count == 0
    )
    neutral_institutional_context = (
        institutional_count >= 2
        and misinformation_count == 0
        and sensational_count == 0
        and fake_count == 0
        and exclamation_count == 0
        and uppercase_word_count <= 2
    )

    # Blend with trained ML model (primary signal) and apply only light heuristic bias
    model_prediction = ml_model.predict(text)
    if model_prediction:
        model_fake = float(model_prediction['fake_percentage'])
        total_signal = max(fake_signal + real_signal, 1.0)
        heuristic_bias = ((fake_signal - real_signal) / total_signal) * 12.0
        heuristic_bias = max(min(heuristic_bias, 10.0), -10.0)
        fake_percentage = model_fake + heuristic_bias
    else:
        fake_percentage = heuristic_fake

    # Strong explicit misinformation should push FAKE upward
    if misinformation_count >= 2 and fake_percentage < 62:
        fake_percentage = 62
    elif misinformation_count >= 1 and sensational_count >= 2 and fake_percentage < 58:
        fake_percentage = 58

    # Trusted formal reporting should reduce false FAKEs
    if trusted_report_context and fake_percentage > 40:
        fake_percentage -= min(14, fake_percentage - 40)

    # Neutral institutional reporting should not be pushed hard to FAKE
    if neutral_institutional_context and fake_percentage > 48:
        fake_percentage -= min(10, fake_percentage - 48)

    fake_percentage = min(max(fake_percentage, 1), 99)
    real_percentage = 100 - fake_percentage
    score_margin = abs(real_percentage - fake_percentage)

    # Determine final prediction using calibrated model thresholds when available
    if model_prediction and 'thresholds' in model_prediction:
        threshold_info = model_prediction.get('thresholds', {})
        real_threshold = float(threshold_info.get('real_threshold', 45.0))
        fake_threshold = float(threshold_info.get('fake_threshold', 55.0))

        # Short snippets/headlines carry less context; be conservative before calling FAKE.
        if short_text_mode and misinformation_count == 0 and sensational_count == 0:
            real_threshold = min(real_threshold + 4.0, 52.0)
            fake_threshold = min(fake_threshold + 8.0, 78.0)

        if fake_percentage >= fake_threshold:
            prediction = "FAKE"
        elif fake_percentage <= real_threshold:
            prediction = "REAL"
        else:
            prediction = "UNCERTAIN"

        confidence = float(model_prediction.get('confidence', 55.0))
        confidence = (confidence * 0.8) + (min(95.0, 50.0 + (score_margin * 0.6)) * 0.2)
    else:
        if score_margin < 8:
            prediction = "UNCERTAIN"
            confidence = max(45, 55 - (score_margin / 2))
        elif real_percentage > fake_percentage:
            prediction = "REAL"
            confidence = min(98, 50 + (score_margin * 0.9))
        else:
            prediction = "FAKE"
            confidence = min(98, 50 + (score_margin * 0.9))

    # Final safety checks
    if prediction == "REAL" and misinformation_count >= 2:
        prediction = "FAKE"
        confidence = max(confidence, 72)
    
    # Override for unsubstantiated death/extreme claims without citation
    # These are almost always fake when stated baldly without source
    if (
        unsubstantiated_claim_count > 0
        and citation_count == 0
        and trusted_source_count == 0
        and word_count < 30  # Very short claim with no context
    ):
        if prediction != "FAKE":
            prediction = "FAKE"
            confidence = max(confidence, 78)
        else:
            confidence = max(confidence, 85)

    if (
        prediction == "UNCERTAIN"
        and trusted_report_context
        and misinformation_count == 0
        and sensational_count == 0
        and fake_count == 0
        and fake_percentage <= 55
    ):
        prediction = "REAL"
        confidence = max(confidence, 68)

    if prediction == "FAKE" and trusted_report_context and misinformation_count == 0 and fake_percentage < 65:
        prediction = "UNCERTAIN"
        confidence = min(confidence, 62)

    if prediction == "FAKE" and neutral_institutional_context and fake_percentage < 72:
        prediction = "UNCERTAIN"
        confidence = min(confidence, 64)

    if (
        prediction == "FAKE"
        and misinformation_count == 0
        and sensational_count == 0
        and fake_count == 0
        and exclamation_count == 0
        and fake_percentage < 68
    ):
        prediction = "UNCERTAIN"
        confidence = min(confidence, 62)

    if prediction == "UNCERTAIN" and neutral_institutional_context and fake_percentage <= 52:
        prediction = "REAL"
        confidence = max(confidence, 66)

    if prediction == "UNCERTAIN" and neutral_institutional_context and institutional_count >= 3 and fake_percentage <= 62:
        prediction = "REAL"
        confidence = max(confidence, 63)

    # Reduce unnecessary UNCERTAIN outcomes when one side has clearer evidence.
    if prediction == "UNCERTAIN":
        has_fake_cues = misinformation_count >= 1 or sensational_count >= 1 or fake_count >= 1
        has_real_cues = citation_count >= 1 or trusted_source_count >= 1 or institutional_count >= 2 or real_count >= 1

        if has_real_cues and not has_fake_cues and fake_percentage <= 58:
            prediction = "REAL"
            confidence = max(confidence, 62)
        elif has_fake_cues and fake_percentage >= 56:
            prediction = "FAKE"
            confidence = max(confidence, 62)
        elif score_margin >= 12 and (has_real_cues or has_fake_cues or word_count >= 90):
            prediction = "REAL" if real_percentage > fake_percentage else "FAKE"
            confidence = max(confidence, 60)
    
    # Determine categories
    categories = []
    if 'breaking' in text_lower and '!' in text:
        categories.append({"name": "clickbait", "confidence": 80, "description": "Uses sensational headlines"})
    if any(word in text_lower for word in ['government', 'secret', 'plot', 'conspiracy']):
        categories.append({"name": "propaganda", "confidence": 70, "description": "Contains political messaging"})
    if 'joke' in text_lower or 'satire' in text_lower or 'humor' in text_lower:
        categories.append({"name": "satire", "confidence": 85, "description": "Appears to be humorous content"})
    if 'fake' in text_lower or 'false' in text_lower:
        categories.append({"name": "misinformation", "confidence": 75, "description": "Contains false claims"})
    
    # Default category if none detected
    if not categories:
        categories.append({"name": "general", "confidence": 50, "description": "General news content"})
    
    # Extract potential dates
    date_patterns = [
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # DD-MM-YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
        r'\b\d{4}\b'  # Just year
    ]
    
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            dates_found.append({
                "date": match,
                "context": "Found in text",
                "consistent": True
            })
    
    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "real_percentage": round(real_percentage, 2),
        "fake_percentage": round(fake_percentage, 2),
        "categories": categories,
        "text_preview": text[:300] + ("..." if len(text) > 300 else ""),
        "word_count": len(text.split()),
        "dates_found": dates_found[:3],  # Limit to 3 dates
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_insights": {
            "ml_enabled": bool(model_prediction),
            "ml_prediction": model_prediction['prediction'] if model_prediction else "N/A",
            "ml_confidence": model_prediction['confidence'] if model_prediction else 0,
            "model_name": model_prediction['model'] if model_prediction else "rule_based",
            "score_margin": round(score_margin, 2),
            "rule_based_fake": round(heuristic_fake, 2),
            "rule_based_real": round(heuristic_real, 2),
            "misinformation_hits": misinformation_count,
            "sensational_hits": sensational_count,
                "institutional_hits": institutional_count,
                "short_text_mode": short_text_mode,
                "recommended_min_words": 40,
        }
    }

# Source analysis function
def analyze_source(url):
    """Analyze the credibility of a news source"""
    if not url:
        return {
            "trust_score": 50,
            "status": "No URL provided",
            "source_type": "Unknown",
            "analysis": "Cannot analyze without URL"
        }
    
    import urllib.parse
    
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        
        trusted_sources = load_trusted_sources()["trusted_sources"]
        
        # Check if domain is in trusted sources
        if domain in trusted_sources:
            trust_score = trusted_sources[domain]["score"]
            source_type = trusted_sources[domain]["type"]
            analysis = "Trusted news source"
        else:
            # Check for suspicious patterns
            suspicious = ['.xyz', '.top', '.club', 'fake', 'clickbait', 'rumor']
            if any(pattern in domain for pattern in suspicious):
                trust_score = 25
                source_type = "Questionable"
                analysis = "Suspicious domain pattern detected"
            elif 'blog' in domain or 'wordpress' in domain:
                trust_score = 40
                source_type = "Personal Blog"
                analysis = "Personal blog, verify with official sources"
            else:
                trust_score = 50
                source_type = "Unknown"
                analysis = "Domain not in our trusted database"
        
        # Check for HTTPS
        ssl_secured = url.startswith('https')
        if ssl_secured:
            trust_score += 5
        
        return {
            "domain": domain,
            "trust_score": min(trust_score, 100),
            "source_type": source_type,
            "ssl_secured": ssl_secured,
            "analysis": analysis
        }
    except Exception as e:
        return {
            "trust_score": 30,
            "status": f"Error analyzing URL: {str(e)}",
            "source_type": "Error",
            "analysis": "Could not parse URL"
        }

# Routes
@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/detect')
def detect():
    """Detection page"""
    sample = request.args.get('sample', '')
    return render_template('detect.html', sample_text=sample)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/faq')
def faq():
    """FAQ page"""
    real_accuracy = ml_model.metrics.get('accuracy')
    acc_str = f'{round(real_accuracy, 1)}%' if real_accuracy else 'dependent on dataset size'
    faqs = [
        {"q": "How accurate is your system?",
         "a": f"Based on the current training data the model achieves approximately {acc_str} accuracy. "
              f"Accuracy improves automatically every 6 hours as the scheduler fetches the latest "
              f"verified headlines from Reuters, BBC, AP, Guardian and others and retrains the model."},
        {"q": "Does the model improve automatically?",
         "a": "Yes. An auto-retrain scheduler runs every 6 hours. It pulls fresh real-world headlines "
              f"from 6 trusted RSS feeds, adds them as REAL samples, and retrains the model. "
              f"You can also trigger a manual refresh from the Live Feed dashboard."},
        {"q": "What types of fake news can you detect?",
         "a": "We detect 7 categories: clickbait, disinformation, hoaxes, junk news, misinformation, propaganda, and satire."},
        {"q": "What is the UNCERTAIN verdict?",
         "a": "UNCERTAIN means the AI could not confidently classify the content. This happens with short snippets, "
              "neutral language, or topics that lack strong signals. Always cross-check UNCERTAIN results with trusted sources."},
        {"q": "How much text should I paste?",
         "a": "For stable results use at least 40 words. Very short headlines or phrases often return UNCERTAIN by design."},
        {"q": "Can I analyze URLs?",
         "a": "Yes. The URL tab auto-extracts the full article body using a web scraper before running the analysis. "
              "You can also manually review the extracted text before submitting."},
        {"q": "Is it free to use?", "a": "Yes, the detection service is completely free."},
        {"q": "How long does analysis take?", "a": "Most analyses complete in 2–3 seconds."},
        {"q": "Is this always correct?",
         "a": "No. It is a decision-support tool. Always verify high-impact claims with at least two independent trusted sources."},
    ]
    return render_template('faq.html', faqs=faqs)

@app.route('/research')
def research():
    """Research page"""
    return render_template('research.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')


@app.route('/api/site-stats', methods=['GET'])
def api_site_stats():
    """Return comprehensive site statistics for dashboard and home page."""
    model_status = ml_model.status()
    feedback     = _load_feedback()
    site_stats   = _load_stats()
    with _dashboard_lock:
        dash_count = len(_dashboard_cache.get('headlines', []))
        fetched_at = _dashboard_cache.get('fetched_at')
    return jsonify({
        'success': True,
        'model': {
            'loaded':            model_status.get('model_loaded'),
            'dataset_size':      model_status.get('dataset_size', 0),
            'accuracy':          model_status.get('metrics', {}).get('accuracy', 0),
            'f1_score':          model_status.get('metrics', {}).get('f1_score', 0),
            'roc_auc':           model_status.get('metrics', {}).get('roc_auc'),
            'trained_at':        model_status.get('metrics', {}).get('trained_at'),
            'class_distribution':model_status.get('metrics', {}).get('class_distribution', {}),
            'class_balance_ratio':model_status.get('metrics', {}).get('class_balance_ratio', 0),
            'decision_thresholds':model_status.get('metrics', {}).get('decision_thresholds', {}),
            'train_samples':     model_status.get('metrics', {}).get('train_samples', 0),
            'test_samples':      model_status.get('metrics', {}).get('test_samples', 0),
            'holdout':           model_status.get('metrics', {}).get('holdout_calibration', {}),
        },
        'votes':     feedback.get('summary', {'agree': 0, 'disagree': 0, 'total': 0}),
        'analyses':  site_stats,
        'dashboard': {
            'headlines_cached': dash_count,
            'last_fetched': fetched_at.isoformat() if fetched_at else None,
        },
        'scheduler': _scheduler_status,
    })

@app.route('/analyze', methods=['POST'])
@_rl('30 per hour')
def analyze():
    """Analyze news endpoint"""
    try:
        data = request.get_json() or {}
        text = data.get('text', '').strip()
        url = data.get('url', '').strip()
        channel = str(data.get('channel', 'text')).lower().strip()
        is_whatsapp = bool(data.get('is_whatsapp')) or channel in {'wa', 'whatsapp', 'forward'}

        # Server-side length cap to prevent memory abuse
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        if not text and not url:
            return jsonify({
                'error': True,
                'message': 'Please provide either text or URL to analyze',
            }), 400

        # Analyze text
        text_analysis = analyze_text(text) if text else {
            "prediction": "UNKNOWN",
            "confidence": 50,
            "real_percentage": 50,
            "fake_percentage": 50,
            "categories": [],
            "text_preview": "No text provided",
            "word_count": 0
        }
        
        # Analyze source if URL provided
        source_analysis = analyze_source(url) if url else {
            "trust_score": 50,
            "source_type": "Text input",
            "analysis": "Analyzed from text input only"
        }

        # Find where similar news is circulating
        circulation_sources = lookup_circulation_sources(text) if text else []
        if url:
            circulation_sources.insert(0, {
                "title": "Original URL provided for analysis",
                "link": url,
                "source": source_analysis.get('domain', 'Provided Source'),
                "published": "Provided by user"
            })
        
        # Calculate overall confidence
        text_confidence = text_analysis.get('confidence', 50)
        source_confidence = source_analysis.get('trust_score', 50)
        
        if url:  # If URL provided, give more weight to source
            overall_confidence = (text_confidence * 0.4) + (source_confidence * 0.6)
        else:  # If only text, use text confidence
            overall_confidence = text_confidence
        
        # Generate evidence
        evidence = []
        if text_analysis['prediction'] == "REAL":
            evidence.append({
                "type": "positive",
                "description": "Contains factual reporting language",
                "supporting": True
            })
        elif text_analysis['prediction'] == "FAKE":
            evidence.append({
                "type": "negative",
                "description": "Uses sensationalist language",
                "supporting": False
            })
        
        if source_analysis.get('trust_score', 50) > 70:
            evidence.append({
                "type": "positive",
                "description": f"Source ({source_analysis.get('domain', 'N/A')}) is credible",
                "supporting": True
            })
        
        # Suggested actions
        suggested_actions = [
            {
                "title": "Verify with multiple sources",
                "description": "Check if other reputable news outlets report the same story",
                "icon": "fa-search",
                "link": "https://news.google.com"
            },
            {
                "title": "Check publication date",
                "description": "Old news can be misleading if presented as current",
                "icon": "fa-calendar"
            },
            {
                "title": "Review author credibility",
                "description": "Check the author's background and expertise",
                "icon": "fa-user-check"
            }
        ]
        
        # Compile final result
        result = {
            "success": True,
            "text_analysis": text_analysis,
            "source_analysis": source_analysis,
            "circulation_sources": circulation_sources,
            "input_channel": channel,
            "overall": {
                "prediction": text_analysis['prediction'],
                "confidence": round(overall_confidence, 2),
                "real_percentage": text_analysis.get('real_percentage', 50),
                "fake_percentage": text_analysis.get('fake_percentage', 50)
            },
            "dates_found": text_analysis.get('dates_found', []),
            "confidence_breakdown": {
                "text_analysis": text_analysis.get('confidence', 50),
                "source_credibility": source_analysis.get('trust_score', 50),
                "linguistic_patterns": text_analysis.get('confidence', 50),
                "temporal_consistency": 60 if text_analysis.get('dates_found') else 50
            },
            "evidence": evidence,
            "suggested_actions": suggested_actions,
            "analysis_timestamp": datetime.now().isoformat(),
            "reading_time": f"{max(1, len(text.split()) // 200)} min"
        }

        if is_whatsapp:
            result['whatsapp_analysis'] = analyze_whatsapp_forward(text)
        
        # Store in session for result page
        session['last_analysis'] = result

        # Track analysis count
        try:
            _increment_analysis()
        except Exception:
            pass

        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": True,
            "message": f"Analysis error: {str(e)}"
        }), 500

@app.route('/result')
def result():
    """Result display page"""
    result_data = session.get('last_analysis', {})
    if not result_data:
        # Session expired or direct URL visit — tell the user why
        return redirect(url_for('detect') + '?no_result=1')
    return render_template('result.html', result=result_data)

@app.route('/api/check', methods=['POST'])
def api_check():
    """Simple API endpoint"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    analysis = analyze_text(text)
    return jsonify(analysis)


@app.route('/api/check-url', methods=['POST'])
def api_check_url():
    """API endpoint for URL/domain credibility checks"""
    payload = request.get_json() or {}
    url = str(payload.get('url', '')).strip()

    if not url:
        return jsonify({"error": "No url provided"}), 400

    source_analysis = analyze_source(url)
    return jsonify(source_analysis)


@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Return ML model status and metrics"""
    return jsonify(ml_model.status())


@app.route('/api/model-retrain', methods=['POST'])
@_rl('3 per hour')
def model_retrain():
    """Force retrain ML model from dataset"""
    admin_error = _require_admin()
    if admin_error:
        return admin_error
    retrain_success = ml_model.train_model(force=True)
    if not retrain_success:
        return jsonify({
            "success": False,
            "message": ml_model.last_error or "Retraining failed",
            "status": ml_model.status(),
        }), 500

    return jsonify({
        "success": True,
        "message": "Model retrained successfully",
        "status": ml_model.status(),
    })


@app.route('/api/model-add-sample', methods=['POST'])
@_rl('10 per hour')
def model_add_sample():
    """Add labeled sample for continuous learning"""
    admin_error = _require_admin()
    if admin_error:
        return admin_error
    payload = request.get_json() or {}
    text = str(payload.get('text', '')).strip()
    label = payload.get('is_fake', None)
    auto_retrain = bool(payload.get('auto_retrain', False))

    if len(text) < 10:
        return jsonify({
            'success': False,
            'message': 'Text must be at least 10 characters',
        }), 400

    if label not in [0, 1, '0', '1', False, True]:
        return jsonify({
            'success': False,
            'message': 'is_fake must be 0 or 1',
        }), 400

    is_fake = int(label)
    added = ml_model.add_training_sample(text, is_fake)
    if not added:
        return jsonify({
            'success': False,
            'message': ml_model.last_error or 'Could not save sample',
        }), 500

    retrained = False
    if auto_retrain:
        retrained = ml_model.train_model(force=True)

    return jsonify({
        'success': True,
        'message': 'Training sample added successfully',
        'retrained': retrained,
        'status': ml_model.status(),
    })


@app.route('/api/model-ingest-and-retrain', methods=['POST'])
@_rl('2 per hour')
def model_ingest_and_retrain():
    """Ingest external CSV datasets and optionally retrain model."""
    admin_error = _require_admin()
    if admin_error:
        return admin_error
    payload = request.get_json() or {}
    auto_retrain = bool(payload.get('auto_retrain', True))
    force_retrain = bool(payload.get('force_retrain', False))

    ingest_report = ingest_external_datasets(
        raw_dir=str(RAW_DATASET_DIR),
        output_csv=str(TRAINING_DIR / 'external_ingested.csv'),
    )

    if ingest_report.get('rows_ingested', 0) == 0:
        return jsonify({
            'success': False,
            'message': 'No valid rows ingested. Add Kaggle/UCI CSV files into data/raw_datasets and try again.',
            'ingest_report': ingest_report,
            'status': ml_model.status(),
        }), 400

    distribution = ingest_report.get('label_distribution', {})
    ingested_real = int(distribution.get('real', 0))
    ingested_fake = int(distribution.get('fake', 0))

    if ingested_real == 0 or ingested_fake == 0:
        return jsonify({
            'success': False,
            'message': 'Ingested data is one-sided (only REAL or only FAKE). Add both classes before retraining.',
            'ingest_report': ingest_report,
            'status': ml_model.status(),
        }), 400

    minority = min(ingested_real, ingested_fake)
    majority = max(ingested_real, ingested_fake)
    imbalance_ratio = (minority / majority) if majority else 0

    if imbalance_ratio < 0.15 and auto_retrain and not force_retrain:
        return jsonify({
            'success': False,
            'message': 'Ingested data is highly imbalanced. Retrain blocked to protect model quality. Use force_retrain=true to override.',
            'ingest_report': ingest_report,
            'status': ml_model.status(),
        }), 400

    retrained = False
    if auto_retrain:
        retrained = ml_model.train_model(force=True)

    return jsonify({
        'success': True,
        'message': 'External datasets ingested successfully',
        'retrained': retrained,
        'imbalance_ratio': round(imbalance_ratio, 3),
        'ingest_report': ingest_report,
        'status': ml_model.status(),
    })


@app.route('/api/model-refresh-latest-retrain', methods=['POST'])
@_rl('3 per hour')
def model_refresh_latest_retrain():
    """Refresh latest reliable RSS samples and retrain model."""
    admin_error = _require_admin()
    if admin_error:
        return admin_error
    payload = request.get_json() or {}
    auto_retrain = bool(payload.get('auto_retrain', True))
    max_items = int(payload.get('max_items', 250))
    max_items = max(50, min(max_items, 600))

    refresh_report = refresh_latest_real_news_dataset(max_items=max_items)
    if not refresh_report.get('success'):
        return jsonify({
            'success': False,
            'message': refresh_report.get('message', 'Latest data refresh failed'),
            'refresh_report': refresh_report,
            'status': ml_model.status(),
        }), 400

    retrained = False
    if auto_retrain:
        retrained = ml_model.train_model(force=True)

    return jsonify({
        'success': True,
        'message': 'Latest reliable data refreshed successfully',
        'retrained': retrained,
        'refresh_report': refresh_report,
        'status': ml_model.status(),
    })

# ─────────────────────────────────────────────
# FEATURE 1 — Full URL Article Scraper
# ─────────────────────────────────────────────
def scrape_article_from_url(url):
    """Scrape the full article body text from a URL using BeautifulSoup."""
    if not SCRAPER_AVAILABLE:
        return None, 'Scraper library not available'
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/124.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        resp = http_requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'lxml')

        # Remove boilerplate tags
        for tag in soup(['script', 'style', 'nav', 'header', 'footer',
                         'aside', 'form', 'noscript', 'iframe', 'advertisement']):
            tag.decompose()

        # Try known article containers first
        article_text = ''
        for selector in ['article', '[role="main"]', 'main', '.article-body',
                         '.post-content', '.entry-content', '.story-body',
                         '#article-body', '.article__body', '.content-body']:
            container = soup.select_one(selector)
            if container:
                paragraphs = container.find_all('p')
                article_text = ' '.join(p.get_text(' ', strip=True) for p in paragraphs)
                if len(article_text.split()) >= 30:
                    break

        # Fallback: all <p> tags
        if len(article_text.split()) < 30:
            paragraphs = soup.find_all('p')
            article_text = ' '.join(p.get_text(' ', strip=True) for p in paragraphs)

        article_text = re.sub(r'\s+', ' ', article_text).strip()
        article_text = article_text[:8000]  # cap at 8k chars

        title = ''
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            title = title_tag.get_text(' ', strip=True)[:200]

        if len(article_text.split()) < 15:
            return None, 'Could not extract enough article text from that URL'

        return {'text': article_text, 'title': title, 'url': url}, None
    except Exception as err:
        return None, f'Failed to fetch URL: {str(err)}'


@app.route('/api/scrape-url', methods=['POST'])
def api_scrape_url():
    """Scrape full article text from a URL and return it."""
    payload = request.get_json() or {}
    url = str(payload.get('url', '')).strip()
    if not url:
        return jsonify({'success': False, 'message': 'No URL provided'}), 400
    data, err = scrape_article_from_url(url)
    if err:
        return jsonify({'success': False, 'message': err}), 422
    return jsonify({'success': True, **data})


# ─────────────────────────────────────────────
# FEATURE 2 — WhatsApp Forward Detector
# ─────────────────────────────────────────────
def analyze_whatsapp_forward(text):
    """Detect forwarded-message patterns common in WhatsApp/Telegram fake news."""
    text_lower = text.lower()

    forward_signals = [
        r'forward(?:ed)?\s+(?:this|it|to|message)',
        r'share\s+(?:this|now|immediately|before)',
        r'please\s+(?:forward|share|spread)',
        r'send\s+(?:this|to\s+everyone|to\s+all)',
        r'copy\s+(?:and\s+paste|paste\s+and\s+share)',
        r'\d+\s+times\s+forwarded',
        r'circulating\s+(?:on\s+)?(?:whatsapp|telegram|social)',
        r'viral\s+(?:message|news|post)',
        r'going\s+viral',
        r'broadcast\s+(?:list|message)',
    ]
    chain_signals = [
        r'send\s+to\s+\d+\s+(?:people|friends)',
        r'forward\s+to\s+\d+',
        r'if\s+you\s+don\'t\s+(?:share|forward)',
        r'within\s+\d+\s+(?:hours|minutes|days)',
        r'bad\s+luck',
        r'good\s+luck\s+(?:if|to)',
        r'must\s+(?:share|forward)',
    ]
    panic_signals = [
        r'\b(?:urgent|emergency|warning|alert|breaking)\b',
        r'\b(?:immediately|right\s+now|asap)\b',
        r'\b(?:danger|threat|attack|crisis)\b.{0,30}\b(?:confirmed|verified|proven)\b',
        r'tell\s+(?:everyone|all\s+your)',
        r'spread\s+(?:the\s+)?(?:word|news|message)',
    ]

    fwd_count = sum(1 for p in forward_signals if re.search(p, text_lower))
    chain_count = sum(1 for p in chain_signals if re.search(p, text_lower))
    panic_count = sum(1 for p in panic_signals if re.search(p, text_lower))

    exclamations = text.count('!')
    all_caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
    emoji_count  = len(re.findall(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]', text))
    question_marks = text.count('?')

    risk_score = (
        fwd_count * 20
        + chain_count * 25
        + panic_count * 12
        + exclamations * 3
        + all_caps_words * 4
        + emoji_count * 2
        + question_marks * 2
    )
    risk_score = min(risk_score, 100)

    if risk_score >= 65:
        risk_level = 'HIGH'
        risk_color = 'fake'
    elif risk_score >= 35:
        risk_level = 'MEDIUM'
        risk_color = 'uncertain'
    else:
        risk_level = 'LOW'
        risk_color = 'real'

    flags = []
    if fwd_count:   flags.append({'icon': 'fa-share-nodes',  'text': f'{fwd_count} forwarding cue(s) detected'})
    if chain_count: flags.append({'icon': 'fa-link',          'text': f'{chain_count} chain-letter pattern(s) found'})
    if panic_count: flags.append({'icon': 'fa-triangle-exclamation', 'text': f'{panic_count} panic/urgency phrase(s)'})
    if exclamations >= 3: flags.append({'icon': 'fa-exclamation', 'text': f'{exclamations} exclamation marks'})
    if all_caps_words >= 3: flags.append({'icon': 'fa-font', 'text': f'{all_caps_words} ALL-CAPS words'})

    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'forward_signals': fwd_count,
        'chain_signals':   chain_count,
        'panic_signals':   panic_count,
        'flags': flags,
    }


@app.route('/api/whatsapp-check', methods=['POST'])
def api_whatsapp_check():
    """Analyze a WhatsApp/Telegram forwarded message."""
    payload = request.get_json() or {}
    text = str(payload.get('text', '')).strip()
    if len(text) < 5:
        return jsonify({'success': False, 'message': 'Text too short'}), 400
    wa_result = analyze_whatsapp_forward(text)
    text_result = analyze_text(text)
    return jsonify({
        'success': True,
        'whatsapp': wa_result,
        'text_analysis': text_result,
    })


# ─────────────────────────────────────────────
# FEATURE 3 — User Feedback / Voting
# ─────────────────────────────────────────────
def _load_feedback():
    if FEEDBACK_PATH.exists():
        try:
            with open(FEEDBACK_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'votes': [], 'summary': {'agree': 0, 'disagree': 0, 'total': 0}}


def _save_feedback(data):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """Store user vote on whether the ML prediction was correct."""
    payload = request.get_json() or {}
    vote = str(payload.get('vote', '')).strip().lower()   # 'agree' or 'disagree'
    prediction = str(payload.get('prediction', '')).strip().upper()
    text_preview = str(payload.get('text_preview', ''))[:200]
    if vote not in ('agree', 'disagree'):
        return jsonify({'success': False, 'message': 'vote must be agree or disagree'}), 400

    data = _load_feedback()
    data['votes'].append({
        'vote': vote,
        'prediction': prediction,
        'text_preview': text_preview,
        'timestamp': datetime.now().isoformat(),
    })
    data['summary']['agree']    = sum(1 for v in data['votes'] if v['vote'] == 'agree')
    data['summary']['disagree'] = sum(1 for v in data['votes'] if v['vote'] == 'disagree')
    data['summary']['total']    = len(data['votes'])
    _save_feedback(data)
    return jsonify({'success': True, 'summary': data['summary']})


@app.route('/api/feedback-stats', methods=['GET'])
def api_feedback_stats():
    """Return aggregated voting statistics."""
    data = _load_feedback()
    return jsonify({'success': True, 'summary': data['summary']})


# ─────────────────────────────────────────────
# FEATURE 4 — Live Credibility Dashboard
# ─────────────────────────────────────────────
_dashboard_cache = {'headlines': [], 'fetched_at': None}
_dashboard_lock = Lock()


def _fetch_dashboard_headlines(max_items=30):
    """Fetch headlines from RSS feeds and analyse each for credibility."""
    global _dashboard_cache

    # Serve cache if fresh (< 10 min)
    with _dashboard_lock:
        cached_at = _dashboard_cache['fetched_at']
        cached_headlines = list(_dashboard_cache.get('headlines', []))
    if cached_at:
        age = (datetime.now() - cached_at).total_seconds()
        if age < 600 and cached_headlines:
            return cached_headlines

    feeds = [
        ('Reuters',     'https://feeds.reuters.com/reuters/topNews'),
        ('BBC',         'https://feeds.bbci.co.uk/news/rss.xml'),
        ('AP News',     'https://feeds.apnews.com/apnews/topnews'),
        ('Al Jazeera',  'https://www.aljazeera.com/xml/rss/all.xml'),
        ('The Guardian','https://www.theguardian.com/world/rss'),
        ('NPR',         'https://feeds.npr.org/1001/rss.xml'),
    ]

    headlines = []
    seen = set()
    per_source = max(4, max_items // len(feeds))

    for source_name, feed_url in feeds:
        try:
            req = urllib.request.Request(
                feed_url,
                headers={'User-Agent': 'Mozilla/5.0 (VeriCleri Dashboard)'},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                payload = resp.read()
            root = ET.fromstring(payload)
            count = 0
            for item in root.findall('.//item'):
                title = (item.findtext('title') or '').strip()
                link  = (item.findtext('link')  or '').strip()
                desc  = (item.findtext('description') or '').strip()
                pub   = (item.findtext('pubDate') or '').strip()
                if not title or title.lower() in seen:
                    continue
                seen.add(title.lower())

                combined = re.sub(r'<[^>]+>', ' ', f'{title}. {desc}')
                combined = html.unescape(re.sub(r'\s+', ' ', combined)).strip()
                if len(combined) < 20:
                    combined = title

                result = analyze_text(combined)
                headlines.append({
                    'title':      title,
                    'link':       link,
                    'source':     source_name,
                    'published':  pub,
                    'prediction': result.get('prediction', 'UNCERTAIN'),
                    'confidence': result.get('confidence', 50),
                    'real_pct':   result.get('real_percentage', 50),
                    'fake_pct':   result.get('fake_percentage', 50),
                    'text':       combined[:300],
                })
                count += 1
                if count >= per_source:
                    break
        except Exception:
            continue
        if len(headlines) >= max_items:
            break

    with _dashboard_lock:
        _dashboard_cache = {'headlines': headlines, 'fetched_at': datetime.now()}
    return headlines


@app.route('/dashboard')
def dashboard():
    """Live credibility dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/dashboard-headlines', methods=['GET'])
def api_dashboard_headlines():
    """Return live-analysed headlines for the dashboard."""
    try:
        max_items = int(request.args.get('max', 30))
        headlines = _fetch_dashboard_headlines(max_items=min(max_items, 50))
        with _dashboard_lock:
            fetched_at = _dashboard_cache.get('fetched_at')
        real_count      = sum(1 for h in headlines if h['prediction'] == 'REAL')
        fake_count      = sum(1 for h in headlines if h['prediction'] == 'FAKE')
        uncertain_count = sum(1 for h in headlines if h['prediction'] == 'UNCERTAIN')
        return jsonify({
            'success':   True,
            'headlines': headlines,
            'stats': {
                'total':     len(headlines),
                'real':      real_count,
                'fake':      fake_count,
                'uncertain': uncertain_count,
            },
            'fetched_at': fetched_at.isoformat() if fetched_at else None,
        })
    except Exception as err:
        return jsonify({'success': False, 'message': str(err)}), 500


# ─────────────────────────────────────────────────────
# SCHEDULER STATUS API
# ─────────────────────────────────────────────────────
@app.route('/api/scheduler-status', methods=['GET'])
def api_scheduler_status():
    """Return auto-retrain scheduler info."""
    return jsonify({
        'success': True,
        'scheduler': _scheduler_status,
        'model': {
            'accuracy': ml_model.metrics.get('accuracy'),
            'trained_at': ml_model.metrics.get('trained_at'),
            'dataset_size': ml_model.dataset_size,
        },
    })


@app.route('/api/scheduler-run-now', methods=['POST'])
@_rl('2 per hour')
def api_scheduler_run_now():
    """Manually trigger one auto-retrain cycle immediately."""
    admin_error = _require_admin()
    if admin_error:
        return admin_error
    if _bg_scheduler is not None and _scheduler_status.get('enabled'):
        try:
            _bg_scheduler.modify_job('auto_retrain', next_run_time=datetime.now())
            return jsonify({'success': True, 'message': 'Retrain triggered via scheduler'})
        except Exception as err:
            pass
    # Fallback: run synchronously
    _auto_retrain_job()
    return jsonify({'success': True, 'message': 'Retrain completed', 'result': _scheduler_status.get('last_result')})


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error="Server error"), 500

if __name__ == '__main__':
    # Create necessary folders if they don't exist
    folders = [
        'templates',
        'static/css',
        'static/js',
        'data',
        'data/training',
        'data/raw_datasets',
        'utils',
        'ml_model',
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("=" * 60)
    print("FAKE NEWS DETECTION WEBSITE - VeriCleri")
    print("=" * 60)
    print("Project structure created successfully!")
    print("Starting server...")
    print("Open: http://localhost:5000")
    print("Detection: http://localhost:5000/detect")
    print("Dashboard: http://localhost:5000/dashboard")
    print("=" * 60)
    print("Press CTRL+C to stop the server")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)