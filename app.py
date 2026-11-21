"""SpamLens Streamlit application: English SMS spam checker (COMP6885001).

Run from the project folder:
    streamlit run app.py

Needs models/ (weights + calibration JSON) and results/ (tables and figures) next to this file.
Both are produced by notebooks/traditional_models.ipynb and notebooks/transformer_models.ipynb.
User messages stay in session memory only.
"""

import csv
import html
import io
import json
import logging
import os
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import expit, logit

# Model-loading problems are written to the terminal that runs the app (never message text)
logger = logging.getLogger("spamlens")

# ---------------------------------------------------------------------------
# Text normalization (identical to the notebooks)
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|ly|me|co|info|io|uk|au|in|us|xyz|top|link|site|online|club|app|gl|biz)\b(?:/\S*)?",
    re.I,
)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MASK_TAGS = re.compile(r"<(?:#|DECIMAL|TIME)>", re.I)
MASK_URL = re.compile(r"<URL>", re.I)
MASK_EMAIL = re.compile(r"<EMAIL>", re.I)
DIGIT = r"[0-9X]"  # a real digit, or the uppercase X that the datasets use to hide digits
PHONE_PATTERN = re.compile(rf"(?<![A-Za-z0-9])\+?{DIGIT}(?:[\s\-()]?{DIGIT}){{6,}}(?![A-Za-z0-9])")
MONEY_PATTERN = re.compile(
    rf"(?:[£$€₹]|\b(?:rs|inr|usd|gbp|eur|php|ngn|ksh|aud|sgd)\.?)\s?{DIGIT}[0-9X,.]*(?:\s?(?:k|m|lakh|lac|lacs|crore|cr)\b)?",
    re.I,
)
NUMBER_PATTERN = re.compile(rf"(?<![A-Za-z0-9]){DIGIT}*\d{DIGIT}*(?:[.,:/]{DIGIT}+)*(?![A-Za-z0-9])|\bX{{2,}}\b|(?<![A-Za-z'\-])X(?![A-Za-z'\-])")
CONTROL_TOKENS = re.compile(r"</?s>|<pad>|<unk>|<mask>|<extra_id_\d+>")


def normalize(text: str) -> str:
    """Replace URLs, emails, money, phone numbers and other numbers (and the datasets' masks) with placeholders."""
    text = unicodedata.normalize("NFKC", str(text))
    text = MASK_URL.sub(" [URL] ", text)
    text = MASK_EMAIL.sub(" [EMAIL] ", text)
    text = MASK_TAGS.sub(" [NUM] ", text)
    text = EMAIL_PATTERN.sub(" [EMAIL] ", text)
    text = URL_PATTERN.sub(" [URL] ", text)
    text = MONEY_PATTERN.sub(" [MONEY] ", text)
    text = PHONE_PATTERN.sub(" [PHONE] ", text)
    text = NUMBER_PATTERN.sub(" [NUM] ", text)
    text = re.sub(r"(?:\[NUM\]\s*){2,}", "[NUM] ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_lowercase(text: str) -> str:
    """Input for the case-sensitive models (RoBERTa + SetFit, FLAN-T5)."""
    return CONTROL_TOKENS.sub(" ", normalize(text).lower())

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
REPO_URL = "https://github.com/Verrelllikecoding/spam-lens"
HF_MODEL_ID = None  # optional Hugging Face Hub id for DistilBERT when models/distilbert is absent

DEFAULT_MODEL = "distilbert"
MAX_CHARS = 1000
BATCH_MAX_ROWS = 1000
EXPLAIN_MAX_WORDS = 60
EXPLAIN_TOP_K = 8
CPU_THREADS = 12
MAX_LENGTH = {"distilbert": 128, "flan_t5": 128}

CLASSICAL_NAMES = ("tfidf_nb", "tfidf_lr")
EXTENSION_MODEL_NAMES = ("roberta_setfit", "flan_t5")  # calibration stored in models/calibration_extensions.json

MODEL_LABELS = {
    "distilbert": "DistilBERT",
    "tfidf_lr": "TF-IDF + Logistic Regression",
    "tfidf_nb": "TF-IDF + Multinomial NB",
    "roberta_setfit": "RoBERTa + SetFit",
    "flan_t5": "FLAN-T5",
}
MODEL_DESCRIPTIONS = {
    "distilbert": "A compact transformer that learns message context.",
    "tfidf_lr": "Word and character patterns with logistic regression.",
    "tfidf_nb": "A simple, interpretable word-pattern baseline.",
    "roberta_setfit": "A RoBERTa sentence encoder fine-tuned with SetFit contrastive learning and a logistic-regression head.",
    "flan_t5": "An encoder-decoder language model, fine-tuned for spam detection.",
}
# Models whose weights are absent from a fresh checkout (they are not stored in Git).
OPTIONAL_MODELS = {"distilbert", "roberta_setfit", "flan_t5"}
BAND_LABELS = {
    "likely_spam": "Likely spam",
    "suspicious": "Suspicious — verify before acting",
    "likely_ham": "Likely legitimate",
}
DISCLAIMER = (
    "This is an automated estimate and may be wrong. Do not click links or share "
    "codes/passwords in unexpected messages; verify with the official sender."
)
LANGUAGE_WARNING = (
    "This message may not be in English. The model was trained on English SMS, "
    "so this result is less reliable."
)
MAX_CSV_BYTES = 2_000_000

DATA_SOURCES = [
    {"source": "UCI SMS Spam Collection", "license": "CC BY 4.0", "citation": "Almeida and Hidalgo (2011), DOI 10.24432/C5CC84"},
    {"source": "Super SMS Dataset", "license": "Research release; academic use; no redistribution", "citation": "Salman et al. (2024), DOI 10.1109/ACCESS.2024.3364671"},
]

# Anonymized held-out test messages shown as one-click examples.
EXAMPLES = [{'title': 'Spam: prize claim',
  'text': 'Ur cash-balance is currently [MONEY] - to maximize ur cash-in now send CASH to [NUM] only '
          '[MONEY]/msg. CC: [PHONE] PO BOX [NUM] [LOCATION]'},
 {'title': 'Spam: recipe link',
  'text': 'This Crispy Fish Pakora recipe is so easy to make, you just need few ingredients to put together! '
          'Click the link: [URL]'},
 {'title': 'Spam: medicine offer',
  'text': 'Apollo Authentic Medicines Home Delivered in [NUM] Hrs Use Code [CODE] USER [NUM] for [NUM] '
          'Savings Download Now [URL] to book your first order'},
 {'title': 'Ham: checking in', 'text': 'Are you ignoring me?'},
 {'title': 'Ham: project plans', 'text': 'Hey, we r now deciding role for the project.. What do u wanna do?'},
 {'title': 'Ham: funny video', 'text': 'Rofl, dat was a funny video!'}]

# Hand-written borderline messages (no ground truth); also used by notebooks/transformer_models.ipynb.
AMBIGUOUS_EXAMPLES = [{'id': 'A01',
  'category': 'Security alert without a link',
  'text': 'URGENT: your account was locked after an unusual login attempt. Please call the number on the '
          'back of your card.',
  'why_ambiguous': 'Urgent scam-like wording, but it sends the reader to a trusted number and asks for '
                   'nothing.',
  'lean': 'unclear'},
 {'id': 'A02',
  'category': 'One-time code',
  'text': 'Your OTP is 482913. Do NOT share this code with anyone. If you did not request it, call 1500888.',
  'why_ambiguous': 'A normal transactional code, yet the same text appears in account-takeover attempts.',
  'lean': 'ham'},
 {'id': 'A03',
  'category': 'Possible wrong-number opener',
  'text': "Hi, is this still Sarah's number? We met at the conference last month and I wanted to follow up.",
  'why_ambiguous': 'Could be a real contact or the opener of a trust-building scam. Nothing in the text '
                   'decides it.',
  'lean': 'unclear'},
 {'id': 'A04',
  'category': 'Family impersonation',
  'text': "Hey it's mum, I broke my phone so this is my new number. Can you text me back when you're free?",
  'why_ambiguous': 'Sounds personal, and it is also the first step of a well-known family-impersonation '
                   'scam.',
  'lean': 'unclear'},
 {'id': 'A05',
  'category': 'Delivery notice with a link',
  'text': 'Your package could not be delivered today. Reschedule here: https://track.example.com/a8f2',
  'why_ambiguous': 'Couriers send this every day, and so do parcel scammers. Only the sender and link can '
                   'separate them.',
  'lean': 'unclear'},
 {'id': 'A06',
  'category': 'Spam words in a personal message',
  'text': "You won!! Free drinks on me tonight at Joe's, come celebrate my promotion with us",
  'why_ambiguous': 'Uses the words won, free, and promotion, but it is an invitation between friends.',
  'lean': 'ham'},
 {'id': 'A07',
  'category': 'Opted-in business promotion',
  'text': 'Reminder: 20% off your next check-up if you book before Friday. Reply STOP to unsubscribe. - '
          'Bright Smile Dental',
  'why_ambiguous': 'A legitimate business and possibly wanted by the recipient, yet still marketing.',
  'lean': 'spam'},
 {'id': 'A08',
  'category': 'Very short and urgent',
  'text': "Call me now, it's urgent",
  'why_ambiguous': 'Too short to judge. It suits both a worried friend and a callback scam.',
  'lean': 'ham'},
 {'id': 'A09',
  'category': 'Too-good job offer',
  'text': 'We reviewed your CV and would like to offer you a part-time role at $45/hour from home. WhatsApp '
          'HR on +1 555 010 7788.',
  'why_ambiguous': 'Recruiters do text applicants, but high pay and moving the chat to WhatsApp are scam '
                   'signs.',
  'lean': 'spam'},
 {'id': 'A10',
  'category': 'Billing receipt with a link',
  'text': 'Thanks for your payment of $120.50. Your balance is now $0. Questions? Visit example.com/billing',
  'why_ambiguous': 'A plain receipt that has the money amount and link typical of spam.',
  'lean': 'ham'},
 {'id': 'A11',
  'category': 'Charity appeal',
  'text': 'Flood relief: your $5 can feed a family this week. Text GIVE to 70123. Msg rates apply.',
  'why_ambiguous': 'Real charities fundraise this way, and scammers copy the format after disasters.',
  'lean': 'unclear'},
 {'id': 'A12',
  'category': 'Two-word question',
  'text': 'Free tonight?',
  'why_ambiguous': 'Contains a classic spam word, and has almost no other context.',
  'lean': 'ham'}]

SAMPLE_MESSAGES = ['Ur cash-balance is currently [MONEY] - to maximize ur cash-in now send CASH to [NUM] only [MONEY]/msg. CC: '
 '[PHONE] PO BOX [NUM] [LOCATION]',
 'This Crispy Fish Pakora recipe is so easy to make, you just need few ingredients to put together! Click '
 'the link: [URL]',
 'Apollo Authentic Medicines Home Delivered in [NUM] Hrs Use Code [CODE] USER [NUM] for [NUM] Savings '
 'Download Now [URL] to book your first order',
 'Thankyou for your interest in SBI Credit Card Please click herethis SMS to [PHONE] within the next [NUM] '
 'hour to authorize SBI Card to access your bureau record',
 'Are you ignoring me?',
 'Hey, we r now deciding role for the project.. What do u wanna do?',
 'Rofl, dat was a funny video!',
 'Alright i have a new goal now',
 'They at clubhouse alr.',
 'No then it costs money']

TEAM = [{'Name': 'Raynald Arvan Lim'},
 {'Name': 'Hans Ewaldo Kristiawan'},
 {'Name': 'Christian Verrell'},
 {'Name': 'Dorian Joel Robert Melenotte'},
 {'Name': 'Samuel Tristan Gani'},
 {'Name': 'Thania Calista'}]


# ---------------------------------------------------------------------------
# Model loading: every scorer returns raw (uncalibrated) spam logits
# ---------------------------------------------------------------------------
def _torch_threads(torch) -> None:
    torch.set_num_threads(min(CPU_THREADS, os.cpu_count() or 1))


class RawScorer:
    """TF-IDF pipelines (joblib) and Hugging Face sequence classifiers (DistilBERT, FLAN-T5)."""

    def __init__(self, name: str):
        self.name = name
        if name in CLASSICAL_NAMES:
            self.model = joblib.load(MODELS / f"{name}.joblib")
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        _torch_threads(torch)
        local = MODELS / name
        hub_id = HF_MODEL_ID if name == "distilbert" else None
        location = str(local) if (local / "config.json").exists() else hub_id
        if not location:
            raise FileNotFoundError(f"{name} weights are absent")
        self.tokenizer = AutoTokenizer.from_pretrained(location)
        self.model = AutoModelForSequenceClassification.from_pretrained(location).to("cpu").eval()

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty(0, dtype=float)
        if self.name in CLASSICAL_NAMES:
            raw = self.model.predict_proba([normalize(text) for text in texts])[:, 1]
            return logit(np.clip(raw, 1e-6, 1 - 1e-6))
        prepare = normalize if self.name == "distilbert" else prepare_lowercase
        logits = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), 32):
                batch = [prepare(text) for text in texts[start:start + 32]]
                encoded = self.tokenizer(batch, truncation=True, padding=True, max_length=MAX_LENGTH[self.name], return_tensors="pt")
                outputs = self.model(**encoded).logits
                logits.extend((outputs[:, 1] - outputs[:, 0]).numpy().astype(float))
        return np.asarray(logits)


class SetFitScorer:
    """RoBERTa + SetFit body with its logistic-regression head, on CPU."""

    name = "roberta_setfit"

    def __init__(self):
        location = MODELS / "roberta_setfit"
        if not (location / "model_head.pkl").exists():
            raise FileNotFoundError("roberta_setfit weights are absent")
        import torch
        from setfit import SetFitModel

        _torch_threads(torch)
        self.model = SetFitModel.from_pretrained(str(location), device="cpu")

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty(0, dtype=float)
        prepared = [prepare_lowercase(text) for text in texts]
        spam = np.asarray(self.model.predict_proba(prepared, batch_size=32, as_numpy=True))[:, 1]
        return logit(np.clip(spam.astype(float), 1e-6, 1 - 1e-6))


def make_scorer(name: str):
    return SetFitScorer() if name == "roberta_setfit" else RawScorer(name)


# ---------------------------------------------------------------------------
# Calibrated predictions with validation-chosen thresholds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Prediction:
    label: str
    spam_probability: float
    risk_band: str
    model_name: str
    latency_ms: float


class ModelPredictor:
    """Temperature-scaled probability and risk bands; thresholds come from models/calibration*.json."""

    def __init__(self, name: str):
        self.name = name
        filename = "calibration_extensions.json" if name in EXTENSION_MODEL_NAMES else "calibration.json"
        self.settings = json.loads((MODELS / filename).read_text(encoding="utf-8"))[name]
        self.scorer = make_scorer(name)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return expit(self.scorer.logits(texts) / self.settings["temperature"])

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        started = perf_counter()
        probabilities = self.predict_proba(texts)
        latency = (perf_counter() - started) * 1000 / max(1, len(probabilities))
        results = []
        for probability in probabilities:
            spam = probability >= self.settings["threshold"]
            band = "likely_spam" if spam else "suspicious" if probability >= self.settings["threshold_low"] else "likely_ham"
            results.append(Prediction("spam" if spam else "ham", float(probability), band, self.name, latency))
        return results


def load_predictor(name: str) -> ModelPredictor:
    return ModelPredictor(name)


# ---------------------------------------------------------------------------
# Input guards, batch CSV handling, explanations
# ---------------------------------------------------------------------------
def guard_text(text: str, max_chars: int = MAX_CHARS) -> tuple[str, bool]:
    """Trim blank input and enforce the character limit before inference."""
    text = text.strip()
    return text[:max_chars], len(text) > max_chars


def likely_non_english(text: str, vocabulary: set[str]) -> bool:
    tokens = re.findall(r"\b[^\W\d_]{3,}\b", normalize(text).lower())
    return len(tokens) >= 5 and sum(token in vocabulary for token in tokens) / len(tokens) < 0.4


def parse_batch(payload: bytes, max_rows: int = BATCH_MAX_ROWS) -> pd.DataFrame:
    """Validate a UTF-8 CSV before passing any content to a model."""
    if len(payload) > MAX_CSV_BYTES:
        raise ValueError("The CSV is larger than 2 MB. Upload a smaller file.")
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Save the CSV as UTF-8 before uploading it.") from error
    try:
        header = next(csv.reader(io.StringIO(decoded)))
        if header.count("text") != 1:
            raise ValueError("The CSV must contain exactly one column named text.")
        frame = pd.read_csv(io.StringIO(decoded), dtype=str, keep_default_na=False)
    except (StopIteration, pd.errors.EmptyDataError, pd.errors.ParserError, csv.Error) as error:
        raise ValueError("The file could not be read as a CSV with a text column.") from error
    if not 1 <= len(frame) <= max_rows:
        raise ValueError(f"Upload between 1 and {max_rows:,} message rows.")
    if frame.text.str.strip().eq("").any():
        raise ValueError("Every row needs a non-empty message in the text column.")
    return frame[["text"]].copy()


def predict_batch(frame: pd.DataFrame, predictor: ModelPredictor, max_chars: int = MAX_CHARS,
                  progress: Callable[[float], None] | None = None) -> pd.DataFrame:
    texts = [guard_text(text, max_chars)[0] for text in frame.text]
    predictions = []
    for start in range(0, len(texts), 32):
        predictions.extend(asdict(result) for result in predictor.predict(texts[start:start + 32]))
        if progress:
            progress(min((start + 32) / len(texts), 1.0))
    result = pd.DataFrame(predictions)
    result.insert(0, "text", texts)
    return result


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Keep exports in memory and neutralize spreadsheet formula prefixes."""
    output = frame.copy()
    if "text" in output:
        output["text"] = output.text.map(lambda text: "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text)
    return output.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


@dataclass(frozen=True)
class WordEffect:
    word: str
    word_index: int
    start: int
    end: int
    effect_pp: float


def explain(predictor: ModelPredictor, text: str, top_k: int = EXPLAIN_TOP_K, max_words: int = EXPLAIN_MAX_WORDS) -> list[WordEffect]:
    """Leave-one-word-out: remove each of the first max_words words and measure the probability change."""
    matches = list(re.finditer(r"\S+", text))[:max_words]
    if not matches or top_k <= 0:
        return []
    variants = [text[:match.start()] + text[match.end():] for match in matches]
    probabilities = predictor.predict_proba([text, *variants])
    changes = (probabilities[0] - probabilities[1:]) * 100
    selected = np.argsort(-np.abs(changes), kind="stable")[:top_k]
    return [WordEffect(matches[i].group(), int(i), matches[i].start(), matches[i].end(), float(changes[i])) for i in selected]


def highlighted_text(text: str, effects: list[WordEffect]) -> str:
    """Escape all user content and label direction independently of color."""
    parts, position = [], 0
    for effect in sorted(effects, key=lambda item: item.start):
        parts.append(html.escape(text[position:effect.start]))
        raises = effect.effect_pp >= 0
        direction = "raises spam score" if raises else "lowers spam score"
        arrow = "▲" if raises else "▼"  # the arrow shows the direction without relying on color
        parts.append(f'<mark class="{"up" if raises else "down"}" title="{direction} by {abs(effect.effect_pp):.1f} pp">'
                     f'{html.escape(text[effect.start:effect.end])} {arrow}</mark>')
        position = effect.end
    parts.append(html.escape(text[position:]))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Streamlit user interface
# ---------------------------------------------------------------------------
STYLE = """
<style>
.block-container {padding-top: 2.2rem; max-width: 900px;}
.hero {background: linear-gradient(135deg, #1e3a8a 0%, #4338ca 55%, #7c3aed 100%); color: #fff;
       padding: 26px 28px 22px; border-radius: 18px; margin-bottom: 1.2rem;}
.hero .eyebrow {letter-spacing: .14em; font-size: .72rem; font-weight: 700; opacity: .8;}
.hero h1 {color: #fff; font-size: 2.1rem; margin: .25rem 0 .35rem; padding: 0;}
.hero p {margin: 0 0 .8rem; opacity: .92; font-size: 1rem;}
.chip {display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
       border-radius: 999px; padding: 2px 11px; font-size: .76rem; margin: 0 6px 4px 0;}
.result-card {border-radius: 16px; padding: 18px 22px 14px; border: 1px solid; margin: .6rem 0 .8rem;}
.result-card.likely_spam {background: #fef2f2; border-color: #fca5a5;}
.result-card.suspicious {background: #fffbeb; border-color: #fcd34d;}
.result-card.likely_ham {background: #f0fdf4; border-color: #86efac;}
.result-top {display: flex; justify-content: space-between; align-items: flex-end; gap: 12px; flex-wrap: wrap;}
.result-label {font-size: 1.3rem; font-weight: 700; color: #111827;}
.result-sub {font-size: .85rem; color: #4b5563; margin-top: 2px;}
.result-prob {font-size: 2.4rem; font-weight: 800; line-height: 1; color: #111827; text-align: right;}
.result-prob small {display: block; font-size: .72rem; font-weight: 600; color: #6b7280; letter-spacing: .04em;}
.gauge {position: relative; height: 12px; border-radius: 999px; background: #e5e7eb; margin: 16px 0 4px;}
.gauge .fill {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px;}
.gauge .threshold {position: absolute; top: -6px; height: 24px; border-left: 2px dashed #111827;}
.gauge-labels {position: relative; height: 18px; font-size: .72rem; color: #6b7280;}
.gauge-labels span {position: absolute; transform: translateX(-50%); white-space: nowrap;}
.result-foot {font-size: .78rem; color: #6b7280; margin-top: 8px;}
.explained {line-height: 2.1; font-size: 1rem; padding: 12px 14px; background: #f9fafb; border-radius: 12px;}
mark.up {background: #fecaca; color: #7f1d1d; border-radius: 6px; padding: 1px 5px;}
mark.down {background: #bbf7d0; color: #14532d; border-radius: 6px; padding: 1px 5px;}
.compare-row {display: grid; grid-template-columns: 170px 1fr 62px 92px; gap: 10px; align-items: center;
              padding: 7px 0; border-bottom: 1px solid #f1f5f9; font-size: .9rem;}
.compare-bar {position: relative; height: 10px; background: #e5e7eb; border-radius: 999px;}
.compare-bar .fill {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px;}
.compare-bar .threshold {position: absolute; top: -4px; height: 18px; border-left: 2px dashed #374151;}
.pill {display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: .75rem; font-weight: 700; text-align: center;}
.pill.likely_spam {background: #fee2e2; color: #991b1b;}
.pill.suspicious {background: #fef3c7; color: #92400e;}
.pill.likely_ham {background: #dcfce7; color: #166534;}
.side-brand {font-size: 1.35rem; font-weight: 800; color: #1e3a8a; margin-bottom: 0;}
.footer {text-align: center; color: #9ca3af; font-size: .78rem; margin-top: 2.5rem;}
@media (max-width: 640px) {.compare-row {grid-template-columns: 110px 1fr 52px; } .compare-row .pill {display: none;}}
</style>
"""
BAND_STYLE = {
    "likely_spam": {"icon": "🚫", "color": "#dc2626", "short": "Spam"},
    "suspicious": {"icon": "⚠️", "color": "#d97706", "short": "Suspicious"},
    "likely_ham": {"icon": "✅", "color": "#16a34a", "short": "Legitimate"},
}
BAND_ADVICE = {
    "likely_spam": "Do not click links, reply, or share codes. Delete it or report it.",
    "suspicious": "Borderline message. Check the sender through an official channel before acting.",
    "likely_ham": "Looks like a normal message. Still be careful with unexpected requests.",
}


def read_table(filename: str) -> pd.DataFrame:
    path = RESULTS / "tables" / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def test_result(name: str) -> dict:
    """This model's row of results/tables/test_results.csv (written by the notebooks)."""
    table = read_table("test_results.csv")
    rows = table[table.model == name].to_dict("records") if "model" in table else []
    return rows[0] if rows else {}


@st.cache_resource(show_spinner=False)
def cached_predictor(name: str):
    """Cache only model resources; no message content enters this shared cache."""
    return load_predictor(name)


def resolve_predictor(requested: str):
    failed = st.session_state.setdefault("failed_models", set())
    try:
        if requested in failed:
            raise RuntimeError("Previously unavailable in this session")
        return cached_predictor(requested)
    except Exception:
        # UI errors intentionally omit exception text, which can contain sensitive input.
        logger.exception("Could not load model %s", requested)
        if requested not in OPTIONAL_MODELS:
            st.error("This model could not load. Check the local model files and try again.")
            st.stop()
        failed.add(requested)
        st.warning(f"{MODEL_LABELS[requested]} is unavailable. Using TF-IDF + Logistic Regression for this session.")
        try:
            return cached_predictor("tfidf_lr")
        except Exception:
            st.error("The fallback model could not load. Restore models/tfidf_lr.joblib and restart the app.")
            st.stop()


def sidebar():
    with st.sidebar:
        st.markdown('<p class="side-brand">🔎 SpamLens</p>', unsafe_allow_html=True)
        st.caption("English SMS spam checker · COMP6885001")
        names = list(MODEL_LABELS)
        requested = st.selectbox("Model", names, index=names.index(DEFAULT_MODEL), format_func=MODEL_LABELS.get, key="selected_model")
        st.caption(MODEL_DESCRIPTIONS[requested])
        if requested in st.session_state.get("failed_models", set()):
            if st.button(f"Retry {MODEL_LABELS[requested]}", key="retry_model"):
                st.session_state["failed_models"].discard(requested)
        with st.spinner("Loading model…"):
            predictor = resolve_predictor(requested)
        metrics = test_result(predictor.name)
        st.divider()
        left, right = st.columns(2)
        left.metric("Threshold", f"{predictor.settings['threshold']:.2f}", help="Spam if the probability is at or above this value (chosen on validation data).")
        if metrics:
            right.metric("Test macro-F1", f"{metrics['macro_f1']:.3f}")
            left.metric("Spam recall", f"{metrics['spam_recall']:.1%}")
            right.metric("False alarms", f"{metrics['ham_fpr']:.2%}", help="Share of legitimate test messages flagged as spam.")
        st.divider()
        st.link_button("View project on GitHub", REPO_URL, width="stretch")
        st.caption("🔒 Your messages stay in this session. They are never saved to files, logs, or analytics.")
    return predictor


def word_vocabulary() -> set[str]:
    model = cached_predictor("tfidf_lr").scorer.model
    word_vectorizer = dict(model.named_steps["features"].transformer_list)["word"]
    return set(word_vectorizer.vocabulary_)


def choose_example(text: str) -> None:
    st.session_state["sms_text"] = text
    st.session_state.pop("last_result", None)
    st.session_state.pop("comparison", None)


def enforce_message_limit() -> None:
    """Enforce the cap before widget creation, keeping evidence for the truncation warning."""
    raw = st.session_state.get("sms_text", "")
    if len(raw) > MAX_CHARS:
        st.session_state["sms_text"] = raw[:MAX_CHARS]
        st.session_state["text_truncated"] = True


def band_of(probability: float, settings: dict) -> str:
    if probability >= settings["threshold"]:
        return "likely_spam"
    return "suspicious" if probability >= settings["threshold_low"] else "likely_ham"


def gauge_html(probability: float, threshold: float, color: str) -> str:
    return (f'<div class="gauge"><div class="fill" style="width:{probability * 100:.1f}%;background:{color}"></div>'
            f'<div class="threshold" style="left:{threshold * 100:.1f}%"></div></div>'
            f'<div class="gauge-labels"><span style="left:2%">0%</span>'
            f'<span style="left:{min(max(threshold * 100, 14), 86):.1f}%">spam threshold {threshold:.0%}</span>'
            f'<span style="left:97%">100%</span></div>')


def display_result(text, result, predictor):
    style = BAND_STYLE[result.risk_band]
    ece = test_result(result.model_name).get("ece", 1.0)
    # "Probability" wording only when the test calibration error (ECE) is small
    wording = "SPAM PROBABILITY" if ece <= 0.03 else "SPAM SCORE"
    st.markdown(
        f'<div class="result-card {result.risk_band}">'
        f'<div class="result-top"><div><div class="result-label">{style["icon"]} {BAND_LABELS[result.risk_band]}</div>'
        f'<div class="result-sub">{html.escape(BAND_ADVICE[result.risk_band])}</div></div>'
        f'<div class="result-prob"><small>{wording}</small>{result.spam_probability:.1%}</div></div>'
        f'{gauge_html(result.spam_probability, predictor.settings["threshold"], style["color"])}'
        f'<div class="result-foot">{MODEL_LABELS[result.model_name]} · {result.latency_ms:.1f} ms · {html.escape(DISCLAIMER)}</div>'
        f'</div>', unsafe_allow_html=True)

    with st.expander("🔍 Why this result?", expanded=True):
        cache = st.session_state.setdefault("explanations", {})
        key = (result.model_name, text)
        try:
            if key not in cache:
                with st.spinner("Measuring the effect of each word…"):
                    cache[key] = explain(predictor, text, EXPLAIN_TOP_K, EXPLAIN_MAX_WORDS)
                while len(cache) > 5:
                    del cache[next(iter(cache))]
            effects = cache[key]
            st.markdown('<div class="explained">' + highlighted_text(text, effects) + "</div>", unsafe_allow_html=True)
            st.caption("🟥 ▲ raises the spam score · 🟩 ▼ lowers it. Each word is removed once and the change in probability is measured.")
            table = pd.DataFrame([{"Word": item.word, "Effect (percentage points)": round(item.effect_pp, 2)} for item in effects])
            st.dataframe(table, hide_index=True, width="stretch")
            if len(text.split()) > EXPLAIN_MAX_WORDS:
                st.caption("Explanation covers the first 60 words; the prediction uses the full message within the model's token limit.")
        except Exception:
            st.info("The prediction is ready, but its explanation is temporarily unavailable.")

    with st.expander("⚖️ Compare all five models on this message"):
        st.caption("Useful for borderline messages. Each model uses its own threshold (dashed line). Loading every model the first time takes a while.")
        comparison = st.session_state.get("comparison")
        if st.button("Run all models", key="compare_models"):
            rows = []
            with st.spinner("Running all five models…"):
                for name in MODEL_LABELS:
                    try:
                        model = cached_predictor(name)
                    except Exception:
                        logger.exception("Could not load model %s", name)
                        rows.append((name, None, None))
                        continue
                    try:
                        probability = float(model.predict_proba([text])[0])
                        rows.append((name, probability, model.settings))
                    except Exception as error:
                        logger.error("Model %s could not score the message (%s)", name, type(error).__name__)
                        rows.append((name, None, None))
            comparison = st.session_state["comparison"] = (text, rows)
        if comparison and comparison[0] == text:
            parts = []
            for name, probability, settings in comparison[1]:
                if probability is None:
                    parts.append(f'<div class="compare-row"><b>{MODEL_LABELS[name]}</b><span>unavailable</span><span></span><span></span></div>')
                    continue
                band = band_of(probability, settings)
                parts.append(
                    f'<div class="compare-row"><b>{MODEL_LABELS[name]}</b>'
                    f'<div class="compare-bar"><div class="fill" style="width:{probability * 100:.1f}%;background:{BAND_STYLE[band]["color"]}"></div>'
                    f'<div class="threshold" style="left:{settings["threshold"] * 100:.1f}%"></div></div>'
                    f'<span style="text-align:right">{probability:.0%}</span><span class="pill {band}">{BAND_STYLE[band]["short"]}</span></div>')
            st.markdown("".join(parts), unsafe_allow_html=True)


def single_tab(predictor):
    st.markdown("#### Paste a message and check it before you act")
    with st.expander("✨ Try an example", expanded=True):
        columns = st.columns(3)
        for index, example in enumerate(EXAMPLES):
            columns[index % 3].button(example["title"], key=f"example_{index}", on_click=choose_example, args=(example["text"],), width="stretch")
    with st.expander("🤔 Try an ambiguous example"):
        st.caption("Hand-written borderline messages with no single correct answer. Hover a button to see why it is ambiguous, then compare the models.")
        columns = st.columns(2)
        for index, example in enumerate(AMBIGUOUS_EXAMPLES):
            columns[index % 2].button(f"{example['id']} · {example['category']}", key=f"ambiguous_{index}", on_click=choose_example, args=(example["text"],), width="stretch", help=example["why_ambiguous"])
    # Streamlit's max_chars deserializer truncates before callbacks can detect it,
    # so the same 1,000-character cap is enforced here to show a truthful warning.
    enforce_message_limit()
    text = st.text_area("Message", placeholder="Paste an SMS here…", key="sms_text", on_change=enforce_message_limit, height=130)
    clean, truncated = guard_text(text, MAX_CHARS)
    left, right = st.columns([3, 1])
    left.caption(f"{len(text):,} / {MAX_CHARS:,} characters")
    if truncated or st.session_state.pop("text_truncated", False):
        st.warning("Your message was longer than 1,000 characters and was truncated.")
    if clean and likely_non_english(clean, word_vocabulary()):
        st.warning(LANGUAGE_WARNING)
    if right.button("Check message", type="primary", disabled=not clean, key="check_message", width="stretch"):
        with st.spinner("Checking message…"):
            try:
                result = predictor.predict([clean])[0]
                st.session_state["last_result"] = (clean, result)
                st.session_state.pop("comparison", None)
                history = st.session_state.setdefault("history", [])
                history.insert(0, {"Message": clean, "Result": BAND_LABELS[result.risk_band], "Spam score": f"{result.spam_probability:.1%}", "Model": MODEL_LABELS[result.model_name]})
                del history[5:]
            except Exception:
                st.error("This message could not be checked. Please try again.")
    if not clean:
        st.info("Paste a message or choose an example to begin.")
    previous = st.session_state.get("last_result")
    if previous and previous[0] == clean and previous[1].model_name == predictor.name:
        display_result(*previous, predictor)
    history = st.session_state.get("history", [])
    if history:
        with st.expander("🕘 Recent checks · this session only"):
            st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
            if st.button("Clear history", key="clear_history"):
                for key in ("history", "last_result", "explanations", "comparison"):
                    st.session_state.pop(key, None)
                st.rerun()


def batch_tab(predictor):
    st.markdown("#### Check many messages at once")
    st.write("Upload a UTF-8 CSV with a `text` column: up to 1,000 rows and 2 MB.")
    sample = pd.DataFrame({"text": SAMPLE_MESSAGES}).to_csv(index=False, lineterminator="\n").encode("utf-8")
    st.download_button("⬇️ Download sample CSV", sample, file_name="sample_messages.csv", mime="text/csv", key="sample_csv")
    upload = st.file_uploader("Messages CSV", type=["csv"], key="batch_file")
    if upload is None:
        return
    try:
        frame = parse_batch(upload.getvalue(), BATCH_MAX_ROWS)
    except ValueError as error:
        st.error(str(error))
        return
    st.caption(f"{len(frame):,} messages ready · {MODEL_LABELS[predictor.name]}")
    if frame.text.str.len().gt(MAX_CHARS).any():
        st.warning("Messages longer than 1,000 characters will be truncated before checking.")
    if st.button("Check batch", type="primary", key="check_batch"):
        progress = st.progress(0.0)
        try:
            result = predict_batch(frame, predictor, MAX_CHARS, progress.progress)
            st.session_state["batch_result"] = (upload.file_id, predictor.name, result)
        except Exception:
            st.error("The batch could not be checked. Please try again with a smaller file.")
        finally:
            progress.empty()
    previous = st.session_state.get("batch_result")
    if previous and previous[:2] == (upload.file_id, predictor.name):
        result = previous[2]
        counts = result.risk_band.value_counts()
        columns = st.columns(3)
        for column, band in zip(columns, BAND_LABELS, strict=True):
            column.metric(f"{BAND_STYLE[band]['icon']} {BAND_STYLE[band]['short']}", int(counts.get(band, 0)))
        view = result[["text", "spam_probability", "risk_band"]].assign(risk_band=result.risk_band.map(BAND_LABELS))
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={"text": "Message", "risk_band": "Result",
                                    "spam_probability": st.column_config.ProgressColumn("Spam probability", format="percent", min_value=0.0, max_value=1.0)})
        st.caption(DISCLAIMER)
        st.download_button("⬇️ Download results CSV", csv_bytes(result), file_name="spamlens_predictions.csv", mime="text/csv", key="results_csv")


def percent_columns(frame: pd.DataFrame, columns: list[str]) -> dict:
    labels = {"spam_precision": "Spam precision", "spam_recall": "Spam recall", "ham_fpr": "False alarms (ham)"}
    return {column: st.column_config.NumberColumn(labels[column], format="percent") for column in columns if column in frame}


def performance_tab(predictor):
    st.markdown("#### Measured on held-out test data")
    st.caption("Models are trained in the notebooks; thresholds are chosen on validation data. No training runs inside this app.")
    results = read_table("test_results.csv")
    metrics = test_result(predictor.name)
    if metrics:
        st.markdown(f"**{MODEL_LABELS[predictor.name]}** (selected model)")
        columns = st.columns(4)
        columns[0].metric("Macro-F1", f"{metrics['macro_f1']:.4f}")
        columns[1].metric("Spam precision", f"{metrics['spam_precision']:.2%}")
        columns[2].metric("Spam recall", f"{metrics['spam_recall']:.2%}")
        columns[3].metric("False alarms (ham)", f"{metrics['ham_fpr']:.2%}")
    if not results.empty:
        st.markdown("**All models · balanced test set (2,873 messages)**")
        table = results[["model", "macro_f1", "spam_precision", "spam_recall", "ham_fpr", "ece"]].sort_values("macro_f1", ascending=False)
        table = table.assign(model=table.model.map(MODEL_LABELS))
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"model": "Model", "macro_f1": st.column_config.NumberColumn("Macro-F1", format="%.4f"),
                                    "ece": st.column_config.NumberColumn("ECE", format="%.4f"),
                                    **percent_columns(table, ["spam_precision", "spam_recall", "ham_fpr"])})
    confusion_figure = RESULTS / "figures" / f"confusion_{predictor.name}.png"
    if confusion_figure.exists():
        left, middle, right = st.columns([1, 2, 1])
        middle.image(str(confusion_figure), caption=f"Confusion matrix · {MODEL_LABELS[predictor.name]}", width="stretch")
    natural = read_table("natural_prevalence.csv")
    if not natural.empty:
        st.markdown("**Realistic inbox · 13.4% spam**")
        table = natural[["model", "spam_precision", "spam_recall", "false_alarms_per_1000_ham"]]
        table = table.assign(model=table.model.map(MODEL_LABELS))
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"model": "Model", "false_alarms_per_1000_ham": st.column_config.NumberColumn("False alarms per 1,000 ham", format="%.1f"),
                                    **percent_columns(table, ["spam_precision", "spam_recall"])})
        st.caption("A 50/50 test set makes precision look better than in a real inbox, where most messages are legitimate.")
    split = read_table("data_split.csv")
    if not split.empty:
        st.markdown("**Dataset**")
        total = int(split[split["split"] != "unused_ham_pool"]["total"].sum())
        st.write(f"{total:,} curated messages · equal ham/spam · seed 42 · train/validation/test 70/15/15")
        st.dataframe(pd.DataFrame(DATA_SOURCES), hide_index=True, width="stretch")
        with st.expander("Messages removed during curation"):
            st.dataframe(read_table("curation_summary.csv"), hide_index=True, width="stretch")


def about_tab():
    st.markdown("#### An assistive first-line SMS filter")
    st.write("SpamLens estimates whether an English SMS resembles spam. It cannot verify a sender or guarantee that a message is safe.")
    columns = st.columns(3)
    columns[0].info("**1. Normalize**\n\nLinks, numbers, money and phone numbers become placeholders.")
    columns[1].info("**2. Estimate**\n\nA calibrated model gives the spam probability.")
    columns[2].info("**3. Explain**\n\nThe words that moved the score are highlighted.")
    st.markdown("**Data sources and attribution**")
    st.markdown("[UCI SMS Spam Collection](https://doi.org/10.24432/C5CC84), Almeida & Hidalgo (2011), CC BY 4.0; Almeida et al., DocEng 2011, DOI 10.1145/2034691.2034742.")
    st.markdown("[Super SMS Dataset](https://doi.org/10.1109/ACCESS.2024.3364671), Salman, Ikram & Kaafar (IEEE Access, 2024). Author-published research data without an explicit license file; this project uses it for non-commercial academic work and does not redistribute it.")
    st.markdown("[DistilBERT](https://arxiv.org/abs/1910.01108), Sanh et al. (2019), Apache-2.0 pretrained model. Project code: MIT.")
    st.markdown("[RoBERTa](https://arxiv.org/abs/1907.11692), Liu et al. (2019), trained with [SetFit](https://arxiv.org/abs/2209.11055), Tunstall et al. (2022). [FLAN-T5](https://arxiv.org/abs/2210.11416), Chung et al. (2022), Apache-2.0 pretrained model `google/flan-t5-base`.")
    st.markdown("**Limitations**")
    st.write("English SMS only; uneven regional and dialect coverage; training sources end in 2024; spam changes over time. Published labels contain borderline promotional or transactional messages. No URL reputation or sender checks are performed.")
    st.write("Obfuscated wording and informal or mixed-language messages can cause errors. The score is an automated estimate; verify unexpected requests independently.")
    st.markdown("**Privacy**")
    st.write("Messages and uploads are processed in memory and are never written to files, logs, or analytics. Recent checks and explanations belong only to this session and clear when its connection ends. Use Clear history to remove recent single-message checks immediately.")
    st.caption(DISCLAIMER)
    st.markdown("**Team · COMP6885001 — Natural Language Processing**")
    st.dataframe(pd.DataFrame(TEAM), hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="SpamLens · SMS Spam Checker", page_icon="🔎", layout="centered")
    st.markdown(STYLE, unsafe_allow_html=True)
    predictor = sidebar()
    st.markdown(
        '<div class="hero"><div class="eyebrow">SPAMLENS · ENGLISH SMS SPAM DETECTION</div>'
        "<h1>Pause. Check. Verify.</h1>"
        "<p>Paste a suspicious SMS and get a calibrated spam probability, the words behind it, and a second opinion from five models.</p>"
        '<span class="chip">5 models</span><span class="chip">19,152 curated messages</span>'
        '<span class="chip">Calibrated probabilities</span><span class="chip">Nothing is stored</span></div>',
        unsafe_allow_html=True)
    tabs = st.tabs(["✉️ Check a message", "📄 Batch check", "📊 Model performance", "ℹ️ About"])
    for tab, render in zip(tabs[:3], (single_tab, batch_tab, performance_tab), strict=True):
        with tab:
            render(predictor)
    with tabs[3]:
        about_tab()
    st.markdown('<div class="footer">SpamLens · COMP6885001 Natural Language Processing · an automated estimate, not a guarantee</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
