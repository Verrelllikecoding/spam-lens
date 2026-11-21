# SpamLens: English SMS Spam Detection

COMP6885001 Natural Language Processing project. SpamLens compares two classical models and three transformers for English SMS spam detection, and serves them in a Streamlit app with calibrated probabilities, word-level explanations, and a five-model comparison.

## Results (balanced test set, 2,873 messages)

| Model | Macro-F1 | Spam precision | Spam recall | Ham false-positive rate |
|---|---|---|---|---|
| TF-IDF + Multinomial Naive Bayes | 0.9673 | 99.27% | 94.15% | 0.70% |
| TF-IDF + Logistic Regression | 0.9854 | 99.08% | 97.98% | 0.90% |
| **DistilBERT** | **0.9889** | 99.30% | **98.47%** | 0.70% |
| RoBERTa + SetFit | 0.9843 | **99.43%** | 97.42% | **0.56%** |
| FLAN-T5 | 0.9875 | 99.30% | 98.19% | 0.70% |

All numbers come from `results/tables/`, which the notebooks write. More tables: realistic 13.4% spam rate, McNemar tests, ambiguous examples, robustness, and error analysis.

## Project structure

```
app.py                          Streamlit app (single file)
requirements.txt                Packages for running the app
notebooks/
  traditional_models.ipynb      Data download and curation, Naive Bayes, Logistic Regression
  transformer_models.ipynb      DistilBERT, RoBERTa + SetFit, FLAN-T5, comparison of all five
models/                         Saved models and calibration (thresholds, temperatures)
results/tables/, figures/       Results written by the notebooks
results/error_analysis.md       Categorized errors of DistilBERT
```

## Run the app on your computer

Requirements: Python 3.12 and about 3 GB of free disk space.

1. Clone the repository and open the folder.

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Create a virtual environment and install the packages.

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```

3. Add the transformer weights. They are too large for GitHub (about 1.6 GB). Download the three folders `distilbert`, `roberta_setfit` and `flan_t5` from [Google Drive](https://drive.google.com/drive/folders/1ZaPDtUqgwVaMBn2mMw2JV5uM0hezqQnc?usp=sharing) and put them inside `models/`. Without them the app still runs with the two TF-IDF models.

4. Start the app. It opens at http://localhost:8501.

   ```bash
   .venv\Scripts\python -m streamlit run app.py
   ```

## Train the models again (Google Colab)

1. Upload both notebooks to Google Colab and choose a T4 GPU (`Runtime > Change runtime type`).
2. Run `traditional_models.ipynb` first. It downloads the two datasets, curates them, saves the train/validation/test splits to Google Drive (`MyDrive/SpamLens`), and trains the classical models.
3. Run `transformer_models.ipynb`. After the first `pip install`, click `Runtime > Restart session` and run it again from the top.
4. Copy `models/` and `results/` from Google Drive into this folder.

## Data

- UCI SMS Spam Collection: Almeida & Hidalgo (2011), CC BY 4.0, DOI 10.24432/C5CC84.
- Super SMS Dataset: Salman, Ikram & Kaafar, IEEE Access (2024), DOI 10.1109/ACCESS.2024.3364671. Research use only; this repository does not redistribute it. The notebook downloads it from the authors.

## Team

Raynald Arvan Lim, Hans Ewaldo Kristiawan, Christian Verrell, Dorian Joel Robert Melenotte, Samuel Tristan Gani, Thania Calista.
