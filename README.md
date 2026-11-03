
# Bitcoin AML Transaction Monitoring System

An end-to-end **Anti-Money Laundering (AML) Transaction Monitoring and Explainable AI (XAI) Pipeline** built for Bitcoin blockchain transactions. This system combines machine learning classification, post-hoc model interpretability via SHAP, generative AI compliance justifications (powered by Groq and Llama 3.3), and an interactive **Streamlit** dashboard.

---

## 🚀 Key Features

* **Real-World Dataset Ingestion:** Automatically downloads and processes the [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) via `kagglehub`, mapping thousands of Bitcoin transactions with anonymized network and local features.
* **Time-Aware Machine Learning:** Trains a `RandomForestClassifier` with balanced class weights using a strict *time-based train-test split* to prevent data leakage and mimic real-world deployment constraints.
* **Explainable AI (SHAP):** Utilizes `shap.TreeExplainer` to extract feature-level contributions (distinguishing between local transaction dynamics and aggregated neighbor structures) for every flagged transaction.
* **LLM-Powered Compliance Notes:** Integrates with the **Groq API** (`llama-3.3-70b-versatile`) to automatically generate human-readable, audit-ready compliance justifications explaining *why* a transaction was flagged.
* **Interactive Dashboard:** Deploys a clean **Streamlit** web app showcasing risk probabilities, top contributing features, and generated audit narratives.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Core Libraries:** Pandas, NumPy, Scikit-Learn, SHAP
* **GenAI / LLM:** Groq SDK (`llama-3.3-70b-versatile`)
* **Frontend / UI:** Streamlit
* **Tunneling / Hosting:** Ngrok / Localtunnel (for Google Colab deployment)

---

## 📂 Project Structure

```text
├── app.py                 # Streamlit dashboard script
├── flagged_results.json   # Processed transactions with SHAP values and LLM notes
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies (or inline pip installs)

```

---

## ⚙️ Installation & Setup

### 1. Prerequisites

Make sure you have Python installed along with an active **Groq API Key** (obtainable via [Groq Console](https://console.groq.com/)).

### 2. Install Dependencies

```bash
pip install kagglehub pandas scikit-learn shap groq streamlit pyngrok

```

### 3. Set Up API Keys

If running inside **Google Colab**, store your API key in Colab's secrets manager (`userdata`):

* Key name: `GROQ_API_KEY`
* Value: `your_groq_api_key_here`

If running locally, set it as an environment variable:

```bash
export GROQ_API_KEY="your_groq_api_key_here"

```

---

## 🏃‍♂️ How to Run the Pipeline

1. **Run the Data Ingestion & Model Training:**
Execute the data processing and Random Forest training script to generate predictions and calculate SHAP explanations.
2. **Generate Compliance Notes:**
Run the Groq API integration loop to produce auditable text justifications for high-risk transactions, exporting them to `flagged_results.json`.
3. **Launch the Streamlit App:**
Run the Streamlit dashboard locally or tunnel it using `pyngrok`:
```bash
streamlit run app.py

```



---

## 📊 Model Performance

The Random Forest model handles severe class imbalance (licit vs. illicit transactions) using `class_weight="balanced"`, achieving strong results on later-timestep evaluation splits:

* **Overall Accuracy:** ~98%
* **Illicit Class Precision:** ~0.99
* **Illicit Class Recall:** ~0.69

---

## 💡 Example Compliance Note Output

> *"This Bitcoin transaction was flagged as potentially illicit with a high probability of 0.87 due to a combination of local transaction features and an aggregated neighbor-transaction feature. The strongest contributing signals came from `feat_163` (an aggregated neighbor-transaction feature) and `feat_55` (a local transaction feature), which increased the illicit-risk score by 0.0459 and 0.0434, respectively..."*

