# PawAI: Multi-Agent System for Personalized Dog Care

An intelligent multi-agent system for personalised dog care advice covering health triage, nutrition planning, and behaviour training. Built on a quantised Llama-3.2-3B-Instruct model with FAISS-based RAG, served through a Flask backend and Streamlit chat UI.

---

## Architecture

```
User Question
     │
  Router (LLM JSON plan)
     │
     ├── HealthAgent     -> symptom analysis, 4-level triage classification
     ├── NutritionAgent  -> calorie estimation, meal planning (Python-computed)
     └── TrainingAgent   -> positive reinforcement behaviour plans
     │
  Synthesizer -> Final Answer
     │
  Streamlit UI (via Flask backend + ngrok tunnel)
```

**Key components:**

| File                     | Role                                       |
| ------------------------ | ------------------------------------------ |
| `config.py`              | Paths, model name, RAG settings            |
| `data_ingestion.py`      | Fetches and chunks veterinary corpora      |
| `rag_pipeline.py`        | FAISS index build and retrieval            |
| `llm.py`                 | Llama model loading and generation         |
| `agents.py`              | HealthAgent, NutritionAgent, TrainingAgent |
| `orchestrator.py`        | Router, run_agent(), synthesizer           |
| `app.py`                 | Flask backend (POST /chat, GET /health)    |
| `streamlit_app_ngrok.py` | Streamlit chat UI                          |

---

## Prerequisites

- Google Colab with a **T4 GPU** runtime (Runtime -> Change runtime type -> T4 GPU)
- A free [ngrok account](https://dashboard.ngrok.com/get-started/your-authtoken) for the public UI URL
- A [Hugging Face account](https://huggingface.co/settings/tokens) with access to `meta-llama/Llama-3.2-3B-Instruct`
- A [NCBI account](https://www.ncbi.nlm.nih.gov/account/) for the PubMed E-utilities API

---

## Required Colab Secrets

All credentials are stored as **Colab Secrets** (🔑 icon in the left sidebar -> "Add new secret").

| Secret name        | Where to get it                                                                                          | Used for                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `HF_TOKEN`         | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)                                 | Downloading Llama-3.2-3B-Instruct                         |
| `NCBI_API_KEY`     | [ncbi.nlm.nih.gov/account/settings](https://www.ncbi.nlm.nih.gov/account/settings/)                      | PubMed corpus ingestion (10 req/s vs 3 req/s without key) |
| `NCBI_EMAIL`       | Your email address                                                                                       | Required by NCBI E-utilities terms of service             |
| `NGROK_AUTH_TOKEN` | [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken) | Public tunnel to the Streamlit UI                         |

> **Note:** All four secrets must be added before running any cell. The notebook will raise a clear error if a required secret is missing.

---

## Installation

### Step 1 : Open the notebook in Colab

Upload `main.ipynb` to Google Drive and open it, or use File -> Upload notebook directly in Colab.

### Step 2 : Set runtime to T4 GPU

```
Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU -> Save
```

### Step 3 : Add Colab Secrets

Click the 🔑 icon in the left sidebar and add all four secrets listed in the table above. Enable "Notebook access" for each one.

### Step 4 : Upload project files

Run **Section 1** (Environment Setup & Configuration) and upload all of the following files when prompted:

```
config.py
data_ingestion.py
rag_pipeline.py
llm.py
agents.py
orchestrator.py
app.py
streamlit_app_ngrok.py
```

### Step 4b : Upload corpus files

In the same upload dialog, upload the following corpus files from the `corpus/` folder:

```
corpus/health_corpus.jsonl
corpus/nutrition_corpus.jsonl
corpus/training_corpus.jsonl
```

These files will be uploaded to `/content/corpus/` in Colab (the path is configured in environment variables).

### Step 5 : Install dependencies

Run **Section 1** (Environment Setup & Configuration). This installs:

```
huggingface-hub >= 1.3.0
transformers
torch
sentence-transformers
faiss-cpu
biopython
langchain-core
flask
flask-cors
streamlit
pyngrok
```

---

## Running the System

Run the cells in **Section 10** (Deployment - Flask & Streamlit Setup) in this exact order every Colab session:

| Step                      | What it does                            | Wait time |
| ------------------------- | --------------------------------------- | --------- |
| Install Dependencies      | Install Streamlit, pyngrok, Flask       | ~30s      |
| Configure ngrok Tunnel    | Set ngrok auth token from Colab Secrets | instant   |
| Launch Flask Backend      | Start Flask backend + load LLM          | **~60s**  |
| Verify Flask Health       | Confirm Flask is up                     | instant   |
| Launch Streamlit Frontend | Open ngrok tunnel -> start Streamlit    | ~10s      |

After the final step completes, you will see:

```
✅ Your Dog AI Agent is LIVE at:
   👉 https://xxxx-xx-xx-xxx-xx.ngrok-free.app
```

Open that URL in any browser. The UI will be accessible from any device on any network for the duration of the Colab session.

```python
# View logs
!cat /content/flask.log
!cat /content/streamlit.log
```

---

## Example Usage

### Health Query : Triage

**Dog profile:** Bella, 9yr female Labrador Retriever, arthritis, chicken allergy

**Question:** `My dog ate a whole bar of chocolate. What do I do?`

**Response:**

- 🚨 **Emergency** badge
- Symptoms listed: vomiting, diarrhea, increased heart rate, seizures in severe cases
- ASPCA Poison Control number surfaced

---

### Nutrition Query : Calorie Estimation

**Question:** `Suggest a meal plan for my overweight dog.`

**Response:**

- RER and MER computed in Python with full factor breakdown:
  - Activity × age × neuter × health × breed-specific adjustments
- Three-meal daily schedule with protein sources, portion sizes, and calorie counts
- Allergen-safe ingredients throughout

---

### Training Query : Behaviour Plan

**Question:** `My dog barks excessively when left alone. What should I do?`

**Response:**

- 🎓 **Training** badge
- Three-phase plan: safe space establishment -> departure cue desensitisation -> counter-conditioning
- Positive reinforcement only : no punishment techniques
- Health profile integrated (arthritis, low activity noted in assessment)

---

## Triage Levels

The HealthAgent classifies every health query into one of four levels using a three-layer system:

| Level       | Badge | Meaning                         | Examples                                      |
| ----------- | ----- | ------------------------------- | --------------------------------------------- |
| `HOME_CARE` | 🏠    | Minor issue, manageable at home | Small scrape, brief hiccups                   |
| `MONITOR`   | 👁    | Watch closely 24–48h            | Single vomiting episode, mild lethargy        |
| `VET_VISIT` | 🏥    | Schedule vet appointment        | Limping for 3+ days, recurring ear scratching |
| `EMERGENCY` | 🚨    | Seek care immediately           | Toxin ingestion, collapse, pale gums, seizure |

---

## Project Structure

```
main.ipynb   <- main notebook
config.py                        <- configuration
data_ingestion.py                <- corpus fetching (PubMed, AKC, ASPCA, AAFCO)
rag_pipeline.py                  <- FAISS index build + retrieval
llm.py                           <- model loading + generation
agents.py                        <- HealthAgent, NutritionAgent, TrainingAgent
orchestrator.py                  <- router + run_agent() + synthesizer
app.py                           <- Flask backend
streamlit_app_ngrok.py           <- Streamlit UI
```

---

## Notes

- **First run only:** Data ingestion (Section 1) fetches ~500 documents from PubMed and builds FAISS indexes. This takes 3–5 minutes and only needs to run once per session if using Google Drive persistence.
- **Model load time:** The LLM downloads ~2GB on the very first run. Subsequent runs in the same session reuse the cached weights.
- **Session limits:** Free Colab sessions disconnect after ~12 hours of inactivity. The ngrok tunnel and Flask process will need to be restarted after reconnecting.
- **T4 GPU memory:** The model runs in 4-bit NF4 quantisation and uses ~4GB VRAM. Do not run other GPU-intensive cells while the Flask backend is active.
