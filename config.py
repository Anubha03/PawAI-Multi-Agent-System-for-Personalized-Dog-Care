"""
config.py – Central configuration for the project.
"""

import os

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
FALLBACK_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

HF_TOKEN = os.getenv("HF_TOKEN", "")   

MAX_NEW_TOKENS = 1024
TEMPERATURE    = 0.0   
TOP_P          = 0.95
TOP_K_SAMPLING = 50

EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K      = 4     
CHUNK_SIZE     = 600
CHUNK_OVERLAP  = 80

CORPUS_DIR     = os.getenv("CORPUS_DIR", "/content")
HEALTH_CORPUS  = os.path.join(CORPUS_DIR, "health_corpus.jsonl")
NUTRITION_CORPUS = os.path.join(CORPUS_DIR, "nutrition_corpus.jsonl")
TRAINING_CORPUS  = os.path.join(CORPUS_DIR, "training_corpus.jsonl")

INDEX_DIR = os.getenv("INDEX_DIR", "./faiss_indexes")

NCBI_EMAIL      = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY    = os.getenv("NCBI_API_KEY", "")  
PUBMED_MAX_IDS = 20

PUBMED_FETCH_BATCH_SIZE = 20   

PUBMED_SLEEP = 0.15 if os.getenv("NCBI_API_KEY") else 1.0

PUBMED_HEALTH_QUERIES = [
    "canine symptoms diagnosis treatment veterinary",
    "dog disease triage emergency veterinary clinical signs",
    "canine vomiting diarrhea lethargy symptoms management",
    "dog skin allergy dermatitis treatment",
    "canine musculoskeletal lameness orthopedic",
    "dog poisoning toxicology emergency treatment",
    "canine respiratory disease cough dyspnea",
    "dog urinary tract infection renal disease",
]

PUBMED_NUTRITION_QUERIES = [
    "dog nutrition dietary requirements feeding guidelines",
    "canine obesity weight management diet",
    "dog breed specific nutrition large small breed",
    "puppy senior dog nutritional needs life stage",
    "canine food allergy elimination diet",
    "AAFCO dog food nutritional adequacy",
    "raw diet commercial dog food comparison canine",
]

PUBMED_TRAINING_QUERIES = [
    "dog behavior positive reinforcement training",
    "canine aggression fear anxiety behavior modification",
    "puppy socialization training early learning",
    "dog separation anxiety treatment behavioral",
    "canine obedience training clicker operant conditioning",
    "dog cognitive enrichment mental stimulation",
    "problem behavior barking destructive chewing dog",
]

NUTRITION_URLS = [
    "https://www.akc.org/expert-advice/nutrition/dog-nutrition-tips/",
    "https://www.akc.org/expert-advice/nutrition/what-can-dogs-eat/",
    "https://www.fda.gov/animal-veterinary/animal-food-feeds/pet-food",
    "https://www.aspca.org/pet-care/dog-care/dog-nutrition-tips",
]

TRAINING_URLS = [
    "https://www.akc.org/expert-advice/training/",
    "https://www.aspca.org/pet-care/dog-care/common-dog-behavior-issues",
]

HEALTH_URLS = [
    "https://www.akc.org/expert-advice/health/",
    "https://www.aspca.org/pet-care/animal-poison-control/toxic-and-non-toxic-plants",
]

TRIAGE_LEVELS = {
    "HOME_CARE":  "✅ Home Care – manageable at home with monitoring",
    "MONITOR":    "👁️  Monitor – watch closely; see vet if no improvement in 24–48 h",
    "VET_VISIT":  "⚠️  Vet Visit Recommended – schedule an appointment",
    "EMERGENCY":  "🚨 Emergency – seek veterinary care IMMEDIATELY",
}
