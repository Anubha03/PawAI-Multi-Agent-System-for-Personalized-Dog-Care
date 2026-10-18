"""
Flask backend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

import warnings
warnings.filterwarnings('ignore')

import config
from rag_pipeline import RAGPipeline
from agents import DogProfile
from orchestrator import run_agent
import llm

app = Flask(__name__)
CORS(app)  

rag = RAGPipeline()
rag.build_indexes(force_rebuild=False)

llm.load_model(use_4bit=True)

def build_dog_profile(data: dict) -> DogProfile:
    return DogProfile(
        name           = data.get("name", "Dog"),
        breed          = data.get("breed", "Mixed Breed"),
        age_years      = float(data.get("age", 3)),
        weight_kg      = float(data.get("weight", 20)),
        sex            = data.get("sex", "male"),
        neutered       = bool(data.get("neutered", True)),
        allergies      = [a.strip() for a in data.get("allergies", "").split(",") if a.strip()],
        health_issues  = [h.strip() for h in data.get("healthIssues", "").split(",") if h.strip()],
        activity_level = data.get("activityLevel", "moderate"),
    )

@app.route("/chat", methods=["POST"])
def chat():
    """
    Request body (JSON):
    {
      "question": "My dog vomited twice. Should I be worried?",
      "dog": {
        "name": "Buddy", "breed": "Labrador Retriever",
        "age": "2", "weight": "30", "sex": "male",
        "neutered": true, "allergies": "chicken",
        "healthIssues": "", "activityLevel": "high"
      }
    }

    Response (JSON):
    {
      "answer": "...",
      "agents": ["health"],
      "triage": "MONITOR"   // only present for health queries
    }
    """
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    dog_data = body.get("dog", {})
    dog = build_dog_profile(dog_data)

    try:
        result = run_agent(
            question      = question,
            rag           = rag,
            dog           = dog,
            show_evidence = False,
        )

        agents_used = [step["agent"] for step in result.get("plan", [])]

        return jsonify({
            "answer":  result["final_answer"],
            "agents":  agents_used,
            "triage":  result.get("triage"),         
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model": llm.get_model_name()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
