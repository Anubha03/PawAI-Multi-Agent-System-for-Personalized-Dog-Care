"""
orchestrator.py

Router produces a JSON plan:
  {"plan": [{"agent": "health|nutrition|training", "question": "..."}]}

Multi-agent: a single user message may activate 1–3 sub-agents.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional

from config import MAX_NEW_TOKENS
import llm
from rag_pipeline import RAGPipeline
from agents import HealthAgent, NutritionAgent, TrainingAgent, DogProfile

log = logging.getLogger(__name__)

AGENT_DESCRIPTIONS = """
Available agents:
1. health
   Use for: symptoms, illness, injury, medication questions, triage, poisoning,
            vet recommendations, physical condition assessment.
2. nutrition
   Use for: diet, food, meal planning, calorie needs, weight management, allergies,
            safe/unsafe foods, puppy or senior feeding.
3. training
   Use for: behaviour problems, obedience, socialisation, separation anxiety,
            aggression, barking, pulling on leash, tricks, mental enrichment.
"""

def _make_router_prompt(question: str) -> str:
    return f"""
You are an intent router for a Dog Owner AI assistant.

{AGENT_DESCRIPTIONS}

Routing rules:
- Choose the MINIMAL set of agents needed.
- A question may need multiple agents (e.g., "My overweight Labrador pulls on leash" → nutrition + training).
- Always rewrite the "question" field to be specific to that agent's domain.
- Return ONLY valid JSON. No markdown, no explanation.

Schema:
{{
  "plan": [
    {{"agent": "<health|nutrition|training>", "question": "<domain-specific question>"}}
  ]
}}

User question: {question}

JSON:
""".strip()

def route_question(question: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": "Output strictly valid JSON only. No extra text."},
        {"role": "user",   "content": _make_router_prompt(question)},
    ]
    raw = llm.generate(messages, temperature=0.0, max_new_tokens=256)
    cleaned = raw.strip().strip("```json").strip("```").strip()

    if "plan" not in cleaned:
        json_match = re.search(r'\{.*?"plan".*?\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

    try:
        plan = json.loads(cleaned)
        if "plan" not in plan or not isinstance(plan["plan"], list):
            raise ValueError("Invalid plan schema")
    
        valid_agents = {"health", "nutrition", "training"}
        for step in plan["plan"]:
            if step.get("agent") not in valid_agents:
                raise ValueError(f"Unknown agent: {step.get('agent')}")
        return plan
    except Exception:
       
        q = question.lower()
        if any(w in q for w in ["eat", "feed", "diet", "food", "calorie", "weight"]):
            agent = "nutrition"
        elif any(w in q for w in ["train", "bark", "bite", "pull", "behav", "aggress"]):
            agent = "training"
        else:
            agent = "health"
        return {"plan": [{"agent": agent, "question": question}]}


def _extract_json(text: str) -> dict:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found")
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise ValueError("Unbalanced JSON braces")
    return json.loads(text[start:end + 1])

def run_plan(
    plan: Dict[str, Any],
    rag: RAGPipeline,
    dog: Optional[DogProfile] = None,
    show_evidence: bool = True,
) -> List[Dict[str, Any]]:
   
    agents = {
        "health":    HealthAgent(rag),
        "nutrition": NutritionAgent(rag),
        "training":  TrainingAgent(rag),
    }

    trace = []
    for step in plan["plan"]:
        agent_name = step["agent"]
        question   = step.get("question", "")
        agent = agents[agent_name]

        log.info(f"  Running {agent_name} agent for: '{question}'")
        result = agent.run(question, dog=dog, show_evidence=show_evidence)

        trace.append({
            "agent":    agent_name,
            "question": question,
            "result":   result,
        })
    return trace

def synthesize_answer(
    original_question: str,
    trace: List[Dict[str, Any]],
) -> str:
    
    if len(trace) == 1:
        return trace[0]["result"]["answer"]

    sections = []
    for t in trace:
        agent_name = t["agent"].upper()
        answer     = t["result"]["answer"]
        sections.append(f"=== {agent_name} AGENT ===\n{answer}")

    combined = "\n\n".join(sections)

    all_doc_ids = []
    for t in trace:
        for ev in t["result"].get("evidence", []):
            did = ev.get("doc_id")
            if did and did not in all_doc_ids:
                all_doc_ids.append(did)

    citation_rule = ""
    if all_doc_ids:
        sample = ", ".join([f"[{d}]" for d in all_doc_ids[:4]])
        citation_rule = (
            f"Available doc_ids: {sample} (and others).\n"
            "Include at least one citation in your synthesis.\n"
        )

    messages = [
        {"role": "system", "content": (
            "You are a helpful Dog Owner AI assistant synthesising advice from multiple expert agents.\n"
            "Create a coherent, well-structured final response that addresses the user's original question.\n"
            "Do NOT contradict any agent's advice. Prioritise safety recommendations.\n"
            f"{citation_rule}"
            "Be concise. Use clear section headers where appropriate."
        )},
        {"role": "user", "content": (
            f"Original question: {original_question}\n\n"
            f"Expert agent outputs:\n{combined}\n\n"
            "Please synthesize a unified, helpful response:"
        )},
    ]

    return llm.generate(messages, max_new_tokens=1024, temperature=0.1)

def run_agent(
    question: str,
    rag: RAGPipeline,
    dog: Optional[DogProfile] = None,
    show_evidence: bool = True,
) -> Dict[str, Any]:
   
    print(f"Question: {question}")
    if dog:
        print(f"Dog:      {dog.summary()}")
      
    plan = route_question(question)
    print(f"\n📋 Router Plan:")
    print(json.dumps(plan["plan"], indent=2))

    trace = run_plan(plan, rag, dog=dog, show_evidence=show_evidence)

    final_answer = synthesize_answer(question, trace)

    triage = None
    triage_label = None
    for t in trace:
        if t["agent"] == "health":
            triage       = t["result"].get("triage")
            triage_label = t["result"].get("triage_label")

    result = {
        "question":     question,
        "plan":         plan["plan"],
        "trace":        trace,
        "final_answer": final_answer,
    }
    if triage:
        result["triage"]       = triage
        result["triage_label"] = triage_label

    print(f"\n💬 Final Answer:\n{final_answer}")
    if triage_label:
        print(f"\n{triage_label}")

    return result
