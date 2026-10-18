"""
Agents:
  HealthAgent     – symptom analysis, triage, cautionary diagnosis
  NutritionAgent  – meal planning, dietary recommendations
  TrainingAgent   – customized training plans

"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from langchain_core.documents import Document

from config import TRIAGE_LEVELS, RAG_TOP_K
import llm
from rag_pipeline import RAGPipeline

log = logging.getLogger(__name__)

@dataclass
class DogProfile:
    name:          str   = "Dog"
    breed:         str   = "Mixed breed"
    age_years:     float = 3.0
    weight_kg:     float = 15.0
    sex:           str   = "male"    
    neutered:      bool  = True
    allergies:     List[str] = field(default_factory=list)
    health_issues: List[str] = field(default_factory=list)
    activity_level: str  = "moderate"  
    body_condition_score: Optional[int] = None  
    def summary(self) -> str:
        al  = ", ".join(self.allergies)     if self.allergies     else "none known"
        hi  = ", ".join(self.health_issues) if self.health_issues else "none"
        bcs = f", BCS {self.body_condition_score}/9" if self.body_condition_score else ""
        return (
            f"{self.name} is a {'neutered ' if self.neutered else ''}{self.sex} "
            f"{self.breed}, {self.age_years} years old, {self.weight_kg} kg{bcs}. "
            f"Known allergies: {al}. Pre-existing conditions: {hi}. "
            f"Activity level: {self.activity_level}."
        )

class BaseAgent:
    DOMAIN: str = "general"
    SYSTEM_PROMPT: str = "You are a helpful dog care assistant."

    def __init__(self, rag: RAGPipeline):
        self.rag = rag

    def _retrieve(self, query: str, k: int = RAG_TOP_K) -> List[Document]:
        return self.rag.retrieve(query, domain=self.DOMAIN, k=k)

    def _build_rag_messages(
        self,
        question: str,
        context_docs: List[Document],
        dog: Optional[DogProfile] = None,
        extra_instructions: str = "",
    ) -> List[Dict[str, str]]:
        ctx_blocks = []
        for d in context_docs:
            doc_id = d.metadata.get("doc_id", "unknown")
            ctx_blocks.append(f"[{doc_id}]\n{d.page_content.strip()}")
        context = "\n\n".join(ctx_blocks)

        allergy_block = ""
        if dog and dog.allergies:
            allergen_list = ", ".join(dog.allergies)
            allergy_block = (
                f"\n** ALLERGY WARNING – HARD RULE **\n"
                f"{dog.name} is allergic to: {allergen_list}.\n"
                f"You MUST NOT recommend any food, treat, or ingredient containing "
                f"these allergens under ANY circumstances, even if the retrieved context "
                f"suggests them as a general recommendation. "
                f"If a standard recommendation conflicts with the allergy, you MUST "
                f"explicitly state a safe alternative instead (e.g. plain boiled turkey "
                f"or white fish instead of chicken).\n"
                f"* END ALLERGY WARNING *\n"
            )

        dog_info = f"\nDog profile: {dog.summary()}\n{allergy_block}" if dog else ""
        disclaimer_instruction = (
            "\nDISCLAIMER REQUIREMENT: You MUST end every response with the following "
            "disclaimer block, on its own line, exactly as written:\n"
            "---\n"
            "DISCLAIMER: I am an AI assistant, NOT a licensed veterinarian. "
            "The information above is for general guidance only and does not constitute "
            "professional veterinary advice, diagnosis, or treatment. "
            "Please consult a qualified, licensed veterinarian before making any "
            "decisions about your dog's health, diet, or care.\n"
            "---\n"
        )

        system = (
            f"{self.SYSTEM_PROMPT}\n"
            "Answer the user's question using ONLY the provided veterinary context.\n"
            "If the context is insufficient, say so clearly.\n"
            "You MUST cite sources using doc_id in square brackets, e.g., [health_a1b2c3d4].\n"
            "Never cite sources not in the context.\n"
            "\n"
            "CRITICAL GROUNDING RULES:\n"
            "1. The DOG PROFILE above describes THIS specific dog. "
               "Only attribute conditions, history, or traits to the dog "
               "if they are explicitly listed in the dog profile.\n"
            "2. The CONTEXT below contains veterinary reference material. "
               "If the context mentions a disease (e.g. hypothyroidism) as a "
               "possible cause, do NOT state the dog HAS that disease — "
               "say it IS A POSSIBLE CAUSE to investigate.\n"
            "3. Never invent medical history, diagnoses, or conditions "
               "not present in the dog profile.\n"
            f"{disclaimer_instruction}"
            f"{extra_instructions}"
        )

        user = (
            f"{dog_info}"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    DISCLAIMER_TEXT = (
        "\n---\n"
        "DISCLAIMER: I am an AI assistant, NOT a licensed veterinarian. "
        "The information above is for general guidance only and does not constitute "
        "professional veterinary advice, diagnosis, or treatment. "
        "Please consult a qualified, licensed veterinarian before making any "
        "decisions about your dog's health, diet, or care.\n"
        "---"
    )

    def _enforce_disclaimer(self, answer: str) -> str:
        if "NOT a licensed veterinarian" not in answer:
            answer = answer.rstrip() + self.DISCLAIMER_TEXT
        return answer

    def _check_allergens(self, answer: str, dog: Optional[DogProfile]) -> str:
        if not dog or not dog.allergies:
            return answer
        flagged = []
        answer_lower = answer.lower()
        for allergen in dog.allergies:
            if allergen.lower() in answer_lower:
                flagged.append(allergen)
        if flagged:
            warning = (
                f"⚠️ ALLERGY ALERT: This response mentions "
                f"{', '.join(flagged)}, which {dog.name} is allergic to. "
                f"Please disregard any suggestion involving these ingredients "
                f"and consult your vet for allergen-safe alternatives.\n\n"
            )
            answer = warning + answer
        return answer

    def run(
        self,
        question: str,
        dog: Optional[DogProfile] = None,
        show_evidence: bool = True,
        k: int = RAG_TOP_K,
    ) -> Dict[str, Any]:
        docs = self._retrieve(question, k=k)

        if show_evidence:
            self.rag.show_evidence(question, docs)

        messages = self._build_rag_messages(question, docs, dog=dog)
        answer   = llm.generate(messages, temperature=0.0)

        answer = self._check_allergens(answer, dog)
        answer = self._enforce_disclaimer(answer)

        return {
            "agent":    self.DOMAIN,
            "question": question,
            "answer":   answer,
            "evidence": self.rag.extract_evidence_list(docs),
        }
  
class HealthAgent(BaseAgent):
    DOMAIN = "health"
    SYSTEM_PROMPT = (
        "You are a knowledgeable veterinary assistant helping dog owners understand "
        "their dog's health. You analyse symptoms and provide cautionary guidance. "
        "⚠️ IMPORTANT: You are NOT a licensed veterinarian. Always advise the owner "
        "to consult a licensed veterinarian for diagnosis and treatment. "
        "Do not prescribe medications or dosages."
    )

    TRIAGE_INSTRUCTION = (
        "\nAfter your explanation, you MUST conclude with a TRIAGE LEVEL on its own line:\n"
        "TRIAGE: <one of: HOME_CARE | MONITOR | VET_VISIT | EMERGENCY>\n\n"
        "Definitions:\n"
        "  HOME_CARE  – minor issue, fully explained cause, no distress\n"
        "  MONITOR    – single isolated episode, dog otherwise normal, watch 24–48h\n"
        "  VET_VISIT  – symptom present >24h, recurring, infection, limping, not eating\n"
        "  EMERGENCY  – poisoning, collapse, pale/white/blue gums, seizure, "
        "breathing difficulty, uncontrolled bleeding\n"
    )

    _EMERGENCY_TRIGGERS = [
        "xylitol", "chocolate", "grapes", "raisins", "onion", "garlic",
        "ibuprofen", "acetaminophen", "tylenol", "advil", "antifreeze",
        "rat poison", "rodenticide", "bleach", "pesticide", "mothball",
        "collapsed", "collapse", "unconscious", "not breathing", "stopped breathing",
        "pale gums", "white gums", "blue gums", "grey gums",
        "seizure", "convuls", "fitting",
        "bloat", "distended abdomen", "swollen belly",
        "can't stand", "cannot stand", "unable to stand", "dragging legs",
        "paralys", "uncontrolled bleeding", "spurting blood",
    ]
    _VET_VISIT_TRIGGERS = [
        "limping for", "lame for", "for 3 days", "for two days", "for a week",
        "keeps scratching", "scratching his ears", "scratching her ears",
        "shaking his head", "shaking her head",
        "discharge from", "not eating for", "hasn't eaten",
        "lump", "mass on", "growth on", "bump on",
        "for several days", "recurring",
    ]

    _TRIAGE_CLASSIFIER_PROMPT = (
        "You are a veterinary triage classifier. "
        "Output EXACTLY one word — nothing else: HOME_CARE, MONITOR, VET_VISIT, or EMERGENCY.\n\n"
        "Rules:\n"
        "  EMERGENCY  – poisoning/toxin, collapse, pale/white/blue gums, seizure, "
        "breathing difficulty, uncontrolled bleeding\n"
        "  VET_VISIT  – symptom present >24h, infection, limping, ear/eye problem, "
        "recurring issue, not eating >24h, lump/mass\n"
        "  MONITOR    – single isolated episode, dog otherwise normal and alert\n"
        "  HOME_CARE  – very minor, fully explained cause, no distress\n\n"
        "Output one word only."
    )

    def run(
        self,
        question: str,
        dog: Optional[DogProfile] = None,
        show_evidence: bool = True,
        k: int = RAG_TOP_K,
    ) -> Dict[str, Any]:
        docs = self._retrieve(question, k=k)

        if show_evidence:
            self.rag.show_evidence(question, docs)

        forced_triage = self._prescreen_triage(question)

        messages = self._build_rag_messages(
            question, docs, dog=dog,
            extra_instructions=self.TRIAGE_INSTRUCTION,
        )
        answer = llm.generate(messages, temperature=0.0, max_new_tokens=1024)

        answer = self._check_allergens(answer, dog)
        answer = self._enforce_disclaimer(answer)

        triage = forced_triage or self._classify_triage(question, answer)

        return {
            "agent":        self.DOMAIN,
            "question":     question,
            "answer":       answer,
            "triage":       triage,
            "triage_label": TRIAGE_LEVELS.get(triage, triage),
            "evidence":     self.rag.extract_evidence_list(docs),
        }

    @staticmethod
    def _prescreen_triage(question: str) -> Optional[str]:
        q = question.lower()
        if any(t in q for t in HealthAgent._EMERGENCY_TRIGGERS):
            return "EMERGENCY"
        if any(t in q for t in HealthAgent._VET_VISIT_TRIGGERS):
            return "VET_VISIT"
        return None

    def _classify_triage(self, question: str, answer: str) -> str:
        messages = [
            {"role": "system", "content": self._TRIAGE_CLASSIFIER_PROMPT},
            {"role": "user",   "content": (
                f"Situation: {question}\n\n"
                f"Vet response summary: {answer[:500]}"
            )},
        ]
        result = llm.generate(messages, temperature=0.0, max_new_tokens=8).strip().upper()
        result = result.split()[0] if result.split() else ""
        if result in ("HOME_CARE", "MONITOR", "VET_VISIT", "EMERGENCY"):
            return result
        return HealthAgent._parse_triage(answer)

    @staticmethod
    def _parse_triage(text: str) -> str:
        match = re.search(
            r'TRIAGE:\s*(HOME_CARE|MONITOR|VET_VISIT|EMERGENCY)',
            text, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        t = text.lower()

        if any(w in t for w in [
            "immediate", "emergency", "life-threatening", "poison", "toxic",
            "collapsed", "pale gums", "seizure", "unconscious",
            "call poison control", "not breathing",
        ]):
            return "EMERGENCY"

        if any(w in t for w in [
            "schedule", "appointment", "days", "persists", "worsening",
            "limping", "ear infection", "scratching ears", "has been",
            "for 3 days", "persistent", "recurring", "several days",
        ]):
            return "VET_VISIT"

        if any(w in t for w in ["vet", "veterinarian", "clinic"]):
            return "VET_VISIT"

        if any(w in t for w in ["monitor", "watch", "24 hours", "48 hours", "bland diet"]):
            return "MONITOR"

        return "MONITOR"  

_BREED_SIZE_MAP: Dict[str, str] = {
    # Toy (< 5 kg)
    "chihuahua": "toy", "yorkshire terrier": "toy", "pomeranian": "toy",
    "maltese": "toy", "toy poodle": "toy", "papillon": "toy",
    # Small (5–10 kg)
    "miniature poodle": "small", "shih tzu": "small", "pug": "small",
    "french bulldog": "small", "cavalier king charles spaniel": "small",
    "cocker spaniel": "small", "beagle": "small", "dachshund": "small",
    "jack russell terrier": "small", "west highland white terrier": "small",
    # Medium (10–25 kg)
    "border collie": "medium", "australian shepherd": "medium",
    "standard poodle": "medium", "brittany": "medium",
    "bull terrier": "medium", "basenji": "medium",
    "whippet": "medium", "vizsla": "medium",
    # Large (25–45 kg)
    "labrador retriever": "large", "golden retriever": "large",
    "german shepherd": "large", "german shepherd mix": "large",
    "doberman pinscher": "large", "boxer": "large",
    "siberian husky": "large", "weimaraner": "large",
    "dalmatian": "large", "flat-coated retriever": "large",
    "belgian malinois": "large",
    # Giant (> 45 kg)
    "great dane": "giant", "saint bernard": "giant",
    "newfoundland": "giant", "mastiff": "giant",
    "bernese mountain dog": "giant", "irish wolfhound": "giant",
    "great pyrenees": "giant", "leonberger": "giant",
    "rottweiler": "giant",
}

_SENIOR_THRESHOLDS: Dict[str, int] = {
    "toy": 10, "small": 9, "medium": 8, "large": 7, "giant": 5
}


def _classify_size(weight_kg: float, breed: str) -> str:
    """Return size class from weight, falling back to breed lookup."""
    if weight_kg < 5:  return "toy"
    if weight_kg < 10: return "small"
    if weight_kg < 25: return "medium"
    if weight_kg < 45: return "large"
    return _BREED_SIZE_MAP.get(breed.lower().strip(), "giant")


def _ideal_weight_kg(actual_kg: float, bcs: Optional[int]) -> float:
    if bcs is None or bcs == 5:
        return actual_kg
    deviation = (bcs - 5) * 0.10
    lean_mass = actual_kg * (1 - max(deviation, 0))
    return lean_mass if bcs > 5 else actual_kg


def _health_modifier(health_issues: List[str]) -> Tuple[float, List[str]]:

    CONDITION_DELTA: Dict[str, Tuple[float, str]] = {
        "obesity":        (-0.20, "Obesity: reduce MER ~20%; target 1–2% weight loss/week."),
        "overweight":     (-0.15, "Overweight: reduce MER ~15%."),
        "underweight":    (+0.20, "Underweight: increase MER ~20% until target weight reached."),
        "diabetes":       (+0.00, "Diabetes: calorie target unchanged; prioritise low-GI diet. Vet supervision required."),
        "hypothyroidism": (-0.15, "Hypothyroidism: reduce MER ~15%; monitor weight monthly."),
        "hyperthyroidism":(+0.20, "Hyperthyroidism: increase MER ~20%; monitor weight closely."),
        "cardiac":        (-0.10, "Cardiac disease: mild calorie restriction; reduce sodium. Vet-guided."),
        "renal":          (-0.10, "Renal disease: moderate restriction; phosphorus-controlled diet. Vet-guided."),
        "liver":          (-0.10, "Liver disease: moderate restriction; protein quality critical. Vet-guided."),
        "cancer":         (+0.10, "Cancer: increased metabolic demand; high-quality protein essential."),
        "pancreatitis":   (-0.10, "Pancreatitis: low-fat diet; may reduce caloric density."),
        "arthritis":      (-0.05, "Arthritis: mild reduction to support weight management."),
        "anxiety":        (+0.00, "Anxiety: calorie needs unchanged; behaviour intervention primary."),
        "epilepsy":       (+0.00, "Epilepsy: calorie needs unchanged; avoid high-fat ketogenic unless vet-prescribed."),
    }
    total_delta = 0.0
    notes: List[str] = []
    for issue in health_issues:
        key = issue.lower().strip()
        for pattern, (delta, note) in CONDITION_DELTA.items():
            if pattern in key:
                total_delta += delta
                notes.append(note)
                break
    total_delta = max(-0.40, min(0.40, total_delta))
    return 1.0 + total_delta, notes


def _breed_modifier(breed: str) -> Tuple[float, Optional[str]]:
  
    b = breed.lower()
    obesity_prone = ["labrador", "beagle", "cocker spaniel", "pug",
                     "basset hound", "dachshund", "cairn terrier"]
    sighthounds   = ["greyhound", "whippet", "saluki", "borzoi", "afghan"]
    for pattern in obesity_prone:
        if pattern in b:
            return 0.90, f"{breed} is obesity-prone; applying 10% calorie reduction."
    for pattern in sighthounds:
        if pattern in b:
            return 1.05, f"{breed} has higher lean-mass ratio; 5% calorie increase."
    return 1.0, None

class NutritionAgent(BaseAgent):
    DOMAIN = "nutrition"
    SYSTEM_PROMPT = (
    "You are a canine nutrition specialist assistant. "
    "You provide science-based dietary advice based on the dog's breed, age, weight, "
    "and health status following AAFCO guidelines and peer-reviewed research. "
    ""
    "⚠️ Always note that specific medical dietary needs should be confirmed with a veterinarian. "
    "Do not recommend homemade diets without professional oversight for dogs with health conditions."
)

    NUTRITION_INSTRUCTION = (
        "\nSection '1. DAILY CALORIE ESTIMATE' is already written and will be prepended "
        "to your response — DO NOT write it yourself.\n\n"
        "Write ONLY the following four sections. Each has required content listed below:\n\n"

        "2. MEAL PLAN\n"
        "   • State the number of meals per day and the kcal target per meal "
        "(use the pre-computed values provided).\n"
        "   • Give a concrete daily schedule, e.g.:\n"
        "       Morning meal (7am): [food type + approximate portion]\n"
        "       Evening meal (6pm): [food type + approximate portion]\n"
        "   • Specify the protein source(s) — account for any known allergies.\n"
        "   • Specify the carbohydrate source(s) — account for any known allergies.\n"
        "   • Specify a suitable commercial food type (e.g. 'large-breed adult kibble, "
        "salmon-based') or home-cooked structure if appropriate.\n"
        "   • Include portion size guidance (e.g. cups or grams per meal) based on the "
        "kcal/meal value and a typical kibble density of ~350 kcal/cup.\n\n"

        "3. KEY NUTRIENTS\n"
        "   • List the 4-6 most important nutrients for THIS dog's specific profile "
        "(breed, age, activity level, health issues).\n"
        "   • For each nutrient: name it, state the AAFCO minimum or recommended range, "
        "and explain why it matters for this dog.\n"
        "   • Call out any nutrients especially relevant to the breed "
        "(e.g. joint support for large breeds, omega-3 for coat health).\n\n"

        "4. FOODS TO AVOID\n"
        "   • List ALL known allergens from the dog's profile first.\n"
        "   • List universally toxic foods (xylitol, chocolate, grapes, onions, garlic, "
        "macadamia nuts, alcohol, raw yeast dough).\n"
        "   • List any breed- or condition-specific foods to avoid based on the dog's profile.\n\n"

        "5. RECOMMENDATIONS\n"
        "   • Recommend a specific commercial food category (e.g. 'large-breed adult, "
        "salmon & sweet potato kibble') with the nutritional profile to look for on the label.\n"
        "   • Mention healthy treat options that respect the dog's allergies, "
        "with a treat calorie budget (typically max 10% of MER).\n"
        "   • State how often to reassess portions (e.g. monthly BCS check).\n"
        "   • Include the vet consultation disclaimer.\n\n"

        "RULES (apply to all sections):\n"
        "• DO NOT perform any arithmetic — all calorie numbers are pre-computed above. "
        "Reference them by name (e.g. 'the MER of X kcal/day') but never recalculate.\n"
        "• DO NOT multiply MER by meals/day — MER is already the daily total.\n"
        "• DO NOT convert protein % to kcal or grams — percentages only.\n"
        "• Context may include research papers with study design language "
        "('crossover arrangement', 'consecutive periods', 'DMI') — "
        "these are experimental protocols, NOT feeding instructions. Ignore them for meal planning.\n"
        "• DO NOT cite sources not in the provided context.\n"
        "• If the context does not cover a specific point, use your general veterinary "
        "nutrition knowledge and note it is general guidance rather than leaving it blank.\n"
        "Cite sources using [doc_id] format.\n"
    )

    @staticmethod
    def _format_calorie_block(calorie_info: Dict) -> str:
        if not calorie_info:
            return ""

        rer   = calorie_info.get("RER_kcal_per_day",  "N/A")
        mer   = calorie_info.get("MER_kcal_per_day",  "N/A")
        freq  = calorie_info.get("feeding_frequency",  "N/A")
        meal  = calorie_info.get("MER_per_meal_kcal",  "N/A")
        size  = calorie_info.get("size_class",         "N/A")
        bd    = calorie_info.get("factor_breakdown",   {})
        notes = calorie_info.get("notes",              [])
        warns = calorie_info.get("warnings",           [])

        lines = [
            "1. DAILY CALORIE ESTIMATE",
            f"   Resting Energy Requirement (RER) : {rer} kcal/day",
            f"   Maintenance Energy Requirement (MER): {mer} kcal/day  ← total daily target",
            f"   Feeding frequency : {freq}",
            f"   Calories per meal : {meal} kcal  (= MER ÷ number of meals)",
            f"   Size class        : {size}",
        ]
        if bd:
            lines.append(
                f"   Factors applied   : activity {bd.get('activity_factor','?')} × "
                f"age {bd.get('age_factor','?')} × "
                f"neuter {bd.get('neuter_factor','?')} × "
                f"health {bd.get('health_factor','?')} × "
                f"breed {bd.get('breed_factor','?')} "
                f"= composite {bd.get('composite_factor','?')}"
            )
        for n in notes:
            lines.append(f"   • {n}")
        for w in warns:
            lines.append(f"   ⚠️  {w}")
        lines.append("")   

        lines += [
            " CALORIE REFERENCE (do not reprint this block; use values by name if needed) ",
            f"RER={rer} kcal/day | MER={mer} kcal/day | "
            f"per_meal={meal} kcal | freq={freq}",
            "DO NOT multiply MER by meals — it is already the daily total.",
            "",
        ]
        return "\n".join(lines)

    def run(
        self,
        question: str,
        dog: Optional[DogProfile] = None,
        show_evidence: bool = True,
        k: int = RAG_TOP_K,
    ) -> Dict[str, Any]:
        retrieval_query = question
        if dog:
            retrieval_query = (
                f"{question} {dog.breed} {dog.age_years} year old "
                f"{dog.weight_kg}kg dog nutrition"
            )

        docs = self._retrieve(retrieval_query, k=k)

        if show_evidence:
            self.rag.show_evidence(question, docs)
          
        calorie_info = {}
        calorie_block = ""
        if dog:
            calorie_info  = self._estimate_calories(dog)
            calorie_block = self._format_calorie_block(calorie_info)

        messages = self._build_rag_messages(
            question, docs, dog=dog,
            extra_instructions=calorie_block + "\n" + self.NUTRITION_INSTRUCTION,
        )
        llm_answer = llm.generate(messages, temperature=0.0, max_new_tokens=1024)

        section1 = "\n".join(calorie_block.split("\n")[:calorie_block.split("\n").index("")]) \
            if calorie_info else ""
        answer = (section1 + "\n\n" + llm_answer).strip() if section1 else llm_answer

        return {
            "agent":        self.DOMAIN,
            "question":     question,
            "answer":       answer,
            "calorie_info": calorie_info,
            "evidence":     self.rag.extract_evidence_list(docs),
        }

    @staticmethod
    def _estimate_calories(dog: DogProfile) -> Dict:
        
        cal_warnings: List[str] = []
        notes:        List[str] = []

        VALID_ACTIVITY = ("low", "moderate", "high", "working", "sport")
        ACTIVITY_FACTORS = {
            "low":      1.2,   
            "moderate": 1.6,   
            "high":     2.0,   
            "working":  3.0,   
            "sport":    4.0,   
        }

        weight_kg = getattr(dog, "weight_kg", None)
        if weight_kg is None:
            raise ValueError("weight_kg is required but was not provided.")
        try:
            weight_kg = float(weight_kg)
        except (TypeError, ValueError):
            raise ValueError(f"weight_kg must be a number, got {weight_kg!r}")
        if weight_kg <= 0:
            raise ValueError(f"weight_kg must be positive, got {weight_kg}")

        age_years = getattr(dog, "age_years", None)
        age_assumed = False
        if age_years is None:
            age_years = 3.0
            age_assumed = True
            cal_warnings.append(
                "age_years not provided; assuming 3 years (adult). "
                "Puppy/senior adjustments will NOT be applied."
            )
        else:
            try:
                age_years = float(age_years)
            except (TypeError, ValueError):
                age_years = 3.0
                age_assumed = True
                cal_warnings.append(
                    f"age_years value {dog.age_years!r} is not a valid number; "
                    "assuming 3 years (adult)."
                )
            if age_years < 0:
                age_years = 3.0
                age_assumed = True
                cal_warnings.append(
                    f"age_years is negative ({dog.age_years}); assuming 3 years (adult)."
                )

        raw_activity = getattr(dog, "activity_level", None)
        if raw_activity is None or str(raw_activity).strip() == "":
            activity_level = "moderate"
            cal_warnings.append(
                "activity_level not provided; defaulting to 'moderate' (1.6×RER). "
                "Provide 'low', 'moderate', 'high', 'working', or 'sport' for accuracy."
            )
        else:
            activity_level = str(raw_activity).strip().lower()
            if activity_level not in VALID_ACTIVITY:
                activity_level = "moderate"
                cal_warnings.append(
                    f"activity_level '{raw_activity}' is not recognised "
                    f"(valid: {', '.join(VALID_ACTIVITY)}); defaulting to 'moderate'."
                )
        activity_factor = ACTIVITY_FACTORS[activity_level]
      
        neutered = getattr(dog, "neutered", None)
        if neutered is None:
            neutered = False
            cal_warnings.append(
                "neutered status not provided; assuming intact. "
                "If neutered, actual MER is ~15% lower."
            )

        breed = getattr(dog, "breed", None)
        if not breed or str(breed).strip() == "":
            breed = "unknown"
            cal_warnings.append(
                "breed not provided; breed-specific metabolic adjustment skipped."
            )

        health_issues = getattr(dog, "health_issues", None) or []

        bcs = getattr(dog, "body_condition_score", None)
        if bcs is not None:
            try:
                bcs = int(bcs)
                if not (1 <= bcs <= 9):
                    cal_warnings.append(
                        f"BCS {bcs} is outside the valid 1–9 range; "
                        "ignoring BCS weight correction."
                    )
                    bcs = None
            except (TypeError, ValueError):
                cal_warnings.append(
                    f"BCS value {dog.body_condition_score!r} is not a valid integer; "
                    "ignoring BCS weight correction."
                )
                bcs = None

        ideal_kg = _ideal_weight_kg(weight_kg, bcs)
        if ideal_kg != weight_kg:
            notes.append(
                f"BCS {bcs}/9: using estimated ideal weight {ideal_kg:.1f} kg "
                f"(actual {weight_kg:.1f} kg) for RER calculation."
            )

        rer = 70.0 * (ideal_kg ** 0.75)

        size = _classify_size(weight_kg, breed)

        age_factor = 1.0
        if not age_assumed:
            if age_years < 1.0:
                max_puppy = 2.5 if size == "giant" else 3.0
                age_factor = max_puppy - (max_puppy - 2.0) * age_years
                notes.append(
                    f"Puppy ({age_years:.1f} yr): age factor {age_factor:.2f}. "
                    "Feed 3–4× daily; transition to adult food near 12 months "
                    "(small/medium) or 18–24 months (large/giant)."
                )
            else:
                senior_age = _SENIOR_THRESHOLDS.get(size, 7)
                if age_years >= senior_age:
                    reduction = min(0.05 * (age_years - senior_age), 0.25)
                    age_factor = 1.0 - reduction
                    notes.append(
                        f"Senior ({age_years:.0f} yr, threshold {senior_age} yr "
                        f"for {size} breed): age factor {age_factor:.2f}. "
                        "Monitor BCS monthly; some seniors need MORE calories due "
                        "to poor nutrient absorption."
                    )

        neuter_factor = 1.0
        if neutered and age_years >= 1.0:
            neuter_factor = 0.85
            notes.append("Neutered: MER reduced by 15% (Laflamme 1997).")
        elif neutered and age_years < 1.0:
            neuter_factor = 0.95
            notes.append("Neutered puppy: minor 5% reduction; growth needs dominate.")

        health_factor, health_notes = _health_modifier(health_issues)
        notes.extend(health_notes)

        breed_factor, breed_note = _breed_modifier(breed)
        if breed_note:
            notes.append(breed_note)

        composite_factor = (
            activity_factor * age_factor * neuter_factor
            * health_factor * breed_factor
        )
        composite_factor = max(0.8, min(6.0, composite_factor))
        mer = rer * composite_factor

        if age_years < 0.33:
            feeding_freq = "4× per day"
        elif age_years < 0.5:
            feeding_freq = "3–4× per day"
        elif age_years < 1.0:
            feeding_freq = "3× per day"
        elif age_years >= _SENIOR_THRESHOLDS.get(size, 7):
            feeding_freq = "2–3× per day (seniors benefit from smaller, more frequent meals)"
        else:
            feeding_freq = "2× per day"

        meals = int(feeding_freq[0])

        return {
            "dog_name":          getattr(dog, "name", "Unknown"),
            "actual_weight_kg":  weight_kg,
            "ideal_weight_kg":   round(ideal_kg, 1),
            "size_class":        size,
            "RER_kcal_per_day":  round(rer, 1),
            "MER_kcal_per_day":  round(mer, 1),
            "MER_per_meal_kcal": round(mer / meals, 1),
            "feeding_frequency": feeding_freq,
            "assumed_fields":    (
                ["age_years (→ 3.0 yr)"] if age_assumed else []
            ) + (
                ["activity_level (→ moderate)"]
                if activity_level == "moderate" and
                   (not raw_activity or raw_activity not in VALID_ACTIVITY) else []
            ) + (
                ["neutered (→ intact)"]
                if getattr(dog, "neutered", None) is None else []
            ),
            "factor_breakdown": {
                "activity_factor":  round(activity_factor,  3),
                "age_factor":       round(age_factor,        3),
                "neuter_factor":    round(neuter_factor,     3),
                "health_factor":    round(health_factor,     3),
                "breed_factor":     round(breed_factor,      3),
                "composite_factor": round(composite_factor,  3),
            },
            "notes":    notes,
            "warnings": cal_warnings,
            "note": (                         
                "⚠️  Evidence-based estimates only. Calorie needs vary ±20% between "
                "individuals. Adjust based on BCS (target 4–5/9). Consult your "
                "veterinarian for dogs with health conditions, pregnant/lactating "
                "females, or puppies."
            ),
        }


class TrainingAgent(BaseAgent):
    DOMAIN = "training"
    SYSTEM_PROMPT = (
        "You are a certified-dog-trainer assistant with expertise in positive reinforcement "
        "and evidence-based behaviour science (Applied Animal Behaviour Science, AVSAB). "
        "You create personalised, humane training plans. "
        "⚠️ For severe aggression or anxiety, always recommend consulting a certified "
        "applied animal behaviourist (CAAB) or veterinary behaviourist (DACVB)."
    )

    TRAINING_INSTRUCTION = (
        "\nStructure your response as:\n"
        "1. ASSESSMENT – brief assessment of the dog's situation\n"
        "2. TRAINING PLAN – week-by-week or phase-by-phase steps\n"
        "3. TECHNIQUE – specific methods and cues to use\n"
        "4. COMMON MISTAKES – what to avoid\n"
        "5. PROGRESS MILESTONES – how to know it's working\n"
        "Base your plan on positive reinforcement principles only. "
        "Cite your sources using [doc_id] format.\n"
    )

    def run(
        self,
        question: str,
        dog: Optional[DogProfile] = None,
        show_evidence: bool = True,
        k: int = RAG_TOP_K,
    ) -> Dict[str, Any]:
        retrieval_query = question
        if dog:
            retrieval_query = f"{question} {dog.breed} {dog.age_years} year old dog training"

        docs = self._retrieve(retrieval_query, k=k)

        if show_evidence:
            self.rag.show_evidence(question, docs)

        messages = self._build_rag_messages(
            question, docs, dog=dog,
            extra_instructions=self.TRAINING_INSTRUCTION,
        )
        answer = llm.generate(messages, temperature=0.2, max_new_tokens=1024)

        answer = self._check_allergens(answer, dog)
        answer = self._enforce_disclaimer(answer)

        return {
            "agent":    self.DOMAIN,
            "question": question,
            "answer":   answer,
            "evidence": self.rag.extract_evidence_list(docs),
        }
