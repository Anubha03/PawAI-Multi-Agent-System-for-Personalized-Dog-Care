"""
llm.py

Loads the LLM (Llama-3.2-3B-Instruct with 4-bit quantisation for Colab)
and exposes a single generate() function used by all agents.

"""

import logging
import torch
from typing import List, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from config import (
    MODEL_NAME, FALLBACK_MODEL_NAME, HF_TOKEN,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K_SAMPLING,
)

log = logging.getLogger(__name__)

_tokenizer = None
_model = None
_loaded_model_name = None


def _get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(model_name: str = MODEL_NAME, use_4bit: bool = True) -> None:
    global _tokenizer, _model, _loaded_model_name

    token = HF_TOKEN or None

    log.info(f"Loading tokenizer: {model_name}")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=token,
        )
    except Exception as e:
        if model_name != FALLBACK_MODEL_NAME:
            log.warning(
                f"Could not load '{model_name}': {e}\n"
                f"Falling back to '{FALLBACK_MODEL_NAME}'"
            )
            load_model(FALLBACK_MODEL_NAME, use_4bit=use_4bit)
            return
        raise

    log.info(f"Loading model: {model_name} (4bit={use_4bit})")
    load_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )
    if use_4bit and torch.cuda.is_available():
        load_kwargs["quantization_config"] = _get_bnb_config()
    else:
        load_kwargs["torch_dtype"] = torch.float16

    try:
        _model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        if model_name != FALLBACK_MODEL_NAME:
            log.warning(
                f"Could not load '{model_name}': {e}\n"
                f"Falling back to '{FALLBACK_MODEL_NAME}'"
            )
            load_model(FALLBACK_MODEL_NAME, use_4bit=use_4bit)
            return
        raise

    _loaded_model_name = model_name
    log.info(f"✅  Model loaded: {_loaded_model_name}")


def get_model_name() -> str:
    return _loaded_model_name or "not loaded"


def generate(
    messages: List[Dict[str, str]],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K_SAMPLING,
) -> str:
    
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=_tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)

    with torch.no_grad():
        outputs = _model.generate(**gen_kwargs)

    response = _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    return response
