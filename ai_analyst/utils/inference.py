from ai_analyst.analysis_kit.config import AnalysisConfig
from contextlib import contextmanager
import torch
import gc
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

@contextmanager
def gemma3_session(config: AnalysisConfig):
    """
    Yield (model, tokenizer) and guarantee GPU memory is released afterwards.
    This way we keep the memory consumption as low as possible and avoid
    running out of memory faster than necessary.
    
    Args:
        config (AnalysisConfig): Configuration object containing model settings
    """
    bnb_cfg = None
    if config.load_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    accelerator = Accelerator()
    tokenizer   = AutoTokenizer.from_pretrained(config.model_path)
    _           = AutoProcessor.from_pretrained(config.model_path, use_fast=True)  # not used but required
    model       = Gemma3ForConditionalGeneration.from_pretrained(
        config.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_cfg,
    )
    model = accelerator.prepare(model)

    try:
        yield model, tokenizer
    finally:
        try:
            model.cpu()
        except Exception:
            pass
        del model, tokenizer, accelerator
        torch.cuda.empty_cache()
        gc.collect()

def analyst_inference(messages, *, config: AnalysisConfig) -> str:
    """One assistant turn given `messages` (Open‑AI style list of dicts).
    
    Args:
        messages: List of message dictionaries
        config (AnalysisConfig): Configuration object containing model settings
    """
    with gemma3_session(config) as (model, tokenizer):
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).to(model.device)

        prompt_len = inputs.shape[1]
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=config.max_tokens_gen,
                eos_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        return reply.strip() 