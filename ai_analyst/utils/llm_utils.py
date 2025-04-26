from ai_analyst.classes import _DummyClient
from ai_analyst.general_utils.image_utils import _decode_plot_if_any
from ai_analyst.general_utils.pdf_utils import save_conversation_to_pdf
from ai_analyst.general_utils.text_utils import extract_text_and_code, summarize_conversation, strip_string_quotes
from ai_analyst.analysis_kit.analyse import AnalysisConfig
from contextlib import contextmanager, redirect_stdout
import io
import pandas as pd
import torch
import gc
import time
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
    


def decide_if_continue_or_not(
        latest_text: str, 
        client: _DummyClient, 
        model_id: str, 
        data_about: str):
    decider_chat = client.chats.create(model=model_id)
    decider_prompt = (
        "You are the DECIDER LLM. Analyse the following Analyst‑LLM output:\n\n"
        f"{latest_text}\n\n"
        "Should the Analyst continue? Reply \"no\" if we are finished, otherwise say anything else.\n\n"
        f"(Remember: data already loaded as df about {data_about} with columns {list(df.columns)})"
    )
    decider_reply = decider_chat.send_message(decider_prompt).text.strip()
    return decider_reply.lower() != "no", decider_reply


def run_tool_code(code_str: str, conversation_log: list, tmp_dir: str):
    global executed_calls_cache
    conversation_log.append(("TOOL_CODE", f"```tool_code\n{code_str}\n```"))

    if code_str in executed_calls_cache:
        cached = executed_calls_cache[code_str]
        if not _decode_plot_if_any(cached, conversation_log, tmp_dir):
            conversation_log.append(("TOOL", f"[Cache Hit] {cached.strip()}"))
        return f"```tool_output\n[Cache Hit]\n{cached.strip()}\n```"

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            val = eval(code_str)
        except Exception as e:
            err = f"Error: {e}"
            conversation_log.append(("TOOL", err))
            executed_calls_cache[code_str] = err
            return f"```tool_output\n{err}\n```"

    out_txt = buf.getvalue()
    if val is not None:
        out_txt += ("\n" if out_txt else "") + (str(val) if not isinstance(val, str) else val)

    if not _decode_plot_if_any(out_txt, conversation_log, tmp_dir):
        conversation_log.append(("TOOL", out_txt.strip() or "(no output)"))
    executed_calls_cache[code_str] = out_txt
    return f"```tool_output\n{out_txt.strip()}\n```"


def chat_with_tools(
        user_message: str,
        client: _DummyClient,
        model_id: str,
        conversation_log: list,
        final_answer: str,
        iterations: int,
        sleep_secs: int = 0,
        data_about: str = "",
        tmp_dir: str = "/kaggle/working/_plots",
        pdf_path: str = "/kaggle/working/report.pdf",
        ) -> str:
    conversation_log = []
    chat = client.chats.create(model=model_id)

    response = chat.send_message(user_message)
    model_text = response.text
    final_answer, iterations = "", 0

    while True:
        pre, code_block, post = extract_text_and_code(model_text)

        if pre:
            conversation_log.append(("LLM", pre)); final_answer += pre + "\n"
        if code_block:
            tool_out = run_tool_code(code_block, conversation_log, tmp_dir)
        if post:
            conversation_log.append(("LLM", post)); final_answer += post + "\n"

        cont, decider_txt = decide_if_continue_or_not(
            latest_text=model_text,
            client=client,
            model_id=model_id,
            data_about=data_about
        )
        conversation_log.append(("DECIDER", decider_txt))
        if not cont or iterations >= 5:
            break
        iterations += 1
        time.sleep(sleep_secs)

        _, summary = summarize_conversation(conversation_log)
        next_msg = f"Conversation so far (summary):\n{summary}\n\nContinue."
        model_text = chat.send_message(next_msg).text

    save_conversation_to_pdf(conversation_log, pdf_path)
    return strip_string_quotes(final_answer.strip())
