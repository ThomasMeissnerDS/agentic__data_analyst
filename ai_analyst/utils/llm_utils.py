from ai_analyst.classes.dummy_client import _DummyClient
from ai_analyst.general_utils.image_utils import _decode_plot_if_any
from ai_analyst.general_utils.pdf_utils import save_conversation_to_pdf
from ai_analyst.general_utils.text_utils import extract_text_and_code, summarize_conversation, strip_string_quotes
from ai_analyst.analysis_kit.config import AnalysisConfig
from contextlib import contextmanager, redirect_stdout
import io
import pandas as pd
import torch
import gc
import time
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration
from ai_analyst.utils.analysis_toolkit import (
    correlation,
    describe_df,
    groupby_aggregate,
    groupby_aggregate_multi,
    filter_data,
    boxplot_all_columns,
    correlation_matrix,
    scatter_matrix_all_numeric,
    line_plot_over_time,
    outlier_rows
)

# Global cache for tool execution results
executed_calls_cache = {}

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


def decide_if_continue_or_not(
        latest_text: str, 
        client: _DummyClient, 
        model_id: str, 
        data_about: str,
        df: pd.DataFrame,
        config: AnalysisConfig = None):
    decider_chat = client.chats.create(model=model_id)
    decider_prompt = (
        "You are the DECIDER LLM. Analyse the following Analyst‑LLM output:\n\n"
        f"{latest_text}\n\n"
        "Should the Analyst continue? Reply \"no\" if we are finished, otherwise say anything else.\n\n"
        f"(Remember: data already loaded as df about {data_about} with columns {list(df.columns)})"
    )
    decider_reply = decider_chat.send_message(decider_prompt, config=config).text.strip()
    return decider_reply.lower() != "no", decider_reply


def run_tool_code(code_str: str, conversation_log: list, tmp_dir: str):
    """Execute tool code and return its output."""
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
        config: AnalysisConfig = None,
        df: pd.DataFrame = None,
        ) -> str:
    conversation_log = []
    chat = client.chats.create(model=model_id)

    # Add instruction for interpretation to the initial prompt
    user_message += """
    
    IMPORTANT: After each tool call, you MUST provide a clear interpretation of the results. For example:
    - For correlation_matrix(): Explain the strongest correlations and any interesting patterns
    - For boxplot_all_columns(): Describe the distribution characteristics and any outliers
    - For scatter_matrix_all_numeric(): Point out any notable relationships between variables
    - For line_plot_over_time(): Explain trends, seasonality, or other temporal patterns
    
    Your analysis should be data-driven and focus on actionable insights.
    """

    response = chat.send_message(user_message, config=config)
    model_text = response.text
    final_answer, iterations = "", 0

    while True:
        pre, code_block, post = extract_text_and_code(model_text)

        if pre:
            conversation_log.append(("LLM", pre)); final_answer += pre + "\n"
        if code_block:
            tool_out = run_tool_code(code_block, conversation_log, tmp_dir)
            # Add a reminder for interpretation if the tool output is a visualization
            if "base64" in tool_out:
                next_msg = f"Tool output:\n{tool_out}\n\nPlease provide a detailed interpretation of these results. Focus on key insights and patterns."
                model_text = chat.send_message(next_msg, config=config).text
                continue
        if post:
            conversation_log.append(("LLM", post)); final_answer += post + "\n"

        cont, decider_txt = decide_if_continue_or_not(
            latest_text=model_text,
            client=client,
            model_id=model_id,
            data_about=data_about,
            df=df,
            config=config
        )
        conversation_log.append(("DECIDER", decider_txt))
        if not cont or iterations >= config.max_iterations:
            break
        iterations += 1
        time.sleep(sleep_secs)

        _, summary = summarize_conversation(conversation_log)
        next_msg = f"Conversation so far (summary):\n{summary}\n\nContinue with your analysis, making sure to interpret any visualizations or statistical results."
        model_text = chat.send_message(next_msg, config=config).text

    save_conversation_to_pdf(conversation_log, pdf_path)
    return strip_string_quotes(final_answer.strip())
