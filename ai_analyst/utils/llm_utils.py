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
    outlier_rows,
    scatter_plot
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
    chat = client.chats.create(model=model_id)

    # Create the tool information string that will be included in every message
    tool_info = f"""
    You are an Analyst LLM. You have these Python tool functions to assist you:
    1. correlation("column1_name", "column2_name") - Returns correlation coefficient between two columns
    2. groupby_aggregate("groupby_column", "agg_column", "agg_func") - Groups data and applies aggregation function
    3. groupby_aggregate_multi(["groupby_col1", "groupby_col2"], {{"agg_col": "agg_func"}}) - Groups by multiple columns and applies multiple aggregations
    4. filter_data("column_name", "operator", value) - Returns filtered dataframe based on condition
    5. boxplot_all_columns() - Creates boxplots for all numeric columns
    6. correlation_matrix() - Returns correlation matrix for all numeric columns
    7. scatter_matrix_all_numeric() - Creates scatter plots between all numeric columns
    8. line_plot_over_time("date_col", "value_col", agg_func="mean", freq="D") - Creates time series plot with aggregation
    9. outlier_rows("column_name", z_threshold=3.0) - Returns rows identified as outliers based on z-score
    10. scatter_plot("x_column", "y_column", hue_col="optional_color_column") - Creates a scatter plot between two columns with optional color encoding

    You cannot ask for additional functions. These are the only functions you can use.

    IMPORTANT: All column names must be provided as strings (in quotes). For example:
    - CORRECT: correlation("Rainfall", "Temperature")
    - INCORRECT: correlation(Rainfall, Temperature)

    The dataset is already in a global 'df'. The data is about {config.data_about}.
    Available columns in the dataset: {df.columns.tolist()}
    
    You can call any tool by producing a code block with:
    ```tool_code
    <function_call_here>
    ```
    
    IMPORTANT: After each tool call, you MUST provide a clear interpretation of the results. Here is a list 
    of the functions you can use, and an example explanation for each of them:
    - For correlation_matrix(): Explain the strongest correlations and any interesting patterns
    - For boxplot_all_columns(): Describe the distribution characteristics and any outliers
    - For scatter_matrix_all_numeric(): Point out any notable relationships between variables
    - For line_plot_over_time(): Explain trends, seasonality, or other temporal patterns
    - For outlier_rows(): Explain the outliers and their significance
    - For describe_df(): Explain the summary statistics of the data
    - For groupby_aggregate(): Explain the aggregation results
    - For groupby_aggregate_multi(): Explain the aggregation results
    - For filter_data(): Explain the filtered data
    - For boxplot_all_columns(): Explain the boxplot results
    - For correlation_matrix(): Explain the correlation matrix results
    - For scatter_plot(): Explain the scatter plot results

    ANALYSIS STRATEGY GUIDELINES:
    1. Start with a broad overview using describe_df() to understand the data distribution
    2. Use boxplot_all_columns() to identify potential outliers and distribution characteristics
    3. If you find interesting patterns in the boxplots, investigate specific columns with scatter_plot()
    4. Use correlation_matrix() sparingly - only when you have a specific hypothesis about relationships
    5. When using correlation_matrix(), follow up with specific scatter_plot() calls for the most interesting relationships
    6. Use groupby_aggregate() or groupby_aggregate_multi() to explore categorical relationships
    7. If you have time-series data, use line_plot_over_time() to identify trends
    8. Use outlier_rows() to investigate specific columns that show potential anomalies
    9. Avoid repetitive analysis - if you've already examined a relationship, move on to new insights
    10. Each iteration should focus on a different aspect of the data

    ONLY USE THE FUNCTIONS THAT ARE LISTED ABOVE. Do not write any code that is not in this list.
    
    Your analysis should be data-driven and focus on actionable insights. Each iteration should provide new, non-repetitive insights about the data.
    """

    # Add tool information to the initial message
    initial_message = user_message + tool_info
    response = chat.send_message(initial_message, config=config)
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
                next_msg = f"Tool output:\n{tool_out}\n\n{tool_info}\n\nPlease provide a detailed interpretation of these results. Focus on key insights and patterns."
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
        next_msg = f"Conversation so far (summary):\n{summary}\n\n{tool_info}\n\nContinue with your analysis, making sure to interpret any visualizations or statistical results."
        model_text = chat.send_message(next_msg, config=config).text

    save_conversation_to_pdf(conversation_log, pdf_path, config)
    return strip_string_quotes(final_answer.strip())
