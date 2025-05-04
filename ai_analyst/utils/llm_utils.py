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
import logging
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
import os

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
    # Set memory management settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        # Enable memory efficient attention
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
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
        attn_implementation="flash_attention_2",  # Use flash attention for better memory efficiency
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


def decide_if_continue_or_not(
        latest_text: str, 
        client: _DummyClient, 
        model_id: str, 
        data_about: str,
        df: pd.DataFrame,
        conversation_log: list = None,
        config: AnalysisConfig = None):
    decider_chat = client.chats.create(model=model_id)
    
    # Extract tool calls from conversation log
    tool_calls = []
    if conversation_log:
        for kind, content in conversation_log:
            if kind == "TOOL_CODE":
                tool_calls.append(content)
    
    decider_prompt = (
        "You are the DECIDER LLM. Your role is to guide the analysis process by:\n"
        "1. Evaluating the current analysis progress\n"
        "2. Suggesting specific areas for further investigation\n"
        "3. Providing clear directions for the next analysis steps\n\n"
        "Previous tool calls made in this analysis:\n"
    )
    
    if tool_calls:
        decider_prompt += "\n".join([f"- {call}" for call in tool_calls]) + "\n\n"
    else:
        decider_prompt += "No previous tool calls have been made.\n\n"
    
    decider_prompt += (
        """
        The analyst has these Python tool functions available:
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
        11. analyze_missing_value_impact("column_name", "target_column") - Analyzes the impact of missing values in a column on regression with target variable
        12. histogram_plot("column_name", bins=30) - Creates a histogram with KDE for a numeric column
        13. qq_plot("column_name") - Creates a Q-Q plot to assess normality of a numeric column
        14. density_plot("column_name") - Creates a density plot for a numeric column
        15. anova_test("group_column", "value_column") - Performs ANOVA test to compare means across groups
        16. chi_square_test("categorical_col1", "categorical_col2") - Performs chi-square test of independence between categorical variables
        17. t_test("numeric_col1", "numeric_col2") - Performs independent t-test between two numeric columns
        18. seasonal_decomposition("date_col", "value_col", freq="D") - Performs seasonal decomposition of time series data
        19. autocorrelation_plot("column_name", lags=30) - Creates an autocorrelation plot for time series data
        20. create_interaction("numeric_col1", "numeric_col2") - Creates an interaction term between two numeric columns
        21. bin_numeric_column("column_name", bins=5) - Creates bins for a numeric column \n
        """

        f"(Remember: data already loaded as df about {data_about})\n"

        "Analyze the following Analyst-LLM output:\n\n"
        f"{latest_text}\n\n"
        "Provide your response in this format:\n"
        "CONTINUE: [yes/no]\n"
        "SUGGESTIONS (add a maximum of 3):\n"
        "- [Specific analysis suggestion 1]\n"
        "- [Specific analysis suggestion 2]\n"
        "...\n\n"
        "Important: Keep your answer concise as we have limited GPU memory to process your answer!\n"
        "The CONTINUE part is always mandatory though. As we aim for a deep analysis, do not stop the analysis too early.\n"
        "Make sure that CONTINUE has to be at the start of the line. Otherwise the parsing will lead to wrong results.\n"
        "If in doubt return CONTINUE: yes\n"
    )
    decider_reply = decider_chat.send_message(decider_prompt, config=config).text.strip()
    
    # Parse the decider's response
    continue_analysis = True  # Default to True to avoid early stops
    suggestions = []
    
    lines = decider_reply.split('\n')
    found_continue = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # More flexible CONTINUE parsing
        if line.lower().startswith('continue:'):
            found_continue = True
            # Extract the value after CONTINUE:
            value = line[8:].strip().lower()
            # Only set to False if explicitly "no"
            continue_analysis = value != 'no'
            print(f"Decider CONTINUE parsing: line='{line}', value='{value}', continue_analysis={continue_analysis}")
        elif line.startswith('-'):
            suggestions.append(line[1:].strip())
    
    # If no CONTINUE line was found, log a warning but keep continue_analysis as True
    if not found_continue:
        print(f"Warning: No CONTINUE line found in decider response. Defaulting to continue_analysis=True")
        print(f"Full decider response:\n{decider_reply}")
    
    return continue_analysis, decider_reply, suggestions


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


def create_meaningful_summary(
        current_step: tuple,  # (kind, content) of current step
        previous_summary: str,  # Previous summary to build upon
        client: _DummyClient, 
        model_id: str, 
        config: AnalysisConfig
    ) -> str:
    """Create or update a meaningful summary of the analysis using an LLM."""
    summary_chat = client.chats.create(model=model_id)
    
    # Format the current step for summarization
    kind, content = current_step
    if kind == "LLM":
        current_step_text = f"Analyst: {content}"
    elif kind == "TOOL_CODE":
        current_step_text = f"Tool Call: {content}"
    elif kind == "TOOL":
        current_step_text = f"Tool Output: {content}"
    elif kind == "DECIDER":
        current_step_text = f"Decision: {content}"
    else:
        current_step_text = f"{kind}: {content}"
    
    summary_prompt = f"""
    You are a data analysis summarizer. Your task is to update the analysis summary with the latest step.
    
    Here is the current summary of the analysis so far:
    {previous_summary}
    
    Here is the latest step in the analysis:
    {current_step_text}
    
    Please update the summary to incorporate this new information. Your updated summary should:
    1. Maintain the key insights discovered so far
    2. Add any new insights from the latest step
    3. Update the analysis progress
    4. Keep the summary focused on the most significant findings
    5. Be concise and under 100 words total
    
    Format your response as:
    SUMMARY:
    [Your updated summary here - must be under 100 words]
    
    KEY FINDINGS:
    - [Key finding 1]
    - [Key finding 2]

    Important: Keep your answer concise as we have limited GPU memory to process your answer!\n
    ...
    """
    
    response = summary_chat.send_message(summary_prompt, config=config)
    summary_text = response.text.strip()
    
    # Extract the summary part
    summary_lines = summary_text.split('\n')
    summary = ""
    for line in summary_lines:
        if line.startswith('SUMMARY:'):
            summary = line[8:].strip()
            break
    
    # If no summary found, use the first line
    if not summary:
        summary = summary_lines[0]
    
    # Truncate if over 100 words
    words = summary.split()
    if len(words) > 100:
        truncated_summary = ' '.join(words[:100])
        print(f"Warning: Summary truncated from {len(words)} to 100 words")
        return truncated_summary
    
    return summary


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
    conversation_log = []  # Full conversation log for PDF generation
    current_summary = ""   # Growing summary of the analysis
    chat = client.chats.create(model=model_id)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create a minimal tool info string that will be included in every message
    tool_info = f"""
    You are an Analyst LLM. You have access to data analysis tools.
    The dataset is already in a global 'df'. The data is about {data_about}.
    Available columns in the dataset: {df.columns.tolist()}
    
    You can call any tool by producing a code block with:
    ```tool_code
    <function_call_here>
    ```
    
    After each tool call, you MUST provide a clear interpretation of the results.
    Keep your analysis focused and avoid repetitive analysis.
    """

    # Add tool information to the initial message
    initial_message = user_message + tool_info
    response = chat.send_message(initial_message, config=config)
    model_text = response.text
    current_summary = ""
    final_answer, iterations = "", 0

    while True:
        # Log memory usage and input length before each iteration
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"Memory usage before iteration {iterations}:")
            logger.info(f"Allocated: {memory_allocated:.2f} GB")
            logger.info(f"Reserved: {memory_reserved:.2f} GB")
        
        pre, code_block, post = extract_text_and_code(model_text)
        
        # Log input length
        input_length = len(model_text)
        logger.info(f"Input length for iteration {iterations}: {input_length} tokens")

        if pre:
            conversation_log.append(("LLM", pre))
            final_answer += pre + "\n"
            # Update summary with the new content, keeping it concise
            current_summary = create_meaningful_summary(("LLM", pre), current_summary, client, model_id, config)
            logger.info(f"Pre-text current_summary (iter {iterations}): {current_summary[:100]}...")
            
        if code_block:
            tool_out = run_tool_code(code_block, conversation_log, tmp_dir)
            # Add a reminder for interpretation if the tool output is a visualization
            if "base64" in tool_out:
                next_msg = f"Tool output:\n{tool_out}\n\n{tool_info}\n\nPlease provide a detailed interpretation of these results. Focus on key insights and patterns."
                model_text = chat.send_message(next_msg, config=config).text
                continue
            # Update summary with the tool call and output, keeping it concise
            current_summary = create_meaningful_summary(("TOOL", model_text), current_summary, client, model_id, config)
            logger.info(f"Tool call current_summary (iter {iterations}): {current_summary[:100]}...")
            
        if post:
            conversation_log.append(("LLM", post))
            final_answer += post + "\n"
            # Update summary with the new content, keeping it concise
            current_summary = create_meaningful_summary(("LLM", post), current_summary, client, model_id, config)
            logger.info(f"Post-text current_summary (iter {iterations}): {current_summary[:100]}...")

        logger.info(f"Conversation log length (iter {iterations}): {len(conversation_log)}")

        cont, decider_txt, suggestions = decide_if_continue_or_not(
            latest_text=model_text,
            client=client,
            model_id=model_id,
            data_about=data_about,
            df=df,
            conversation_log=[("LLM", current_summary)],  # Only pass the current summary
            config=config
        )
        conversation_log.append(("DECIDER", decider_txt))
        # Update summary with the decision, keeping it concise
        current_summary = create_meaningful_summary(("DECIDER", decider_txt + "\n".join([f"- {s}" for s in suggestions])), current_summary, client, model_id, config)
        
        if not cont or iterations >= config.max_iterations:
            logger.info(f"Max iterations reached or decision to stop analysis: decision: {cont}, iteration: {iterations}")
            break
        iterations += 1
        time.sleep(sleep_secs)
        
        # Prepare next message with minimal context
        context = f"Analysis Summary (last {min(3, len(conversation_log))} steps):\n{current_summary}\n\n"
        next_msg = f"{context}Previous suggestions:\n" + "\n\n" + tool_info
        
        # Clear memory before next iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        model_text = chat.send_message(next_msg, config=config).text

    save_conversation_to_pdf(conversation_log, pdf_path, config)
    return strip_string_quotes(final_answer.strip())
