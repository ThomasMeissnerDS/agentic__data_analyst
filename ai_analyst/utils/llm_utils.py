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
        "Analyze the following Analyst-LLM output:\n\n"
        f"{latest_text}\n\n"
        "Provide your response in this format:\n"
        "CONTINUE: [yes/no]\n"
        "SUGGESTIONS:\n"
        "- [Specific analysis suggestion 1]\n"
        "- [Specific analysis suggestion 2]\n"
        "...\n\n"
        
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
        10. scatter_plot("x_column", "y_column", hue_col="optional_color_column") - Creates a scatter plot between two columns with optional color encoding \n
        """

        f"(Remember: data already loaded as df about {data_about})"
    )
    decider_reply = decider_chat.send_message(decider_prompt, config=config).text.strip()
    
    # Parse the decider's response
    continue_analysis = False
    suggestions = []
    
    lines = decider_reply.split('\n')
    for line in lines:
        if line.startswith('CONTINUE:'):
            continue_analysis = 'yes' in line.lower()
        elif line.startswith('-'):
            suggestions.append(line[1:].strip())
    
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


def create_meaningful_summary(conversation_log: list, client: _DummyClient, model_id: str, config: AnalysisConfig) -> str:
    """Create a meaningful summary of the conversation using an LLM."""
    summary_chat = client.chats.create(model=model_id)
    
    # Format the conversation for summarization
    formatted_conversation = []
    for kind, content in conversation_log:
        if kind == "LLM":
            formatted_conversation.append(f"Analyst: {content}")
        elif kind == "TOOL_CODE":
            formatted_conversation.append(f"Tool Call: {content}")
        elif kind == "TOOL":
            formatted_conversation.append(f"Tool Output: {content}")
        elif kind == "DECIDER":
            formatted_conversation.append(f"Decision: {content}")
    
    conversation_text = "\n".join(formatted_conversation)
    
    summary_prompt = f"""
    You are a data analysis summarizer. Your task is to create a concise but meaningful summary of the data analysis process so far.
    
    Here is the conversation history:
    {conversation_text}
    
    Please provide a summary that:
    1. Captures the key insights discovered
    2. Lists the main analysis steps taken
    3. Highlights any interesting patterns or relationships found
    4. Notes any important decisions made
    5. Keeps the summary focused on the most significant findings
    
    Format your response as:
    SUMMARY:
    [Your summary here]
    
    KEY FINDINGS:
    - [Key finding 1]
    - [Key finding 2]
    ...
    """
    
    response = summary_chat.send_message(summary_prompt, config=config)
    return response.text.strip()


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
    21. bin_numeric_column("column_name", bins=5) - Creates bins for a numeric column

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
    - For analyze_missing_value_impact(): Explain how different missing value treatments affect the regression results
    - For histogram_plot(): Explain the distribution shape, modality, and any skewness
    - For qq_plot(): Explain the normality of the distribution and any deviations
    - For density_plot(): Explain the probability density and any peaks or modes
    - For anova_test(): Explain the significance of group differences and effect size
    - For chi_square_test(): Explain the relationship between categorical variables
    - For t_test(): Explain the significance of mean differences between groups
    - For seasonal_decomposition(): Explain the trend, seasonality, and residual patterns
    - For autocorrelation_plot(): Explain the temporal dependencies and patterns
    - For create_interaction(): Explain the new interaction feature created
    - For bin_numeric_column(): Explain the binning strategy and distribution

    ANALYSIS STRATEGY GUIDELINES:
    1. Start with functions that give a broad overview of the data
    2. Use boxplot_all_columns() to identify potential outliers and distribution characteristics
    3. If you find interesting patterns in the boxplots, investigate specific columns with scatter_plot()
    4. Use correlation_matrix() sparingly - only when you have a specific hypothesis about relationships
    5. When using correlation_matrix(), follow up with specific scatter_plot() calls for the most interesting relationships
    6. Use groupby_aggregate() or groupby_aggregate_multi() to explore categorical relationships
    7. If you have time-series data, use line_plot_over_time() to identify trends
    8. Use outlier_rows() to investigate specific columns that show potential anomalies
    9. Use analyze_missing_value_impact() to understand how missing values affect relationships
    10. Use histogram_plot(), qq_plot(), and density_plot() to understand the distribution of numeric variables
    11. Use anova_test() and t_test() to compare means across groups
    12. Use chi_square_test() to analyze relationships between categorical variables
    13. For time series data, use seasonal_decomposition() and autocorrelation_plot() to understand patterns
    14. Use create_interaction() to explore potential interaction effects
    15. Use bin_numeric_column() to discretize continuous variables when appropriate
    16. Avoid repetitive analysis - if you've already examined a relationship, move on to new insights
    17. Each iteration should focus on a different aspect of the data

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

        print(f"Conversation log (iter {iterations}):", conversation_log)

        cont, decider_txt, suggestions = decide_if_continue_or_not(
            latest_text=model_text,
            client=client,
            model_id=model_id,
            data_about=data_about,
            df=df,
            conversation_log=conversation_log,
            config=config
        )
        conversation_log.append(("DECIDER", decider_txt))
        if not cont or iterations >= config.max_iterations:
            break
        iterations += 1
        time.sleep(sleep_secs)
        
        # Create meaningful summary every 5 iterations
        if iterations % 5 == 0:
            summary = create_meaningful_summary(conversation_log, client, model_id, config)
            # Keep only the last 3 items and add the summary
            conversation_log = conversation_log[-3:] + [("SUMMARY", summary)]
            
        # Prepare next message with suggestions and context
        context = f"Previous analysis summary:\n{conversation_log[-1][1]}\n\n" if len(conversation_log) > 0 else ""
        next_msg = f"{context}Previous suggestions:\n" + "\n".join([f"- {s}" for s in suggestions]) + "\n\n" + tool_info
        model_text = chat.send_message(next_msg, config=config).text

    save_conversation_to_pdf(conversation_log, pdf_path, config)
    return strip_string_quotes(final_answer.strip())
