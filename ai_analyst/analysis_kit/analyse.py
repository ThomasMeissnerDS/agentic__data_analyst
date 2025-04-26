import pandas as pd
from typing import Optional
from ai_analyst.classes import _DummyClient
from ai_analyst.utils.llm_utils import chat_with_tools
from ai_analyst.analysis_kit.config import AnalysisConfig

# Default client and model
client = _DummyClient()
model_id = "gemma-3-local"

def analyse_data(
    data: pd.DataFrame,
    config: Optional[AnalysisConfig] = None,
) -> pd.DataFrame:
    """Analyse data using the specified model and configuration.
    
    Args:
        data (pd.DataFrame): Input data to analyze
        config (Optional[AnalysisConfig]): Configuration for the analysis.
            If None, uses default configuration.
        
    Returns:
        pd.DataFrame: Analyzed data with insights
    """
    if config is None:
        config = AnalysisConfig()
    
    # Make df available in global scope
    global df
    df = data
    
    # Create the complex prompt with the configuration
    complex_prompt = f"""
    You are an Analyst LLM. You have these Python tool functions to assist you:
    1. correlation(column1_name, column2_name)
    2. groupby_aggregate(groupby_column, agg_column, agg_func)
    3. groupby_aggregate_multi(groupby_cols, agg_dict)
    3. filter_data(column_name, operator, value)
    4. boxplot_all_columns()
    5. correlation_matrix()
    6. scatter_matrix_all_numeric()
    7. line_plot_over_time(date_col, value_col, agg_func='mean', freq='D')
    8. outlier_rows(column_name, z_threshold=3.0)

    The dataset is already in a global 'df'. The data is about {config.data_about}.
    You can call any tool by producing a code block with:
    ```tool_code
    <function_call_here>
    """
    
    final_text = chat_with_tools(
        user_message=complex_prompt,
        client=client,
        model_id=config.model_path,
        conversation_log=[],
        final_answer="",
        iterations=0,
        sleep_secs=config.sleep_seconds,
        data_about=config.data_about,
        tmp_dir=config.tmp_dir,
        pdf_path=config.pdf_path,
        config=config
    )
    print("FINAL ANALYST ANSWER ===")
    print(final_text)
    print(f"PDF conversation saved at {config.pdf_path}")
    return client.chats.create(model=config.model_path).send_message(data, config=config)
