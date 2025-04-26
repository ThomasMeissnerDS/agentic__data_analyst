import pandas as pd
import os
from dataclasses import dataclass
from typing import Optional
from ai_analyst.classes import _DummyClient
from ai_analyst.utils.llm_utils import chat_with_tools

@dataclass
class AnalysisConfig:
    """Configuration class for data analysis.
    
    Attributes:
        data_path (str): Path to the input data file
        model_path (str): Path to the model directory
        load_4bit (bool): Whether to load the model in 4-bit precision
        max_tokens_gen (int): Maximum number of tokens to generate
        pdf_path (str): Path to save the analysis report PDF
        sleep_seconds (int): Sleep time between API calls
        data_about (str): Description of the dataset
        max_total_chars (int): Maximum total characters for analysis
        max_item_chars (int): Maximum characters per analysis item
        keep_last_n_items (int): Number of recent items to keep
        tmp_dir (str): Directory for temporary plot files
        target_column (str): Name of the target column for analysis
    """
    data_path: str = "your_path/train.csv"
    model_path: str = "/kaggle/input/gemma-3/transformers/gemma-3-12b-it/1/"
    load_4bit: bool = True
    max_tokens_gen: int = 1024
    pdf_path: str = "/kaggle/working/report.pdf"
    sleep_seconds: int = 0
    data_about: str = ""
    max_total_chars: int = 15_000
    max_item_chars: int = 2_000
    keep_last_n_items: int = 3
    tmp_dir: str = "/kaggle/working/_plots"
    target_column: str = ""

    def __post_init__(self):
        """Create temporary directory if it doesn't exist and validate critical fields."""
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        # Validate critical fields
        if self.data_about == "":
            raise ValueError(
                "data_about cannot be empty. Please provide a description of your dataset. "
                "Example: 'The dataset contains customer transaction data for fraud detection'"
            )
        
        if self.target_column == "":
            raise ValueError(
                "target_column cannot be empty. Please specify the name of your target column. "
                "Example: 'fraud' or 'sales_amount'"
            )
        
        # Additional validation for data_path
        if self.data_path == "your_path/train.csv":
            import warnings
            warnings.warn(
                "data_path is set to the default placeholder value. "
                "Please set it to your actual data path before running the analysis.",
                UserWarning
            )

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
        pdf_path=config.pdf_path
    )
    print("FINAL ANALYST ANSWER ===")
    print(final_text)
    print(f"PDF conversation saved at {config.pdf_path}")
    return client.chats.create(model=config.model_path).send_message(data)
