import os
from dataclasses import dataclass

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