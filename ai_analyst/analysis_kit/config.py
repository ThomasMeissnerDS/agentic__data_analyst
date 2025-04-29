import os
from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    """Configuration class for data analysis.
    
    Attributes:
        data_path (str): Path to the input data file
        model_path (str): Path to the model directory (for local models)
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
        max_iterations (int): Maximum number of iterations for the analysis
        font_path (str): Path to the font file for PDF generation. If None, will try to use system default.
        font_output_dir (str): Directory where font files will be copied for PDF generation
        font_family (str): Font family name to use in PDF generation
        default_font_path (str): Default font path to use if no custom font is specified
        use_api (bool): Whether to use API-based model instead of local model
        api_key (str): API key for the model service. If None, will try to get from environment or Kaggle secrets
        api_model_id (str): Model ID to use with the API
        api_secret_label (str): Label for the API key in Kaggle secrets
    """
    data_path: str = "your_path/train.csv"
    model_path: str = "/google/gemma-3-12b-it/"
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
    max_iterations: int = 5
    font_path: str = None
    font_output_dir: str = "/kaggle/working/_fonts"
    font_family: str = "DejaVu"
    default_font_path: str = "/kaggle/input/dejavusans-bold-ttf/DejaVuSans-Bold.ttf"
    use_api: bool = False
    api_key: str = None
    api_model_id: str = "gemma-3-27b-it"
    api_secret_label: str = "GEMINI_API_KEY"

    def __post_init__(self):
        """Create temporary directory if it doesn't exist and validate critical fields."""
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.font_output_dir, exist_ok=True)
        
        # If no font path is specified, use the default
        if self.font_path is None:
            self.font_path = self.default_font_path
        
        # If using API and no API key is provided, try to get it from environment or Kaggle secrets
        if self.use_api and self.api_key is None:
            try:
                from kaggle_secrets import UserSecretsClient
                self.api_key = os.getenv(self.api_secret_label, 
                                       UserSecretsClient().get_secret(self.api_secret_label))
            except ImportError:
                self.api_key = os.getenv(self.api_secret_label)
            
            if not self.api_key:
                raise ValueError(
                    f"API key not found. Please provide it either through the config, "
                    f"environment variable {self.api_secret_label}, or Kaggle secrets."
                )
        
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