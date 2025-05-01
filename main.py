import numpy as np
import pandas as pd
import os

import torch

from ai_analyst.analysis_kit.analyse import analyse_data, _validate_pdf_requirements
from ai_analyst.analysis_kit.config import AnalysisConfig

os.environ['TRANSFORMERS_OFFLINE'] = '1'  # not sure if needed
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


if __name__ == "__main__":
    DATA_PATH = "/home/thomas/Downloads/train.csv"
    df      = pd.read_csv(DATA_PATH)
    TARGET  = "rainfall"

    ai_cfg = AnalysisConfig(
        data_path=DATA_PATH,
        model_path="google/gemma-3-12b-it",  # default
        data_about="This dataset contains information about rainfall",
        target_column=TARGET,
        pdf_path="/home/thomas/Downloads/report.pdf",
        tmp_dir="/home/thomas/Downloads/_plots",
        font_output_dir="/home/thomas/Downloads/_fonts",
        default_font_path="/home/thomas/Downloads/DejaVuSans-Bold.ttf",
        max_iterations=10
    )

    _validate_pdf_requirements(ai_cfg)
    analyse_data(df, ai_cfg)