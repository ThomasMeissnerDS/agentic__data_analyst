import os
import shutil
from ai_analyst.analysis_kit.config import AnalysisConfig


def copy_files(
        config: AnalysisConfig,
        src: str = None,
        tgt_dir: str = None,
        tgt_name: str = "DejaVuSans-Bold.ttf",
        ) -> str:
    """
    Copy a file from the input directory to the working directory.
    
    Args:
        config: AnalysisConfig instance containing font paths
        src: Source file path. If None, uses config.font_path
        tgt_dir: Target directory path. If None, uses config.font_output_dir
        tgt_name: Target file name
        
    Returns:
        str: Path to the copied file in the working directory
    """
    # Use config values if not explicitly provided
    src = src or config.font_path
    tgt_dir = tgt_dir or config.font_output_dir
    
    # If no font path is specified, return None
    if not src:
        return None
        
    # Create target directory if it doesn't exist
    os.makedirs(tgt_dir, exist_ok=True)
    
    # Construct the full target path
    target_path = os.path.join(tgt_dir, tgt_name)
    
    # Copy the file if it exists
    if os.path.exists(src):
        shutil.copyfile(src, target_path)
        return target_path
    else:
        raise FileNotFoundError(f"Source font file not found: {src}")
