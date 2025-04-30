import pandas as pd
from typing import Optional
from ai_analyst.classes import _DummyClient
from ai_analyst.utils.llm_utils import chat_with_tools
from ai_analyst.analysis_kit.config import AnalysisConfig
from ai_analyst.analysis_kit import (
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

def _validate_pdf_requirements(config: AnalysisConfig) -> None:
    """Validate all requirements for PDF generation.
    
    Args:
        config (AnalysisConfig): Configuration object containing paths and settings
        
    Raises:
        ImportError: If fpdf package is not installed
        FileNotFoundError: If font file is not found and no default is available
        PermissionError: If required directories are not writable
    """
    # Check if fpdf is installed
    try:
        import fpdf
    except ImportError:
        raise ImportError(
            "The 'fpdf' package is required for PDF generation. "
            "Please install it with: pip install fpdf"
        )
    
    # Check if font is specified and available
    if config.font_path:
        if not os.path.exists(config.font_path):
            raise FileNotFoundError(
                f"Specified font file not found: {config.font_path}. "
                "Please ensure the font file exists or set font_path to None to use system default."
            )
    
    # Check if output directory is writable
    pdf_dir = os.path.dirname(config.pdf_path)
    if pdf_dir:
        os.makedirs(pdf_dir, exist_ok=True)
        if not os.access(pdf_dir, os.W_OK):
            raise PermissionError(
                f"Cannot write to PDF output directory: {pdf_dir}. "
                "Please ensure you have write permissions."
            )
    
    # Check if temporary directory is writable
    os.makedirs(config.tmp_dir, exist_ok=True)
    if not os.access(config.tmp_dir, os.W_OK):
        raise PermissionError(
            f"Cannot write to temporary directory: {config.tmp_dir}. "
            "Please ensure you have write permissions."
        )
    
    # Check if font output directory is writable
    os.makedirs(config.font_output_dir, exist_ok=True)
    if not os.access(config.font_output_dir, os.W_OK):
        raise PermissionError(
            f"Cannot write to font output directory: {config.font_output_dir}. "
            "Please ensure you have write permissions."
        )

# Default client and model
client = _DummyClient()
model_id = "gemma-3-local"

def _refine_analysis_content(conversation_log: list, config: AnalysisConfig) -> list:
    """Refine and organize the analysis content to create a more polished report.
    
    Args:
        conversation_log (list): List of tuples containing the conversation history
        config (AnalysisConfig): Configuration object containing settings
        
    Returns:
        list: Refined conversation log with organized content
    """
    # Initialize the appropriate client based on config
    if config.use_api:
        from ai_analyst.classes import APIChat
        client = APIChat(config)
        model_id = config.api_model_id
    else:
        client = _DummyClient()
        model_id = config.model_path
    
    # Extract and organize content from the conversation log
    analysis_content = {
        "executive_summary": [],
        "data_overview": [],
        "key_insights": [],
        "visual_analysis": [],
        "statistical_analysis": [],
        "conclusions": []
    }
    
    # Track visualizations and their context
    visualizations = []
    current_section = "data_overview"
    for kind, content in conversation_log:
        if kind == "LLM":
            # Add content to appropriate section
            analysis_content[current_section].append(content)
        elif kind == "TOOL":
            # Add tool results to statistical analysis
            analysis_content["statistical_analysis"].append(f"Analysis Result: {content}")
        elif kind == "TOOL_IMG":
            # Store visualization with its context
            visualizations.append({
                "path": content,
                "context": analysis_content[current_section][-1] if analysis_content[current_section] else "Visualization"
            })
        elif kind == "DECIDER":
            # Add decision to appropriate section
            analysis_content[current_section].append(f"Analysis Decision: {content}")
    
    # Create a prompt for the refinement LLM
    refinement_prompt = f"""
    You are a Data Analysis Report Refiner. Your task is to take the raw analysis content and create a polished, professional report.
    
    Here is the organized content from the analysis:
    
    1. Data Overview:
    {chr(10).join(analysis_content['data_overview'])}
    
    2. Key Insights:
    {chr(10).join(analysis_content['key_insights'])}
    
    3. Visual Analysis:
    {len(visualizations)} visualizations available with their context
    
    4. Statistical Analysis:
    {chr(10).join(analysis_content['statistical_analysis'])}
    
    5. Conclusions:
    {chr(10).join(analysis_content['conclusions'])}
    
    Please create a well-structured report with the following sections:
    1. Table of Contents - List all sections with page numbers
    2. Executive Summary - A concise overview of the key findings
    3. Data Overview - Initial observations about the dataset
    4. Key Insights - The most important findings from the analysis
    5. Visual Analysis - Organized presentation of visualizations with clear interpretations
    6. Statistical Analysis - Summary of statistical findings
    7. Conclusions and Recommendations - Actionable insights and next steps
    
    Rules:
    - Create a clear table of contents at the beginning
    - Remove any redundant information
    - Ensure each visualization is presented with clear context and interpretation
    - Maintain a professional and analytical tone
    - Focus on actionable insights
    - Keep the most relevant visualizations and remove duplicates
    - Ensure smooth transitions between sections
    - Place visualizations close to their relevant text descriptions
    - Ensure proper page breaks and layout
    
    Please provide the refined content in a structured format that can be easily converted to a PDF.
    """
    
    # Get the refined content from the LLM
    chat = client.chats.create(model=model_id)
    response = chat.send_message(refinement_prompt, config=config)
    refined_content = response.text
    
    # Convert the refined content back into the conversation log format
    refined_log = []
    
    # Add the refined content as a single LLM message
    refined_log.append(("LLM", refined_content))
    
    # Preserve the visualizations with their context
    for viz in visualizations:
        refined_log.append(("LLM", f"Visualization Context: {viz['context']}"))
        refined_log.append(("TOOL_IMG", viz['path']))
    
    return refined_log

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
    
    # Validate PDF generation requirements
    _validate_pdf_requirements(config)
    
    # Initialize the appropriate client based on config
    if config.use_api:
        from ai_analyst.classes import APIChat
        client = APIChat(config)
        model_id = config.api_model_id
    else:
        client = _DummyClient()
        model_id = config.model_path
    
    # Set the global DataFrame for the analysis functions
    import ai_analyst.utils.analysis_toolkit as toolkit
    toolkit.df = data
    
    # Get available columns for the prompt
    available_columns = list(data.columns)
    
    # Create the complex prompt with the configuration
    complex_prompt = f"""
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
    Available columns in the dataset: {available_columns}
    
    You can call any tool by producing a code block with:
    ```tool_code
    <function_call_here>
    ```
    """
    
    # Initialize conversation log
    conversation_log = []
    
    final_text = chat_with_tools(
        user_message=complex_prompt,
        client=client,
        model_id=model_id,
        conversation_log=conversation_log,
        final_answer="",
        iterations=0,
        sleep_secs=config.sleep_seconds,
        data_about=config.data_about,
        tmp_dir=config.tmp_dir,
        pdf_path=config.pdf_path,
        config=config,
        df=data
    )
    
    # Refine the analysis content
    refined_log = _refine_analysis_content(conversation_log, config)
    
    # Save the refined content to PDF
    from ai_analyst.general_utils.pdf_utils import save_conversation_to_pdf
    save_conversation_to_pdf(refined_log, config.pdf_path, config)
    
    print("FINAL ANALYST ANSWER ===")
    print(final_text)
    print(f"PDF conversation saved at {config.pdf_path}")
    return client.chats.create(model=model_id).send_message(str(data), config=config)
