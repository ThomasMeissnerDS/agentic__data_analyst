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
        "visual_analysis": [],
        "statistical_analysis": [],
        "conclusions": []
    }
    
    # Track visualizations and their context
    visualizations = []
    current_section = "executive_summary"  # Initialize with default section
    current_context = None
    
    for kind, content in conversation_log:
        if kind == "LLM":
            if content.startswith("Visualization Context:"):
                current_context = content
            elif content.startswith("Executive Summary"):
                current_section = "executive_summary"
            elif content.startswith("Data Overview"):
                current_section = "data_overview"
            elif content.startswith("Visual Analysis"):
                current_section = "visual_analysis"
            elif content.startswith("Statistical Analysis"):
                current_section = "statistical_analysis"
            elif content.startswith("Conclusions"):
                current_section = "conclusions"
            else:
                # Add content to appropriate section
                if current_section in analysis_content:
                    analysis_content[current_section].append(content)
                else:
                    # If no section is set, add to executive summary
                    analysis_content["executive_summary"].append(content)
        elif kind == "TOOL":
            # Add tool results to statistical analysis
            analysis_content["statistical_analysis"].append(f"Analysis Result: {content}")
        elif kind == "TOOL_IMG":
            # Store visualization with its context
            visualizations.append({
                "path": content,
                "context": current_context or "Visualization",
                "section": current_section
            })
            current_context = None
        elif kind == "DECIDER":
            # Add decision to appropriate section
            analysis_content[current_section].append(f"Analysis Decision: {content}")
    
    # Create refined content for each section in specific order
    refined_sections = {}
    section_order = [
        ("visual_analysis", "Visual Analysis"),
        ("statistical_analysis", "Statistical Analysis"),
        ("data_overview", "Data Overview"),
        ("executive_summary", "Executive Summary"),
        ("conclusions", "Conclusions")
    ]
    
    for section_key, section_title in section_order:
        # Create a prompt for each section
        section_prompt = f"""
        You are a Data Analysis Report Refiner. Your task is to create a polished section for the report.
        
        Section: {section_title}
        
        Content to refine:
        {chr(10).join(analysis_content[section_key])}
        
        {f"Relevant visualizations: {len([v for v in visualizations if v['section'] == section_key])}" if section_key in ['visual_analysis', 'statistical_analysis'] else ""}
        
        Please create a well-structured section with:
        1. A clear section header
        2. The main content
        3. Any relevant visualizations with their context (if applicable)
        4. A brief summary of key points
        
        Rules:
        - Maintain a professional and analytical tone
        - Focus on actionable insights
        - Remove any redundant information
        - Ensure smooth transitions between points
        - Place visualizations close to their relevant text descriptions
        """
        
        # Get the refined content for this section
        chat = client.chats.create(model=model_id)
        response = chat.send_message(section_prompt, config=config)
        refined_sections[section_key] = response.text
    
    # Create the final report structure
    final_report = f"""
    Table of Contents
    ================
    1. Executive Summary
    2. Data Overview
    3. Visual Analysis
    4. Statistical Analysis
    5. Conclusions
    
    Executive Summary
    ================
    {refined_sections['executive_summary']}
    
    Data Overview
    ============
    {refined_sections['data_overview']}
    
    Visual Analysis
    ==============
    {refined_sections['visual_analysis']}
    
    Statistical Analysis
    ==================
    {refined_sections['statistical_analysis']}
    
    Conclusions
    ==========
    {refined_sections['conclusions']}
    """
    
    # Convert the refined content back into the conversation log format
    refined_log = []
    
    # Add the final report
    refined_log.append(("LLM", final_report))
    
    # Add visualizations in their respective sections
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
