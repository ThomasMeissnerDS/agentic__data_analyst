from fpdf import FPDF
from ai_analyst.general_utils.file_utils import copy_files
import os
from datetime import datetime
import pkg_resources
from ai_analyst.analysis_kit.config import AnalysisConfig


class PDF(FPDF):
    def __init__(self, font_path: str = None, font_family: str = "DejaVu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register fonts before any page is added
        if font_path:
            self.add_font(font_family, "", font_path, uni=True)
            self.add_font(font_family, "B", font_path, uni=True)
            self.add_font(font_family, "I", font_path, uni=True)
        self.font_family = font_family

    def header(self):
        # Attempt to locate logo via pkg_resources, fallback to relative path
        try:
            logo_path = pkg_resources.resource_filename('ai_analyst', 'resources/logo.png')
        except Exception:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(current_dir, '..', 'resources', 'logo.png')

        if os.path.exists(logo_path):
            self.image(logo_path, 10, 8, 33)

        # Header title
        self.set_font(self.font_family, 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Data Analysis Report', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font(self.font_family, 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')


def save_conversation_to_pdf(
    conversation_log,
    pdf_path,
    config: AnalysisConfig,
):
    # Copy the font file to the working directory and get the new path
    working_font_path = copy_files(config)
    
    # Create PDF instance with fonts already registered
    pdf = PDF(working_font_path, config.font_family)
    pdf.alias_nb_pages()
    pdf.add_page()  # header() can now use font safely

    # Styles and metadata
    pdf.set_text_color(51, 51, 51)
    pdf.set_draw_color(200, 200, 200)

    # Title page
    pdf.set_font(config.font_family, "B", 24)
    pdf.cell(0, 20, "Data Analysis Report", ln=True, align="C")
    pdf.set_font(config.font_family, "", 12)
    pdf.cell(
        0,
        10,
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ln=True,
        align="C",
    )
    pdf.ln(20)

    # Table of contents
    pdf.set_font(config.font_family, "B", 16)
    pdf.cell(0, 10, "Table of Contents", ln=True)
    pdf.ln(10)

    # Track sections for table of contents
    sections = []
    current_section = None
    pdf.set_font(config.font_family, "", 11)

    # First pass: collect sections and their page numbers
    for kind, content in conversation_log:
        if kind == "LLM":
            if content.startswith("Table of Contents"):
                continue
            elif content.startswith("1.") or content.startswith("2.") or content.startswith("3.") or content.startswith("4.") or content.startswith("5.") or content.startswith("6.") or content.startswith("7."):
                sections.append((content, pdf.page_no()))
        elif kind == "TOOL_IMG":
            if current_section:
                sections.append((f"Visualization in {current_section}", pdf.page_no()))

    # Add table of contents
    pdf.set_font(config.font_family, "", 11)
    for section, page in sections:
        pdf.cell(0, 10, f"{section} ......................... {page}", ln=True)
    pdf.ln(20)

    # Second pass: add content with proper layout
    for kind, content in conversation_log:
        if kind == "LLM":
            if content.startswith("Table of Contents"):
                continue
            elif content.startswith("Visualization Context:"):
                # Store the context for the next visualization
                current_section = content
            else:
                # Check if we need a new page
                if pdf.get_y() > 250:  # If less than 20mm left on page
                    pdf.add_page()
                
                pdf.set_font(config.font_family, "", 11)
                pdf.set_fill_color(248, 248, 248)
                pdf.multi_cell(0, 5, content, fill=True)
                pdf.ln(5)

        elif kind == "TOOL_IMG":
            # Check if we need a new page
            if pdf.get_y() > 200:  # If less than 50mm left on page
                pdf.add_page()
            
            # Add visualization context if available
            if current_section:
                pdf.set_font(config.font_family, "I", 10)
                pdf.multi_cell(0, 5, current_section)
                pdf.ln(5)
            
            # Calculate image size to fit page (smaller size)
            img_width = 140  # Reduced from 180
            img_height = 90  # Reduced from 120
            
            # Get current position
            x = pdf.get_x()
            y = pdf.get_y()
            
            # Draw border
            pdf.set_draw_color(100, 100, 100)
            pdf.rect(x, y, img_width, img_height)
            
            # Add image if it exists
            if os.path.exists(content):
                pdf.image(content, x=x, y=y, w=img_width)
                pdf.set_font(config.font_family, "I", 8)
                pdf.cell(0, 5, "Analysis Visualization", ln=True)
            else:
                pdf.set_font(config.font_family, "I", 8)
                pdf.multi_cell(0, 5, f"Image not found: {content}")
            
            pdf.ln(10)
            current_section = None  # Reset context after visualization

        elif kind == "TOOL_CODE":
            # Check if we need a new page
            if pdf.get_y() > 250:  # If less than 20mm left on page
                pdf.add_page()
            
            pdf.set_fill_color(240, 240, 240)
            pdf.set_draw_color(200, 200, 200)
            pdf.set_font(config.font_family, "", 10)
            x, y = pdf.get_x(), pdf.get_y()
            pdf.rect(x, y, 190, 20)
            pdf.multi_cell(0, 5, content, fill=True)
            pdf.ln(5)

        elif kind == "TOOL":
            # Check if we need a new page
            if pdf.get_y() > 250:  # If less than 20mm left on page
                pdf.add_page()
            
            pdf.set_font(config.font_family, "I", 10)
            pdf.set_fill_color(252, 252, 252)
            pdf.multi_cell(0, 5, content, fill=True)
            pdf.ln(5)

        elif kind == "DECIDER":
            # Check if we need a new page
            if pdf.get_y() > 250:  # If less than 20mm left on page
                pdf.add_page()
            
            pdf.set_text_color(0, 102, 204)
            pdf.set_font(config.font_family, "B", 10)
            pdf.multi_cell(0, 5, f"Decision: {content}")
            pdf.ln(5)
            pdf.set_text_color(51, 51, 51)

        pdf.set_font(config.font_family, "", 11)

    # Footer generation info
    pdf.set_y(-15)
    pdf.set_font(config.font_family, "I", 8)
    pdf.cell(0, 10, f"Generated by AI Analyst | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")

    pdf.output(pdf_path)
    print(f"PDF saved → {pdf_path}")
