from fpdf import FPDF
from ai_analyst.general_utils.file_utils import copy_files
import os
from datetime import datetime
import pkg_resources
from ai_analyst.analysis_kit.config import AnalysisConfig


class PDF(FPDF):
    def __init__(self, font_path: str = None, font_family: str = "DejaVu", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Register custom font if provided
        if font_path:
            self.add_font(font_family, "", font_path, uni=True)
            self.add_font(font_family, "B", font_path, uni=True)
            self.add_font(font_family, "I", font_path, uni=True)
        self.font_family = font_family

        # Turn on automatic page breaks with a 15 mm bottom margin
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        # Try pkg_resources first, fallback to relative path
        try:
            logo_path = pkg_resources.resource_filename('ai_analyst', 'resources/logo.png')
        except Exception:
            curr = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(curr, '..', 'resources', 'logo.png')

        if os.path.exists(logo_path):
            self.image(logo_path, 10, 8, 33)

        self.set_font(self.font_family, 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Data Analysis Report', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_family, 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def strip_html(self, raw: str) -> str:
        """Remove any HTML tags (e.g. <h2>, </h2>) before rendering."""
        return re.sub(r'<[^>]+>', '', raw)

    def ensure_space(self, needed_height: float):
        """
        If there isn't enough room on the current page to fit
        'needed_height' (in user units), add a new page.
        """
        if self.get_y() + needed_height > self.page_break_trigger:
            self.add_page()


def save_conversation_to_pdf(
    conversation_log,
    pdf_path: str,
    config: AnalysisConfig,
):
    """Saves the refined conversation (with LLM text, code blocks, decisions, and images)
    into a paginated PDF, ensuring no overlaps and stripping stray HTML."""
    # Copy fonts into working directory
    working_font_path = copy_files(config)

    # Initialize PDF and pagination settings
    pdf = PDF(working_font_path, config.font_family)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_text_color(51, 51, 51)
    pdf.set_draw_color(200, 200, 200)

    # — Title Page —
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

    # — First pass: collect TOC entries & visualizations —
    sections = []       # list of (section_title, page_num)
    visualizations = [] # list of dicts with keys: path, context
    current_section = None

    for kind, content in conversation_log:
        if kind == "LLM":
            txt = content.strip()
            if txt.startswith("Table of Contents"):
                continue
            # Detect numbered or roman‐numeral headings
            elif any(txt.startswith(f"{i}.") for i in range(1, 10)) or \
                 any(txt.startswith(r) for r in ["I.", "II.", "III.", "IV.", "V."]):
                sections.append((txt, pdf.page_no()))
            elif txt.startswith("Visualization Context:"):
                current_section = txt
        elif kind == "TOOL_IMG":
            if current_section:
                visualizations.append({
                    "path": content,
                    "context": current_section
                })
            current_section = None

    # — Render Table of Contents —
    pdf.set_font(config.font_family, "B", 16)
    pdf.cell(0, 10, "Table of Contents", ln=True)
    pdf.ln(5)
    pdf.set_font(config.font_family, "", 11)
    for title, page in sections:
        pdf.cell(0, 8, f"{title} ...... {page}", ln=True)
    pdf.ln(15)

    # — Second pass: render body text, code, tool outputs, decisions —
    for kind, content in conversation_log:
        if kind == "LLM":
            if content.startswith(("Table of Contents", "Visualization Context:")):
                continue
            text = pdf.strip_html(content)
            # Estimate height: ~5 mm per line plus small padding
            lines = text.count("\n") + 1
            needed_h = lines * 5 + 5
            pdf.ensure_space(needed_h)
            pdf.set_font(config.font_family, "", 11)
            pdf.set_fill_color(248, 248, 248)
            pdf.multi_cell(0, 5, text, fill=True)
            pdf.ln(5)

        elif kind == "TOOL_CODE":
            # Reserve ~20 mm for code blocks
            pdf.ensure_space(20)
            pdf.set_fill_color(240, 240, 240)
            pdf.set_draw_color(200, 200, 200)
            pdf.set_font(config.font_family, "", 10)
            x, y = pdf.get_x(), pdf.get_y()
            pdf.rect(x, y, 190, 20)
            pdf.multi_cell(0, 5, content, fill=True)
            pdf.ln(5)

        elif kind == "TOOL":
            text = content
            lines = text.count("\n") + 1
            needed_h = lines * 5 + 5
            pdf.ensure_space(needed_h)
            pdf.set_font(config.font_family, "I", 10)
            pdf.set_fill_color(252, 252, 252)
            pdf.multi_cell(0, 5, text, fill=True)
            pdf.ln(5)

        elif kind == "DECIDER":
            text = f"Decision: {content}"
            lines = text.count("\n") + 1
            needed_h = lines * 5 + 5
            pdf.ensure_space(needed_h)
            pdf.set_text_color(0, 102, 204)
            pdf.set_font(config.font_family, "B", 10)
            pdf.multi_cell(0, 5, text)
            pdf.ln(5)
            pdf.set_text_color(51, 51, 51)

    # — Visualizations section —
    if visualizations:
        pdf.add_page()
        pdf.set_font(config.font_family, "B", 16)
        pdf.cell(0, 10, "Visualizations", ln=True)
        pdf.ln(10)

        for viz in visualizations:
            # Reserve space: image height + caption + buffer
            img_h, caption_h, buffer = 90, 5, 10
            pdf.ensure_space(img_h + caption_h + buffer)

            pdf.set_font(config.font_family, "I", 10)
            pdf.multi_cell(0, 5, viz["context"])
            pdf.ln(3)

            x, y = pdf.get_x(), pdf.get_y()
            pdf.set_draw_color(100, 100, 100)
            pdf.rect(x, y, 140, img_h)

            if os.path.exists(viz["path"]):
                pdf.image(viz["path"], x=x, y=y, w=140)
                pdf.set_font(config.font_family, "I", 8)
                pdf.cell(0, 5, "Analysis Visualization", ln=True)
            else:
                pdf.set_font(config.font_family, "I", 8)
                pdf.multi_cell(0, 5, f"Image not found: {viz['path']}")

            pdf.ln(buffer)

    # — Footer metadata —
    pdf.set_y(-15)
    pdf.set_font(config.font_family, "I", 8)
    pdf.cell(
        0,
        10,
        f"Generated by AI Analyst | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        align="C"
    )

    pdf.output(pdf_path)
    print(f"PDF saved → {pdf_path}")
