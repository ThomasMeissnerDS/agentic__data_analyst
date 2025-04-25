from fpdf import FPDF
from ai_analyst.general_utils.file_utils import copy_files


def save_conversation_to_pdf(
        conversation_log, 
        pdf_path,
        font_path: str = "/kaggle/working/output/dejavusans-bold-ttf/DejaVuSans-Bold.ttf",
        ):
    pdf = FPDF()
    pdf.add_page()
    copy_files()
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 11)

    for kind, content in conversation_log:
        if kind == "TOOL_IMG":
            pdf.image(content, w=180)  # full width
            pdf.ln(2)
            continue
        if kind in {"LLM", "DECIDER", "TOOL", "TOOL_CODE"}:
            pdf.multi_cell(0, 5, f"[{kind}] {content}")
            pdf.ln(2)
    pdf.output(pdf_path)
    print(f"PDF saved → {pdf_path}")
