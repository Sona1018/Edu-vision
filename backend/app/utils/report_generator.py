import os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "storage", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_excel_report(evaluation: dict) -> str:
    """
    Generates an Excel report from an evaluation result.
    Returns the file path.
    """
    eval_id = evaluation.get("evaluation_id", "report")
    filepath = os.path.join(REPORTS_DIR, f"evaluation_{eval_id}.xlsx")
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Evaluation Summary"
    
    # Styles - Clean Monochromatic Theme (Black/Charcoal/White)
    header_fill = PatternFill(start_color="18181B", end_color="18181B", fill_type="solid") # Dark charcoal
    sub_fill = PatternFill(start_color="27272A", end_color="27272A", fill_type="solid")
    alt_fill = PatternFill(start_color="F4F4F5", end_color="F4F4F5", fill_type="solid")
    
    title_font = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    meta_label_font = Font(name="Calibri", size=10, bold=True, color="27272A")
    meta_val_font = Font(name="Calibri", size=10, bold=False, color="000000")
    data_font = Font(name="Calibri", size=10, color="000000")
    
    thin_border = Border(
        left=Side(style='thin', color='E4E4E7'),
        right=Side(style='thin', color='E4E4E7'),
        top=Side(style='thin', color='E4E4E7'),
        bottom=Side(style='thin', color='E4E4E7')
    )
    
    # Title block
    ws.merge_cells("A1:G1")
    ws["A1"] = "INTELLIPLACE AI ANSWER SHEET EVALUATION REPORT"
    ws["A1"].font = title_font
    ws["A1"].fill = header_fill
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 36
    
    # Metadata Block
    summary = evaluation.get("summary", {})
    metadata = [
        ("Evaluation ID:", eval_id, "Evaluation Date:", evaluation.get("timestamp", "N/A")),
        ("Total Marks Obtained:", f"{summary.get('total_obtained', 0)} / {summary.get('total_max', 0)}", "Score Percentage:", f"{summary.get('percentage', 0)}%"),
        ("Questions Evaluated:", str(summary.get("total_questions", 0)), "Flags Requiring Review:", str(summary.get("review_flags_count", 0)))
    ]
    
    row_idx = 3
    for row_data in metadata:
        ws.cell(row=row_idx, column=1, value=row_data[0]).font = meta_label_font
        ws.cell(row=row_idx, column=2, value=row_data[1]).font = meta_val_font
        ws.cell(row=row_idx, column=4, value=row_data[2]).font = meta_label_font
        ws.cell(row=row_idx, column=5, value=row_data[3]).font = meta_val_font
        row_idx += 1
        
    row_idx += 1
    
    # Table Headers
    headers = [
        "Q#", "Max Marks", "Marks Awarded", "Status", "Student Answer", 
        "Evaluation Explanation", "Missing Concepts / Suggestions"
    ]
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=row_idx, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = sub_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border
    ws.row_dimensions[row_idx].height = 25
    
    # Data Rows
    row_idx += 1
    questions = evaluation.get("question_evaluations", [])
    for q_idx, q in enumerate(questions):
        status = "Needs Review" if q.get("needs_review") else "Evaluated"
        missing_list = q.get("missing_concepts", [])
        suggestions = q.get("suggestions", [])
        
        notes = []
        if missing_list:
            notes.append("Missing: " + "; ".join(missing_list))
        if suggestions:
            notes.append("Suggestions: " + " ".join(suggestions))
        notes_str = "\n".join(notes) if notes else "None"
        
        row_values = [
            f"Q{q.get('question_number')}",
            q.get("max_marks", 0),
            q.get("awarded_marks", 0),
            status,
            q.get("student_answer", "").strip(),
            q.get("explanation", "").strip(),
            notes_str
        ]
        
        for col_idx, val in enumerate(row_values, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.font = data_font
            cell.border = thin_border
            if q_idx % 2 == 1:
                cell.fill = alt_fill
            if col_idx in [1, 2, 3, 4]:
                cell.alignment = Alignment(horizontal="center", vertical="top")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        
        ws.row_dimensions[row_idx].height = 45
        row_idx += 1
        
    # Column widths
    col_widths = {1: 8, 2: 12, 3: 14, 4: 14, 5: 35, 6: 40, 7: 35}
    for col, width in col_widths.items():
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = width
        
    wb.save(filepath)
    return filepath


def generate_pdf_report(evaluation: dict) -> str:
    """
    Generates a PDF report from an evaluation result.
    Returns the file path.
    """
    eval_id = evaluation.get("evaluation_id", "report")
    filepath = os.path.join(REPORTS_DIR, f"evaluation_{eval_id}.pdf")
    
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#09090B'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#71717A'),
        spaceAfter=15
    )
    
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#18181B'),
        spaceBefore=12,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#27272A')
    )
    
    bold_style = ParagraphStyle(
        'BoldBody',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#18181B')
    )
    
    story = []
    
    # Title
    story.append(Paragraph("INTELLIPLACE EVALUATION REPORT", title_style))
    story.append(Paragraph("Automated Handwritten Answer Sheet Evaluation Powered by OCR & AI Assessment", subtitle_style))
    
    # Summary Box
    summary = evaluation.get("summary", {})
    summary_data = [
        [
            Paragraph("<b>Evaluation ID:</b>", body_style),
            Paragraph(eval_id, body_style),
            Paragraph("<b>Date:</b>", body_style),
            Paragraph(str(evaluation.get("timestamp", "N/A"))[:19], body_style)
        ],
        [
            Paragraph("<b>Total Score:</b>", body_style),
            Paragraph(f"<b>{summary.get('total_obtained', 0)} / {summary.get('total_max', 0)}</b> ({summary.get('percentage', 0)}%)", bold_style),
            Paragraph("<b>Status:</b>", body_style),
            Paragraph(f"{'Flagged for Review' if summary.get('review_flags_count', 0) > 0 else 'All Clear'}", body_style)
        ],
        [
            Paragraph("<b>Questions Evaluated:</b>", body_style),
            Paragraph(str(summary.get("total_questions", 0)), body_style),
            Paragraph("<b>Review Flags:</b>", body_style),
            Paragraph(str(summary.get("review_flags_count", 0)), body_style)
        ]
    ]
    
    summary_table = Table(summary_data, colWidths=[120, 150, 100, 170])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4F4F5')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E4E4E7')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E4E4E7')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # Question Wise Details
    story.append(Paragraph("QUESTION-WISE DETAILED EVALUATION", section_title))
    
    questions = evaluation.get("question_evaluations", [])
    for q in questions:
        q_num = q.get("question_number")
        awarded = q.get("awarded_marks", 0)
        max_m = q.get("max_marks", 0)
        needs_rev = q.get("needs_review", False)
        
        status_text = " [FLAGGED FOR REVIEW]" if needs_rev else ""
        header_text = f"<b>Question {q_num} — Score: {awarded} / {max_m} Marks</b>{status_text}"
        
        q_box_data = [
            [Paragraph(header_text, bold_style)],
            [Paragraph(f"<b>Extracted Student Answer:</b><br/>{q.get('student_answer', 'No answer provided')}", body_style)],
            [Paragraph(f"<b>Reference Answer:</b><br/>{q.get('reference_answer', 'N/A')}", body_style)],
            [Paragraph(f"<b>AI Assessment & Rubric Breakdown:</b><br/>{q.get('explanation', '')}", body_style)]
        ]
        
        if q.get("missing_concepts"):
            missing_txt = ", ".join(q.get("missing_concepts"))
            q_box_data.append([Paragraph(f"<b>Missing Key Concepts:</b><br/>{missing_txt}", body_style)])
            
        if q.get("incorrect_statements"):
            inc_txt = "; ".join(q.get("incorrect_statements"))
            q_box_data.append([Paragraph(f"<b>Incorrect / Contradictory Statements:</b><br/>{inc_txt}", body_style)])
            
        if q.get("suggestions"):
            sugg_txt = " ".join(q.get("suggestions"))
            q_box_data.append([Paragraph(f"<b>Suggestions for Improvement:</b><br/>{sugg_txt}", body_style)])
            
        q_table = Table(q_box_data, colWidths=[540])
        bg_color = colors.HexColor('#FAFAFA') if not needs_rev else colors.HexColor('#F4F4F5')
        q_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), colors.HexColor('#E4E4E7')),
            ('BACKGROUND', (0,1), (-1,-1), bg_color),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#D4D4D8')),
            ('LINEBELOW', (0,0), (0,-1), 0.5, colors.HexColor('#E4E4E7')),
            ('PADDING', (0,0), (-1,-1), 6),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        
        story.append(KeepTogether(q_table))
        story.append(Spacer(1, 10))
        
    doc.build(story)
    return filepath
