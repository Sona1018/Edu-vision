import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

pdf_path = os.path.join(os.path.dirname(__file__), "sample_student_sheet.pdf")
doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
styles = getSampleStyleSheet()

title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, leading=20)
h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, leading=16, spaceBefore=10)
body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=14)

story = [
    Paragraph("Assignment - 1: Marketing Management", title_style),
    Paragraph("Student Name: Riya Mishra | Roll: 1630102023 | Subject: MM", body_style),
    Spacer(1, 15),
    Paragraph("Q1. Define Marketing. Explain nature of marketing.", h2_style),
    Paragraph("Answer: Marketing is the activity, set of institutions and processes for creating, communicating, delivering, and exchanging offerings that have value for customers, clients, partners, and society at large.", body_style),
    Paragraph("Nature of Marketing:<br/>- Customer-Centric Orientation: Understanding consumer behaviour and expectations.<br/>- Value Creation and Delivery: Creating superior value through marketing mix.<br/>- Exchange-Based: Marketing operates on voluntary exchange where both buyer and seller gain value.<br/>- Continuous and Dynamic Process: Evolving with changes in technology and consumer preferences.<br/>- Integrated activities: Coordinates with production, finance, and HR.", body_style),
    Spacer(1, 15),
    Paragraph("2. What all are scope of marketing and importance of marketing?", h2_style),
    Paragraph("Answer: Scope of Marketing includes marketing of goods and services, ideas (social campaigns like Swachh Bharat), place marketing, experiences (theme parks, concerts), digital marketing, and person marketing.", body_style),
    Paragraph("Importance of Marketing includes customer relationship development, business revenue growth, competitive positioning, and consumer satisfaction.", body_style)
]

doc.build(story)
print("Sample student PDF created at:", pdf_path)
