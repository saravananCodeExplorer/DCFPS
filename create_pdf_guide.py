import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)

pdf_filename = r"c:\Users\HINATA\Music\Bakya Akka\Simple_Explanation_and_Run_Guide.pdf"
doc = SimpleDocTemplate(
    pdf_filename,
    pagesize=letter,
    rightMargin=40, leftMargin=40,
    topMargin=40, bottomMargin=40
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=22,
    leading=26,
    textColor=colors.HexColor('#1A365D'),
    spaceAfter=10
)

subtitle_style = ParagraphStyle(
    'DocSubtitle',
    parent=styles['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=12,
    leading=16,
    textColor=colors.HexColor('#4A5568'),
    spaceAfter=15
)

h1_style = ParagraphStyle(
    'SectionH1',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=14,
    leading=18,
    textColor=colors.HexColor('#2B6CB0'),
    spaceBefore=12,
    spaceAfter=8
)

body_style = ParagraphStyle(
    'BodyTextCustom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=10,
    leading=14,
    textColor=colors.HexColor('#2D3748'),
    spaceAfter=8
)

bold_body_style = ParagraphStyle(
    'BoldBodyText',
    parent=body_style,
    fontName='Helvetica-Bold'
)

code_style = ParagraphStyle(
    'CodeBox',
    parent=styles['Normal'],
    fontName='Courier',
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor('#1A202C'),
    backColor=colors.HexColor('#EDF2F7'),
    borderColor=colors.HexColor('#CBD5E0'),
    borderWidth=1,
    borderPadding=6,
    spaceBefore=4,
    spaceAfter=8
)

story = []

# Title Header
story.append(Paragraph("Data Center Failure Prediction System", title_style))
story.append(Paragraph("<b>Simple Guide:</b> How to Run the Project and How it Works (PhD Methodology M1)", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2B6CB0'), spaceAfter=15))

# Section 1: Simple Explanation
story.append(Paragraph("1. What is this Project? (Simple Explanation)", h1_style))
story.append(Paragraph(
    "Imagine a massive computer server room (data center) with thousands of machines running 24/7. "
    "Occasionally, servers break down unexpectedly. However, <b>before a server crashes, it leaves early warning signs</b> — "
    "such as CPU usage spiking, memory filling up, or disk speeds slowing down.",
    body_style
))
story.append(Paragraph(
    "This project builds an <b>Artificial Intelligence (AI) Early Warning System using XGBoost</b> that detects these warning signs "
    "<b>30 minutes BEFORE the server actually fails</b>. This gives system administrators time to safely move tasks to another computer before a crash occurs.",
    body_style
))

story.append(Spacer(1, 10))

# Section 2: How to Run
story.append(Paragraph("2. How to Run the Project (Step-by-Step)", h1_style))
story.append(Paragraph("You can run the entire project in <b>one single step</b> from your terminal:", body_style))

story.append(Paragraph("<b>Step 1:</b> Open Command Prompt or Terminal.", body_style))
story.append(Paragraph("<b>Step 2:</b> Navigate to your project folder:", body_style))
story.append(Paragraph("cd \"c:\\Users\\HINATA\\Music\\Bakya Akka\"", code_style))

story.append(Paragraph("<b>Step 3:</b> Run the python pipeline script (make sure to include <b>.py</b>):", body_style))
story.append(Paragraph("python run_pipeline.py", code_style))

story.append(Paragraph(
    "<i>Note: Running <code>python run_pipeline</code> without <code>.py</code> will cause an error! Always type <code>python run_pipeline.py</code>.</i>",
    body_style
))

story.append(Spacer(1, 10))

# Section 3: How the Pipeline Works
story.append(Paragraph("3. How the Pipeline Works (The 2 Modules)", h1_style))
story.append(Paragraph(
    "The codebase is split into two logical modules to ensure clean data processing and <b>zero cheating (no data leakage)</b>:",
    body_style
))

module_table_data = [
    [Paragraph("<b>Module & File</b>", bold_body_style), Paragraph("<b>What it Does in Simple Terms</b>", bold_body_style)],
    [
        Paragraph("<b>Module 1</b><br/><code>Module1_Dataset_...ipynb</code>", body_style),
        Paragraph(
            "1. <b>Simulates Server Telemetry:</b> Generates 50,000 continuous time-series rows for 25 machines.<br/>"
            "2. <b>Calculates 52 Rolling Features:</b> Computes averages, spikes, and trends over past 25m, 75m, and 300m.<br/>"
            "3. <b>Checks History Rules:</b> Ensures feature math strictly uses <i>past</i> data ($t' \\le t$).",
            body_style
        )
    ],
    [
        Paragraph("<b>Module 2</b><br/><code>Module2_Forward_...ipynb</code>", body_style),
        Paragraph(
            "1. <b>Creates Warning Labels:</b> Flags $y=1$ if a server will fail in the next 30 minutes.<br/>"
            "2. <b>Chronological Split:</b> Divides data strictly by time (70% Train, 15% Val, 15% Test).<br/>"
            "3. <b>5-Point Audit:</b> Programmatically confirms 100% zero data leakage.<br/>"
            "4. <b>Trains AI Models:</b> Trains XGBoost, Random Forest, and SVM classifiers.",
            body_style
        )
    ]
]

t_mod = Table(module_table_data, colWidths=[150, 380])
t_mod.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
]))
story.append(t_mod)

story.append(Spacer(1, 12))

# Section 4: Results
story.append(Paragraph("4. Understanding the Results", h1_style))
story.append(Paragraph("When you run <code>python run_pipeline.py</code>, it trains the models and tests them on unseen future data. Here are the final benchmark results:", body_style))

results_data = [
    [Paragraph("<b>Model</b>", bold_body_style), Paragraph("<b>Threshold</b>", bold_body_style), Paragraph("<b>Precision</b>", bold_body_style), Paragraph("<b>Recall</b>", bold_body_style), Paragraph("<b>F1-Score</b>", bold_body_style), Paragraph("<b>PR-AUC</b>", bold_body_style), Paragraph("<b>ROC-AUC</b>", bold_body_style)],
    [Paragraph("<b>Proposed XGBoost</b>", bold_body_style), Paragraph("0.860", body_style), Paragraph("65.8%", body_style), Paragraph("<b>100.0%</b>", bold_body_style), Paragraph("<b>79.3%</b>", bold_body_style), Paragraph("<b>0.854</b>", bold_body_style), Paragraph("<b>0.999</b>", bold_body_style)],
    [Paragraph("Random Forest Baseline", body_style), Paragraph("0.470", body_style), Paragraph("55.2%", body_style), Paragraph("100.0%", body_style), Paragraph("71.1%", body_style), Paragraph("0.796", body_style), Paragraph("0.999", body_style)],
    [Paragraph("SVM Baseline", body_style), Paragraph("0.430", body_style), Paragraph("69.8%", body_style), Paragraph("77.1%", body_style), Paragraph("73.3%", body_style), Paragraph("0.823", body_style), Paragraph("0.999", body_style)],
]

t_res = Table(results_data, colWidths=[120, 65, 65, 65, 65, 65, 65])
t_res.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
    ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#EBF8FF')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
]))
story.append(t_res)

story.append(Spacer(1, 8))
story.append(Paragraph("<b>Key Findings:</b>", bold_body_style))
story.append(Paragraph("• <b>Proposed XGBoost AI</b> achieves the best overall performance with an F1-Score of <b>79.3%</b> and PR-AUC of <b>0.854</b>.", body_style))
story.append(Paragraph("• <b>100% Recall:</b> The system caught <i>every single failure precursor event</i> on test data without missing any!", body_style))
story.append(Paragraph("• <b>Advance Warning Time:</b> Provides an average lead time of <b>30 minutes</b> before the server crashes.", body_style))

story.append(Spacer(1, 10))

# Section 5: Where Outputs Are Saved
story.append(Paragraph("5. Where Outputs Are Saved", h1_style))
story.append(Paragraph("All generated output files are stored cleanly in two subfolders:", body_style))
story.append(Paragraph("• <code>module1_outputs/</code>: Raw telemetry parquet file & engineered feature parquet file.", body_style))
story.append(Paragraph("• <code>module2_outputs/</code>:", body_style))
story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <code>xgboost_m1_precursor_model.json</code> (Saved trained AI model)", body_style))
story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <code>m1_benchmark_results.csv</code> (CSV table of all metrics)", body_style))
story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <code>m1_performance_and_feature_importance.png</code> (High-res chart of PR Curves & Top 15 Features)", body_style))
story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <code>m1_lead_time_distribution.png</code> (Chart showing 30-min lead time distribution)", body_style))

doc.build(story)
print(f"PDF successfully generated: {pdf_filename}")
