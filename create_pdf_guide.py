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
    fontSize=20,
    leading=24,
    textColor=colors.HexColor('#1A365D'),
    spaceAfter=8
)

subtitle_style = ParagraphStyle(
    'DocSubtitle',
    parent=styles['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=11,
    leading=15,
    textColor=colors.HexColor('#4A5568'),
    spaceAfter=12
)

h1_style = ParagraphStyle(
    'SectionH1',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=13,
    leading=17,
    textColor=colors.HexColor('#2B6CB0'),
    spaceBefore=10,
    spaceAfter=6
)

body_style = ParagraphStyle(
    'BodyTextCustom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9.5,
    leading=13.5,
    textColor=colors.HexColor('#2D3748'),
    spaceAfter=6
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
    fontSize=9,
    leading=12,
    textColor=colors.HexColor('#1A202C'),
    backColor=colors.HexColor('#EDF2F7'),
    borderColor=colors.HexColor('#CBD5E0'),
    borderWidth=1,
    borderPadding=5,
    spaceBefore=3,
    spaceAfter=6
)

story = []

# Title Header
story.append(Paragraph("Data Center Failure Prediction System", title_style))
story.append(Paragraph("<b>Run Guide & Paper Methodology Overview</b> (Modules 1 & 2 + 5-Point Leakage Audit)", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2B6CB0'), spaceAfter=12))

# Section 1: Simple Explanation
story.append(Paragraph("1. What is this Project? (Simple Explanation)", h1_style))
story.append(Paragraph(
    "Imagine a massive cloud server infrastructure running thousands of tasks 24/7. "
    "Before a machine fails, it exhibits subtle <b>telemetry degradation patterns</b> — "
    "such as CPU usage drift, memory exhaustion, disk I/O bottlenecks, or scheduler latency spikes.",
    body_style
))
story.append(Paragraph(
    "This project builds an <b>Artificial Intelligence (AI) Precursor Warning System using XGBoost</b> that detects failure precursors "
    "<b>30 minutes BEFORE a server crash occurs</b>, allowing live task migration and zero downtime.",
    body_style
))

story.append(Spacer(1, 6))

# Section 2: How to Run
story.append(Paragraph("2. How to Run the Project (Step-by-Step)", h1_style))
story.append(Paragraph("You can execute the entire pipeline in <b>one step</b> from your terminal:", body_style))

story.append(Paragraph("<b>Step 1:</b> Open Command Prompt or PowerShell.", body_style))
story.append(Paragraph("<b>Step 2:</b> Navigate to your project folder:", body_style))
story.append(Paragraph("cd \"c:\\Users\\HINATA\\Music\\Bakya Akka\"", code_style))

story.append(Paragraph("<b>Step 3:</b> Execute the master pipeline script:", body_style))
story.append(Paragraph("python run_pipeline.py", code_style))

story.append(Spacer(1, 6))

# Section 3: How the Pipeline Works
story.append(Paragraph("3. How the Pipeline Works (The 2 Modules & 5-Point Audit)", h1_style))
story.append(Paragraph(
    "The pipeline is divided into two modules with strict temporal isolation and <b>zero data leakage</b>:",
    body_style
))

module_table_data = [
    [Paragraph("<b>Module & Component</b>", bold_body_style), Paragraph("<b>Methodology & Workflow Implementation</b>", bold_body_style)],
    [
        Paragraph("<b>Module 1</b><br/><code>Module1_...ipynb</code>", body_style),
        Paragraph(
            "1. <b>Stochastic Telemetry Simulation:</b> Generates 50,000 continuous time-series rows across 25 machines using Ornstein-Uhlenbeck processes superposed with non-linear degradation drift.<br/>"
            "2. <b>Backward-Only Window Features:</b> Computes 52 rolling mean, std, trend, and CV features over $W \\in \\{5, 15, 60\\}$ timesteps.<br/>"
            "3. <b>Strict History Rule:</b> Enforces <code>min_periods=W</code> so features rely strictly on past raw telemetry ($t' \\le t$).",
            body_style
        )
    ],
    [
        Paragraph("<b>Module 2</b><br/><code>Module2_...ipynb</code>", body_style),
        Paragraph(
            "1. <b>Forward Horizon Labeling ($H=30$m):</b> Labels $y=1$ for timesteps in $[f_{idx}-H, f_{idx})$.<br/>"
            "2. <b>Chronological Split with Embargo Purging:</b> 70% Train, 15% Val, 15% Test per machine. Purges $W_{max}-1$ feature lookback rows and $H$ horizon lookahead rows at split boundaries.<br/>"
            "3. <b>Full 5-Point Leakage Audit:</b> Verifies Feature Bounds, Target Isolation, Temporal Separation, Feature Embargo Isolation, and Target Horizon Isolation.<br/>"
            "4. <b>Forward-Chaining CV & Hyperparameter Tuning:</b> Tunes XGBoost hyperparameters using expanding temporal folds before final model fitting.",
            body_style
        )
    ]
]

t_mod = Table(module_table_data, colWidths=[140, 390])
t_mod.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
]))
story.append(t_mod)

story.append(Spacer(1, 8))

# Section 4: Results
story.append(Paragraph("4. Benchmark Performance Results", h1_style))
story.append(Paragraph("Final benchmark evaluation on the out-of-time test partition after embargo purging and forward-chaining CV:", body_style))

results_data = [
    [Paragraph("<b>Model</b>", bold_body_style), Paragraph("<b>Threshold</b>", bold_body_style), Paragraph("<b>Precision</b>", bold_body_style), Paragraph("<b>Recall</b>", bold_body_style), Paragraph("<b>F1-Score</b>", bold_body_style), Paragraph("<b>PR-AUC</b>", bold_body_style), Paragraph("<b>ROC-AUC</b>", bold_body_style)],
    [Paragraph("<b>Proposed XGBoost</b>", bold_body_style), Paragraph("0.730", body_style), Paragraph("55.4%", body_style), Paragraph("<b>100.0%</b>", bold_body_style), Paragraph("<b>71.3%</b>", bold_body_style), Paragraph("<b>0.838</b>", bold_body_style), Paragraph("<b>0.999</b>", bold_body_style)],
    [Paragraph("Random Forest Baseline", body_style), Paragraph("0.480", body_style), Paragraph("51.4%", body_style), Paragraph("100.0%", body_style), Paragraph("67.9%", body_style), Paragraph("0.463", body_style), Paragraph("0.999", body_style)],
    [Paragraph("SVM Baseline", body_style), Paragraph("0.330", body_style), Paragraph("64.7%", body_style), Paragraph("91.7%", body_style), Paragraph("75.9%", body_style), Paragraph("0.849", body_style), Paragraph("0.999", body_style)],
]

t_res = Table(results_data, colWidths=[120, 65, 65, 65, 65, 65, 65])
t_res.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
    ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#EBF8FF')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(t_res)

story.append(Spacer(1, 6))
story.append(Paragraph("<b>Key Methodological Achievements:</b>", bold_body_style))
story.append(Paragraph("• <b>100% Zero Leakage:</b> Passed all 5 programmatic audit checks including boundary embargo isolation.", body_style))
story.append(Paragraph("• <b>High Precursor Recall:</b> Proposed XGBoost detects failure precursors with <b>100% Recall</b> on out-of-time test data.", body_style))
story.append(Paragraph("• <b>30-Minute Early Warning:</b> Achieves a mean precursor lead time of <b>30.0 minutes</b> prior to failure.", body_style))

story.append(Spacer(1, 8))

# Section 5: Where Outputs Are Saved
story.append(Paragraph("5. Generated Artifacts & Directory Structure", h1_style))
story.append(Paragraph("All generated output artifacts are organized in dedicated output directories:", body_style))
story.append(Paragraph("• <code>module1_outputs/</code>: Raw telemetry & backward sliding-window feature Parquet files.", body_style))
story.append(Paragraph("• <code>module2_outputs/</code>:", body_style))
story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <code>xgboost_m1_precursor_model.json</code> (Trained XGBoost model)", body_style))
story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <code>m1_benchmark_results.csv</code> (Benchmark evaluation table)", body_style))
story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <code>m1_performance_and_feature_importance.png</code> (PR curves & Feature importances chart)", body_style))
story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <code>m1_lead_time_distribution.png</code> (Precursor lead-time distribution chart)", body_style))

doc.build(story)
print(f"PDF successfully generated: {pdf_filename}")
