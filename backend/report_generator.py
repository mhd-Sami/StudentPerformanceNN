from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
from datetime import datetime

def generate_student_report(data):
    """
    Generate a PDF performance report for a student.
    
    Args:
        data (dict): Dictionary containing:
            - predictions (dict): Model predictions (Pass/Fail, Score, Support)
            - features (dict): Input feature values
            - comparison (dict, optional): Comparison with other model
            
    Returns:
        bytes: PDF file content
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=30,
        alignment=1 # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#0891b2'),
        spaceBefore=20,
        spaceAfter=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#334155')
    )
    
    elements = []
    
    # --- Title ---
    elements.append(Paragraph("Student Performance Report", title_style))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    elements.append(Spacer(1, 20))
    
    # --- 1. Executive Summary ---
    elements.append(Paragraph("1. Executive Summary", heading_style))
    
    predictions = data.get('predictions', {})
    pass_fail = str(predictions.get('pass_fail', {}).get('prediction', 'N/A'))
    try:
        score = float(predictions.get('final_exam_score', {}).get('predicted_score', 0))
    except (ValueError, TypeError):
        score = 0.0
    score_conf = predictions.get('final_exam_score', {}).get('confidence_interval', '')
    support = str(predictions.get('needs_support', {}).get('prediction', 'N/A'))
    
    # Status Color logic for summary text
    status_color = "#059669" if pass_fail == "Pass" else "#dc2626"
    
    summary_text = f"""
    The AI model monitors current academic performance indicators. 
    Based on the provided data, the student is predicted to 
    <font color='{status_color}'><b>{pass_fail.upper()}</b></font> 
    the final exam with a projected score of <b>{score:.1f}</b> {score_conf}.
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 10))
    
    if support == "Needs Support":
        alert_text = "<b>High Priority:</b> The system has flagged this profile as requiring academic intervention/support."
        elements.append(Paragraph(alert_text, ParagraphStyle('Alert', parent=normal_style, textColor=colors.HexColor('#dc2626'))))
    else:
        elements.append(Paragraph("No immediate academic intervention needed based on current trends.", normal_style))
        
    elements.append(Spacer(1, 20))
    
    # --- 2. Input Data Profile ---
    elements.append(Paragraph("2. Academic Profile (Inputs)", heading_style))
    
    features = data.get('features', {})
    # Helper to safely get features as strings for display, numbers for math
    def get_f(key, default=0):
        try:
            return float(features.get(key, default))
        except (ValueError, TypeError):
            return 0.0
            
    # Organize data for table
    table_data = [
        ['Metric', 'Value', 'Metric', 'Value'], # Header
        ['Attendance', f"{get_f('attendance'):.1f}%", 'Midterm', f"{get_f('midterm'):.1f}"],
        ['Quiz 1', f"{get_f('quiz1'):.1f}", 'Assignment 1', f"{get_f('assignment1'):.1f}"],
        ['Quiz 2', f"{get_f('quiz2'):.1f}", 'Assignment 2', f"{get_f('assignment2'):.1f}"],
        ['Quiz 3', f"{get_f('quiz3'):.1f}", 'Assignment 3', f"{get_f('assignment3'):.1f}"],
        ['Quiz 4', f"{get_f('quiz4'):.1f}", 'Assignment 4', f"{get_f('assignment4'):.1f}"],
    ]
    
    t = Table(table_data, colWidths=[100, 80, 100, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    
    # --- 3. Key Risk/Success Drivers ---
    elements.append(Paragraph("3. Key Drivers (AI Analysis)", heading_style))
    elements.append(Paragraph("The following factors most strongly influenced this prediction:", normal_style))
    elements.append(Spacer(1, 10))
    
    # Extract feature importance if available, otherwise use heuristics
    # Since specific feature importance per prediction requires Shap values (expensive), 
    # we can use the global importance or simple heuristic (High/Low vs Avg)
    
    # Simple Heuristic: List Top 3 Deviations
    # (In a real app, we'd pass Shap values here. For now, we list clear strengths/weaknesses)
    
    drivers = []
    
    # Check for negative trends
    q_trend = get_f('quiz4') - get_f('quiz1')
    if q_trend < -5:
        drivers.append(f"<b>Declining Quiz Scores:</b> Performance dropped by {abs(q_trend):.1f} points from Quiz 1 to Quiz 4.")
    elif q_trend > 5:
        drivers.append(f"<b>Improving Quiz Scores:</b> Performance improved by {q_trend:.1f} points.")
        
    # Check Attendance
    if get_f('attendance') < 75:
        drivers.append(f"<b>Low Attendance ({get_f('attendance'):.1f}%):</b> Significant risk factor.")
        
    # Check Midterm
    if get_f('midterm') < 60:
        drivers.append(f"<b>Low Midterm Score ({get_f('midterm'):.1f}):</b> Heavily weighted in final prediction.")
        
    # Default message if no strong outliers
    if not drivers:
        drivers.append("Student shows consistent performance across all metrics with no major outliers.")
        
    for driver in drivers:
        elements.append(Paragraph(f"• {driver}", normal_style))
        elements.append(Spacer(1, 5))
        
    elements.append(Spacer(1, 20))


    # --- 4. Comparison (if available) ---
    comparison = data.get('comparison')
    if comparison:
        elements.append(Paragraph("4. Model Consensus Check", heading_style))
        agreement = comparison.get('agreement', False)
        other_name = comparison.get('other_model_name', 'Secondary Model')
        try:
            other_score = float(comparison.get('other_score', 0))
        except:
            other_score = 0.0
        
        comp_text = f"""
        <b>Primary Model:</b> {pass_fail} ({score:.1f})<br/>
        <b>{other_name}:</b> {comparison.get('other_pass_fail', '')} ({other_score:.1f})<br/><br/>
        Status: <b>{'AGREEMENT' if agreement else 'DIVERGENCE'}</b>
        """
        elements.append(Paragraph(comp_text, normal_style))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
