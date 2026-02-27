"""
Forensic Report Generator for Audio Analysis

This module generates comprehensive forensic reports for deepfake audio detection.
"""

from datetime import datetime
from typing import Dict, Optional
import hashlib
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from io import BytesIO


class ForensicReportGenerator:
    """
    Generates detailed forensic reports for audio analysis results.
    """
    
    def __init__(self):
        """Initialize the forensic report generator."""
        self.report_version = "1.0"
    
    def generate_report(
        self,
        prediction: Dict,
        transcript: str,
        audio_metrics: Dict,
        filename: str,
        file_hash: Optional[str] = None
    ) -> Dict:
        """
        Generate a comprehensive forensic report.
        
        Args:
            prediction: Prediction results from the model
            transcript: Transcribed text from audio
            audio_metrics: Audio technical metrics
            filename: Original filename
            file_hash: SHA-256 hash of the audio file
            
        Returns:
            Dictionary containing the complete forensic report
        """
        timestamp = datetime.now()
        
        # Determine authenticity assessment
        confidence = prediction['confidence']
        label = prediction['label']
        
        if label == 'real':
            authenticity = "AUTHENTIC"
            risk_level = "LOW" if confidence > 0.9 else "MODERATE"
        else:
            authenticity = "SYNTHETIC/MANIPULATED"
            risk_level = "HIGH" if confidence > 0.9 else "MODERATE"
        
        # Generate detailed analysis
        report = {
            'metadata': {
                'report_id': self._generate_report_id(filename, timestamp),
                'report_version': self.report_version,
                'generated_at': timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'analysis_tool': 'EchoShield v1.0',
                'model': 'BiLSTM Deep Learning Model'
            },
            'file_information': {
                'filename': filename,
                'file_hash_sha256': file_hash or 'N/A',
                'file_size': self._format_file_size(audio_metrics.get('file_size', 0)),
                'duration': f"{audio_metrics.get('duration', 0):.2f} seconds",
                'sample_rate': f"{audio_metrics.get('sample_rate', 0)} Hz",
                'format': self._get_file_format(filename)
            },
            'analysis_results': {
                'authenticity_assessment': authenticity,
                'confidence_score': f"{confidence * 100:.2f}%",
                'risk_level': risk_level,
                'prediction_label': label.upper(),
                'real_probability': f"{prediction['probabilities']['real'] * 100:.2f}%",
                'fake_probability': f"{prediction['probabilities']['fake'] * 100:.2f}%"
            },
            'technical_analysis': {
                'peak_amplitude': f"{audio_metrics.get('peak_amplitude', 0):.4f}",
                'rms_level': f"{audio_metrics.get('rms_level', 0):.1f} dB",
                'dynamic_range': f"{audio_metrics.get('dynamic_range', 0):.1f} dB",
                'zero_crossing_rate': f"{audio_metrics.get('zero_crossings', 0)} per second",
                'spectral_centroid': f"{audio_metrics.get('spectral_centroid', 0)} Hz",
                'noise_floor': f"{audio_metrics.get('noise_floor', 0):.1f} dB"
            },
            'transcript': {
                'text': transcript,
                'language': 'Auto-detected',
                'word_count': len(transcript.split()) if not transcript.startswith('[') else 0
            },
            'indicators': self._generate_indicators(prediction, audio_metrics),
            'recommendations': self._generate_recommendations(label, confidence, risk_level),
            'disclaimer': (
                "This analysis is provided for informational purposes only. "
                "While our AI model achieves high accuracy, no automated system is perfect. "
                "This report should be used as part of a comprehensive verification process "
                "and not as the sole basis for critical decisions. "
                "Always verify with multiple sources and expert analysis when necessary."
            )
        }
        
        return report
    
    def generate_pdf(self, report: Dict) -> BytesIO:
        """
        Generate a PDF version of the forensic report.
        
        Args:
            report: Report dictionary from generate_report()
            
        Returns:
            BytesIO buffer containing the PDF
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        # Container for PDF elements
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a2e'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#16213e'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        normal_style = styles['Normal']
        
        # Title
        elements.append(Paragraph("FORENSIC AUDIO ANALYSIS REPORT", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Metadata Section
        elements.append(Paragraph("Report Metadata", heading_style))
        metadata_data = [
            ['Report ID:', report['metadata']['report_id']],
            ['Generated:', report['metadata']['generated_at']],
            ['Analysis Tool:', report['metadata']['analysis_tool']],
            ['Model:', report['metadata']['model']],
            ['Report Version:', report['metadata']['report_version']]
        ]
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4.5*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(metadata_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # File Information
        elements.append(Paragraph("File Information", heading_style))
        file_data = [
            ['Filename:', report['file_information']['filename']],
            ['File Hash (SHA-256):', report['file_information']['file_hash_sha256']],
            ['File Size:', report['file_information']['file_size']],
            ['Duration:', report['file_information']['duration']],
            ['Sample Rate:', report['file_information']['sample_rate']],
            ['Format:', report['file_information']['format']]
        ]
        file_table = Table(file_data, colWidths=[2*inch, 4.5*inch])
        file_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(file_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Analysis Results
        elements.append(Paragraph("Analysis Results", heading_style))
        auth_color = colors.HexColor('#28a745') if report['analysis_results']['authenticity_assessment'] == 'AUTHENTIC' else colors.HexColor('#dc3545')
        
        results_data = [
            ['Authenticity Assessment:', report['analysis_results']['authenticity_assessment']],
            ['Confidence Score:', report['analysis_results']['confidence_score']],
            ['Risk Level:', report['analysis_results']['risk_level']],
            ['Prediction Label:', report['analysis_results']['prediction_label']],
            ['Real Probability:', report['analysis_results']['real_probability']],
            ['Fake Probability:', report['analysis_results']['fake_probability']]
        ]
        results_table = Table(results_data, colWidths=[2*inch, 4.5*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('BACKGROUND', (1, 0), (1, 0), auth_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(results_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Technical Analysis
        elements.append(Paragraph("Technical Analysis", heading_style))
        tech_data = [
            ['Peak Amplitude:', report['technical_analysis']['peak_amplitude']],
            ['RMS Level:', report['technical_analysis']['rms_level']],
            ['Dynamic Range:', report['technical_analysis']['dynamic_range']],
            ['Zero Crossing Rate:', report['technical_analysis']['zero_crossing_rate']],
            ['Spectral Centroid:', report['technical_analysis']['spectral_centroid']],
            ['Noise Floor:', report['technical_analysis']['noise_floor']]
        ]
        tech_table = Table(tech_data, colWidths=[2*inch, 4.5*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(tech_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Transcript
        elements.append(Paragraph("Transcript", heading_style))
        transcript_text = report['transcript']['text']
        if len(transcript_text) > 500:
            transcript_text = transcript_text[:500] + "..."
        elements.append(Paragraph(f"<i>{transcript_text}</i>", normal_style))
        elements.append(Paragraph(f"<b>Word Count:</b> {report['transcript']['word_count']}", normal_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Indicators
        elements.append(Paragraph("Detection Indicators", heading_style))
        for indicator in report['indicators']:
            elements.append(Paragraph(f"• {indicator}", normal_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        elements.append(Paragraph("Recommendations", heading_style))
        for recommendation in report['recommendations']:
            elements.append(Paragraph(f"• {recommendation}", normal_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Disclaimer
        elements.append(Paragraph("Disclaimer", heading_style))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=normal_style,
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_JUSTIFY
        )
        elements.append(Paragraph(report['disclaimer'], disclaimer_style))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
    
    def _generate_report_id(self, filename: str, timestamp: datetime) -> str:
        """Generate a unique report ID."""
        data = f"{filename}{timestamp.isoformat()}".encode()
        return f"FR-{hashlib.md5(data).hexdigest()[:12].upper()}"
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def _get_file_format(self, filename: str) -> str:
        """Extract file format from filename."""
        ext = os.path.splitext(filename)[1].upper()
        return ext[1:] if ext else 'Unknown'
    
    def _generate_indicators(self, prediction: Dict, audio_metrics: Dict) -> list:
        """Generate list of detection indicators."""
        indicators = []
        
        label = prediction['label']
        confidence = prediction['confidence']
        
        if label == 'fake':
            indicators.append(f"AI model detected synthetic patterns with {confidence*100:.1f}% confidence")
            indicators.append("Audio exhibits characteristics consistent with AI-generated speech")
            
            # Technical indicators
            if audio_metrics.get('dynamic_range', 0) < 20:
                indicators.append("Low dynamic range may indicate audio processing/synthesis")
            if audio_metrics.get('noise_floor', 0) < -50:
                indicators.append("Unusually low noise floor suggests digital generation")
        else:
            indicators.append(f"AI model classified audio as authentic with {confidence*100:.1f}% confidence")
            indicators.append("Audio characteristics consistent with natural human speech")
            indicators.append("No significant synthetic patterns detected")
        
        return indicators
    
    def _generate_recommendations(self, label: str, confidence: float, risk_level: str) -> list:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if label == 'fake':
            recommendations.append("Exercise caution when using or sharing this audio")
            recommendations.append("Verify the source and context of this audio file")
            recommendations.append("Consider additional forensic analysis if used for critical decisions")
            if confidence < 0.8:
                recommendations.append("Moderate confidence suggests seeking expert verification")
        else:
            recommendations.append("Audio appears authentic based on AI analysis")
            if confidence < 0.8:
                recommendations.append("Moderate confidence suggests additional verification may be beneficial")
            recommendations.append("Always verify audio authenticity through multiple methods when possible")
        
        recommendations.append("Keep original audio file for future reference")
        recommendations.append("Document the chain of custody if used for legal purposes")
        
        return recommendations
