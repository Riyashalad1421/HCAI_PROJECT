from django.shortcuts import render
from django.http import HttpResponse
import io
from reportlab.pdfgen import canvas
from pathlib import Path
import pandas as pd
#from .utils import simulate_rating_impact


def project4_landing(request):
    """
    This is the landing page for Project 4.
    It includes a link to download the PDF and a button to start the study.
    """
    return render(request, 'project4/landing.html')


def project4_study(request):
    """
    Simulated user interaction for Project 4 Task 3.
    """
    BASE_DIR = Path(__file__).resolve().parent
    movies_path = BASE_DIR / "data" / "movies.csv"
    movies_df = pd.read_csv(movies_path)
    movie_title_to_id = dict(zip(movies_df['title'], movies_df['movieId']))

    selected_movie = "Inception"
    prediction_impact = simulate_rating_impact(selected_movie, movie_title_to_id)

    return render(request, 'project4/study.html', {
        'movie_title': selected_movie,
        'prediction_impact': prediction_impact
    })

def project4_download_pdf(request):
    """
    Dynamically generate a simple PDF with explanation (for now just placeholder text).
    Later, you can use ReportLab or xhtml2pdf to include Task 1 and Task 2 content.
    """
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica", 14)
    p.drawString(100, 800, "Project 4 - Method and User Study Description")
    p.drawString(100, 770, "This PDF will contain Task 1 and Task 2 content.")
    p.drawString(100, 740, "You can update this content later using a template or static file.")
    p.showPage()
    p.save()

    buffer.seek(0)
    return HttpResponse(buffer, content_type='application/pdf')
