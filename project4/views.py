from .utils import simulate_rating_impact
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import io
from reportlab.pdfgen import canvas
from pathlib import Path
import pandas as pd
from .models import Feedback 
import logging

logger = logging.getLogger(__name__)

def project4_landing(request):
    """
    This is the landing page for Project 4.
    It includes a link to download the PDF and a button to start the study.
    """
    return render(request, 'project4/landing.html')


def project4_study(request):
    BASE_DIR = Path(__file__).resolve().parent
    movies_path = BASE_DIR / "data" / "movies.csv"
    
    try:
        movies_df = pd.read_csv(movies_path)
        movie_title_to_id = dict(zip(movies_df['title'], movies_df['movieId']))
        all_titles = sorted(movie_title_to_id.keys())
        logger.info(f"Loaded {len(all_titles)} movies from dataset")
    except Exception as e:
        logger.error(f"Error loading movies: {str(e)}")
        # Fallback movie list for demo
        all_titles = [
            "Toy Story (1995)",
            "Jumanji (1995)", 
            "Grumpier Old Men (1995)",
            "Waiting to Exhale (1995)",
            "Father of the Bride Part II (1995)"
        ]
        movie_title_to_id = {title: i+1 for i, title in enumerate(all_titles)}

    selected_movie = None
    prediction_impact = None
    feedback_submitted = False
    error_message = None

    if request.method == "POST":
        selected_movie = request.POST.get("movie_title", "").strip()
        logger.info(f"User selected movie: '{selected_movie}'")
        
        if 'submit_feedback' in request.POST:
            # Save feedback
            try:
                Feedback.objects.create(
                    movie_title=selected_movie if selected_movie else "Unknown",
                    helpfulness=request.POST.get("feedback", "None"),
                    comments=request.POST.get("comments", "")
                )
                feedback_submitted = True
                logger.info(f"Feedback saved for movie: {selected_movie}")
            except Exception as e:
                logger.error(f"Error saving feedback: {str(e)}")
                error_message = "Error saving feedback. Please try again."

        # Calculate prediction impact if a movie is selected
        if selected_movie and selected_movie in movie_title_to_id:
            try:
                logger.info(f"Calculating prediction impact for: {selected_movie}")
                prediction_impact = simulate_rating_impact(selected_movie, movie_title_to_id)
                
                # Debug: log the results
                for rating, movies in prediction_impact.items():
                    logger.debug(f"{rating}: {len(movies)} movies - {movies}")
                    
                # Validate that we got different results
                if prediction_impact:
                    all_recommendations = []
                    for movies in prediction_impact.values():
                        all_recommendations.extend(movies)
                    
                    unique_recommendations = len(set(all_recommendations))
                    total_recommendations = len(all_recommendations)
                    
                    logger.info(f"Generated {unique_recommendations} unique recommendations out of {total_recommendations} total")
                    
                    # Check if results seem too similar (potential issue)
                    if unique_recommendations < total_recommendations * 0.7:
                        logger.warning("Recommendations seem too similar - check model diversity")
                        
            except Exception as e:
                logger.error(f"Error calculating prediction impact: {str(e)}")
                error_message = f"Error calculating recommendations for '{selected_movie}'. Please try a different movie."
                prediction_impact = None
        elif selected_movie and selected_movie not in movie_title_to_id:
            error_message = f"Movie '{selected_movie}' not found in our database."
            logger.warning(f"Movie not found: {selected_movie}")

    # Add some analytics data for the other tabs
    try:
        recent_feedback = Feedback.objects.all().order_by('-submitted_at')[:10]
        total_submissions = Feedback.objects.count()
        
        # Calculate some basic stats
        feedback_stats = {
            'total_submissions': total_submissions,
            'recent_feedback': recent_feedback,
        }
    except Exception as e:
        logger.error(f"Error fetching analytics: {str(e)}")
        feedback_stats = {
            'total_submissions': 0,
            'recent_feedback': [],
        }

    context = {
        'movie_title': selected_movie,
        'prediction_impact': prediction_impact,
        'all_titles': all_titles,
        'feedback_submitted': feedback_submitted,
        'error_message': error_message,
        **feedback_stats
    }
    
    return render(request, 'project4/study.html', context)


# Add a debug endpoint to test the prediction system
def debug_predictions(request):
    """Debug endpoint to test prediction generation"""
    if request.method == 'GET':
        movie_title = request.GET.get('movie', 'Toy Story (1995)')
        
        BASE_DIR = Path(__file__).resolve().parent
        movies_path = BASE_DIR / "data" / "movies.csv"
        
        try:
            movies_df = pd.read_csv(movies_path)
            movie_title_to_id = dict(zip(movies_df['title'], movies_df['movieId']))
        except:
            movie_title_to_id = {"Toy Story (1995)": 1}
        
        predictions = simulate_rating_impact(movie_title, movie_title_to_id)
        
        return JsonResponse({
            'movie': movie_title,
            'predictions': predictions,
            'debug_info': {
                'total_movies_in_db': len(movie_title_to_id),
                'movie_found': movie_title in movie_title_to_id,
                'movie_id': movie_title_to_id.get(movie_title)
            }
        })


def project4_download_pdf(request):
    """
    Generate PDF with explanation
    """
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica", 14)
    p.drawString(100, 800, "Project 4 - Method and User Study Description")
    p.drawString(100, 770, "This PDF contains Task 1 and Task 2 content.")
    p.drawString(100, 740, "Recommendation Impact Visualization Study")
    p.showPage()
    p.save()

    buffer.seek(0)
    return HttpResponse(buffer, content_type='application/pdf')


def project4_feedback_review(request):
    feedbacks = Feedback.objects.all().order_by('-id')
    return render(request, 'project4/feedback_review.html', {
        'feedbacks': feedbacks,
        'page_title': "Feedback Review"
    })


def project4_export_feedback(request):
    """
    Export feedback data as CSV
    """
    import csv
    from django.http import HttpResponse
    from datetime import datetime
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="feedback_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['ID', 'Movie Title', 'Helpfulness', 'Comments', 'Submitted At'])
    
    feedbacks = Feedback.objects.all().order_by('-submitted_at')
    for feedback in feedbacks:
        writer.writerow([
            feedback.id,
            feedback.movie_title,
            feedback.helpfulness,
            feedback.comments,
            feedback.submitted_at.strftime('%Y-%m-%d %H:%M:%S')
        ])
    
    return response