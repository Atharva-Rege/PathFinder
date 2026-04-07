from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'jobs', views.JobViewSet)
router.register(r'candidates', views.CandidateViewSet)
router.register(r'interactions', views.InteractionViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('auth/register/', views.register_user),
    path('auth/login/', views.login_user),
    path('auth/me/', views.me),
    path('auth/update/', views.update_profile),
    path('recommend/jobs/', views.recommend_jobs),
    path('recommend/candidates/', views.recommend_candidates),
    path('parse-resume/', views.parse_resume),
    path('parse-job-pdf/', views.parse_job_pdf),
    path('notify-candidate/', views.notify_candidate),
    path('log-interaction/', views.log_interaction),
]
