from django.contrib import admin
from .models import UserProfile, Job, Candidate, Interaction

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role')
    search_fields = ('user__username', 'role')

@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ('title', 'job_ID', 'company', 'employmentType')
    search_fields = ('title', 'company', 'job_ID')
    list_filter = ('employmentType', 'experience')

@admin.register(Candidate)
class CandidateAdmin(admin.ModelAdmin):
    list_display = ('name', 'candidate_ID', 'email', 'experience_bucket')
    search_fields = ('name', 'email', 'candidate_ID')
    list_filter = ('experience_bucket', 'contract_preference', 'source')

@admin.register(Interaction)
class InteractionAdmin(admin.ModelAdmin):
    list_display = ('candidate', 'job', 'type', 'timestamp')
    list_filter = ('type',)
    search_fields = ('candidate__name', 'job__title')
