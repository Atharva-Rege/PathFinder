from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=50) # 'candidate' or 'recruiter'

    def __str__(self):
        return f"{self.user.username} - {self.role}"

class Job(models.Model):
    job_ID = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=255)
    description = models.TextField()
    skills = models.JSONField(default=list)
    employmentType = models.CharField(max_length=100, null=True, blank=True)
    experience = models.CharField(max_length=100, null=True, blank=True)
    categories = models.JSONField(default=list)
    company = models.CharField(max_length=255, null=True, blank=True)
    salary = models.CharField(max_length=100, null=True, blank=True)
    dateAdded = models.BigIntegerField()

    def __str__(self):
        return self.title

class Candidate(models.Model):
    candidate_ID = models.CharField(max_length=255, unique=True)
    name = models.CharField(max_length=255)
    email = models.EmailField(null=True, blank=True)
    phone = models.CharField(max_length=30, null=True, blank=True)
    description = models.TextField()
    skills = models.JSONField(default=list)
    contract_preference = models.CharField(max_length=100, null=True, blank=True)
    source = models.CharField(max_length=100, null=True, blank=True)
    experience_bucket = models.CharField(max_length=100, null=True, blank=True)
    salary_current = models.FloatField(null=True, blank=True)
    timestamp = models.BigIntegerField()

    def __str__(self):
        return self.name

class Interaction(models.Model):
    INTERACTION_CHOICES = (
        ('application', 'Application'),
        ('shortlist', 'Shortlist'),
    )
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name='interactions')
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='interactions')
    type = models.CharField(max_length=50, choices=INTERACTION_CHOICES)
    timestamp = models.BigIntegerField()

    class Meta:
        unique_together = ('candidate', 'job', 'type')
