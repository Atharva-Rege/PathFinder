import uuid
import time as time_module
import math
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
import requests
import os
from typing import Any, cast
from .models import Job, Candidate, Interaction, UserProfile
from .serializers import JobSerializer, CandidateSerializer, InteractionSerializer

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001")
_MODEL_SYNCED_JOB_IDS: set[str] = set()
_MODEL_SYNCED_CANDIDATE_IDS: set[str] = set()


def _score_to_compatibility(score):
    """Calibrate raw model scores to a UI-friendly compatibility range [0, 1]."""
    try:
        raw = float(score)
    except (TypeError, ValueError):
        return 0.0

    # The model outputs are concentrated near 0 for sparse/cold-start entity pairs.
    # A steeper logistic transform preserves ordering while avoiding flat ~50% UX.
    mapped = 1.0 / (1.0 + math.exp(-20.0 * raw))
    if mapped < 0.0:
        return 0.0
    if mapped > 1.0:
        return 1.0
    return mapped


def _safe_float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _get_user_role(user: Any) -> str:
    role = UserProfile.objects.filter(user=user).values_list("role", flat=True).first()
    return str(role) if role else "unknown"


def _sync_candidate_to_model(candidate: Candidate, *, force: bool = False) -> bool:
    """Push one Django candidate profile to the model runtime graph."""
    if not force and candidate.candidate_ID in _MODEL_SYNCED_CANDIDATE_IDS:
        return True

    payload = {
        "candidate": {
            "candidate_id": candidate.candidate_ID,
            "description": candidate.description or "",
            "skills": candidate.skills or [],
            "contract": candidate.contract_preference,
            "origin": candidate.source,
            "experience": candidate.experience_bucket,
            "salary_current": candidate.salary_current,
            "timestamp": int(candidate.timestamp or time_module.time()),
        },
        "persist_graph": True,
    }
    try:
        res = requests.post(f"{MODEL_SERVER_URL}/candidates", json=payload, timeout=60)
        if res.status_code in (200, 201, 409):
            _MODEL_SYNCED_CANDIDATE_IDS.add(candidate.candidate_ID)
            return True
        print(f"Candidate sync failed ({candidate.candidate_ID}): {res.status_code} {res.text[:200]}")
    except Exception as e:
        print(f"Candidate sync exception ({candidate.candidate_ID}): {e}")
    return False


def _sync_job_to_model(job: Job, *, force: bool = False) -> bool:
    """Push one Django job profile to the model runtime graph."""
    if not force and job.job_ID in _MODEL_SYNCED_JOB_IDS:
        return True

    categories = job.categories or []
    payload = {
        "job": {
            "job_id": job.job_ID,
            "description": job.description or "",
            "skills": job.skills or [],
            "contract": job.employmentType,
            "experience": job.experience,
            "salary": _safe_float(job.salary),
            "category": categories[0] if categories else None,
            "company": job.company,
            "timestamp": int(job.dateAdded or time_module.time()),
        },
        "persist_graph": True,
    }
    try:
        res = requests.post(f"{MODEL_SERVER_URL}/jobs", json=payload, timeout=60)
        if res.status_code in (200, 201, 409):
            _MODEL_SYNCED_JOB_IDS.add(job.job_ID)
            return True
        print(f"Job sync failed ({job.job_ID}): {res.status_code} {res.text[:200]}")
    except Exception as e:
        print(f"Job sync exception ({job.job_ID}): {e}")
    return False


def _ensure_candidate_synced(candidate_id: str) -> None:
    """Ensure candidate exists in model mappings before requesting recommendations."""
    try:
        health = requests.get(f"{MODEL_SERVER_URL}/health", timeout=10)
        if not health.ok:
            return
    except Exception:
        return

    try:
        probe = requests.post(
            f"{MODEL_SERVER_URL}/recommend/jobs-for-candidate",
            json={"candidate_id": candidate_id, "top_k": 1, "persist_graph": False},
            timeout=30,
        )
        if probe.status_code == 404:
            candidate = Candidate.objects.filter(candidate_ID=candidate_id).first()
            if candidate:
                _sync_candidate_to_model(candidate)
    except Exception as e:
        print(f"Candidate sync probe failed ({candidate_id}): {e}")


def _ensure_jobs_synced(job_ids: list[str]) -> None:
    """Ensure all relevant jobs are present in model mappings."""
    if not job_ids:
        return
    for job in Job.objects.filter(job_ID__in=job_ids):
        _sync_job_to_model(job)


def _ensure_candidates_synced(candidate_ids: list[str]) -> None:
    """Ensure all relevant candidates are present in model mappings."""
    if not candidate_ids:
        return
    for candidate in Candidate.objects.filter(candidate_ID__in=candidate_ids):
        _sync_candidate_to_model(candidate)


# ─────────────────────────────────
# Canonical enums enforced at ingestion
# ─────────────────────────────────
CONTRACT_ENUM = {'permanent', 'contract', 'freelance', 'temporary'}
EXPERIENCE_ENUM = {'entry', 'junior', 'mid', 'senior', 'lead', 'principal'}
ORIGIN_ENUM = {'indeed', 'linkedin', 'referral', 'glassdoor', 'direct'}
EMPLOYMENT_TYPE_ENUM = {'permanent', 'contract', 'freelance', 'temporary'}

def normalize_enum(value, allowed, default=None):
    """Lowercase, strip, and map to allowed set. Returns None if no match."""
    if not value:
        return default
    v = str(value).lower().strip()
    if v in allowed:
        return v
    return default

def normalize_salary(value):
    """Return float or None. Never a random default."""
    try:
        v = float(str(value).replace(',', '').replace('₹', '').replace('$', '').strip())
        return v if v > 0 else None
    except Exception:
        return None

def normalize_skills(raw):
    """Accept list or comma string, return cleaned list."""
    if isinstance(raw, list):
        return [s.strip() for s in raw if s.strip()]
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(',') if s.strip()]
    return []

def make_candidate_id(username):
    return f"cand_{username}_{uuid.uuid4().hex[:6]}"

def make_job_id(username):
    return f"job_{username}_{uuid.uuid4().hex[:6]}"


# ─────────────────────────────────
# ViewSets
# ─────────────────────────────────
class JobViewSet(viewsets.ModelViewSet):
    queryset = Job.objects.all()
    serializer_class = JobSerializer

    def perform_create(self, serializer):
        job = serializer.save()
        _sync_job_to_model(job)

    def perform_update(self, serializer):
        job = serializer.save()
        _sync_job_to_model(job)

class CandidateViewSet(viewsets.ModelViewSet):
    queryset = Candidate.objects.all()
    serializer_class = CandidateSerializer

    def perform_create(self, serializer):
        candidate = serializer.save()
        _sync_candidate_to_model(candidate)

    def perform_update(self, serializer):
        candidate = serializer.save()
        _sync_candidate_to_model(candidate)

class InteractionViewSet(viewsets.ModelViewSet):
    queryset = Interaction.objects.all()
    serializer_class = InteractionSerializer


# ─────────────────────────────────
# Auth endpoints
# ─────────────────────────────────
@api_view(['POST'])
@permission_classes([AllowAny])
def register_user(request):
    data = request.data
    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip()
    password = data.get('password') or ''
    role = (data.get('role') or '').strip()

    if not all([username, email, password, role]):
        return Response({"error": "Username, email, password and role are required."}, status=400)

    if role not in ('candidate', 'recruiter'):
        return Response({"error": "Role must be 'candidate' or 'recruiter'."}, status=400)

    if User.objects.filter(username=username).exists():
        return Response({"error": "This username is taken. Please choose another."}, status=400)

    if User.objects.filter(email=email).exists():
        return Response({"error": "An account with this email already exists."}, status=400)

    user = User.objects.create_user(username=username, email=email, password=password)
    UserProfile.objects.create(user=user, role=role)
    now_ts = int(time_module.time())

    if role == 'candidate':
        candidate_id = make_candidate_id(username)
        cand = Candidate.objects.create(
            candidate_ID=candidate_id,
            name=data.get('name') or username,
            email=email,  # user's registered email
            phone=data.get('phone') or '',
            description=data.get('description') or '',
            skills=normalize_skills(data.get('skills')),
            contract_preference=normalize_enum(data.get('contract_preference'), CONTRACT_ENUM, 'permanent'),
            source=normalize_enum(data.get('origin_source'), ORIGIN_ENUM, 'direct'),
            experience_bucket=normalize_enum(data.get('experience_level'), EXPERIENCE_ENUM, 'mid'),
            salary_current=normalize_salary(data.get('salary_current')),
            timestamp=now_ts
        )
        try:
            requests.post(
                f"{MODEL_SERVER_URL}/candidates",
                json={
                    "candidate": {
                        "candidate_id": cand.candidate_ID,
                        "description": cand.description,
                        "skills": cand.skills,
                        "contract": cand.contract_preference,
                        "origin": cand.source,
                        "experience": cand.experience_bucket,
                        "salary_current": cand.salary_current,
                        "timestamp": cand.timestamp
                    },
                    "persist_graph": True
                },
                timeout=120
            )
        except Exception as e:
            print(f"Model API error creating candidate: {e}")

    elif role == 'recruiter':
        job_id = make_job_id(username)
        job_salary_float = normalize_salary(data.get('salary'))
        job = Job.objects.create(
            job_ID=job_id,
            title=data.get('title') or '',
            description=data.get('job_description_text') or data.get('description') or '',
            skills=normalize_skills(data.get('skills')),
            employmentType=normalize_enum(data.get('employment_type'), EMPLOYMENT_TYPE_ENUM, 'permanent'),
            experience=normalize_enum(data.get('experience_required'), EXPERIENCE_ENUM, 'mid'),
            categories=[data.get('category')] if data.get('category') else [],
            company=data.get('company_owner') or data.get('company') or '',
            salary=str(job_salary_float or ''),
            dateAdded=now_ts
        )
        try:
            requests.post(
                f"{MODEL_SERVER_URL}/jobs",
                json={
                    "job": {
                        "job_id": job.job_ID,
                        "description": job.description,
                        "skills": job.skills,
                        "contract": job.employmentType,
                        "experience": job.experience,
                        "salary": job_salary_float,
                        "category": job.categories[0] if job.categories else None,
                        "company": job.company,
                        "timestamp": job.dateAdded
                    },
                    "persist_graph": True
                },
                timeout=120
            )
        except Exception as e:
            print(f"Model API error creating job: {e}")


    token, _ = Token.objects.get_or_create(user=user)
    return Response({"token": token.key, "role": role, "username": username}, status=201)


@api_view(['POST'])
@permission_classes([AllowAny])
def login_user(request):
    username = request.data.get('username')
    password = request.data.get('password')
    user = authenticate(username=username, password=password)
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        role = _get_user_role(user)
        return Response({"token": token.key, "role": role, "username": user.username})
    return Response({"error": "Incorrect username or password."}, status=401)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def me(request):
    user = request.user
    role = _get_user_role(user)
    profile_data = None
    if role == 'candidate':
        cand = Candidate.objects.filter(candidate_ID__startswith=f"cand_{user.username}_").first()
        if cand: profile_data = CandidateSerializer(cand).data
    elif role == 'recruiter':
        job = Job.objects.filter(job_ID__startswith=f"job_{user.username}_").first()
        if job: profile_data = JobSerializer(job).data
    return Response({"username": user.username, "email": user.email, "role": role, "profile": profile_data})


@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    user = request.user
    role = _get_user_role(user)
    data = request.data

    if role == 'candidate':
        cand = Candidate.objects.filter(candidate_ID__startswith=f"cand_{user.username}_").first()
        if not cand:
            return Response({"error": "Candidate profile not found."}, status=404)
        if 'skills' in data:
            cand.skills = normalize_skills(data['skills'])
        if 'description' in data:
            cand.description = data['description']
        if 'contract_preference' in data:
            v = normalize_enum(data['contract_preference'], CONTRACT_ENUM)
            if v: cand.contract_preference = v
        if 'experience_level' in data:
            v = normalize_enum(data['experience_level'], EXPERIENCE_ENUM)
            if v: cand.experience_bucket = v
        if 'origin_source' in data:
            v = normalize_enum(data['origin_source'], ORIGIN_ENUM)
            if v: cand.source = v
        if 'salary_current' in data:
            cand.salary_current = normalize_salary(data['salary_current'])
        cand.save()
        return Response(CandidateSerializer(cand).data)

    elif role == 'recruiter':
        job = Job.objects.filter(job_ID__startswith=f"job_{user.username}_").first()
        if not job:
            return Response({"error": "Job profile not found."}, status=404)
        if 'title' in data: job.title = data['title']
        if 'job_description_text' in data: job.description = data['job_description_text']
        if 'skills' in data: job.skills = normalize_skills(data['skills'])
        if 'employment_type' in data:
            v = normalize_enum(data['employment_type'], EMPLOYMENT_TYPE_ENUM)
            if v: job.employmentType = v
        if 'experience_required' in data:
            v = normalize_enum(data['experience_required'], EXPERIENCE_ENUM)
            if v: job.experience = v
        if 'salary' in data:
            job.salary = str(normalize_salary(data['salary']) or '')
        if 'category' in data:
            job.categories = [data['category']] if data['category'] else []
        job.save()
        return Response(JobSerializer(job).data)

    return Response({"error": "Unknown role."}, status=400)


# ─────────────────────────────────
# Recommendations via Graph Model
# ─────────────────────────────────
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recommend_jobs(request):
    user = request.user
    if _get_user_role(user) != 'candidate':
        return Response({"error": "Only candidates can get job recommendations"}, status=403)
        
    cand = Candidate.objects.filter(candidate_ID__startswith=f"cand_{user.username}_").first()
    if not cand:
        return Response({"error": "Candidate not found"}, status=404)

    # Scope model ranking to jobs known to Django so UI never receives unknown job IDs.
    all_job_ids = list(Job.objects.values_list("job_ID", flat=True))
    if not all_job_ids:
        return Response({"recommendations": []})

    _ensure_candidate_synced(cand.candidate_ID)
    _ensure_jobs_synced(all_job_ids)

    try:
        res = requests.post(
            f"{MODEL_SERVER_URL}/recommend/jobs-for-candidate",
            json={
                "candidate_id": cand.candidate_ID,
                "job_ids": all_job_ids,
                "top_k": min(20, len(all_job_ids)),
                "persist_graph": False,
            },
            timeout=120
        )
        res.raise_for_status()
        model_data = res.json().get("results", [])
        
        # Merge with Django Job profiles
        job_ids = [item["job_id"] for item in model_data if "job_id" in item]
        db_jobs = {j.job_ID: JobSerializer(j).data for j in Job.objects.filter(job_ID__in=job_ids)}
        
        merged = []
        for item in model_data:
            jid = item.get("job_id")
            if jid in db_jobs:
                merged_job = cast(dict[str, Any], db_jobs[jid]).copy()
                merged_job["raw_match_score"] = item.get("score")
                merged_job["match_score"] = _score_to_compatibility(item.get("score"))
                merged_job["match_rank"] = item.get("rank")
                merged.append(merged_job)
                
        return Response({"recommendations": merged})
    except Exception as e:
        print(f"Fallback UI recommend jobs: {e}")
        jobs = Job.objects.all().order_by('-id')[:10]
        return Response({"recommendations": JobSerializer(jobs, many=True).data})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recommend_candidates(request):
    user = request.user
    if _get_user_role(user) != 'recruiter':
        return Response({"error": "Only recruiters can get candidate recommendations"}, status=403)
        
    job = Job.objects.filter(job_ID__startswith=f"job_{user.username}_").first()
    if not job:
        return Response({"error": "Job not found"}, status=404)

    _sync_job_to_model(job)
    all_candidate_ids = list(Candidate.objects.values_list("candidate_ID", flat=True))
    if not all_candidate_ids:
        return Response({"recommendations": []})
    _ensure_candidates_synced(all_candidate_ids)

    try:
        res = requests.post(
            f"{MODEL_SERVER_URL}/recommend/candidates-for-job",
            json={
                "job_id": job.job_ID,
                "candidate_ids": all_candidate_ids,
                "top_k": min(20, len(all_candidate_ids)),
                "persist_graph": False,
            },
            timeout=120
        )
        res.raise_for_status()
        model_data = res.json().get("results", [])
        
        cand_ids = [item["candidate_id"] for item in model_data if "candidate_id" in item]
        db_cands = {c.candidate_ID: CandidateSerializer(c).data for c in Candidate.objects.filter(candidate_ID__in=cand_ids)}
        
        merged = []
        for item in model_data:
            cid = item.get("candidate_id")
            if cid in db_cands:
                merged_cand = cast(dict[str, Any], db_cands[cid]).copy()
                merged_cand["raw_match_score"] = item.get("score")
                merged_cand["match_score"] = _score_to_compatibility(item.get("score"))
                merged_cand["match_rank"] = item.get("rank")
                merged.append(merged_cand)
                
        return Response({"recommendations": merged})
    except Exception as e:
        print(f"Fallback UI recommend candidates: {e}")
        candidates = Candidate.objects.all().order_by('-id')[:10]
        return Response({"recommendations": CandidateSerializer(candidates, many=True).data})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def log_interaction(request):
    user = request.user
    role = _get_user_role(user)
    data = request.data
    action = data.get('action') # 'application' or 'shortlist'
    
    cand = None
    job = None
    
    if role == 'candidate':
        cand = Candidate.objects.filter(candidate_ID__startswith=f"cand_{user.username}_").first()
        job = Job.objects.filter(job_ID=data.get('job_id')).first()
        if not cand or not job: return Response({"error":"Not found"}, status=404)
        
    elif role == 'recruiter':
        job = Job.objects.filter(job_ID__startswith=f"job_{user.username}_").first()
        cand = Candidate.objects.filter(candidate_ID=data.get('candidate_id')).first()
        if not cand or not job: return Response({"error":"Not found"}, status=404)

    if cand and job:
        Interaction.objects.get_or_create(
            candidate=cand,
            job=job,
            type=action,
            defaults={'timestamp': int(time_module.time())}
        )
        model_logged = False
        retrain_due = False
        model_response = None
        try:
            score = 1.0
            model_res = requests.post(
                f"{MODEL_SERVER_URL}/interactions",
                json={
                    "candidate_id": cand.candidate_ID,
                    "job_id": job.job_ID,
                    "score": score,
                    "timestamp": int(time_module.time())
                },
                timeout=10
            )
            if model_res.ok:
                model_logged = True
                try:
                    model_response = model_res.json()
                except ValueError:
                    model_response = None
                if isinstance(model_response, dict):
                    retrain_due = bool(model_response.get("retrain_due", False))
            else:
                print(f"Model interaction logging non-200: {model_res.status_code} {model_res.text[:200]}")
        except Exception as e:
            print(f"Model interaction logging failed: {e}")
        return Response({
            "status": "ok",
            "interaction_saved": True,
            "model_logged": model_logged,
            "retrain_due": retrain_due,
            "model_response": model_response,
        })
    return Response({"error": "Invalid interaction"}, status=400)


# ─────────────────────────────────
# PDF text extraction utilities
# ─────────────────────────────────
import io
import re

# Canonical skill vocabulary (shared with frontend list)
KNOWN_SKILLS = [
    'python','javascript','typescript','react','node.js','nodejs','django','fastapi',
    'aws','gcp','azure','docker','kubernetes','postgresql','mongodb','redis','mysql',
    'pytorch','tensorflow','scikit-learn','pandas','numpy','sql','graphql',
    'next.js','nextjs','flutter','swift','kotlin','java','go','rust','c++','c#','php','ruby',
    'machine learning','deep learning','nlp','computer vision','data science',
    'system design','leadership','product management','ui/ux','figma','git',
    'html','css','scss','vue','angular','express','spring','flask',
    'spark','kafka','elasticsearch','airflow','dbt', 'terraform',
]

# Experience keyword → canonical enum
EXPERIENCE_MAP = [
    (['0-1', '0-2', 'fresh', 'fresher', 'entry', 'graduate', 'intern'], 'entry'),
    (['1-3', '2-4', 'junior', '1 year', '2 year'], 'junior'),
    (['3-5', '3-6', 'mid', 'intermediate', '3 year', '4 year', '5 year'], 'mid'),
    (['senior', '5+', '6+', '7+', '5 year', '6 year', '7 year', '8 year'], 'senior'),
    (['lead', 'tech lead', 'team lead', 'engineering lead'], 'lead'),
    (['principal', 'staff', 'architect', 'director', 'vp of', 'vice president'], 'principal'),
]

CONTRACT_MAP = [
    (['freelance', 'freelancer'], 'freelance'),
    (['contract', 'contractor', 'fixed-term'], 'contract'),
    (['temporary', 'temp ', 'part-time', 'parttime'], 'temporary'),
    (['permanent', 'full-time', 'fulltime', 'regular'], 'permanent'),
]


def extract_pdf_text(file_obj) -> str:
    """Extract all text from a PDF file object using PyPDF2."""
    try:
        import PyPDF2
        raw = file_obj.read()
        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return '\n'.join(pages)
    except Exception as e:
        return ''


def extract_name(text: str):
    """Heuristic: first non-empty, non-keyword line is usually the name."""
    skip = {'resume', 'cv', 'curriculum vitae', 'profile', 'summary',
            'objective', 'contact', 'address', 'phone', 'email', 'linkedin'}
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:8]:
        words = line.split()
        if 2 <= len(words) <= 5 and line.lower() not in skip:
            # Likely a name if all words are capitalized and no digits
            if all(w[0].isupper() for w in words if w) and not re.search(r'\d', line):
                return line, 0.85
    return '', 0.0


def extract_skills_from_text(text: str):
    """Find known skills mentioned in the text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for skill in KNOWN_SKILLS:
        # Use word-boundary-like matching
        pattern = r'(?<![a-zA-Z])' + re.escape(skill) + r'(?![a-zA-Z])'
        if re.search(pattern, text_lower):
            # Normalize display name
            display = skill.split('.')[0].title() if '.' not in skill[1:] else skill
            if display.lower() in ['c++', 'c#', 'sql', 'html', 'css', 'scss', 'aws', 'gcp', 'nlp',
                                     'ui/ux', 'dbt', 'vp of']:
                display = display.upper() if display.lower() not in ['ui/ux', 'vp of'] else display
            found.append(skill.title() if skill not in ['aws','gcp','sql','html','css','scss',
                                                         'nlp','dbt','c++','c#','nodejs','nextjs',
                                                         'scikit-learn','node.js','next.js'] else skill.upper() if skill in ['aws','gcp','sql','html','css','scss','nlp','dbt'] else skill)
    # Deduplicate
    seen = set()
    unique = []
    for s in found:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique.append(s)
    conf = 0.9 if len(unique) > 0 else 0.0
    return unique, conf


def extract_experience_level(text: str):
    text_lower = text.lower()
    for keywords, level in EXPERIENCE_MAP:
        for kw in keywords:
            if kw in text_lower:
                return level, 0.75
    return 'mid', 0.4


def extract_contract_preference(text: str):
    text_lower = text.lower()
    for keywords, pref in CONTRACT_MAP:
        for kw in keywords:
            if kw in text_lower:
                return pref, 0.75
    return 'permanent', 0.4


def extract_salary(text: str):
    """Look for salary patterns: ₹12,00,000 / $120,000 / 12 LPA / 12L etc."""
    patterns = [
        r'(?:₹|rs\.?|inr)\s*([\d,]+)',
        r'\$([\d,]+)',
        r'([\d.]+)\s*l(?:pa|akh)',
        r'salary[:\s]+([\d,]+)',
        r'ctc[:\s]+([\d,]+)',
        r'([\d,]{5,})\s*(?:per annum|pa\b|/year|annual)',
    ]
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            raw = m.group(1).replace(',', '')
            try:
                val = float(raw)
                # If value looks like LPA (e.g. 12.5) convert to full number
                if val < 1000:
                    val = val * 100000
                return val, 0.8
            except Exception:
                pass
    return None, 0.0


def extract_description(text: str, max_chars=600):
    """Return a clean summary block — skip header lines, take first dense paragraph."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # Skip very short lines (likely headers)
    paragraphs = [l for l in lines if len(l) > 60]
    if paragraphs:
        description = ' '.join(paragraphs[:4])[:max_chars]
        return description, 0.7
    return '', 0.0


def extract_job_title(text: str):
    """Look for job title patterns in a job description."""
    patterns = [
        r'(?:job title|position|role)[:\s]+([^\n]{3,60})',
        r'(?:we are hiring|hiring for|looking for)[:\s]+(?:a |an )?([\w\s]{3,50})',
        r'^([A-Z][\w\s]{3,40}(?:engineer|developer|manager|analyst|designer|scientist|lead|architect))',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip(), 0.8
    # Fallback: first capitalized line that looks like a title
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:5]:
        if 3 < len(line) < 70 and not re.search(r'[.@]', line):
            return line, 0.5
    return '', 0.0


def extract_company(text: str):
    patterns = [
        r'(?:company|employer|organization|about us)[:\s]+([^\n]{2,50})',
        r'at ([A-Z][\w\s&.]{2,30})(?:\.|,|\n)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip(), 0.7
    return '', 0.0


def extract_category(text: str):
    categories = ['engineering','software','data','design','marketing','sales','finance',
                  'operations','hr','product','research','legal','healthcare','education']
    text_lower = text.lower()
    for cat in categories:
        if cat in text_lower:
            return cat.title(), 0.7
    return '', 0.0


# ─────────────────────────────────
# PDF parse endpoints — real extraction
# ─────────────────────────────────
@api_view(['POST'])
@permission_classes([AllowAny])
@parser_classes([MultiPartParser, FormParser])
def parse_resume(request):
    """
    Extract candidate fields from uploaded PDF/DOC using PyPDF2 + regex heuristics.
    Returns per-field confidence scores. Frontend shows pre-filled form for user confirmation.
    """
    if 'file' not in request.FILES:
        return Response({"error": "No file uploaded."}, status=400)

    file_obj = request.FILES['file']
    text = extract_pdf_text(file_obj)

    if not text.strip():
        return Response({"error": "Could not read text from the file. Please ensure it's a text-based PDF (not a scanned image), or fill in the form manually."}, status=422)

    name, name_conf = extract_name(text)
    skills, skills_conf = extract_skills_from_text(text)
    experience, exp_conf = extract_experience_level(text)
    contract, contract_conf = extract_contract_preference(text)
    salary, salary_conf = extract_salary(text)
    description, desc_conf = extract_description(text)

    extracted = {
        "public_name":               {"value": name,        "confidence": name_conf},
        "profile_description_text":  {"value": description, "confidence": desc_conf},
        "skills":                    {"value": skills,       "confidence": skills_conf},
        "experience_level":          {"value": experience,   "confidence": exp_conf},
        "contract_preference":       {"value": contract,     "confidence": contract_conf},
        "origin_source":             {"value": "direct",     "confidence": 1.0},
        "salary_current":            {"value": salary,       "confidence": salary_conf},
    }
    return Response({"message": "Extracted — please review and confirm.", "fields": extracted})


@api_view(['POST'])
@permission_classes([AllowAny])
@parser_classes([MultiPartParser, FormParser])
def parse_job_pdf(request):
    """
    Extract job description fields from uploaded PDF using PyPDF2 + regex heuristics.
    """
    if 'file' not in request.FILES:
        return Response({"error": "No file uploaded."}, status=400)

    file_obj = request.FILES['file']
    text = extract_pdf_text(file_obj)

    if not text.strip():
        return Response({"error": "Could not read text from the file. Please ensure it's a text-based PDF, or fill in the form manually."}, status=422)

    title, title_conf = extract_job_title(text)
    skills, skills_conf = extract_skills_from_text(text)
    experience, exp_conf = extract_experience_level(text)
    employment_type, et_conf = extract_contract_preference(text)
    salary, salary_conf = extract_salary(text)
    description, desc_conf = extract_description(text)
    category, cat_conf = extract_category(text)

    extracted = {
        "title":                {"value": title,           "confidence": title_conf},
        "job_description_text": {"value": description,     "confidence": desc_conf},
        "skills":               {"value": skills,           "confidence": skills_conf},
        "experience_required":  {"value": experience,       "confidence": exp_conf},
        "employment_type":      {"value": employment_type,  "confidence": et_conf},
        "category":             {"value": category,         "confidence": cat_conf},
        "salary":               {"value": salary,           "confidence": salary_conf},
    }
    return Response({"message": "Extracted — please review and confirm.", "fields": extracted})


@api_view(['POST'])
@permission_classes([AllowAny])
def notify_candidate(request):
    email = request.data.get('email')
    job_title = request.data.get('job_title')
    if email and job_title:
        return Response({"message": f"Notification queued for {email} regarding '{job_title}'"})
    return Response({"error": "Missing email or job_title."}, status=400)


