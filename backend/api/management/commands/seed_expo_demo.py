import csv
import os
import time
from pathlib import Path

import requests
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from rest_framework.authtoken.models import Token

from api.models import Candidate, Interaction, Job, UserProfile


DEMO_PAIRS = [
    {
        "track": "Backend Python",
        "job_title": "Backend Python Engineer",
        "job_description": "Build FastAPI and Django microservices with PostgreSQL, Redis, and Docker.",
        "candidate_name": "Aarav Sharma",
        "candidate_description": "Backend engineer focused on Python APIs, SQL modeling, caching, and cloud deployment.",
        "skills": ["python", "django", "fastapi", "postgresql", "redis", "docker", "sql"],
        "contract": "permanent",
        "experience": "mid",
        "category": "engineering",
        "company": "Astra Systems",
        "salary_job": "2200000",
        "salary_current": 1800000,
        "source": "linkedin",
    },
    {
        "track": "Frontend React",
        "job_title": "Frontend React Developer",
        "job_description": "Build React and TypeScript UIs with state management, component systems, and testing.",
        "candidate_name": "Ira Mehta",
        "candidate_description": "Frontend developer with React, TypeScript, accessibility, and performance optimization experience.",
        "skills": ["javascript", "typescript", "react", "html", "css", "next.js", "git"],
        "contract": "permanent",
        "experience": "mid",
        "category": "frontend",
        "company": "Nova Commerce",
        "salary_job": "1900000",
        "salary_current": 1500000,
        "source": "indeed",
    },
    {
        "track": "Data Science",
        "job_title": "Data Scientist",
        "job_description": "Design ML experiments, feature pipelines, and model evaluation for product insights.",
        "candidate_name": "Kabir Rao",
        "candidate_description": "Data scientist specializing in statistics, scikit-learn pipelines, and experiment analysis.",
        "skills": ["python", "pandas", "numpy", "scikit-learn", "sql", "machine learning", "data science"],
        "contract": "permanent",
        "experience": "senior",
        "category": "data",
        "company": "Insight Forge",
        "salary_job": "2600000",
        "salary_current": 2150000,
        "source": "linkedin",
    },
    {
        "track": "DevOps",
        "job_title": "DevOps Engineer",
        "job_description": "Own CI/CD, infrastructure as code, Kubernetes operations, and cloud observability.",
        "candidate_name": "Riya Bansal",
        "candidate_description": "DevOps engineer experienced with AWS, Terraform, Kubernetes, and delivery pipelines.",
        "skills": ["aws", "docker", "kubernetes", "terraform", "python", "git", "redis"],
        "contract": "permanent",
        "experience": "senior",
        "category": "devops",
        "company": "CloudBridge",
        "salary_job": "2800000",
        "salary_current": 2350000,
        "source": "referral",
    },
    {
        "track": "ML Engineering",
        "job_title": "Machine Learning Engineer",
        "job_description": "Train and serve deep learning models with PyTorch and production MLOps workflows.",
        "candidate_name": "Vihaan Kapoor",
        "candidate_description": "ML engineer working on PyTorch training, model serving, and feature engineering.",
        "skills": ["python", "pytorch", "tensorflow", "deep learning", "ml", "docker", "kafka"],
        "contract": "permanent",
        "experience": "mid",
        "category": "ml",
        "company": "Neuron Stack",
        "salary_job": "3000000",
        "salary_current": 2500000,
        "source": "direct",
    },
    {
        "track": "Mobile Flutter",
        "job_title": "Flutter Mobile Developer",
        "job_description": "Build scalable cross-platform apps with Flutter, clean architecture, and API integrations.",
        "candidate_name": "Ananya Joshi",
        "candidate_description": "Mobile engineer focused on Flutter UI architecture, API integration, and performance.",
        "skills": ["flutter", "dart", "javascript", "graphql", "firebase", "git", "ui/ux"],
        "contract": "contract",
        "experience": "mid",
        "category": "mobile",
        "company": "Pulse Apps",
        "salary_job": "1700000",
        "salary_current": 1450000,
        "source": "linkedin",
    },
    {
        "track": "QA Automation",
        "job_title": "QA Automation Engineer",
        "job_description": "Design automation frameworks, API tests, and reliability checks for releases.",
        "candidate_name": "Dev Malhotra",
        "candidate_description": "QA automation specialist for API testing, CI, and stable release pipelines.",
        "skills": ["python", "javascript", "selenium", "pytest", "postman", "sql", "git"],
        "contract": "permanent",
        "experience": "junior",
        "category": "qa",
        "company": "Quality Grid",
        "salary_job": "1600000",
        "salary_current": 1250000,
        "source": "indeed",
    },
    {
        "track": "Product",
        "job_title": "Product Manager",
        "job_description": "Drive roadmap, prioritize features, coordinate engineering and business outcomes.",
        "candidate_name": "Sana Ali",
        "candidate_description": "Product manager with strong execution in discovery, analytics, and cross-functional leadership.",
        "skills": ["product management", "leadership", "sql", "figma", "data science", "system design"],
        "contract": "permanent",
        "experience": "senior",
        "category": "product",
        "company": "Orbit Labs",
        "salary_job": "3200000",
        "salary_current": 2800000,
        "source": "referral",
    },
    {
        "track": "Design",
        "job_title": "UI UX Designer",
        "job_description": "Lead UX research, prototype flows, and production-ready design systems.",
        "candidate_name": "Nisha Verma",
        "candidate_description": "Designer focused on interaction design, design systems, and usability validation.",
        "skills": ["ui/ux", "figma", "css", "html", "product management", "leadership"],
        "contract": "freelance",
        "experience": "mid",
        "category": "design",
        "company": "Canvas Works",
        "salary_job": "1550000",
        "salary_current": 1300000,
        "source": "direct",
    },
    {
        "track": "Java Platform",
        "job_title": "Java Spring Backend Engineer",
        "job_description": "Develop high-throughput Java services, Spring APIs, and resilient distributed systems.",
        "candidate_name": "Arjun Nair",
        "candidate_description": "Backend engineer skilled in Java, Spring, SQL tuning, and system design.",
        "skills": ["java", "spring", "sql", "postgresql", "docker", "kafka", "system design"],
        "contract": "permanent",
        "experience": "senior",
        "category": "backend",
        "company": "Vertex Finance",
        "salary_job": "2900000",
        "salary_current": 2450000,
        "source": "linkedin",
    },
]


class Command(BaseCommand):
    help = "Seed 10 expo candidates and 10 expo recruiters/jobs, then sync them to the model graph."

    def add_arguments(self, parser):
        parser.add_argument(
            "--password",
            default="ExpoDemo@2026",
            help="Password assigned to all seeded users.",
        )
        parser.add_argument(
            "--output",
            default="demo_credentials.csv",
            help="CSV output path for generated credentials (relative to backend/ if not absolute).",
        )
        parser.add_argument(
            "--model-url",
            default=os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:8001"),
            help="Model API base URL used for graph syncing.",
        )
        parser.add_argument(
            "--skip-graph",
            action="store_true",
            help="Seed only Django DB and skip graph API sync calls.",
        )

    def handle(self, *args, **options):
        password = options["password"]
        model_url = options["model_url"].rstrip("/")
        skip_graph = bool(options["skip_graph"])
        now_ts = int(time.time())

        created_candidates = []
        created_jobs = []
        credential_rows = []

        for idx, item in enumerate(DEMO_PAIRS, start=1):
            cand_username = f"expo_cand_{idx:02d}"
            rec_username = f"expo_rec_{idx:02d}"

            candidate_id = f"cand_{cand_username}_001"
            job_id = f"job_{rec_username}_001"

            cand_user = self._upsert_user(
                username=cand_username,
                email=f"{cand_username}@demo.local",
                password=password,
                role="candidate",
            )
            rec_user = self._upsert_user(
                username=rec_username,
                email=f"{rec_username}@demo.local",
                password=password,
                role="recruiter",
            )

            candidate, _ = Candidate.objects.update_or_create(
                candidate_ID=candidate_id,
                defaults={
                    "name": item["candidate_name"],
                    "email": cand_user.email,
                    "phone": f"+91-90000{idx:05d}",
                    "description": item["candidate_description"],
                    "skills": item["skills"],
                    "contract_preference": item["contract"],
                    "source": item["source"],
                    "experience_bucket": item["experience"],
                    "salary_current": item["salary_current"],
                    "timestamp": now_ts,
                },
            )

            job, _ = Job.objects.update_or_create(
                job_ID=job_id,
                defaults={
                    "title": item["job_title"],
                    "description": item["job_description"],
                    "skills": item["skills"],
                    "employmentType": item["contract"],
                    "experience": item["experience"],
                    "categories": [item["category"]],
                    "company": item["company"],
                    "salary": item["salary_job"],
                    "dateAdded": now_ts,
                },
            )

            Interaction.objects.update_or_create(
                candidate=candidate,
                job=job,
                type="shortlist",
                defaults={"timestamp": now_ts},
            )
            Interaction.objects.update_or_create(
                candidate=candidate,
                job=job,
                type="application",
                defaults={"timestamp": now_ts},
            )

            Token.objects.get_or_create(user=cand_user)
            Token.objects.get_or_create(user=rec_user)

            created_candidates.append(candidate)
            created_jobs.append(job)

            credential_rows.append(
                {
                    "role": "candidate",
                    "username": cand_username,
                    "password": password,
                    "email": cand_user.email,
                    "profile_id": candidate_id,
                    "track": item["track"],
                    "paired_id": job_id,
                }
            )
            credential_rows.append(
                {
                    "role": "recruiter",
                    "username": rec_username,
                    "password": password,
                    "email": rec_user.email,
                    "profile_id": job_id,
                    "track": item["track"],
                    "paired_id": candidate_id,
                }
            )

        graph_result = {
            "candidates": 0,
            "jobs": 0,
            "interactions": 0,
            "errors": [],
        }
        if not skip_graph:
            self._sync_to_graph(model_url, created_candidates, created_jobs, graph_result, now_ts)

        output_path = Path(options["output"])
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_credentials_csv(output_path, credential_rows)

        self.stdout.write(self.style.SUCCESS("Expo demo seed completed."))
        self.stdout.write(f"Candidates in DB: {len(created_candidates)}")
        self.stdout.write(f"Jobs in DB: {len(created_jobs)}")
        if skip_graph:
            self.stdout.write("Graph sync: skipped")
        else:
            self.stdout.write(
                f"Graph sync: candidates={graph_result['candidates']} jobs={graph_result['jobs']} interactions={graph_result['interactions']}"
            )
            if graph_result["errors"]:
                self.stdout.write(self.style.WARNING("Graph sync warnings:"))
                for error in graph_result["errors"]:
                    self.stdout.write(f"- {error}")
        self.stdout.write(f"Credentials file: {output_path}")

    def _upsert_user(self, username, email, password, role):
        user, _ = User.objects.get_or_create(username=username, defaults={"email": email})
        if user.email != email:
            user.email = email
        user.set_password(password)
        user.save()
        UserProfile.objects.update_or_create(user=user, defaults={"role": role})
        return user

    def _sync_to_graph(self, model_url, candidates, jobs, graph_result, now_ts):
        for candidate in candidates:
            payload = {
                "candidate": {
                    "candidate_id": candidate.candidate_ID,
                    "description": candidate.description or "",
                    "skills": candidate.skills or [],
                    "contract": candidate.contract_preference,
                    "origin": candidate.source,
                    "experience": candidate.experience_bucket,
                    "salary_current": candidate.salary_current,
                    "timestamp": int(candidate.timestamp or now_ts),
                },
                "persist_graph": True,
                "force_update": True,
            }
            ok, error = self._post_json(f"{model_url}/candidates", payload)
            if ok:
                graph_result["candidates"] += 1
            elif error:
                graph_result["errors"].append(error)

        for job in jobs:
            category = job.categories[0] if job.categories else None
            payload = {
                "job": {
                    "job_id": job.job_ID,
                    "description": job.description or "",
                    "skills": job.skills or [],
                    "contract": job.employmentType,
                    "experience": job.experience,
                    "salary": self._to_float(job.salary),
                    "category": category,
                    "company": job.company,
                    "timestamp": int(job.dateAdded or now_ts),
                },
                "persist_graph": True,
                "force_update": True,
            }
            ok, error = self._post_json(f"{model_url}/jobs", payload)
            if ok:
                graph_result["jobs"] += 1
            elif error:
                graph_result["errors"].append(error)

        for candidate, paired_job in zip(candidates, jobs):
            payload = {
                "candidate_id": candidate.candidate_ID,
                "job_id": paired_job.job_ID,
                "score": 1.0,
                "timestamp": now_ts,
            }
            ok, error = self._post_json(f"{model_url}/interactions", payload)
            if ok:
                graph_result["interactions"] += 1
            elif error:
                graph_result["errors"].append(error)

    def _post_json(self, url, payload):
        try:
            response = requests.post(url, json=payload, timeout=45)
            if response.status_code in (200, 201, 409):
                return True, None
            return False, f"{url} -> {response.status_code}: {response.text[:180]}"
        except Exception as exc:
            return False, f"{url} -> exception: {exc}"

    def _to_float(self, value):
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _write_credentials_csv(self, output_path, rows):
        fieldnames = ["role", "username", "password", "email", "profile_id", "track", "paired_id"]
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
