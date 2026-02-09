import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

LAYER1_URL = os.environ.get("LAYER1_URL", "http://localhost:8000/site")
LAYER2_PORT = int(os.environ.get("LAYER2_PORT", "8001"))

VACANCIES_PATH = Path(os.environ.get(
    "VACANCIES_PATH",
    str(BASE_DIR.parent / "enriched_vacancies.json.gz"),
))

JOBBERT_MODEL = os.environ.get("JOBBERT_MODEL", "TechWolf/JobBERT-v2")
FAISS_INDEX_PATH = DATA_DIR / "vacancy_index.faiss"
EMBEDDINGS_PATH = DATA_DIR / "vacancy_embeddings.npy"
VACANCY_META_PATH = DATA_DIR / "vacancy_metadata.jsonl"
UNIQUE_TITLES_PATH = DATA_DIR / "unique_titles.json"

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

RETRIEVAL_TOP_K = 10
RERANK_THRESHOLD = 3.0
CLUSTER_DISTANCE_THRESHOLD = 0.35
JOBBERT_BATCH_SIZE = 256
JOBBERT_MAX_TOKENS = 64

EXCLUSION_TITLE_KEYWORDS = [
    "software engineer", "software developer", "frontend developer",
    "backend developer", "full stack developer", "fullstack developer",
    "web developer", "mobile developer", "devops", "sre",
    "site reliability", "infrastructure engineer", "platform engineer",
    "cloud engineer", "data engineer", "etl developer",
    "qa engineer", "test engineer", "quality assurance engineer",
    "sdet", "automation engineer", "product manager",
    "it administrator", "system administrator", "sysadmin",
    "network administrator", "database administrator", "dba",
    "release engineer", "build engineer", "mlops",
]
