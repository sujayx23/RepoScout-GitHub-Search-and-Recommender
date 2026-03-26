import os
import json
import hashlib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from githubApi import GitHubAPI, RepoProcessor, SemanticAnalyzer, KeywordExtractor, RelatedRepoFinder

# Optional Redis caching
try:
    import redis
    _redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
        socket_connect_timeout=2,
    )
    _redis_client.ping()
    REDIS_AVAILABLE = True
except Exception:
    _redis_client = None
    REDIS_AVAILABLE = False

CACHE_TTL = 3600  # 1 hour

def cache_get(key: str):
    if not REDIS_AVAILABLE:
        return None
    try:
        value = _redis_client.get(key)
        return json.loads(value) if value else None
    except Exception:
        return None

def cache_set(key: str, value):
    if not REDIS_AVAILABLE:
        return
    try:
        _redis_client.setex(key, CACHE_TTL, json.dumps(value))
    except Exception:
        pass

def make_cache_key(prefix: str, *args) -> str:
    raw = prefix + "|" + "|".join(str(a) for a in args)
    return hashlib.sha256(raw.encode()).hexdigest()


# Initialise shared components once at startup
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

github_api = GitHubAPI(token=GITHUB_TOKEN)
repo_processor = RepoProcessor()
semantic_analyzer = SemanticAnalyzer()
keyword_extractor = KeywordExtractor()
related_repo_finder = RelatedRepoFinder(github_api, repo_processor, semantic_analyzer, keyword_extractor)

app = FastAPI(title="RepoScout API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------

class RecommendRequest(BaseModel):
    github_url: str


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "redis": REDIS_AVAILABLE}


@app.get("/search")
def search(topic: str = Query(..., min_length=1)):
    """Search repositories by topic and return ranked results."""
    cache_key = make_cache_key("search", topic)
    cached = cache_get(cache_key)
    if cached is not None:
        return {"source": "cache", "results": cached}

    repos = github_api.search_repos_by_topic(topic)
    if not repos:
        return {"source": "api", "results": []}

    ranked = related_repo_finder.rank_repositories(repos, topic)
    cache_set(cache_key, ranked)
    return {"source": "api", "results": ranked}


@app.post("/recommend")
def recommend(body: RecommendRequest):
    """Find repositories related to a given GitHub repo URL."""
    url = body.github_url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="github_url is required")

    cache_key = make_cache_key("recommend", url)
    cached = cache_get(cache_key)
    if cached is not None:
        return {"source": "cache", "results": cached}

    results = related_repo_finder.find_related_repositories(url)
    cache_set(cache_key, results)
    return {"source": "api", "results": results}


@app.get("/metadata")
def metadata(url: str = Query(..., min_length=1)):
    """Get structured metadata for a GitHub repository URL."""
    cache_key = make_cache_key("metadata", url)
    cached = cache_get(cache_key)
    if cached is not None:
        return {"source": "cache", "metadata": cached}

    try:
        owner, repo = github_api.extract_repo_details(url)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL")

    raw = github_api.get_repo_metadata(owner, repo)
    if raw is None:
        raise HTTPException(status_code=404, detail="Repository not found")

    structured = repo_processor.preprocess_metadata(raw)
    cache_set(cache_key, structured)
    return {"source": "api", "metadata": structured}
