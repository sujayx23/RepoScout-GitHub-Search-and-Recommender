# RepoScout

**RepoScout** is an intelligent GitHub repository discovery tool that combines semantic search (SBERT), keyword extraction (KeyBERT), and multi-factor ranking to surface the most relevant repositories for any topic or GitHub URL.

## Features

- **Topic Search** — enter any keyword or phrase, get ranked results
- **URL-based Discovery** — paste a GitHub repo URL; RepoScout extracts its keywords and finds semantically similar repos
- **Smart Ranking** — normalized scoring across stars, forks, activity, semantic relevance, and recency
- **Fork & Archive Filtering** — forks and archived repos are excluded from results
- **Redis Caching** — all GitHub API responses cached for 1 hour; falls back gracefully when Redis is unavailable
- **Modern Dark UI** — responsive React frontend, no external UI libraries

## Scoring Algorithm

```
final_score = (norm_stars × 0.35)
            + (norm_forks  × 0.20)
            + (norm_activity × 0.15)
            + (relevance   × 0.20)
            + (recency     × 0.10)
```

- **norm_stars / norm_forks / norm_activity** — each metric normalized to [0, 1] relative to the current result set
- **relevance** — cosine similarity via SBERT (`all-mpnet-base-v2`), blended with a language-match bonus
- **recency** — based on `updated_at`: ≤30 days → 1.0, ≤90 days → 0.75, ≤365 days → 0.5, older → 0.25

## Project Structure

```
RepoScout/
├── backend/
│   ├── githubApi.py        # Core logic: GitHub API, ranking, semantic analysis
│   ├── main.py             # FastAPI application
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.css
│   ├── nginx.conf
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
├── .env.example
└── README.md
```

## Quick Start (Docker Compose)

### 1. Get a GitHub Personal Access Token

1. Go to **GitHub → Settings → Developer settings → Personal access tokens**
2. Generate a token with `public_repo` scope (or `repo` for private repos)

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your token:
#   GITHUB_TOKEN=ghp_...
```

### 3. Build and run

```bash
docker compose up --build
```

| Service  | URL                    |
|----------|------------------------|
| Frontend | http://localhost:3000  |
| Backend  | http://localhost:8000  |
| Redis    | localhost:6379         |

### 4. Stop

```bash
docker compose down
```

## Local Development (without Docker)

### Backend

```bash
cd backend
pip install -r requirements.txt

# Set your token
export GITHUB_TOKEN=ghp_...

uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install

# Point at backend
echo "VITE_API_URL=http://localhost:8000" > .env.local

npm run dev
# Open http://localhost:5173
```

## API Reference

All responses include a `"source"` field (`"api"` or `"cache"`).

---

### `GET /health`

Health check. Returns Redis availability.

**Response**
```json
{ "status": "ok", "redis": true }
```

---

### `GET /search?topic={topic}`

Search GitHub repositories by topic and return ranked results.

| Param   | Type   | Required | Description           |
|---------|--------|----------|-----------------------|
| `topic` | string | Yes      | Search query / topic  |

**Response**
```json
{
  "source": "api",
  "results": [
    {
      "name": "pytorch",
      "owner": "pytorch",
      "url": "https://github.com/pytorch/pytorch",
      "description": "Tensors and Dynamic neural networks in Python...",
      "language": "Python",
      "stars": 82000,
      "forks": 22000,
      "activity": 130,
      "relevance": 0.9124,
      "recency": 1.0,
      "final_score": 0.9341
    }
  ]
}
```

---

### `POST /recommend`

Find repositories related to a given GitHub URL.

**Request body**
```json
{ "github_url": "https://github.com/owner/repo" }
```

**Response** — same shape as `/search`

---

### `GET /metadata?url={github_url}`

Fetch structured metadata for a GitHub repository.

| Param | Type   | Required | Description      |
|-------|--------|----------|------------------|
| `url` | string | Yes      | Full GitHub URL  |

**Response**
```json
{
  "source": "api",
  "metadata": {
    "Repository Name": "pytorch",
    "Owner": "pytorch",
    "Description": "...",
    "Stars": 82000,
    "Forks": 22000,
    "Open Issues": 14000,
    "Primary Language": "Python",
    "Created At": "2016-08-13T05:25:09Z",
    "Updated At": "2024-05-01T12:00:00Z",
    "License": "BSD 3-Clause",
    "Clone URL": "https://github.com/pytorch/pytorch.git",
    "Topics": ["deep-learning", "neural-network", "pytorch"]
  }
}
```

---

## Environment Variables

| Variable     | Default     | Description                              |
|--------------|-------------|------------------------------------------|
| `GITHUB_TOKEN` | `""`      | GitHub Personal Access Token             |
| `REDIS_HOST` | `localhost` | Redis hostname                           |
| `REDIS_PORT` | `6379`      | Redis port                               |
| `VITE_API_URL` | `http://localhost:8000` | Backend URL (build-time for frontend) |

## GitHub API Rate Limits

| Auth state    | Requests / hour |
|---------------|-----------------|
| No token      | 60              |
| With token    | 5,000           |

A token is strongly recommended. Set `GITHUB_TOKEN` in your `.env` file.

## Tech Stack

| Layer    | Technology                              |
|----------|-----------------------------------------|
| Backend  | Python 3.11, FastAPI, uvicorn           |
| ML       | SBERT (`all-mpnet-base-v2`), KeyBERT    |
| Cache    | Redis                                   |
| Frontend | React 18, Vite                          |
| Serving  | nginx                                   |
| Infra    | Docker Compose                          |

## License

MIT
