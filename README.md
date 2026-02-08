# RepoScout

**RepoScout: GitHub Search and Recommender** is an intelligent repository recommendation system that uses semantic analysis and machine learning to help you discover GitHub repositories related to your interests or existing projects.

## Features

- **Smart Repository Discovery**: Find repositories similar to a given GitHub URL using semantic analysis
- **Topic-Based Search**: Search for repositories by topic with intelligent ranking
- **Metadata Extraction**: Extract comprehensive metadata from any GitHub repository
- **Semantic Similarity**: Uses sentence transformers (SBERT) to compute relevance scores
- **Keyword Extraction**: Automatically extracts key terms using KeyBERT
- **Multi-Factor Ranking**: Ranks repositories based on:
  - Stars (40%)
  - Forks (20%)
  - Activity Score (20%)
  - Semantic Relevance (20%)

## How It Works

RepoScout combines multiple techniques to provide accurate repository recommendations:

1. **Keyword Extraction**: Uses KeyBERT to extract relevant keywords from repository descriptions
2. **Semantic Analysis**: Employs sentence transformers to compute semantic similarity between repositories
3. **GitHub API Integration**: Fetches repository metadata including stars, forks, issues, and activity metrics
4. **Intelligent Ranking**: Combines popularity metrics with semantic relevance to rank results

## Installation

### Prerequisites

- Python 3.7+
- GitHub Personal Access Token (PAT)

### Dependencies

Install the required packages:

```bash
pip install requests torch transformers sentence-transformers keybert
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/sujayx23/RepoScout-GitHub-Search-and-Recommender.git
cd RepoScout-GitHub-Search-and-Recommender
```

2. Get a GitHub Personal Access Token:
   - Go to GitHub Settings > Developer settings > Personal access tokens
   - Generate a new token with `repo` scope
   - Copy the token

3. Add your token to the code:
   - Open `backend/githubApi.py`
   - Replace the empty string on line 224 with your token:
     ```python
     github_token = "your_github_token_here"
     ```

## Usage

### 1. Find Related Repositories by URL

```python
from backend.githubApi import *

github_api = GitHubAPI(token="your_token")
repo_processor = RepoProcessor()
semantic_analyzer = SemanticAnalyzer()
keyword_extractor = KeywordExtractor()
related_repo_finder = RelatedRepoFinder(github_api, repo_processor, semantic_analyzer, keyword_extractor)

# Find repositories similar to a target repo
target_url = "https://github.com/pytorch/pytorch"
related_repos = related_repo_finder.find_related_repositories(target_url)

for repo in related_repos:
    print(f"{repo['name']} - Stars: {repo['stars']} - Score: {repo['final_score']}")
```

### 2. Search Repositories by Topic

```python
topic = "machine learning"
searched_repos = github_api.search_repos_by_topic(topic)
ranked_repos = related_repo_finder.rank_repositories(searched_repos, topic)

for repo in ranked_repos[:5]:
    print(f"{repo['owner']}/{repo['name']}: {repo['url']}")
```

### 3. Get Repository Metadata

```python
owner, repo_name = github_api.extract_repo_details("https://github.com/pytorch/pytorch")
metadata = github_api.get_repo_metadata(owner, repo_name)
structured_data = repo_processor.preprocess_metadata(metadata)

print(json.dumps(structured_data, indent=4))
```

### 4. Check Rate Limit

```python
github_api.check_rate_limit()
```

## API Components

### GitHubAPI
- `get_repo_metadata(owner, repo)`: Fetch repository metadata
- `search_repos_by_topic(topic)`: Search repositories by topic
- `fetch_additional_repo_data(repo_full_name)`: Get forks and activity metrics
- `check_rate_limit()`: Check GitHub API rate limit status

### SemanticAnalyzer
- Uses `sentence-transformers/all-MiniLM-L6-v2` model
- Computes cosine similarity between text embeddings

### KeywordExtractor
- Extracts important keywords from repository descriptions
- Uses KeyBERT with customizable n-gram ranges

### RelatedRepoFinder
- Combines all components to find and rank related repositories
- Builds intelligent search queries from repository metadata
- Returns top 10 most relevant repositories

## Scoring Algorithm

The final score for each repository is calculated as:

```
final_score = (stars × 0.4) + (forks × 0.2) + (activity_score × 0.2) + (relevance × 0.2)
```

Where:
- **Stars**: Repository stargazers count
- **Forks**: Number of forks
- **Activity Score**: Sum of recent commits, open issues, and pull requests
- **Relevance**: Semantic similarity score (0-1) computed using SBERT

## Example Output

```json
{
  "name": "tensorflow",
  "owner": "tensorflow",
  "url": "https://github.com/tensorflow/tensorflow",
  "stars": 175000,
  "forks": 88000,
  "activity": 450,
  "relevance": 0.85,
  "final_score": 192500.17
}
```

## Limitations

- GitHub API rate limits apply (60 requests/hour without token, 5000/hour with token)
- First run downloads the sentence transformer model (~80MB)
- Semantic analysis requires computational resources for embedding generation

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is open source and available under the MIT License.

## Author

Built with Python, PyTorch, and the GitHub API.
