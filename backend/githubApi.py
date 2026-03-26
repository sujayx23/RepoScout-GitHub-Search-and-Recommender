import requests
import json
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import util
from urllib.parse import quote
from keybert import KeyBERT
import time
from datetime import datetime, timezone

class GitHubAPI:
    def __init__(self, token=None):
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            self.headers["Authorization"] = f"token {token}"

    def check_rate_limit(self):
        response = requests.get("https://api.github.com/rate_limit", headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            core = data["rate"]
            print(f"Core Limit: {core['limit']}")
            print(f"Core Remaining: {core['remaining']}")
            print(f"Core Reset: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(core['reset']))}")
        else:
            print(f"Failed to fetch rate limit status: {response.status_code}")

    def extract_repo_details(self, url):
        parts = url.rstrip("/").split("/")
        return parts[-2], parts[-1]  # (owner, repo)

    def get_repo_metadata(self, owner, repo):
        url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching metadata for {owner}/{repo}: {response.status_code}, {response.json()}")
            return None

    def fetch_additional_repo_data(self, repo_full_name):
        # Get Forks
        forks_url = f"https://api.github.com/repos/{repo_full_name}"
        forks_response = requests.get(forks_url, headers=self.headers)
        forks_count = forks_response.json().get("forks_count", 0) if forks_response.status_code == 200 else 0

        # Get Open Issues
        open_issues_url = f"https://api.github.com/repos/{repo_full_name}/issues"
        issues_response = requests.get(open_issues_url, headers=self.headers)
        open_issues_count = len(issues_response.json()) if issues_response.status_code == 200 else 0

        # Get Pull Requests
        pulls_url = f"https://api.github.com/repos/{repo_full_name}/pulls"
        pulls_response = requests.get(pulls_url, headers=self.headers)
        pr_count = len(pulls_response.json()) if pulls_response.status_code == 200 else 0

        # Get Recent Commits
        commits_url = f"https://api.github.com/repos/{repo_full_name}/commits"
        commits_response = requests.get(commits_url, headers=self.headers)
        recent_commits = len(commits_response.json()) if commits_response.status_code == 200 else 0

        # Compute Activity Score
        activity_score = recent_commits + open_issues_count + pr_count

        return forks_count, activity_score

    def search_repos_by_topic(self, topic):
        encoded_query = quote(topic)
        url = f"https://api.github.com/search/repositories?q={encoded_query}"
        print(f"Searching GitHub with query: {url}")
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            repos = data.get("items", [])
            print(f"Found {len(repos)} repositories for the topic: {topic}")
            return repos
        else:
            print(f"Failed to fetch data for topic '{topic}'. Status Code: {response.status_code}")
            return []

class RepoProcessor:
    def preprocess_metadata(self, data):
        structured_data = {
            "Repository Name": data.get("name"),
            "Owner": data.get("owner", {}).get("login"),
            "Description": data.get("description"),
            "Stars": data.get("stargazers_count"),
            "Forks": data.get("forks_count"),
            "Open Issues": data.get("open_issues_count"),
            "Primary Language": data.get("language"),
            "Created At": data.get("created_at"),
            "Updated At": data.get("updated_at"),
            "License": data.get("license", {}).get("name") if data.get("license") else "No License",
            "Clone URL": data.get("clone_url"),
            "Contributors URL": data.get("contributors_url"),
            "Topics": data.get("topics", []),
        }
        return structured_data

class SemanticAnalyzer:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, texts):
        """Accept a single string or a list of strings; always returns a 2-D tensor (N, H)."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings  # shape (N, H)

    def compute_similarity(self, text1, text2):
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return util.pytorch_cos_sim(emb1, emb2).item()

    def compute_batch_similarity(self, query, descriptions):
        """Encode query once and all descriptions in one forward pass; return a list of scores."""
        query_emb = self.get_embedding(query)           # (1, H)
        desc_embs = self.get_embedding(descriptions)    # (N, H)
        scores = util.pytorch_cos_sim(query_emb, desc_embs)[0]  # (N,)
        return scores.tolist()

class KeywordExtractor:
    def __init__(self):
        self.kw_model = KeyBERT()

    def extract_keywords(self, text, top_n=3):
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=top_n
        )
        return [kw[0] for kw in keywords]

class RelatedRepoFinder:
    def __init__(self, github_api, repo_processor, semantic_analyzer, keyword_extractor):
        self.github_api = github_api
        self.repo_processor = repo_processor
        self.semantic_analyzer = semantic_analyzer
        self.keyword_extractor = keyword_extractor

    def build_query_from_repo(self, owner, repo):
        metadata = self.github_api.get_repo_metadata(owner, repo)
        if not metadata:
            return None

        description = metadata.get("description", "")
        language = metadata.get("language", "")
        topics = metadata.get("topics", [])

        top_keywords = self.keyword_extractor.extract_keywords(description, top_n=3) if description else []

        if top_keywords:
            search_terms = " ".join(top_keywords)
        else:
            search_terms = ""

        base_query = f"{search_terms} in:name,description,topics"
        if language:
            base_query += f" language:{language}"

        if len(base_query) > 256:
            fixed_part = f" in:name,description,topics"
            if language:
                fixed_part += f" language:{language}"
            allowed_length = 256 - len(fixed_part) - 1
            truncated_search_terms = search_terms[:allowed_length]
            base_query = f"{truncated_search_terms}{fixed_part}"

        print("The link query:", base_query)
        return base_query

    def rank_repositories(self, repos, topic):
        # Filter forks and archived repos
        filtered = [r for r in repos if not r.get("fork", False) and not r.get("archived", False)]
        if not filtered:
            return []

        # First pass: collect all non-embedding data and build description list
        raw_data = []
        descriptions = []
        for repo_data in filtered:
            repo_full_name = repo_data["full_name"]
            forks_count, activity_score = self.github_api.fetch_additional_repo_data(repo_full_name)
            description = repo_data.get("description", "")
            language = repo_data.get("language", "")
            stars = repo_data["stargazers_count"]
            updated_at = repo_data.get("updated_at", "")

            if not description:
                topics = repo_data.get("topics", [])
                description = " ".join(topics) if topics else repo_data.get("name", "")

            # Recency score from updated_at
            recency_score = 0.25
            if updated_at:
                try:
                    updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    days_ago = (datetime.now(timezone.utc) - updated_dt).days
                    if days_ago <= 30:
                        recency_score = 1.0
                    elif days_ago <= 90:
                        recency_score = 0.75
                    elif days_ago <= 365:
                        recency_score = 0.5
                except Exception:
                    pass

            descriptions.append(description)
            raw_data.append({
                "repo_data": repo_data,
                "stars": stars,
                "forks": forks_count,
                "activity": activity_score,
                "language": language,
                "recency": recency_score,
            })

        # Single batch forward pass for all descriptions
        similarities = self.semantic_analyzer.compute_batch_similarity(topic, descriptions)

        # Attach relevance scores using batch results
        for entry, sim in zip(raw_data, similarities):
            language_match = 1 if topic.lower() in (entry["language"] or "").lower() else 0
            entry["relevance"] = (sim * 0.8) + (language_match * 0.2)

        # Normalize stars, forks, activity to [0, 1]
        max_stars = max(r["stars"] for r in raw_data) or 1
        max_forks = max(r["forks"] for r in raw_data) or 1
        max_activity = max(r["activity"] for r in raw_data) or 1

        ranked_repos = []
        for r in raw_data:
            norm_stars = r["stars"] / max_stars
            norm_forks = r["forks"] / max_forks
            norm_activity = r["activity"] / max_activity

            final_score = (
                (norm_stars * 0.35) +
                (norm_forks * 0.20) +
                (norm_activity * 0.15) +
                (r["relevance"] * 0.20) +
                (r["recency"] * 0.10)
            )

            repo_data = r["repo_data"]
            ranked_repos.append({
                "name": repo_data["name"],
                "owner": repo_data["owner"]["login"],
                "url": repo_data["html_url"],
                "description": repo_data.get("description", ""),
                "language": repo_data.get("language", ""),
                "stars": r["stars"],
                "forks": r["forks"],
                "activity": r["activity"],
                "relevance": round(r["relevance"], 4),
                "recency": r["recency"],
                "final_score": round(final_score, 4),
            })

        ranked_repos = sorted(ranked_repos, key=lambda x: x["final_score"], reverse=True)[:10]
        return ranked_repos

    def find_related_repositories(self, github_url):
        owner, repo = self.github_api.extract_repo_details(github_url)
        query_text = self.build_query_from_repo(owner, repo)

        if not query_text:
            print("Could not build query.")
            return []

        repos = self.github_api.search_repos_by_topic(query_text)
        if not repos:
            print("No repos found for the generated query.")
            return []

        return self.rank_repositories(repos, query_text)

# --- Main Execution ---
if __name__ == "__main__":
    github_token = "" # insert your PAT here
    github_api = GitHubAPI(token=github_token)
    repo_processor = RepoProcessor()
    semantic_analyzer = SemanticAnalyzer()
    keyword_extractor = KeywordExtractor()
    related_repo_finder = RelatedRepoFinder(github_api, repo_processor, semantic_analyzer, keyword_extractor)

    github_api.check_rate_limit()
    # Example of Part 3 - search by URL 
    target_repo_url = "https://github.com/weiaicunzai/awesome-image-classification"
    related_repositories = related_repo_finder.find_related_repositories(target_repo_url)

    print("\nTop 10 related repositories:")
    for repo in related_repositories:
        print(json.dumps(repo, indent=2))

    # Example of Part 1  - metadata
    owner, repo_name = github_api.extract_repo_details(target_repo_url)
    metadata = github_api.get_repo_metadata(owner, repo_name)
    if metadata:
        structured_metadata = repo_processor.preprocess_metadata(metadata)
        print("\nMetadata for the target repository:")
        print(json.dumps(structured_metadata, indent=4))

    # Example of Part 2 functionality (searching by a topic)
    topic = "Machine learning projects"
    searched_repos = github_api.search_repos_by_topic(topic)
    if searched_repos:
        ranked_searched_repos = related_repo_finder.rank_repositories(searched_repos, topic)
        print(f"\nTop 5 repositories related to '{topic}':")
        for repo in ranked_searched_repos[:5]:
            print(json.dumps(repo, indent=2))