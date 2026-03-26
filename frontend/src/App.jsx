import { useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function isGithubUrl(input) {
  return /^(https?:\/\/)?(www\.)?github\.com\/.+\/.+/i.test(input.trim())
}

function StarIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
    </svg>
  )
}

function ForkIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <circle cx="6" cy="6" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="12" cy="18" r="2"/>
      <line x1="6" y1="8" x2="6" y2="12"/><line x1="18" y1="8" x2="18" y2="12"/>
      <path d="M6 12 Q6 16 12 16 Q18 16 18 12"/>
    </svg>
  )
}

function RepoCard({ repo }) {
  const recencyLabel =
    repo.recency >= 1.0 ? 'Active' :
    repo.recency >= 0.75 ? 'Recent' :
    repo.recency >= 0.5 ? 'Moderate' : 'Inactive'

  const recencyClass =
    repo.recency >= 1.0 ? 'badge-green' :
    repo.recency >= 0.75 ? 'badge-blue' :
    repo.recency >= 0.5 ? 'badge-yellow' : 'badge-gray'

  return (
    <div className="repo-card">
      <div className="card-header">
        <div className="repo-name-row">
          <a href={repo.url} target="_blank" rel="noopener noreferrer" className="repo-name">
            <span className="owner">{repo.owner}</span>
            <span className="sep">/</span>
            <span className="name">{repo.name}</span>
          </a>
          <span className={`badge ${recencyClass}`}>{recencyLabel}</span>
        </div>
        {repo.language && <span className="language-tag">{repo.language}</span>}
      </div>

      {repo.description && (
        <p className="repo-description">{repo.description}</p>
      )}

      <div className="card-stats">
        <span className="stat">
          <StarIcon /> {Number(repo.stars).toLocaleString()}
        </span>
        <span className="stat">
          <ForkIcon /> {Number(repo.forks).toLocaleString()}
        </span>
        <span className="stat relevance-stat">
          Relevance&nbsp;{(repo.relevance * 100).toFixed(0)}%
        </span>
        <span className="stat score-stat">
          Score&nbsp;{(repo.final_score * 100).toFixed(1)}
        </span>
      </div>
    </div>
  )
}

function Spinner() {
  return (
    <div className="spinner-wrap">
      <div className="spinner" />
      <p>Searching repositories…</p>
    </div>
  )
}

function EmptyState({ onSuggestion }) {
  const suggestions = ['machine learning', 'web scraping', 'data visualization', 'fastapi', 'react hooks']
  return (
    <div className="empty-state">
      <div className="empty-icon">🔭</div>
      <h2>Discover GitHub Repositories</h2>
      <p>Enter a topic or paste a GitHub URL to find related repos.</p>
      <div className="suggestions">
        <p className="suggestions-label">Try searching for:</p>
        <div className="suggestion-chips">
          {suggestions.map(s => (
            <button key={s} className="chip" onClick={() => onSuggestion(s)}>{s}</button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [input, setInput] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [searchedTerm, setSearchedTerm] = useState('')

  async function handleSearch(value) {
    const query = (value !== undefined ? value : input).trim()
    if (!query) return

    setLoading(true)
    setError(null)
    setResults(null)
    setSearchedTerm(query)

    try {
      let res
      if (isGithubUrl(query)) {
        res = await fetch(`${API_BASE}/recommend`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ github_url: query }),
        })
      } else {
        res = await fetch(`${API_BASE}/search?topic=${encodeURIComponent(query)}`)
      }

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `Server error ${res.status}`)
      }

      const data = await res.json()
      setResults(data.results || [])
    } catch (e) {
      setError(e.message || 'Something went wrong. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') handleSearch()
  }

  function handleSuggestion(s) {
    setInput(s)
    handleSearch(s)
  }

  const urlDetected = isGithubUrl(input) && input.trim()

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-icon">🔭</span>
          <span className="logo-text">RepoScout</span>
        </div>
        <p className="tagline">Discover GitHub repositories powered by semantic search</p>
      </header>

      <main className="main">
        <div className="search-bar">
          <input
            className="search-input"
            type="text"
            placeholder="Search by topic or paste a GitHub URL…"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            className="search-btn"
            onClick={() => handleSearch()}
            disabled={loading || !input.trim()}
          >
            {loading ? '…' : 'Search'}
          </button>
        </div>

        {urlDetected && (
          <p className="mode-hint">URL detected — will find similar repositories</p>
        )}

        {loading && <Spinner />}

        {error && (
          <div className="error-state">
            <span className="error-icon">⚠️</span>
            <p>{error}</p>
          </div>
        )}

        {!loading && !error && results === null && (
          <EmptyState onSuggestion={handleSuggestion} />
        )}

        {!loading && !error && results !== null && results.length === 0 && (
          <div className="empty-results">
            <p>No repositories found for <strong>"{searchedTerm}"</strong>. Try a different query.</p>
          </div>
        )}

        {!loading && !error && results && results.length > 0 && (
          <>
            <p className="result-count">
              {results.length} result{results.length !== 1 ? 's' : ''} for <strong>"{searchedTerm}"</strong>
            </p>
            <div className="results-grid">
              {results.map(repo => (
                <RepoCard key={repo.url} repo={repo} />
              ))}
            </div>
          </>
        )}
      </main>

      <footer className="footer">
        <p>Powered by GitHub API · SBERT (all-mpnet-base-v2) · KeyBERT</p>
      </footer>
    </div>
  )
}
