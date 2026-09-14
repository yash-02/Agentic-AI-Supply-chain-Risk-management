# GraphRAG-Based Supply Chain Analyzer

A minimal, working GraphRAG-style project for supply-chain and enterprise risk analysis.

The app ingests financial filing text and news articles, extracts company/supplier/product/risk relationships, stores them as a local knowledge graph, and answers questions using graph traversal plus source evidence snippets.

## What It Does

- Builds a graph of companies, suppliers, products, countries, and risk events.
- Ingests local SEC 10-K-style text files and news JSON files.
- Supports real-time news ingestion through an optional RSS feed.
- Answers questions such as: `How does a semiconductor shortage affect Apple?`
- Shows graph paths and evidence snippets in a browser UI.

## Tech Used

- Python standard library
- SQLite for local graph storage by default, with an optional Neo4j backend (see below)
- Built-in HTTP server for API and UI
- Optional Ollama local LLM or OpenAI for relationship extraction, answer generation, and embeddings
- Rule-based relationship extraction and keyword-only retrieval as automatic fallbacks when no LLM is available
- Optional RSS feed polling for live news updates

This version is intentionally dependency-free so it runs immediately with zero setup: SQLite storage, rule-based extraction, and templated answers all work with nothing installed. Every LLM-dependent step degrades gracefully instead of failing - if Ollama/OpenAI is available, it's used for both relationship extraction and answer generation; if not, deterministic fallbacks (keyword extraction, evidence-snippet templating) take over automatically, per chunk / per question.

### How relationship extraction works

Each ingested chunk is sent to the active LLM with a strict prompt: return only a JSON list of `{source, source_label, relation, target, target_label, confidence}` objects, using a fixed vocabulary of entity types (`Company`, `Supplier`, `Product`, `Country`, `RiskEvent`, `NewsArticle`) and relations (`RELIES_ON`, `DEPENDS_ON`, `OPERATES_IN`, `EXPOSED_TO`, `REPORTS`, `AFFECTS`). The raw output is never trusted as-is - every relationship is checked against `RELATION_SIGNATURES` (which entity-type pairs are valid for each relation) before being written, and entity names are canonicalized (stripping corporate suffixes, matching known aliases) so `"NVIDIA Inc."` and `"NVIDIA Corporation"` land on the same node. If the LLM isn't configured, the call fails, or nothing parses, the chunk falls back to the original keyword-based extractor.

### How retrieval works (hybrid keyword + vector search)

Every chunk is also embedded at ingest time (Ollama's `nomic-embed-text` by default, or OpenAI embeddings) and the vector is stored alongside the chunk. At question time, two independent rankings are computed - term-frequency keyword overlap (the original signal) and cosine similarity against the question's own embedding - and fused with [Reciprocal Rank Fusion](https://en.wikipedia.org/wiki/Reciprocal_rank_fusion) (`RRF_K = 60`, the same default Elasticsearch/Azure AI Search use for hybrid search), rather than a hand-tuned weighted sum of two differently-scaled scores. Each evidence item in the API/UI is tagged `matched_by: keyword | vector | hybrid` so it's visible which signal actually surfaced it. If no embedding provider is configured/reachable, this reduces to exactly the original keyword-only ranking - RRF over a single ranking preserves its order, so there's no regression when running fully offline.

This was validated with a deliberately adversarial query (paraphrased so hard it shares zero exact tokens with the target chunk, e.g. asking about a "geological tremor abroad" instead of the stored article's "earthquake in Taiwan"): pure keyword scoring ranked the correct chunk *last* out of four candidates (a generic word like "manufacturer" happened to appear more in unrelated filings), while cosine similarity on embeddings ranked it first by a clear margin. RRF fusion partially, not perfectly, corrects for this since it weights both rankings equally - a real, known limitation of unweighted RRF, and the reason production hybrid-search systems often weight semantic search higher or add a reranking pass for paraphrase-heavy queries.

## Run

```bash
python app.py --ingest
python app.py --ask "How does a semiconductor shortage affect Apple?"
python app.py --serve
```

Open:

```text
http://127.0.0.1:8000
```

## Enable Free Local LLM Q&A With Ollama

Install Ollama from:

```text
https://ollama.com/download
```

Pull a small local model, plus an embedding model for vector search:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Run the app with Ollama:

```powershell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.2:3b"
$env:OLLAMA_EMBED_MODEL="nomic-embed-text"
python app.py --serve
```

Ollama's local API normally runs at:

```text
http://localhost:11434
```

If your Ollama server uses a different URL:

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
```

## Enable OpenAI LLM Q&A

PowerShell:

```powershell
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_MODEL="gpt-4.1-mini"
$env:OPENAI_EMBED_MODEL="text-embedding-3-small"
python app.py --serve
```

Default mode is:

```powershell
$env:LLM_PROVIDER="auto"
```

In `auto` mode, the app tries Ollama first, then OpenAI, then local fallback.

Now the app answers general questions about the ingested context, for example:

```text
Tell me about profits of Apple
How does export control risk affect NVIDIA?
What suppliers are connected to Tesla?
```

The LLM is instructed to answer only from the retrieved context. If the answer is not present in the filings/news, it should say the current context does not include that fact.

## Enable Neo4j Graph Storage

By default the graph lives in a local SQLite file - no setup required. To use a real graph
database instead, install the driver and point the app at a running Neo4j server:

```bash
pip install -r requirements.txt   # installs the neo4j driver
```

Quickest way to get a Neo4j server running locally (Docker):

```bash
docker run -d --name graphrag-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/<your-password> neo4j:5
```

Then run the app with:

```powershell
$env:GRAPH_BACKEND="neo4j"
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="<your-password>"
python app.py --serve
```

Browse the graph directly at `http://localhost:7474` with Cypher, e.g.:

```cypher
MATCH (c:Company {name: "Apple Inc."})-[r]-(other)
RETURN c, r, other
```

`GRAPH_BACKEND` defaults to `auto`: if a Neo4j server is reachable at `NEO4J_URI` it's used automatically (the `/api/status` endpoint and the UI's "Graph DB" tile show which backend is active); otherwise the app falls back to SQLite with no error. Set `GRAPH_BACKEND=neo4j` to fail loudly instead of silently falling back (useful if you want to be sure you're actually testing against Neo4j).

Node/edge storage and multi-hop traversal both become native Cypher in this mode (a single variable-length path query, `MATCH p = (start)-[*1..3]-(other)`) instead of the hand-rolled BFS-over-SQL the SQLite backend uses - text chunks and their retrieval scoring stay identical either way.

## Real-Time News

The app can refresh news from any RSS feed:

```bash
python app.py --rss "https://news.google.com/rss/search?q=semiconductor%20shortage%20supply%20chain"
```

For background polling while the server runs:

PowerShell:

```powershell
$env:NEWS_RSS_URL="https://news.google.com/rss/search?q=semiconductor%20shortage%20supply%20chain"
$env:NEWS_POLL_SECONDS="900"
python app.py --serve
```

If `NEWS_RSS_URL` is not set, the UI's `Refresh RSS` button reloads the included sample news.

## API

### Status

```http
GET /api/status
```

### Rebuild Graph

```http
POST /api/ingest
Content-Type: application/json

{"reset": true}
```

### Ask Question

```http
POST /api/ask
Content-Type: application/json

{"question": "How does export control risk affect NVIDIA?"}
```

### Refresh News

```http
POST /api/news/refresh
```

## Project Structure

```text
app.py                 Main app, API, graph stores (SQLite + Neo4j), extractor, retriever, UI
requirements.txt       Optional - only needed to enable the Neo4j backend
data/filings/          Sample 10-K-style source text
data/news/             Sample news articles
storage/               Local SQLite database, created at runtime (used unless Neo4j is configured)
```

## Resume Description

Built a GraphRAG-Based Supply Chain Analyzer that maps relationships between companies, suppliers, products, countries, and risk events using financial filings and real-time news. The system uses an LLM to convert unstructured text into a validated knowledge graph (Neo4j or SQLite), and combines multi-hop graph traversal with evidence retrieval to answer natural-language supply-chain risk questions - falling back to deterministic rules at every LLM-dependent step.

## Resume Bullets

- Developed a GraphRAG-style supply-chain risk analyzer that uses an LLM to extract typed company/supplier/product/geography/risk relationships from filings and news, with schema validation (entity-type-pair whitelisting, alias-based canonicalization) and a deterministic rule-based fallback.
- Implemented a pluggable graph storage layer (Neo4j via Cypher, or a zero-dependency local SQLite fallback) with multi-hop traversal to surface indirect supply-chain exposure a keyword search would miss.
- Implemented context-grounded LLM Q&A over graph paths and retrieved evidence snippets to answer questions about financial performance and supply-chain exposure, with a templated local fallback when no LLM is available.
- Built hybrid retrieval combining lexical (term-frequency) and semantic (embedding cosine similarity) chunk search via reciprocal rank fusion, validated against adversarial paraphrased queries to confirm the vector signal recovers matches keyword search alone misses.
- Added real-time news ingestion through RSS refresh and background polling to update risk events connected to tracked companies and suppliers.
