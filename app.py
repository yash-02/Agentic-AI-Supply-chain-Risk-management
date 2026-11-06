import argparse
import hashlib
import html
import json
import math
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


APP_NAME = "GraphRAG-Based Supply Chain Analyzer"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
NEWS_DIR = DATA_DIR / "news"
STORAGE_DIR = ROOT / "storage"
DB_PATH = STORAGE_DIR / "supply_chain.db"


def _load_dotenv() -> None:
    """Minimal stdlib-only .env loader (no external dependency): sets os.environ
    from KEY=VALUE lines in .env, without overriding vars already set in the shell."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


COMPANY_ALIASES = {
    "Apple Inc.": ["apple", "apple inc", "aapl"],
    "NVIDIA Corporation": ["nvidia", "nvidia corporation", "nvda"],
    "Tesla, Inc.": ["tesla", "tesla inc", "tsla"],
    "Taiwan Semiconductor Manufacturing Company": ["tsmc", "taiwan semiconductor", "taiwan semiconductor manufacturing"],
    "Foxconn": ["foxconn", "hon hai"],
    "Samsung Electronics": ["samsung", "samsung electronics"],
    "Panasonic": ["panasonic"],
    "CATL": ["catl", "contemporary amperex"],
}

SUPPLIERS = {
    "Taiwan Semiconductor Manufacturing Company": ["tsmc", "taiwan semiconductor"],
    "Foxconn": ["foxconn", "hon hai"],
    "Samsung Electronics": ["samsung electronics", "samsung"],
    "Panasonic": ["panasonic"],
    "CATL": ["catl", "contemporary amperex"],
}

PRODUCTS = {
    "semiconductors": ["semiconductor", "semiconductors", "chips", "chip"],
    "batteries": ["battery", "batteries", "battery cells"],
    "lithium": ["lithium"],
    "rare earth materials": ["rare earth", "rare earth materials"],
    "electric vehicles": ["electric vehicles", "evs", "vehicles"],
    "consumer electronics": ["iphone", "consumer electronics", "devices"],
    "artificial intelligence accelerators": ["ai accelerators", "gpus", "graphics processors", "accelerators"],
}

COUNTRIES = [
    "China",
    "Taiwan",
    "Japan",
    "South Korea",
    "United States",
    "Germany",
    "Mexico",
]

RISK_KEYWORDS = {
    "Semiconductor shortage": ["semiconductor shortage", "chip shortage", "shortage of chips", "chip supply"],
    "Supplier concentration": ["single source", "sole source", "limited number of suppliers", "third-party suppliers"],
    "Export controls": ["export control", "export controls", "export restrictions", "sanctions"],
    "Tariff risk": ["tariff", "tariffs", "trade duties"],
    "Port congestion": ["port congestion", "shipping delays", "logistics delays"],
    "Factory shutdown": ["factory shutdown", "plant shutdown", "production halt"],
    "Labor strike": ["strike", "labor disruption", "work stoppage"],
    "Geopolitical risk": ["geopolitical", "cross-strait", "military conflict"],
    "Natural disaster": ["earthquake", "flood", "typhoon", "wildfire"],
    "Commodity price volatility": ["commodity price", "raw material price", "price volatility"],
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}

QUERY_SYNONYMS = {
    "profit": ["profit", "profits", "profitability", "net", "income", "earnings"],
    "profits": ["profit", "profits", "profitability", "net", "income", "earnings"],
    "revenue": ["revenue", "sales", "net", "income", "profit"],
    "earnings": ["earnings", "profit", "profits", "net", "income"],
    "shortage": ["shortage", "shortages", "supply", "disruption"],
    "supplier": ["supplier", "suppliers", "manufacturer", "manufacturers"],
    "risk": ["risk", "risks", "exposure", "disruption"],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    return value


def canonical_key(label: str, name: str) -> str:
    return f"{label}:{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}"


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokenize(text: str) -> List[str]:
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in STOPWORDS or len(token) <= 1:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    return tokens


def expand_query_terms(question: str) -> Counter:
    terms = Counter(tokenize(question))
    expanded = Counter(terms)
    for term, count in terms.items():
        for synonym in QUERY_SYNONYMS.get(term, []):
            expanded[synonym] += count
    return expanded


RRF_K = 60  # standard reciprocal-rank-fusion damping constant (same default Elasticsearch/Azure
            # AI Search hybrid retrieval use) - fuses two rankings by position, not raw score,
            # so keyword TF scores and cosine similarities never need to be rescaled to match.


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def keyword_score_chunks(chunks: List[Dict], q_terms: Counter) -> Dict[str, float]:
    scores = {}
    for chunk in chunks:
        terms = Counter(json.loads(chunk["terms"]))
        score = 0.0
        for term, q_count in q_terms.items():
            if term in terms:
                score += (1 + math.log(terms[term])) * q_count
        if score > 0:
            scores[chunk["id"]] = score
    return scores


CHROMA_DIR = STORAGE_DIR / "chroma"


class ChromaVectorIndex:
    """Thin wrapper over a local persistent Chroma collection - a real ANN-capable vector
    store, replacing the pure-Python linear cosine scan over JSON-decoded embeddings. Kept
    intentionally dumb: chunk text/metadata stays in the graph store (SQLite/Neo4j) as the
    source of truth; Chroma only ever holds id + embedding + enough metadata to be useful
    standalone, and is queried purely for "which chunk ids are nearest," nothing else."""

    def __init__(self):
        import chromadb  # deferred: only required when this backend is actually used

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(name="chunks", metadata={"hnsw:space": "cosine"})

    def upsert(self, chunk_id: str, embedding: List[float], metadata: Dict) -> None:
        self.collection.upsert(ids=[chunk_id], embeddings=[embedding], metadatas=[metadata])

    def query(self, embedding: List[float], top_k: int) -> Dict[str, float]:
        count = self.collection.count()
        if count == 0:
            return {}
        result = self.collection.query(query_embeddings=[embedding], n_results=min(top_k, count))
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        # cosine space: distance = 1 - cosine_similarity, so similarity = 1 - distance
        return {chunk_id: max(0.0, 1.0 - dist) for chunk_id, dist in zip(ids, distances)}

    def reset(self) -> None:
        self.client.delete_collection("chunks")
        self.collection = self.client.get_or_create_collection(name="chunks", metadata={"hnsw:space": "cosine"})


_VECTOR_INDEX = None
_VECTOR_INDEX_READY = False


def get_vector_index() -> Optional[ChromaVectorIndex]:
    """Lazily create (once per process) and cache the Chroma index, same auto/none pattern
    as GRAPH_BACKEND: try it if available, silently fall back to the in-row JSON scan if
    chromadb isn't installed or anything about it fails, and never once fail the request."""
    global _VECTOR_INDEX, _VECTOR_INDEX_READY
    if _VECTOR_INDEX_READY:
        return _VECTOR_INDEX
    _VECTOR_INDEX_READY = True
    if os.getenv("VECTOR_BACKEND", "auto").strip().lower() == "none":
        return None
    try:
        _VECTOR_INDEX = ChromaVectorIndex()
        print(f"[vector] using Chroma at {CHROMA_DIR}")
    except ImportError:
        print("[vector] chromadb not installed (pip install chromadb) - using in-row embedding scan.")
    except Exception as exc:
        print(f"[vector] could not start Chroma ({exc}) - using in-row embedding scan.")
    return _VECTOR_INDEX


def _vector_score_chunks_scan(chunks: List[Dict], question_embedding: List[float]) -> Dict[str, float]:
    scores = {}
    for chunk in chunks:
        raw = chunk.get("embedding")
        if not raw:
            continue
        try:
            vec = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        similarity = cosine_similarity(question_embedding, vec)
        if similarity > 0:
            scores[chunk["id"]] = similarity
    return scores


def vector_score_chunks(chunks: List[Dict], question_embedding: Optional[List[float]]) -> Dict[str, float]:
    if not question_embedding:
        return {}
    candidate_ids = {chunk["id"] for chunk in chunks}
    index = get_vector_index()
    if index is not None:
        try:
            # `chunks` may already be a restricted subset (e.g. only chunks linked to a
            # matched company node) - asking Chroma for only len(candidate_ids) nearest
            # neighbors would search the WHOLE collection and could miss that specific
            # subset entirely. query() clamps to the true collection size internally, so
            # asking for an intentionally oversized top_k safely means "give me every
            # vector, ranked" and filtering to candidate_ids afterward stays correct.
            raw_scores = index.query(question_embedding, top_k=100_000)
            return {cid: score for cid, score in raw_scores.items() if cid in candidate_ids and score > 0}
        except Exception as exc:
            print(f"[vector] Chroma query failed ({exc}) - falling back to in-row scan for this question.")
    return _vector_score_chunks_scan(chunks, question_embedding)


def reciprocal_rank_fusion(*score_maps: Dict[str, float]) -> Counter:
    fused = Counter()
    for score_map in score_maps:
        for rank, chunk_id in enumerate(sorted(score_map, key=score_map.get, reverse=True), start=1):
            fused[chunk_id] += 1.0 / (RRF_K + rank)
    return fused


def rank_chunks(chunks: List[Dict], question: str, preferred_chunk_ids: set, limit: int) -> List[Dict]:
    """Hybrid retrieval: fuse lexical (term-frequency) and semantic (embedding cosine
    similarity) rankings via reciprocal rank fusion, then boost chunks already linked to the
    graph-matched entity. If no embedding provider is configured/reachable, this is exactly
    the original keyword-only behavior - RRF over a single ranking preserves its order."""
    q_terms = expand_query_terms(question)
    if not q_terms:
        return []
    keyword_scores = keyword_score_chunks(chunks, q_terms)
    if not keyword_scores:
        return []
    question_embedding = embed_text(question)
    vector_scores = vector_score_chunks(chunks, question_embedding)
    fused = reciprocal_rank_fusion(keyword_scores, vector_scores) if vector_scores else Counter(keyword_scores)

    for chunk_id in preferred_chunk_ids:
        if chunk_id in keyword_scores or chunk_id in vector_scores:
            fused[chunk_id] += 1.0  # RRF scores top out well under 1.0, so this reliably wins ties

    chunk_by_id = {chunk["id"]: chunk for chunk in chunks}
    ranked_ids = [cid for cid, _ in fused.most_common(limit) if cid in chunk_by_id]
    results = []
    for chunk_id in ranked_ids:
        matched_by = "hybrid" if chunk_id in keyword_scores and chunk_id in vector_scores else (
            "vector" if chunk_id in vector_scores else "keyword"
        )
        results.append({**chunk_by_id[chunk_id], "score": round(fused[chunk_id], 4), "matched_by": matched_by})
    return results


def contains_any(text_l: str, phrases: Iterable[str]) -> bool:
    return any(phrase.lower() in text_l for phrase in phrases)


def detect_company_from_title(title: str) -> Optional[str]:
    title_l = title.lower()
    for company, aliases in COMPANY_ALIASES.items():
        if contains_any(title_l, aliases):
            return company
    return None


def split_chunks(text: str, max_words: int = 180) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


class GraphStore:
    def __init__(self, db_path: Path = DB_PATH):
        STORAGE_DIR.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.init_schema()

    def init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    name TEXT NOT NULL,
                    aliases TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    confidence REAL DEFAULT 0.6,
                    evidence TEXT DEFAULT '',
                    source_ref TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE(source_id, relation, target_id, source_ref)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT DEFAULT '',
                    published_at TEXT DEFAULT '',
                    text TEXT NOT NULL,
                    terms TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunk_nodes (
                    chunk_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    PRIMARY KEY(chunk_id, node_id)
                );
                """
            )
            existing_columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(chunks)")}
            if "embedding" not in existing_columns:
                self.conn.execute("ALTER TABLE chunks ADD COLUMN embedding TEXT DEFAULT ''")

    def reset(self) -> None:
        with self.lock, self.conn:
            self.conn.executescript(
                """
                DELETE FROM chunk_nodes;
                DELETE FROM edges;
                DELETE FROM chunks;
                DELETE FROM nodes;
                """
            )

    def close(self) -> None:
        self.conn.close()

    def upsert_node(self, label: str, name: str, aliases: Optional[List[str]] = None) -> str:
        name = normalize_name(name)
        node_id = canonical_key(label, name)
        aliases = aliases or []
        with self.lock, self.conn:
            existing = self.conn.execute("SELECT aliases FROM nodes WHERE id = ?", (node_id,)).fetchone()
            if existing:
                current = set(json.loads(existing["aliases"] or "[]"))
                current.update(aliases)
                self.conn.execute("UPDATE nodes SET aliases = ? WHERE id = ?", (json.dumps(sorted(current)), node_id))
            else:
                self.conn.execute(
                    "INSERT INTO nodes(id, label, name, aliases, created_at) VALUES (?, ?, ?, ?, ?)",
                    (node_id, label, name, json.dumps(sorted(set(aliases))), now_iso()),
                )
        return node_id

    def upsert_edge(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        confidence: float,
        evidence: str,
        source_ref: str,
    ) -> str:
        edge_id = hash_text(f"{source_id}|{relation}|{target_id}|{source_ref}")[:24]
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO edges(id, source_id, relation, target_id, confidence, evidence, source_ref, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (edge_id, source_id, relation, target_id, confidence, evidence[:900], source_ref, now_iso()),
            )
        return edge_id

    def upsert_chunk(
        self,
        source_type: str,
        title: str,
        text: str,
        url: str = "",
        published_at: str = "",
        embedding: str = "",
    ) -> str:
        chunk_id = hash_text(f"{source_type}|{title}|{url}|{text}")[:24]
        terms = Counter(tokenize(text))
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO chunks(id, source_type, title, url, published_at, text, terms, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, source_type, title, url, published_at, text, json.dumps(terms), embedding, now_iso()),
            )
        return chunk_id

    def link_chunk_node(self, chunk_id: str, node_id: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO chunk_nodes(chunk_id, node_id) VALUES (?, ?)",
                (chunk_id, node_id),
            )

    def count(self, table: str) -> int:
        row = self.conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
        return int(row["c"])

    def nodes(self, label: Optional[str] = None) -> List[sqlite3.Row]:
        if label:
            return self.conn.execute("SELECT * FROM nodes WHERE label = ? ORDER BY name", (label,)).fetchall()
        return self.conn.execute("SELECT * FROM nodes ORDER BY label, name").fetchall()

    def find_company(self, query: str) -> Optional[sqlite3.Row]:
        query_l = query.lower()
        for node in self.nodes("Company"):
            aliases = json.loads(node["aliases"] or "[]")
            names = [node["name"].lower()] + [a.lower() for a in aliases]
            if any(alias and alias in query_l for alias in names):
                return node
        return None

    def graph_for_company(self, company_name: str, depth: int = 3) -> Dict:
        company = self.find_company(company_name)
        if not company:
            return {"nodes": [], "edges": [], "paths": []}
        paths = self.traverse(company["id"], depth=depth)
        node_ids = {company["id"]}
        edge_rows = []
        for path in paths:
            for edge in path:
                node_ids.add(edge["source_id"])
                node_ids.add(edge["target_id"])
                edge_rows.append(edge)
        nodes = []
        if node_ids:
            placeholders = ",".join("?" for _ in node_ids)
            nodes = self.conn.execute(f"SELECT * FROM nodes WHERE id IN ({placeholders})", tuple(node_ids)).fetchall()
        unique_edges = {edge["id"]: edge for edge in edge_rows}
        return {
            "nodes": [dict(row) for row in nodes],
            "edges": [dict(row) for row in unique_edges.values()],
            "paths": [[dict(e) for e in path] for path in paths],
        }

    def traverse(self, start_id: str, depth: int = 3, limit: int = 40) -> List[List[sqlite3.Row]]:
        queue = deque([(start_id, [], {start_id})])
        paths = []
        while queue and len(paths) < limit:
            node_id, path, visited = queue.popleft()
            if len(path) >= depth:
                continue
            edges = self.conn.execute(
                """
                SELECT e.*, s.name AS source_name, s.label AS source_label, t.name AS target_name, t.label AS target_label
                FROM edges e
                JOIN nodes s ON s.id = e.source_id
                JOIN nodes t ON t.id = e.target_id
                WHERE e.source_id = ? OR e.target_id = ?
                ORDER BY e.confidence DESC
                LIMIT 30
                """,
                (node_id, node_id),
            ).fetchall()
            for edge in edges:
                next_id = edge["target_id"] if edge["source_id"] == node_id else edge["source_id"]
                if next_id in visited:
                    continue
                next_path = path + [edge]
                paths.append(next_path)
                queue.append((next_id, next_path, visited | {next_id}))
        return paths

    def search_chunks(self, question: str, limit: int = 5, preferred_node_id: Optional[str] = None) -> List[Dict]:
        rows = [dict(row) for row in self.conn.execute("SELECT * FROM chunks").fetchall()]
        preferred_chunks = set()
        if preferred_node_id:
            preferred_chunks = {
                row["chunk_id"]
                for row in self.conn.execute("SELECT chunk_id FROM chunk_nodes WHERE node_id = ?", (preferred_node_id,))
            }
            if preferred_chunks:
                rows = [row for row in rows if row["id"] in preferred_chunks]
        return rank_chunks(rows, question, preferred_chunks, limit)


EXTRACTION_LABELS = ["Company", "Supplier", "Product", "Country", "RiskEvent", "NewsArticle"]
EXTRACTION_RELATIONS = ["RELIES_ON", "DEPENDS_ON", "OPERATES_IN", "EXPOSED_TO", "REPORTS", "AFFECTS"]

# A weak/local model will sometimes emit a relation with plausible-looking but semantically
# backwards or nonsensical label pairs (e.g. RiskEvent -[REPORTS]-> NewsArticle). Enum
# membership alone doesn't catch that, so every (source_label, target_label) pair is checked
# against this whitelist before an edge is accepted.
RELATION_SIGNATURES = {
    "RELIES_ON": {("Company", "Company"), ("Company", "Supplier")},
    "DEPENDS_ON": {("Company", "Product")},
    "OPERATES_IN": {("Company", "Country")},
    "EXPOSED_TO": {("Company", "RiskEvent")},
    "REPORTS": {("NewsArticle", "RiskEvent")},
    "AFFECTS": {("RiskEvent", "Product"), ("RiskEvent", "Country")},
}

CORP_SUFFIX_RE = re.compile(r"[,]?\s*(incorporated|inc\.?|corporation|corp\.?|co\.?|ltd\.?|llc|plc|company)\s*$", re.IGNORECASE)

EXTRACTION_INSTRUCTIONS = f"""
You are a financial-risk knowledge graph extraction engine. Read the supplied text (a chunk
from an SEC filing or a news article) and extract supply-chain and risk relationships as a
strict JSON array. Output ONLY the JSON array - no prose, no markdown code fences.

Each array item must be an object with exactly these keys:
  source: string, full consistent entity name (e.g. "Apple Inc.", "Taiwan Semiconductor Manufacturing Company")
  source_label: one of {EXTRACTION_LABELS}
  relation: one of {EXTRACTION_RELATIONS}
  target: string, full consistent entity name
  target_label: one of {EXTRACTION_LABELS}
  confidence: number between 0 and 1, how directly the text states this relationship

Relation meanings:
  RELIES_ON: a Company relies on another Company/Supplier for manufacturing, parts, or fabrication.
  DEPENDS_ON: a Company depends on a Product or material category (e.g. semiconductors, lithium, batteries).
  OPERATES_IN: a Company has operations, manufacturing, or material exposure in a Country.
  EXPOSED_TO: a Company is exposed to a RiskEvent (e.g. "Semiconductor shortage", "Export controls", "Tariff risk").
  REPORTS: only for news articles - the NewsArticle (use the article title as source) reports on a RiskEvent.
  AFFECTS: only for news articles - a RiskEvent affects a Product or a Country.

Rules:
- Extract only relationships stated or clearly implied by the text. Never invent facts not present in the text.
- Use short, canonical risk names for RiskEvent (e.g. "Semiconductor shortage", not a full sentence).
- If the text is a filing (not news), do not emit NewsArticle/REPORTS/AFFECTS relationships.
- Return at most 12 relationships. Return [] if none apply.
""".strip()


def _strip_corp_suffix(name_l: str) -> str:
    prev = None
    while prev != name_l:
        prev = name_l
        name_l = CORP_SUFFIX_RE.sub("", name_l).strip()
    return name_l


def canonicalize_entity_name(label: str, name: str) -> str:
    name = normalize_name(name)
    name_l = name.lower()
    if label in ("Company", "Supplier"):
        bare = _strip_corp_suffix(name_l)
        for canonical, aliases in {**COMPANY_ALIASES, **SUPPLIERS}.items():
            candidates = [canonical.lower()] + [a.lower() for a in aliases]
            if name_l in candidates or bare in {_strip_corp_suffix(c) for c in candidates}:
                return canonical
    elif label == "Product":
        for canonical, aliases in PRODUCTS.items():
            if name_l == canonical.lower() or name_l in [a.lower() for a in aliases]:
                return canonical
    elif label == "RiskEvent":
        for canonical, phrases in RISK_KEYWORDS.items():
            if name_l == canonical.lower() or name_l in [p.lower() for p in phrases]:
                return canonical
    elif label == "Country":
        for canonical in COUNTRIES:
            if name_l == canonical.lower():
                return canonical
    return name


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text)
    return text.strip()


def parse_llm_relationships(raw_text: str, source_type: str, source_ref: str, source_text: str) -> Optional[List[Dict]]:
    cleaned = _strip_json_fence(raw_text)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        items = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None

    evidence = source_text[:700].strip()
    relationships = []
    for item in items[:15]:
        if not isinstance(item, dict):
            continue
        source_label = str(item.get("source_label", "")).strip()
        target_label = str(item.get("target_label", "")).strip()
        relation = str(item.get("relation", "")).strip().upper()
        source = normalize_name(str(item.get("source", "")))
        target = normalize_name(str(item.get("target", "")))
        if source_label not in EXTRACTION_LABELS or target_label not in EXTRACTION_LABELS:
            continue
        if relation not in EXTRACTION_RELATIONS:
            continue
        if (source_label, target_label) not in RELATION_SIGNATURES[relation]:
            continue
        if not source or not target or source.lower() == target.lower():
            continue
        if source_type != "news" and (
            relation in {"REPORTS", "AFFECTS"} or "NewsArticle" in (source_label, target_label)
        ):
            continue
        try:
            confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = max(0.05, min(0.95, confidence))
        relationships.append(
            {
                "source_label": source_label,
                "source": canonicalize_entity_name(source_label, source),
                "relation": relation,
                "target_label": target_label,
                "target": canonicalize_entity_name(target_label, target),
                "confidence": round(confidence, 2),
                "evidence": evidence,
                "source_ref": source_ref,
            }
        )
    return relationships


def llm_extract_relationships(text: str, source_type: str, title: str, source_ref: str) -> Optional[List[Dict]]:
    if not llm_status()["enabled"]:
        return None
    prompt = f"Source type: {source_type}\nTitle: {title}\n\nText:\n{text}"
    raw_text, _error, _mode = call_llm_raw(EXTRACTION_INSTRUCTIONS, prompt)
    if not raw_text:
        return None
    return parse_llm_relationships(raw_text, source_type, source_ref, text)


def extract_relationships_hybrid(text: str, source_type: str, title: str, source_ref: str) -> Tuple[List[Dict], str]:
    llm_result = llm_extract_relationships(text, source_type, title, source_ref)
    if llm_result is not None:
        return llm_result, "llm"
    return extract_relationships(text, source_type, title, source_ref), "rule"


def extract_relationships(text: str, source_type: str, title: str, source_ref: str) -> List[Dict]:
    text_l = text.lower()
    relationships = []

    primary_company = detect_company_from_title(title) if source_type == "filing" else None
    companies = [primary_company] if primary_company else []
    if not companies:
        for company, aliases in COMPANY_ALIASES.items():
            if contains_any(text_l, aliases):
                companies.append(company)

    suppliers = []
    for supplier, aliases in SUPPLIERS.items():
        if contains_any(text_l, aliases):
            suppliers.append(supplier)

    products = []
    for product, aliases in PRODUCTS.items():
        if contains_any(text_l, aliases):
            products.append(product)

    countries = [country for country in COUNTRIES if country.lower() in text_l]

    risks = []
    for risk, phrases in RISK_KEYWORDS.items():
        if contains_any(text_l, phrases):
            risks.append(risk)

    if "third-party" in text_l and ("supplier" in text_l or "manufacturer" in text_l):
        suppliers.append("Third-party manufacturers")
        risks.append("Supplier concentration")

    evidence = text[:700].strip()

    for company in companies:
        for supplier in suppliers:
            if company != supplier:
                relationships.append(
                    {
                        "source_label": "Company",
                        "source": company,
                        "relation": "RELIES_ON",
                        "target_label": "Supplier" if supplier == "Third-party manufacturers" else "Company",
                        "target": supplier,
                        "confidence": 0.72,
                        "evidence": evidence,
                        "source_ref": source_ref,
                    }
                )
        for product in products:
            relationships.append(
                {
                    "source_label": "Company",
                    "source": company,
                    "relation": "DEPENDS_ON",
                    "target_label": "Product",
                    "target": product,
                    "confidence": 0.7,
                    "evidence": evidence,
                    "source_ref": source_ref,
                }
            )
        for country in countries:
            relationships.append(
                {
                    "source_label": "Company",
                    "source": company,
                    "relation": "OPERATES_IN",
                    "target_label": "Country",
                    "target": country,
                    "confidence": 0.58,
                    "evidence": evidence,
                    "source_ref": source_ref,
                }
            )
        for risk in risks:
            relationships.append(
                {
                    "source_label": "Company",
                    "source": company,
                    "relation": "EXPOSED_TO",
                    "target_label": "RiskEvent",
                    "target": risk,
                    "confidence": 0.76 if source_type == "news" else 0.68,
                    "evidence": evidence,
                    "source_ref": source_ref,
                }
            )

    if source_type == "news" and risks:
        article_name = title[:160]
        for risk in risks:
            relationships.append(
                {
                    "source_label": "NewsArticle",
                    "source": article_name,
                    "relation": "REPORTS",
                    "target_label": "RiskEvent",
                    "target": risk,
                    "confidence": 0.82,
                    "evidence": evidence,
                    "source_ref": source_ref,
                }
            )
            for product in products:
                relationships.append(
                    {
                        "source_label": "RiskEvent",
                        "source": risk,
                        "relation": "AFFECTS",
                        "target_label": "Product",
                        "target": product,
                        "confidence": 0.72,
                        "evidence": evidence,
                        "source_ref": source_ref,
                    }
                )
            for country in countries:
                relationships.append(
                    {
                        "source_label": "RiskEvent",
                        "source": risk,
                        "relation": "AFFECTS",
                        "target_label": "Country",
                        "target": country,
                        "confidence": 0.66,
                        "evidence": evidence,
                        "source_ref": source_ref,
                    }
                )

    return relationships


NEO4J_NODE_LABELS = set(EXTRACTION_LABELS)  # whitelist - Cypher can't parameterize labels, so
                                             # anything interpolated into a label/rel-type string
                                             # must be checked against a fixed set first, since
                                             # these values ultimately originate from LLM output.


class Neo4jGraphStore:
    """Graph backend on real Neo4j. Same public interface as GraphStore (SQLite) so the rest
    of the app - ingestion, traversal, retrieval, the API - doesn't know or care which one is
    active. Chunks and their term-frequency scoring stay identical to the SQLite version; only
    node/edge storage and traversal become native Cypher instead of hand-rolled BFS-over-SQL."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase  # deferred: only required when this backend is used

        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.init_schema()

    def init_schema(self) -> None:
        with self.driver.session(database=self.database) as session:
            for label in NEO4J_NODE_LABELS:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")

    def reset(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        self.driver.close()

    def upsert_node(self, label: str, name: str, aliases: Optional[List[str]] = None) -> str:
        if label not in NEO4J_NODE_LABELS:
            raise ValueError(f"Refusing to write unknown node label: {label!r}")
        name = normalize_name(name)
        node_id = canonical_key(label, name)
        aliases = set(aliases or [])
        with self.driver.session(database=self.database) as session:
            existing = session.run(f"MATCH (n:{label} {{id: $id}}) RETURN n.aliases AS aliases", id=node_id).single()
            if existing and existing["aliases"]:
                aliases.update(json.loads(existing["aliases"]))
            session.run(
                f"""
                MERGE (n:{label} {{id: $id}})
                ON CREATE SET n.name = $name, n.aliases = $aliases, n.created_at = $now
                ON MATCH SET n.aliases = $aliases
                """,
                id=node_id,
                name=name,
                aliases=json.dumps(sorted(aliases)),
                now=now_iso(),
            )
        return node_id

    def upsert_edge(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        confidence: float,
        evidence: str,
        source_ref: str,
    ) -> str:
        if relation not in EXTRACTION_RELATIONS:
            raise ValueError(f"Refusing to write unknown relation type: {relation!r}")
        edge_id = hash_text(f"{source_id}|{relation}|{target_id}|{source_ref}")[:24]
        with self.driver.session(database=self.database) as session:
            session.run(
                f"""
                MATCH (s {{id: $source_id}}), (t {{id: $target_id}})
                MERGE (s)-[r:{relation} {{source_ref: $source_ref}}]->(t)
                ON CREATE SET r.id = $edge_id, r.confidence = $confidence,
                              r.evidence = $evidence, r.created_at = $now
                """,
                source_id=source_id,
                target_id=target_id,
                source_ref=source_ref,
                edge_id=edge_id,
                confidence=confidence,
                evidence=evidence[:900],
                now=now_iso(),
            )
        return edge_id

    def upsert_chunk(
        self,
        source_type: str,
        title: str,
        text: str,
        url: str = "",
        published_at: str = "",
        embedding: str = "",
    ) -> str:
        chunk_id = hash_text(f"{source_type}|{title}|{url}|{text}")[:24]
        terms = Counter(tokenize(text))
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (c:Chunk {id: $id})
                ON CREATE SET c.source_type = $source_type, c.title = $title, c.url = $url,
                              c.published_at = $published_at, c.text = $text,
                              c.terms = $terms, c.embedding = $embedding, c.created_at = $now
                """,
                id=chunk_id,
                source_type=source_type,
                title=title,
                url=url,
                published_at=published_at,
                text=text,
                terms=json.dumps(terms),
                embedding=embedding,
                now=now_iso(),
            )
        return chunk_id

    def link_chunk_node(self, chunk_id: str, node_id: str) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (c:Chunk {id: $chunk_id}), (n {id: $node_id})
                MERGE (c)-[:MENTIONS]->(n)
                """,
                chunk_id=chunk_id,
                node_id=node_id,
            )

    def count(self, table: str) -> int:
        with self.driver.session(database=self.database) as session:
            if table == "chunks":
                record = session.run("MATCH (c:Chunk) RETURN count(c) AS c").single()
            elif table == "edges":
                record = session.run(
                    "MATCH ()-[r]->() WHERE type(r) <> 'MENTIONS' RETURN count(r) AS c"
                ).single()
            else:
                record = session.run("MATCH (n) WHERE NOT n:Chunk RETURN count(n) AS c").single()
            return int(record["c"])

    def nodes(self, label: Optional[str] = None) -> List[Dict]:
        # Project explicit scalars from Cypher rather than returning raw node objects - the
        # driver's `.data()` strips label metadata off Node objects, so asking Cypher for
        # `labels(n)[0]` directly is more robust than introspecting driver objects in Python.
        if label:
            query = "MATCH (n) WHERE $label IN labels(n) RETURN n.id AS id, $label AS label, n.name AS name, n.aliases AS aliases"
        else:
            query = "MATCH (n) WHERE NOT n:Chunk RETURN n.id AS id, labels(n)[0] AS label, n.name AS name, n.aliases AS aliases"
        with self.driver.session(database=self.database) as session:
            rows = session.run(query, label=label).data()
        return sorted(rows, key=lambda r: (r["label"], r["name"]))

    def find_company(self, query: str) -> Optional[Dict]:
        query_l = query.lower()
        for node in self.nodes("Company"):
            aliases = json.loads(node["aliases"] or "[]")
            names = [node["name"].lower()] + [a.lower() for a in aliases]
            if any(alias and alias in query_l for alias in names):
                return node
        return None

    def graph_for_company(self, company_name: str, depth: int = 3) -> Dict:
        company = self.find_company(company_name)
        if not company:
            return {"nodes": [], "edges": [], "paths": []}

        paths = self.traverse(company["id"], depth=depth)
        node_ids = {company["id"]}
        edge_rows = []
        for path in paths:
            for edge in path:
                node_ids.add(edge["source_id"])
                node_ids.add(edge["target_id"])
                edge_rows.append(edge)
        unique_nodes = {}
        for edge in edge_rows:
            unique_nodes[edge["source_id"]] = {
                "id": edge["source_id"],
                "label": edge["source_label"],
                "name": edge["source_name"],
            }
            unique_nodes[edge["target_id"]] = {
                "id": edge["target_id"],
                "label": edge["target_label"],
                "name": edge["target_name"],
            }
        unique_edges = {edge["id"]: edge for edge in edge_rows}
        return {
            "nodes": list(unique_nodes.values()),
            "edges": list(unique_edges.values()),
            "paths": paths,
        }

    def traverse(self, start_id: str, depth: int = 3, limit: int = 40) -> List[List[Dict]]:
        # A single variable-length Cypher path query replaces the hand-rolled BFS the SQLite
        # backend needs - this is the traversal Neo4j is actually built for. `depth` is an
        # int we control (not user input), so it's safe to interpolate into the range bound;
        # Cypher has no parameter syntax for that position. The relationship list is projected
        # into plain maps inside Cypher itself (via startNode/endNode/labels/type) rather than
        # returned as raw Path/Relationship objects, so the Python side only ever handles
        # ordinary dicts regardless of driver version.
        query = f"""
        MATCH p = (start {{id: $start_id}})-[rels*1..{depth}]-(other)
        WHERE start.id <> other.id AND ALL(rel IN rels WHERE type(rel) <> 'MENTIONS')
        RETURN [rel IN rels | {{
            id: rel.id,
            source_id: startNode(rel).id,
            source_name: startNode(rel).name,
            source_label: labels(startNode(rel))[0],
            relation: type(rel),
            target_id: endNode(rel).id,
            target_name: endNode(rel).name,
            target_label: labels(endNode(rel))[0],
            confidence: rel.confidence,
            evidence: rel.evidence,
            source_ref: rel.source_ref
        }}] AS edges
        ORDER BY length(p) ASC
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            records = session.run(query, start_id=start_id, limit=limit).data()
        return [record["edges"] for record in records]

    def search_chunks(self, question: str, limit: int = 5, preferred_node_id: Optional[str] = None) -> List[Dict]:
        with self.driver.session(database=self.database) as session:
            rows = session.run("MATCH (c:Chunk) RETURN c").data()
            preferred_chunks = set()
            if preferred_node_id:
                preferred_chunks = {
                    record["c"]["id"]
                    for record in session.run(
                        "MATCH (c:Chunk)-[:MENTIONS]->(n {id: $node_id}) RETURN c",
                        node_id=preferred_node_id,
                    ).data()
                }

        chunks = [dict(record["c"]) for record in rows]
        if preferred_chunks:
            chunks = [c for c in chunks if c["id"] in preferred_chunks] or chunks
        return rank_chunks(chunks, question, preferred_chunks, limit)


def connect_neo4j_store() -> Optional[Neo4jGraphStore]:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    try:
        return Neo4jGraphStore(uri, user, password, database)
    except ImportError:
        print("[graph] neo4j driver not installed (pip install neo4j) - using local SQLite graph store.")
        return None
    except Exception as exc:
        print(f"[graph] could not connect to Neo4j at {uri} ({exc}) - using local SQLite graph store.")
        return None


def build_store():
    """Backend selection follows the same graceful-degradation pattern as the LLM calls:
    prefer Neo4j when GRAPH_BACKEND asks for or allows it and a server is reachable, otherwise
    fall back to the dependency-free SQLite store so the app always runs with zero setup."""
    backend = os.getenv("GRAPH_BACKEND", "auto").strip().lower()
    if backend in {"auto", "neo4j"}:
        store = connect_neo4j_store()
        if store:
            print(f"[graph] using Neo4j at {os.getenv('NEO4J_URI', 'bolt://localhost:7687')}")
            return store
        if backend == "neo4j":
            raise RuntimeError("GRAPH_BACKEND=neo4j but no Neo4j server could be reached.")
    print("[graph] using local SQLite graph store.")
    return GraphStore()


def ingest_text(store: GraphStore, source_type: str, title: str, text: str, url: str = "", published_at: str = "") -> Dict:
    chunks = split_chunks(text)
    total_edges = 0
    total_chunks = 0
    embedded_chunks = 0
    method_counts = Counter()
    for chunk in chunks:
        embedding_vec = embed_text(chunk)
        embedding_json = json.dumps(embedding_vec) if embedding_vec else ""
        if embedding_vec:
            embedded_chunks += 1
        chunk_id = store.upsert_chunk(source_type, title, chunk, url, published_at, embedding_json)
        total_chunks += 1
        if embedding_vec:
            vector_index = get_vector_index()
            if vector_index is not None:
                try:
                    vector_index.upsert(chunk_id, embedding_vec, {"title": title, "source_type": source_type, "url": url})
                except Exception as exc:
                    print(f"[vector] Chroma upsert failed for a chunk ({exc}) - it stays searchable via the in-row scan.")
        source_ref = url or title
        relationships, method = extract_relationships_hybrid(chunk, source_type, title, source_ref)
        method_counts[method] += 1
        for rel in relationships:
            source_id = store.upsert_node(rel["source_label"], rel["source"], COMPANY_ALIASES.get(rel["source"], []))
            target_id = store.upsert_node(rel["target_label"], rel["target"], COMPANY_ALIASES.get(rel["target"], []))
            store.upsert_edge(source_id, rel["relation"], target_id, rel["confidence"], rel["evidence"], rel["source_ref"])
            store.link_chunk_node(chunk_id, source_id)
            store.link_chunk_node(chunk_id, target_id)
            total_edges += 1
    return {
        "chunks": total_chunks,
        "edges": total_edges,
        "llm_chunks": method_counts["llm"],
        "rule_chunks": method_counts["rule"],
        "embedded_chunks": embedded_chunks,
    }


def ingest_filings(store: GraphStore) -> Dict:
    total = {"files": 0, "chunks": 0, "edges": 0, "llm_chunks": 0, "rule_chunks": 0, "embedded_chunks": 0}
    for path in sorted(FILINGS_DIR.glob("*.txt")):
        result = ingest_text(store, "filing", path.stem.replace("_", " "), path.read_text(encoding="utf-8"))
        total["files"] += 1
        total["chunks"] += result["chunks"]
        total["edges"] += result["edges"]
        total["llm_chunks"] += result["llm_chunks"]
        total["rule_chunks"] += result["rule_chunks"]
        total["embedded_chunks"] += result["embedded_chunks"]
    return total


def ingest_news_file(store: GraphStore) -> Dict:
    total = {"articles": 0, "chunks": 0, "edges": 0, "llm_chunks": 0, "rule_chunks": 0, "embedded_chunks": 0}
    for path in sorted(NEWS_DIR.glob("*.json")):
        articles = json.loads(path.read_text(encoding="utf-8"))
        for article in articles:
            text = " ".join(
                [
                    article.get("title", ""),
                    article.get("description", ""),
                    article.get("content", ""),
                ]
            ).strip()
            if not text:
                continue
            result = ingest_text(
                store,
                "news",
                article.get("title", "Untitled article"),
                text,
                article.get("url", ""),
                article.get("published_at", ""),
            )
            total["articles"] += 1
            total["chunks"] += result["chunks"]
            total["edges"] += result["edges"]
            total["llm_chunks"] += result["llm_chunks"]
            total["rule_chunks"] += result["rule_chunks"]
            total["embedded_chunks"] += result["embedded_chunks"]
    return total


def ingest_rss(store: GraphStore, rss_url: str) -> Dict:
    with urllib.request.urlopen(rss_url, timeout=20) as response:
        raw = response.read()
    root = ET.fromstring(raw)
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    total = {"articles": 0, "chunks": 0, "edges": 0, "llm_chunks": 0, "rule_chunks": 0, "embedded_chunks": 0}
    for item in items[:30]:
        title = (item.findtext("title") or item.findtext("{http://www.w3.org/2005/Atom}title") or "Untitled").strip()
        link = item.findtext("link") or ""
        atom_link = item.find("{http://www.w3.org/2005/Atom}link")
        if atom_link is not None and atom_link.attrib.get("href"):
            link = atom_link.attrib["href"]
        description = item.findtext("description") or item.findtext("summary") or ""
        published = item.findtext("pubDate") or item.findtext("{http://www.w3.org/2005/Atom}updated") or now_iso()
        text = f"{title}. {description}"
        result = ingest_text(store, "news", title, text, link, published)
        total["articles"] += 1
        total["chunks"] += result["chunks"]
        total["edges"] += result["edges"]
        total["llm_chunks"] += result["llm_chunks"]
        total["rule_chunks"] += result["rule_chunks"]
        total["embedded_chunks"] += result["embedded_chunks"]
    return total


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def ollama_is_running(timeout: float = 0.35) -> bool:
    try:
        with urllib.request.urlopen(f"{ollama_base_url()}/api/tags", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def llm_status() -> Dict:
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    ollama_running = ollama_is_running()
    ollama_model = os.getenv("OLLAMA_MODEL", "phi3.5")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if provider == "ollama":
        enabled = ollama_running
        active = "Ollama" if ollama_running else "Ollama not running"
        model = ollama_model
    elif provider == "openai":
        enabled = bool(openai_key)
        active = "OpenAI Responses API" if openai_key else "OpenAI key missing"
        model = openai_model if openai_key else ""
    elif provider == "gemini":
        enabled = bool(gemini_key)
        active = "Gemini" if gemini_key else "Gemini key missing"
        model = gemini_model_name if gemini_key else ""
    else:
        enabled = ollama_running or bool(openai_key) or bool(gemini_key)
        if ollama_running:
            active = "Ollama"
            model = ollama_model
        elif openai_key:
            active = "OpenAI Responses API"
            model = openai_model
        elif gemini_key:
            active = "Gemini"
            model = gemini_model_name
        else:
            active = "local fallback"
            model = ""

    return {
        "enabled": enabled,
        "provider": active,
        "configured_provider": provider,
        "model": model,
        "ollama_base_url": ollama_base_url(),
        "agent_enabled": bool(gemini_key),
    }


def extract_response_text(payload: Dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text") or content.get("output_text")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def build_llm_context(question: str, graph_paths: List[str], evidence: List[Dict]) -> Tuple[str, str]:
    evidence_block = "\n\n".join(
        [
            (
                f"Evidence {idx + 1}\n"
                f"Title: {item['title']}\n"
                f"Type: {item['source_type']}\n"
                f"Published: {item['published_at'] or 'unknown'}\n"
                f"Source: {item['url'] or item['title']}\n"
                f"Text: {item['snippet']}"
            )
            for idx, item in enumerate(evidence)
        ]
    )
    graph_block = "\n".join([f"- {path}" for path in graph_paths]) or "No graph paths found."
    prompt = f"""
Question:
{question}

Graph paths:
{graph_block}

Retrieved evidence:
{evidence_block or "No evidence snippets found."}
""".strip()

    instructions = """
You are a context-grounded financial and supply-chain Q&A assistant.
Answer only from the supplied graph paths and evidence snippets.
If the context does not contain the requested fact, say that the current context does not include it.
Do not invent current financial results, stock prices, suppliers, or news.

Output format - follow this exactly:
- 2-5 concise bullets, each citing evidence as [Evidence 1], [Evidence 2], etc.
- Optionally, one final line: "Strongest path: <the single most relevant graph path in plain English>"
- Nothing else. No introduction, no summary/conclusion section, no restating the bullets in different words afterward.

Write the answer exactly once. Do not draft it, then redo or rephrase it under a second heading like
"Answer:" or "In summary:" - if you catch yourself about to repeat a point already made, stop instead.
Do not comment on your own reasoning (e.g. "this path is more/less direct than that one") - just state
the facts.
""".strip()
    return instructions, prompt


def _ollama_chat_raw(instructions: str, prompt: str, timeout: float = 90) -> Tuple[Optional[str], Optional[str]]:
    if not ollama_is_running(timeout=0.8):
        return None, f"Ollama is not running at {ollama_base_url()}."

    model = os.getenv("OLLAMA_MODEL", "phi3.5")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        f"{ollama_base_url()}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = (result.get("message", {}) or {}).get("content", "").strip()
        if text:
            return text, None
        return None, "Ollama response did not contain message content."
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return None, f"Ollama API error {exc.code}: {detail[:500]}"
    except Exception as exc:
        return None, f"Ollama call failed: {exc}"


def _openai_respond_raw(
    instructions: str, prompt: str, timeout: float = 45, max_output_tokens: int = 700
) -> Tuple[Optional[str], Optional[str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "OPENAI_API_KEY is not set."

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    endpoint = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")

    payload = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = extract_response_text(result)
        if text:
            return text, None
        return None, "LLM response did not contain text."
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return None, f"OpenAI API error {exc.code}: {detail[:500]}"
    except Exception as exc:
        return None, f"LLM call failed: {exc}"


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def gemini_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _gemini_extract_text(result: Dict) -> str:
    try:
        parts = result["candidates"][0]["content"]["parts"]
        return "\n".join(p["text"] for p in parts if "text" in p).strip()
    except (KeyError, IndexError, TypeError):
        return ""


def _gemini_request(body: Dict, model: Optional[str] = None, timeout: float = 45) -> Tuple[Optional[Dict], Optional[str]]:
    api_key = gemini_api_key()
    if not api_key:
        return None, "GEMINI_API_KEY is not set."
    url = f"{GEMINI_API_BASE}/models/{model or gemini_model()}:generateContent?key={api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return None, f"Gemini API error {exc.code}: {detail[:500]}"
    except Exception as exc:
        return None, f"Gemini call failed: {exc}"


def _gemini_chat_raw(instructions: str, prompt: str, timeout: float = 45) -> Tuple[Optional[str], Optional[str]]:
    body = {
        "system_instruction": {"parts": [{"text": instructions}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    result, error = _gemini_request(body, timeout=timeout)
    if error:
        return None, error
    text = _gemini_extract_text(result)
    if text:
        return text, None
    return None, "Gemini response did not contain text."


def call_llm_raw(instructions: str, prompt: str) -> Tuple[Optional[str], Optional[str], str]:
    """Provider-agnostic LLM call: tries Ollama, then OpenAI, then Gemini per LLM_PROVIDER,
    same fallback order used for answer generation. Used for both Q&A and extraction."""
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    errors = []

    if provider in {"auto", "ollama"}:
        text, error = _ollama_chat_raw(instructions, prompt)
        if text:
            return text, None, "ollama"
        if error:
            errors.append(error)
        if provider == "ollama":
            return None, "; ".join(errors), "local"

    if provider in {"auto", "openai"}:
        text, error = _openai_respond_raw(instructions, prompt, max_output_tokens=900)
        if text:
            return text, None, "openai"
        if error:
            errors.append(error)
        if provider == "openai":
            return None, "; ".join(errors), "local"

    if provider in {"auto", "gemini"}:
        text, error = _gemini_chat_raw(instructions, prompt)
        if text:
            return text, None, "gemini"
        if error:
            errors.append(error)

    return None, "; ".join(errors) if errors else None, "local"


def _ollama_embed_raw(texts: List[str], timeout: float = 30) -> Optional[List[List[float]]]:
    if not ollama_is_running(timeout=0.8):
        return None
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    payload = {"model": model, "input": texts}
    request = urllib.request.Request(
        f"{ollama_base_url()}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        embeddings = result.get("embeddings")
        if embeddings and len(embeddings) == len(texts):
            return embeddings
        return None
    except Exception:
        return None


def _openai_embed_raw(texts: List[str], timeout: float = 30) -> Optional[List[List[float]]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    endpoint = os.getenv("OPENAI_EMBEDDINGS_URL", "https://api.openai.com/v1/embeddings")
    payload = {"model": model, "input": texts}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        data = result.get("data", [])
        if len(data) == len(texts):
            return [item["embedding"] for item in data]
        return None
    except Exception:
        return None


def _gemini_embed_raw(texts: List[str], timeout: float = 30) -> Optional[List[List[float]]]:
    api_key = gemini_api_key()
    if not api_key:
        return None
    model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    embeddings = []
    for text in texts:
        url = f"{GEMINI_API_BASE}/models/{model}:embedContent?key={api_key}"
        body = {"content": {"parts": [{"text": text}]}}
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            values = result.get("embedding", {}).get("values")
            if not values:
                return None
            embeddings.append(values)
        except Exception:
            return None
    return embeddings


def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Same provider-fallback shape as call_llm_raw, but for embeddings: prefer Ollama, then
    OpenAI, then Gemini, per LLM_PROVIDER. Returns None (not a list of zeros) when no embedding
    provider is available, so callers can tell 'no vector signal' apart from 'this chunk embedded
    to zero'."""
    if not texts:
        return []
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    if provider in {"auto", "ollama"}:
        result = _ollama_embed_raw(texts)
        if result:
            return result
        if provider == "ollama":
            return None
    if provider in {"auto", "openai"}:
        result = _openai_embed_raw(texts)
        if result:
            return result
        if provider == "openai":
            return None
    if provider in {"auto", "gemini"}:
        result = _gemini_embed_raw(texts)
        if result:
            return result
    return None


def embed_text(text: str) -> Optional[List[float]]:
    result = embed_texts([text])
    return result[0] if result else None


def call_ollama_llm(question: str, graph_paths: List[str], evidence: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    instructions, prompt = build_llm_context(question, graph_paths, evidence)
    return _ollama_chat_raw(instructions, prompt)


def call_openai_llm(question: str, graph_paths: List[str], evidence: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    instructions, prompt = build_llm_context(question, graph_paths, evidence)
    return _openai_respond_raw(instructions, prompt)


def call_context_llm(question: str, graph_paths: List[str], evidence: List[Dict]) -> Tuple[Optional[str], Optional[str], str]:
    instructions, prompt = build_llm_context(question, graph_paths, evidence)
    return call_llm_raw(instructions, prompt)


def local_context_answer(company: Optional[sqlite3.Row], question: str, graph_paths: List[str], evidence: List[Dict]) -> str:
    if not evidence and not graph_paths:
        known = "Apple, NVIDIA, Tesla, TSMC, Foxconn, Samsung, Panasonic, or CATL"
        return f"I do not have enough matching context to answer that. Try a question about {known}, or ingest more filings/news."

    if company:
        subject = company["name"]
    else:
        subject = "the available context"

    q_terms = set(expand_query_terms(question))
    asks_financials = bool(q_terms & {"profit", "profits", "profitability", "revenue", "earnings", "income", "sales"})
    asks_risk = bool(q_terms & {"risk", "risks", "shortage", "supplier", "disruption", "tariff", "export"})

    if asks_financials:
        intro = f"Based on the retrieved context, here is what I found about {subject}'s financial performance:"
    elif asks_risk:
        intro = f"Based on the retrieved context, here is what I found about {subject}'s supply-chain risk:"
    else:
        intro = f"Based on the retrieved context, here is what I found about {subject}:"

    lines = [intro]
    if graph_paths and not asks_financials:
        lines.append(f"- Strongest graph signal: {graph_paths[0]}")
    for idx, item in enumerate(evidence[:3]):
        lines.append(f"- Evidence {idx + 1}: {item['snippet']}")
    lines.append("- If the exact fact is not shown above, the current local context does not contain it yet.")
    return "\n".join(lines)


def build_answer(store: GraphStore, question: str) -> Dict:
    company = store.find_company(question)
    relevant_chunks = store.search_chunks(question, limit=6, preferred_node_id=company["id"] if company else None)
    graph = store.graph_for_company(company["name"] if company else question, depth=3)
    q_terms = set(expand_query_terms(question))
    asks_financials = bool(q_terms & {"profit", "profits", "profitability", "revenue", "earnings", "income", "sales"})
    asks_risk = bool(q_terms & {"risk", "risks", "shortage", "supplier", "disruption", "tariff", "export"})

    paths = []
    seen_paths = set()
    for path in graph["paths"][:8]:
        labels = []
        for edge in path:
            labels.append(f"{edge['source_name']} -[{edge['relation']}]-> {edge['target_name']}")
        if labels:
            path_text = " | ".join(labels)
            if path_text not in seen_paths:
                paths.append(path_text)
                seen_paths.add(path_text)

    evidence = []
    for chunk in relevant_chunks:
        snippet = chunk["text"][:700].strip()
        evidence.append(
            {
                "title": chunk["title"],
                "source_type": chunk["source_type"],
                "url": chunk["url"],
                "published_at": chunk["published_at"],
                "snippet": snippet,
                "score": chunk["score"],
                "matched_by": chunk.get("matched_by", "keyword"),
            }
        )

    paths_for_answer = paths if asks_risk or not asks_financials else []
    answer, llm_error, answer_mode = call_context_llm(question, paths_for_answer, evidence)
    if not answer:
        answer = local_context_answer(company, question, paths, evidence)

    return {
        "question": question,
        "company": dict(company) if company else None,
        "answer": answer,
        "answer_mode": answer_mode,
        "llm": llm_status(),
        "llm_error": llm_error,
        "graph_paths": paths,
        "evidence": evidence,
        "graph": {"nodes": graph["nodes"], "edges": graph["edges"]},
    }


AGENT_TOOLS = [
    {
        "name": "search_chunks",
        "description": (
            "Hybrid keyword+vector search over ingested SEC filing and news text chunks already "
            "in the local knowledge graph. Use this first to ground claims in ingested documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query or question"}},
            "required": ["query"],
        },
    },
    {
        "name": "find_entity",
        "description": "Resolve a company/entity name to a graph node, confirming it exists and returning its id.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Entity or company name"}},
            "required": ["name"],
        },
    },
    {
        "name": "expand_neighbors",
        "description": (
            "Walk the knowledge graph outward from an entity to find multi-hop relationships - "
            "e.g. suppliers of suppliers, or risk events connected to a company two hops away. "
            "Use this for questions a single chunk search can't answer because the connection "
            "spans more than one relationship."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_name": {"type": "string", "description": "Entity/company name to expand from"},
                "depth": {"type": "integer", "description": "Relationship hops to traverse, 1-4 (default 2)"},
            },
            "required": ["entity_name"],
        },
    },
    {
        "name": "list_entities",
        "description": "List known graph entities of a given label: Company, Supplier, Product, Country, RiskEvent, or NewsArticle.",
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string", "description": "Node label to list"}},
            "required": ["label"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the live web for recent or current news not present in the local ingested "
            "documents. Use only when the question needs up-to-date information the graph and "
            "chunk search don't have - e.g. 'any news this week about X'."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Web search query"}},
            "required": ["query"],
        },
    },
]


def web_search_live(query: str) -> str:
    """Live web search via Gemini's built-in Google Search grounding tool - no separate
    search API key needed, just the Gemini key already configured for the agent."""
    if not query.strip():
        return "No query provided."
    body = {
        "contents": [{"role": "user", "parts": [{"text": query}]}],
        "tools": [{"google_search": {}}],
    }
    result, error = _gemini_request(body, timeout=30)
    if error:
        return f"Web search failed: {error}"
    text = _gemini_extract_text(result)
    sources = []
    try:
        chunks = result["candidates"][0]["groundingMetadata"]["groundingChunks"]
        for chunk in chunks:
            web = chunk.get("web", {})
            if web.get("uri"):
                sources.append(f"{web.get('title', web['uri'])} - {web['uri']}")
    except (KeyError, IndexError, TypeError):
        pass
    if not text:
        return "Web search returned no results."
    if sources:
        text += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources[:5])
    return text


def find_any_entity(store: GraphStore, query: str) -> Optional[Dict]:
    """Like store.find_company, but matches nodes of any label - a Supplier, RiskEvent,
    Product, etc, not just Company. Needed so expand_neighbors/find_entity can start a
    traversal from any entity, not only the companies find_company was written for."""
    query_l = query.lower().strip()
    for node in store.nodes():
        node_d = dict(node) if not isinstance(node, dict) else node
        aliases = json.loads(node_d.get("aliases") or "[]")
        names = [node_d["name"].lower()] + [a.lower() for a in aliases]
        if any(alias and alias in query_l for alias in names):
            return node_d
    return None


def dispatch_tool(store: GraphStore, name: str, args: Dict) -> str:
    try:
        if name == "search_chunks":
            chunks = store.search_chunks(args.get("query", ""), limit=5)
            if not chunks:
                return "No matching chunks found in the local graph."
            return "\n".join(
                f"- [{c['title']}] ({c['source_type']}, {c['published_at'] or 'undated'}): {c['text'][:400]}"
                for c in chunks
            )

        if name == "find_entity":
            row = find_any_entity(store, args.get("name", ""))
            if not row:
                return f"No entity found matching '{args.get('name', '')}'."
            return f"Found: {row['name']} (id={row['id']}, label={row['label']})"

        if name == "expand_neighbors":
            entity_name = args.get("entity_name", "")
            depth = int(args.get("depth") or 2)
            row = find_any_entity(store, entity_name)
            if not row:
                return f"No entity found matching '{entity_name}'."
            paths = store.traverse(row["id"], depth=depth, limit=25)
            lines = []
            for path in paths:
                labels = [f"{edge['source_name']} -[{edge['relation']}]-> {edge['target_name']}" for edge in path]
                if labels:
                    lines.append(" | ".join(labels))
            return "\n".join(lines) if lines else f"No relationships found within {depth} hops of {row['name']}."

        if name == "list_entities":
            rows = store.nodes(args.get("label") or None)
            if not rows:
                return f"No entities found for label '{args.get('label', '')}'."
            return ", ".join(row["name"] for row in rows[:30])

        if name == "web_search":
            return web_search_live(args.get("query", ""))

        return f"Unknown tool: {name}"
    except Exception as exc:
        return f"Tool '{name}' failed: {exc}"


AGENT_INSTRUCTIONS = """
You are an agentic supply-chain risk analyst. You have tools to search ingested filings/news,
resolve and expand entities in a knowledge graph, and search the live web for recent news.

Use tools to gather evidence before answering. Prefer search_chunks and expand_neighbors for
questions about the ingested companies (Apple, NVIDIA, Tesla, TSMC, Foxconn, Samsung, Panasonic,
CATL). Only use web_search when the question explicitly needs current/recent information the
local graph won't have.

If the question names multiple entities (e.g. "Apple, NVIDIA, Tesla"), you must gather evidence
for EACH one separately - call expand_neighbors and/or search_chunks once per named entity, even
if an early result happens to mention another named entity in passing. Do not treat a shared
supplier or a related edge as sufficient coverage for an entity you have not directly queried.
Only stop calling tools once every named entity has been directly covered by at least one tool
call. Do not call the same tool with the same arguments twice.

When you have enough evidence, answer in plain text (no further tool calls):
- 2-5 concise bullets, citing which tool result each claim came from.
- Do not invent facts not returned by a tool call.
- If tools returned nothing relevant, say so plainly instead of guessing.
""".strip()


def run_agent(store: GraphStore, question: str, max_steps: int = 5) -> Dict:
    if not gemini_api_key():
        return {
            "question": question,
            "answer": None,
            "error": "GEMINI_API_KEY is not set - the agent requires Gemini for tool calling.",
            "trace": [],
        }

    contents = [{"role": "user", "parts": [{"text": question}]}]
    trace = []

    for _ in range(max_steps):
        body = {
            "system_instruction": {"parts": [{"text": AGENT_INSTRUCTIONS}]},
            "contents": contents,
            "tools": [{"function_declarations": AGENT_TOOLS}],
        }
        result, error = _gemini_request(body, timeout=45)
        if error:
            return {"question": question, "answer": None, "error": error, "trace": trace}

        try:
            model_parts = result["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            return {"question": question, "answer": None, "error": "Gemini returned no candidates.", "trace": trace}

        function_calls = [p["functionCall"] for p in model_parts if "functionCall" in p]
        if not function_calls:
            answer = _gemini_extract_text(result)
            return {"question": question, "answer": answer or None, "error": None if answer else "Empty response.", "trace": trace}

        contents.append({"role": "model", "parts": model_parts})

        response_parts = []
        for call in function_calls:
            tool_name = call.get("name", "")
            tool_args = call.get("args", {}) or {}
            observation = dispatch_tool(store, tool_name, tool_args)
            trace.append({"tool": tool_name, "args": tool_args, "result": observation[:1000]})
            response_parts.append(
                {"functionResponse": {"name": tool_name, "response": {"result": observation}}}
            )
        contents.append({"role": "user", "parts": response_parts})

    return {
        "question": question,
        "answer": None,
        "error": f"Agent did not finish within {max_steps} tool-call steps.",
        "trace": trace,
    }


def ensure_sample_data() -> None:
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    NEWS_DIR.mkdir(parents=True, exist_ok=True)

    samples = {
        "apple_10k_sample.txt": """
Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories.
Apple reports revenue, gross margin, operating income, net income, and diluted earnings per share as key measures of profitability.
In this sample filing context, Apple's profitability is driven by product sales, services revenue, pricing, component costs, and foreign exchange movements.
The Company relies on outsourcing partners and third-party manufacturers for many components and finished products.
Some components are obtained from single or limited sources and could be affected by supplier concentration.
Apple depends on semiconductors, displays, memory, and other components used in consumer electronics.
Manufacturing and assembly operations are concentrated in China and other parts of Asia, including Taiwan and South Korea.
Supply shortages, logistics delays, tariffs, export controls, or geopolitical risk could materially affect production and sales.
Apple works with suppliers including Foxconn, Taiwan Semiconductor Manufacturing Company, and Samsung Electronics.
""",
        "nvidia_10k_sample.txt": """
NVIDIA Corporation designs graphics processors, systems-on-chip, and artificial intelligence accelerators.
NVIDIA reports revenue, gross profit, operating income, net income, and diluted earnings per share to describe financial performance.
In this sample filing context, NVIDIA's profitability is influenced by AI accelerator demand, data center sales, manufacturing costs, and export control restrictions.
NVIDIA depends on a limited number of third-party suppliers and contract manufacturers for semiconductors and advanced packaging.
The company relies on Taiwan Semiconductor Manufacturing Company and other foundries for wafer fabrication.
Export controls and sanctions may restrict sales of advanced AI accelerators to certain countries.
Shortages of semiconductors, substrate materials, or manufacturing capacity could delay product shipments.
Geopolitical risk in Taiwan and trade restrictions involving China could affect NVIDIA's supply chain and demand.
""",
        "tesla_10k_sample.txt": """
Tesla, Inc. designs, develops, manufactures, and sells electric vehicles and energy storage products.
Tesla reports automotive revenue, gross profit, operating income, net income, and operating margin to describe profitability.
In this sample filing context, Tesla's profitability is affected by vehicle pricing, battery costs, manufacturing scale, lithium prices, and logistics costs.
Tesla depends on suppliers for battery cells, lithium, rare earth materials, semiconductors, and vehicle components.
The company sources battery cells from suppliers including Panasonic and CATL.
Raw material price volatility, lithium shortages, supplier concentration, and logistics delays could affect vehicle production.
Manufacturing operations and supplier networks span the United States, China, Germany, Japan, and Mexico.
Tariffs, port congestion, labor disruption, or natural disasters may increase costs or interrupt supply.
""",
    }

    for filename, content in samples.items():
        path = FILINGS_DIR / filename
        if not path.exists():
            path.write_text(content.strip() + "\n", encoding="utf-8")

    news_path = NEWS_DIR / "sample_news.json"
    if not news_path.exists():
        news_path.write_text(
            json.dumps(
                [
                    {
                        "title": "Chip supply concerns rise after earthquake disrupts Taiwan production",
                        "description": "A natural disaster in Taiwan has raised semiconductor shortage concerns for companies that rely on TSMC and advanced chip manufacturing.",
                        "content": "The disruption could affect companies including Apple and NVIDIA through chip supply and advanced packaging delays.",
                        "url": "https://example.com/taiwan-chip-earthquake",
                        "published_at": "2026-09-04T09:00:00Z",
                    },
                    {
                        "title": "New export controls target advanced AI accelerators",
                        "description": "Regulators announced export restrictions covering advanced artificial intelligence accelerators and semiconductor equipment.",
                        "content": "NVIDIA may face demand and shipment risk if export controls limit sales to China.",
                        "url": "https://example.com/ai-export-controls",
                        "published_at": "2026-09-04T10:30:00Z",
                    },
                    {
                        "title": "Lithium price volatility pressures battery supply chains",
                        "description": "Battery makers and electric vehicle companies face commodity price volatility as lithium markets tighten.",
                        "content": "Tesla, Panasonic, and CATL are exposed to battery material costs and potential lithium shortages.",
                        "url": "https://example.com/lithium-battery-risk",
                        "published_at": "2026-09-04T11:15:00Z",
                    },
                ],
                indent=2,
            ),
            encoding="utf-8",
        )


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GraphRAG Supply Chain Analyzer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f5ef;
      --panel: #ffffff;
      --ink: #18211f;
      --muted: #64706b;
      --line: #d9ddd6;
      --accent: #0e766e;
      --accent-2: #bf5b21;
      --soft: #eef5f3;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      padding: 28px clamp(18px, 4vw, 56px) 18px;
      border-bottom: 1px solid var(--line);
      background: #fffdf8;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 46px);
      letter-spacing: 0;
    }
    p { line-height: 1.55; }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 420px) 1fr;
      gap: 18px;
      padding: 18px clamp(18px, 4vw, 56px) 36px;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    label {
      display: block;
      margin-bottom: 8px;
      font-weight: 700;
    }
    textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      font: inherit;
      resize: vertical;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 11px 14px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary { background: var(--accent-2); }
    button:disabled { opacity: .55; cursor: wait; }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
    .status {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 10px;
      margin-top: 14px;
    }
    .stat {
      background: var(--soft);
      border-radius: 6px;
      padding: 10px;
    }
    .stat strong { display: block; font-size: 22px; }
    .answer {
      font-size: 18px;
      background: #fffaf0;
      border-left: 4px solid var(--accent-2);
      padding: 14px;
      border-radius: 6px;
      white-space: pre-wrap;
    }
    .path, .evidence, .edge {
      border-top: 1px solid var(--line);
      padding: 12px 0;
    }
    .muted { color: var(--muted); }
    .pill {
      display: inline-block;
      padding: 3px 8px;
      margin: 3px 4px 3px 0;
      border-radius: 999px;
      background: var(--soft);
      color: #24524e;
      font-size: 13px;
      font-weight: 700;
    }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
      .status { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>GraphRAG-Based Supply Chain Analyzer</h1>
    <p class="muted">Ask supply-chain risk questions backed by a local graph built from filings and real-time-style news.</p>
  </header>
  <main>
    <aside>
      <label for="question">Question</label>
      <textarea id="question" rows="5">How does a semiconductor shortage affect Apple?</textarea>
      <div class="actions">
        <button id="askBtn" onclick="ask()">Ask</button>
        <button class="secondary" id="ingestBtn" onclick="ingest()">Rebuild Graph</button>
        <button class="secondary" id="rssBtn" onclick="refreshNews()">Refresh RSS</button>
      </div>
      <div class="status">
        <div class="stat"><span class="muted">Nodes</span><strong id="nodes">0</strong></div>
        <div class="stat"><span class="muted">Edges</span><strong id="edges">0</strong></div>
        <div class="stat"><span class="muted">Chunks</span><strong id="chunks">0</strong></div>
        <div class="stat"><span class="muted">LLM</span><strong id="llm">Off</strong></div>
        <div class="stat"><span class="muted">Graph DB</span><strong id="graphBackend">SQLite</strong></div>
        <div class="stat"><span class="muted">Vector DB</span><strong id="vectorBackend">In-row</strong></div>
      </div>
      <p class="muted">Sample companies: Apple, NVIDIA, Tesla.</p>
    </aside>
    <section>
      <div id="result">
        <p class="muted">Run ingestion or ask a question to inspect the graph.</p>
      </div>
    </section>
  </main>
  <script>
    async function api(path, options) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }
    async function status() {
      const data = await api("/api/status");
      document.getElementById("nodes").textContent = data.nodes;
      document.getElementById("edges").textContent = data.edges;
      document.getElementById("chunks").textContent = data.chunks;
      document.getElementById("llm").textContent = data.llm_enabled ? "On" : "Off";
      document.getElementById("graphBackend").textContent = data.graph_backend === "neo4j" ? "Neo4j" : "SQLite";
      document.getElementById("vectorBackend").textContent = data.vector_backend === "chroma" ? "Chroma" : "In-row";
    }
    async function ingest() {
      const btn = document.getElementById("ingestBtn");
      btn.disabled = true;
      btn.textContent = "Building...";
      try {
        const data = await api("/api/ingest", { method: "POST", body: JSON.stringify({ reset: true }) });
        await status();
        const llmChunks = data.filings.llm_chunks + data.news.llm_chunks;
        const ruleChunks = data.filings.rule_chunks + data.news.rule_chunks;
        const embeddedChunks = data.filings.embedded_chunks + data.news.embedded_chunks;
        document.getElementById("result").innerHTML = `<div class="answer">Graph rebuilt: ${data.filings.files} filings and ${data.news.articles} news articles processed.<br>Extraction: ${llmChunks} chunk(s) via LLM, ${ruleChunks} chunk(s) via rule-based fallback.<br>Embeddings: ${embeddedChunks} chunk(s) vectorized for semantic search.</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Rebuild Graph";
      }
    }
    async function refreshNews() {
      const btn = document.getElementById("rssBtn");
      btn.disabled = true;
      btn.textContent = "Refreshing...";
      try {
        const data = await api("/api/news/refresh", { method: "POST" });
        await status();
        document.getElementById("result").innerHTML = `<div class="answer">${data.message}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Refresh RSS";
      }
    }
    async function ask() {
      const btn = document.getElementById("askBtn");
      btn.disabled = true;
      btn.textContent = "Thinking...";
      try {
        const question = document.getElementById("question").value;
        const data = await api("/api/ask", { method: "POST", body: JSON.stringify({ question }) });
        render(data);
        await status();
      } finally {
        btn.disabled = false;
        btn.textContent = "Ask";
      }
    }
    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    }
    function render(data) {
      const paths = data.graph_paths.map(p => `<div class="path">${escapeHtml(p)}</div>`).join("") || `<p class="muted">No graph paths found.</p>`;
      const evidence = data.evidence.map(e => `
        <div class="evidence">
          <strong>${escapeHtml(e.title)}</strong>
          <span class="pill">${escapeHtml(e.source_type)}</span>
          ${e.matched_by ? `<span class="pill">matched by: ${escapeHtml(e.matched_by)}</span>` : ""}
          <p>${escapeHtml(e.snippet)}</p>
          <p class="muted">Score: ${e.score}${e.url ? ` | ${escapeHtml(e.url)}` : ""}</p>
        </div>
      `).join("") || `<p class="muted">No matching evidence chunks found.</p>`;
      const nodes = data.graph.nodes.slice(0, 24).map(n => `<span class="pill">${escapeHtml(n.label)}: ${escapeHtml(n.name)}</span>`).join("");
      const mode = `<p class="muted">Answer mode: ${escapeHtml(data.answer_mode)}${data.llm_error ? ` | ${escapeHtml(data.llm_error)}` : ""}</p>`;
      document.getElementById("result").innerHTML = `
        <div class="answer">${escapeHtml(data.answer)}</div>
        ${mode}
        <h2>Connected Nodes</h2>
        <div>${nodes || '<p class="muted">No nodes.</p>'}</div>
        <h2>Graph Paths</h2>
        ${paths}
        <h2>Evidence</h2>
        ${evidence}
      `;
    }
    status();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    store: GraphStore = None  # type: ignore

    def _send_json(self, payload: Dict, status: int = 200) -> None:
        raw = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> Dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path.startswith("/api/status"):
            llm = llm_status()
            self._send_json(
                {
                    "app": APP_NAME,
                    "nodes": self.store.count("nodes"),
                    "edges": self.store.count("edges"),
                    "chunks": self.store.count("chunks"),
                    "llm_enabled": llm["enabled"],
                    "llm_provider": llm["provider"],
                    "llm_model": llm["model"],
                    "graph_backend": "neo4j" if isinstance(self.store, Neo4jGraphStore) else "sqlite",
                    "vector_backend": "chroma" if get_vector_index() is not None else "in-row",
                }
            )
            return
        if self.path.startswith("/api/companies"):
            self._send_json({"companies": [dict(row) for row in self.store.nodes("Company")]})
            return
        self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/api/ingest":
            body = self._read_json()
            if body.get("reset", True):
                self.store.reset()
                vector_index = get_vector_index()
                if vector_index is not None:
                    vector_index.reset()
            filings = ingest_filings(self.store)
            news = ingest_news_file(self.store)
            self._send_json({"filings": filings, "news": news})
            return
        if self.path == "/api/ask":
            body = self._read_json()
            question = body.get("question", "").strip()
            if not question:
                self._send_json({"error": "question is required"}, 400)
                return
            if self.store.count("chunks") == 0:
                ingest_filings(self.store)
                ingest_news_file(self.store)
            self._send_json(build_answer(self.store, question))
            return
        if self.path == "/api/agent-ask":
            body = self._read_json()
            question = body.get("question", "").strip()
            if not question:
                self._send_json({"error": "question is required"}, 400)
                return
            if self.store.count("chunks") == 0:
                ingest_filings(self.store)
                ingest_news_file(self.store)
            self._send_json(run_agent(self.store, question))
            return
        if self.path == "/api/news/refresh":
            rss_url = os.getenv("NEWS_RSS_URL", "").strip()
            if not rss_url:
                result = ingest_news_file(self.store)
                self._send_json(
                    {
                        "message": "NEWS_RSS_URL is not set, so sample news was refreshed instead.",
                        "result": result,
                    }
                )
                return
            try:
                result = ingest_rss(self.store, rss_url)
                self._send_json({"message": f"RSS refresh complete from {rss_url}", "result": result})
            except Exception as exc:
                self._send_json({"error": f"RSS refresh failed: {exc}"}, 500)
            return
        self._send_json({"error": "Not found"}, 404)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[server] {self.address_string()} - {fmt % args}")


def background_news_worker(store: GraphStore) -> None:
    rss_url = os.getenv("NEWS_RSS_URL", "").strip()
    if not rss_url:
        return
    interval = int(os.getenv("NEWS_POLL_SECONDS", "900"))
    while True:
        try:
            result = ingest_rss(store, rss_url)
            print(f"[news] refreshed {result['articles']} RSS items")
        except Exception as exc:
            print(f"[news] refresh failed: {exc}")
        time.sleep(interval)


def run_server(port: int) -> None:
    ensure_sample_data()
    store = build_store()
    if store.count("chunks") == 0:
        ingest_filings(store)
        ingest_news_file(store)

    Handler.store = store
    worker = threading.Thread(target=background_news_worker, args=(store,), daemon=True)
    worker.start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"{APP_NAME} running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    finally:
        store.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--serve", action="store_true", help="start the browser app")
    parser.add_argument("--port", type=int, default=8000, help="server port")
    parser.add_argument("--ingest", action="store_true", help="rebuild graph from local filings and news")
    parser.add_argument("--ask", type=str, help="ask a supply-chain risk question")
    parser.add_argument("--rss", type=str, help="ingest one RSS feed URL now")
    args = parser.parse_args()

    ensure_sample_data()
    store = build_store()

    if args.ingest:
        store.reset()
        vector_index = get_vector_index()
        if vector_index is not None:
            vector_index.reset()
        print(json.dumps({"filings": ingest_filings(store), "news": ingest_news_file(store)}, indent=2))
    if args.rss:
        print(json.dumps(ingest_rss(store, args.rss), indent=2))
    if args.ask:
        if store.count("chunks") == 0:
            ingest_filings(store)
            ingest_news_file(store)
        result = build_answer(store, args.ask)
        print(result["answer"])
        if result["graph_paths"]:
            print("\nGraph paths:")
            for path in result["graph_paths"][:8]:
                print(f"- {path}")
        if result["evidence"]:
            print("\nEvidence:")
            for item in result["evidence"][:5]:
                source = item["url"] or item["title"]
                print(f"- [{item['source_type']}] {item['snippet']} ({source})")
    if args.serve:
        run_server(args.port)
    if not any([args.ingest, args.ask, args.serve, args.rss]):
        parser.print_help()
    if not args.serve:
        store.close()


if __name__ == "__main__":
    cli()
