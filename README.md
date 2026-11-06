# Agentic Supply Chain Risk Intelligence System

An autonomous AI-powered platform for supply-chain risk intelligence that connects financial disclosures, live news, and knowledge-graph relationships to detect and explain risk exposure across firms, suppliers, and upstream dependencies.

## Overview

This project was designed to go beyond traditional keyword search by combining:

- knowledge graphs for multi-hop relationship tracing
- financial filing analysis for company and supplier context
- live news ingestion for emerging risk signals
- LLM-based reasoning and tool orchestration for investigative workflows

The system helps answer questions such as:

- Which suppliers or geographies are linked to a company’s operational risk?
- What events are causing a downstream disruption?
- How do supplier concentration, geopolitical exposure, or export controls affect a target organization?

## What I built

- Developed an autonomous AI agent for multi-hop supply-chain risk analysis using knowledge graphs, financial filings, and live news.
- Designed an LLM-driven orchestration layer with function calling to dynamically execute graph traversals, hybrid entity lookups, and semantic web searches for complex investigations.
- Integrated Gemini API for agentic reasoning, Neo4j for graph-based relationship mapping, and ChromaDB with vector embeddings for document retrieval.
- Implemented a retrieval pipeline that combines structured entity relationships with unstructured evidence to support explainable risk tracing.

## Core capabilities

- Multi-hop risk tracing across companies, suppliers, products, countries, and risk events
- Hybrid retrieval using graph + semantic search + source-grounded evidence
- Financial filing ingestion from SEC-style 10-K documents and related disclosures
- News-driven monitoring of evolving supply-chain disruptions
- Agentic reasoning to decompose complex queries and gather evidence from multiple sources
- Explainable outputs with linked relationships, source snippets, and reasoning steps

## Technology stack

- Python
- Gemini API
- Neo4j
- ChromaDB
- SQLite / local graph storage for lightweight runs
- Vector embeddings and semantic retrieval
- LLM tool orchestration and function calling

## Architecture

The system follows a modular agentic design:

1. Ingest financial filings and news sources
2. Extract entities and relationships
3. Store graph structure and retrieval metadata
4. Retrieve relevant evidence using hybrid search
5. Ask an LLM to reason over the graph and supporting documents
6. Return risk findings with traceable source evidence

## Example use cases

- "How does a semiconductor shortage affect Apple?"
- "Which suppliers are connected to Tesla’s EV supply chain risk?"
- "What export-control or geopolitical exposures are linked to NVIDIA?"
- "Trace supplier concentration risk across the electronics value chain."

## Project structure

- `app.py` – main application logic and ingestion/query interface
- `data/` – sample filings and news data used for demonstration
- `storage/` – local database and vector store artifacts
- `docs/` – project documentation and architecture assets

## Run locally

```bash
python app.py --ingest
python app.py --ask "How does a semiconductor shortage affect Apple?"
python app.py --serve
```

Then open:

```text
http://127.0.0.1:8000
```

## Notes

This project demonstrates an agentic supply-chain intelligence workflow that blends structured graph reasoning with semantic retrieval and LLM-based analysis, making it suitable for risk investigation, supplier monitoring, and decision support in complex operational environments.
