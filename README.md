# Codebase Analysis using LLM

A Python program that analyzes a given codebase, extracts structured knowledge using a local Large Language Model (LLM), and generates machine-readable JSON output. Built with LangChain and Ollama for cost-free, privacy-safe code intelligence extraction.

## Overview

This tool clones any GitHub/GitLab/Bitbucket repository, processes its source code in token-safe chunks, and uses **Hugging Face models** (from the Hub) to extract rich, structured insights. Models download on first use and run locally.

**Why Local LLM?**

- Eliminates API cost
- Keeps proprietary code secure
- Enables offline analysis

## Knowledge Extracted

| Category | Description |
|----------|-------------|
| **Project overview** | Purpose, tech stack, statistics |
| **Classes** | Name, type (class/interface/enum), annotations, responsibilities |
| **Methods** | Name, signature, description, annotations, visibility |
| **REST endpoints** | HTTP method, path, handler, description |
| **Entity mappings** | JPA entities, table names, key fields, relationships |
| **Dependencies** | Imports, injections, extends, implements |
| **Config properties** | Keys, values/types, purpose |
| **Design patterns** | Repository, Service Layer, DTO, Mapper, etc. |
| **Security aspects** | JWT, role-based access, CORS |
| **Error handling** | Global handlers, custom exceptions |
| **Complexity notes** | Coupling, nesting, noteworthy implementation details |
| **API summary** | Deduplicated endpoint catalog with statistics |

## Prerequisites

1. **Python 3.10+**
2. **Git** – For cloning the repository
3. **~2–4 GB RAM** for smaller models (Qwen2-0.5B, SmolLM); **~6–8 GB** for larger (Qwen2.5-1.5B, Phi-2)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web UI (recommended)

```bash
streamlit run app_ui.py
```

1. Enter a GitHub repo URL (e.g. `https://github.com/spring-projects/spring-petclinic`)
2. Click **Load Repo** to clone
3. Select a Hugging Face model in the sidebar (default: Qwen2.5-1.5B-Instruct)
4. Select a folder from the dropdown (e.g. `src/main/java`)
5. Click **Run Analysis** (first run downloads the model from Hugging Face Hub)
6. View results in Overview, Files, or Full JSON tabs

### Command line

```bash
# Full analysis (downloads model on first run)
python main.py --repo https://github.com/owner/repo

# With subfolder and limit
python main.py --repo https://github.com/owner/repo --folder src/main/java --limit 20

# Use a different Hugging Face model
python main.py -r https://github.com/owner/repo --model Qwen/Qwen2-0.5B-Instruct
```

## Approach

1. **Clone** – Use GitPython to clone the target repository
2. **Load** – Recursively read supported files (.java, .yaml, .yml, .properties, .kts); exclude test code and large SQL data
3. **Chunk** – Use `RecursiveCharacterTextSplitter` (1500 chars, 200 overlap) to respect token limits
4. **Analyze** – File-type-aware prompts (Controller, Entity, Config, etc.) for targeted extraction; parallel processing (4 workers)
5. **Aggregate** – Deduplicate and merge results; generate architecture summary via LLM
6. **Output** – Write structured JSON with statistics

## Methodology

- **LangChain** – LLM orchestration, prompts, text splitting
- **File-type detection** – Tailored prompts for Controllers, Entities, Repositories, Config
- **RecursiveCharacterTextSplitter** – Preserves method/class boundaries
- **Parallel processing** – ThreadPoolExecutor for concurrent LLM calls
- **Deduplication** – Merge duplicate methods, endpoints, classes across chunks
- **ChatOllama** – Local inference with deepseek-coder

## Best Practices Implemented

| Practice | Implementation |
|----------|----------------|
| Token limits | Chunk size 1500 chars; no request exceeds typical context |
| Structured output | Explicit JSON schema in prompt; parse and validate |
| Extensibility | Modular components; easy to swap LLM |
| Code-specialized | deepseek-coder model via Ollama |
| Error handling | `errors="ignore"` on read; fallback for invalid JSON |
| Parallelism | ThreadPoolExecutor for faster analysis |
| Deduplication | Merge duplicate extractions across chunks |

## Assumptions

- Transformers and PyTorch are installed; models download from Hugging Face Hub on first use
- Git is available for cloning
- Network access for initial repository clone
- Focus on main source code; test code excluded

## Limitations

- No cross-file context – each chunk is analyzed in isolation
- LLM may occasionally hallucinate details
- Chunk boundaries may split methods across chunks
- Analysis quality depends on model capability

## Output Schema

`analysis_output.json` structure:

```json
{
  "project_overview": {
    "name": "spring-rest-sakila",
    "purpose": "REST API for Sakila DVD rental database",
    "stack": ["Spring Boot", "JPA", "HATEOAS", "Querydsl", "MapStruct", "JWT"],
    "statistics": {
      "total_files_analyzed": 195,
      "total_classes": 120,
      "total_methods": 450,
      "total_rest_endpoints": 45,
      "design_patterns": ["DTO", "Repository", "Service Layer"],
      "security_aspects": ["JWT authentication"]
    },
    "api_summary": {
      "total_endpoints": 45,
      "endpoints": [{"http_method": "GET", "path": "/api/v1/actors", "handler_method": "getActors"}]
    }
  },
  "architecture_summary": "Prose summary generated by LLM covering purpose, stack, components, and API surface.",
  "files": [
    {
      "file": "src/main/java/.../ActorController.java",
      "classes": [{"name": "ActorController", "type": "class", "annotations": ["@RestController"]}],
      "methods": [...],
      "rest_endpoints": [{"http_method": "GET", "path": "/api/v1/actors", "handler_method": "getActors"}],
      "entity_mappings": null,
      "dependencies": [...],
      "design_patterns": null,
      "security_aspects": null,
      "complexity_notes": "Uses Spring HATEOAS for hypermedia"
    }
  ]
}
```

## Deployment (Streamlit Community Cloud)

Deploy the web UI for free on [Streamlit Community Cloud](https://share.streamlit.io):

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app** and choose:
   - **Repository:** `Shankyhurkin09/Codebase-Analyzer`
   - **Branch:** `main`
   - **Main file path:** `app_ui.py`
3. Click **Deploy**. The app will build and run (first run may be slower while dependencies install).

**Note:** The app uses local LLM (Hugging Face models). On Streamlit Cloud, model download and inference run in the cloud; for heavy use consider memory limits and model size (smaller models like Qwen2-0.5B are more suitable for free tier).

## Project Structure

```
codebase_experiments/
├── app_ui.py            # Streamlit web UI (repo URL, folder selection, results)
├── main.py              # CLI entry point, parallel orchestration
├── repo_loader.py       # Clone, folder structure, load source files
├── chunker.py           # Token-safe code chunking
├── llm_engine.py        # LangChain + Ollama, file-type-aware prompts
├── output_writer.py     # Aggregation, deduplication, JSON output
├── analysis_output.json # Generated structured output
├── requirements.txt     # Dependencies
└── README.md            # This file
```
