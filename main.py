"""Codebase analysis using Hugging Face LLM: main entry point."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from chunker import chunk_code
from llm_engine import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_ID,
    analyze_chunk,
    check_hf_available,
    generate_architecture_summary,
)
from output_writer import aggregate_results, save_json
from repo_loader import clone_repo, load_code_files, _repo_name_from_url

MAX_WORKERS = 2  # HF models are memory-intensive; fewer workers


def _run_analysis(chunks: list[dict], model_id: str | None = None) -> list[dict]:
    """Run analysis on chunks using Hugging Face LLM."""
    results = []
    total = len(chunks)

    def _analyze(c):
        return analyze_chunk(c, model_id=model_id)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(_analyze, c): i for i, c in enumerate(chunks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"  Error on chunk {idx + 1} ({chunks[idx].get('file', '?')}): {e}")
                results.append({
                    "file": chunks[idx]["file"],
                    "chunk_index": chunks[idx].get("chunk_index", 0),
                    "analysis": {"error": str(e)},
                })
            done = len(results)
            if done % 5 == 0 or done == total:
                print(f"  Processed {done}/{total} chunks")

    results.sort(key=lambda r: (r["file"], r["chunk_index"]))
    return results


def main(
    repo_url: str,
    folder: str = ".",
    limit: int | None = None,
    model_id: str | None = None,
) -> None:
    """Run the full analysis pipeline using Hugging Face models."""
    if not (repo_url or "").strip():
        print("Error: repository URL is required. Use --repo <url>")
        return

    ok, msg = check_hf_available()
    if not ok:
        print(f"Error: {msg}")
        print("Install: pip install transformers torch accelerate")
        return

    model_id = model_id or DEFAULT_MODEL_ID
    print(f"Using model: {model_id}")

    print("Cloning repository...")
    repo_path = clone_repo(repo_url.strip())
    if not repo_path:
        print("Error: could not clone repository. Check the URL and network.")
        return

    print(f"Repo at: {repo_path}")
    subfolder = (folder or ".").strip().replace("\\", "/")
    print(f"Loading code files from: {subfolder or '.'}")
    code_files = load_code_files(repo_path, subfolder)
    print(f"Loaded {len(code_files)} files")

    if not code_files:
        print("No supported code files found. Try --folder . or another path (e.g. src/main/java).")
        return

    print("Chunking code...")
    chunks = chunk_code(code_files)
    print(f"Created {len(chunks)} chunks")

    if limit is not None:
        chunks = chunks[:limit]
        print(f"Limiting to first {limit} chunks")

    print(f"Analyzing chunks with Hugging Face LLM (workers: {MAX_WORKERS})...")
    results = _run_analysis(chunks, model_id=model_id)

    repo_name = _repo_name_from_url(repo_url)
    print("Aggregating and deduplicating results...")
    aggregated = aggregate_results(results, repo_name=repo_name, repo_url=repo_url.strip())

    print("Generating architecture summary...")
    summary_text = generate_architecture_summary(aggregated, model_id=model_id)
    if summary_text:
        aggregated["architecture_summary"] = summary_text

    print("Saving to analysis_output.json...")
    save_json(aggregated)
    stats = aggregated.get("project_overview", {}).get("statistics", {})
    print(f"Done. Extracted: {stats.get('total_classes', 0)} classes, {stats.get('total_methods', 0)} methods, {stats.get('total_rest_endpoints', 0)} endpoints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a codebase with Hugging Face models.")
    parser.add_argument("--repo", "-r", type=str, default="", help="Repository URL")
    parser.add_argument("--folder", "-f", type=str, default=".", help="Subfolder to analyze (default: .)")
    parser.add_argument("--limit", type=int, default=None, help="Limit chunks to process")
    parser.add_argument("--model", type=str, default=None, help=f"Hugging Face model ID (default: {AVAILABLE_MODELS[0]})")
    args = parser.parse_args()
    main(
        repo_url=args.repo,
        folder=args.folder,
        limit=args.limit,
        model_id=args.model or None,
    )
