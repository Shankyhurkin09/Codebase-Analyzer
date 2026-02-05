"""Codebase analysis using Hugging Face LLM: main entry point."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from chunker import chunk_code
from llm_engine import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_ID,
    analyze_chunk,
    check_hf_available,
    download_all_models,
    download_model,
    generate_architecture_summary,
    get_downloaded_models,
)
from output_writer import aggregate_results, save_json
from repo_loader import clone_repo, load_code_files, _repo_name_from_url

MAX_WORKERS = 2  # 4 can cause "paging file too small" on limited RAM


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

    downloaded = get_downloaded_models()
    if not downloaded:
        print("Error: No models in models/hf/. Run: python main.py --download-all")
        return
    model_id = model_id or (DEFAULT_MODEL_ID if DEFAULT_MODEL_ID in downloaded else downloaded[0])
    if model_id not in downloaded:
        print(f"Error: Model '{model_id}' not downloaded. Available: {', '.join(downloaded)}")
        return
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
    files_read = [cf["file"] for cf in code_files]
    aggregated = aggregate_results(results, repo_name=repo_name, repo_url=repo_url.strip(), files_read=files_read)

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
    parser.add_argument("--download", "-d", action="store_true", help="Download one model to models/hf/ (no analysis)")
    parser.add_argument("--download-all", action="store_true", help="Download ALL models for app (no analysis). Run before hosting.")
    args = parser.parse_args()

    if args.download_all:
        ok, msg = check_hf_available()
        if not ok:
            print(f"Error: {msg}")
            sys.exit(1)
        n = download_all_models()
        print(f"\n{n}/{len(AVAILABLE_MODELS)} models ready. Start app: python run_app.py")
        sys.exit(0 if n > 0 else 1)

    if args.download:
        ok, msg = check_hf_available()
        if not ok:
            print(f"Error: {msg}")
            sys.exit(1)
        model = args.model or DEFAULT_MODEL_ID
        success = download_model(model)
        if success:
            print(f"Ready. Run: python main.py -r <repo_url> --model {model}")
        sys.exit(0 if success else 1)

    main(
        repo_url=args.repo,
        folder=args.folder,
        limit=args.limit,
        model_id=args.model or None,
    )
