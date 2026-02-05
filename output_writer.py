"""Output writer: merge analyses, deduplicate, and write structured JSON."""

import json
from collections import defaultdict


OUTPUT_FILE = "analysis_output.json"


def _endpoint_key(x: dict) -> str:
    return f"{x.get('http_method', '')} {x.get('path', '')}"


def _deduplicate_by_key(items: list[dict], key_field: str = "name", key_func=None) -> list[dict]:
    """Remove duplicates based on key field, merging descriptions."""
    seen: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if key_func:
            key = key_func(item)
        else:
            key = item.get(key_field) or item.get("path") or str(item)
        if key not in seen:
            seen[key] = item.copy()
        elif "description" in item and item["description"]:
            desc = seen[key].get("description", "")
            if desc and item["description"] not in desc:
                seen[key]["description"] = f"{desc}; {item['description']}"
            else:
                seen[key]["description"] = item["description"]
    return list(seen.values())


def _merge_lists_unique(items_list: list, key_field: str = "name", key_func=None) -> list:
    """Merge list of items, deduplicating by key."""
    merged = []
    for items in items_list:
        if isinstance(items, list):
            merged.extend(items)
    return _deduplicate_by_key(merged, key_field, key_func)


def _merge_flat_list(items_list: list) -> list:
    """Merge and deduplicate flat lists (strings)."""
    seen = set()
    result = []
    for items in items_list:
        if isinstance(items, list):
            for x in items:
                if isinstance(x, str) and x and x not in seen:
                    seen.add(x)
                    result.append(x)
                elif isinstance(x, dict) and x.get("name") and x["name"] not in seen:
                    seen.add(x["name"])
                    result.append(x)
    return result


def aggregate_results(results: list[dict], repo_name: str | None = None, repo_url: str | None = None) -> dict:
    """
    Aggregate per-chunk analyses into a unified structure with deduplication.
    repo_name: optional display name (e.g. from URL). repo_url: optional source URL.
    """
    project_overview = {
        "name": (repo_name or "").strip() or "codebase",
        "repo_url": (repo_url or "").strip() or None,
        "purpose": None,
        "stack": [],
        "statistics": {},
    }

    # Global collections across all files
    all_rest_endpoints = []
    all_dependencies = []
    all_design_patterns = set()
    all_security_aspects = set()
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_file[r["file"]].append(r)

    # Derive purpose from first analysis
    for r in results:
        purpose = r.get("analysis", {}).get("project_purpose")
        if purpose:
            project_overview["purpose"] = purpose
            break

    files_list = []
    total_methods = 0
    total_classes = 0
    total_endpoints = 0

    for file_path in sorted(by_file.keys()):
        chunks = sorted(by_file[file_path], key=lambda x: x["chunk_index"])

        file_analysis = {
            "file": file_path,
            "classes": [],
            "methods": [],
            "rest_endpoints": [],
            "entity_mappings": [],
            "dependencies": [],
            "config_properties": [],
            "design_patterns": [],
            "security_aspects": [],
            "error_handling": [],
            "key_constants": [],
            "complexity_notes": None,
        }

        for c in chunks:
            a = c.get("analysis", {})
            if not isinstance(a, dict):
                continue

            if a.get("classes"):
                file_analysis["classes"] = _merge_lists_unique(
                    [file_analysis["classes"], a["classes"]], "name"
                )
            if a.get("methods"):
                file_analysis["methods"] = _merge_lists_unique(
                    [file_analysis["methods"], a["methods"]], "name"
                )
            if a.get("rest_endpoints"):
                file_analysis["rest_endpoints"] = _merge_lists_unique(
                    [file_analysis["rest_endpoints"], a["rest_endpoints"]], key_func=_endpoint_key
                )
            if a.get("entity_mappings"):
                file_analysis["entity_mappings"] = _merge_lists_unique(
                    [file_analysis["entity_mappings"], a["entity_mappings"]], "entity"
                )
            if a.get("dependencies"):
                file_analysis["dependencies"] = _merge_lists_unique(
                    [file_analysis["dependencies"], a["dependencies"]], "target"
                )
            if a.get("config_properties"):
                file_analysis["config_properties"] = _merge_lists_unique(
                    [file_analysis["config_properties"], a["config_properties"]], "key"
                )
            if a.get("design_patterns"):
                for p in a["design_patterns"]:
                    if isinstance(p, str):
                        all_design_patterns.add(p)
                    elif isinstance(p, dict) and p.get("name"):
                        all_design_patterns.add(p["name"])
            if a.get("security_aspects"):
                for s in (a["security_aspects"] or []):
                    if isinstance(s, str):
                        all_security_aspects.add(s)
            if a.get("error_handling"):
                file_analysis["error_handling"] = _merge_flat_list(
                    [file_analysis["error_handling"], a["error_handling"]]
                )
            if a.get("key_constants"):
                file_analysis["key_constants"] = _merge_lists_unique(
                    [file_analysis["key_constants"], a["key_constants"]], "name"
                )
            if a.get("complexity_notes"):
                parts = file_analysis["complexity_notes"] or ""
                file_analysis["complexity_notes"] = f"{parts} {a['complexity_notes']}".strip()

            all_rest_endpoints.extend(file_analysis.get("rest_endpoints", []))
            all_dependencies.extend(file_analysis.get("dependencies", []))

        total_methods += len(file_analysis["methods"])
        total_classes += len(file_analysis["classes"])
        total_endpoints += len(file_analysis["rest_endpoints"])

        # Clean empty lists
        file_analysis = {k: v if v else None for k, v in file_analysis.items()}

        files_list.append(file_analysis)

    project_overview["statistics"] = {
        "total_files_analyzed": len(files_list),
        "total_classes": total_classes,
        "total_methods": total_methods,
        "total_rest_endpoints": total_endpoints,
        "design_patterns": sorted(all_design_patterns),
        "security_aspects": sorted(all_security_aspects),
    }
    # Stack derived from design patterns and tech inferred by LLM
    project_overview["stack"] = sorted(all_design_patterns)

    api_summary = _deduplicate_by_key(all_rest_endpoints, key_func=_endpoint_key)
    project_overview["api_summary"] = {
        "total_endpoints": len(api_summary),
        "endpoints": api_summary[:50],  # Top 50 for overview
    }

    return {
        "project_overview": project_overview,
        "files": files_list,
    }


def save_json(data: dict, path: str = OUTPUT_FILE) -> None:
    """Write data to JSON file with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
