"""Codebase Analysis UI - Streamlit web interface."""

import json
import os
import streamlit as st

from chunker import chunk_code
from llm_engine import (
    DEFAULT_MODEL_ID,
    analyze_chunk,
    check_hf_available,
    generate_architecture_summary,
    get_downloaded_models,
)
from output_writer import aggregate_results
from repo_loader import REPOS_DIR, clone_repo, get_folder_structure, get_selectable_folders, load_code_files, _repo_name_from_url

# Page config
st.set_page_config(
    page_title="Codebase Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean, readable UI
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
    h1 { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem; }
    h2, h3, h4 { font-weight: 600; color: #16213e; margin-top: 1.25rem; margin-bottom: 0.5rem; }
    p { line-height: 1.6; color: #334155; }
    .stExpander { border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 0.5rem; }
    .stTextInput input { border-radius: 8px; border: 1px solid #e2e8f0; }
    .stButton > button { border-radius: 8px; font-weight: 500; padding: 0.5rem 1.25rem; }
    [data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
    [data-testid="stMetricValue"] { font-size: 1.25rem; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def run_analysis(
    repo_path: str,
    subfolder: str,
    limit: int | None,
    model_id: str | None = None,
    repo_name: str | None = None,
    repo_url: str | None = None,
    include_summary: bool = True,
) -> dict:
    """Run full analysis pipeline using Hugging Face LLM."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    subfolder = (subfolder or ".").strip().replace("\\", "/")
    code_files = load_code_files(repo_path, subfolder)
    if not code_files:
        return {"error": f"No supported files found in folder: {subfolder or '.'}. Try another folder or '.' for the whole repo."}

    chunks = chunk_code(code_files)
    if limit:
        chunks = chunks[:limit]

    results = []
    total = len(chunks)
    progress = st.progress(0, text="Analyzing with Hugging Face LLM...")

    def _analyze(c):
        return analyze_chunk(c, model_id=model_id)

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(_analyze, c): i for i, c in enumerate(chunks)}
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"file": chunks[idx]["file"], "chunk_index": chunks[idx].get("chunk_index", 0), "analysis": {"error": str(e)}})
            done += 1
            progress.progress(done / total, text=f"Processing {done}/{total} chunks")

    progress.empty()
    results.sort(key=lambda r: (r["file"], r["chunk_index"]))
    files_read = [cf["file"] for cf in code_files]
    aggregated = aggregate_results(results, repo_name=repo_name, repo_url=repo_url, files_read=files_read)

    if "error" not in aggregated and include_summary:
        with st.spinner("Generating architecture summary..."):
            summary_text = generate_architecture_summary(aggregated, model_id=model_id)
            if summary_text:
                aggregated["architecture_summary"] = summary_text

    return aggregated


def _render_tree(nodes: list[dict], depth: int = 0) -> None:
    for n in nodes:
        if n.get("children"):
            with st.expander(f"📁 {n['label']}", expanded=depth < 2):
                _render_tree(n["children"], depth + 1)
        else:
            st.caption(f"  {'  ' * depth}📁 {n['label']}")


def main():
    st.title("🔍 Codebase Analyzer")
    st.caption("Analyze repository structure with Hugging Face models")

    with st.sidebar:
        hf_ok, hf_msg = check_hf_available()
        st.markdown("### 🤖 Model (pre-downloaded, ready for inference)")
        downloaded = get_downloaded_models()
        if not hf_ok:
            st.warning(f"⚠️ {hf_msg}")
            model_id = None
        elif not downloaded:
            st.error("No models in models/hf/. Run:")
            st.code("python main.py --download-all", language="bash")
            model_id = None
        else:
            # Prefer Qwen2-0.5B (faster) when available for ~30s response
            fast_model = "Qwen/Qwen2-0.5B-Instruct"
            default_idx = downloaded.index(fast_model) if fast_model in downloaded else 0
            model_id = st.selectbox(
                "Select model",
                options=downloaded,
                index=default_idx,
                help="All models are pre-downloaded — instant inference.",
            )
        st.markdown("### ⚙️ Settings")
        force_reclone = st.checkbox("Force re-clone (delete existing)", value=False)
        include_summary = st.checkbox("Include architecture summary", value=True, help="Uncheck for faster analysis")
        chunk_limit = st.number_input("Chunk limit (0 = all)", min_value=0, value=8, step=2, help="Lower = faster (~30s). 8 recommended.")
        st.markdown("---")
        st.markdown("### 📖 How it works")
        st.markdown("""
        1. Enter a GitHub repo URL
        2. Click **Load Repo** to clone
        3. Select a folder to analyze
        4. Click **Run Analysis**
        """)

    st.markdown("#### 1. Repository URL")
    col1, col2 = st.columns([3, 1])
    with col1:
        repo_url = st.text_input(
            "Repository URL",
            value=st.session_state.get("repo_url", ""),
            placeholder="https://github.com/owner/repo or GitLab/Bitbucket URL",
            label_visibility="collapsed",
        )
    with col2:
        load_clicked = st.button("📥 Load Repo", use_container_width=True)

    if st.sidebar.button("🔄 Clear & load new repo"):
        st.session_state.pop("repo_path", None)
        st.rerun()

    if "repo_path" not in st.session_state:
        st.session_state.repo_path = None

    if load_clicked and repo_url:
        with st.spinner("Cloning repository..."):
            try:
                repo_path = clone_repo(repo_url.strip(), force=force_reclone)
                if repo_path:
                    st.session_state.repo_path = repo_path
                    st.session_state.repo_url = repo_url.strip()
                    st.success(f"Repository loaded at `{repo_path}`")
                else:
                    st.error("Please enter a valid repository URL.")
            except Exception as e:
                st.error(f"Failed to clone: {e}")

    repo_path = st.session_state.repo_path
    if repo_path is None and repo_url and ("github.com" in repo_url or "gitlab.com" in repo_url or "bitbucket.org" in repo_url):
        folder = _repo_name_from_url(repo_url.strip())
        path = os.path.join(REPOS_DIR, folder)
        if os.path.exists(path):
            repo_path = path
            st.session_state.repo_path = path
            if "repo_url" not in st.session_state:
                st.session_state.repo_url = repo_url.strip()

    if repo_path:
        st.markdown("#### 2. Select folder to analyze")
        st.caption("Pick any folder — analysis includes that folder and all subfolders.")
        folders = get_selectable_folders(repo_path)
        tree_data = get_folder_structure(repo_path)

        col_tree, col_sel = st.columns([2, 1])
        with col_tree:
            st.caption("Folder structure")
            with st.container():
                for node in tree_data:
                    if node.get("children"):
                        with st.expander(f"📁 {node['label']}", expanded=True):
                            _render_tree(node["children"])
                    else:
                        st.caption(f"📁 {node['label']}")
        with col_sel:
            default_idx = folders.index("src/main/java") if "src/main/java" in folders else 0
            selected_folder = st.selectbox(
                "Analyze this folder",
                options=folders,
                index=min(default_idx, len(folders) - 1),
                help="Use '.' for the whole repo.",
            )
            folder_to_analyze = (selected_folder or ".").replace("\\", "/").strip()

        st.markdown("#### 3. Run analysis")
        analyze_clicked = st.button(
            "▶️ Run Analysis", type="primary",
            disabled=not hf_ok or model_id is None,
            help="Select a model in the sidebar" if model_id is None else None,
        )

        if analyze_clicked and hf_ok and model_id:
            limit = int(chunk_limit) if chunk_limit else None
            repo_url_for_analysis = st.session_state.get("repo_url", "") or ""
            repo_name_for_analysis = _repo_name_from_url(repo_url_for_analysis) if repo_url_for_analysis else os.path.basename(repo_path)
            with st.spinner("Running analysis..."):
                result = run_analysis(
                    repo_path, folder_to_analyze, limit,
                    model_id=model_id,
                    repo_name=repo_name_for_analysis,
                    repo_url=repo_url_for_analysis or None,
                    include_summary=include_summary,
                )

            if "error" in result:
                st.error(result["error"])
                if "No supported files" in str(result.get("error", "")):
                    st.info("Try: check **Force re-clone** and Load Repo again, or select a different folder.")
            else:
                st.balloons()
                st.success("Analysis complete!")

                stats = result.get("project_overview", {}).get("statistics", {})
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("📄 Files read", stats.get("total_files_read", stats.get("total_files_analyzed", 0)))
                c2.metric("📊 Analyzed", stats.get("total_files_analyzed", 0))
                c3.metric("🏛️ Classes", stats.get("total_classes", 0))
                c4.metric("⚙️ Methods", stats.get("total_methods", 0))
                c5.metric("🔗 Endpoints", stats.get("total_rest_endpoints", 0))

                col_dl, col_md, _ = st.columns([1, 1, 2])
                with col_dl:
                    st.download_button("📥 Download JSON", data=json.dumps(result, indent=2, ensure_ascii=False), file_name="analysis_output.json", mime="application/json")
                with col_md:
                    po = result.get("project_overview", {})
                    stats = po.get("statistics", {})
                    md_lines = [
                        f"# {po.get('name', 'Codebase')} – Analysis Report\n\n",
                        f"**Repository:** {po.get('repo_url', '—')}\n\n",
                        f"**Purpose:** {po.get('purpose') or '—'}\n\n",
                        f"**Stack:** {', '.join(po.get('stack', [])) or '—'}\n\n",
                        f"**Files read:** {stats.get('total_files_read', stats.get('total_files_analyzed', 0))} | "
                        f"**Analyzed:** {stats.get('total_files_analyzed', 0)} | "
                        f"**Classes:** {stats.get('total_classes', 0)} | "
                        f"**Methods:** {stats.get('total_methods', 0)} | "
                        f"**Endpoints:** {stats.get('total_rest_endpoints', 0)}\n\n",
                    ]
                    if result.get("architecture_summary"):
                        md_lines.append("## Architecture Summary\n\n" + result["architecture_summary"] + "\n\n")
                    md_lines.append("## All Files Read\n\n")
                    for fp in result.get("files_read", [f.get("file") for f in result.get("files", [])]):
                        md_lines.append(f"- `{fp}`\n")
                    md_lines.append("\n## Files & Methods\n\n")
                    for f in result.get("files", []):
                        md_lines.append(f"### {f.get('file', '')}\n\n")
                        for m in (f.get("methods") or [])[:20]:
                            md_lines.append(f"- **{m.get('name', '')}**: {m.get('description', '')}\n")
                        if not (f.get("methods") or f.get("classes")):
                            md_lines.append("*(Read only – not analyzed in detail)*\n")
                        md_lines.append("\n")
                    st.download_button("📄 Export Markdown", data="".join(md_lines), file_name="analysis_report.md", mime="text/markdown")

                tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📁 Files", "📄 All files read", "🔧 Full JSON"])
                with tab1:
                    po = result.get("project_overview", {})
                    st.markdown("#### 📖 Architecture summary")
                    if result.get("architecture_summary"):
                        st.markdown(result["architecture_summary"])
                    else:
                        st.info("Enable **Include architecture summary** in settings for a prose overview.")
                    st.divider()
                    st.markdown("#### 📌 Quick facts")
                    st.markdown(f"**Purpose:** {po.get('purpose') or '—'}")
                    stack = po.get("stack", [])
                    st.markdown(f"**Tech stack & patterns:** {', '.join(stack) if stack else '—'}")
                    if po.get("repo_url"):
                        st.markdown(f"**Repository:** [{po['repo_url']}]({po['repo_url']})")
                    if po.get("api_summary", {}).get("endpoints"):
                        st.markdown("#### 🔗 Sample endpoints")
                        for ep in po["api_summary"]["endpoints"][:10]:
                            st.markdown(f"- `{ep.get('http_method', '')} {ep.get('path', '')}` — {ep.get('description', '')}")
                    st.markdown("#### ⚙️ Sample methods")
                    sample_methods = []
                    for f in result.get("files", [])[:10]:
                        for m in (f.get("methods") or [])[:2]:
                            desc = (m.get("description") or "").strip()
                            if desc:
                                sample_methods.append((f.get("file", "").split("/")[-1].split("\\")[-1], m.get("name", ""), desc))
                    if sample_methods:
                        for file_name, method_name, desc in sample_methods[:12]:
                            st.markdown(f"- **{file_name}** → `{method_name}`: {desc[:120]}{'…' if len(desc) > 120 else ''}")
                    else:
                        st.caption("See the **Files** tab for per-file details.")
                with tab2:
                    files_list = result.get("files", [])
                    search_query = st.text_input("🔍 Filter by file or method name", placeholder="e.g. Controller, getCustomer", key="files_search")
                    if search_query:
                        q = search_query.strip().lower()
                        files_list = [
                            f for f in files_list
                            if q in (f.get("file") or "").lower()
                            or any(q in (m.get("name") or "").lower() for m in (f.get("methods") or []))
                        ]
                    for f in files_list:
                        has_detail = bool(f.get("methods") or f.get("classes") or f.get("rest_endpoints"))
                        label = f"📄 {f.get('file', '')}" + (" ✓" if has_detail else " (read only)")
                        with st.expander(label, expanded=has_detail):
                            if f.get("classes"):
                                st.markdown("**Classes**")
                                for c in f["classes"]:
                                    st.markdown(f"- `{c.get('name', '')}` — {c.get('description', '—')}")
                                st.markdown("")
                            if f.get("methods"):
                                st.markdown("**Methods**")
                                for m in f["methods"]:
                                    st.markdown(f"**`{m.get('name', '')}`**")
                                    if m.get("signature"):
                                        st.caption(f"`{m['signature']}`")
                                    st.markdown(f"{m.get('description', '—')}")
                                    st.markdown("")
                            if f.get("rest_endpoints"):
                                st.markdown("**REST endpoints**")
                                for e in f["rest_endpoints"]:
                                    st.markdown(f"- `{e.get('http_method', '')} {e.get('path', '')}` — {e.get('description', '')}")
                            if not has_detail:
                                st.caption("File was read but not analyzed in detail (chunk limit may apply).")
                with tab3:
                    st.markdown("All files loaded from the selected folder:")
                    files_read = result.get("files_read", [fp.get("file") for fp in result.get("files", [])])
                    for fp in sorted(files_read):
                        st.markdown(f"- `{fp}`")
                    if not files_read:
                        st.caption("No files listed.")
                with tab4:
                    st.json(result)


if __name__ == "__main__":
    main()
