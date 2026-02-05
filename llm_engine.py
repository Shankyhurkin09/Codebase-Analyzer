"""LLM engine: Hugging Face Transformers for advanced code analysis."""

import json
import os
import re

from langchain.prompts import PromptTemplate

# Project-local model storage (downloads here instead of ~/.cache/huggingface)
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hf")
os.makedirs(_MODELS_DIR, exist_ok=True)

# Best code-capable models from Hugging Face (smaller = faster, larger = better quality)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
AVAILABLE_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/phi-2",
    "bigcode/starcoder2-3b",
]

_llm = None
_current_model_id: str | None = None

# Comprehensive extraction schema for maximum information
ADVANCED_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["code", "file_path", "file_type"],
    template="""You are a senior software architect performing deep code analysis.

Analyze this code and return ONLY valid JSON (no markdown, no code blocks, no extra text).
Be concise. For each method, one short "description" sentence.

Expected structure - extract ALL applicable fields:
{{
  "project_purpose": "Brief purpose if inferable, else null",
  "classes": [
    {{"name": "ClassName", "type": "class|interface|enum", "description": "Role/responsibility", "annotations": ["@RestController", etc]}}
  ],
  "methods": [
    {{"name": "methodName", "signature": "returnType methodName(params)", "description": "One clear sentence: what this method is used for, when it is called, and what it returns or does. Be specific.", "annotations": ["@GetMapping"], "visibility": "public|private|protected"}}
  ],
  "rest_endpoints": [
    {{"http_method": "GET|POST|PUT|DELETE|PATCH", "path": "/api/v1/...", "handler_method": "methodName", "description": "Endpoint purpose"}}
  ],
  "entity_mappings": [
    {{"entity": "EntityName", "table": "table_name", "key_fields": ["id"], "relationships": ["OneToMany with X"]}}
  ],
  "dependencies": [
    {{"type": "import|inject|extends|implements", "target": "FullyQualifiedName", "purpose": "Why used"}}
  ],
  "config_properties": [
    {{"key": "property.name", "value_or_type": "string|int|...", "description": "Purpose"}}
  ],
  "design_patterns": ["Repository", "Service Layer", "DTO", "Mapper", etc],
  "security_aspects": ["JWT auth", "role-based", "CORS", etc] or null,
  "error_handling": ["GlobalExceptionHandler", "custom exceptions", etc] or null,
  "complexity_notes": "Cyclomatic complexity, nesting, coupling, noteworthy implementation details",
  "key_constants": [{{"name": "CONSTANT", "value": "...", "description": "Purpose"}}] or null
}}

File: {file_path}
File type hint: {file_type}

Code:
{code}
""",
)

ARCHITECTURE_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary_json"],
    template="""You are a senior software architect. Based on the following structured analysis of a codebase, write a short architecture summary (2-4 paragraphs) that covers:
1. What the project does and its high-level purpose
2. Main technical stack and patterns used
3. Key components (controllers, services, domains) and how they relate
4. API surface and any notable security or design decisions

Write in clear, professional prose. No bullet lists—use paragraphs only.

Analysis summary:
{summary_json}
""",
)


def download_model(model_id: str | None = None) -> bool:
    """Download and store a model to models/hf/. Uses cached copy if already present."""
    model_id = (model_id or DEFAULT_MODEL_ID).strip()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Downloading {model_id} to {_MODELS_DIR}...")
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=_MODELS_DIR)
        AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=_MODELS_DIR,
        )
        print(f"  [OK] {model_id}")
        return True
    except Exception as e:
        print(f"  [FAIL] {model_id}: {e}")
        return False


def download_all_models() -> int:
    """Download all AVAILABLE_MODELS to models/hf/. Returns count of successfully downloaded."""
    ok = 0
    for m in AVAILABLE_MODELS:
        if download_model(m):
            ok += 1
    return ok


def get_downloaded_models() -> list[str]:
    """Return model IDs that exist and are complete in models/hf/ (have model weights)."""
    if not os.path.isdir(_MODELS_DIR):
        return []
    prefix = "models--"
    found = []
    for name in os.listdir(_MODELS_DIR):
        if not name.startswith(prefix):
            continue
        snap_dir = os.path.join(_MODELS_DIR, name, "snapshots")
        if not os.path.isdir(snap_dir):
            continue
        for snapshot in os.listdir(snap_dir):
            snap_path = os.path.join(snap_dir, snapshot)
            if not os.path.isdir(snap_path):
                continue
            files = os.listdir(snap_path)
            has_weights = (
                "model.safetensors" in files
                or "pytorch_model.bin" in files
                or "model.safetensors.index.json" in files
                or any(f.endswith(".safetensors") for f in files)
            )
            if has_weights:
                model_id = name[len(prefix):].replace("--", "/")
                found.append(model_id)
                break
    return sorted(found) if found else []


def check_hf_available() -> tuple[bool, str]:
    """Verify Hugging Face transformers is installed. Returns (success, message)."""
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True, "Hugging Face models available"
    except ImportError as e:
        return False, f"Install transformers and torch: pip install transformers torch accelerate\n{e}"
    except Exception as e:
        return False, str(e)


def _format_prompt_for_instruct(model_id: str, prompt: str) -> str:
    """Wrap prompt in chat format for instruct models."""
    model_lower = model_id.lower()
    if "qwen" in model_lower or "chat" in model_lower:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if "phi" in model_lower:
        return f"Instruct: {prompt}\nOutput:"
    if "smol" in model_lower:
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    return prompt


def get_llm(model_id: str | None = None, local_files_only: bool = True):
    """Lazy-load the Hugging Face LLM from models/hf/. Uses only pre-downloaded models (no network)."""
    global _llm, _current_model_id
    model_id = (model_id or DEFAULT_MODEL_ID).strip()
    if _llm is None or _current_model_id != model_id:
        _current_model_id = model_id
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=_MODELS_DIR,
            local_files_only=local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=_MODELS_DIR,
            local_files_only=local_files_only,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        llm_raw = HuggingFacePipeline(pipeline=pipe)
        # Wrap to add instruct formatting
        _llm = _InstructLLMWrapper(llm_raw, model_id)
    return _llm


class _InstructLLMWrapper:
    """Wraps HuggingFacePipeline to format prompts for instruct models."""

    def __init__(self, llm, model_id: str):
        self._llm = llm
        self._model_id = model_id

    def invoke(self, prompt: str) -> str:
        formatted = _format_prompt_for_instruct(self._model_id, prompt)
        return self._llm.invoke(formatted)


def _detect_file_type(file_path: str) -> str:
    """Infer file type for prompt context."""
    fp = (file_path or "").lower().replace("\\", "/")
    if "controller" in fp:
        return "Controller - focus on REST endpoints"
    if "entity" in fp or "domain" in fp:
        return "Entity/Model - focus on JPA mappings, relationships"
    if "repository" in fp:
        return "Repository - focus on data access, custom queries"
    if "service" in fp:
        return "Service - focus on business logic"
    if "config" in fp or "application" in fp:
        return "Configuration - focus on properties, beans"
    if ".yaml" in fp or ".yml" in fp or ".properties" in fp:
        return "Config file - focus on properties, endpoints"
    if ".kts" in fp:
        return "Build config - focus on dependencies, plugins"
    if "exception" in fp or "filter" in fp or "security" in fp:
        return "Cross-cutting - focus on error handling, security"
    return "General source"


def _extract_json_from_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = content.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("{"):
                text = p
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"raw_output": content}


def analyze_chunk(chunk: dict, model_id: str | None = None) -> dict:
    """
    Analyze a single code chunk with the Hugging Face LLM.
    Returns dict with: file, chunk_index, analysis (parsed JSON).
    """
    llm = get_llm(model_id)
    file_type = _detect_file_type(chunk.get("file", ""))

    prompt = ADVANCED_ANALYSIS_PROMPT.format(
        code=chunk["chunk"],
        file_path=chunk["file"],
        file_type=file_type,
    )

    response = llm.invoke(prompt)
    content = response if isinstance(response, str) else str(response)
    parsed = _extract_json_from_response(content)

    return {
        "file": chunk["file"],
        "chunk_index": chunk.get("chunk_index", 0),
        "analysis": parsed,
    }


def generate_architecture_summary(aggregated: dict, model_id: str | None = None) -> str:
    """Generate a short prose architecture summary. Uses one LLM call."""
    try:
        po = aggregated.get("project_overview", {})
        stats = po.get("statistics", {})
        api = po.get("api_summary", {})
        summary = {
            "name": po.get("name"),
            "purpose": po.get("purpose"),
            "stack": po.get("stack"),
            "total_files": stats.get("total_files_analyzed"),
            "total_classes": stats.get("total_classes"),
            "total_methods": stats.get("total_methods"),
            "total_endpoints": stats.get("total_rest_endpoints"),
            "design_patterns": stats.get("design_patterns"),
            "security_aspects": stats.get("security_aspects"),
            "sample_endpoints": (api.get("endpoints") or [])[:15],
            "file_names": [f.get("file") for f in aggregated.get("files", [])[:30]],
        }
        llm = get_llm(model_id)
        prompt = ARCHITECTURE_SUMMARY_PROMPT.format(summary_json=json.dumps(summary, indent=2))
        response = llm.invoke(prompt)
        content = response if isinstance(response, str) else str(response)
        return (content or "").strip()
    except Exception:
        return ""
