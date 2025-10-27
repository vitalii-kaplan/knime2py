# rag/query_openai_file.py
# Edit a SINGLE file using OpenAI and return the FULL, updated file.
#
# What it does
# ------------
# - Reads the target file from disk (source of truth).
# - Pulls a *small* amount of repository context:
#     * a few STRUCTURE.md chunks (so the model knows the layout),
#     * top-k retrieved chunks relevant to your edit request (excluding the target file).
# - Builds a strict prompt that tells the model to output ONLY the full new file
#   between <<BEGIN_FILE>> and <<END_FILE>> markers (no commentary).
# - Dynamically computes max_output_tokens = min(requested, context_window - input_tokens - safety).
# - Enforces token budget and fails fast if there is no headroom.
# - Prints ONLY the updated file content to stdout (if the markers are present).
# - If --rewrite is provided, also overwrites the target file with the updated content.
#
# Usage
# -----
#   python -m rag.query_openai_file <path/to/file> --edit "your change request"
# Options:
#   --top-k N                     (default: 6) number of non-target context chunks to include
#   --structure-max-chunks N      (default: 1) number of STRUCTURE.md chunks to include
#   --file-hint-max-chunks N      (default: 8) per-file cap when a filename is mentioned in the edit text
#   --model NAME                  (default: gpt-4o-mini)
#   --max-output TOKENS           (default: 10000) upper bound; actual value is computed to fit
#   --safety TOKENS               (default: 1024) safety margin for prompt budgeting
#   --raw                         print model response as-is (don’t parse markers)
#   --rewrite                     overwrite the target file with the updated content
#
# Requirements
# ------------
#   pip install openai chromadb sentence-transformers python-dotenv tiktoken

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

# Load .env early (OPENAI_API_KEY, optional overrides)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from rag.rag_utils import (
    RAGConfig,
    load_config_from_env,
    structure_chunks,
    retrieve_with_structure_and_hints,
    ensure_prompt_fits,
    resolve_context_window,
    print_mode_banner,
    extract_between_markers,
    lang_for,
    count_tokens,
)

# ---------------- Config defaults ----------------
CFG: RAGConfig = load_config_from_env(default_embed_backend=os.getenv("RAG_EMBED_BACKEND", "openai"))

STRUCTURE_MAX_CHUNKS_DEFAULT = int(os.getenv("RAG_STRUCTURE_MAX_CHUNKS", "1"))
FILE_HINT_MAX_CHUNKS_DEFAULT = int(os.getenv("RAG_FILE_HINT_MAX_CHUNKS", "8"))  # per hinted file

# Generation (OpenAI)
OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_CTX_MAP = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "o4-mini": 200_000,
    "o3": 200_000,
}
OPENAI_CONTEXT_WINDOW = resolve_context_window(OPENAI_MODEL, _OPENAI_CTX_MAP, default=128_000)
OPENAI_MAX_OUTPUT_DEFAULT = int(os.getenv("OPENAI_MAX_OUTPUT", "10000"))
RAG_SAFETY_MARGIN_TOKENS_DEFAULT = int(os.getenv("RAG_SAFETY_MARGIN_TOKENS", "1024"))

# ---------------- Edit system prompt (strict markers, no commentary) ----------------
EDIT_SYS = (
    "You are a senior software engineer.\n"
    "Task: Update the SINGLE target source file exactly as requested. Output ONLY the complete, updated file contents.\n"
    "STRICT OUTPUT FORMAT:\n"
    "<<BEGIN_FILE>>\n"
    "<entire updated file content with no extra prose>\n"
    "<<END_FILE>>\n"
    "No explanations, no headers, no code fences. If unsure, say you cannot complete safely.\n"
)

# ---------------- Prompt construction ----------------
def _build_prompt(
    target_path: Path,
    file_text: str,
    edit_request: str,
    struct_chunks: List[Tuple[str, dict]],
    aux_chunks: List[Tuple[str, dict]],
) -> str:
    blocks: List[str] = []

    if struct_chunks:
        struct_formatted = "\n\n".join(
            f"{m.get('path','?')}#chunk-{m.get('chunk','?')}\n{doc}" for doc, m in struct_chunks
        )
        blocks.append("### Repository structure (excerpt)\n" + struct_formatted)

    if aux_chunks:
        aux_formatted = "\n\n".join(
            f"{m.get('path','?')}#chunk-{m.get('chunk','?')}\n{doc}" for doc, m in aux_chunks
        )
        blocks.append("### Related context\n" + aux_formatted)

    lang = lang_for(target_path)
    blocks.append(
        "### Target file (current contents)\n"
        f"PATH: {target_path}\n"
        f"LANG: {lang or 'text'}\n"
        "BEGIN_ORIGINAL\n"
        f"{file_text}\n"
        "END_ORIGINAL"
    )

    instruction = (
        "### Edit request\n"
        f"{edit_request}\n\n"
        "### Output rules (MANDATORY)\n"
        "- Rewrite the file to satisfy the request, preserving project style/imports/paths.\n"
        "- Output only the new file content wrapped exactly as:\n"
        "<<BEGIN_FILE>>\n"
        "<entire updated file>\n"
        "<<END_FILE>>\n"
        "- No explanations, no extra lines before/after markers, no code fences.\n"
    )

    context = "\n\n".join(blocks)
    return f"{EDIT_SYS}\n\n{context}\n\n{instruction}"

# ---------------- OpenAI call (with dynamic max_output) ----------------
def _llm_openai(prompt: str, model: str, requested_max_output: int, ctx_window: int, safety: int) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ImportError("OpenAI client not installed. `pip install openai`.") from e
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    # Compute input token usage and dynamic headroom
    input_tokens = count_tokens(prompt, tokenizer_hint="cl100k_base")
    available = ctx_window - input_tokens - safety
    if available <= 0:
        raise RuntimeError(
            "Prompt too large for model context window.\n"
            f"- Model: {model}\n"
            f"- Context window: {ctx_window} tokens\n"
            f"- Input tokens (est.): {input_tokens}\n"
            f"- Safety margin: {safety}\n"
            f"=> Available for output: {available} (no headroom)\n\n"
            "Fix: reduce context (lower --top-k or structure/file-hint slots), split your request, "
            "or switch to a larger-context model."
        )
    computed_max_output = max(1, min(requested_max_output, available))

    # Print computed value early (before the request)
    print(
        f"[RAG] tokens: input={input_tokens}, ctx={ctx_window}, safety={safety} "
        f"-> computed_max_output={computed_max_output} (requested={requested_max_output})",
        flush=True,
    )

    # Guard using the computed headroom
    ensure_prompt_fits(
        prompt,
        ctx_limit=ctx_window,
        max_output=computed_max_output,
        safety=safety,
        model_name=model,
        tokenizer_hint="cl100k_base",
    )

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=computed_max_output,
    )
    return resp.choices[0].message.content or ""

# ---------------- Main edit flow ----------------
def _edit_file(
    target_path: Path,
    edit_request: str,
    top_k: int,
    struct_max: int,
    file_hint_max: int,
    model: str,
    requested_max_output: int,
    safety: int,
    raw_out: bool,
) -> str:
    if not target_path.exists():
        raise FileNotFoundError(f"Target file does not exist: {target_path}")
    if not target_path.is_file():
        raise IsADirectoryError(f"Target path is not a file: {target_path}")

    file_text = target_path.read_text(encoding="utf-8", errors="ignore")

    # 1) Minimal structure context
    struct_chunks = structure_chunks(CFG, struct_max) if struct_max > 0 else []

    # 2) Hint + vector chunks (exclude the target file so model doesn't parrot it)
    aux_chunks = retrieve_with_structure_and_hints(
        CFG,
        query=edit_request,
        top_k=top_k,
        struct_max=0,  # we already pulled structure separately
        file_hint_max=file_hint_max,
        rerank=False,
        exclude_paths=[str(target_path)],
    )

    # 3) Build strict prompt with markers
    prompt = _build_prompt(
        target_path=target_path,
        file_text=file_text,
        edit_request=edit_request,
        struct_chunks=struct_chunks,
        aux_chunks=aux_chunks,
    )

    # 4) LLM call with dynamic max_output
    ctx_window = resolve_context_window(model, _OPENAI_CTX_MAP, default=128_000)
    resp = _llm_openai(
        prompt,
        model=model,
        requested_max_output=requested_max_output,
        ctx_window=ctx_window,
        safety=safety,
    )

    if raw_out:
        return resp

    body = extract_between_markers(resp)
    if body is None:
        raise RuntimeError(
            "Model response did not contain required markers.\n"
            "Expected format:\n<<BEGIN_FILE>>\n<entire updated file>\n<<END_FILE>>"
        )
    return body

# ---------------- CLI ----------------
def main(argv: Sequence[str] | None = None) -> int:
    global OPENAI_MODEL, OPENAI_CONTEXT_WINDOW

    p = argparse.ArgumentParser(
        description="Rewrite a SINGLE file with OpenAI and print the FULL updated file to stdout."
    )
    p.add_argument("path", help="Path to the target file to edit (must exist).")
    p.add_argument(
        "--edit",
        required=True,
        help="Plain-English instruction describing the change you want in this file.",
    )
    p.add_argument("--top-k", type=int, default=6, help="Number of non-target context chunks to include.")
    p.add_argument("--structure-max-chunks", type=int, default=STRUCTURE_MAX_CHUNKS_DEFAULT)
    p.add_argument("--file-hint-max-chunks", type=int, default=FILE_HINT_MAX_CHUNKS_DEFAULT)
    p.add_argument("--model", type=str, default=OPENAI_MODEL, help="OpenAI chat model to use.")
    p.add_argument("--max-output", type=int, default=OPENAI_MAX_OUTPUT_DEFAULT, help="Upper bound for output tokens.")
    p.add_argument("--safety", type=int, default=RAG_SAFETY_MARGIN_TOKENS_DEFAULT, help="Safety margin tokens.")
    p.add_argument("--raw", action="store_true", help="Print raw model response (don’t parse markers).")
    p.add_argument("--rewrite", action="store_true", help="Overwrite the target file with the updated content.")

    args = p.parse_args(argv)

    # Allow override of OPENAI_MODEL at runtime
    OPENAI_MODEL = args.model
    OPENAI_CONTEXT_WINDOW = resolve_context_window(OPENAI_MODEL, _OPENAI_CTX_MAP, default=128_000)

    target_path = (CFG.repo_root / args.path).resolve() if not os.path.isabs(args.path) else Path(args.path).resolve()

    # Banner (prints requested numbers; a second line will print computed max before request)
    try:
        size = target_path.stat().st_size
    except Exception:
        size = "?"
    print_mode_banner(
        CFG,
        gen_label=f"openai({OPENAI_MODEL})",
        model=OPENAI_MODEL,
        top_k=args.top_k,
        rerank=False,
        context_window=OPENAI_CONTEXT_WINDOW,
        max_output=args.max_output,
        safety=args.safety,
        extras=[f"editing={target_path} (bytes={size})",
                f"struct_slots={args.structure_max_chunks}",
                f"file_hint_slots={args.file_hint_max_chunks} per file",
                f"rewrite={'on' if args.rewrite else 'off'}"],
        warn_openai_embed=(CFG.embed_backend == "openai"),
        warn_openai_gen=True,
    )

    try:
        updated = _edit_file(
            target_path=target_path,
            edit_request=args.edit,
            top_k=args.top_k,
            struct_max=args.structure_max_chunks,
            file_hint_max=args.file_hint_max_chunks,
            model=OPENAI_MODEL,
            requested_max_output=args.max_output,
            safety=args.safety,
            raw_out=args.raw,
        )

        # If --rewrite, write to disk. If --raw was used, extract payload first.
        if args.rewrite:
            to_write = updated
            if args.raw:
                payload = extract_between_markers(updated)
                if payload is None:
                    raise RuntimeError(
                        "Cannot --rewrite because --raw was used and the response lacks required markers.\n"
                        "Remove --raw or ensure the model outputs the <<BEGIN_FILE>>/<<END_FILE>> wrapped content."
                    )
                to_write = payload
            target_path.write_text(to_write, encoding="utf-8")
            try:
                nbytes = len(to_write.encode("utf-8"))
            except Exception:
                nbytes = "?"
            print(f"[RAG] wrote {target_path} ({nbytes} bytes)", file=sys.stderr, flush=True)

        # Always print to stdout (keep old behavior; you can redirect if desired)
        print(updated, end="" if updated.endswith("\n") else "\n")
        return 0

    except KeyboardInterrupt:
        print("Aborted.")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
