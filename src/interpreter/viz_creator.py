from __future__ import annotations

import base64
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from interpreter.text_utils import strip_json_fences

VIZ_MAX_RETRIES = 10


def _ensure_marker_foam(case_dir: Path) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    marker = case_dir / f"{case_dir.name}.foam"
    if not marker.exists():
        marker.touch()
    return marker


def _run_script(script_path: Path, cwd: Path) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=600,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return -1, "", f"Runner exception: {e}"


def _images_to_blocks(image_paths: List[Path], max_images: int = 16) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for p in image_paths[:max_images]:
        if not p.exists() or not p.is_file():
            continue
        try:
            b = p.read_bytes()
            b64 = base64.b64encode(b).decode("utf-8")
            ext = p.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif ext == ".gif":
                mime = "image/gif"
            else:
                mime = "image/png"
            url = f"data:{mime};base64,{b64}"
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        except Exception:
            continue
    return blocks


def viz_creator(
    llm: Any,
    foam_output_dir: Path,
    viz_dir: Path,
    what_to_visualize: str,
    user_requirement: str,
    reference_viz_script: Optional[str] = None,
    max_retries: int = VIZ_MAX_RETRIES,
) -> Dict[str, Any]:
    foam_output_dir = foam_output_dir.expanduser().resolve()
    viz_dir = viz_dir.expanduser().resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)

    case_label = foam_output_dir.parent.name if foam_output_dir.parent else foam_output_dir.name

    def _log(msg: str) -> None:
        print(f"[viz_creator {case_label}] {msg}", flush=True)

    _log(f"case_dir={foam_output_dir} viz_dir={viz_dir}")

    for old_png in viz_dir.glob("*.png"):
        try:
            old_png.unlink()
        except Exception:
            pass

    marker_foam = _ensure_marker_foam(foam_output_dir)

    script_system = (
        "You write PyVista+matplotlib Python scripts to visualize OpenFOAM cases.\n"
        "Requirements (CFD paper-quality figures only):\n"
        "- Load the case using PyVista from the given foam_output_dir.\n"
        "- The marker .foam file to load is always the given marker_name.\n"
        "- Use off_screen=True plotters only (no interactive windows).\n"
        "- Save all figures as PNG files into viz_dir.\n"
        "- Use PyVista for all field visualizations; do NOT use matplotlib to draw 2D contour/filled-field plots.\n"
        "- Matplotlib may be used only for 1D line plots where data are first sampled from PyVista.\n"
        "- Output ONLY raw Python code. No markdown fences.\n"
        "- First line must be an import statement.\n"
    )

    script_user_tpl = (
        "User requirement:\n{user_requirement}\n\nWhat to visualize:\n{what_to_visualize}\n\n{reference_block}"
        "foam_output_dir:\n{foam_output_dir}\n\nviz_dir:\n{viz_dir}\n\nmarker_name:\n{marker_name}\n\n"
        "Previous error:\n{previous_error}\n\nPrevious script:\n{previous_script}\n\n"
        "Write a complete Python script: pyvista as pv, save PNGs to viz_dir, exit 0 on success.\n"
    )

    viz_check_system = (
        "You check visualization output quality only (not physics). "
        "Return ONLY JSON: {\"viz_acceptable\": bool, \"reason\": \"string\"}."
    )

    viz_check_user_tpl = (
        "User requirement:\n{user_requirement}\n\nRequested:\n{what_to_visualize}\n\n"
        "Images follow. Are they readable and show requested plot types? JSON only."
    )

    last_error = ""
    last_script = ""
    images: List[Path] = []
    attempt = 0

    if not reference_viz_script:
        existing_script = viz_dir / "viz_script.py"
        if existing_script.is_file():
            try:
                reference_viz_script = existing_script.read_text(encoding="utf-8")
            except Exception:
                reference_viz_script = None

    if reference_viz_script:
        reference_block = (
            "Reference script:\n" + reference_viz_script + "\n\n"
        )
    else:
        reference_block = ""

    for attempt in range(1, max_retries + 1):
        user_prompt = script_user_tpl.format(
            user_requirement=user_requirement,
            what_to_visualize=what_to_visualize,
            reference_block=reference_block,
            foam_output_dir=str(foam_output_dir),
            viz_dir=str(viz_dir),
            marker_name=marker_foam.name,
            previous_error=last_error or "(none)",
            previous_script=last_script or "(none - first attempt)",
        )
        script_msgs = [
            SystemMessage(content=script_system),
            HumanMessage(content=user_prompt),
        ]
        try:
            resp = llm.invoke(script_msgs)
            script_text = getattr(resp, "content", str(resp))
        except Exception as e:
            last_error = f"LLM error while generating script: {e}"
            _log(f"LLM error: {e}")
            continue

        script_text = strip_json_fences(script_text)
        lines = script_text.lstrip().splitlines()
        if lines and lines[0].strip().lower() in {"python", "bash", "sh"}:
            script_text = "\n".join(lines[1:])
        script_path = viz_dir / "viz_script.py"
        script_path.write_text(script_text, encoding="utf-8")

        rc, out, err = _run_script(script_path, cwd=foam_output_dir)
        pngs = sorted(p for p in viz_dir.glob("*.png") if p.is_file())

        if rc != 0 or not pngs:
            snippet = (err or out or "Unknown error")[-4000:]
            last_error = f"Return code: {rc}\nSTDOUT:\n{out[-1000:]}\nSTDERR:\n{snippet}\n"
            last_script = script_text
            for p in pngs:
                try:
                    p.unlink()
                except Exception:
                    pass
            continue

        img_blocks = _images_to_blocks(pngs)
        if not img_blocks:
            last_error = "Generated images could not be read/encoded."
            for p in pngs:
                try:
                    p.unlink()
                except Exception:
                    pass
            continue

        viz_user = viz_check_user_tpl.format(
            user_requirement=user_requirement,
            what_to_visualize=what_to_visualize,
        )
        content: List[Any] = [{"type": "text", "text": viz_user}]
        content.extend(img_blocks)
        viz_msgs = [
            SystemMessage(content=viz_check_system),
            HumanMessage(content=content),
        ]
        try:
            viz_resp = llm.invoke(viz_msgs)
            raw = getattr(viz_resp, "content", str(viz_resp))
            parsed = json.loads(strip_json_fences(raw))
            viz_ok = bool(parsed.get("viz_acceptable", False))
        except Exception as e:
            viz_ok = True
            _log(f"viz check parse fallback: {e}")

        if viz_ok:
            _log(f"attempt {attempt}: viz accepted, {len(pngs)} images")
            images = pngs
            last_error = ""
            break

        last_error = "Viz quality check rejected; retrying."
        last_script = script_text
        for p in pngs:
            try:
                p.unlink()
            except Exception:
                pass

        time.sleep(min(2.0, 0.25 * (1 + random.random())))

    if images:
        _log(f"SUCCESS: {len(images)} images after {attempt} attempt(s)")
    else:
        _log(f"FAILED after {attempt} attempt(s)")

    return {
        "ok": bool(images),
        "images": [str(p) for p in images],
        "attempts": attempt,
        "last_error": last_error,
        "foam_output_dir": str(foam_output_dir),
        "viz_dir": str(viz_dir),
        "marker_foam": str(marker_foam),
    }
