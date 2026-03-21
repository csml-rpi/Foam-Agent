"""Flow-field narrative analysis (cfd-scientist AnalysisAgent pattern, single OpenFOAM case)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from interpreter.viz_creator import viz_creator

try:
    from PIL import Image
except ImportError:
    Image = None


def _downscale_image_bytes(b: bytes, max_dimension: int, fmt: str = "PNG") -> bytes:
    if Image is None:
        return b
    img = Image.open(io.BytesIO(b))
    w, h = img.size
    if w <= max_dimension and h <= max_dimension:
        return b
    scale = min(max_dimension / w, max_dimension / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    if fmt.upper() == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _image_paths_to_blocks(
    image_paths: List[Path],
    max_images: int = 12,
    max_dimension: Optional[int] = None,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for p in image_paths[:max_images]:
        if not p.exists() or not p.is_file():
            continue
        try:
            b = p.read_bytes()
            if max_dimension is not None and Image is not None:
                ext = p.suffix.lower()
                fmt = "PNG" if ext == ".png" else "JPEG" if ext in (".jpg", ".jpeg") else "PNG"
                b = _downscale_image_bytes(b, max_dimension, fmt)
            b64 = base64.b64encode(b).decode("utf-8")
            ext = p.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif ext == ".png":
                mime = "image/png"
            else:
                mime = "image/gif"
            url = f"data:{mime};base64,{b64}"
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        except Exception:
            continue
    return blocks


def _is_image_dimension_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "2000" in str(exc) and ("dimension" in msg or "pixels" in msg)


class FlowFieldAnalysisAgent:
    MAX_EXP_VIZ = 10

    def __init__(self, llm: Any):
        self.llm = llm

    def decide_visualizations(self, user_requirement: str, topic: str) -> str:
        max_viz = getattr(self, "MAX_EXP_VIZ", 10)
        system = (
            "You are a CFD expert. Given the user requirement, decide what visualizations best explain "
            "the flow field: contours, streamlines, slices, profiles, mesh views. "
            "For localized features use zoomed views. "
            f"At most {max_viz} distinct visualization types. Output only a short spec, no code."
        )
        user = (
            f"Topic: {topic}\n\nUser requirement:\n{user_requirement[:4000]}\n\n"
            f"What to plot (max {max_viz} types)?"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{u}")])
        out = (prompt | self.llm).invoke({"u": user})
        return getattr(out, "content", str(out)).strip()

    def create_analysis_viz(
        self,
        foam_output_dir: Path,
        viz_dir: Path,
        user_requirement: str,
        viz_spec: str,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        viz_dir.mkdir(parents=True, exist_ok=True)
        ref_script = foam_output_dir / "interpreter_viz" / "viz_script.py"
        reference_code = ""
        if ref_script.is_file():
            try:
                reference_code = ref_script.read_text(encoding="utf-8")
            except Exception:
                pass
        if verbose:
            print(f"[FlowAnalysis] viz -> {viz_dir}", flush=True)
        result = viz_creator(
            self.llm,
            foam_output_dir=foam_output_dir,
            viz_dir=viz_dir,
            what_to_visualize=viz_spec,
            user_requirement=user_requirement,
            reference_viz_script=reference_code or None,
        )
        return {
            "ok": result.get("ok", False),
            "images": result.get("images", []),
            "viz_dir": result.get("viz_dir", str(viz_dir)),
            "attempts": result.get("attempts", 0),
            "error": result.get("last_error", ""),
        }

    def describe_flow_from_images(
        self,
        user_requirement: str,
        case_name: str,
        topic: str,
        image_paths: List[Path],
        max_images: int = 8,
        verbose: bool = False,
    ) -> str:
        system = (
            "You are a CFD expert. The simulation passed an automated check. "
            "Explain in clear prose: (1) what the flow field looks like, "
            "(2) how it relates to the user requirement, "
            "(3) notable physics and caveats. Short bullets allowed."
        )
        max_retries = 4
        dims: List[Optional[int]] = [None, 1999, 1500, 1000]
        for attempt in range(max_retries):
            max_dim = dims[attempt]
            parts: List[Any] = [
                {
                    "type": "text",
                    "text": f"Case: {case_name}\nTopic: {topic}\n\nRequirement:\n{user_requirement[:2000]}\n\n",
                }
            ]
            blocks = _image_paths_to_blocks(image_paths, max_images=max_images, max_dimension=max_dim)
            if not blocks:
                return "No figures to analyze; check analysis_viz folder."
            parts.extend(blocks)
            parts.append({"type": "text", "text": "\nDescribe the flow field as requested."})
            messages = [SystemMessage(content=system), HumanMessage(content=parts)]
            try:
                out = self.llm.invoke(messages)
                return getattr(out, "content", str(out)).strip()
            except Exception as e:
                if _is_image_dimension_error(e) and attempt < max_retries - 1:
                    if verbose:
                        print("[FlowAnalysis] retry smaller images", flush=True)
                    continue
                raise
        return ""

    def run_for_case(
        self,
        foam_output_dir: Path,
        user_requirement: str,
        case_name: str,
        topic: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        foam_output_dir = foam_output_dir.expanduser().resolve()
        t = topic or case_name or "OpenFOAM"
        viz_spec = self.decide_visualizations(user_requirement, t)
        viz_dir = foam_output_dir / "analysis_viz"
        viz_summary = self.create_analysis_viz(
            foam_output_dir, viz_dir, user_requirement, viz_spec, verbose=verbose
        )
        paths = [Path(p) for p in viz_summary.get("images", [])]
        if paths:
            text = self.describe_flow_from_images(
                user_requirement, case_name, t, paths, verbose=verbose
            )
        else:
            text = "No analysis figures produced. " + str(viz_summary.get("error") or "")
        return {
            "analysis_text": text,
            "viz_spec": viz_spec,
            "visualization": viz_summary,
            "image_paths": [str(p) for p in paths],
        }
