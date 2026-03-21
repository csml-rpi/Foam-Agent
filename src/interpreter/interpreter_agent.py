"""ResultsInterpreterAgent (from cfd-scientist); uses injected LangChain llm and Foam-Agent viz_creator."""

from __future__ import annotations

import base64
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from interpreter.text_utils import strip_json_fences
from interpreter.viz_creator import viz_creator

VIZ_MAX_RETRIES = 10


class ResultsInterpreterAgent:
    MAX_EXP_VIZ = 10

    def __init__(self, llm: Any, prompts: Dict[str, Any]):
        self.llm = llm
        self.prompts = prompts or {}

    @staticmethod
    def _extract_output_dir(experiment_result: Dict[str, Any]) -> Optional[Path]:
        out = experiment_result.get("output_dir")
        if isinstance(out, str) and out.strip():
            p = Path(out)
            return p if p.exists() else None
        cmd = experiment_result.get("cmd")
        if isinstance(cmd, list):
            for i, tok in enumerate(cmd):
                if tok == "--output" and i + 1 < len(cmd):
                    p = Path(str(cmd[i + 1]))
                    return p if p.exists() else None
        return None

    @staticmethod
    def _locate_foam_dataset(foam_output_dir: Path) -> Optional[Path]:
        if not foam_output_dir.exists():
            return None
        foam_files = sorted(
            foam_output_dir.rglob("*.foam"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if foam_files:
            return foam_files[0]
        marker = foam_output_dir / "case.foam"
        try:
            marker.touch(exist_ok=True)
            return marker
        except Exception:
            return None

    @staticmethod
    def _get_case_structure(output_dir: Path, foam_path: Path) -> Dict[str, Any]:
        times: List[str] = []
        variables: List[str] = []
        try:
            for d in output_dir.iterdir():
                if not d.is_dir():
                    continue
                name = d.name
                if name in ("constant", "system"):
                    continue
                try:
                    float(name)
                    times.append(name)
                except ValueError:
                    continue
            times = sorted(times, key=lambda x: float(x))
        except Exception:
            pass
        for tdir in [output_dir / "0", output_dir / times[0] if times else None]:
            if tdir is None or not tdir.exists():
                continue
            for f in tdir.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    variables.append(f.name)
            if variables:
                break
        if not variables:
            try:
                import pyvista as pv
                mesh = pv.read(str(foam_path))
                variables = list(getattr(mesh, "array_names", []))
            except Exception:
                pass
        return {"times": times, "variables": variables}

    @staticmethod
    def _image_path_to_data_url(image_path: Path) -> Optional[str]:
        if not image_path.exists() or not image_path.is_file():
            return None
        try:
            b = image_path.read_bytes()
            b64 = base64.b64encode(b).decode("utf-8")
            ext = image_path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                return f"data:image/jpeg;base64,{b64}"
            if ext == ".png":
                return f"data:image/png;base64,{b64}"
            if ext == ".gif":
                return f"data:image/gif;base64,{b64}"
            return f"data:image/png;base64,{b64}"
        except Exception:
            return None

    @staticmethod
    def _image_paths_to_content(image_paths: List[Path], max_images: int = 20) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for p in image_paths[:max_images]:
            url = ResultsInterpreterAgent._image_path_to_data_url(p)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    def _invoke_vision_llm(
        self,
        user_requirement: str,
        image_paths: List[Path],
        system_prompt: str,
        user_prompt_template: str,
        max_retries: int = 10,
    ) -> str:
        text = user_prompt_template.format(user_requirement=user_requirement)
        image_blocks = self._image_paths_to_content(image_paths)
        if not image_blocks:
            text += "\n(No images provided.)"
        content: List[Any] = [{"type": "text", "text": text}]
        content.extend(image_blocks)

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=content)]
        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                out = self.llm.invoke(messages)
                return getattr(out, "content", str(out)) if out else ""
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_retryable = (
                    "throttl" in err_str
                    or "too many requests" in err_str
                    or "rate" in err_str
                    or "validation error" in err_str
                )
                if attempt >= max_retries or not is_retryable:
                    raise
                delay = min(60.0, 1.0 * (2**attempt)) + random.uniform(0, 0.1)
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        return ""

    @staticmethod
    def _user_requirement_text(idea_json: Dict[str, Any], experiment_spec: Dict[str, Any]) -> str:
        parts: List[str] = []
        if isinstance(idea_json, dict) and idea_json.get("description"):
            parts.append(str(idea_json["description"]).strip())
        if isinstance(experiment_spec, dict):
            if experiment_spec.get("description"):
                parts.append(str(experiment_spec["description"]).strip())
            if experiment_spec.get("case_name") and not parts:
                parts.append(str(experiment_spec["case_name"]).strip())
        return "\n".join(p for p in parts if p) or "No user requirement provided."

    def _plan_what_to_visualize(self, user_req: str, case_structure: Dict[str, Any]) -> str:
        times = [str(t) for t in (case_structure.get("times", []) or [])]
        vars_ = [str(v) for v in (case_structure.get("variables", []) or [])]
        max_viz = getattr(self, "MAX_EXP_VIZ", 10)
        system = (
            "You are a CFD visualization planner. Given the user requirement, solution times, and field variables, "
            "describe in plain language what visualizations to generate for scientific assessment. "
            f"At most {max_viz} distinct visualization types. Do NOT write code. Short paragraph only."
        )
        user_t = (
            "User requirement:\n{user_requirement}\n\nTimes:\n{times}\n\nVariables:\n{variables}\n\n"
            "Describe what to visualize (fields, times, plot types). Plain text only."
        )
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user_t)])
        chain = prompt | self.llm
        content = chain.invoke(
            {
                "user_requirement": user_req,
                "times": ", ".join(times) if times else "(none)",
                "variables": ", ".join(vars_) if vars_ else "(none)",
            }
        ).content
        text = str(content or "").strip()
        if not text:
            raise ValueError("LLM returned empty visualization plan")
        return text

    def _text_only_interpret(
        self,
        user_req: str,
        solver_log_tail: str,
        experiment_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        system = self.prompts.get("system_prompt", "You are a CFD results interpreter.")
        user_t = self.prompts.get(
            "user_prompt",
            "USER REQUIREMENT:\n{user_requirement}\n\nSOLVER LOG:\n{solver_log}\n\nReturn JSON.",
        )
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user_t)])
        chain = prompt | self.llm
        content = chain.invoke({"user_requirement": user_req, "solver_log": solver_log_tail}).content
        try:
            parsed = json.loads(strip_json_fences(content))
        except Exception:
            parsed = {"raw": content, "parse_error": True}
        rc = experiment_results.get("returncode")
        parsed.setdefault("rerun_required", rc != 0)
        parsed.setdefault("simulation_success", rc == 0)
        parsed.setdefault("requirement_met", False)
        parsed.setdefault("viz_ok", False)
        parsed.setdefault("viz_attempts", [])
        return parsed

    def interpret(
        self,
        idea_json: Dict[str, Any],
        experiment_spec: Dict[str, Any],
        experiment_results: Dict[str, Any],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        sim_id = experiment_spec.get("simulation_id", "?") if isinstance(experiment_spec, dict) else "?"
        if verbose:
            print(f"[Interpreter] Interpreting {sim_id}...", flush=True)
        user_req = self._user_requirement_text(idea_json, experiment_spec)
        output_dir = self._extract_output_dir(experiment_results)
        solver_log_payload = self._collect_solver_log_tails(experiment_results, n_lines=20)
        solver_log_tail = solver_log_payload.get("tail_text", "") or "(no solver log found)"

        if output_dir is None:
            return self._text_only_interpret(user_req, solver_log_tail, experiment_results)

        foam_path = self._locate_foam_dataset(output_dir)
        if foam_path is None:
            return self._text_only_interpret(user_req, solver_log_tail, experiment_results)

        case_structure = self._get_case_structure(output_dir, foam_path)
        times = [str(t) for t in (case_structure.get("times", []) or [])]
        nonzero_times: List[str] = []
        for t in times:
            try:
                if float(t) != 0.0:
                    nonzero_times.append(t)
            except Exception:
                continue
        if not nonzero_times:
            times_str = ", ".join(times) if times else "(no time folders discovered)"
            augmented_tail = (
                f"Detected time folders in foam_output_dir: {times_str}. "
                "Only '0' (or none) usually means the solver did not advance.\n\n"
                f"{solver_log_tail}"
            )
            base = self._text_only_interpret(user_req, augmented_tail, experiment_results)
            base["rerun_required"] = True
            base["simulation_success"] = False
            base["requirement_met"] = False
            base["viz_ok"] = False
            base.setdefault("viz_attempts", [])
            base["viz_attempts"].append(
                {
                    "attempt": 0,
                    "viz_ok": False,
                    "reason": "Case did not advance beyond time 0; skipping visualization and requesting rerun.",
                }
            )
            return base

        viz_base = output_dir / "interpreter_viz"
        viz_base.mkdir(parents=True, exist_ok=True)

        if verbose:
            print("[Interpreter] Running viz_creator (PyVista)...", flush=True)
        viz_result = viz_creator(
            self.llm,
            foam_output_dir=output_dir,
            viz_dir=viz_base,
            what_to_visualize=self._plan_what_to_visualize(user_req, case_structure),
            user_requirement=user_req,
            reference_viz_script=None,
            max_retries=VIZ_MAX_RETRIES,
        )

        image_paths: List[Path] = [Path(p) for p in viz_result.get("images", [])]
        viz_attempts: List[Dict[str, Any]] = []

        if not viz_result.get("ok") or not image_paths:
            base = self._text_only_interpret(user_req, solver_log_tail, experiment_results)
            reason = viz_result.get("last_error", "viz_creator failed or produced no images")
            base.setdefault("viz_attempts", [])
            base["viz_attempts"].append(
                {
                    "attempt": viz_result.get("attempts", 0),
                    "viz_ok": False,
                    "reason": reason,
                }
            )
            base["viz_ok"] = False
            return base

        viz_attempts.append(
            {
                "attempt": viz_result.get("attempts", 1),
                "viz_ok": True,
                "reason": "viz_creator accepted visualizations",
            }
        )

        system_interp = self.prompts.get(
            "interpretation_system_prompt",
            "You are a CFD results interpreter. Return JSON with simulation_success, requirement_met, issues, rerun_required, summary, reasons.",
        )
        user_interp = self.prompts.get(
            "interpretation_user_prompt",
            "User requirement:\n{user_requirement}\n\nImages from the run. Return JSON only.",
        )
        if verbose:
            print("[Interpreter] Invoking vision LLM for interpretation...", flush=True)
        content = self._invoke_vision_llm(user_req, image_paths, system_interp, user_interp)
        try:
            parsed = json.loads(strip_json_fences(content))
        except Exception:
            parsed = {"raw": content, "parse_error": True}

        parsed.setdefault("rerun_required", False)
        parsed.setdefault("simulation_success", True)
        parsed.setdefault("requirement_met", False)
        parsed.setdefault("viz_ok", bool(viz_attempts and viz_attempts[-1].get("viz_ok")))
        parsed.setdefault("viz_attempts", viz_attempts)
        parsed.setdefault("case_structure", case_structure)
        if verbose:
            print(
                f"[Interpreter] Done: rerun_required={parsed.get('rerun_required')} requirement_met={parsed.get('requirement_met')}",
                flush=True,
            )
        return parsed

    def _collect_solver_log_tails(
        self, experiment_result: Dict[str, Any], n_lines: int = 20
    ) -> Dict[str, Any]:
        output_dir = self._extract_output_dir(experiment_result)
        if output_dir is None:
            return {"output_dir": None, "files": [], "tail_text": ""}
        preferred = [
            "log.icoFoam",
            "log.pisoFoam",
            "log.pimpleFoam",
            "log.simpleFoam",
            "log.rhoPimpleFoam",
            "log.rhoSimpleFoam",
        ]
        files: List[Path] = []
        for name in preferred:
            p = output_dir / name
            if p.exists() and p.is_file():
                files.append(p)
        if not files:
            files = sorted(
                [p for p in output_dir.glob("log.*") if p.is_file() and "foam" in p.name.lower()]
            )
        chunks: List[str] = []
        for p in files:
            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
                tail = "\n".join(lines[-n_lines:])
                if tail.strip():
                    chunks.append(f"--- {p.name} (last {n_lines} lines) ---\n{tail}")
            except Exception:
                pass
        return {
            "output_dir": str(output_dir),
            "files": [str(p) for p in files],
            "tail_text": "\n\n".join(chunks),
        }
