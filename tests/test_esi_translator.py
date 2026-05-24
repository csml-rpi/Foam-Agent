"""Unit tests for Foundation v10 -> ESI translation middleware."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from translation.esi_translator import (  # noqa: E402
    ESITranslator,
    _find_dict_block,
    convert_case_to_esi_if_needed,
)


@pytest.fixture
def rules_path() -> Path:
    return ROOT / "src" / "translation" / "esi_translation_rules.json"


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    (case / "constant").mkdir()
    (case / "0").mkdir()
    return case


def _write_icofoam_case(case: Path) -> None:
    (case / "system" / "controlDict").write_text(
        "FoamFile { version 2.0; }\n\napplication     icoFoam;\n",
        encoding="utf-8",
    )
    (case / "system" / "fvSolution").write_text(
        """FoamFile { version 2.0; }

solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0;
    }
}

PISO
{
    nCorrectors     2;
}
""",
        encoding="utf-8",
    )
    (case / "constant" / "physicalProperties").write_text(
        "FoamFile { version 2.0; }\n\nnu              0.01;\n",
        encoding="utf-8",
    )


def _write_ras_case(case: Path) -> None:
    (case / "system" / "controlDict").write_text(
        "application     simpleFoam;\n",
        encoding="utf-8",
    )
    (case / "constant" / "momentumTransport").write_text(
        """
simulationType  RAS;
RAS
{
    model kEpsilon;
    turbulence      on;
}
""",
        encoding="utf-8",
    )


class TestESITranslator:
    def test_renames_physical_and_momentum_files(self, case_dir: Path, rules_path: Path) -> None:
        _write_icofoam_case(case_dir)
        ESITranslator(case_dir, rules_path=rules_path).run_translation_pipeline()

        assert not (case_dir / "constant" / "physicalProperties").exists()
        assert (case_dir / "constant" / "thermophysicalProperties").is_file()

    def test_injects_pfinal_and_pref_for_icofoam(self, case_dir: Path, rules_path: Path) -> None:
        _write_icofoam_case(case_dir)
        ESITranslator(case_dir, rules_path=rules_path).run_translation_pipeline()

        fv = (case_dir / "system" / "fvSolution").read_text()
        assert "pFinal" in fv
        assert "$p;" in fv
        assert "pRefCell" in fv
        assert "pRefValue" in fv

    def test_provisions_transport_properties_for_icofoam(
        self, case_dir: Path, rules_path: Path
    ) -> None:
        _write_icofoam_case(case_dir)
        ESITranslator(case_dir, rules_path=rules_path).run_translation_pipeline()

        transport = case_dir / "constant" / "transportProperties"
        assert transport.is_file()
        assert "nu" in transport.read_text()

    def test_ras_model_keyword_swap(self, case_dir: Path, rules_path: Path) -> None:
        _write_ras_case(case_dir)
        ESITranslator(case_dir, rules_path=rules_path).run_translation_pipeline()

        turb = (case_dir / "constant" / "turbulenceProperties").read_text()
        assert "RASModel kEpsilon;" in turb
        assert "model kEpsilon;" not in turb

    def test_blacklisted_solver_raises(self, case_dir: Path, rules_path: Path) -> None:
        (case_dir / "system" / "controlDict").write_text(
            "application     adjointShapeOptimisationFoam;\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="adjointShapeOptimisationFoam"):
            ESITranslator(case_dir, rules_path=rules_path).run_translation_pipeline()

    def test_strips_markdown_fences(self, case_dir: Path, rules_path: Path) -> None:
        (case_dir / "system" / "controlDict").write_text(
            "```foam\napplication icoFoam;\n```\n",
            encoding="utf-8",
        )
        (case_dir / "system" / "fvSolution").write_text(
            "solvers { p { solver PCG; relTol 0; } }\nPISO { nCorrectors 2; }\n",
            encoding="utf-8",
        )
        ESITranslator(case_dir, rules_path=rules_path).run_translation_pipeline()
        text = (case_dir / "system" / "controlDict").read_text()
        assert "```" not in text
        assert "FoamFile" in text

    def test_skips_allrun_and_logs(self, case_dir: Path, rules_path: Path) -> None:
        allrun = case_dir / "Allrun"
        allrun.write_text("#!/bin/sh\necho test\n", encoding="utf-8")
        log_file = case_dir / "log.icoFoam"
        log_file.write_text("Execution log\n", encoding="utf-8")
        
        ESITranslator(case_dir, rules_path=rules_path).run_translation_pipeline()
        
        # Verify they don't have the FoamFile header
        assert "FoamFile" not in allrun.read_text(encoding="utf-8")
        assert "FoamFile" not in log_file.read_text(encoding="utf-8")

    def test_chemkin_properties_reset(self, case_dir: Path, rules_path: Path) -> None:
        (case_dir / "system" / "controlDict").write_text(
            "application icoFoam;\n",
            encoding="utf-8",
        )
        (case_dir / "system" / "fvSolution").write_text(
            "solvers { p { solver PCG; relTol 0; } }\nPISO { nCorrectors 2; }\n",
            encoding="utf-8",
        )
        (case_dir / "constant" / "physicalProperties").write_text(
            "Chemkin thermo database leak\n",
            encoding="utf-8",
        )
        ESITranslator(case_dir, rules_path=rules_path).run_translation_pipeline()
        props = (case_dir / "constant" / "thermophysicalProperties").read_text()
        assert "nu" in props
        assert "Chemkin" not in props

    def test_noop_for_foundation_fork(self, case_dir: Path, rules_path: Path) -> None:
        _write_icofoam_case(case_dir)
        original = (case_dir / "constant" / "physicalProperties").read_text()

        class FoundationConfig:
            openfoam_fork = "foundation"

        convert_case_to_esi_if_needed(case_dir, FoundationConfig())
        assert (case_dir / "constant" / "physicalProperties").exists()
        assert (case_dir / "constant" / "physicalProperties").read_text() == original


class TestFindDictBlock:
    def test_finds_nested_p_block(self) -> None:
        content = "solvers\n{\n    p\n    {\n        solver PCG;\n    }\n}\n"
        span = _find_dict_block(content, "p")
        assert span is not None
        start, end = span
        assert content[start:end].strip().startswith("p")


@pytest.mark.integration
def test_extract_and_translate_cavity_from_database(tmp_path: Path, rules_path: Path) -> None:
    """Requires database/raw/openfoam_tutorials_details.txt."""
    db = ROOT / "database" / "raw" / "openfoam_tutorials_details.txt"
    if not db.is_file():
        pytest.skip("RAG database not available")

    sys.path.insert(0, str(ROOT / "scripts"))
    from extract_v10_case import extract_case  # noqa: E402

    out = tmp_path / "cavity"
    extract_case(
        out,
        "cavity",
        db_path=db,
        solver="icoFoam",
        category="cavity",
    )
    ESITranslator(out, rules_path=rules_path).run_translation_pipeline()

    assert (out / "constant" / "thermophysicalProperties").is_file()
    assert "icoFoam" in (out / "system" / "controlDict").read_text()
    fv = (out / "system" / "fvSolution").read_text()
    assert "pFinal" in fv
