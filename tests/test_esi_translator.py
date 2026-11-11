"""Legacy ESI conversion preserves physical inputs and native contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from translation.esi_translator import ESITranslator, convert_case_to_esi_if_needed


ROOT = Path(__file__).resolve().parent.parent


def _native_missing_entry(argv):
    import subprocess

    entry = argv[argv.index('-entry') + 1]
    raise subprocess.CalledProcessError(1, argv, stderr=(
        '\n--> FOAM FATAL IO ERROR:\nCannot find entry ' + entry + '\n\n'
        '    From function int main(int, char**)\n    in file foamDictionary.C at line 637.\n'))


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
    @pytest.mark.parametrize('application', ['boundaryFoam'])
    def test_unsupported_actual_application_preserves_every_source(self, case_dir, application):
        _write_icofoam_case(case_dir)
        (case_dir / 'system/controlDict').write_text(f'application {application};\n')
        before = {p.relative_to(case_dir): p.read_bytes() for p in case_dir.rglob('*') if p.is_file()}
        with pytest.raises(ValueError, match='not available|Cannot establish solver-dependent|adjointShapeOptimisationFoam'):
            ESITranslator(case_dir).run_translation_pipeline()
        assert {p.relative_to(case_dir): p.read_bytes() for p in case_dir.rglob('*') if p.is_file()} == before

    @pytest.mark.parametrize('source,compressed', [
        ('physicalProperties', False),
        ('momentumTransport', True),
    ])
    def test_required_regional_mapping_stops_before_any_input_changes(
            self, case_dir, source, compressed, monkeypatch):
        import gzip

        _write_icofoam_case(case_dir)
        region = case_dir / 'constant/fluid'
        region.mkdir()
        data = ('FoamFile {version 2.0; class dictionary; object ' + source + ';}\n'
                + ('simulationType laminar;' if source == 'momentumTransport' else 'nu 0.000123;'))
        regional = region / (source + ('.gz' if compressed else ''))
        regional.write_bytes(gzip.compress(data.encode()) if compressed else data.encode())
        declaration = case_dir / ('constant/regionProperties' + ('.gz' if compressed else ''))
        data = 'regions (fluid (fluid));'
        declaration.write_bytes(gzip.compress(data.encode()) if compressed else data.encode())
        (region / 'unknownPrivateProperties').write_bytes(b'Supplied private input\x00\xff')
        (case_dir / 'constant/unrelated').write_text('Extra supplied data\n')
        calls = []

        def active_regions(case):
            calls.append(case)
            return {'fluid': region / 'polyMesh'}

        monkeypatch.setattr('services.meshing._active_mesh_directories', active_regions)
        before = {p.relative_to(case_dir): p.read_bytes() for p in case_dir.rglob('*') if p.is_file()}

        with pytest.raises(ValueError, match=f'Unsupported regional.*constant/fluid/{source}.*inputs preserved'):
            ESITranslator(case_dir).run_translation_pipeline()

        assert calls == [case_dir]
        assert {p.relative_to(case_dir): p.read_bytes() for p in case_dir.rglob('*') if p.is_file()} == before

    @pytest.mark.parametrize(('family', 'model'), [('RAS', 'SpalartAllmaras'), ('LES', 'WALE')])
    def test_turbulence_family_keyword_preserves_model_and_coefficients(self, case_dir, family, model):
        source = case_dir / 'constant/momentumTransport'
        source.write_text(
            f'simulationType {family}; {family} {{\n'
            '// model is a keyword only in the active dictionary\n'
            f' model {model};\n turbulence on;\n'
            ' extra model value;\n'
            ' coefficients { model auxiliary; label "} model ignored;"; }\n}\n')
        ESITranslator(case_dir).run_translation_pipeline()
        result = (case_dir / 'constant/turbulenceProperties').read_text()
        assert f'{family}Model {model};' in result
        assert '// model is a keyword only in the active dictionary' in result
        assert 'extra model value;' in result
        assert 'coefficients { model auxiliary; label "} model ignored;"; }' in result

    @pytest.mark.parametrize('thermo_type', ['hePsiThermo', 'heRhoThermo'])
    def test_supplied_thermodynamics_type_and_properties_are_preserved(self, case_dir, thermo_type):
        properties = case_dir / 'constant/thermophysicalProperties'
        content = (
            'FoamFile { version 2.0; class dictionary; object thermophysicalProperties; }\n'
            f'thermoType {{ type {thermo_type}; mixture pureMixture; transport const; '
            'thermo hConst; equationOfState perfectGas; specie specie; energy sensibleInternalEnergy; }\n'
            'mixture { specie { molWeight 28.9; } thermodynamics { Cp 1005; Hf 0; } '
            'transport { mu 0.00002; Pr 0.7; } }')
        properties.write_text(content)
        ESITranslator(case_dir).run_translation_pipeline()
        assert properties.read_text() == content

    def test_structural_renames_remove_gzip_shadows_and_can_repeat(self, case_dir):
        import gzip
        _write_icofoam_case(case_dir)
        _write_ras_case(case_dir)
        for name in ('physicalProperties', 'momentumTransport'):
            (case_dir / f'constant/{name}.gz').write_bytes(gzip.compress(b'ignored shadow contents\n'))
        ESITranslator(case_dir).run_translation_pipeline()
        for name in ('physicalProperties', 'momentumTransport'):
            assert not (case_dir / f'constant/{name}').exists()
            assert not (case_dir / f'constant/{name}.gz').exists()
        assert 'nu              0.01;' in (case_dir / 'constant/transportProperties').read_text()
        before = {p.relative_to(case_dir): p.read_bytes() for p in case_dir.rglob('*') if p.is_file()}
        ESITranslator(case_dir).run_translation_pipeline()
        assert {p.relative_to(case_dir): p.read_bytes() for p in case_dir.rglob('*') if p.is_file()} == before

    @pytest.mark.parametrize('compressed', [True])
    def test_chemkin_comment_and_extra_fields_do_not_change_physics(self, case_dir, rules_path, compressed):
        import gzip
        _write_icofoam_case(case_dir)
        (case_dir / 'constant/physicalProperties').write_text(
            'FoamFile {class dictionary;}\n// Chemkin comparison only\nnu 0.000123;\nextraField retained;\n')
        helper = case_dir / 'constant/helper.py'
        helper.write_text('print("Chemkin viscosityModel")\n')
        original = helper.read_bytes()
        assets = {'Allrun': '#!/bin/sh\necho test\n', 'log.icoFoam': 'Execution log\n'}
        for name, supplied in assets.items():
            (case_dir / name).write_text(supplied)
        if compressed:
            path = case_dir / 'constant/physicalProperties'
            path.with_suffix('.gz').write_bytes(gzip.compress(path.read_bytes()))
            path.unlink()
        ESITranslator(case_dir, rules_path=rules_path).run_translation_pipeline()
        path = case_dir / ('constant/transportProperties' + ('.gz' if compressed else ''))
        content = gzip.decompress(path.read_bytes()).decode() if compressed else path.read_text()
        assert 'nu 0.000123;' in content
        assert 'extraField retained;' in content
        assert helper.read_bytes() == original
        assert all((case_dir / name).read_text() == supplied for name, supplied in assets.items())
        compile(helper.read_text(), str(helper), 'exec')

    @pytest.mark.parametrize('target', ['foundation-v10', 'esi-v2006'])
    def test_explicit_target_bypasses_legacy_converter(self, case_dir, monkeypatch, target):
        from types import SimpleNamespace
        monkeypatch.setattr(ESITranslator, 'run_translation_pipeline', lambda _: pytest.fail('legacy called'))
        convert_case_to_esi_if_needed(case_dir, SimpleNamespace(openfoam_target=target, openfoam_fork='esi'))


@pytest.mark.integration
def test_extract_and_translate_cavity_from_database(tmp_path: Path, rules_path: Path) -> None:
    """Requires database/foundation-v10/raw/openfoam_tutorials_details.txt."""
    db = ROOT / "database" / "foundation-v10" / "raw" / "openfoam_tutorials_details.txt"
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


@pytest.mark.parametrize('properties,selection,effective,compressed', [
    ('physicalProperties', 'macro', 'constant', False),
    ('transportProperties', 'include', 'CrossPowerLaw', True),
    ('physicalProperties', 'pattern', 'constant', False),
])
def test_indirect_transport_alias_preserves_unsupported_conversion_and_allows_custom_values(
        case_dir, monkeypatch, properties, compressed, selection, effective):
    import gzip
    import subprocess

    (case_dir / 'system/controlDict').write_text('application simpleFoam;\n')
    (case_dir / 'constant/momentumTransport').write_text('simulationType laminar;\n')
    selector = {'macro': 'chosen ' + effective + ';\ntransportModel $chosen;',
                'include': '#include "transportSelection"',
                'pattern': '"transport.*" ' + effective + ';'}[selection]
    content = (f'FoamFile {{version 2.0; class dictionary; object {properties};}}\n'
               + selector + '\nnu 0.000123;\nunknownExtra retained;')
    path = case_dir / ('constant/' + properties + ('.gz' if compressed else ''))
    path.write_bytes(gzip.compress(content.encode()) if compressed else content.encode())
    fragment = case_dir / 'constant/transportSelection'
    fragment.write_text('transportModel ' + effective + ';\n')
    (case_dir / 'private.bin').write_bytes(b'\x00\xff supplied private input')
    before = {p.relative_to(case_dir): p.read_bytes() for p in case_dir.rglob('*') if p.is_file()}
    calls = []

    def native(argv, **kwargs):
        assert argv[argv.index('-entry') - 1] == 'constant/' + properties
        entry = argv[argv.index('-entry') + 1]
        calls.append(entry)
        if entry == 'viscosityModel':
            return _native_missing_entry(argv)
        assert entry == 'transportModel'
        key = '"transport.*"' if selection == 'pattern' else 'transportModel'
        return subprocess.CompletedProcess(argv, 0, f'{key} {effective};\n', '')

    monkeypatch.setattr('translation.esi_translator.run_openfoam_utility', native)
    if effective == 'constant':
        with pytest.raises(ValueError, match='transportModel conversion.*effective constant.*inputs preserved'):
            ESITranslator(case_dir).run_translation_pipeline()
        assert {p.relative_to(case_dir): p.read_bytes() for p in case_dir.rglob('*') if p.is_file()} == before
    else:
        ESITranslator(case_dir).run_translation_pipeline()
        target = case_dir / ('constant/transportProperties' + ('.gz' if compressed else ''))
        actual = gzip.decompress(target.read_bytes()).decode() if compressed else target.read_text()
        assert selector in actual and 'nu 0.000123;' in actual and 'unknownExtra retained;' in actual
        assert fragment.read_bytes() == before[fragment.relative_to(case_dir)]
        assert (case_dir / 'private.bin').read_bytes() == before[Path('private.bin')]
    assert calls == (['transportModel'] if selection == 'macro' else ['viscosityModel', 'transportModel'])


@pytest.mark.parametrize('version,supplied_final', [
    ('10', True),
    ('2006', False),
])
def test_native_included_pressure_final_controls_are_preserved(case_dir, monkeypatch, version, supplied_final):
    import subprocess

    _write_icofoam_case(case_dir)
    fragment = case_dir / 'system/pressureSettings'
    fragment.write_text('pFinal {solver PCG; tolerance 9e-9; relTol 0.8;}\n' if supplied_final
                        else 'suppliedMetadata retained;\n')
    content = ('FoamFile {version 2.0; class dictionary; object fvSolution;}\n'
               'solvers {\n#include "pressureSettings"\n'
               'p {solver PCG; tolerance 1e-6; relTol 0.1;}\n}\nPISO {nCorrectors 2;}\n')
    path = case_dir / 'system/fvSolution'
    path.write_text(content)
    before_fragment = fragment.read_bytes()
    calls = []
    monkeypatch.setenv('WM_PROJECT_VERSION', version)
    def native(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, 'p\npFinal\n' if supplied_final else 'suppliedMetadata\np\n', '')
    monkeypatch.setattr('translation.esi_translator.run_openfoam_utility', native)

    ESITranslator(case_dir).run_translation_pipeline()

    assert calls == [['foamDictionary', *(['-expand'] if version == '10' else []),
                      'system/fvSolution', '-entry', 'solvers', '-keywords']]
    assert fragment.read_bytes() == before_fragment
    if supplied_final:
        assert path.read_text() == content.strip()
    else:
        assert '$p;' in path.read_text() and 'relTol          0;' in path.read_text()


@pytest.mark.parametrize('stage', ['property', 'pressure'])
def test_unresolved_native_conversion_facts_preserve_the_complete_case(case_dir, monkeypatch, stage):
    _write_icofoam_case(case_dir)
    path = case_dir / ('constant/physicalProperties' if stage == 'property' else 'system/fvSolution')
    if stage == 'property':
        path.write_text('nu 0.001;\n#include "unavailableSelection"\n')
    else:
        path.write_text('solvers {\n#include "unavailableSelection"\np {solver PCG;}\n}\n')
    before = {path.relative_to(case_dir): path.read_bytes() for path in case_dir.rglob('*') if path.is_file()}
    def native(*args, **kwargs):
        raise RuntimeError('Selected native dictionary facts are unavailable')
    monkeypatch.setattr('translation.esi_translator.run_openfoam_utility', native)

    with pytest.raises(ValueError, match='Cannot establish native legacy ESI entries.*inputs preserved'):
        ESITranslator(case_dir).run_translation_pipeline()

    assert {path.relative_to(case_dir): path.read_bytes() for path in case_dir.rglob('*') if path.is_file()} == before
