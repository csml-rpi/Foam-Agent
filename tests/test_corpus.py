"""Native corpus construction, extraction, runtime selection, and verification."""

import gzip
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

from database.script.tutorial_parser import find_cases, save_cases_to_file
import pytest
from scripts.extract_v10_case import extract_case


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('failure', ['unterminated_file', 'relative_folder'])
def test_failed_extraction_preserves_existing_case_and_outside_files(tmp_path, failure):
    raw = tmp_path / 'raw'
    raw.mkdir()
    output = tmp_path / 'case'
    output.mkdir()
    previous = output / 'supplied-input'
    previous.write_text('previous valid case')
    outside = tmp_path / 'outside'
    outside.mkdir()
    external = outside / 'supplied-input'
    external.write_text('external input')
    folder, filename = ('../outside', 'supplied-input') if failure == 'relative_folder' else ('system', 'controlDict')
    save_cases_to_file([dict(case_name='sample', domain='fluid', category='tutorial', solver='icoFoam',
        allrun='None', entries=[dict(folder_name=folder, file_name=filename,
                                    content='replacement input')])], raw)
    database = raw / 'openfoam_tutorials_details.txt'
    if failure == 'unterminated_file':
        database.write_text(database.read_text().replace('</file_content>', ''))

    with pytest.raises((LookupError, ValueError)):
        extract_case(output, 'sample', db_path=database, overwrite=True)

    assert previous.read_text() == 'previous valid case'
    assert external.read_text() == 'external input'
    assert {path.name for path in output.iterdir()} == {'supplied-input'}
    assert not list(tmp_path.glob('.foamagent-extract-*'))


def test_initializer_sources_selected_runtime_and_preserves_external_tutorials(tmp_path):
    runtime = tmp_path / 'selected OpenFOAM'
    (runtime / 'etc').mkdir(parents=True)
    commands = runtime / 'platforms/linux64GccDPInt32Opt/bin'
    commands.mkdir(parents=True)
    tutorials = tmp_path / 'external tutorials'
    system = tutorials / 'incompressible/nativeFoam/sample/system'
    system.mkdir(parents=True)
    (system / 'controlDict').write_text('application nativeFoam;\n')
    (system.parent / 'Allrun').write_text('#!/bin/sh\nnativeFoam\n')
    library = runtime / 'selected library'
    (runtime / 'etc/bashrc').write_text(
        f'export WM_PROJECT_DIR={shlex.quote(str(runtime))}\n'
        'export WM_PROJECT_VERSION=10\n'
        f'export FOAM_APPBIN={shlex.quote(str(commands))}\n'
        f'export LD_LIBRARY_PATH={shlex.quote(str(library))}\n'
        f'export FOAM_TUTORIALS={shlex.quote(str(runtime / "tutorials"))}\n')
    utility = commands / 'nativeUtility'
    utility.write_text(
        '#!/bin/sh\n'
        f'[ "$LD_LIBRARY_PATH" = {shlex.quote(str(library))} ] || {{ echo foreign-library >&2; exit 127; }}\n'
        f'[ "$FOAM_TUTORIALS" = {shlex.quote(str(tutorials))} ] || {{ echo wrong-tutorials >&2; exit 1; }}\n'
        '[ "$1" = -help ] || exit 2\nprintf "NATIVE_HELP\\n"\n')
    utility.chmod(0o755)
    output = tmp_path / 'corpus'
    environment = dict(os.environ, LD_LIBRARY_PATH=str(tmp_path / 'foreign library'),
                       WM_PROJECT_DIR=str(tmp_path / 'foreign runtime'), WM_PROJECT_VERSION='2006')
    result = subprocess.run([
        sys.executable, str(ROOT / 'init_database.py'), '--openfoam_path', str(runtime),
        '--openfoam_target', 'foundation-v10', '--tutorials_path', str(tutorials),
        '--database_path', str(output), '--raw_only',
    ], cwd=tmp_path,
        env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    help_text = (output / 'raw/openfoam_command_help.txt').read_text()
    assert '<help_text>NATIVE_HELP\n</help_text>' in help_text
    assert 'foreign-library' not in help_text and 'wrong-tutorials' not in help_text
    manifest = json.loads((output / 'raw/foamagent_target.json').read_text())
    assert manifest['openfoam_target'] == 'foundation-v10'
    assert manifest['wm_project_version'] == '10'


def _initializer(tmp_path, monkeypatch, **overrides):
    import init_database

    runtime = tmp_path / 'runtime'
    (runtime / 'tutorials').mkdir(parents=True)
    args = SimpleNamespace(openfoam_path=str(runtime), openfoam_target='foundation-v10',
        database_path=str(tmp_path / 'corpus'), tutorials_path='', embedding_provider='',
        embedding_model='', force=False, raw_only=False)
    for name, value in overrides.items():
        setattr(args, name, value)
    monkeypatch.setattr(init_database, 'parse_args', lambda: args)
    monkeypatch.setattr(init_database, '_read_openfoam_version',
        lambda *a: init_database.TARGET_RUNTIME_VERSIONS[args.openfoam_target])
    monkeypatch.setenv('WM_PROJECT_DIR', str(runtime))
    monkeypatch.setenv('FOAM_TUTORIALS', str(runtime / 'tutorials'))
    return init_database, args, Path(args.database_path) / 'raw'


@pytest.mark.parametrize('model', ['', 'Qwen/Qwen3-Embedding-0.6B'])
def test_initializer_reuses_the_indices_its_builders_actually_generate(tmp_path, monkeypatch, model):
    initializer, args, raw = _initializer(tmp_path, monkeypatch, embedding_model=model,
        embedding_provider='huggingface' if model else '')
    commands = []
    index_names = ('command_help', 'allrun_scripts', 'tutorials_structure', 'tutorials_details')

    def generate(command, **kwargs):
        commands.append(command)
        if 'tutorial_parser.py' in command:
            assert not (raw / 'foamagent_target.json').exists()
            for name in initializer.REQUIRED_RAW_CORPUS_FILES:
                (raw / name).write_text('Native corpus content\n')
        else:
            argv = shlex.split(command)
            selected_model = next((arg.split('=', 1)[1] for arg in argv
                                   if arg.startswith('--embedding_model=')), 'text-embedding-3-small')
            if model:
                assert '--embedding_provider=huggingface' in argv
            index = next(name for name in index_names if f'faiss_{name}.py' in command)
            destination = initializer._embedding_index_dir(Path(args.database_path), selected_model) / f'openfoam_{index}'
            destination.mkdir(parents=True)
            for name in ('index.faiss', 'index.pkl'):
                (destination / name).write_bytes(b'Generated index')

    monkeypatch.setattr(initializer, 'run_command', generate)
    initializer.main()
    assert len(commands) == 5
    manifest = json.loads((raw / 'foamagent_target.json').read_text())
    assert manifest['openfoam_target'] == 'foundation-v10'
    # Reusing complete assets must preserve the evidence of their actual source.
    manifest['source'] = {'archive': 'source actually used for these assets'}
    (raw / 'foamagent_target.json').write_text(json.dumps(manifest))
    before = (raw / 'foamagent_target.json').read_bytes()
    initializer.main()
    assert len(commands) == 5
    assert (raw / 'foamagent_target.json').read_bytes() == before


def test_gzip_text_survives_tutorial_serialization_and_extraction_without_source_changes(tmp_path):
    tutorials = tmp_path / 'tutorials'
    case = tutorials / 'incompressible/icoFoam/sample'
    contents = {
        'system/controlDict.gz': 'FoamFile {class dictionary; object controlDict;}\napplication icoFoam;\n',
        '0/U.gz': ('FoamFile {class volVectorField; object U;}\n'
                   'dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (1 0 0);\nboundaryField {}\n'),
        'system/fvSchemes': 'ddtSchemes {default Euler;}\n',
    }
    for relative, content in contents.items():
        path = case / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(content.encode()) if path.suffix == '.gz' else content.encode())
    for name in ('binary.dat', 'binary.dat.gz'):
        path = case / 'constant' / name
        path.parent.mkdir(exist_ok=True)
        data = b'\x00\xff binary geometry'
        path.write_bytes(gzip.compress(data) if path.suffix == '.gz' else data)
    before = {path.relative_to(case): path.read_bytes() for path in case.rglob('*') if path.is_file()}

    cases, stats = find_cases(str(tutorials))
    entries = {'/'.join((entry['folder_name'], entry['file_name'])): entry['content']
               for entry in cases[0]['entries']}
    assert entries == contents
    assert stats['files_skipped_encoding'] == 2
    raw = tmp_path / 'raw'
    raw.mkdir()
    save_cases_to_file(cases, raw)
    output = tmp_path / 'extracted'
    extract_case(output, 'sample', db_path=raw / 'openfoam_tutorials_details.txt')

    for relative, content in contents.items():
        path = output / relative
        if path.suffix == '.gz':
            assert path.read_bytes().startswith(b'\x1f\x8b')
            assert gzip.decompress(path.read_bytes()).decode() == content
        else:
            assert path.read_text() == content
    assert not (output / 'constant/binary.dat').exists()
    assert not (output / 'constant/binary.dat.gz').exists()
    assert {path.relative_to(case): path.read_bytes() for path in case.rglob('*') if path.is_file()} == before
