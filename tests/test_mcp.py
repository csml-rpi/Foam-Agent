"""MCP tool contracts, scheduling, cancellation, and protocol-safe diagnostics."""

import asyncio
import gzip
import importlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
from threading import Event, Lock, get_ident
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def review_case(tmp_path, monkeypatch):
    server = importlib.import_module('src.mcp.fastmcp_server')
    case = tmp_path / 'transient-case'
    (case / 'system').mkdir(parents=True)
    calls = SimpleNamespace(retrieval=[], corpus=[], review=[], native=[])
    config = SimpleNamespace(openfoam_target='foundation-v10', openfoam_fork='foundation',
                             max_time_limit=17, searchdocs=7)
    monkeypatch.setattr(server, 'global_config', config)
    monkeypatch.setattr(server, 'require_target_corpus', lambda cfg: calls.corpus.append(cfg))

    def retrieve(**kwargs):
        calls.retrieval.append(kwargs)
        return ('Actual solver reference', '', '', '', None)

    def diagnose(user, system):
        calls.review.append((user, system))
        return 'Offline diagnosis'

    monkeypatch.setattr(server, 'retrieve_references', retrieve)
    monkeypatch.setattr(server, 'get_llm_service', lambda **kw: SimpleNamespace(invoke=diagnose))
    from services import solver_execution

    def native(argv, **kwargs):
        calls.native.append((argv, kwargs))
        pytest.fail('An established literal must not require native execution')

    monkeypatch.setattr(solver_execution, 'run_openfoam_utility', native)

    def review():
        request = server.ReviewRequest(case_dir=str(case) + '/', errors=['Supplied runtime error'],
                                       user_requirement='Preserve supplied transient physics')
        result = server._review(request, lambda *args: None)
        assert result.model_dump() == {'analysis': 'Offline diagnosis'}
        assert len(calls.review) == 1
        user, _ = calls.review[0]
        assert 'Supplied runtime error' in user and request.user_requirement in user
        return user

    return case, config, calls, review


@pytest.mark.parametrize('application,compressed', [
    ('"/opt/native/rhoPimpleFoam"', False),
    ('rhoPimpleFoam', True),
])
def test_review_uses_current_literal_application_without_a_runtime(review_case, application, compressed):
    case, config, calls, review = review_case
    content = ('label "application wrongSolver; #include \\"wrong\\"";\n'
               '// application wrongSolver;\n'
               'settings {application wrongSolver; #include "nestedSettings"\n$nested}\n'
               'application ' + application + ';\n')
    path = case / ('system/controlDict.gz' if compressed else 'system/controlDict')
    before = gzip.compress(content.encode()) if compressed else content.encode()
    path.write_bytes(before)

    user = review()

    assert calls.retrieval == [dict(case_name='transient-case', case_solver='rhoPimpleFoam',
        case_domain='', case_category='', searchdocs=7, config=config)]
    assert calls.corpus == [config] and calls.native == []
    assert 'Actual solver reference' in user and 'rhoPimpleFoam' in user
    assert path.read_bytes() == before


class Context:
    def __init__(self):
        self.thread = get_ident()

    async def info(self, *args):
        assert get_ident() == self.thread

    error = report_progress = info


def test_long_runs_leave_protocol_responsive_and_serialize_only_their_owner(tmp_path, monkeypatch):
    from services import run_local, solver_execution
    from services.output_safety import prepare_output_directory
    server = importlib.import_module('src.mcp.fastmcp_server')
    first = prepare_output_directory(tmp_path / 'first', overwrite=False)
    second = prepare_output_directory(tmp_path / 'second', overwrite=False)
    first_work = first / 'work'
    first_work.mkdir()
    for case in (first, first_work, second):
        (case / 'Allrun').write_text('icoFoam\n')
    main_thread = get_ident()
    active, peak = {}, {'total': 0, 'same': 0}
    guard = Lock()
    heartbeat = False

    def execute(script, out, err, case, *args, **kwargs):
        assert get_ident() != main_thread, 'Native execution blocks the protocol loop'
        owner = first if Path(case).is_relative_to(first) else second
        with guard:
            active[owner] = active.get(owner, 0) + 1
            peak['total'] = max(peak['total'], sum(active.values()))
            peak['same'] = max(peak['same'], active[owner])
        time.sleep(0.15)
        assert heartbeat, 'Protocol loop did not respond during execution'
        (Path(case) / '1').mkdir(exist_ok=True)
        (Path(case) / '1/U').write_text('FoamFile {class volVectorField;}\n')
        Path(out).write_text('Exec : icoFoam\nEnd\n')
        Path(err).write_text('')
        with guard:
            active[owner] -= 1
        return {'returncode': 0, 'timed_out': False}

    monkeypatch.setattr(run_local, 'run_command', execute)
    monkeypatch.setattr(solver_execution, 'run_openfoam_utility', lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 0, 'application icoFoam;', ''))

    from services.static_review import StaticReviewResult
    import utils

    def review(*args, **kwargs):
        assert get_ident() != main_thread
        return StaticReviewResult(summary='Inputs inspected', findings=[])

    monkeypatch.setattr(utils, 'LLMService', lambda cfg: SimpleNamespace(invoke=review))

    async def scenario():
        nonlocal heartbeat
        ctx = Context()
        run = getattr(server.run, 'fn', server.run)
        requests = [asyncio.create_task(run(server.RunSimulationRequest(case_dir=str(case)), ctx))
                    for case in (first, first_work, second)]
        await asyncio.sleep(0.02)
        heartbeat = True
        inspect = getattr(server.static_review, 'fn', server.static_review)
        report = await inspect(server.StaticReviewRequest(case_dir=str(second), user_requirement='Run supplied case'), ctx)
        assert report.inspected_files and not any(task.done() for task in requests)
        responses = await asyncio.gather(*requests)
        assert all(response.status == 'success' for response in responses)

    asyncio.run(scenario())
    assert peak['same'] == 1 and peak['total'] == 2


def test_cancelled_waiter_does_not_release_a_running_case(tmp_path):
    from src.mcp.dispatch import run_tool

    entered, release, second_entered = Event(), Event(), Event()

    def first(request, notify):
        entered.set()
        assert release.wait(timeout=3)

    def second(request, notify):
        second_entered.set()

    async def scenario():
        ctx = Context()
        task = asyncio.create_task(run_tool(first, None, ctx, case_dir=tmp_path))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            following = asyncio.create_task(run_tool(second, None, ctx, case_dir=tmp_path))
            await asyncio.sleep(0.03)
            assert not second_entered.is_set()
        finally:
            release.set()
        await following
        assert second_entered.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize('subtasks', [[{'folder': '../outside', 'file': 'U'}]])
def test_invalid_generation_request_preserves_existing_output(tmp_path, monkeypatch, subtasks):
    from config import Config
    from services.output_safety import prepare_output_directory
    server = importlib.import_module('src.mcp.fastmcp_server')
    case = prepare_output_directory(tmp_path / 'case', overwrite=False)
    (case / '0').mkdir()
    original = case / '0/U'
    original.write_text('previous valid field')
    monkeypatch.setattr(server, 'global_config', Config(case_dir=str(case)))
    monkeypatch.setattr(server, 'require_target_corpus', lambda _: None)
    monkeypatch.setattr(server, 'get_llm_service', lambda **kw: pytest.fail('Invalid request reached model setup'))
    request = server.GenerateFilesRequest(case_name='case', subtasks=subtasks, user_requirement='Preserve supplied case',
        case_solver='icoFoam', case_domain='fluid', case_category='tutorial', overwrite_output=True)
    with pytest.raises(ValueError):
        asyncio.run(getattr(server.input_writer, 'fn', server.input_writer)(request, Context()))
    assert original.read_text() == 'previous valid field'


def test_real_stdio_server_outputs_only_protocol_messages(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with (tmp_path / 'server.stderr').open('w+') as stderr:
        process = subprocess.Popen([sys.executable, '-m', 'src.mcp.cli'], cwd=root,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=stderr, text=True,
                                   env=dict(os.environ, PYTHONUNBUFFERED='1'))
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)

        def send(message):
            process.stdin.write(json.dumps(message) + '\n')
            process.stdin.flush()

        def receive(request_id):
            while True:
                assert selector.select(timeout=20), 'MCP response timed out'
                line = process.stdout.readline()
                assert line, 'MCP server exited'
                message = json.loads(line)
                if message.get('id') == request_id:
                    assert 'error' not in message, message
                    return message['result']

        try:
            send(dict(jsonrpc='2.0', id=1, method='initialize', params={
                'protocolVersion': '2024-11-05', 'capabilities': {},
                'clientInfo': {'name': 'offline-regression', 'version': '1'},
            }))
            assert receive(1)['serverInfo']['name'] == 'Foam-Agent'
            send(dict(jsonrpc='2.0', method='notifications/initialized'))
            send(dict(jsonrpc='2.0', id=2, method='tools/list', params={}))
            assert {tool['name'] for tool in receive(2)['tools']} >= {'run', 'run_case', 'plan'}
        finally:
            process.terminate()
            process.communicate(timeout=10)
            selector.close()
