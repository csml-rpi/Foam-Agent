"""Lazy model services, provider-specific credentials, and shared retrieval."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_service_import_and_standalone_execution_do_not_construct_models(tmp_path):
    code = '''
import utils
def forbidden(*args, **kwargs):
    raise AssertionError("Model constructed during import or standalone execution")
utils.LLMService = forbidden
from services.case_paths import safe_case_path
from services import plan, input_writer, run_local, run_hpc, static_review, visualization
assert run_local.run_allrun_and_collect_errors("missing-case")[0]["file"] == "Allrun"
assert safe_case_path("case", "0", "U").name == "U"
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def _config(provider):
    return SimpleNamespace(model_provider=provider, model_version='offline-test-model',
                           temperature=1, structured_output_method='auto')


@pytest.mark.parametrize('provider,host,own_key', [
    ('deepseek', 'api.deepseek.com', 'FAKE_DEEPSEEK_KEY'),
    ('openai', 'api.openai.com', 'FAKE_OPENAI_KEY'),
])
def test_actual_sdk_request_uses_only_the_selected_provider_key(monkeypatch, provider, host, own_key):
    import httpx
    import utils

    monkeypatch.setenv('OPENAI_API_KEY', 'FAKE_OPENAI_KEY')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'FAKE_DEEPSEEK_KEY')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=dict(id='offline-response', object='chat.completion',
            created=0, model='offline-test-model', choices=[dict(index=0,
                message=dict(role='assistant', content='Offline response'), finish_reason='stop')],
            usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2)))

    service = utils.LLMService(_config(provider))
    service.llm.root_client._client.close()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        service.llm.root_client._client = client
        response = service.llm.invoke([dict(role='user', content='Offline transport test')])
    assert response.content == 'Offline response' and len(requests) == 1
    assert requests[0].url.host == host
    assert requests[0].headers['Authorization'] == 'Bearer ' + own_key
