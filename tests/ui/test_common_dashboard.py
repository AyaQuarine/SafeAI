"""Tests de la vue de démonstration : aucune inférence réelle ni réseau externe."""
import json
from pathlib import Path
from fastapi.testclient import TestClient
from safeai.ui.app import create_app


def test_common_route_and_static_assets():
    client = TestClient(create_app())
    page = client.get('/common')
    assert page.status_code == 200
    assert 'Common V1' in page.text
    assert 'ne lance ni Whisper' in page.text
    assert client.get('/static/common.js').status_code == 200
    assert client.get('/static/common.css').status_code == 200
    assert client.get('/').status_code == 200
    assert '/common' in client.get('/').text


def test_example_pairs_share_timeline_and_come_from_fixture():
    client = TestClient(create_app())
    for mode in ('completed', 'empty', 'partial'):
        r1 = client.get(f'/static/examples/{mode}_aligned.json')
        r2 = client.get(f'/static/examples/{mode}_diarization.json')
        assert r1.status_code == r2.status_code == 200
        a, d = r1.json(), r2.json()
        assert a['session_id'] == d['session_id']
        assert a['duration_s'] == d['duration_s']
        assert a['provenance'] == 'synthetic_fixture'
        assert a['status'] == mode
        assert all('synthetic' in str(x).lower() or 'EXEMPLE_FICTIF' in str(x) for x in [a['provenance'],a['source']])
        if mode == 'empty':
            assert not a['utterances']
        if mode == 'partial':
            assert sum(t['overlap'] for t in d['turns']) == 2


def test_microphone_can_be_disabled_in_container(monkeypatch):
    monkeypatch.setenv('SAFEAI_DISABLE_MICROPHONE', '1')
    client = TestClient(create_app())
    assert client.get('/api/capabilities').json()['microphone_enabled'] is False
    assert client.get('/api/devices').json()['devices'] == []
    result=client.post('/api/session/start', data={'consent': 'true'})
    assert result.status_code == 503
