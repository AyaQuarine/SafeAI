"""Valide un couple de sorties Common V1 sans dépendance ML.

Usage: python scripts/verify_common_pair.py aligned.json diarization.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def verify(aligned: dict, diar: dict) -> dict:
    for key in ('session_id','duration_s'):
        if key not in aligned or key not in diar:
            raise ValueError(f'{key} manquant')
    if aligned['session_id'] != diar['session_id']:
        raise ValueError('session_id différent entre aligned et diarization')
    if abs(float(aligned['duration_s']) - float(diar['duration_s'])) > .02:
        raise ValueError('durée différente entre aligned et diarization')
    if aligned.get('status') not in {'completed','empty','partial','failed'}:
        raise ValueError('statut aligned invalide')
    turns = diar.get('turns')
    utterances = aligned.get('utterances')
    if not isinstance(turns,list) or not isinstance(utterances,list):
        raise ValueError('turns ou utterances invalide')
    for kind, items in (('tour',turns),('énoncé',utterances)):
        for t in items:
            if t.get('session_id') != aligned['session_id'] or not 0 <= float(t['start_s']) < float(t['end_s']) <= float(aligned['duration_s'])+.02:
                raise ValueError(f'{kind} hors session')
    intersections=[]
    for i,a in enumerate(turns):
        for b in turns[i+1:]:
            if not a.get('speaker_id') or not b.get('speaker_id') or a['speaker_id']==b['speaker_id']:
                continue
            start,end=max(a['start_s'],b['start_s']),min(a['end_s'],b['end_s'])
            if end>start: intersections.append((start,end))
    return dict(session_id=aligned['session_id'],status=aligned['status'],duration_s=aligned['duration_s'],utterances=len(utterances),turns=len(turns),overlap_marked_turns=sum(bool(t.get('overlap')) for t in turns),intersections=len(intersections))


def main(argv:list[str]|None=None)->int:
    args=argv or sys.argv[1:]
    if len(args)!=2:
        print('Usage: python scripts/verify_common_pair.py aligned.json diarization.json',file=sys.stderr)
        return 2
    try:
        aligned=json.loads(Path(args[0]).read_text(encoding='utf-8'))
        diar=json.loads(Path(args[1]).read_text(encoding='utf-8'))
        print(json.dumps(verify(aligned,diar),ensure_ascii=False,indent=2))
        return 0
    except (ValueError,KeyError,TypeError,FileNotFoundError,json.JSONDecodeError) as exc:
        print(f'Invalide: {exc}',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
