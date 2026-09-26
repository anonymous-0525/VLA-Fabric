"""Frozen Mic conditions and evidence-based checkpoint selection, without GPU imports."""
import csv
import json
from pathlib import Path
import shutil

from pi05_fabric.provenance import _sha256 as file_sha256
from pi05_fabric.evaluation.robotwin2_protocol import load_condition_csv, audit_robotwin_rollout_rows

TASK = 'robotwin_handover_mic'
EVALUATION_PROTOCOL = {'prediction_horizon': 50, 'execution_horizon': 25,
                       'flow_steps': 10, 'maximum_steps': 800, 'official_step_limit': 600}


def read(path):
    return json.loads(Path(path).read_text())


def _conditions(root):
    identity, conditions = {}, {}
    for split, first in [('validation', 0), ('fresh', 200)]:
        directory = Path(root) / split
        manifest = read(directory / 'manifest.json')
        digest = file_sha256(directory / 'conditions.csv')
        if (manifest['task'] != TASK or manifest['split'] != split
                or manifest['condition_count'] != 200 or manifest['frozen_sha256'] != digest
                or not manifest['expert_validated']):
            raise ValueError('conditions approval mismatch')
        conditions[split] = load_condition_csv(directory / 'conditions.csv', expected_ids=tuple(range(first, first+200)))
        identity[split] = {'csv_sha256': digest, 'manifest_sha256': file_sha256(directory / 'manifest.json')}
    if {c.env_seed for c in conditions['validation']} & {c.env_seed for c in conditions['fresh']}:
        raise ValueError('condition seed overlap')
    return identity, conditions


def snapshot_conditions(source, target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    identity, _ = _conditions(source)
    target.mkdir(parents=True)
    for split in identity:
        (target / split).mkdir()
        for filename in ['conditions.csv', 'manifest.json']:
            shutil.copy2(Path(source) / split / filename, target / split / filename)
    load_snapshot(target, identity)
    return identity


def load_snapshot(root, identity):
    actual, conditions = _conditions(root)
    if actual != identity:
        raise ValueError('condition snapshot changed')
    return conditions


def checkpoint_identity(checkpoint, runs_root):
    checkpoint, runs_root = Path(checkpoint).resolve(), Path(runs_root).resolve()
    run = next((p for p in checkpoint.parents if p.parent == runs_root), None)
    if run is None or not run.name.startswith(('PI05-ROBOTWIN2-HANDOVER-MIC-V2-FULL-DIRECT-20K-', 'PI05-ROBOTWIN2-HANDOVER-MIC-V2-FULL-DIRECT-30K-')):
        raise ValueError('not a formal Mic run')
    record = read(run / 'training_run.json')
    manifest = read(checkpoint / 'manifest.json')
    if (not record['four_gpu_gate_passed'] or record['global_batch'] != 12
            or record['config']['task'] != TASK or record['config']['steps'] not in (20000, 30000)
            or manifest['stage'] != 'pi_native_v2_full_direct'
            or manifest['step'] not in ((10000,20000,30000) if record['config']['steps']==30000 else (5000,10000,20000))):
        raise ValueError('formal Mic contract mismatch')
    for seed in ['model_seed', 'training_seed']:
        if record['config'][seed] != manifest[seed]:
            raise ValueError('formal seed mismatch')
    if file_sha256(checkpoint / 'state.msgpack') != manifest['state_sha256']:
        raise ValueError('checkpoint state changed')
    return {'checkpoint': str(checkpoint), 'run': str(run), 'step': manifest['step'],
            'model_sha256': manifest['model_sha256'], 'state_sha256': manifest['state_sha256'],
            'manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
            'training_run_sha256': file_sha256(run / 'training_run.json'), 'protocol': manifest['protocol']}


def _selection(evaluations, runs_root):
    candidates, shared = [], None
    for directory in evaluations:
        directory = Path(directory).resolve()
        record = read(directory / 'manifest.json')
        actual = checkpoint_identity(record['checkpoint']['checkpoint'], runs_root)
        if actual != record['checkpoint'] or record['task'] != TASK or record['split'] != 'validation':
            raise ValueError('evaluation checkpoint identity changed')
        conditions = load_snapshot(directory / 'conditions_snapshot', record['conditions'])['validation']
        protocol = record['evaluation_protocol']
        if any(protocol.get(k) != v for k, v in EVALUATION_PROTOCOL.items()):
            raise ValueError('evaluation protocol mismatch')
        comparable = {'run': actual['run'], 'training_run_sha256': actual['training_run_sha256'],
                      'protocol': actual['protocol'], 'conditions': record['conditions'], 'evaluation_protocol': protocol}
        if shared is None:
            shared = comparable
        elif comparable != shared:
            raise ValueError('candidate protocols or conditions differ')
        with (directory / 'merged/rollouts.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        audit_robotwin_rollout_rows(rows, conditions=conditions)
        if any(r['task'] != TASK or r['instruction'] != c.instruction for r,c in zip(rows,conditions)):
            raise ValueError('task or instruction mismatch')
        successes = sum(int(r['success']) for r in rows)
        for path in ['COMPLETE.json', 'merged/summary.json']:
            summary = read(directory / path)
            if (summary['status'] != 'PASS' or summary['trials'] != 200 or summary['successes'] != successes
                    or summary['errors'] or summary['timeouts']):
                raise ValueError('incomplete or inconsistent development evidence')
        if any(r['error'].strip() or int(r['timeout']) for r in rows):
            raise ValueError('development infrastructure failures')
        candidates.append({'checkpoint': actual, 'successes': successes, 'evaluation': str(directory),
                           'rollouts_sha256': file_sha256(directory / 'merged/rollouts.csv')})
    total = read(Path(shared['run']) / 'training_run.json')['config']['steps'] if shared else 0
    expected_steps = [10000,20000,30000] if total == 30000 else [5000,10000,20000]
    if sorted(c['checkpoint']['step'] for c in candidates) != expected_steps:
        raise ValueError('require all three unique candidates')
    candidates.sort(key=lambda c: c['checkpoint']['step'])
    selected = min(candidates, key=lambda c: (-c['successes'], c['checkpoint']['step']))
    return {'selected_checkpoint': selected['checkpoint'], 'candidates': candidates,
            'conditions': shared['conditions'], 'evaluation_protocol': shared['evaluation_protocol']}


def freeze_selection(evaluations, runs_root, output):
    result = _selection(evaluations, runs_root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    return result


def validate_frozen_selection(path, checkpoint, conditions, evaluation_protocol, runs_root):
    frozen = read(path)
    actual = _selection([c['evaluation'] for c in frozen['candidates']], runs_root)
    if (actual != frozen or checkpoint != frozen['selected_checkpoint']
            or conditions != frozen['conditions'] or evaluation_protocol != frozen['evaluation_protocol']):
        raise ValueError('frozen selection identity mismatch')
