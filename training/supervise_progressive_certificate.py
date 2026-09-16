"""Run one frozen certificate through completion and independent audit.

This wrapper leaves the archived evaluation code untouched. Progress comes only
from completed block counts; results are evaluated only after the runner exits.
It never changes a policy, sample budget, alpha allocation, or acceptance rule.
"""
import argparse
from datetime import datetime
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

# Only the already-running shared service owns CUDA.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/progressive-upgrades'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def check_preflight(folder, service_path):
    import progressive_campaign as campaign
    manifest = load(folder / 'manifest.json')
    campaign.verify(manifest, folder)
    assert manifest['purpose'] == 'certificate'
    assert manifest['deals'] == 2048
    service, validation = load(service_path), load(folder / 'validation-result.json')
    digest = sha(folder / 'manifest.json')
    spec = manifest['preflight_protocol']
    assert validation['status'] == 'passed' and validation['manifest_sha256'] == digest
    assert validation['decisions'] == spec['decisions_per_policy'] * len(spec['policies'])
    assert {r['policy'] for r in validation['reports']} == set(spec['policies'])
    assert service['status'] == 'ready' and digest in service['manifests']
    assert service['streams'] == validation['streams'] == spec['streams']
    assert validation['clients'] == spec['clients']
    assert service['source_sha256'] == sha(ROOT / 'training/progressive_inference_service.py')
    assert sha(ROOT / spec['script']) == spec['script_sha256']
    for record in validation['reports']:
        path = ROOT / record['path']
        assert sha(path) == record['sha256']
        report = load(path)
        assert report['status'] == 'passed' and report['policy'] == record['policy']
        assert report['decisions'] == spec['decisions_per_policy']
        assert report['clients'] == spec['clients'] and report['streams'] == spec['streams']
        assert report['verifier_sha256'] == spec['script_sha256']
        assert report['service_source_sha256'] == service['source_sha256']
        assert report['source_trace_sha256'] == load(OUT / spec['archive'] / 'results.json')['trace_sha256']
        assert report['selected_inputs_sha256'] == sha(folder / f'verification-inputs-{record["policy"]}.json')
    audit = validation['selection_audit']
    assert sha(ROOT / audit['path']) == audit['sha256']
    assert load(ROOT / audit['path'])['status'] == 'passed'
    state = load(OUT / 'state.json')
    assert state['pending_certificate']['run'] == manifest['name']
    assert state['pending_certificate']['candidate'] == manifest['versions'][1]
    assert manifest['predecessor'] == state['active_baseline']
    assert manifest['certificate_attempt'] == len(state['certificate_attempts']) + 1
    return manifest, service


def progress(folder, deals):
    # Read only structural identifiers; never calculate interim scores or wins.
    counts = {3: 0, 4: 0}
    trace = folder / 'blocks.jsonl'
    if trace.exists():
        with trace.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.endswith('\n'):
                    break  # An in-progress append is not a completed block.
                block = json.loads(line)
                players = block['players']
                assert players in counts and 0 <= block['block'] < deals
                counts[players] += 1
    assert all(n <= deals for n in counts.values())
    return dict(blocks=sum(counts.values()), total=2 * deals,
                three_player_blocks=counts[3], four_player_blocks=counts[4],
                games=counts[3] * 6 + counts[4] * 8)


def publish(folder, info):
    info['updated'] = now()
    save(folder / 'supervisor-status.json', info)
    state, work = load(OUT / 'state.json'), load(OUT / 'work-status.json')
    work.update(updated=info['updated'], accepted_count=len(state['accepted_upgrades']),
                pending_certificate=state.get('pending_certificate'),
                current_goal_turn_classification=info['status'])
    work['active_runs'] = [folder.name] if info.get('runner_active') else []
    work['current_verified_progress'][folder.name] = dict(info)
    work['certificate_004_preflight'] = 'certificate-004/validation-result.json'
    work['selection_development_008_independent_audit'] = 'selection-development-008-independent-audit.json'
    awake = folder / 'awake-request.json'
    if awake.exists():
        work['awake_helper'] = load(awake)
    if info.get('audit_passed'):
        audit = load(OUT / 'independent-completion-audit.json')
        work['independent_audit'].update(last_verified_completed_attempts=audit['completed_attempts'],
                                        verified_games=audit['verified_games'], last_observed_at=audit['observed_at'])
    work['next_steps'] = info['next_steps']
    save(OUT / 'work-status.json', work)
    print(json.dumps({k: info[k] for k in ('updated', 'status', 'blocks', 'total', 'games')}), flush=True)


def main(args):
    from manage_progressive_workers import processes
    folder = (OUT / args.run).resolve()
    assert folder.parent == OUT.resolve() and folder.name.startswith('certificate-')
    # A Windows file lock is released even if the supervisor crashes.
    with (folder / 'supervisor.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        service_path = folder / 'service-state.json'
        manifest, service = check_preflight(folder, service_path)
        needle = str((folder / 'manifest.json').relative_to(ROOT)).replace('\\', '/')
        live = processes()
        assert any(p['pid'] == service['pid'] and 'training/progressive_inference_service.py' in p['command'].replace('\\', '/') for p in live)
        assert not any(needle in p['command'].replace('\\', '/') and
                       any(driver in p['command'] for driver in ('run_progressive_with_service.py', 'progressive_campaign.py')) for p in live), 'A runner already exists'
        info = dict(run=folder.name, status='starting', started_at=now(), supervisor_pid=os.getpid(),
                    manifest_sha256=sha(folder / 'manifest.json'), supervisor_sha256=sha(__file__),
                    runner_active=False, service_pid=service['pid'], workers=8, streams=service['streams'],
                    next_steps=['Finish the unchanged frozen certificate, then independently audit all completed attempts.'],
                    **progress(folder, manifest['deals']))
        flags = subprocess.CREATE_NO_WINDOW
        options = dict(cwd=ROOT, creationflags=flags)
        runner = awake = None
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        with (folder / f'runner-{stamp}.stdout.log').open('w', encoding='utf-8') as stdout, \
             (folder / f'runner-{stamp}.stderr.log').open('w', encoding='utf-8') as stderr, \
             (folder / f'finalization-{stamp}.log').open('w', encoding='utf-8') as final_log:
            try:
                runner = subprocess.Popen([sys.executable, '-u', 'training/run_progressive_with_service.py',
                    '--manifest', needle, '--service', str(service_path.relative_to(ROOT)), '--workers', '8'],
                    stdout=stdout, stderr=stderr, **options)
                info.update(runner_pid=runner.pid, runner_active=True, status='running',
                            runner_stdout=str(stdout.name), runner_stderr=str(stderr.name))
                awake = subprocess.Popen([sys.executable, '-u', 'training/keep_progressive_awake.py',
                    '--manifest', needle, '--service', str(service_path.relative_to(ROOT))],
                    stdout=final_log, stderr=final_log, **options)
                info['awake_pid'] = awake.pid
                publish(folder, info)
                while runner.poll() is None:
                    try:
                        runner.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                    info.update(progress(folder, manifest['deals']))
                    publish(folder, info)
                info.update(runner_active=False, runner_exit_code=runner.returncode)
                assert runner.returncode == 0, f'Runner exited {runner.returncode}; resume the same manifest after diagnosis'
                info['status'] = 'independent-audit'
                publish(folder, info)
                state = load(OUT / 'state.json')
                command = [sys.executable, 'training/audit_progressive_goal.py']
                if state['goal_complete']:
                    command.append('--require-complete')
                subprocess.run(command, stdout=final_log, stderr=final_log, check=True, **options)
                attempt = next(a for a in state['certificate_attempts'] if a['run'] == folder.name)
                info.update(audit_passed=True, certificate_passed=attempt['passed'], goal_complete=state['goal_complete'],
                            accepted_count=len(state['accepted_upgrades']), completed_at=now(),
                            status='complete' if state['goal_complete'] else 'completed-certificate-not-accepted',
                            next_steps=['Three accepted upgrades independently verified; review the final results.'] if state['goal_complete'] else
                            ['Continue fresh development using audited opponent-proxies-v3-devshift and continuation-v3-fourp; preserve this failure and its alpha allocation.'])
            except BaseException as error:
                info.update(status='needs-attention', error=f'{type(error).__name__}: {error}',
                            runner_active=runner is not None and runner.poll() is None,
                            next_steps=['Inspect supervisor and runner logs; keep the same frozen manifest and all completed blocks.'])
                raise
            finally:
                # Never shut down a service while a still-live runner is using it.
                if runner is None or runner.poll() is not None:
                    stopped = subprocess.run([sys.executable, 'training/progressive_inference_service.py',
                        '--state', str(service_path.relative_to(ROOT)), '--stop'],
                        stdout=final_log, stderr=final_log, timeout=60, **options)
                    info['service_stop_exit_code'] = stopped.returncode
                    if awake is not None:
                        try:
                            awake.wait(timeout=40)
                        except subprocess.TimeoutExpired:
                            awake.terminate()
                            awake.wait(timeout=10)
                        info['awake_exit_code'] = awake.returncode
                publish(folder, info)
                subprocess.run([sys.executable, 'training/report_progressive_campaign.py'],
                               stdout=final_log, stderr=final_log, check=True, **options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    main(parser.parse_args())
