"""Bounded structured progress; full subprocess diagnostics stay in private logs."""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PREFIX = 'STARMAPR_PROGRESS '


def initialize_progress():
    if not os.getenv('STARMAPR_PROGRESS_FILE'):
        folder = Path('06_jobs') / ('cli-' + uuid.uuid4().hex)
        folder.mkdir(parents=True, mode=0o700)
        os.environ['STARMAPR_PROGRESS_FILE'] = str(folder.resolve() / 'progress.json')


def emit_progress(phase, state='running', **fields):
    event = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                 phase=phase, state=state, actor=os.getenv('STARMAPR_ACTOR'), **fields)
    path = os.getenv('STARMAPR_PROGRESS_FILE')
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
        temp.write_text(json.dumps(event), encoding='utf-8')
        os.replace(temp, target)
    print(PREFIX + json.dumps(event), file=sys.stderr, flush=True)


def run_logged(command, *, check=False, capture_output=True, text=True,
               encoding='utf-8', errors='replace', **kwargs):
    """subprocess.run-compatible capture with bounded tails and streamed phase events.

    Worker output timestamps indicate output, not useful progress. Idle heartbeats
    deliberately do not replace the latest nested phase/progress.json record.
    """
    phase = Path(str(command[1] if len(command) > 1 else command[0])).stem
    initialize_progress()
    log_path = Path(os.environ['STARMAPR_PROGRESS_FILE']).parent / (phase + '-' + uuid.uuid4().hex[:8] + '.log')
    emit_progress(phase, 'started', log=str(log_path))
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    env.update(kwargs.pop('env', {}) or {})
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding=encoding, errors=errors, env=env, **kwargs)
    messages = queue.Queue()

    def pump(stream, name):
        try:
            for line in iter(stream.readline, ''):
                messages.put((name, line))
        finally:
            stream.close()
            messages.put((name, None))

    for name in ('stdout', 'stderr'):
        threading.Thread(target=pump, args=(getattr(process, name), name), daemon=True).start()
    tails = dict(stdout='', stderr='')
    closed = 0
    last_output = None
    last_event = time.monotonic()
    try:
        with log_path.open('w', encoding='utf-8') as log:
            os.chmod(log_path, 0o600)
            while closed < 2:
                try:
                    name, line = messages.get(timeout=1)
                except queue.Empty:
                    if time.monotonic() - last_event >= 30:
                        # No claim of actual progress when a process is merely alive.
                        print(PREFIX + json.dumps(dict(phase=phase, state='waiting',
                              last_output_at=last_output)), file=sys.stderr, flush=True)
                        last_event = time.monotonic()
                    continue
                if line is None:
                    closed += 1
                    continue
                last_output = datetime.now(timezone.utc).isoformat()
                log.write(name + ': ' + line)
                log.flush()
                if line.startswith(PREFIX):
                    print(line.rstrip(), file=sys.stderr, flush=True)
                    last_event = time.monotonic()
                else:
                    tails[name] = (tails[name] + line)[-16000:]
    except BaseException:
        process.terminate()
        process.wait()
        raise
    code = process.wait()
    emit_progress(phase, 'completed' if code == 0 else 'failed',
                  exit_code=code, last_output_at=last_output, log=str(log_path))
    if check and code:
        raise subprocess.CalledProcessError(code, command, tails['stdout'], tails['stderr'])
    return subprocess.CompletedProcess(command, code, tails['stdout'], tails['stderr'])
