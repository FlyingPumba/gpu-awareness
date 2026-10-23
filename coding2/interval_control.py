"""Run the authorized backfill and sequential train/generate/judge intervals.

This local controller uses the existing two SSH aliases and detached stage
wrappers. A controller interruption leaves remote jobs and all records intact;
the same command reconnects to their recorded state without duplicating jobs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace/coding-awareness"
PYTHON = REMOTE + "/.venv-coding002/bin/python"
JUDGE_PYTHON = REMOTE + "/.venv-coding002-judge/bin/python"
HOSTS = {"worker": "runpod3", "coordinator": "runpod2"}


def command(args, **kwargs):
    try:
        return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs).stdout
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr, flush=True)
        raise


def ssh(host, shell):
    return command(["ssh", "-o", "BatchMode=yes", host, shell])


def remote(host, source):
    return json.loads(command(["ssh", "-o", "BatchMode=yes", host,
                               f"cd {REMOTE} && {PYTHON} -"], input=source))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def probe(host, folder, name):
    return remote(host, f'''
from pathlib import Path
import json,subprocess
p=Path({folder!r}); name={name!r}
pid=(p/(name+'.pid')).read_text().strip() if (p/(name+'.pid')).exists() else None
status=p/(name+'-status.json')
print(json.dumps({{'exit':(p/(name+'.exit')).read_text().strip() if (p/(name+'.exit')).exists() else None,
'status':json.loads(status.read_text()) if status.exists() else None,
'process':subprocess.run(['ps','-p',pid,'-o','args='],text=True,capture_output=True).stdout.strip() if pid else '',
'tmux':subprocess.run(['tmux','list-sessions','-F','#{{session_name}}'],text=True,capture_output=True).stdout.splitlines(),
'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()}}))
''')


def stage(host, folder, name, module, arguments, python=PYTHON):
    state = probe(host, folder, name)
    if state["exit"] is not None:
        if state["exit"] != "0":
            raise RuntimeError(f"{host} {folder}/{name} exited {state['exit']}; preserve and inspect its log")
        return
    if state["process"]:
        if module not in state["process"] or folder not in state["process"]:
            raise RuntimeError("Recorded PID does not belong to the expected stage")
        return
    if state["gpu"] or any(s.startswith("coding-002") for s in state["tmux"]):
        raise RuntimeError(f"GPU or experiment session already active on {host}")
    session = "coding-002-interval-" + Path(folder).name + "-" + name
    shell = f"cd {REMOTE} && exec " + shlex.join([
        "bash", "coding/stage-session.sh", folder, name, python, "-m", module, *arguments])
    ssh(host, "tmux new-session -d -s " + shlex.quote(session) + " " + shlex.quote(shell))


def wait_stage(host, folder, name):
    while True:
        state = probe(host, folder, name)
        if state["exit"] is not None:
            if state["exit"] != "0":
                raise RuntimeError(f"{host} {folder}/{name} exited {state['exit']}")
            if not state["process"] and not state["gpu"]:
                return state
        if state["status"] and state["status"].get("event") == "failed":
            raise RuntimeError(state["status"].get("error", "Stage failed"))
        time.sleep(15)


def collect_directory(host, remote_folder, local_folder):
    local_folder.mkdir(parents=True, exist_ok=True)
    command(["rsync", "-az", "--exclude=wandb", host + ":" + REMOTE + "/" + remote_folder + "/", str(local_folder) + "/"])


class Controller:
    def __init__(self, run_id):
        self.run_id = run_id
        self.run = "runs/" + run_id
        self.local = ROOT.parent / "coding-runs" / run_id
        self.plan = json.loads((self.local / "interval/plan.json").read_text())
        self.cfg = json.loads((ROOT / "coding2/config.json").read_text())
        self.state_path = self.local / "interval/controller-state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "run_id": run_id, "current_step": self.plan["resume_step"], "backfill_complete": False,
            "training_terminal": False}

    def update(self, **fields):
        if fields.get("phase") and fields["phase"] != "failed":
            self.state.pop("error", None)
        self.state.update(fields, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        atomic_json(self.state_path, self.state)
        print(json.dumps(self.state), flush=True)

    def checkpoint(self, step):
        path = self.run + f"/checkpoint-{step:06d}.pt"
        info = remote(HOSTS["coordinator"], f'''
from pathlib import Path
import hashlib,json,torch
p=Path({path!r}); s=torch.load(p,map_location='cpu',weights_only=False)
c=json.loads(Path('coding2/config.json').read_text())
assert s['step']=={step} and s['config']==c
assert all(t.dtype==torch.float32 for t in s['adapters'].values())
assert all(float(x['step'])=={step} for x in s['optimizer']['state'].values())
print(json.dumps({{'step':s['step'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
'selection':s['selection'],'adapter_sha256':hashlib.sha256(b''.join(t.contiguous().numpy().tobytes() for t in s['adapters'].values())).hexdigest()}}))
''')
        local_path = self.local / Path(path).name
        def backup_local():
            if not local_path.exists() or hashlib.sha256(local_path.read_bytes()).hexdigest() != info["sha256"]:
                command(["rsync", "-az", HOSTS["coordinator"] + ":" + REMOTE + "/" + path, str(local_path)])
            assert hashlib.sha256(local_path.read_bytes()).hexdigest() == info["sha256"]
        # Reuse the worker's existing training SSH link so checkpoints do not
        # traverse the local machine's upload connection a second time.
        transfer = '''set -euo pipefail
source /workspace/gpu-awareness-private/worker-link.env
ssh_command=(ssh -i "$PAIR_SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes
 -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$PAIR_KNOWN_HOSTS" -p "$PAIR_SSH_PORT")
printf -v rsync_ssh '%q ' "${ssh_command[@]}"
rsync -az -e "$rsync_ssh" "$PAIR_SSH_TARGET:"''' + shlex.quote(REMOTE + "/" + path) + " " + shlex.quote(REMOTE + "/" + path)
        # These independent copies used to serialize the local download and
        # Pod-to-Pod transfer. Both still finish and hash-check before generation.
        with ThreadPoolExecutor(max_workers=2) as pool:
            local_copy = pool.submit(backup_local)
            worker_copy = pool.submit(ssh, HOSTS["worker"], "bash -c " + shlex.quote(transfer))
            local_copy.result()
            worker_copy.result()
        received = remote(HOSTS["worker"], f"from pathlib import Path\nimport hashlib,json\nprint(json.dumps(hashlib.sha256(Path({path!r}).read_bytes()).hexdigest()))")
        assert received == info["sha256"]
        return path, info

    def interval(self, step, backfill):
        folder = self.run + f"/interval/step-{step:06d}"
        local_folder = self.local / f"interval/step-{step:06d}"
        if (local_folder / "verification.json").exists():
            receipt = json.loads((local_folder / "verification.json").read_text())
            assert receipt["step"] == step and receipt["run_id"] == self.run_id
            return
        self.update(phase="interval_generation", evaluation_step=step, backfill=backfill)
        checkpoint, info = self.checkpoint(step)
        for role, host in HOSTS.items():
            remote(host, f'''
from pathlib import Path
import hashlib,json,shutil
p=Path({folder!r}); p.mkdir(parents=True,exist_ok=True)
meta=p/'input-provenance.json'
value={{'step':{step},'backfill':{backfill!r},'checkpoint':{info!r},'problem_ids':{self.cfg['selection_problem_ids']!r}}}
if meta.exists(): assert json.loads(meta.read_text())==value
else:
 meta.write_text(json.dumps(value,indent=2)+'\\n')
 shutil.copy2('coding2/config.json',p/'config.json')
 shutil.copytree('coding2',p/'source-coding2',ignore=shutil.ignore_patterns('__pycache__','results'))
 manifest={{str(x.relative_to(p)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (p/'source-coding2').rglob('*') if x.is_file()}}
 (p/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
print(json.dumps({{'ready':True}}))
''')
            stage(host, folder, role, "coding2.evaluate", ["--config", "coding2/config.json", "--role", role,
                  "--output", folder, "--checkpoint", checkpoint, "--selection-subset"])
        for role, host in HOSTS.items():
            wait_stage(host, folder, role)
        # Copy only this completed role's files; the coordinator retains its own source and records.
        local_folder.mkdir(parents=True, exist_ok=True)
        command(["rsync", "-az", "--include=worker*", "--exclude=*", HOSTS["worker"] + ":" + REMOTE + "/" + folder + "/", str(local_folder) + "/"])
        files = sorted(str(p) for p in local_folder.glob("worker*") if p.is_file())
        command(["rsync", "-az", *files, HOSTS["coordinator"] + ":" + REMOTE + "/" + folder + "/"])
        self.update(phase="interval_judging", evaluation_step=step)
        judge_folder = folder + "/judge"
        stage(HOSTS["coordinator"], judge_folder, "judge", "coding2.judge", ["--config", "coding2/config.json",
              "--input", folder + "/worker-generations.jsonl", folder + "/coordinator-generations.jsonl",
              "--output", judge_folder, "--interval-step", str(step)], JUDGE_PYTHON)
        wait_stage(HOSTS["coordinator"], judge_folder, "judge")
        self.update(phase="interval_verification", evaluation_step=step)
        # The stage has finished uploading its numeric W&B history before audit.
        time.sleep(10)
        result = ssh(HOSTS["coordinator"], f"cd {REMOTE} && HF_HOME=/workspace/hf_cache " +
                     shlex.join([JUDGE_PYTHON, "-m", "coding2.interval_audit", folder, str(step)]))
        print(result, flush=True)
        collect_directory(HOSTS["coordinator"], folder, local_folder)
        receipt = json.loads((local_folder / "verification.json").read_text())
        for name, expected in receipt["files"].items():
            p = local_folder / name
            assert p.stat().st_size == expected["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest() == expected["sha256"]
        self.update(phase="interval_complete", evaluation_step=step)

    def train_segment(self, start, end):
        states = {role: probe(host, self.run, role) for role, host in HOSTS.items()}
        already_finished = all(s["exit"] == "0" and s["status"] and (
            (s["status"]["event"] == "paused" and s["status"]["step"] == end) or
            (s["status"]["event"] == "completed" and start < s["status"]["step"] <= end))
            for s in states.values())
        active = any(s["process"] for s in states.values())
        if active:
            for role, state in states.items():
                if state["process"] and ("coding2/" not in state["process"] and "coding2.train" not in state["process"]):
                    raise RuntimeError("Unexpected process owns a training PID")
        elif not already_finished:
            if any(s["gpu"] for s in states.values()):
                raise RuntimeError("An existing GPU job prevents the next training segment")
            self.update(phase="training_start", segment_start=start, segment_end=end)
            print(command(["python3", "coding2/launch.py", self.run_id, "--resume",
                           self.run + f"/checkpoint-{start:06d}.pt", "--pause-after-update", str(end)], cwd=ROOT), flush=True)
        self.update(phase="training", segment_start=start, segment_end=end)
        completed = {role: wait_stage(host, self.run, role) for role, host in HOSTS.items()}
        actual_step = completed["coordinator"]["status"]["step"]
        terminal = completed["coordinator"]["status"]["event"] == "completed"
        assert start < actual_step <= end and (terminal or actual_step == end)
        for role, state in completed.items():
            assert state["status"]["step"] == actual_step and state["status"]["event"] in ("paused", "completed")
        assert completed["worker"]["status"]["event"] == completed["coordinator"]["status"]["event"]
        atomic_json(self.local / f"interval/step-{actual_step:06d}/training-boundary.json",
                    {"step": actual_step, "terminal": terminal, "roles": {r: s["status"] for r, s in completed.items()}})
        return actual_step, terminal

    def run_all(self):
        # Verify the repaired live path once before working through historical
        # checkpoints. No further training advances until that interval is judged.
        if not self.state["backfill_complete"] and self.state["current_step"] == self.plan["resume_step"]:
            start = self.state["current_step"]
            end = min(start + self.cfg["eval_every"], self.cfg["steps"])
            end, terminal = self.train_segment(start, end)
            self.interval(end, False)
            self.update(current_step=end, training_terminal=terminal)
        if not self.state["backfill_complete"]:
            for step in self.plan["backfill_steps"]:
                self.interval(step, True)
            self.update(backfill_complete=True)
        while not self.state["training_terminal"] and self.state["current_step"] < self.cfg["steps"]:
            start = self.state["current_step"]
            cadence = (self.plan["judge_every_updates"]
                       if start >= self.plan.get("judge_cadence_effective_after_step", float("inf"))
                       else self.cfg["eval_every"])
            end = min((start // cadence + 1) * cadence, self.cfg["steps"])
            end, terminal = self.train_segment(start, end)
            if end % cadence == 0:
                self.interval(end, False)
            self.update(current_step=end, training_terminal=terminal)
        self.update(phase="training_and_intervals_complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    os.chdir(ROOT)
    controller = Controller(args.run_id)
    lock_path = controller.local / "interval/controller.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (lock_path.parent / "controller.pid").write_text(str(os.getpid()) + "\n")
        try:
            controller.run_all()
        except BaseException:
            controller.update(phase="failed", error=traceback.format_exc())
            raise
