"""Launch one approved paired run over existing SSH aliases; no Pod provisioning."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def ssh(host, command, check=True):
    return subprocess.run(["ssh", host, command], check=check, text=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--worker", default="runpod1")
    parser.add_argument("--coordinator", default="runpod2")
    parser.add_argument("--remote-root", default="/workspace/coding-awareness")
    parser.add_argument("--resume", help="Coordinator checkpoint path relative to remote root")
    args = parser.parse_args()
    suite = json.loads(Path("coding/suite.json").read_text())
    run = next(x for x in suite["runs"] if x["run_id"] == args.run_id)
    output = "runs/" + args.run_id
    root = shlex.quote(args.remote_root)
    session = shlex.quote(args.run_id)
    # One shared pair: refuse duplicate launches while either task session is live.
    for host in (args.coordinator, args.worker):
        for candidate in suite["runs"]:
            live = ssh(host, "tmux has-session -t " + shlex.quote(candidate["run_id"]), check=False)
            if live.returncode == 0:
                raise RuntimeError(f"Existing suite session on {host}: {candidate['run_id']}")
    for role, host in (("coordinator", args.coordinator), ("worker", args.worker)):
        # Old events remain in the append-only logs; startup must not read stale
        # failure/exit markers from the previous attempt at this same run.
        ssh(host, f"cd {root} && rm -f " + shlex.join([
            output + f"/{role}.exit", output + f"/{role}-status.json"]))
        command = ["bash", "coding/session.sh", role, run["config"], output]
        if role == "coordinator" and args.resume:
            command += ["--resume", args.resume]
        shell = f"cd {root} && exec " + shlex.join(command)
        ssh(host, "tmux new-session -d -s " + session + " " + shlex.quote(shell))
        if role == "coordinator":
            while True:
                result = ssh(host, f"cd {root} && cat {shlex.quote(output + '/coordinator-status.json')}", check=False)
                if result.returncode == 0:
                    state = json.loads(result.stdout)
                    if state["event"] == "waiting_for_worker":
                        break
                    if state["event"] == "failed":
                        raise RuntimeError(state["error"])
                code = ssh(host, f"cd {root} && cat {shlex.quote(output + '/coordinator.exit')}", check=False)
                if code.returncode == 0:
                    raise RuntimeError(f"Coordinator exited before worker launch: {code.stdout.strip()}")
                time.sleep(5)
    print(json.dumps({"run_id": args.run_id, "launched": True}))


if __name__ == "__main__":
    main()
