"""Sequence fresh baseline, paired training, coding intervals and final evaluation.

Uses the shared trainer and original generator, with vLLM for new judge stages.
The already-started baseline retains its original judge. A process/file lock
prevents a second local controller; stage PID and GPU
ownership checks allow reconnecting without duplicating remote jobs.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import traceback

from coding2.interval_control import (Controller, ROOT, REMOTE, PYTHON, JUDGE_PYTHON,
    HOSTS, atomic_json, command, remote, ssh, wait_stage)

RUN = 'induced-coding-language'
CONFIG = 'induced_coding_language/config.json'
SUITE = 'induced_coding_language/suite.json'


class InducedController(Controller):
    def __init__(self):
        super().__init__(RUN, CONFIG, SUITE, RUN)

    def stage(self, host, folder, name, module, arguments, python=PYTHON):
        if module == 'coding2.judge' and folder != self.run+'/baseline/judge':
            suite = json.loads((ROOT/self.suite_path).read_text())
            module = 'coding2.judge_vllm'
            python = suite['judge_python']
            arguments = [*arguments,'--runtime',suite['judge_runtime']]
        return super().stage(host,folder,name,module,arguments,python)

    def prepare_full(self, folder, stage, step, checkpoint):
        for host in HOSTS.values():
            remote(host, f'''
from pathlib import Path
import hashlib,json,shutil
p=Path({folder!r}); p.mkdir(parents=True,exist_ok=True)
cfg=json.loads(Path({self.config_path!r}).read_text())
assert cfg=={self.cfg!r}
value={{'run_id':{self.run_id!r},'stage':{stage!r},'step':{step},'checkpoint':{checkpoint!r}}}
meta=p/'input-provenance.json'
if meta.exists(): assert json.loads(meta.read_text())==value
else:
 meta.write_text(json.dumps(value,indent=2)+'\\n')
 shutil.copy2({self.config_path!r},p/'config.json')
 for name in ('coding2','induced_coding_language'):
  shutil.copytree(name,p/('source-'+name),ignore=shutil.ignore_patterns('__pycache__','results'))
 manifest={{str(x.relative_to(p)):hashlib.sha256(x.read_bytes()).hexdigest() for d in p.glob('source-*') if d.is_dir() for x in d.rglob('*') if x.is_file()}}
 (p/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
print(json.dumps({{'ready':True}}))
''')

    def full_evaluation(self, stage, step):
        folder = self.run + '/' + stage
        local = self.local / stage
        if (local/'verification.json').exists():
            receipt = json.loads((local/'verification.json').read_text())
            assert receipt['run_id'] == self.run_id and receipt['step'] == step
            return
        checkpoint, info = self.checkpoint(step) if step else (None, None)
        self.prepare_full(folder, stage, step, info)
        self.update(phase=stage+'_generation', evaluation_step=step)
        for role, host in HOSTS.items():
            args = ['--config',self.config_path,'--role',role,'--output',folder]
            if checkpoint:
                args += ['--checkpoint',checkpoint]
            self.stage(host,folder,role,'coding2.evaluate',args)
        for role, host in HOSTS.items():
            wait_stage(host,folder,role)
            print(ssh(host, f'cd {REMOTE} && HF_HOME=/workspace/hf_cache ' + shlex.join([
                PYTHON,'-m','induced_coding_language.audit','generation',stage,'--role',role])),flush=True)
        local.mkdir(parents=True,exist_ok=True)
        command(['rsync','-azc','--include=worker*','--exclude=*',
                 HOSTS['worker']+':'+REMOTE+'/'+folder+'/',str(local)+'/'])
        files = sorted(str(p) for p in local.glob('worker*') if p.is_file())
        command(['rsync','-azc',*files,HOSTS['coordinator']+':'+REMOTE+'/'+folder+'/'])
        self.update(phase=stage+'_judging',evaluation_step=step)
        judge = folder+'/judge'
        self.stage(HOSTS['coordinator'],judge,'judge','coding2.judge',[
            '--config',self.config_path,'--input',folder+'/worker-generations.jsonl',
            folder+'/coordinator-generations.jsonl','--output',judge],JUDGE_PYTHON)
        wait_stage(HOSTS['coordinator'],judge,'judge')
        self.update(phase=stage+'_verification',evaluation_step=step)
        print(ssh(HOSTS['coordinator'],f'cd {REMOTE} && HF_HOME=/workspace/hf_cache '+shlex.join([
            JUDGE_PYTHON,'-m','induced_coding_language.audit','full',stage])),flush=True)
        command(['rsync','-azc','--exclude=wandb',HOSTS['coordinator']+':'+REMOTE+'/'+folder+'/',str(local)+'/'])
        receipt = json.loads((local/'verification.json').read_text())
        for name, expected in receipt['files'].items():
            p = local/name
            assert p.stat().st_size == expected['bytes']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == expected['sha256']
        self.update(phase=stage+'_complete',evaluation_step=step)

    def run_all(self):
        if not self.state.get('baseline_complete'):
            self.full_evaluation('baseline',0)
            self.update(baseline_complete=True,backfill_complete=True)
        while not self.state['training_terminal'] and self.state['current_step'] < self.cfg['steps']:
            start = self.state['current_step']
            end = min((start//100+1)*100,self.cfg['steps'])
            end, terminal = self.train_segment(start,end)
            if end % 100 == 0:
                self.interval(end,False)
            self.update(current_step=end,training_terminal=terminal)
        best = remote(HOSTS['coordinator'],f"from pathlib import Path\nimport json\nprint(Path({(self.run+'/best.json')!r}).read_text())")
        atomic_json(self.local/'best.json',best)
        self.full_evaluation('selected',best['best_step'])
        self.update(phase='evaluation_complete',selected_step=best['best_step'])


def main():
    os.chdir(ROOT)
    control = InducedController()
    lock_path = control.local/'interval/controller.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (lock_path.parent/'controller.pid').write_text(str(os.getpid())+'\n')
        try:
            control.run_all()
        except BaseException:
            control.update(phase='failed',error=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
