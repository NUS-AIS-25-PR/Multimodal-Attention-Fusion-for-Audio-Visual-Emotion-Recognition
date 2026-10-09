"""External experiment observer/sequence. Never edits trainer/results/history."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time

REPO=Path('/home/louis/projects/MultimodalEmotionRecognition')
LOG=Path(__file__).resolve().parent
OUT=REPO/'outputs/speaker_independent_tracked_v1'
PYTHON=REPO/'.venv/bin/python'
COMMIT='0ae967afc654d3234620862437d63f2f33ab57e7'
MODELS=('audio','video','gated','chumachenko_ia','xattn')
state={'status':'starting','command':None,'model':None,'audio_gate':'pending','models':{},'commands':[]}
stop=threading.Event()
mutex=threading.RLock()

def now():return datetime.now(timezone.utc).isoformat()
def event(kind,**values):
    row={'utc':now(),'monotonic':time.monotonic(),'event':kind,**values}
    with mutex:
        with (LOG/'timeline.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return row

def save_state():
    with mutex:
        temporary=LOG/'status.tmp'
        temporary.write_text(json.dumps(state,indent=2))
        temporary.replace(LOG/'status.json')

def source_hash():
    h=hashlib.sha256()
    for p in sorted((REPO/'src').rglob('*.py')):
        h.update(str(p.relative_to(REPO/'src')).encode());h.update(p.read_bytes())
    return h.hexdigest()

BASE_SOURCE=source_hash()
def guard():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==COMMIT,'Git changed; stop next command'
    assert source_hash()==BASE_SOURCE,'Source changed; stop next command'
    subprocess.run(['git','diff','--exit-code','HEAD','--','src'],cwd=REPO,check=True)

def histories_gate():
    p=OUT/'fold_01/audio/history.csv'
    if not p.exists():return
    with p.open(newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)<3:return
    for epoch,row in enumerate(rows[:3],1):
        assert int(row['epoch'])==epoch
        for key,value in row.items():
            if key.startswith(('train_','val_')):
                score=float(value);assert math.isfinite(score),f'Nonfinite {key}'
                if not key.endswith('_loss'):assert 0<=score<=1
    with mutex:
        if state['audio_gate']=='pending':
            state['audio_gate']='passed';state['audio_first_three_epochs']=rows[:3]
            event('audio_three_epoch_local_history_gate_passed',rows=rows[:3]);save_state()
            print('AUDIO INITIAL 3-EPOCH GATE PASSED',flush=True)

def monitor():
    with (LOG/'gpu_samples.jsonl').open('a',buffering=1) as f:
        while not stop.is_set():
            try:
                r=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                sample={'utc':now(),'command':state['command'],'model':state['model'],'returncode':r.returncode,'gpu_csv':r.stdout.strip()}
                f.write(json.dumps(sample)+'\n')
                if state['audio_gate']=='pending':histories_gate()
            except Exception as exc:
                event('monitor_error',error_type=type(exc).__name__)
            stop.wait(10)

def summarize():
    report={'git_commit':COMMIT,'source_fingerprint':BASE_SOURCE,'output_root':str(OUT),'state':state,
            'timing_note':'Epoch wall intervals observed from setup/epoch stdout timestamps; include train+val and between-epoch artifact overhead. GPU usage sampled every 10 seconds, whole-device usage.'}
    for model in MODELS:
        path=OUT/'fold_01'/model/'metrics.json'
        if path.exists():
            record=json.loads(path.read_text())
            state['models'].setdefault(model,{}).update(best_epoch=record['best_epoch'],val_macro_f1=record['val_macro_f1'],test=record['test'])
    (LOG/'execution_summary.json').write_text(json.dumps(report,indent=2))


def main():
    guard()
    assert not OUT.exists(),'Production output root already exists; do not overwrite'
    ready=json.loads((REPO/'outputs/fold1_tracked_prelaunch/readiness.json').read_text())
    assert ready['complete'] and ready['pairs']==1440 and ready['git_commit']==COMMIT
    commands=[]
    for model in ('gated','chumachenko_ia','xattn'):
        commands.append([str(PYTHON),'-u','-m','revision.run','--data-root',str(REPO/'data'),'--fold','1','--model',model,
            '--num-workers','-1','--output-root',str(OUT),'--wandb-mode','offline','--wandb-project','ieee-spmb-2026',
            '--wandb-group','spmb2026-canonical-tracked-v1'])
    figure=[str(PYTHON),'-m','revision.figures','--input-root',str(OUT),'--output-dir',str(REPO/'outputs/paper_figures_fold1_tracked_v1')]
    (LOG/'commands.json').write_text(json.dumps({'commit':COMMIT,'source_fingerprint':BASE_SOURCE,'cwd':str(REPO),
        'env':{'PYTHONPATH':'src'},'training_commands':commands,'figure_command':figure,
        'restriction':'Fold 1 only, no concat; never reuse interrupted/random-split checkpoints'},indent=2))
    env=os.environ.copy();env['PYTHONPATH']='src'
    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
    state['status']='running';save_state()
    try:
        with (LOG/'epoch_timings.csv').open('x',newline='',buffering=1) as timings:
            writer=csv.DictWriter(timings,fieldnames=['model','epoch','observed_epoch_wall_seconds','utc']);writer.writeheader()
            for model,cmd in zip(('gated','chumachenko_ia','xattn'),commands):
                guard();state.update(command=model,model=None);save_state()
                start=event('command_start',model=model,command=cmd)
                print('START',model,start['utc'],flush=True)
                with (LOG/f'{model}.log').open('x',buffering=1) as raw:
                    child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                    event('child_pid',model=model,pid=child.pid)
                    for line in child.stdout:
                        raw.write(line)
                        clean=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',line).strip()
                        match=re.match(r'\[REVISION\] fold=1 model=(\w+) smoke=False',clean)
                        if match:
                            name=match[1];assert name in MODELS
                            state['model']=name;entry=event('model_start',model=name)
                            state['models'][name]={'start_utc':entry['utc'],'start_monotonic':entry['monotonic'],'epochs':0}
                            print('MODEL',name,entry['utc'],flush=True);save_state()
                        name=state['model']
                        if name and clean.startswith('[INFO] Trainable parameters after stage setup:'):
                            state['models'][name]['last_epoch_boundary']=time.monotonic()
                            event('epoch_loop_start',model=name)
                        epoch=re.match(r'Epoch (\d+) \|',clean)
                        if epoch:
                            t=time.monotonic();m=state['models'][name]
                            seconds=t-m.get('last_epoch_boundary',t)
                            writer.writerow({'model':name,'epoch':int(epoch[1]),'observed_epoch_wall_seconds':round(seconds,3),'utc':now()});timings.flush()
                            m.update(last_epoch_boundary=t,epochs=int(epoch[1]),last_epoch_seconds=seconds,last_epoch_line=clean)
                            event('epoch_finished',model=name,epoch=int(epoch[1]),seconds=seconds,line=clean)
                            print(name,clean,'wall_seconds',round(seconds,2),flush=True);save_state()
                        if 'Early stopping triggered' in clean:
                            state['models'][name]['early_stopping']=True;event('early_stopping',model=name,line=clean)
                        if clean.startswith('Best val macro-F1:'):
                            entry=event('model_finished',model=name,line=clean);m=state['models'][name]
                            m.update(end_utc=entry['utc'],runtime_seconds=entry['monotonic']-m['start_monotonic'])
                            print('FINISHED',name,clean,flush=True);save_state()
                        if re.search(r'warning|traceback|out of memory|\bnan\b',clean,re.I):event('diagnostic',model=name,line=clean[:3000])
                        if re.search(r'CUDA out of memory|\b(?:nan|inf)\b',clean,re.I):
                            event('fatal_numeric_or_oom',model=name,line=clean);child.terminate()
                    code=child.wait()
                entry={'model':model,'exit_code':code,'runtime_seconds':time.monotonic()-start['monotonic']}
                state['commands'].append(entry);event('command_finished',**entry);save_state()
                print('EXIT',model,code,flush=True)
                if code:raise RuntimeError(f'{model} command failed; subsequent commands not launched')
                histories_gate();assert state['audio_gate']=='passed','Audio 3-epoch history gate did not pass'
            guard();state.update(status='generating_figures',command='figures',model=None);save_state()
            with (LOG/'figures.log').open('x') as f:
                code=subprocess.run(figure,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
            assert code==0,'Figure generation failed'
            state['status']='complete';event('all_fold1_commands_and_figures_complete');print('COMPLETE',now(),flush=True)
            return 0
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        event('sequence_failed',error_type=type(exc).__name__,error=str(exc));print('FAILED',str(exc),flush=True)
        return 1
    finally:
        stop.set();monitor_thread.join(timeout=15);save_state();summarize()

if __name__=='__main__':sys.exit(main())
