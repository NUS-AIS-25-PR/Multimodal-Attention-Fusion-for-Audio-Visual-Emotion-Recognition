"""External execution observer. Repository source/settings/results remain untouched."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import threading
import time

REPO=Path('/home/louis/projects/MultimodalEmotionRecognition')
LOG=Path('/home/louis/projects/MultimodalEmotionRecognition/outputs/six_fold_tracked_recovery_20261009T095133Z')
ROOT=REPO/'outputs/speaker_independent_tracked_v1'
PYTHON=REPO/'.venv/bin/python'
COMMIT='0ae967afc654d3234620862437d63f2f33ab57e7'
SOURCE='e0ac3ba8452d283be16cda83add02069020b5e917f0e760eb79e134c07da9f6b'
METHODS=('audio','video','gated','chumachenko_ia','xattn')
PACKAGES=json.loads((REPO/'outputs/fold1_tracked_prelaunch/packages.json').read_text())
SNAPSHOT=json.loads((LOG/'completed_preservation_snapshot.json').read_text())
PRIOR=json.loads((LOG/'prior_execution_state.json').read_text())
STATE={'status':'starting','command':None,'fold':None,'model':None,'models':{k:v for k,v in PRIOR['models'].items() if 'end_utc' in v},'commands':list(PRIOR['commands']),'skip_count':0,'historical_failures':PRIOR.get('historical_failures',[])+[{'control':json.loads((LOG/'preflight.json').read_text())['prior_control'],'error':PRIOR.get('guard_error'),'model':'video','fold':6,'epochs':13,'cause':'Dependency drift detected by guard; six packages restored to exact Fold1 versions. Computer restarted afterward. Partial video run archived and restarted from epoch1.','archived_artifacts':json.loads((LOG/'preflight.json').read_text())['archived_incomplete']}]}
LOCK=threading.RLock();STOP=threading.Event();CHILD=None

def now():return datetime.now(timezone.utc).isoformat()
def event(kind,**values):
 r={'utc':now(),'monotonic':time.monotonic(),'event':kind,**values}
 with LOCK:
  with (LOG/'timeline.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 return r

def save():
 with LOCK:
  p=LOG/'status.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(LOG/'status.json')

def guard():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==COMMIT,'Git SHA changed'
 h=hashlib.sha256()
 for p in sorted((REPO/'src').rglob('*.py')):
  h.update(str(p.relative_to(REPO/'src')).encode());h.update(p.read_bytes())
 assert h.hexdigest()==SOURCE,'Experiment source changed'
 assert {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}==PACKAGES,'Package environment changed'

def preservation():
 for name,old in SNAPSHOT.items():
  p=Path(name);h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  assert {'sha256':h.hexdigest(),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}==old,'Completed artifact changed: '+name
 event('completed_artifacts_preservation_verified',files=len(SNAPSHOT))


def monitor():
 last_guard=0
 with (LOG/'gpu_samples.jsonl').open('a',buffering=1) as f:
  while not STOP.is_set():
   try:
    if time.monotonic()-last_guard>=60:
     guard();last_guard=time.monotonic()
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    f.write(json.dumps({'utc':now(),'fold':STATE['fold'],'model':STATE['model'],'command':STATE['command'],'returncode':r.returncode,'gpu_csv':r.stdout.strip()})+'\n')
   except AssertionError as e:
    event('fatal_provenance_change',error=str(e));STATE['guard_error']=str(e);save()
    if CHILD is not None:CHILD.terminate()
    STOP.set()
   except Exception as e:event('monitor_error',error_type=type(e).__name__)
   STOP.wait(10)

def execute(label,cmd,env,writer,timings):
 global CHILD
 guard();STATE.update(command=label,fold=None,model=None);save()
 started=event('command_start',model=label,command=cmd);print('START',label,started['utc'],flush=True)
 with (LOG/f'{label}.log').open('x',buffering=1) as raw:
  CHILD=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  event('child_pid',model=label,pid=CHILD.pid)
  for line in CHILD.stdout:
   raw.write(line);text=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',line).strip()
   match=re.match(r'\[REVISION\] fold=(\d+) model=(\w+) smoke=False',text)
   if match:
    fold,model=int(match[1]),match[2];assert 2<=fold<=6 and model in METHODS,'Unexpected training scope'
    STATE.update(fold=fold,model=model);key=f'fold_{fold:02d}/{model}';entry=event('model_start',fold=fold,model=model)
    STATE['models'][key]={'start_utc':entry['utc'],'start_monotonic':entry['monotonic'],'epochs':0}
    print('MODEL',key,entry['utc'],flush=True);save()
   if text.startswith('[REVISION] SKIP complete matching run:'):
    STATE['skip_count']+=1;event('completed_skip',line=text);print(text,flush=True);save()
   key=f"fold_{STATE['fold']:02d}/{STATE['model']}" if STATE['fold'] else None
   if key and text.startswith('[INFO] Trainable parameters after stage setup:'):
    STATE['models'][key]['epoch_boundary']=time.monotonic()
   epoch=re.match(r'Epoch (\d+) \|',text)
   if epoch:
    model=STATE['models'][key];t=time.monotonic();seconds=t-model.get('epoch_boundary',t)
    model.update(epoch_boundary=t,epochs=int(epoch[1]),last_epoch_line=text,last_epoch_seconds=seconds)
    writer.writerow({'fold':STATE['fold'],'model':STATE['model'],'epoch':int(epoch[1]),'observed_epoch_wall_seconds':round(seconds,3),'utc':now()});timings.flush()
    event('epoch_finished',fold=STATE['fold'],model=STATE['model'],epoch=int(epoch[1]),seconds=seconds,line=text)
    print(key,text,'wall_seconds',round(seconds,2),flush=True);save()
   if key and 'Early stopping triggered' in text:
    STATE['models'][key]['early_stopping']=True;event('early_stopping',fold=STATE['fold'],model=STATE['model'])
   if key and text.startswith('Best val macro-F1:'):
    row=event('model_finished',fold=STATE['fold'],model=STATE['model'],line=text);model=STATE['models'][key]
    model.update(end_utc=row['utc'],runtime_seconds=row['monotonic']-model['start_monotonic'])
    print('FINISHED',key,text,flush=True);save()
   if re.search(r'warning|traceback|out of memory|\bnan\b',text,re.I):event('diagnostic',fold=STATE['fold'],model=STATE['model'],line=text[:3000])
   if re.search(r'CUDA out of memory|\b(?:nan|inf)\b',text,re.I):
    event('fatal_numeric_or_oom',line=text);CHILD.terminate()
  code=CHILD.wait();CHILD=None
 entry={'model':label,'exit_code':code,'runtime_seconds':time.monotonic()-started['monotonic']}
 STATE['commands'].append(entry);event('command_finished',**entry);save();print('EXIT',label,code,flush=True)
 assert code==0,label+' failed; later commands not launched'
 assert not STATE.get('guard_error'),STATE.get('guard_error')
 preservation()

def main():
 global CHILD
 env=os.environ.copy();env['PYTHONPATH']='src'
 guard();preservation()
 ready=json.loads((LOG/'preflight.json').read_text());assert ready['passed'] and ready['complete']
 commands=[]
 for model in ('gated','chumachenko_ia','xattn'):
  commands.append([str(PYTHON),'-u','-m','revision.run','--data-root',str(REPO/'data'),'--model',model,'--num-workers','-1',
   '--output-root',str(ROOT),'--wandb-mode','offline','--wandb-project','ieee-spmb-2026','--wandb-group','spmb2026-canonical-tracked-v1'])
 figures=[str(PYTHON),'-m','revision.figures','--input-root',str(ROOT),'--output-dir',str(REPO/'outputs/paper_figures_six_fold_tracked_v1')]
 for model in METHODS:figures.extend(['--model',model])
 (LOG/'commands.json').write_text(json.dumps({'git_commit':COMMIT,'source_fingerprint':SOURCE,'training_commands':commands,
  'figure_command':figures,'env':{'PYTHONPATH':'src'},'no_concat':True,'existing_fold1_must_remain_unchanged':True},indent=2))
 STATE.update(status='running',start_utc=now(),start_monotonic=time.monotonic());save()
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  with (LOG/'epoch_timings.csv').open('x',newline='',buffering=1) as timings:
   writer=csv.DictWriter(timings,fieldnames=['fold','model','epoch','observed_epoch_wall_seconds','utc']);writer.writeheader()
   for model,cmd in zip(('gated','chumachenko_ia','xattn'),commands):execute(model,cmd,env,writer,timings)
  guard();STATE.update(status='finalizing_metrics',command='metrics',fold=None,model=None);save()
  with (LOG/'final_metrics.log').open('x') as f:
   code=subprocess.run([str(PYTHON),'/home/louis/projects/MultimodalEmotionRecognition/outputs/six_fold_tracked_recovery_20261009T095133Z/finalize_six_fold.py',str(LOG)],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  assert code==0,'Final metrics incomplete or invalid'
  STATE.update(status='generating_figures',command='figures');save()
  with (LOG/'figures.log').open('x') as f:code=subprocess.run(figures,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  assert code==0,'Figure generation failed'
  manifest=json.loads((REPO/'outputs/paper_figures_six_fold_tracked_v1/figures_manifest.json').read_text())
  assert set(manifest['models'])==set(METHODS) and not manifest['missing_models'] and not manifest['smoke_not_for_paper']
  assert all(d['complete'] and d['n_folds']==6 for d in manifest['models'].values())
  assert all(sum(sum(row) for row in matrix)==1440 for matrix in manifest['pooled_counts'].values())
  for path in manifest['figures']:
   assert Path(path).is_file() and Path(path).stat().st_size>1000
   if path.endswith('.svg'):assert 'INCOMPLETE' not in Path(path).read_text()
  preservation();guard();STATE.update(status='complete',end_utc=now(),figures_exported=len(manifest['figures']))
  event('all_six_fold_runs_metrics_figures_complete',exports=len(manifest['figures']));print('COMPLETE',now(),flush=True)
  return 0
 except Exception as e:
  STATE.update(status='failed',error_type=type(e).__name__,error=str(e));event('sequence_failed',error=str(e));print('FAILED',str(e),flush=True)
  if CHILD is not None:CHILD.terminate();CHILD.wait()
  return 1
 finally:
  STOP.set();thread.join(timeout=15);STATE['continuation_elapsed_seconds']=time.monotonic()-STATE['start_monotonic'];save()
  (LOG/'execution_summary.json').write_text(json.dumps({'git_commit':COMMIT,'source_fingerprint':SOURCE,'state':STATE,
   'timing_note':'Observed train/val/test process wall durations; GPU samples whole-device usage every 10 seconds.'},indent=2))

if __name__=='__main__':sys.exit(main())
