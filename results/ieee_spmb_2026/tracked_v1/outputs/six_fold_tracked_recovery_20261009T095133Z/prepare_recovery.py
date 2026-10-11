from pathlib import Path
import csv,hashlib,importlib.metadata,json,subprocess,gc
import cv2,torch
from revision.audit import audit_dataset,require_complete
from revision.protocol import expected_run_config,validate_completed_run
from models.wavlm_audio import WavLMAudioEncoder
from utils.face_crop import MediaPipeFaceDetector
from transformers.utils import logging as hf_logging
REPO=Path('/home/louis/projects/MultimodalEmotionRecognition');ROOT=REPO/'outputs/speaker_independent_tracked_v1'
OLD=REPO/'outputs/six_fold_tracked_resume_20261009T044819Z'
C=Path(Path('/tmp/spmb_recovery_control_path').read_text())
COMMIT='0ae967afc654d3234620862437d63f2f33ab57e7';SOURCE='e0ac3ba8452d283be16cda83add02069020b5e917f0e760eb79e134c07da9f6b'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==COMMIT
h=hashlib.sha256()
for p in sorted((REPO/'src').rglob('*.py')):h.update(str(p.relative_to(REPO/'src')).encode());h.update(p.read_bytes())
assert h.hexdigest()==SOURCE
baseline=json.loads((REPO/'outputs/fold1_tracked_prelaunch/packages.json').read_text())
assert {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}==baseline
print('Git, source, all111 package versions match Fold1',flush=True)
audit=audit_dataset(REPO/'data');require_complete(audit)
assert audit['fingerprint']==json.loads((OLD/'preflight.json').read_text())['dataset_fingerprint']
print('Audit:',audit['total_pairs'],'complete:',audit['complete'],flush=True)
completed=[];base=None;failed=ROOT/'fold_06/video'
for fold in sorted(ROOT.glob('fold_*')):
 revision=json.loads((fold/'split.json').read_text())
 assert revision['git_commit']==COMMIT and revision['source_fingerprint']==SOURCE and revision['dataset_fingerprint']==audit['fingerprint']
 assert revision['versions']=={k:importlib.metadata.version(k) for k in ('torch','torchvision','transformers','numpy','scikit-learn')}
 identity={k:v for k,v in revision.items() if k!='split'}
 if base is None:base=identity
 assert base==identity
 for run in sorted(p for p in fold.iterdir() if p.is_dir()):
  if run==failed:continue
  validate_completed_run(run,expected_run_config(revision,run.name));completed.append(run)
assert len(completed)==18
print('Completed runs fully validated:',len(completed),flush=True)
assert failed.exists() and not (failed/'metrics.json').exists()
rows=list(csv.DictReader((failed/'history.csv').open()));assert len(rows)==13
checkpoint=torch.load(failed/'best.pt',map_location='cpu',weights_only=True)
print('Interrupted checkpoint keys:',list(checkpoint),'best epoch',checkpoint.get('epoch'),flush=True)
assert not any(k in checkpoint for k in ('optimizer','scheduler','rng_state','optimizer_state_dict'))
del checkpoint;gc.collect()
video=sorted((REPO/'data/Actor_09').glob('02-*.mp4'))[0]
cap=cv2.VideoCapture(str(video));ok,frame=cap.read();cap.release();assert ok
bbox=MediaPipeFaceDetector().detect_face_bbox(frame);assert bbox and bbox[2]>bbox[0] and bbox[3]>bbox[1]
print('Real MediaPipe bbox',bbox,flush=True)
hf_logging.disable_progress_bar()
model=WavLMAudioEncoder(num_classes=8);assert model.wavlm.config.hidden_size==768
del model;gc.collect();print('Default production WavLM load passed',flush=True)
def snapshot(paths):
 result={}
 for folder in paths:
  for p in sorted(folder.rglob('*')) if folder.is_dir() else [folder]:
   if not p.is_file():continue
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   result[str(p)]={'sha256':h.hexdigest(),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
 return result
# Ensure every earlier preserved completed artifact is still intact.
old_snapshot=json.loads((OLD/'completed_preservation_snapshot.json').read_text())
now_snapshot=snapshot([Path(k) for k in old_snapshot])
assert now_snapshot==old_snapshot
preserved=snapshot(completed+[f/'split.json' for f in ROOT.glob('fold_*')])
(C/'completed_preservation_snapshot.json').write_text(json.dumps(preserved,indent=2))
before=snapshot([failed]);archive=C/'interrupted_runs/fold_06/video';archive.parent.mkdir(parents=True)
(C/'interrupted_snapshot_before.json').write_text(json.dumps(before,indent=2))
failed.rename(archive);after=snapshot([archive])
assert {str(Path(k).relative_to(failed)):v for k,v in before.items()}=={str(Path(k).relative_to(archive)):v for k,v in after.items()}
(C/'interrupted_snapshot_after.json').write_text(json.dumps(after,indent=2))
prior=json.loads((OLD/'status.json').read_text());(C/'prior_execution_state.json').write_text(json.dumps(prior,indent=2))
report={'passed':True,'complete':True,'pairs':audit['total_pairs'],'git_commit':COMMIT,'source_fingerprint':SOURCE,'dataset_fingerprint':audit['fingerprint'],'mediapipe_bbox':bbox,'default_wavlm_load_checks':1,'completed_validated':[str(p) for p in completed],'archived_incomplete':str(archive),'prior_control':str(OLD),'environment_packages_match_fold1':111,'restart_note':'Fold6 video restarts from epoch1; original13epochs preserved, no partial checkpoint warm-start.'}
(C/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
