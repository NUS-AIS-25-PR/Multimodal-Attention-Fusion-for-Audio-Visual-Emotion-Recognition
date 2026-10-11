from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, importlib.util, json
import cv2
from revision.audit import audit_dataset, require_complete
from revision.protocol import expected_run_config, validate_completed_run
from utils.face_crop import MediaPipeFaceDetector
REPO=Path('/home/louis/projects/MultimodalEmotionRecognition')
ROOT=REPO/'outputs/speaker_independent_tracked_v1'
OLD=REPO/'outputs/six_fold_tracked_execution_20261008T192320Z'
spec=importlib.util.spec_from_file_location('previous_observer','/tmp/spmb_continue_six_fold.py')
observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
observer.guard();observer.preservation()
audit=audit_dataset(REPO/'data');require_complete(audit)
assert audit['fingerprint']==json.loads((OLD/'preflight.json').read_text())['dataset_fingerprint']
completed=[]
for fold in sorted(ROOT.glob('fold_*')):
 revision=json.loads((fold/'split.json').read_text())
 assert revision['dataset_fingerprint']==audit['fingerprint']
 for run in sorted(p for p in fold.iterdir() if p.is_dir()):
  if run==ROOT/'fold_02/gated': continue
  validate_completed_run(run, expected_run_config(revision, run.name));completed.append(run)
assert len(completed)==7
failed=ROOT/'fold_02/gated'
assert failed.is_dir() and not (failed/'best.pt').exists() and not (failed/'metrics.json').exists()
with (failed/'history.csv').open() as f: assert len(list(csv.DictReader(f)))==0, 'Failed run contains epochs'
video=sorted((REPO/'data/Actor_09').glob('02-*.mp4'))[0]
cap=cv2.VideoCapture(str(video));ok,frame=cap.read();cap.release();assert ok
detector=MediaPipeFaceDetector();bbox=detector.detect_face_bbox(frame)
assert bbox and bbox[2]>bbox[0] and bbox[3]>bbox[1]
control=REPO/'outputs'/('six_fold_tracked_resume_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
control.mkdir(exist_ok=False)
def snapshot(folder):
 result={}
 for p in sorted(folder.rglob('*')):
  if not p.is_file():continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  result[str(p)]={'sha256':h.hexdigest(),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
 return result
preserved={}
for run in completed: preserved.update(snapshot(run))
for fold in ROOT.glob('fold_*'): preserved.update(snapshot(fold/'split.json') if (fold/'split.json').is_dir() else {str(fold/'split.json'):{'sha256':hashlib.sha256((fold/'split.json').read_bytes()).hexdigest(),'size':(fold/'split.json').stat().st_size,'mtime_ns':(fold/'split.json').stat().st_mtime_ns}})
(control/'completed_preservation_snapshot.json').write_text(json.dumps(preserved,indent=2))
archive=control/'interrupted_runs/fold_02/gated';archive.parent.mkdir(parents=True)
before=snapshot(failed)
(control/'interrupted_snapshot_before.json').write_text(json.dumps(before,indent=2))
failed.rename(archive)
after=snapshot(archive)
assert {str(Path(k).relative_to(failed)):v for k,v in before.items()}=={str(Path(k).relative_to(archive)):v for k,v in after.items()}
(control/'interrupted_snapshot_after.json').write_text(json.dumps(after,indent=2))
prior=json.loads((OLD/'status.json').read_text())
(control/'prior_execution_state.json').write_text(json.dumps(prior,indent=2))
preflight={'passed':True,'complete':True,'pairs':audit['total_pairs'],'git_commit':observer.COMMIT,'source_fingerprint':observer.SOURCE,'dataset_fingerprint':audit['fingerprint'],'mediapipe_bbox':bbox,'default_wavlm_load_checks':2,'completed_validated':[str(p) for p in completed],'archived_incomplete':str(archive),'prior_control':str(OLD),'environment_packages_match_fold1':111}
(control/'preflight.json').write_text(json.dumps(preflight,indent=2))
Path('/tmp/spmb_six_fold_resume_path').write_text(str(control))
print(json.dumps(preflight,indent=2));print('CONTROL',control)
