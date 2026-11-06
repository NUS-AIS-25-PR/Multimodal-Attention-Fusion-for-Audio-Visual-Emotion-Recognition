"""Frozen six-fold exploratory protocol. Exclusive outputs; historical inputs read-only."""
import sys, os, json, csv, hashlib, time, statistics, subprocess, tempfile, shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'late_fusion_probe'))
from probe import ROOT,RUN,read,model,torch,audit_dataset,require_complete,fixed_folds,validate_local_artifacts,PAIR_SERVICE
from data.ravdess import DATASET_FACTORY
from utils.metrics import classification_metrics
from torch.utils.data import DataLoader
K=('accuracy','precision','recall','macro_f1')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
def write(p,rows):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def snapshot():
 return {str(p):dict(sha256=sha(p),mtime_ns=p.stat().st_mtime_ns,bytes=p.stat().st_size) for p in sorted(RUN.rglob('*')) if p.is_file()}
def score(p,y):return classification_metrics(p.argmax(1),y,8)
def select(grid):return min(grid,key=lambda r:(-r['macro_f1'],abs(round(r['alpha']-.5,1)),r['alpha']))
def check(rows,fold,part):
 expected={ '-'.join(f'{x:02d}' for x in (1,p.emotion,p.intensity,p.statement,p.repetition,p.actor)):p for p in PAIRS if p.actor in fold[part+'_actors']}
 assert len(rows)==len(expected)==240
 modalities={}
 for mode in ('audio','video'):
  mapping={}
  for r in rows:
   sid=r['sample_id'];assert sid not in mapping and sid in expected
   p=expected[sid];assert int(r['actor'])==p.actor and int(r['label'])==p.emotion-1
   l=torch.tensor(json.loads(r[mode+'_logits']));q=torch.tensor(json.loads(r[mode+'_probs']))
   assert l.shape==q.shape==(8,) and torch.isfinite(l).all() and torch.isfinite(q).all()
   assert (q>=0).all() and torch.allclose(q.sum(),torch.tensor(1.),atol=1e-6) and torch.allclose(l.softmax(0),q,atol=1e-6)
   mapping[sid]=(q,int(r['label']))
  assert set(mapping)==set(expected);modalities[mode]=mapping
 ids=sorted(expected);assert set(modalities['audio'])==set(modalities['video'])
 return ids,torch.stack([modalities['audio'][s][0] for s in ids]),torch.stack([modalities['video'][s][0] for s in ids]),torch.tensor([modalities['audio'][s][1] for s in ids])
def lightweight():
 import summarize
 with tempfile.TemporaryDirectory() as temp:
  temp=Path(temp)
  for part in ('val','test'):shutil.copyfile(HERE.parent/'late_fusion_probe'/f'fold1_{part}_predictions.csv',temp/f'fold1_{part}_predictions.csv')
  summarize.OUT=temp;summarize.main()
  p=temp/'FOLD1_PROBE_RESULTS.csv';first=p.read_bytes();summarize.main();assert p.read_bytes()==first
  rows=list(csv.DictReader(p.open()));assert len(rows)==6 and set(rows[0])=={'method','alpha',*K}
 assert select([dict(alpha=a,macro_f1=1) for a in (.4,.6)])['alpha']==.4
 assert select([dict(alpha=a,macro_f1=1) for a in (0.,.5,1.)])['alpha']==.5
 dump(HERE/'lightweight_tests.json',dict(status='PASS',checks=['original summary handles extra metrics fields','four metric schema','repeat produces identical six rows, no append','fixed tie rules'],gpu_inference=False))
def infer(n,c,nets,part):
 pp=[p for p in PAIRS if p.actor in c['revision']['split'][part+'_actors']]
 ds=DATASET_FACTORY.create(pp,num_classes=8,num_frames=c['frames'],augment=False,use_face_crop=c['use_face_crop'],use_wavlm=c['use_wavlm'])
 rows=[]
 with torch.inference_mode():
  for v,a,y,meta in DataLoader(ds,batch_size=16,shuffle=False,num_workers=2):
   logits={'audio':nets['audio'](a.cuda()).cpu(),'video':nets['video'](v.cuda()).cpu()}
   for i in range(len(y)):
    sid='-'.join(f'{int(x):02d}' for x in [1,meta['emotion'][i],meta['intensity'][i],meta['statement'][i],meta['repetition'][i],meta['actor'][i]])
    r=dict(fold=n,split=part,sample_id=sid,actor=int(meta['actor'][i]),label=int(y[i]))
    for mode,l in logits.items():r.update({mode+'_logits':json.dumps(l[i].tolist()),mode+'_probs':json.dumps(l[i].softmax(0).tolist())})
    rows.append(r)
 return rows

def main():
 global PAIRS
 torch.set_num_threads(4);torch.manual_seed(42)
 lightweight();before=snapshot();dump(HERE/'historical_before.json',before)
 try:
  audit=audit_dataset(ROOT/'data');require_complete(audit);PAIRS=PAIR_SERVICE.build_pairs(ROOT/'data')
  h=hashlib.sha256()
  for p in sorted((ROOT/'src').rglob('*.py')):h.update(str(p.relative_to(ROOT/'src')).encode());h.update(p.read_bytes())
  dump(HERE/'experiment.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),grid=[i/10 for i in range(11)],selection='validation macro_f1; ties nearest 0.5 then lower',class_order=list(range(8)),source_sha256=h.hexdigest(),data_audit=audit,script_sha256=sha(Path(__file__)),environment_reference=str(HERE.parent/'late_fusion_probe/environment.json')))
  allrows=[];selections=[];timings=[];repro=[]
  for fold in fixed_folds():
   n=fold['fold'];d=HERE/f'fold_{n:02d}';d.mkdir();nets={};sources={};configs={}
   for mode in ('audio','video'):
    src=RUN/f'fold_{n:02d}'/mode;c=read(src/'config.json');m=read(src/'metrics.json');ck=torch.load(src/'best.pt',map_location='cpu',weights_only=True)
    assert ck['config']==c and c['revision']['split']==fold and c['revision']['dataset_fingerprint']==audit['fingerprint'] and c['revision']['source_fingerprint']==h.hexdigest()
    assert ck['epoch']==m['best_epoch'] and ck['val_f1']==m['val_macro_f1'];validate_local_artifacts(src,c,m)
    net=model(c);net.load_state_dict(ck['model'],strict=True);configs[mode]=c
    if n>1:nets[mode]=net.cuda().eval()
    sources[mode]=dict(checkpoint=str(src/'best.pt'),sha256=before[str(src/'best.pt')]['sha256'],config=c,config_path=str(src/'config.json'),best_epoch=ck['epoch'])
    del ck,net
   assert all(configs['audio'][k]==configs['video'][k] for k in ('frames','use_face_crop','use_wavlm'))
   sources['cache_status']='复用已核验缓存' if n==1 else '本轮真实推理'
   if n==1:
    sources['cache']={part:dict(path=str(HERE.parent/'late_fusion_probe'/f'fold1_{part}_predictions.csv'),sha256=sha(HERE.parent/'late_fusion_probe'/f'fold1_{part}_predictions.csv')) for part in ('val','test')}
    sources['cache_provenance_limit']='Prior probe did not bind checkpoint SHA at creation; validated prior script/config/source/data evidence, cache logits-softmax and exact historical test predictions; current input hashes recorded now.'
   dump(d/'sources.json',sources)
   torch.cuda.reset_peak_memory_stats();start=time.perf_counter();parts={}
   for part in ('val','test'):
    if n==1:
     rows=list(csv.DictReader(Path(sources['cache'][part]['path']).open()))
     rows=[dict(fold=n,split=part,**r) for r in rows]
    else:rows=infer(n,configs['audio'],nets,part)
    parts[part]=check(rows,fold,part);write(d/f'{part}_probabilities.csv',rows)
    print('fold',n,part,'validated',len(rows),flush=True)
   torch.cuda.synchronize();timings.append(dict(fold=n,cache_reused=n==1,seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
   ids,a,v,y=parts['test'];mismatch=[]
   for mode,p in [('audio',a),('video',v)]:
    old={r['sample_id']:r for r in csv.DictReader((RUN/f'fold_{n:02d}'/mode/'test_predictions.csv').open())};assert set(old)==set(ids)
    for sid,pred,label in zip(ids,p.argmax(1),y):
     if int(old[sid]['prediction'])!=int(pred) or int(old[sid]['label'])!=int(label):mismatch.append(dict(model=mode,sample_id=sid,old=old[sid],prediction=int(pred),label=int(label)))
   dump(d/'reproduction.json',dict(mismatches=mismatch,checked_predictions=480));assert not mismatch,'unexplained reproduction difference'
   repro.append(dict(fold=n,mismatches=0))
   _,av,vv,yv=parts['val'];grid=[dict(alpha=i/10,**score(i/10*av+(1-i/10)*vv,yv)) for i in range(11)];chosen=select(grid)
   selection=dict(fold=n,selected_alpha=chosen['alpha'],validation_macro_f1=chosen['macro_f1'],grid=grid)
   dump(d/'alpha_selection.json',selection);selections.append(selection) # persisted BEFORE any fusion test scoring
   for name,alpha in [('Equal-weight Late Fusion',.5),('Validation-weighted Late Fusion',chosen['alpha'])]:
    probs=alpha*a+(1-alpha)*v;pred=probs.argmax(1);allrows.append(dict(fold=n,method=name,**score(probs,y)))
    tag='equal' if alpha==.5 and name.startswith('Equal') else 'validation_weighted'
    write(d/f'{tag}_test_predictions.csv',[dict(fold=n,split='test',sample_id=sid,label=int(label),actor=int(sid.split('-')[-1]),prediction=int(pr),probabilities=json.dumps(q.tolist())) for sid,label,pr,q in zip(ids,y,pred,probs)])
    counts=torch.bincount(y*8+pred,minlength=64).reshape(8,8).tolist();dump(d/f'{tag}_confusion_matrix.json',dict(labels=list(range(8)),rows='ground truth',columns='prediction',counts=counts))
   for mode,name in [('audio','Audio-only'),('video','Video-only'),('gated','Gated'),('xattn','XAttn'),('chumachenko_ia','Adapted IA')]:
    m=read(RUN/f'fold_{n:02d}'/mode/'metrics.json');allrows.append(dict(fold=n,method=name,**{k:m['test'][k] for k in K}))
   del nets;torch.cuda.empty_cache()
  write(HERE/'per_fold_metrics.csv',allrows);dump(HERE/'alpha_selection.json',selections);dump(HERE/'runtime.json',timings)
  aggregate=[]
  for method in dict.fromkeys(r['method'] for r in allrows):
   rs=[r for r in allrows if r['method']==method];assert len(rs)==6
   aggregate.append(dict(method=method,**{f'{k}_{stat}':fn([r[k] for r in rs]) for k in K for stat,fn in [('mean',statistics.mean),('std',statistics.stdev)]}))
  write(HERE/'aggregate.csv',aggregate)
  differences=[]
  for n in range(1,7):
   lookup={r['method']:r for r in allrows if r['fold']==n}
   for late in ('Equal-weight Late Fusion','Validation-weighted Late Fusion'):
    for base in ('Audio-only','Gated','XAttn'):
     differences.append(dict(fold=n,method=late,baseline=base,**{k+'_pp':100*(lookup[late][k]-lookup[base][k]) for k in ('accuracy','macro_f1')}))
  write(HERE/'paired_differences.csv',differences)
  counts=[]
  for late in ('Equal-weight Late Fusion','Validation-weighted Late Fusion'):
   for base in ('Audio-only','Gated','XAttn'):
    rs=[r for r in differences if r['method']==late and r['baseline']==base]
    for k in ('accuracy','macro_f1'):
     vals=[r[k+'_pp'] for r in rs];counts.append(dict(method=late,baseline=base,metric=k,improved=sum(x>1e-6 for x in vals),tied=sum(abs(x)<=1e-6 for x in vals),degraded=sum(x< -1e-6 for x in vals)))
  write(HERE/'improvement_counts.csv',counts);dump(HERE/'completion.json',dict(status='COMPLETE',folds=repro))
 finally:
  after=snapshot();dump(HERE/'historical_after.json',after);assert before==after,'Historical inputs changed'
  dump(HERE/'immutability.json',dict(unchanged=True,files=len(before)))
if __name__=='__main__':main()
