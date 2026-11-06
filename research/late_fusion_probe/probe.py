"""Read-only historical-artifact audit; writes only beside this script. No downloads."""
import os
os.environ.update(HF_HUB_OFFLINE='1', HF_LOCAL_ONLY='1')
import csv, inspect, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import torch
from train import build_model
from revision.audit import audit_dataset, require_complete
from revision.protocol import fixed_folds
from revision.artifacts import validate_local_artifacts
from data.ravdess import PAIR_SERVICE
OUT = Path(__file__).parent
RUN = ROOT/'outputs/speaker_independent_tracked_v1'
def read(p): return json.loads(p.read_text())
def model(c):
    kw = {k:v for k,v in c.items() if k in inspect.signature(build_model).parameters}
    kw.update(pretrained_video=False, checkpoint_init=True)
    return build_model(**kw)
def main():
    torch.set_num_threads(4)
    audit = audit_dataset(ROOT/'data')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    pairs = PAIR_SERVICE.build_pairs(ROOT/'data')
    rows=[]; splits=[]
    for s in fixed_folds():
        counts={}
        for part in ('train','val','test'):
            selected=[p for p in pairs if p.actor in s[part+'_actors']]
            keys={(p.emotion,p.intensity,p.statement,p.repetition,p.actor) for p in selected}
            counts[part]={'count':len(selected),'unique':len(keys)}
        splits.append(dict(split=s,counts=counts))
        for mode in ('audio','video'):
            d=RUN/f"fold_{s['fold']:02d}"/mode; p=d/'best.pt'
            r=dict(fold=s['fold'],model=mode,path=str(p),bytes=p.stat().st_size if p.exists() else 0,config=str(d/'config.json'),best_epoch='',load='NOT_RUN',consistency='NOT_RUN',error='')
            try:
                c=read(d/'config.json'); m=read(d/'metrics.json')
                ck=torch.load(p,map_location='cpu',weights_only=True)
                r['best_epoch']=ck['epoch']
                net=model(c); net.load_state_dict(ck['model'],strict=True); del net
                r['load']='PASS (strict keys/shapes)'
                assert ck['config']==c, 'checkpoint/config mismatch'
                assert c['revision']['split']==s, 'fold mismatch'
                assert ck['epoch']==m['best_epoch'] and ck['val_f1']==m['val_macro_f1'], 'best epoch mismatch'
                validate_local_artifacts(d,c,m)
                require_complete(audit)
                assert audit['fingerprint']==c['revision']['dataset_fingerprint'], 'dataset fingerprint mismatch'
                r['consistency']='PASS'
                del ck
            except Exception as e: r['error']=str(e)
            rows.append(r); print(r,flush=True)
    with (OUT/'CHECKPOINT_INVENTORY.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (OUT/'split_audit.json').write_text(json.dumps(splits,indent=2))
    print('ALL_PASS',all(r['consistency']=='PASS' for r in rows))
if __name__=='__main__': main()
