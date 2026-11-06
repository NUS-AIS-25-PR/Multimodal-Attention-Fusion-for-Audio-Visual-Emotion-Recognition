"""Fold 1 only: frozen grid, validation-only selection, test reproduction gate."""
from probe import *
import time
from torch.utils.data import DataLoader
from data.ravdess import DATASET_FACTORY
from utils.metrics import classification_metrics

def scores(p,y): return classification_metrics(p.argmax(1), y, 8)
def main():
    inventory=list(csv.DictReader((OUT/'CHECKPOINT_INVENTORY.csv').open()))
    assert len(inventory)==12 and all(r['consistency']=='PASS' for r in inventory)
    torch.set_num_threads(4); torch.manual_seed(42)
    t=time.monotonic(); nets={}; configs={}
    for mode in ('audio','video'):
        d=RUN/'fold_01'/mode; c=read(d/'config.json'); configs[mode]=c
        net=model(c); net.load_state_dict(torch.load(d/'best.pt',map_location='cpu',weights_only=True)['model']); nets[mode]=net.cuda().eval()
    for k in ('frames','use_face_crop','use_wavlm'):
        assert configs['audio'][k]==configs['video'][k]
    c=configs['audio']; pairs=PAIR_SERVICE.build_pairs(ROOT/'data'); results={}
    with torch.inference_mode():
        for part in ('val','test'):
            pp=[p for p in pairs if p.actor in c['revision']['split'][part+'_actors']]
            ds=DATASET_FACTORY.create(pp,num_classes=8,num_frames=c['frames'],augment=False,use_face_crop=c['use_face_crop'],use_wavlm=c['use_wavlm'])
            rows=[]; al=[]; vl=[]; yl=[]
            for v,a,y,meta in DataLoader(ds,batch_size=16,shuffle=False,num_workers=2):
                aa=nets['audio'](a.cuda()).cpu(); vv=nets['video'](v.cuda()).cpu()
                assert torch.isfinite(aa).all() and torch.isfinite(vv).all()
                ap=aa.softmax(1); vp=vv.softmax(1)
                for i in range(len(y)):
                    sid='-'.join(f'{int(x):02d}' for x in [1,meta['emotion'][i],meta['intensity'][i],meta['statement'][i],meta['repetition'][i],meta['actor'][i]])
                    rows.append(dict(sample_id=sid,actor=int(meta['actor'][i]),label=int(y[i]),audio_logits=json.dumps(aa[i].tolist()),video_logits=json.dumps(vv[i].tolist()),audio_probs=json.dumps(ap[i].tolist()),video_probs=json.dumps(vp[i].tolist())))
                al.append(ap); vl.append(vp); yl.append(y)
                print(part,len(rows),flush=True)
            with (OUT/f'fold1_{part}_predictions.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            results[part]=(torch.cat(al),torch.cat(vl),torch.cat(yl),rows)
    a,v,y,rows=results['test']
    for mode,p in [('audio',a),('video',v)]:
        old={r['sample_id']:r for r in csv.DictReader((RUN/'fold_01'/mode/'test_predictions.csv').open())}
        assert len(old)==len(rows)==240
        mismatches=sum(int(old[r['sample_id']]['prediction'])!=int(pred) or int(old[r['sample_id']]['label'])!=r['label'] for r,pred in zip(rows,p.argmax(1)))
        assert mismatches==0, f'{mode}: {mismatches} test mismatches; fusion comparison stopped'
    av,vv,yv,_=results['val']
    grid=[dict(alpha=i/10,**scores(i/10*av+(1-i/10)*vv,yv)) for i in range(11)]
    chosen=sorted(grid,key=lambda r:(-r['macro_f1'],abs(round(r['alpha']-.5,1)),r['alpha']))[0]['alpha']
    (OUT/'validation_grid.json').write_text(json.dumps(dict(grid=grid,selected_alpha=chosen,tie_rule='closest to 0.5, then smaller alpha'),indent=2))
    rr=[]
    for name,alpha in [('fixed',.5),('calibrated',chosen)]: rr.append(dict(method=name,alpha=alpha,**scores(alpha*a+(1-alpha)*v,y)))
    for mode in ('audio','video','gated','xattn'):
        rr.append(dict(method=mode+'_historical',alpha='',**{k:read(RUN/'fold_01'/mode/'metrics.json')['test'][k] for k in ('accuracy','precision','recall','macro_f1')}))
    with (OUT/'FOLD1_PROBE_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
    (OUT/'runtime.json').write_text(json.dumps(dict(seconds=time.monotonic()-t,peak_cuda_bytes=torch.cuda.max_memory_allocated(),test_prediction_mismatches=0),indent=2))
    print(rr,flush=True)
if __name__=='__main__': main()
