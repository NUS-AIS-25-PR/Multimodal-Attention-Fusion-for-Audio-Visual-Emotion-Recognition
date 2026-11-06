from fold1 import *
def main():
    results={}
    for part in ('val','test'):
        rows=list(csv.DictReader((OUT/f'fold1_{part}_predictions.csv').open()))
        results[part]=(torch.tensor([json.loads(r['audio_probs']) for r in rows]),torch.tensor([json.loads(r['video_probs']) for r in rows]),torch.tensor([int(r['label']) for r in rows]),rows)
    a,v,y,rows=results['test']
    for mode,p in [('audio',a),('video',v)]:
        old={r['sample_id']:r for r in csv.DictReader((RUN/'fold_01'/mode/'test_predictions.csv').open())}
        assert len(old)==len(rows)==240
        assert all(int(old[r['sample_id']]['prediction'])==int(pred) and int(old[r['sample_id']]['label'])==int(r['label']) for r,pred in zip(rows,p.argmax(1)))
    av,vv,yv,_=results['val']
    grid=[dict(alpha=i/10,**scores(i/10*av+(1-i/10)*vv,yv)) for i in range(11)]
    chosen=sorted(grid,key=lambda r:(-r['macro_f1'],abs(round(r['alpha']-.5,1)),r['alpha']))[0]['alpha']
    (OUT/'validation_grid.json').write_text(json.dumps(dict(grid=grid,selected_alpha=chosen,tie_rule='closest to 0.5 then smaller'),indent=2))
    rr=[dict(method=name,alpha=alpha,**scores(alpha*a+(1-alpha)*v,y)) for name,alpha in [('fixed',.5),('calibrated',chosen)]]
    for mode in ('audio','video','gated','xattn'):
        rr.append(dict(method=mode+'_historical',alpha='',**{k:read(RUN/'fold_01'/mode/'metrics.json')['test'][k] for k in ('accuracy','precision','recall','macro_f1')}))
    with (OUT/'FOLD1_PROBE_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
    print(rr)
if __name__=='__main__': main()
