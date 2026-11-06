from probe import *
import hashlib, importlib.metadata, platform
h=hashlib.sha256()
for p in sorted((ROOT/'src').rglob('*.py')):
    h.update(str(p.relative_to(ROOT/'src')).encode());h.update(p.read_bytes())
for f in range(1,7):
    for mode in ('audio','video'):
        c=read(RUN/f'fold_{f:02d}'/mode/'config.json')
        assert c['revision']['source_fingerprint']==h.hexdigest()
        s=c['revision']['split']; a,b,d=[set(s[k+'_actors']) for k in ('train','val','test')]
        assert not(a&b or a&d or b&d) and a|b|d==set(range(1,25))
        assert all(len(x)==n for x,n in zip((a,b,d),(16,4,4)))
versions={k:importlib.metadata.version(k) for k in ('torch','torchvision','transformers','numpy','scikit-learn','mediapipe','opencv-python','librosa')}
(OUT/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),versions=versions,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),source_fingerprint=h.hexdigest(),all_source_fingerprints_match=True),indent=2))
print('PASS source and disjointness')
