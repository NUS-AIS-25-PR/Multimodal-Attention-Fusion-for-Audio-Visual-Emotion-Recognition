"""Independent artifact verification using sklearn, no GPU inference."""
import csv,json,statistics
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
P=Path(__file__).parent
rows=list(csv.DictReader((P/'per_fold_metrics.csv').open()));assert len(rows)==42
for n in range(1,7):
 d=P/f'fold_{n:02d}'
 assert not json.loads((d/'reproduction.json').read_text())['mismatches']
 for tag,name in [('equal','Equal-weight Late Fusion'),('validation_weighted','Validation-weighted Late Fusion')]:
  rr=list(csv.DictReader((d/f'{tag}_test_predictions.csv').open()));assert len(rr)==len({r['sample_id'] for r in rr})==240
  y=[int(r['label']) for r in rr];p=[int(r['prediction']) for r in rr]
  pr,re,f,_=precision_recall_fscore_support(y,p,labels=range(8),average='macro',zero_division=0)
  expected=dict(accuracy=accuracy_score(y,p),precision=pr,recall=re,macro_f1=f)
  actual=next(r for r in rows if int(r['fold'])==n and r['method']==name)
  assert all(abs(float(actual[k])-v)<1e-7 for k,v in expected.items())
  assert confusion_matrix(y,p,labels=range(8)).tolist()==json.loads((d/f'{tag}_confusion_matrix.json').read_text())['counts']
for r in csv.DictReader((P/'aggregate.csv').open()):
 for k in ('accuracy','precision','recall','macro_f1'):
  vals=[float(x[k]) for x in rows if x['method']==r['method']];assert len(vals)==6
  assert abs(float(r[k+'_mean'])-np.mean(vals))<1e-12
  assert abs(float(r[k+'_std'])-np.std(vals,ddof=1))<1e-12
print('PASS: 42 method-fold rows, 12 fusion prediction sets/confusions, seven mean/sample-SD aggregates')
