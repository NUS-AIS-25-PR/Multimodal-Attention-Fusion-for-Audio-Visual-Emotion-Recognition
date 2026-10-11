"""Derive six-fold reports from saved artifacts; no inference, training or artifact edits."""
from pathlib import Path
import csv
import json
import sys
from revision.aggregate import aggregate

ROOT=Path('/home/louis/projects/MultimodalEmotionRecognition/outputs/speaker_independent_tracked_v1')
CONTROL=Path(sys.argv[1])
METHODS=('audio','video','gated','chumachenko_ia','xattn')
METRICS=('accuracy','precision','recall','macro_f1')
agg=aggregate(ROOT)
assert set(agg['methods'])==set(METHODS), 'Unexpected/missing methods'
assert all(agg['methods'][m]['complete_six_folds'] for m in METHODS), 'Cannot finalize incomplete coverage'
state=json.loads((CONTROL/'status.json').read_text())
old=json.loads((ROOT.parent/'fold1_tracked_execution/status.json').read_text())
rows=[]
for model in METHODS:
 for fold in range(1,7):
  path=ROOT/f'fold_{fold:02d}'/model
  record=json.loads((path/'metrics.json').read_text())
  track=json.loads((path/'tracking.json').read_text())
  assert (track['mode'],track['project'],track['group'])==('offline','ieee-spmb-2026','spmb2026-canonical-tracked-v1')
  run=(old['models'][model] if fold==1 else state['models'][f'fold_{fold:02d}/{model}'])
  rows.append({'method':model,'fold':fold,'best_epoch':record['best_epoch'],'val_macro_f1':record['val_macro_f1'],
    'epochs_completed':run['epochs'],'early_stopping':run.get('early_stopping',False),
    **{key:record['test'][key] for key in METRICS},'observed_runtime_seconds':run['runtime_seconds'],
    'wandb_status':track['status'],'metrics_path':str(path/'metrics.json')})
with (ROOT/'per_fold_metrics.csv').open('x',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (ROOT/'mean_std_table.csv').open('x',newline='') as f:
 writer=csv.writer(f);writer.writerow(['method',*(key+' (%, mean ± sample SD)' for key in METRICS)])
 for model in METHODS:
  summary=agg['methods'][model]['summary']
  writer.writerow([model,*[f"{100*summary[key]['mean']:.2f} ± {100*summary[key]['std']:.2f}" for key in METRICS]])
prior_seconds=sum(r['runtime_seconds'] for r in old['commands'])
new_seconds=sum(r['runtime_seconds'] for r in state['commands'])
report={'complete':True,'git_commit':agg['experiment']['git_commit'],'source_fingerprint':agg['experiment']['source_fingerprint'],
 'output_root':str(ROOT),'methods':agg['methods'],'per_fold':rows,'failed_or_partial_folds':[],'historical_failures':state.get('historical_failures',[]),
 'total_training_command_runtime_seconds':prior_seconds+new_seconds,
 'fold1_training_command_runtime_seconds':prior_seconds,'continuation_training_command_runtime_seconds':new_seconds,
 'runtime_note':'Sum of observed process wall runtimes for original Fold 1 and continuation training commands, including validation/test/loading, resume checks, and the preserved failed initialization attempt; excludes idle gap between sessions and final plotting.',
 'tracking':{'mode':'offline','project':'ieee-spmb-2026','group':'spmb2026-canonical-tracked-v1'},
 'aggregate_json':str(ROOT/'aggregate.json'),'aggregate_csv':str(ROOT/'aggregate.csv'),
 'per_fold_csv':str(ROOT/'per_fold_metrics.csv'),'mean_std_table_csv':str(ROOT/'mean_std_table.csv'),
 'final_figure_directory':str(ROOT.parent/'paper_figures_six_fold_tracked_v1')}
(ROOT/'six_fold_report.json').write_text(json.dumps(report,indent=2))
print('Validated 30 complete runs; saved final per-fold and mean ± sample SD reports.')
for model in METHODS:
 print(model,agg['methods'][model]['summary'])
