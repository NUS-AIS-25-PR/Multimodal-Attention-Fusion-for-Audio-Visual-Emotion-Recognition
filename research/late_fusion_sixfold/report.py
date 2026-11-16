"""Render completed immutable experiment tables without inference."""
import csv,json,hashlib,importlib.metadata,subprocess
from pathlib import Path
P=Path(__file__).parent
def read(n):return json.loads((P/n).read_text())
def rows(n):return list(csv.DictReader((P/n).open()))
def table(rs):
 keys=list(rs[0]);return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+['| '+' | '.join(str(r[k]) for k in keys)+' |' for r in rs])
assert read('completion.json')['status']=='COMPLETE' and read('immutability.json')['unchanged']
a=rows('aggregate.csv');diff=rows('paired_differences.csv');runtime=read('runtime.json')
t=[dict(method=r['method'],**{k:f"{100*float(r[k+'_mean']):.2f} ± {100*float(r[k+'_std']):.2f}" for k in ('accuracy','precision','recall','macro_f1')}) for r in a]
d=[{k:(f'{float(v):+.2f}' if k.endswith('_pp') else v) for k,v in r.items()} for r in diff]
s=read('alpha_selection.json')
text='''# Six-fold Late Fusion — COMPLETE

六折全部通过逐 sample_id 复现检查：12 个单模态测试集合，各 240 个预测，零差异。Fold 1 复用已核验缓存；Fold 2–6 实际 GPU 推理。每折 val/test 各 240 个唯一 ID，标签、演员、八类顺序、有限归一化概率及 logits-softmax 一致性均通过。两模态按 sample_id 建索引对齐。

## 七方法比较

单位为百分数，算术均值 ± 样本标准差（ddof=1），固定八类别、zero_division=0。没有用合并预测指标替代六折均值。前五种方法直接引用 outputs/speaker_independent_tracked_v1/fold_XX/{audio,video,gated,xattn,chumachenko_ia}/metrics.json；未重新评估特征融合模型。

'''+table(t)+'\n\n## 验证集调权\n\n'+table([dict(fold=r['fold'],alpha=r['selected_alpha'],validation_macro_f1=f"{r['validation_macro_f1']:.6f}") for r in s])+'''

网格固定 0.0–1.0、步长 0.1；最大验证 Macro-F1，同分选最接近 0.5，再较小权重。每折完整网格和 alpha 在计算融合测试指标前独立保存到 fold_XX/alpha_selection.json。不存在温度缩放或概率校准声明。alpha_selection.json 为汇总副本。

## 逐折差值

正值表示 Late Fusion 高于比较方法，单位百分点。

'''+table(d)+'\n\n## 改善/持平/退化\n\n'+table(rows('improvement_counts.csv'))+'''

## 执行与来源

运行命令：`.venv/bin/python research/late_fusion_sixfold/run.py`；报告：`.venv/bin/python research/late_fusion_sixfold/report.py`。

lightweight_tests.json 记录探针修复验证：从缓存运行原 summarize.main 两次，输出字节一致、恰好六行、仅四项指标；额外 loss/acc/f1 字段不污染 CSV。测试在临时目录进行，没有 GPU 推理，也没有改写原缓存。

experiment.json 保存 commit、分支、环境、源码/脚本指纹和完整数据审计。各折 sources.json 保存完整配置、checkpoint SHA256 和来源路径。val/test_probabilities.csv 保存 fold/split/sample_id/actor/label 和两模态八类 logits/probabilities；两个 *_test_predictions.csv 和 *_confusion_matrix.json 保存融合预测与固定八类计数矩阵。来源文件绝对路径、SHA256、mtime_ns、大小全部在 historical_before/after.json。

'''+f"历史目录 {read('immutability.json')['files']} 个文件执行前后 SHA256、大小和修改时间完全一致。src 未修改，无依赖变更、模型下载、训练或 PR。保留工作区原有无关修改。\n\n"+'''## 资源

'''+table(runtime)+f"\n\nFold 2–6 总推理及缓存校验/写入耗时 {sum(r['seconds'] for r in runtime if not r['cache_reused']):.2f} 秒；最大 PyTorch allocated 显存 {max(r['peak_allocated_bytes'] for r in runtime)/2**30:.2f} GiB，最大 reserved {max(r['peak_reserved_bytes'] for r in runtime)/2**30:.2f} GiB。计时不含模型加载、历史文件哈希和结果汇总；显存不包括驱动及其他进程。\n\n"+'''## 限制与研究解释

Fold 1 原探针缓存生成时未保存 checkpoint SHA256。本轮通过原探针代码、已记录配置/源码/数据指纹、样本完整性、logits-softmax 一致性和两单模态测试预测完全一致进行追溯核验，并记录当前 checkpoint 与缓存哈希；不能把当前哈希声称为生成时签名。Fold 1 不重复 GPU 推理。原数据指纹基于路径/大小/mtime，不是原始媒体内容哈希。

本轮无复现差异，无缺失折；所有折和两个版本均报告。不做显著性检验，不声称统计显著优越。研究者此前已看过旧测试结果及 Fold 1 新结果，因此这是既有研究上的探索性补充比较，不是全新独立确认性实验。特征级模型包含额外联合训练，而本轮仅组合已有单模态预测；比较应描述这一训练成本差异，不推断任意网络或数据集上的普遍结论。
'''
(P/'SIX_FOLD_REPORT.md').write_text(text)
versions={k:importlib.metadata.version(k) for k in ('torch','torchvision','transformers','numpy','scikit-learn','mediapipe','opencv-python','librosa')}
(P/'current_versions.json').write_text(json.dumps(versions,indent=2))
print(table(t));print('alphas',[r['selected_alpha'] for r in s]);print(table(rows('improvement_counts.csv')))
