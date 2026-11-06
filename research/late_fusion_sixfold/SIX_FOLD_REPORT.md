# Six-fold Late Fusion — COMPLETE

六折全部通过逐 sample_id 复现检查：12 个单模态测试集合，各 240 个预测，零差异。Fold 1 复用已核验缓存；Fold 2–6 实际 GPU 推理。每折 val/test 各 240 个唯一 ID，标签、演员、八类顺序、有限归一化概率及 logits-softmax 一致性均通过。两模态按 sample_id 建索引对齐。

## 七方法比较

单位为百分数，算术均值 ± 样本标准差（ddof=1），固定八类别、zero_division=0。没有用合并预测指标替代六折均值。前五种方法直接引用 outputs/speaker_independent_tracked_v1/fold_XX/{audio,video,gated,xattn,chumachenko_ia}/metrics.json；未重新评估特征融合模型。

| method | accuracy | precision | recall | macro_f1 |
|---|---|---|---|---|
| Equal-weight Late Fusion | 71.53 ± 7.09 | 73.75 ± 6.26 | 71.16 ± 5.79 | 70.72 ± 6.37 |
| Validation-weighted Late Fusion | 73.40 ± 7.19 | 75.03 ± 6.87 | 73.44 ± 6.72 | 72.71 ± 7.03 |
| Audio-only | 65.90 ± 9.52 | 69.99 ± 6.51 | 65.89 ± 9.31 | 64.58 ± 9.96 |
| Video-only | 56.04 ± 7.31 | 56.48 ± 8.85 | 54.75 ± 7.17 | 53.22 ± 8.64 |
| Gated | 70.28 ± 7.18 | 75.66 ± 4.84 | 69.47 ± 6.61 | 68.67 ± 7.42 |
| XAttn | 68.75 ± 6.45 | 72.53 ± 6.75 | 68.03 ± 5.94 | 67.23 ± 6.20 |
| Adapted IA | 56.25 ± 3.96 | 60.59 ± 4.91 | 55.34 ± 3.40 | 53.95 ± 4.19 |

## 验证集调权

| fold | alpha | validation_macro_f1 |
|---|---|---|
| 1 | 0.7 | 0.824511 |
| 2 | 0.6 | 0.767919 |
| 3 | 0.5 | 0.720056 |
| 4 | 0.6 | 0.692269 |
| 5 | 0.7 | 0.746311 |
| 6 | 0.6 | 0.821010 |

网格固定 0.0–1.0、步长 0.1；最大验证 Macro-F1，同分选最接近 0.5，再较小权重。每折完整网格和 alpha 在计算融合测试指标前独立保存到 fold_XX/alpha_selection.json。不存在温度缩放或概率校准声明。alpha_selection.json 为汇总副本。

## 逐折差值

正值表示 Late Fusion 高于比较方法，单位百分点。

| fold | method | baseline | accuracy_pp | macro_f1_pp |
|---|---|---|---|---|
| 1 | Equal-weight Late Fusion | Audio-only | +2.50 | +1.94 |
| 1 | Equal-weight Late Fusion | Gated | +13.33 | +17.05 |
| 1 | Equal-weight Late Fusion | XAttn | +8.33 | +11.48 |
| 1 | Validation-weighted Late Fusion | Audio-only | +5.42 | +5.78 |
| 1 | Validation-weighted Late Fusion | Gated | +16.25 | +20.90 |
| 1 | Validation-weighted Late Fusion | XAttn | +11.25 | +15.32 |
| 2 | Equal-weight Late Fusion | Audio-only | +5.00 | +4.62 |
| 2 | Equal-weight Late Fusion | Gated | -0.42 | +0.11 |
| 2 | Equal-weight Late Fusion | XAttn | +7.92 | +8.24 |
| 2 | Validation-weighted Late Fusion | Audio-only | +6.67 | +6.75 |
| 2 | Validation-weighted Late Fusion | Gated | +1.25 | +2.24 |
| 2 | Validation-weighted Late Fusion | XAttn | +9.58 | +10.38 |
| 3 | Equal-weight Late Fusion | Audio-only | +12.92 | +12.48 |
| 3 | Equal-weight Late Fusion | Gated | -6.67 | -7.67 |
| 3 | Equal-weight Late Fusion | XAttn | -5.00 | -5.55 |
| 3 | Validation-weighted Late Fusion | Audio-only | +12.92 | +12.48 |
| 3 | Validation-weighted Late Fusion | Gated | -6.67 | -7.67 |
| 3 | Validation-weighted Late Fusion | XAttn | -5.00 | -5.55 |
| 4 | Equal-weight Late Fusion | Audio-only | +15.42 | +17.84 |
| 4 | Equal-weight Late Fusion | Gated | +1.67 | +2.20 |
| 4 | Equal-weight Late Fusion | XAttn | -2.50 | -2.48 |
| 4 | Validation-weighted Late Fusion | Audio-only | +12.92 | +15.53 |
| 4 | Validation-weighted Late Fusion | Gated | -0.83 | -0.11 |
| 4 | Validation-weighted Late Fusion | XAttn | -5.00 | -4.79 |
| 5 | Equal-weight Late Fusion | Audio-only | +0.00 | +1.29 |
| 5 | Equal-weight Late Fusion | Gated | -2.08 | -1.72 |
| 5 | Equal-weight Late Fusion | XAttn | +2.08 | +2.68 |
| 5 | Validation-weighted Late Fusion | Audio-only | +3.75 | +4.34 |
| 5 | Validation-weighted Late Fusion | Gated | +1.67 | +1.32 |
| 5 | Validation-weighted Late Fusion | XAttn | +5.83 | +5.72 |
| 6 | Equal-weight Late Fusion | Audio-only | -2.08 | -1.28 |
| 6 | Equal-weight Late Fusion | Gated | +1.67 | +2.34 |
| 6 | Equal-weight Late Fusion | XAttn | +5.83 | +6.60 |
| 6 | Validation-weighted Late Fusion | Audio-only | +3.33 | +3.92 |
| 6 | Validation-weighted Late Fusion | Gated | +7.08 | +7.54 |
| 6 | Validation-weighted Late Fusion | XAttn | +11.25 | +11.80 |

## 改善/持平/退化

| method | baseline | metric | improved | tied | degraded |
|---|---|---|---|---|---|
| Equal-weight Late Fusion | Audio-only | accuracy | 4 | 1 | 1 |
| Equal-weight Late Fusion | Audio-only | macro_f1 | 5 | 0 | 1 |
| Equal-weight Late Fusion | Gated | accuracy | 3 | 0 | 3 |
| Equal-weight Late Fusion | Gated | macro_f1 | 4 | 0 | 2 |
| Equal-weight Late Fusion | XAttn | accuracy | 4 | 0 | 2 |
| Equal-weight Late Fusion | XAttn | macro_f1 | 4 | 0 | 2 |
| Validation-weighted Late Fusion | Audio-only | accuracy | 6 | 0 | 0 |
| Validation-weighted Late Fusion | Audio-only | macro_f1 | 6 | 0 | 0 |
| Validation-weighted Late Fusion | Gated | accuracy | 4 | 0 | 2 |
| Validation-weighted Late Fusion | Gated | macro_f1 | 4 | 0 | 2 |
| Validation-weighted Late Fusion | XAttn | accuracy | 4 | 0 | 2 |
| Validation-weighted Late Fusion | XAttn | macro_f1 | 4 | 0 | 2 |

## 执行与来源

运行命令：`.venv/bin/python research/late_fusion_sixfold/run.py`；报告：`.venv/bin/python research/late_fusion_sixfold/report.py`。

lightweight_tests.json 记录探针修复验证：从缓存运行原 summarize.main 两次，输出字节一致、恰好六行、仅四项指标；额外 loss/acc/f1 字段不污染 CSV。测试在临时目录进行，没有 GPU 推理，也没有改写原缓存。

experiment.json 保存 commit、分支、环境、源码/脚本指纹和完整数据审计。各折 sources.json 保存完整配置、checkpoint SHA256 和来源路径。val/test_probabilities.csv 保存 fold/split/sample_id/actor/label 和两模态八类 logits/probabilities；两个 *_test_predictions.csv 和 *_confusion_matrix.json 保存融合预测与固定八类计数矩阵。来源文件绝对路径、SHA256、mtime_ns、大小全部在 historical_before/after.json。

历史目录 463 个文件执行前后 SHA256、大小和修改时间完全一致。src 未修改，无依赖变更、模型下载、训练或 PR。保留工作区原有无关修改。

## 资源

| fold | cache_reused | seconds | peak_allocated_bytes | peak_reserved_bytes |
|---|---|---|---|---|
| 1 | True | 0.39303226199990604 | 0 | 0 |
| 2 | False | 51.42629775600017 | 1067800064 | 1432354816 |
| 3 | False | 47.556477319999885 | 1086546944 | 1117782016 |
| 4 | False | 49.13805480000019 | 1105424896 | 1138753536 |
| 5 | False | 45.75784607200012 | 1124302848 | 1159725056 |
| 6 | False | 43.22711073399978 | 1143180800 | 1180696576 |

Fold 2–6 总推理及缓存校验/写入耗时 237.11 秒；最大 PyTorch allocated 显存 1.06 GiB，最大 reserved 1.33 GiB。计时不含模型加载、历史文件哈希和结果汇总；显存不包括驱动及其他进程。

## 限制与研究解释

Fold 1 原探针缓存生成时未保存 checkpoint SHA256。本轮通过原探针代码、已记录配置/源码/数据指纹、样本完整性、logits-softmax 一致性和两单模态测试预测完全一致进行追溯核验，并记录当前 checkpoint 与缓存哈希；不能把当前哈希声称为生成时签名。Fold 1 不重复 GPU 推理。原数据指纹基于路径/大小/mtime，不是原始媒体内容哈希。

本轮无复现差异，无缺失折；所有折和两个版本均报告。不做显著性检验，不声称统计显著优越。研究者此前已看过旧测试结果及 Fold 1 新结果，因此这是既有研究上的探索性补充比较，不是全新独立确认性实验。特征级模型包含额外联合训练，而本轮仅组合已有单模态预测；比较应描述这一训练成本差异，不推断任意网络或数据集上的普遍结论。

## 主线结论与最终验证

当前结果不支持额外联合训练的特征级融合带来稳定收益。验证集调权 Late Fusion 的平均 Macro-F1 高于 Audio 8.13、Gated 4.04、XAttn 5.47 个百分点；对 Audio 六折全部改善，对 Gated/XAttn 各四折改善、两折退化。因此均值优势不等于逐折一致优势。

独立 verify.py 检查通过：42 个方法×折记录、12 组融合预测和混淆矩阵、七组均值及 ddof=1 标准差。脚本编译通过。git diff --check 仅发现原有 .gitignore 尾随空白，未修改该文件。
