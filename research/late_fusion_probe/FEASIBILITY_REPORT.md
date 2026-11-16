# Late Fusion Feasibility Probe — PASS

2026-11-06，本机运行验证通过。未训练、下载或替换编码器；未修改既有源码、checkpoint、数据或历史结果。所有新材料在本目录。

## 仓库与环境

实际路径 /home/louis/projects/MultimodalEmotionRecognition；基线 main，commit 0ae967afc654d3234620862437d63f2f33ab57e7；工作分支 codex/late-fusion-probe。原有 .gitignore 修改及三个未跟踪文件保留。未提交或创建 PR。

Python 3.10.19、PyTorch 2.10.0+cu130、CUDA 13.0；RTX 5080 Laptop GPU，16GB。CUDA 实际推理成功。详细版本和源码指纹见 environment.json。

指定 results/ieee_spmb_2026/tracked_v1/ 本地不存在。实际证据在 outputs/speaker_independent_tracked_v1/：六折 config.json、split.json、best.pt、history.csv、metrics.json、test_predictions.csv、confusion_matrix.json、tracking.json 及聚合表。历史 test_predictions 只有 sample_id/actor/label/prediction，没有 logits/probabilities，也未找到该运行保存的验证概率。

## 静态发现与运行证据

- fusion.py 的 mode='late' 返回两个 softmax 的等权平均，不是 logits。可直接使用独立 WavLMAudioEncoder 和 VideoNet；本次没有构建或训练新的融合网络。
- train.py 保存验证 macro-F1 首次最优 epoch，并在最终测试前恢复 best.pt。eval.py 已支持 WavLM 原始波形及 checkpoint_init；本次禁用预训练下载、strict=True 恢复完整权重。
- 12/12 checkpoint 实际加载通过，键和形状完全兼容。音频每个 380005889 bytes，视频每个 44801803 bytes；绝对路径、配置路径、epoch 见 CHECKPOINT_INVENTORY.csv。
- checkpoint config 与本地 config 完全相等；最佳 epoch/验证分数与 metrics/history 一致；历史预测、混淆矩阵和指标通过仓库校验器。全部源码指纹及数据指纹与训练记录一致。没有混用其他目录或旧 random-split checkpoint。
- 数据完整、演员互斥、各分区样本数见 DATA_SPLIT_AUDIT.md。文件指纹不是媒体内容哈希；Fold 2–6 未做完整解码推理。
- Fold 1 实际对 val/test 各 240 样本导出两模态八类 logits 和 softmax，见 fold1_*_predictions.csv。两个单模态测试各 240 条预测与历史逐 sample_id 完全一致，标签一致；不是仅比较四舍五入指标。

## Fold 1 融合

预设 alpha={0.0,0.1,...,1.0}，只用验证 macro-F1 选择；同分先最接近 0.5，再较小 alpha。选择 alpha=0.7。未调整温度、预处理、网格或 checkpoint。完整验证得分在 validation_grid.json，测试四项指标和历史描述性对比在 FOLD1_PROBE_RESULTS.csv。

A 等权：Accuracy 0.795833，Macro-Precision 0.811610，Macro-Recall 0.777344，Macro-F1 0.784466。
B 验证校准：Accuracy 0.825000，Macro-Precision 0.831337，Macro-Recall 0.828125，Macro-F1 0.822874。

## 实际执行与限制

运行命令（仓库根目录）：

```bash
.venv/bin/python research/late_fusion_probe/probe.py
.venv/bin/python research/late_fusion_probe/provenance.py
.venv/bin/python research/late_fusion_probe/fold1.py
.venv/bin/python research/late_fusion_probe/summarize.py
```

前两项通过。fold1.py 实际完成全部推理与预测复现检查后，汇总时因 f1/macro_f1 字段名错误退出；summarize.py 首次又遇历史 metrics 含额外 loss 字段的 CSV schema 错误。两项均为探针汇总实现错误，已修复。最终 summarize.py 从已保存概率重算并成功输出全部比较，未重复 GPU 推理。保留首次 fold1_run.log。修复后的完整 fold1.py 未从头重跑；本次完整证据来自首次推理缓存加最终汇总。精确耗时和峰值显存未保存，资源计划仅估算。

结论：值得作为探索性补充 baseline 加入论文，但先完成六折复现及固定规则评估，不能把 Fold 1 当作最终论文结论。研究者已看过旧测试结果，不能声称完全未受研究决策影响的独立确认实验。六折无需重新训练的技术路径已成立，其他五折的实际前向复现仍须运行；详见 SIX_FOLD_EXECUTION_PLAN.md。
