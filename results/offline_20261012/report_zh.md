# LLaVA-Video / NExT-QA 视觉依赖与特征容错实测报告

本报告针对固定60视频、495题子集，48组条件，共23,760个固定答题槽位。模型为LLaVA-Video-7B-Qwen2，语言模型线性层NF4双量化、BF16计算。缺失、运行错误和无效格式均计错；不替换失败答案。

**本轮为离线特征擦除实验，尚不构成原版Testbed的RTC/netem网络实测。**

## 主要对照

| 条件 | 答对/495 | 准确率 | 95%视频分组区间 | 不可用 | 格式无效 | 随机等效判据 |
|---|---:|---:|---:|---:|---:|---|
| text_only | 239/495 | 48.28% | [43.62, 53.03]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| text_timing_16 | 237/495 | 47.88% | [43.48, 52.44]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| native_center_1 | 350/495 | 70.71% | [66.34, 75.10]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| native_center_4 | 355/495 | 71.72% | [67.78, 75.59]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| native_center_16 | 361/495 | 72.93% | [69.20, 76.72]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| native_center_64 | 366/495 | 73.94% | [70.02, 77.80]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| uniform_4 | 369/495 | 74.55% | [70.12, 78.90]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| uniform_16 | 386/495 | 77.98% | [73.64, 82.16]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| uniform_64 | 400/495 | 80.81% | [76.92, 84.52]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| repeat_16 | 354/495 | 71.52% | [67.20, 75.80]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| shuffle_16 | 383/495 | 77.37% | [73.65, 81.10]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| reverse_16 | 382/495 | 77.17% | [72.93, 81.20]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| wrong_video_16 | 192/495 | 38.79% | [34.30, 43.40]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| gaussian_16 | 224/495 | 45.25% | [40.73, 49.90]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |
| zero_16 | 225/495 | 45.45% | [40.92, 50.11]% | 0 | 0 | outside_prespecified_5pp_equivalence_margin |

![基础对照](figures/baseline_controls.png)

## 连续帧与时间覆盖

原生连续帧不抽样，固定取视频中心窗口；等间隔帧覆盖整个视频。本子集连续4帧时间跨度中位数0.1001秒，连续64帧2.1021秒，等间隔64帧33.6903秒。等间隔4/16/64帧覆盖相同起止时刻，帧数增加主要改变时间采样密度。

![帧依赖](figures/frame_dependence.png)

| 成对差异（左减右） | 准确率差（百分点） | 95%视频分组区间（百分点） |
|---|---:|---:|
| visual_gain_16: uniform_16 − text_only | 29.70 | [24.85, 34.48] |
| dense_full_clip_64_vs_4: uniform_64 − uniform_4 | 6.26 | [2.84, 9.84] |
| coverage_at_16_frames: uniform_16 − native_center_16 | 5.05 | [1.03, 9.00] |
| local_window_64_vs_1: native_center_64 − native_center_1 | 3.23 | [-0.19, 6.42] |
| new_frame_information: uniform_16 − repeat_16 | 6.46 | [2.25, 10.64] |
| shuffle_sensitivity: uniform_16 − shuffle_16 | 0.61 | [-1.64, 2.83] |
| reverse_sensitivity: uniform_16 − reverse_16 | 0.81 | [-1.36, 2.97] |
| correct_vs_wrong_video: uniform_16 − wrong_video_16 | 39.19 | [33.89, 44.60] |
| zero_visual_vs_timing_text: zero_16 − text_timing_16 | -2.42 | [-4.71, -0.20] |

上述差异用于区分视觉增益、时间覆盖、新增帧信息和顺序敏感度。区间跨零时，不能据此确认存在相应增益或敏感度。各比较为预先约定的探索性比较，未做多重比较校正。

## 分包特征擦除

在官方双线性空间池化之后、网格换行token插入之前切分。每帧196×3584维，按小端F32序列化，以1080字节特征负载分包，沿用原Testbed负载预算。丢包位置全部置零，保留结构token。离线IID掩码按视频和重复种子固定，同视频各题复用同一已核验视觉前缀；不同擦除率使用嵌套掩码。3组掩码均值的区间以每题跨掩码平均正确值为输入，按视频分组bootstrap，没有将同题的3次预测当作独立问题。0%/100%端点各只生成一次。

![容错曲线](figures/feature_erasure_curve.png)

| 请求擦除率 | 实际缺包比例 | 平均准确率 | 95%视频分组区间 | 各掩码准确率 |
|---:|---:|---:|---:|---|
| 0.00% | 0.00% | 77.98% | [73.64, 82.16]% | 77.98% |
| 5.00% | 5.00% | 77.91% | [73.72, 81.96]% | 77.78%, 77.78%, 78.18% |
| 10.00% | 10.00% | 78.05% | [73.84, 82.09]% | 77.78%, 77.78%, 78.59% |
| 20.00% | 19.99% | 77.78% | [73.77, 81.64]% | 77.17%, 77.98%, 78.18% |
| 30.00% | 29.99% | 77.24% | [73.25, 81.13]% | 77.78%, 77.17%, 76.77% |
| 40.00% | 40.00% | 77.04% | [73.11, 80.90]% | 77.17%, 76.57%, 77.37% |
| 50.00% | 50.00% | 75.89% | [71.75, 79.93]% | 76.16%, 75.56%, 75.96% |
| 60.00% | 59.99% | 75.08% | [71.17, 79.00]% | 76.36%, 73.33%, 75.56% |
| 70.00% | 70.01% | 75.08% | [71.08, 79.11]% | 76.16%, 74.55%, 74.55% |
| 80.00% | 80.01% | 72.86% | [68.94, 76.76]% | 72.93%, 71.72%, 73.94% |
| 90.00% | 90.01% | 65.19% | [61.09, 69.47]% | 65.25%, 64.24%, 66.06% |
| 95.00% | 95.00% | 57.71% | [53.23, 62.26]% | 57.37%, 58.18%, 57.58% |
| 100.00% | 100.00% | 45.45% | [40.92, 50.11]% | 45.45% |

完整保留0%至100%的所有点，不做单调拟合或选择性裁剪。此曲线针对等间隔16帧，不能外推为64帧或实际RTC丢包结果。

## 随机基础性能与解释边界

每题5个选项，均匀随机的理论期望为20%。预先固定随机种子的一次随机选择为83/495（16.77%）；始终选A为121/495（24.44%）。随机生成器的一次实现不必恰好20%。纯文本、全零特征、错视频和独立高斯随机特征是不同对照，不预设它们等同于随机猜测。

预设近似等效判据为90%视频分组bootstrap区间完全处于[15%,25%]；“未发现显著差异”不能代替等效判断。高斯对照为独立N(0,1)，未匹配真实特征分布。带时间文本对照用于识别视觉布局和时间文字提示的影响；错配视频保留目标时间槽文本。

NExT-QA官方已有较高的BlindQA文本基线，换数据集并不保证消除语言偏差。LLaVA-Video公开训练混合包括NExT-QA，属于数据集域有重合的评测；这不证明验证集泄漏，也不能描述为从未接触该数据集的泛化。

## 核验与限制

- 实际接收视觉特征哈希通过 2760/2760 个视频×条件前缀；共 2880 个独立前缀记录。题目复用前缀，不是23,760次独立视觉传输。
- 真实模型1/4帧无损切分的实际LM输入嵌入哈希与完整路径相同；缓存/分块生成在1/4/16帧与完整接收端生成一致。64帧验证了可运行，未声称完成未分块64帧对照。
- GPU为RTX3070 8GB；量化与内存优化配置见runtime_provenance.json，pilot.json保留真实模型一致性核验。模型权重与视频不上传。
- 60视频子集的问题类型构成不同于完整验证集；不将结果当作全验证集或官方BF16成绩。视频来自固定第三方镜像，SHA与镜像记录一致，未核验与官方Google Drive压缩包逐字节相同。
- 运行错误、不可用和格式无效计数分别保留；prefix_audits与attempts记录实际路径。suffix_decode_seconds不包含共同视觉前缀准备，不能称为端到端时延。

## 可复核文件

- [全部条件数值](condition_summary.csv)、[完整擦除曲线](loss_curve.csv)、[逐题成对记录](paired_predictions.csv)
- [原始JSONL答案](predictions/)、[全部统计及类型分组](analysis.json)、[运行身份](run_identity.json)
- [真实模型核验](pilot.json)、[实际特征前缀记录](prefix_audits.jsonl)、[固定条件](../../configs/experiment.json)

来源：[NExT-QA官方仓库](https://github.com/doc-doc/NExT-QA)、[LLaVA-Video模型](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2)、[官方介绍](https://llava-vl.github.io/blog/2024-09-30-llava-video/)、[固定运行源码](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/bce12e479bc4dfee2b9c50c88137b01ff51bd483)。
