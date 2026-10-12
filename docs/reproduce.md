# 复现本轮离线实验

固定配置见 `configs/experiment.json`。主文中的数据准备可在Windows运行；
GPU推理使用Linux/WSL、Python3.12、CUDA12.1和RTX3070 8GB。
先准备并核验全部60视频及原生帧，再执行以下步骤。

## 权重与原版运行源码

```sh
python scripts/download_checkpoint.py --output assets/models/LLaVA-Video-7B-Qwen2 --provenance results/preparation/model_provenance.json --mirror modelscope --workers 8 --block-mib 4
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git vendor/LLaVA-NeXT
git -C vendor/LLaVA-NeXT checkout bce12e479bc4dfee2b9c50c88137b01ff51bd483
```

`--mirror modelscope`仅替换传输来源，先比对镜像权重与固定官方revision的SHA-256，
下载后再次核验整个文件。配置与tokenizer仍来自固定官方revision；
可改用 `--mirror huggingface`，推理权重字节应完全一致。

本机实际通过官方GitHub归档获得源码，并对77个Python包/许可证文件核对Git
blob身份与SHA-256，清单见 `results/preparation/source_package_hashes.json`；
仓库不包含改写的LLaVA源码。加载时仅避免重复下载随后会被完整checkpoint覆盖的
Google初始化权重；缺失、额外或形状不符的参数都会使接受核验失败。

## 独立推理环境

```sh
python3.12 -m venv .venv-inference
.venv-inference/bin/python -m pip install -r requirements-inference.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

本机以独立venv读取现有Testbed中相同版本的Torch运行时，新增依赖装入新venv。
原Testbed的Transformers保持4.44.2，新环境使用4.45.2；未修改原环境。
复现时可以全部新装，无需依赖本机的Testbed路径。

## 真实模型接受核验与正式运行

以下是Bash命令，变量路径按数据实际位置设置：

```sh
NEXTQA_ASSETS="$PWD/assets"
NEXTQA_SOURCE="$PWD/vendor/LLaVA-NeXT"
NEXTQA_MODEL="$NEXTQA_ASSETS/models/LLaVA-Video-7B-Qwen2"
study_args=(
  --source "$NEXTQA_SOURCE"
  --model "$NEXTQA_MODEL"
  --assets "$NEXTQA_ASSETS"
  --output results/offline_20261012
  --plan configs/experiment.json
  --manifest results/preparation/dataset_manifest.json
  --frame-audit results/preparation/frame_cache_audit.json
)
.venv-inference/bin/python scripts/check_pilot.py "${study_args[@]}"
.venv-inference/bin/python scripts/run_study.py "${study_args[@]}" --stage all
```

接受核验先比较真实完整模型和特征接收路径，再检查共同前缀KV复用、分块prefill和
末位置输出头。64帧采用分块路径，不声称未分块64帧对照已完成。
正式运行要求接受结果与模型配置、包版本、条件清单及三个核心实现文件哈希匹配。

编码按4帧一批调用原版视觉编码/投影/池化，每帧特征固定缓存，所有条件复用。
编码期间把未参与视觉运算的词嵌入和语言输出头暂放CPU，完成后移回GPU；
恢复已有缓存时重新计算首批4帧，要求逐帧特征字节哈希一致。
正式语言模型只消费接收到的池化特征，保留原版网格换行处理。
每个视频×条件的共同前缀只prefill一次；后续问题复用同一因果KV前缀。
这降低重复计算，不代表持续视频流或独立逐题网络传输。

已保存错误和无效答案不替换。若运行意外中断，可使用相同命令继续未尝试的槽位；
已有开始记录而未保存答案的尝试在恢复时记为错误。不能删除失败记录后重跑来提高成绩。
`completion.json`是槽位记账完成标志，运行成功、输出有效和实际特征核验另行统计。

## 计分、图表与逐题复核

在单独分析环境执行（本机分析采用Windows Python）：

```sh
python -m pip install -r requirements-analysis.txt
python scripts/analyze_results.py --manifest results/preparation/dataset_manifest.json --plan configs/experiment.json --frame-audit results/preparation/frame_cache_audit.json --results results/offline_20261012
```

输出整体与C/T/D及8类问题统计、预设成对差异、495题固定分母结果、
13点完整擦除曲线与3组掩码结果，以及PNG/PDF图表。
随机等效判据为90%视频分组区间完全落入[15%,25%]，不将“不显著”当成等效。

本轮不执行RTC、网卡netem或network profile；这些真实网络任务必须继续沿用原版
Testbed流程，不能把本离线分包擦除当作其结果。
