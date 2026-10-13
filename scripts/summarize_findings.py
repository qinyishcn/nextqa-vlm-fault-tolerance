"""Write an interpreted report from completed answers, without new inference."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from statistics_eval import cluster_interval, fixed_scores


def pct(value):
    return f'{100 * value:.2f}%'


def ci_text(values):
    return f'[{100 * values[0]:.2f}, {100 * values[1]:.2f}]'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    root = Path(args.results)
    analysis = json.loads((root / 'analysis.json').read_text())
    audit = json.loads((root / 'independent_verification.json').read_text())
    assert audit['status'] == 'passed'
    questions = json.loads(Path(args.manifest).read_text())['questions']
    videos = [q['video_id'] for q in questions]
    summaries = analysis['condition_summaries']
    outcomes = {}
    for name in summaries:
        rows = [json.loads(line) for line in (root / 'predictions' / f'{name}.jsonl').read_text().splitlines()]
        outcomes[name] = fixed_scores(questions, rows)[0]

    # These supplementary paired summaries were added after inference completed.
    # They do not change the preregistered conditions, labels or scoring method.
    loss_differences = []
    for row in analysis['loss_curve']:
        rate = row['requested_erasure_rate']
        names = (['uniform_16'] if rate == 0 else ['zero_16'] if rate == 1 else
                 [f'loss_{round(rate * 100):03d}_s{i}' for i in range(3)])
        mean = np.mean([outcomes[name] for name in names], axis=0)
        difference = (mean - np.array(outcomes['uniform_16'])).tolist()
        interval = cluster_interval(difference, videos)
        interval = [0.0 if abs(value) < 1e-12 else value for value in interval]
        loss_differences.append({'requested_erasure_rate': rate,
            'accuracy_difference_vs_clean': float(np.mean(difference)),
            'ci95_low': interval[0], 'ci95_high': interval[1]})
    temporal = {}
    indices = [i for i, q in enumerate(questions) if q['question_type'].startswith('T')]
    for name, left, right in [
        ('native64_vs_single', 'native_center_64', 'native_center_1'),
        ('different_frames_vs_repeat', 'uniform_16', 'repeat_16'),
        ('correct_order_vs_shuffle', 'uniform_16', 'shuffle_16'),
        ('correct_order_vs_reverse', 'uniform_16', 'reverse_16')]:
        difference = [outcomes[left][i] - outcomes[right][i] for i in indices]
        temporal[name] = {'left': left, 'right': right, 'questions': len(indices),
            'accuracy_difference': float(np.mean(difference)),
            'video_bootstrap_95': cluster_interval(difference, [videos[i] for i in indices])}
    supplementary = {'scope': 'Post-completion exploratory paired summaries; no inference changes',
        'loss_differences': loss_differences, 'temporal_family_contrasts': temporal}
    (root / 'supplementary_analysis.json').write_bytes((json.dumps(supplementary, indent=2) + '\n').encode())
    with (root / 'loss_paired_differences.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(loss_differences[0]))
        writer.writeheader()
        writer.writerows(loss_differences)

    overall = lambda name: summaries[name]['groups']['overall']
    accuracy = lambda name: pct(overall(name)['accuracy'])
    effect = lambda name: analysis['contrasts'][name]
    status = {field: sum(value['overall'][field] for value in summaries.values())
              for field in ['missing_answers', 'unavailable_answers', 'error_records', 'invalid_answers']}
    report = ['# LLaVA-Video / NExT-QA 实测结论', '',
        '已完成固定60视频、495题、48组条件，共23,760次回答。'
        '本机RTX3070 8GB，LLaVA-Video-7B-Qwen2语言模型线性层NF4双量化、BF16计算；'
        '这是该量化配置和固定子集的成绩，不能当作官方BF16全验证集成绩。', '',
        '**结果：视觉有明显增益，多帧内容与时间覆盖有作用，但纯文本和完全错误视觉仍远高于随机。'
        '整体准确率没有显示出明确的帧顺序依赖。**', '',
        '**范围：离线中间特征分包擦除。本轮未执行新的原版Testbed RTC/netem network profile测试。**', '',
        '## 纯文本及错误视觉是否匹配随机选择', '',
        '| 条件 | 答对/495 | 准确率 | 95%视频分组区间 | 随机近似等效用90%区间 |',
        '|---|---:|---:|---:|---:|']
    for name in ['uniform_16', 'uniform_64', 'text_only', 'text_timing_16',
                 'wrong_video_16', 'gaussian_16', 'zero_16']:
        score = overall(name)
        report.append(f'| {name} | {score["correct"]}/495 | {pct(score["accuracy"])} | '
                      f'{ci_text(score["video_bootstrap_95"])}% | {ci_text(score["video_bootstrap_90"])}% |')
    gain = effect('visual_gain_16')
    report += ['', '五选一均匀随机的理论期望为20%，预设±5个百分点近似等效区间为[15%,25%]。'
        '纯文本、错视频、独立高斯特征、全零特征四个总体对照的90%区间均完整高于25%，'
        '**均不匹配随机基础性能**。固定种子的一次随机选择83/495（16.77%），'
        '始终选A为121/495（24.44%）；它们与模型输出分开记录。', '',
        f'16帧正常视觉相对纯文本提高{100 * gain["accuracy_difference"]:.2f}个百分点，'
        f'成对95%区间为{ci_text(gain["video_bootstrap_95"])}个百分点。'
        '换数据集没有消除问题及选项提供的语言线索。'
        '错视频会误导答案，而全零/高斯并不会删除问题与选项；它们也保留视觉结构token和目标时间文字。'
        '因此“视觉完全错误”不等于“随机选择选项”。本实验没有单独量化各类语言线索的因果贡献。', '',
        '![正常及错误视觉对照](figures/baseline_controls.png)', '',
        '## 连续多帧、覆盖范围与帧顺序', '',
        '| 帧数 | 原生FPS连续中心帧 | 覆盖全视频等间隔帧 |', '|---:|---:|---:|']
    for count in [1, 4, 16, 64]:
        native = f'native_center_{count}'
        uniform = f'uniform_{count}' if count != 1 else native
        report.append(f'| {count} | {accuracy(native)} | {accuracy(uniform)} |')
    report += ['', '连续4/16/64帧跨度中位数分别为0.1001/0.5005/2.1021秒；'
        '全视频等间隔帧覆盖约33.69秒，4/16/64帧覆盖相同起止时刻。'
        'NExT-QA问题针对整段视频，没有问题提出时间；连续窗口预先固定在视频中心，未按答案选择事件窗口。', '',
        '| 比较（左减右） | 差值，百分点 | 成对95%区间，百分点 |', '|---|---:|---:|']
    for name, label in [
        ('coverage_at_16_frames', '全视频16帧 − 连续中心16帧'),
        ('dense_full_clip_64_vs_4', '全视频64帧 − 全视频4帧'),
        ('local_window_64_vs_1', '连续64帧 − 中心单帧'),
        ('new_frame_information', '不同16帧 − 中心单帧重复16次'),
        ('shuffle_sensitivity', '原顺序16帧 − 打乱16帧'),
        ('reverse_sensitivity', '原顺序16帧 − 倒放16帧')]:
        row = effect(name)
        report.append(f'| {label} | {100 * row["accuracy_difference"]:.2f} | {ci_text(row["video_bootstrap_95"])} |')
    report += ['', '更广的时间覆盖和更多不同帧提供了增益。连续64帧相对单帧的总体区间仍跨零；'
        '不能据此确认该固定中心窗口的总体增益。原顺序、打乱、倒放分别为'
        f'{accuracy("uniform_16")}、{accuracy("shuffle_16")}、{accuracy("reverse_16")}，'
        '成对差异区间跨零，未确认总体顺序敏感度；这也不能证明模型完全忽略顺序。', '',
        '![帧数与覆盖范围](figures/frame_dependence.png)', '',
        '### 问题类型', '',
        '| 条件 | C因果（251题） | T时间（169题） | D描述（75题） |', '|---|---:|---:|---:|']
    for name in ['text_only', 'native_center_1', 'native_center_64', 'uniform_16', 'uniform_64',
                 'repeat_16', 'shuffle_16', 'reverse_16', 'wrong_video_16', 'zero_16']:
        groups = summaries[name]['groups']
        report.append(f'| {name} | ' + ' | '.join(pct(groups[key]['accuracy']) for key in ['C', 'T', 'D']) + ' |')
    report += ['', 'T类的成对补充分析如下。这些类型内比较及下面的损伤差异区间在推理完成后汇总，'
        '属于探索性结果，未校正多重比较；全部8个细分类结果见analysis.json。', '',
        '| T类比较 | 差值，百分点 | 95%视频分组区间，百分点 |', '|---|---:|---:|']
    for row in temporal.values():
        report.append(f'| {row["left"]} − {row["right"]} | {100 * row["accuracy_difference"]:.2f} | '
                      f'{ci_text(row["video_bootstrap_95"])} |')
    report += ['', 'T类中，连续64帧相对单帧提高9.47个百分点，区间未跨零；'
        '但打乱和倒放的差异区间仍跨零。多帧对时间类题目有帮助，并不直接证明依赖正确时间顺序。', '',
        '## 分包擦除容错曲线', '',
        '切分点为官方视觉投影和空间池化之后、网格换行token之前。16帧池化特征以小端F32传输，'
        '每包1080字节特征负载（沿用原Testbed负载预算，非1500字节净特征），'
        '每个视频缓冲区44,957,696字节、41,628包。缺包对应位置全部置零。'
        '本轮只离线生成IID随机擦除掩码，没有模拟RTC重传、突发丢包、拥塞或真实包头。', '',
        '![完整擦除曲线](figures/feature_erasure_curve.png)', '',
        '| 请求擦除率 | 平均准确率 | 相对无损差值，百分点 | 成对95%区间，百分点 |',
        '|---:|---:|---:|---:|']
    for curve, delta in zip(analysis['loss_curve'], loss_differences):
        report.append(f'| {pct(curve["requested_erasure_rate"])} | {pct(curve["mean_accuracy"])} | '
                      f'{100 * delta["accuracy_difference_vs_clean"]:.2f} | '
                      f'{ci_text([delta["ci95_low"], delta["ci95_high"]])} |')
    report += ['', '0%–40%擦除的点估计变化较小，50%–70%仍约75%–76%；'
        '80%为72.86%，90%为65.19%，95%为57.71%，100%为45.45%。'
        '下降在本次高擦除区间更明显，但没有预先定义或估计统一的“拐点”。'
        '50%点的成对区间上界约为零，不能把数值舍入误差视作明确阈值。'
        '保留全部0%–100%点及小幅非单调变化，不裁掉端点或做单调拟合。'
        '100%端点仍高于随机，意味着高损伤区间存在较高的非视觉任务基础性能。', '',
        '每个中间擦除率使用3组固定掩码，0%/100%确定端点各运行一次。'
        '同视频各题共享掩码；不同擦除率采用嵌套掩码。区间按视频分组，'
        '先对同题跨掩码取平均，没有把3份答案当成1485个独立问题。'
        '仅3组掩码不足以充分刻画任意网络环境的随机性，不能外推到实际RTC network profile。', '',
        '## 核验、复现与边界', '',
        f'- {audit["fixed_slots_independently_scored"]:,}条回答独立计分；'
        f'{audit["official_annotation_questions_matched"]}题的问题、选项和标签与固定官方CSV逐项匹配。',
        f'- 从BF16源特征独立重建{audit["received_feature_buffers_replayed"]}个接收缓冲区、'
        f'{audit["packet_masks_replayed"]}份损伤掩码，缺包数量、置零字节与实际接收SHA-256全部一致。'
        '这证明所声明损伤进入了接收端，不单凭准确率曲线推断容错。',
        f'- 昨晚暂停前{audit["saved_answers_retained_byte_for_byte"]:,}条回答逐字节保留；'
        '恢复只补未尝试槽位。'
        f'最终缺失{status["missing_answers"]}、不可用{status["unavailable_answers"]}、'
        f'运行错误{status["error_records"]}、格式无效{status["invalid_answers"]}。',
        '- 真实模型无损切分和缓存生成一致性边界见pilot.json。64帧验证了分块可运行，'
        '没有声称完成未分块64帧完整路径对照。',
        '- LLaVA-Video公开训练混合包含NExT-QA；这是数据集域有重合的评测，'
        '不证明验证集泄漏，也不声称数据集未见泛化。',
        '- 子集保持视频成组，按固定种子及问题类型选取；不是完整验证集，也没有对视频平均等权计分。'
        '视频镜像SHA核验通过，未与官方Google Drive压缩包做逐字节比对。',
        '- 所有区间为10,000次固定种子视频分组bootstrap，主要比较按预设方法汇总。'
        '未做多重比较校正，不构成实际网络部署的容错验证。', '',
        '## 文件', '',
        '- [完整自动统计报告](report_zh.md)、[48组数值](condition_summary.csv)、[13点曲线及实际擦除率](loss_curve.csv)',
        '- [逐题成对CSV](paired_predictions.csv)、[原始答案](predictions/)、[完整统计及类型分组](analysis.json)',
        '- [独立核验](independent_verification.json)、[补充成对分析](supplementary_analysis.json)、[曲线成对差异CSV](loss_paired_differences.csv)',
        '- [固定实验方案](../../configs/experiment.json)、[预设协议](../../docs/protocol.md)、[复现命令](../../docs/reproduce.md)', '',
        '来源：[NExT-QA官方仓库](https://github.com/doc-doc/NExT-QA)、'
        '[LLaVA-Video-7B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2)、'
        '[LLaVA-Video官方介绍及训练混合](https://llava-vl.github.io/blog/2024-09-30-llava-video/)。', '']
    (root / 'report_findings_zh.md').write_bytes('\n'.join(report).encode())
    print(json.dumps({'report': str(root / 'report_findings_zh.md'), 'supplementary_comparisons': 17}))


if __name__ == '__main__':
    main()
