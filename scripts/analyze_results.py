"""Score every frozen condition and export full-range figures and raw tables."""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nextqa_eval import parse_answer
from statistics_eval import cluster_interval, fixed_scores, summarize_results


def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2) + '\n').encode())


def save_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return f'{100 * value:.2f}%'


def interval_text(interval):
    return '不可估计' if interval is None else f'[{100 * interval[0]:.2f}, {100 * interval[1]:.2f}]%'


def save_figure(fig, folder, name):
    fig.savefig(folder / f'{name}.png', dpi=220, bbox_inches='tight')
    fig.savefig(folder / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--frame-audit', required=True)
    parser.add_argument('--results', required=True)
    args = parser.parse_args()
    root = Path(args.results)
    manifest = json.loads(Path(args.manifest).read_text())
    plan = json.loads(Path(args.plan).read_text())
    frame_audit = json.loads(Path(args.frame_audit).read_text())
    completion = json.loads((root / 'completion.json').read_text())
    if completion['status'] != 'complete':
        raise ValueError('Do not label an unfinished run as a final evaluation')
    questions = manifest['questions']
    videos = [q['video_id'] for q in questions]
    records = {condition['id']: load_jsonl(root / 'predictions' / f'{condition["id"]}.jsonl')
               for condition in plan['conditions']}
    outcomes = {key: fixed_scores(questions, rows)[0] for key, rows in records.items()}
    summaries = {}
    for key, rows in records.items():
        result = summarize_results(questions, rows, records['text_only'])
        differences = [a - b for a, b in zip(outcomes[key], outcomes['uniform_16'])]
        result['paired_uniform_16'] = {'accuracy_difference': float(np.mean(differences)),
            'video_bootstrap_95': cluster_interval(differences, videos)}
        result['prediction_option_counts'] = dict(collections.Counter(
            parse_answer(row.get('raw_response')) or ('UNAVAILABLE' if row.get('raw_response') is None else 'INVALID')
            for row in rows))
        summaries[key] = result
    audits = load_jsonl(root / 'prefix_audits.jsonl')
    unique_audits = {}
    feature_checks = 0
    for row in audits:
        key = row['prefix_id']
        if key in unique_audits:
            previous = unique_audits[key]
            if previous['receiver']['prefix_embedding_sha256'] != row['receiver']['prefix_embedding_sha256']:
                raise ValueError('A resumed prefix changed its actual input embeddings')
        unique_audits[key] = row
    for row in unique_audits.values():
        expected = row['control'].get('condition_feature_sha256')
        if expected:
            received = row['receiver']['receiver']['received_sha256']
            if expected != received:
                raise ValueError('Actual received visual feature bytes differ from the condition')
            feature_checks += 1
    curve = []
    for rate in plan['loss_rates']:
        if rate == 0:
            ids = ['uniform_16']
        elif rate == 1:
            ids = ['zero_16']
        else:
            ids = [f'loss_{round(rate * 100):03d}_s{seed}' for seed in range(3)]
        mean_values = np.mean([outcomes[key] for key in ids], axis=0).tolist()
        observed = [row for row in unique_audits.values() if row['condition_id'] in ids]
        erasures = [row['control']['erasure'] for row in observed if 'erasure' in row['control']]
        total_packets = sum(row['total_packets'] for row in erasures)
        total_bytes = sum(row['feature_bytes'] for row in erasures)
        item = {'requested_erasure_rate': rate, 'mean_accuracy': float(np.mean(mean_values)),
            'video_bootstrap_95': cluster_interval(mean_values, videos),
            'seed_accuracies': [summaries[key]['overall']['accuracy'] for key in ids],
            'fixed_questions_per_seed': len(questions), 'mask_repeats': len(ids),
            'feature_buffer_audits': len(observed), 'expected_feature_buffer_audits': 60 * len(ids),
            'measured_missing_packet_fraction': sum(row['missing_packets'] for row in erasures) / total_packets if total_packets else (0 if rate == 0 and observed else None),
            'measured_zeroed_byte_fraction': sum(row['zeroed_bytes'] for row in erasures) / total_bytes if total_bytes else (0 if rate == 0 and observed else None)}
        item['by_family'] = {}
        for family in ['C', 'T', 'D']:
            indices = [i for i, q in enumerate(questions) if q['question_type'].startswith(family)]
            values = [mean_values[i] for i in indices]
            item['by_family'][family] = {'questions': len(indices), 'accuracy': float(np.mean(values)),
                'video_bootstrap_95': cluster_interval(values, [videos[i] for i in indices])}
        curve.append(item)
    contrasts = {}
    for name, left, right in [
        ('visual_gain_16', 'uniform_16', 'text_only'),
        ('dense_full_clip_64_vs_4', 'uniform_64', 'uniform_4'),
        ('coverage_at_16_frames', 'uniform_16', 'native_center_16'),
        ('local_window_64_vs_1', 'native_center_64', 'native_center_1'),
        ('new_frame_information', 'uniform_16', 'repeat_16'),
        ('shuffle_sensitivity', 'uniform_16', 'shuffle_16'),
        ('reverse_sensitivity', 'uniform_16', 'reverse_16'),
        ('correct_vs_wrong_video', 'uniform_16', 'wrong_video_16'),
        ('zero_visual_vs_timing_text', 'zero_16', 'text_timing_16')]:
        difference = [a - b for a, b in zip(outcomes[left], outcomes[right])]
        contrasts[name] = {'left': left, 'right': right, 'accuracy_difference': float(np.mean(difference)),
            'video_bootstrap_95': cluster_interval(difference, videos)}
    analysis = {'condition_summaries': summaries, 'loss_curve': curve, 'contrasts': contrasts,
        'feature_hash_checks_passed': feature_checks, 'unique_prefix_audits': len(unique_audits),
        'feature_hash_checks_expected': 60 * sum(c['kind'] not in ['text', 'text_timing'] for c in plan['conditions']),
        'slot_accounting': completion, 'intervals': 'Unadjusted exploratory video-cluster bootstrap intervals'}
    save_json(root / 'analysis.json', analysis)
    rows = []
    for key, result in summaries.items():
        score = result['groups']['overall']
        ci = score['video_bootstrap_95'] or [None, None]
        rows.append({'condition_id': key, **result['overall'], 'ci95_low': ci[0], 'ci95_high': ci[1],
            'chance_equivalence': score['chance_equivalence'],
            'delta_vs_text': score['paired_reference']['accuracy_difference'],
            'delta_vs_uniform_16': result['paired_uniform_16']['accuracy_difference']})
    save_csv(root / 'condition_summary.csv', rows)
    save_csv(root / 'loss_curve.csv', [{key: value for key, value in row.items()
        if key not in ['by_family', 'video_bootstrap_95', 'seed_accuracies']} | {
            'ci95_low': row['video_bootstrap_95'][0], 'ci95_high': row['video_bootstrap_95'][1],
            'seed_accuracies': json.dumps(row['seed_accuracies']),
            **{f'{family}_accuracy': row['by_family'][family]['accuracy'] for family in ['C', 'T', 'D']}}
        for row in curve])
    by_id = {key: {row['question_id']: row for row in value} for key, value in records.items()}
    long_rows = []
    for key in records:
        for index, question in enumerate(questions):
            row = by_id[key].get(question['question_id'], {})
            long_rows.append({'condition_id': key, 'question_id': question['question_id'],
                'video_id': question['video_id'], 'question_type': question['question_type'],
                'label': question['label'], 'raw_response': row.get('raw_response'),
                'parsed_answer': parse_answer(row.get('raw_response')), 'correct': outcomes[key][index],
                'text_only_correct': outcomes['text_only'][index], 'uniform16_correct': outcomes['uniform_16'][index],
                'status': row.get('status', 'missing'), 'prefix_id': row.get('prefix_id')})
    save_csv(root / 'paired_predictions.csv', long_rows)

    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    x = [100 * row['requested_erasure_rate'] for row in curve]
    y = [100 * row['mean_accuracy'] for row in curve]
    low = [100 * row['video_bootstrap_95'][0] for row in curve]
    high = [100 * row['video_bootstrap_95'][1] for row in curve]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout='constrained')
    ax.fill_between(x, low, high, alpha=.18, label='95% video-cluster interval')
    ax.plot(x, y, 'o-', linewidth=2, label='16 uniform frames; mean over masks')
    for row in curve:
        ax.scatter([100 * row['requested_erasure_rate']] * len(row['seed_accuracies']),
            [100 * value for value in row['seed_accuracies']], s=16, alpha=.45, color='tab:blue')
    ax.axhline(20, color='black', linestyle='--', label='Uniform random expectation: 20%')
    ax.axhline(100 * summaries['text_only']['overall']['accuracy'], color='tab:orange',
        linestyle=':', label='Text only')
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel='Requested packet erasure (%)',
        ylabel='Task accuracy (%)', title='LLaVA-Video-7B-Qwen2 / NExT-QA: feature erasure')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, loc='best')
    save_figure(fig, figures, 'feature_erasure_curve')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    counts = [1, 4, 16, 64]
    for scheme, label, marker in [('native_center', 'Native consecutive center frames', 'o'),
                                 ('uniform', 'Uniform frames across full clip', 's')]:
        keys = [f'{scheme}_{n}' if n != 1 else 'native_center_1' for n in counts]
        scores = [summaries[key]['groups']['overall'] for key in keys]
        accuracies = np.array([row['accuracy'] for row in scores]) * 100
        intervals = np.array([row['video_bootstrap_95'] for row in scores]).T * 100
        axes[0].errorbar(counts, accuracies, yerr=[accuracies - intervals[0], intervals[1] - accuracies],
            fmt=marker + '-', capsize=3, label=label)
        spans = [float(np.median([row['selection_time_spans_seconds'][f'{scheme}_{n}']
                    for row in frame_audit['video_metadata']])) for n in counts]
        axes[1].plot(counts, spans, marker + '-', label=label)
    for ax in axes:
        ax.set_xscale('log', base=4)
        ax.set_xticks(counts, [str(n) for n in counts])
        ax.set_xlabel('Frames provided')
        ax.grid(alpha=.2)
    axes[0].set(ylim=(0, 100), ylabel='Task accuracy (%)', title='Frame-count and coverage controls')
    axes[1].set(ylabel='Median temporal span (seconds)', title='Temporal coverage differs')
    axes[0].legend(fontsize=8, loc='best')
    save_figure(fig, figures, 'frame_dependence')

    baseline_keys = ['text_only', 'text_timing_16', 'uniform_16', 'repeat_16', 'shuffle_16',
                     'reverse_16', 'wrong_video_16', 'gaussian_16', 'zero_16']
    labels = ['Text only', 'Text + timing', 'Correct video', 'Repeated frame', 'Shuffled frames',
              'Reversed frames', 'Wrong video', 'Gaussian features', 'Zero features']
    fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
    scores = [summaries[key]['groups']['overall'] for key in baseline_keys]
    ys = np.array([row['accuracy'] for row in scores]) * 100
    intervals = np.array([row['video_bootstrap_95'] for row in scores]).T * 100
    ax.bar(range(len(ys)), ys, color=['#df9b46', '#e8bc7a', '#3f8a9b'] + ['#8aafba'] * 6)
    ax.errorbar(range(len(ys)), ys, yerr=[ys - intervals[0], intervals[1] - ys], fmt='none', color='black', capsize=3)
    ax.axhline(20, color='black', linestyle='--', label='Uniform random expectation: 20%')
    ax.set_xticks(range(len(ys)), labels, rotation=30, ha='right')
    ax.set(ylim=(0, 100), ylabel='Task accuracy (%)', title='Normal, absent and incorrect visual information')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=.2)
    save_figure(fig, figures, 'baseline_controls')

    report = ['# LLaVA-Video / NExT-QA 视觉依赖与特征容错实测报告', '',
        '本报告针对固定60视频、495题子集，48组条件，共23,760个固定答题槽位。'
        '模型为LLaVA-Video-7B-Qwen2，语言模型线性层NF4双量化、BF16计算。'
        '缺失、运行错误和无效格式均计错；不替换失败答案。', '',
        '**本轮为离线特征擦除实验，尚不构成原版Testbed的RTC/netem网络实测。**', '',
        '## 主要对照', '',
        '| 条件 | 答对/495 | 准确率 | 95%视频分组区间 | 不可用 | 格式无效 | 随机等效判据 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for key in ['text_only', 'text_timing_16', 'native_center_1', 'native_center_4', 'native_center_16',
                'native_center_64', 'uniform_4', 'uniform_16', 'uniform_64', 'repeat_16', 'shuffle_16',
                'reverse_16', 'wrong_video_16', 'gaussian_16', 'zero_16']:
        score = summaries[key]['groups']['overall']
        report.append(f'| {key} | {score["correct"]}/495 | {percent(score["accuracy"])} | '
            f'{interval_text(score["video_bootstrap_95"])} | {score["unavailable_answers"]} | '
            f'{score["invalid_answers"]} | {score["chance_equivalence"]} |')
    report += ['', '![基础对照](figures/baseline_controls.png)', '', '## 连续帧与时间覆盖', '',
        '原生连续帧不抽样，固定取视频中心窗口；等间隔帧覆盖整个视频。'
        '本子集连续4帧时间跨度中位数0.1001秒，连续64帧2.1021秒，'
        '等间隔64帧33.6903秒。等间隔4/16/64帧覆盖相同起止时刻，帧数增加主要改变时间采样密度。', '',
        '![帧依赖](figures/frame_dependence.png)', '',
        '| 成对差异（左减右） | 准确率差（百分点） | 95%视频分组区间（百分点） |', '|---|---:|---:|']
    for key, row in contrasts.items():
        ci = row['video_bootstrap_95']
        report.append(f'| {key}: {row["left"]} − {row["right"]} | '
            f'{100 * row["accuracy_difference"]:.2f} | [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}] |')
    report += ['', '上述差异用于区分视觉增益、时间覆盖、新增帧信息和顺序敏感度。'
        '区间跨零时，不能据此确认存在相应增益或敏感度。各比较为预先约定的探索性比较，未做多重比较校正。', '',
        '## 分包特征擦除', '',
        '在官方双线性空间池化之后、网格换行token插入之前切分。每帧196×3584维，'
        '按小端F32序列化，以1080字节特征负载分包，沿用原Testbed负载预算。'
        '丢包位置全部置零，保留结构token。离线IID掩码按视频和重复种子固定，'
        '同视频各题复用同一已核验视觉前缀；不同擦除率使用嵌套掩码。'
        '3组掩码均值的区间以每题跨掩码平均正确值为输入，按视频分组bootstrap，'
        '没有将同题的3次预测当作独立问题。0%/100%端点各只生成一次。', '',
        '![容错曲线](figures/feature_erasure_curve.png)', '',
        '| 请求擦除率 | 实际缺包比例 | 平均准确率 | 95%视频分组区间 | 各掩码准确率 |', '|---:|---:|---:|---:|---|']
    for row in curve:
        measured = row['measured_missing_packet_fraction']
        report.append(f'| {percent(row["requested_erasure_rate"])} | '
            f'{percent(measured) if measured is not None else "未核验"} | {percent(row["mean_accuracy"])} | '
            f'{interval_text(row["video_bootstrap_95"])} | {", ".join(percent(value) for value in row["seed_accuracies"])} |')
    report += ['', '完整保留0%至100%的所有点，不做单调拟合或选择性裁剪。'
        '此曲线针对等间隔16帧，不能外推为64帧或实际RTC丢包结果。', '',
        '## 随机基础性能与解释边界', '',
        '每题5个选项，均匀随机的理论期望为20%。预先固定随机种子的一次随机选择为83/495（16.77%）；'
        '始终选A为121/495（24.44%）。随机生成器的一次实现不必恰好20%。'
        '纯文本、全零特征、错视频和独立高斯随机特征是不同对照，不预设它们等同于随机猜测。', '',
        '预设近似等效判据为90%视频分组bootstrap区间完全处于[15%,25%]；'
        '“未发现显著差异”不能代替等效判断。高斯对照为独立N(0,1)，未匹配真实特征分布。'
        '带时间文本对照用于识别视觉布局和时间文字提示的影响；错配视频保留目标时间槽文本。', '',
        'NExT-QA官方已有较高的BlindQA文本基线，换数据集并不保证消除语言偏差。'
        'LLaVA-Video公开训练混合包括NExT-QA，属于数据集域有重合的评测；'
        '这不证明验证集泄漏，也不能描述为从未接触该数据集的泛化。', '',
        '## 核验与限制', '',
        f'- 实际接收视觉特征哈希通过 {feature_checks}/{analysis["feature_hash_checks_expected"]} 个视频×条件前缀；'
        f'共 {len(unique_audits)} 个独立前缀记录。题目复用前缀，不是23,760次独立视觉传输。',
        '- 真实模型1/4帧无损切分的实际LM输入嵌入哈希与完整路径相同；'
        '缓存/分块生成在1/4/16帧与完整接收端生成一致。64帧验证了可运行，未声称完成未分块64帧对照。',
        '- GPU为RTX3070 8GB；量化与内存优化配置见runtime_provenance.json，'
        'pilot.json保留真实模型一致性核验。模型权重与视频不上传。',
        '- 60视频子集的问题类型构成不同于完整验证集；不将结果当作全验证集或官方BF16成绩。'
        '视频来自固定第三方镜像，SHA与镜像记录一致，未核验与官方Google Drive压缩包逐字节相同。',
        '- 运行错误、不可用和格式无效计数分别保留；prefix_audits与attempts记录实际路径。'
        'suffix_decode_seconds不包含共同视觉前缀准备，不能称为端到端时延。', '',
        '## 可复核文件', '',
        '- [全部条件数值](condition_summary.csv)、[完整擦除曲线](loss_curve.csv)、[逐题成对记录](paired_predictions.csv)',
        '- [原始JSONL答案](predictions/)、[全部统计及类型分组](analysis.json)、[运行身份](run_identity.json)',
        '- [真实模型核验](pilot.json)、[实际特征前缀记录](prefix_audits.jsonl)、[固定条件](../../configs/experiment.json)', '',
        '来源：[NExT-QA官方仓库](https://github.com/doc-doc/NExT-QA)、'
        '[LLaVA-Video模型](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2)、'
        '[官方介绍](https://llava-vl.github.io/blog/2024-09-30-llava-video/)、'
        '[固定运行源码](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/bce12e479bc4dfee2b9c50c88137b01ff51bd483)。', '']
    (root / 'report_zh.md').write_bytes('\n'.join(report).encode())
    print(json.dumps({'conditions_scored': len(summaries), 'slots': completion['fixed_slots'],
        'feature_hash_checks': feature_checks, 'figures': 3}, indent=2))


if __name__ == '__main__':
    main()
