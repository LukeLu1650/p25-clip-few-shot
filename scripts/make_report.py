"""Audit completed runs and generate scientific figures and a Markdown report."""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Defaults to repository root; may be a separate staging directory')
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root).resolve()
    figures = output / 'results/figures'
    docs = output / 'docs'
    figures.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    sources = ['results/summary.csv', 'results/coop-sensitivity.csv']
    baseline, sensitivity = [read_csv(root / name) for name in sources]
    raw = {}
    run_paths = {}
    for category, folder in [('baseline', 'results/raw'), ('sensitivity', 'results/sensitivity-raw')]:
        raw[category] = []
        for path in sorted((root / folder).rglob('metrics.json')):
            obj = json.loads(path.read_text(encoding='utf-8'))
            if obj.get('status') == 'complete':
                raw[category].append(obj)
                run_paths[id(obj)] = path.parent

    def key(c):
        return c['dataset'], c['method'], c['shots'], c['n_ctx'], c['ctx_init']

    for name, summary, expected_rows, expected_runs in [
            ('baseline', baseline, 33, 93), ('sensitivity', sensitivity, 4, 12)]:
        if len(summary) != expected_rows or len(raw[name]) != expected_runs:
            raise ValueError(f'{name}: unexpected condition/run count; do not report incomplete matrices')
        groups = defaultdict(list)
        for obj in raw[name]:
            groups[key(obj['config'])].append(obj)
        if len(groups) != expected_rows:
            raise ValueError(f'{name}: duplicate or unexpected scientific conditions')
        for row in summary:
            c = json.loads(row['condition'])
            group = groups[key(c)]
            seeds = [o['config']['seed'] for o in group]
            expected = [1] if c['method'] == 'zero_shot' else [1, 2, 3]
            if sorted(seeds) != expected or int(row['n']) != len(group):
                raise ValueError(f'{name}: incorrect independent seeds')
            scores = [o['accuracy']['0'] for o in group]
            if not math.isclose(statistics.mean(scores), float(row['mean_accuracy']), abs_tol=1e-8):
                raise ValueError('Summary mean differs from raw metrics')
            if len(scores) > 1 and not math.isclose(statistics.stdev(scores), float(row['sample_std']), abs_tol=1e-8):
                raise ValueError('Summary sample standard deviation differs from raw metrics')
            if {o['pool_sha256'] for o in group} != {row['pool_sha256']}:
                raise ValueError('Partition identity differs between summary and runs')
    # Match support/validation/test manifests across adapted methods for every seed/budget.
    shared = defaultdict(set)
    for obj in raw['baseline']:
        c = obj['config']
        if c['method'] != 'zero_shot':
            shared[(c['dataset'], c['shots'], c['seed'])].add(c['manifest_sha256'])
    if any(len(hashes) != 1 for hashes in shared.values()):
        raise ValueError('Methods used different sample manifests')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    colors = {'coop': '#1676A4', 'tip_adapter': '#CE7027', 'zero_shot': '#657080'}
    titles = {'eurosat': 'EuroSAT', 'dtd': 'DTD', 'flowers102': 'Flowers102'}
    names = {'coop': 'CoOp', 'tip_adapter': 'Tip-Adapter', 'zero_shot': 'Zero-shot CLIP'}
    budgets = [1, 2, 4, 8, 16]

    def finish(fig, name):
        fig.savefig(figures / (name + '.png'), dpi=240)
        fig.savefig(figures / (name + '.svg'))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), sharey=True)
    for ax, dataset in zip(axes, titles):
        rows = [r for r in baseline if r['dataset'] == dataset]
        zero = next(float(r['mean_accuracy']) for r in rows if r['method'] == 'zero_shot')
        ax.axhline(zero, ls='--', color=colors['zero_shot'], lw=1.6, label='Zero-shot CLIP')
        for method in ['coop', 'tip_adapter']:
            ordered = sorted([r for r in rows if r['method'] == method], key=lambda r: int(r['shots']))
            means = np.array([float(r['mean_accuracy']) for r in ordered])
            stds = np.array([float(r['sample_std']) for r in ordered])
            ax.plot(range(5), means, 'o-', color=colors[method], lw=2, ms=5, label=names[method])
            ax.fill_between(range(5), means-stds, means+stds, color=colors[method], alpha=.13)
        ax.set(title=titles[dataset], xlabel='Training examples per class', ylim=(0, 100))
        ax.set_xticks(range(5), [str(v) for v in budgets])
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Test top-1 accuracy (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .055))
    fig.suptitle('Few-shot CLIP adaptation under the shared project protocol', y=.97, fontsize=14)
    fig.text(.5, .015, 'Shading: sample standard deviation across seeds 1, 2, 3. Zero-shot is evaluated once. Dataset split protocols differ.', ha='center', fontsize=9, color='#475569')
    fig.subplots_adjust(left=.065, right=.98, top=.83, bottom=.26, wspace=.12)
    finish(fig, 'baseline_accuracy')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    random_rows = sorted([r for r in sensitivity if not json.loads(r['condition'])['ctx_init']], key=lambda r: json.loads(r['condition'])['n_ctx'])
    init_rows = sorted([r for r in sensitivity if json.loads(r['condition'])['n_ctx'] == 4], key=lambda r: bool(json.loads(r['condition'])['ctx_init']))
    for ax, rows, title, labels in [
        (axes[0], random_rows, 'Context length (random initialization)',
         [f"{json.loads(r['condition'])['n_ctx']} tokens ({int(r['trainable_parameters']):,} params)" for r in random_rows]),
        (axes[1], init_rows, 'Initialization (4 context tokens)', ['Random', 'a photo of a'])]:
        for i, row in enumerate(rows):
            mean, std = float(row['mean_accuracy']), float(row['sample_std'])
            ax.errorbar(mean, i, xerr=std, fmt='o', color=colors['coop'], capsize=5, ms=7, lw=1.8)
            ax.text(89.7, i, f'{mean:.2f} ± {std:.2f}', va='center', ha='right', fontsize=9)
        ax.set_yticks(range(len(rows)), labels)
        ax.set(xlim=(75, 90), ylim=(-.65, len(rows)-.35), title=title, xlabel='Test top-1 accuracy (%)')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.18)
    fig.suptitle('CoOp sensitivity: EuroSAT, 16-shot', y=.96, fontsize=14)
    fig.text(.5, .035, 'Mean ± sample SD across three seeds. Accuracy axis is zoomed to 75–90%. No statistical significance claim.', ha='center', fontsize=9, color='#475569')
    fig.subplots_adjust(left=.19, right=.97, top=.77, bottom=.20, wspace=.58)
    finish(fig, 'coop_sensitivity')

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8))
    for ax, dataset in zip(axes, titles):
        rows = [next(r for r in baseline if r['dataset'] == dataset and r['method'] == m and r['shots'] == '16') for m in ['coop', 'tip_adapter']]
        vals = [float(r['mean_cold_adaptation_seconds']) for r in rows]
        ax.bar([0, 1], vals, color=[colors['coop'], colors['tip_adapter']], width=.55)
        ax.set_xticks([0, 1], ['CoOp', 'Tip-Adapter'])
        ax.set_yscale('log')
        ax.set_ylim(1, 1500)
        ax.set_title(titles[dataset])
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
        for i, value in enumerate(vals):
            ax.text(i, value*1.17, f'{value:.1f}s', ha='center', fontsize=10)
    axes[0].set_ylabel('Mean cold adaptation time (seconds; log scale)')
    fig.suptitle('Adaptation cost at 16 shots per class', y=.96, fontsize=14)
    fig.text(.5, .035, 'Includes original support/validation feature extraction, text encoding, training/selection. Excludes model load and test inference.', ha='center', fontsize=8.5, color='#475569')
    fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.21, wspace=.25)
    finish(fig, 'efficiency_16shot')

    # Aggregate confusion rates across fitted models, retaining the shared fixed test set.
    confusions = {}
    top_confusions = []
    for dataset in titles:
        for method in ['coop', 'tip_adapter']:
            selected = [o for o in raw['baseline'] if o['config']['dataset'] == dataset and o['config']['method'] == method and o['config']['shots'] == 16]
            matrix = None
            classes = None
            for obj in selected:
                parent = run_paths[id(obj)]
                c = json.loads((parent / 'confusion_r0.json').read_text())
                if classes is not None and classes != c['classes']:
                    raise ValueError('Confusion class orders disagree')
                classes = c['classes']
                m = np.asarray(c['rows_true_columns_pred'], dtype=float)
                predictions = read_csv(parent / 'predictions_r0.csv')
                score = 100*sum(r['label'] == r['prediction'] for r in predictions)/len(predictions)
                if not math.isclose(score, obj['accuracy']['0'], abs_tol=1e-4):
                    raise ValueError('Predictions disagree with recorded accuracy')
                if int(m.sum()) != len(predictions):
                    raise ValueError('Confusion matrix count differs from predictions')
                matrix = m if matrix is None else matrix + m
            rates = 100 * matrix / matrix.sum(axis=1, keepdims=True)
            confusions[(dataset, method)] = rates
            pairs = [(rates[i, j], i, j) for i in range(len(classes)) for j in range(len(classes)) if i != j]
            for rate, i, j in sorted(pairs, reverse=True)[:2]:
                top_confusions.append((titles[dataset], names[method], classes[i], classes[j], float(rate)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2))
    short = ['Annual crop', 'Forest', 'Herbaceous veg.', 'Highway/road', 'Industrial', 'Pasture', 'Permanent crop', 'Residential', 'River', 'Sea/lake']
    for ax, method in zip(axes, ['coop', 'tip_adapter']):
        rates = confusions[('eurosat', method)]
        im = ax.imshow(rates, vmin=0, vmax=100, cmap='Blues')
        ax.set_xticks(range(10), short, rotation=55, ha='right', fontsize=8)
        ax.set_yticks(range(10), short, fontsize=8)
        ax.set(title=names[method], xlabel='Predicted class', ylabel='True class')
        for i in range(10):
            for j in range(10):
                if i == j or rates[i, j] >= 15:
                    ax.text(j, i, f'{rates[i,j]:.0f}', ha='center', va='center', fontsize=8,
                            color='white' if rates[i,j] > 55 else '#172033')
    fig.subplots_adjust(left=.12, right=.87, top=.83, bottom=.27, wspace=.50)
    cax = fig.add_axes([.90, .30, .015, .48])
    fig.colorbar(im, cax=cax, label='Within-true-class prediction rate (%)')
    fig.suptitle('EuroSAT confusion patterns at 16 shots per class', y=.96, fontsize=14)
    fig.text(.5, .04, 'Mean confusion rates across seeds 1, 2, 3 on the same fixed test images. Diagonal and off-diagonal rates ≥15% are labeled.', ha='center', fontsize=9, color='#475569')
    finish(fig, 'eurosat_confusion_16shot')

    def formatted(row):
        return f"{float(row['mean_accuracy']):.2f}" + (f" ± {float(row['sample_std']):.2f}" if row['sample_std'] else '')

    lines = ['# Baseline results and team handoff', '',
        'Status: the main baseline matrix and planned CoOp sensitivity matrix are complete. This report summarizes project implementations, not strict reproduction of published benchmark numbers.', '',
        '## Evidence and scope', '',
        '- Main matrix: 93 completed runs, summarized as 33 conditions. Each few-shot condition contains seeds 1, 2, 3; zero-shot is evaluated once per dataset.',
        '- CoOp sensitivity: 12 completed runs, summarized as four conditions, all on EuroSAT 16-shot. The 16-token random condition repeats the main baseline setting; it is not an additional independent method.',
        '- Summary means/sample SDs were independently recomputed from raw metrics. Sample-manifest hashes match across CoOp and Tip-Adapter for all 45 dataset/budget/seed combinations.',
        '- At 16-shot, predictions and confusion counts were checked against recorded accuracy for all 18 adapted models.',
        '- Sources: `results/summary.csv`, `results/coop-sensitivity.csv`, and local raw metrics/predictions. Audit details: `results/analysis_checks.json`.', '',
        '## Accuracy versus training budget', '',
        'Accuracy is top-1 percent; ± is sample standard deviation across three sampling/training seeds. Zero-shot has no SD because it is evaluated once.', '',
        '![Baseline curves](../results/figures/baseline_accuracy.png)', '']
    for dataset in titles:
        lines += [f'### {titles[dataset]}', '', '| Shots/class | CoOp | Tip-Adapter | Zero-shot CLIP |', '| --- | --- | --- | --- |']
        zero = next(r for r in baseline if r['dataset'] == dataset and r['method'] == 'zero_shot')
        for shots in budgets:
            a, b = [next(r for r in baseline if r['dataset'] == dataset and r['method'] == m and int(r['shots']) == shots) for m in ['coop', 'tip_adapter']]
            lines.append(f'| {shots} | {formatted(a)} | {formatted(b)} | {formatted(zero)} |')
        lines.append('')
    lines += ['Zero-shot is repeated across rows only as a reference line; these are not five separate zero-shot experiments.', '',
        'CoOp has the highest observed mean accuracy at 16-shot on all three datasets. This does not mean it wins every low-shot condition: Tip-Adapter has higher means at EuroSAT 1-shot, DTD 2-shot, and Flowers102 1-/2-shot. These are descriptive comparisons, not significance claims.', '',
        '## Prompt sensitivity', '',
        '![CoOp sensitivity](../results/figures/coop_sensitivity.png)', '',
        '| Context tokens | Initialization | Accuracy (%) | Learned parameters |', '| --- | --- | --- | --- |']
    for row in random_rows + [r for r in init_rows if json.loads(r['condition'])['ctx_init']]:
        c = json.loads(row['condition'])
        lines.append(f"| {c['n_ctx']} | {c['ctx_init'] or 'Random'} | {formatted(row)} | {int(row['trainable_parameters']):,} |")
    lines += ['', 'The 8-token random condition has the highest observed mean, but also the largest observed SD. Accuracy does not increase monotonically with context length. At four tokens, word initialization has a slightly higher mean and lower observed SD than random initialization. With only three seeds, no statistical significance or universal stability claim is made. Do not change the main baseline setting based on these test scores.', '',
        '## Adaptation cost', '',
        '![Adaptation time](../results/figures/efficiency_16shot.png)', '',
        '| Dataset | CoOp 16-shot (s) | Tip-Adapter 16-shot (s) |', '| --- | --- | --- |']
    for dataset in titles:
        vals = [float(next(r for r in baseline if r['dataset'] == dataset and r['method'] == m and r['shots'] == '16')['mean_cold_adaptation_seconds']) for m in ['coop', 'tip_adapter']]
        lines.append(f'| {titles[dataset]} | {vals[0]:.2f} | {vals[1]:.2f} |')
    lines += ['', 'Timing includes original support/validation feature extraction, text encoding, and optimization/selection. It excludes model loading/downloading and test inference. Cached runs reuse original extraction durations; these are accounting estimates rather than independent cold-start timing replications. Use the same definition for teammate methods before comparing efficiency.', '',
        'CoOp uses 8,192 learned parameters with 16 context tokens; the 4-/8-token variants use 2,048/4,096. Tip-Adapter optimizes zero parameters but stores a labeled feature cache; zero learned parameters does not mean zero memory or zero adaptation cost.', '',
        '## Failure analysis: observations to inspect', '',
        '![EuroSAT confusion](../results/figures/eurosat_confusion_16shot.png)', '',
        'The following are the two largest off-diagonal confusion rates for each adapted method/dataset at 16-shot. Rates are averaged across three models evaluated on the same fixed test set; these are not three independent image datasets.', '',
        '| Dataset | Method | True class | Predicted class | Within-class error rate (%) |', '| --- | --- | --- | --- | --- |']
    for dataset, method, true, predicted, rate in top_confusions:
        lines.append(f'| {dataset} | {method} | {true} | {predicted} | {rate:.2f} |')
    lines += ['', 'Next, inspect representative images for these pairs in the saved predictions and retrieval JSON. Numerical confusion alone does not establish a semantic cause. Human image inspection and a short explanation are still required for the report.', '',
        '## Protocol and interpretation limits', '',
        '- EuroSAT: teammate 60/10/30 split, seed 2026; all 2,700 validation labels. DTD: official partition 1; four validation labels/class (188 total). Flowers102: custom stratified 50/20/30 split, seed 0; four validation labels/class (408 total).',
        '- Training budgets are k images/class, with validation supervision counted separately. The Flowers102 split supports 16-shot but is not its official 10-training-images/class split and is not the exact Zhou split.',
        '- Model: official OpenAI CLIP ViT-B/32; FP32, fixed official image preprocessing. CoOp uses deterministic cached image features and omits the full upstream augmentation/warmup recipe. Tip-Adapter uses one deterministic support view and is not Tip-Adapter-F.',
        '- Report trainable parameters, cache storage, timing definitions, hardware, validation budgets and upstream differences. See `PROTOCOL.md` and `UPSTREAM.md`.',
        '- ResNet-50, standard pixel prompting, multi-resolution pixel prompting and resolution-robustness evaluation are not supplied by these results. The proposed extension has not been demonstrated by this baseline matrix.',
        '- ChatGPT assisted code generation, checks, plots and descriptive reporting. Team members must review the implementation and write their own interpretation and final contribution disclosure.', '',
        '## Rebuild these figures', '',
        'Install optional dependencies from `requirements-report.txt` in an appropriate reporting environment, then run:', '',
        '```powershell', '.\\run.ps1 scripts/make_report.py', '```', '',
        'PNG files are suitable for slides and reports; SVG files preserve vector quality. Original experiment CSVs are not overwritten.']
    (docs / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    audit = {'baseline_conditions': 33, 'baseline_completed_runs': 93, 'sensitivity_conditions': 4,
             'sensitivity_completed_runs': 12, 'summary_means_and_sample_sd_match_raw_metrics': True,
             'cross_method_manifest_checks': len(shared), 'prediction_confusion_checks': 18,
             'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
             'figures': ['baseline_accuracy', 'coop_sensitivity', 'efficiency_16shot', 'eurosat_confusion_16shot'],
             'matplotlib_version': matplotlib.__version__}
    (output / 'results/analysis_checks.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))
    print(f'Report: {docs / "RESULTS.md"}')


if __name__ == '__main__':
    main()
