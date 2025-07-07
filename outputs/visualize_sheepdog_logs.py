import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path

categories = [
    ('Original', 'original', 'blue'),
    ('Adv. A (Objective)', 'adversarial_a', 'orange'),
    ('Adv. B (Neutral)', 'adversarial_b', 'green'),
    ('Adv. C (Emotional)', 'adversarial_c', 'red'),
    ('Adv. D (Sensational)', 'adversarial_d', 'purple'),
    ('Adv. Average', 'adversarial_avg', 'black')
]

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as f:
        content = f.read()

    metrics = {key: {'acc': [], 'prec': [], 'rec': [], 'f1': [],
                     'avg_acc': 0.0, 'avg_metrics': [0.0, 0.0, 0.0]}
               for _, key, _ in categories}

    sections = [
        ('original', '-------------Original-------------'),
        ('adversarial_a', '-------------Adversarial \\(A - Objective\\)------------'),
        ('adversarial_b', '-------------Adversarial \\(B - Neutral\\)------------'),
        ('adversarial_c', '-------------Adversarial \\(C - Emotionally Triggering\\)------------'),
        ('adversarial_d', '-------------Adversarial \\(D - Sensational\\)------------'),
        ('adversarial_avg', '-------------Adversarial \\(Average\\)------------')
    ]

    for section_key, section_pattern in sections:
        section = re.search(f'{section_pattern}\n(.*?)(?=\n\n|\n---------|$)', content, re.DOTALL)
        if section:
            section_text = section.group(1)
            metrics[section_key]['acc'] = extract_list_from_line(section_text, 'All Acc.s:')
            metrics[section_key]['prec'] = extract_list_from_line(section_text, 'All Prec.s:')
            metrics[section_key]['rec'] = extract_list_from_line(section_text, 'All Rec.s:')
            metrics[section_key]['f1'] = extract_list_from_line(section_text, 'All F1.s:')
            metrics[section_key]['avg_acc'] = extract_avg_from_line(section_text, 'Average acc.:')
            metrics[section_key]['avg_metrics'] = extract_avg_metrics_from_line(section_text, 'Average Prec / Rec / F1 \\(macro\\):')

    return metrics

def extract_list_from_line(text, pattern):
    match = re.search(pattern + r'\[(.*?)\]', text)
    if match:
        return [float(x.strip()) for x in match.group(1).split(',')]
    return []

def extract_avg_from_line(text, pattern):
    match = re.search(pattern + r'\s+([\d.]+)', text)
    return float(match.group(1)) if match else 0.0

def extract_avg_metrics_from_line(text, pattern):
    match = re.search(pattern + r'\s+([\d.]+),\s+([\d.]+),\s+([\d.]+)', text)
    return [float(match.group(i)) for i in range(1, 4)] if match else [0.0, 0.0, 0.0]

def print_formatted_metrics(data):
    section_titles = {
        'original': 'Original',
        'adversarial_a': 'Adversarial (A - Objective)',
        'adversarial_b': 'Adversarial (B - Neutral)',
        'adversarial_c': 'Adversarial (C - Emotionally Triggering)',
        'adversarial_d': 'Adversarial (D - Sensational)',
        'adversarial_avg': 'Adversarial (Average)'
    }

    for key in section_titles:
        section = section_titles[key]
        accs = data[key]['acc']
        precs = data[key]['prec']
        recs = data[key]['rec']
        f1s = data[key]['f1']
        avg_acc = data[key]['avg_acc']
        avg_prec, avg_rec, avg_f1 = data[key]['avg_metrics']

        print(f"-------------{section}------------")
        print(f"All Acc.s:[{', '.join(f'{a:.16f}' for a in accs)}]")
        print(f"All Prec.s:[{', '.join(f'{p:.16f}' for p in precs)}]")
        print(f"All Rec.s:[{', '.join(f'{r:.16f}' for r in recs)}]")
        print(f"All F1.s:[{', '.join(f'{f:.16f}' for f in f1s)}]")
        print(f"Average acc.: {avg_acc:.16f}")
        print(f"Average Prec / Rec / F1 (macro): {avg_prec:.16f}, {avg_rec:.16f}, {avg_f1:.16f}\n")

def create_visualizations(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SheepDog Performance Across Test Sets', fontsize=16, fontweight='bold')

    iterations = list(range(1, len(data['original']['acc']) + 1))

    def safe_plot(ax, x, y, *args, **kwargs):
        if len(x) == len(y) and len(x) > 0:
            return ax.plot(x, y, *args, **kwargs)
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes)
            return []

    for label, key, color in categories:
        safe_plot(axes[0, 0], iterations, data[key]['acc'], '-o', label=label, color=color)
        safe_plot(axes[0, 1], iterations, data[key]['prec'], '-o', label=label, color=color)
        safe_plot(axes[1, 0], iterations, data[key]['rec'], '-o', label=label, color=color)
        safe_plot(axes[1, 1], iterations, data[key]['f1'], '-o', label=label, color=color)

    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()

    axes[0, 0].set_title('Accuracy Across Iterations', fontweight='bold')
    axes[0, 1].set_title('Precision Across Iterations', fontweight='bold')
    axes[1, 0].set_title('Recall Across Iterations', fontweight='bold')
    axes[1, 1].set_title('F1 Score Across Iterations', fontweight='bold')

    for ax in axes.flat:
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ax.get_title().split()[0])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Average performance bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(metrics))
    width = 0.15

    for i, (label, key, color) in enumerate(categories):
        values = [data[key]['avg_acc']] + data[key]['avg_metrics']
        ax.bar(x + i*width - width*2.5, values, width, label=label, color=color)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Average Performance Across Test Sets', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Degradation chart
    fig, ax = plt.subplots(figsize=(12, 6))
    orig_acc = data['original']['avg_acc']
    orig_prec, orig_rec, orig_f1 = data['original']['avg_metrics']

    for i, (label, key, color) in enumerate(categories[1:]):
        degradation = [
            ((orig_acc - data[key]['avg_acc']) / orig_acc * 100),
            ((orig_prec - data[key]['avg_metrics'][0]) / orig_prec * 100),
            ((orig_rec - data[key]['avg_metrics'][1]) / orig_rec * 100),
            ((orig_f1 - data[key]['avg_metrics'][2]) / orig_f1 * 100)
        ]
        ax.bar([x + i*width - width*1.5 for x in range(4)], degradation, width, label=label, color=color)

    ax.set_ylabel('Performance Degradation (%)')
    ax.set_title('Performance Degradation Under Adversarial Attacks', fontweight='bold')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_degradation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Box plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Distribution Across Iterations', fontsize=16, fontweight='bold')

    for i, (metric, title, ylabel) in enumerate([
        ('acc', 'Accuracy Distribution', 'Accuracy'),
        ('prec', 'Precision Distribution', 'Precision'),
        ('rec', 'Recall Distribution', 'Recall'),
        ('f1', 'F1 Score Distribution', 'F1 Score')
    ]):
        ax = axes[i//2, i%2]
        box_data = [data[key][metric] for _, key, _ in categories]
        labels = [label for label, _, _ in categories]
        ax.boxplot(box_data, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='blue'),
                   medianprops=dict(color='red'))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    base_dir = Path("/teamspace/studios/this_studio/SheepDog")
    logs_dir = base_dir / "logs"
    output_dir = base_dir / "outputs"
    os.makedirs(output_dir, exist_ok=True)

    iter_files = sorted(logs_dir.glob("*.iter*"))
    if not iter_files:
        print(f"No .iter* files found in {logs_dir}")
        return

    for log_file in iter_files:
        print(f"\nParsing log file: {log_file}")
        data = parse_log_file(log_file)

        sub_output_dir = output_dir / log_file.stem
        os.makedirs(sub_output_dir, exist_ok=True)

        print("\nFormatted Metrics Output:\n")
        print_formatted_metrics(data)

        print("Creating visualizations...")
        create_visualizations(data, sub_output_dir)

if __name__ == "__main__":
    main()
