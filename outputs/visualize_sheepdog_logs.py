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
    """Parse the SheepDog log file and extract metrics for all sections."""
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Initialize dictionary to store all metrics
    metrics = {
        'original': {
            'acc': [], 'prec': [], 'rec': [], 'f1': [],
            'avg_acc': 0.0, 'avg_metrics': [0.0, 0.0, 0.0]
        },
        'adversarial_a': {
            'acc': [], 'prec': [], 'rec': [], 'f1': [],
            'avg_acc': 0.0, 'avg_metrics': [0.0, 0.0, 0.0]
        },
        'adversarial_b': {
            'acc': [], 'prec': [], 'rec': [], 'f1': [],
            'avg_acc': 0.0, 'avg_metrics': [0.0, 0.0, 0.0]
        },
        'adversarial_c': {
            'acc': [], 'prec': [], 'rec': [], 'f1': [],
            'avg_acc': 0.0, 'avg_metrics': [0.0, 0.0, 0.0]
        },
        'adversarial_d': {
            'acc': [], 'prec': [], 'rec': [], 'f1': [],
            'avg_acc': 0.0, 'avg_metrics': [0.0, 0.0, 0.0]
        },
        'adversarial_avg': {
            'acc': [], 'prec': [], 'rec': [], 'f1': [],
            'avg_acc': 0.0, 'avg_metrics': [0.0, 0.0, 0.0]
        }
    }
    
    # Define sections to parse
    sections = [
        ('original', '-------------Original-------------'),
        ('adversarial_a', '-------------Adversarial \\(A - Objective\\)------------'),
        ('adversarial_b', '-------------Adversarial \\(B - Neutral\\)------------'),
        ('adversarial_c', '-------------Adversarial \\(C - Emotionally Triggering\\)------------'),
        ('adversarial_d', '-------------Adversarial \\(D - Sensational\\)------------'),
        ('adversarial_avg', '-------------Adversarial \\(Average\\)------------')
    ]
    
    # Extract metrics for each section
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
    """Extract list of numbers from a line matching the pattern."""
    match = re.search(pattern + r'\[(.*?)\]', text)
    if match:
        numbers_str = match.group(1)
        return [float(x.strip()) for x in numbers_str.split(',')]
    return []

def extract_avg_from_line(text, pattern):
    """Extract a single average value (e.g., accuracy) from a line."""
    match = re.search(pattern + r'\s+([\d.]+)', text)
    if match:
        return float(match.group(1))
    return 0.0

def extract_avg_metrics_from_line(text, pattern):
    """Extract average precision, recall, F1 from a line."""
    match = re.search(pattern + r'\s+([\d.]+),\s+([\d.]+),\s+([\d.]+)', text)
    if match:
        return [float(match.group(1)), float(match.group(2)), float(match.group(3))]
    match = re.search(r'Average Prec / Rec / F1 \(macro\):\s+([\d.]+),\s+([\d.]+),\s+([\d.]+)', text)
    if match:
        return [float(match.group(1)), float(match.group(2)), float(match.group(3))]
    return [0.0, 0.0, 0.0]

def create_visualizations(data, output_dir):
    """Create various visualizations of the metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # Define categories and colors
    categories = [
        ('Original', 'original', 'blue'),
        ('Adv. A (Objective)', 'adversarial_a', 'orange'),
        ('Adv. B (Neutral)', 'adversarial_b', 'green'),
        ('Adv. C (Emotional)', 'adversarial_c', 'red'),
        ('Adv. D (Sensational)', 'adversarial_d', 'purple'),
        ('Adv. Average', 'adversarial_avg', 'black')
    ]
    
    # 1. Iteration-wise performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SheepDog Performance Across Test Sets', fontsize=16, fontweight='bold')
    
    iterations = list(range(1, len(data['original']['acc']) + 1))
    
    def safe_plot(ax, x, y, *args, **kwargs):
        if len(x) == len(y) and len(x) > 0:
            ax.plot(x, y, *args, **kwargs)
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes)
    
    # Plot for each metric
    for label, key, color in categories:
        safe_plot(axes[0, 0], iterations, data[key]['acc'], '-o', label=label, linewidth=2, markersize=6, color=color)
        safe_plot(axes[0, 1], iterations, data[key]['prec'], '-o', label=label, linewidth=2, markersize=6, color=color)
        safe_plot(axes[1, 0], iterations, data[key]['rec'], '-o', label=label, linewidth=2, markersize=6, color=color)
        safe_plot(axes[1, 1], iterations, data[key]['f1'], '-o', label=label, linewidth=2, markersize=6, color=color)
    
    # Configure subplots
    axes[0, 0].set_title('Accuracy Across Iterations', fontweight='bold')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Precision Across Iterations', fontweight='bold')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Recall Across Iterations', fontweight='bold')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('F1 Score Across Iterations', fontweight='bold')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Average performance comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, (label, key, color) in enumerate(categories):
        values = [data[key]['avg_acc']] + data[key]['avg_metrics']
        bars = ax.bar(x + i*width - width*2.5, values, width, label=label, color=color, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
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
    
    # 3. Performance degradation analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    def safe_degradation(orig, adv):
        return ((orig - adv) / orig * 100) if orig != 0 else 0
    
    orig_avg_acc = data['original']['avg_acc']
    orig_avg_metrics = data['original']['avg_metrics']
    
    for i, (label, key, color) in enumerate(categories[1:]):  # Skip Original
        degradation = {
            'Accuracy': safe_degradation(orig_avg_acc, data[key]['avg_acc']),
            'Precision': safe_degradation(orig_avg_metrics[0], data[key]['avg_metrics'][0]),
            'Recall': safe_degradation(orig_avg_metrics[1], data[key]['avg_metrics'][1]),
            'F1 Score': safe_degradation(orig_avg_metrics[2], data[key]['avg_metrics'][2])
        }
        bars = ax.bar([x + i*width - width*1.5 for x in range(len(metrics))], 
                     list(degradation.values()), 
                     width, 
                     label=label, 
                     color=color, 
                     alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height > 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    ax.set_ylabel('Performance Degradation (%)')
    ax.set_title('Performance Degradation Under Adversarial Attacks', fontweight='bold')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_degradation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Box plot for distribution analysis
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
    
    print(f"Visualizations saved to {output_dir}")

def process_all_logs(logs_dir, output_dir):
    logs_dir = Path(logs_dir)
    output_dir = Path(output_dir)
    iter_files = sorted(logs_dir.glob("*.iter*"))
    if not iter_files:
        print(f"No .iter* files found in {logs_dir}")
        return
    for log_file in iter_files:
        print(f"\nProcessing log file: {log_file}")
        data = parse_log_file(log_file)
        sub_output_dir = output_dir / log_file.stem
        os.makedirs(sub_output_dir, exist_ok=True)
        create_visualizations(data, sub_output_dir)
        print(f"Visualizations saved to {sub_output_dir}")

def main():
    # Set up paths
    base_dir = Path("/teamspace/studios/this_studio/SheepDog")
    logs_dir = base_dir / "logs"
    output_dir = base_dir / "outputs"

    # Create outputs directory
    os.makedirs(output_dir, exist_ok=True)

    # Process all log files matching *.iter*
    iter_files = sorted(logs_dir.glob("*.iter*"))
    if not iter_files:
        print(f"No .iter* files found in {logs_dir}")
        return

    for log_file in iter_files:
        print(f"\nParsing log file: {log_file}")
        data = parse_log_file(log_file)
        sub_output_dir = output_dir / log_file.stem
        os.makedirs(sub_output_dir, exist_ok=True)
        print("Creating visualizations...")
        create_visualizations(data, sub_output_dir)

        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        for label, key, _ in categories:
            print(f"\n{label} Test Set:")
            print(f"  Average Accuracy: {data[key]['avg_acc']:.4f}")
            print(f"  Average Precision: {data[key]['avg_metrics'][0]:.4f}")
            print(f"  Average Recall: {data[key]['avg_metrics'][1]:.4f}")
            print(f"  Average F1 Score: {data[key]['avg_metrics'][2]:.4f}")

        print(f"\nPerformance Degradation (Relative to Original):")
        def safe_degradation(orig, adv):
            return ((orig - adv) / orig * 100) if orig != 0 else 0
        orig_avg_acc = data['original']['avg_acc']
        orig_f1 = data['original']['avg_metrics'][2]
        for label, key, _ in categories[1:]:  # Skip Original
            acc_deg = safe_degradation(orig_avg_acc, data[key]['avg_acc'])
            f1_deg = safe_degradation(orig_f1, data[key]['avg_metrics'][2])
            print(f"  {label}:")
            print(f"    Accuracy: {acc_deg:.1f}%")
            print(f"    F1 Score: {f1_deg:.1f}%")

if __name__ == "__main__":
    main()