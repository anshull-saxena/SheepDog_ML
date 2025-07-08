import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import re

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

def create_comparison_visualizations(all_data, log_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8')

    model_colors = ['#1f77b4', '#ff7f0e']

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for metric_idx, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        width = 0.35

        for i, (data, log_name) in enumerate(zip(all_data, log_names)):
            if metric == 'Accuracy':
                values = [data[key]['avg_acc'] for _, key, _ in categories]
            else:
                metric_map = {'Precision': 0, 'Recall': 1, 'F1 Score': 2}
                values = [data[key]['avg_metrics'][metric_map[metric]] for _, key, _ in categories]
            
            bars = ax.bar(x + i * width - width/2, values, width, 
                         label=log_name, color=model_colors[i], alpha=0.8)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Test Set Category')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison Between Models', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _, _ in categories], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.lower()}_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    for i, (data, log_name) in enumerate(zip(all_data, log_names)):
        orig_acc = data['original']['avg_acc']
        orig_prec, orig_rec, orig_f1 = data['original']['avg_metrics']
        
        key = 'adversarial_avg'
        degradation = [
            ((orig_acc - data[key]['avg_acc']) / orig_acc * 100),
            ((orig_prec - data[key]['avg_metrics'][0]) / orig_prec * 100),
            ((orig_rec - data[key]['avg_metrics'][1]) / orig_rec * 100),
            ((orig_f1 - data[key]['avg_metrics'][2]) / orig_f1 * 100)
        ]
        
        bars = ax.bar(x + i * width - width/2, degradation, width, 
                     label=f'{log_name}: Adv. Average', color=model_colors[i], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Performance Degradation (%)')
    ax.set_title('Performance Degradation (Adversarial Average)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'degradation_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        num_logs = int(input("Enter the number of log files to compare (must be 2): "))
        if num_logs != 2:
            print("This script is designed to compare exactly 2 log files")
            return
    except ValueError:
        print("Please enter a valid number")
        return

    log_paths = []
    for i in range(num_logs):
        path = input(f"Enter path to log file {i+1}: ").strip()
        if not os.path.exists(path):
            print(f"Error: File {path} does not exist")
            return
        log_paths.append(path)

    all_data = []
    log_names = []
    for path in log_paths:
        try:
            data = parse_log_file(path)
            all_data.append(data)
            log_names.append(Path(path).stem)
        except Exception as e:
            print(f"Error parsing {path}: {str(e)}")
            return

    output_dir = Path("SheepDog/outputs/comparsion_graphs")
    os.makedirs(output_dir, exist_ok=True)

    print("Creating comparison visualizations...")
    create_comparison_visualizations(all_data, log_names, output_dir)
    print(f"Comparison graphs saved in {output_dir}")

if __name__ == "__main__":
    main()