# SheepDog_Improved

## Overview

SheepDog is a robust fake news detection system that leverages transformer-based models to identify misinformation across different news styles. The project tackles the challenge of detecting fake news that may be reframed in various writing styles to evade detection.

The system uses RoBERTa models fine-tuned on multiple datasets and evaluates performance across adversarial test sets to assess robustness against stylistic variations of news content.

This project was developed under **BITS F464: Machine Learning** at the **Birla Institute of Technology & Science, Pilani**.

## Features

- Fake news detection with both binary and 4-class classification
- Evaluation against adversarial reframing styles:
  - Objective
  - Neutral
  - Emotionally Triggering
  - Sensational
- Support for multiple datasets (Politifact, GossipCop, LUN)
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1)
- Visualization tools for model performance analysis
- Headline classification utility

## Project Structure

    SheepDog/
    ├── checkpoints/           # Trained model checkpoints
    ├── data/
    │   ├── news_articles/     # Original news datasets
    │   ├── adversarial_test/  # Adversarial test sets
    │   ├── reframings/        # Training data reframings
    │   └── veracity_attributions/ # Veracity labels for reframed content
    ├── logs/                  # Training logs with evaluation metrics
    ├── outputs/               # Visualization outputs
    ├── src/
    │   ├── sheepdog.py        # Main model implementation
    │   ├── sheepdog_original.py # Original baseline model
    │   └── infer_headline.py  # Inference script for headline classification
    ├── utils/
    │   └── load_data.py       # Data loading utilities
    └── train.sh              # Training script

## Installation

1. Clone the repository:

    git clone https://github.com/your-username/SheepDog.git
    cd SheepDog

2. Install the required dependencies:

    pip install torch transformers numpy sklearn tqdm matplotlib seaborn pandas

3. Make sure the model checkpoints are properly downloaded (they use Git LFS):

    git lfs pull

## Usage

### Training Models

To train the SheepDog model:

    sh train.sh

You can modify training parameters in the script or pass them as arguments:

    python src/sheepdog.py --dataset_name politifact --batch_size 4 --n_epochs 5 --iters 10

Available dataset options:
- `politifact`: Political news from PolitiFact
- `gossipcop`: Entertainment news from GossipCop
- `lun`: Liar Unlabeled News dataset

### Evaluating Models

The training process automatically evaluates models on the original test set and all adversarial test sets. Results are saved in the `logs/` directory.

To visualize the results:

    cd outputs
    python visualize_sheepdog_logs.py

This generates visualizations showing model performance across different test sets and iterations.

### Classifying Headlines

To classify a single news headline:

    cd src
    python infer_headline.py

The script will prompt you to enter a headline and will output both the 4-class and binary predictions.

## Model Details

SheepDog uses a RoBERTa-based architecture with:
- RobertaClassifier with dual classification heads:
  - 4-class classification (categories of fake/real news)
  - Binary classification (fake vs. real)
- Maximum sequence length of 512 tokens
- Fine-tuned on augmented datasets with varied writing styles
- Evaluation across multiple adversarial test sets

## Results

The model achieves robust performance across different datasets:
- LUN dataset: ~92.8% accuracy on original test set
- Politifact dataset: ~79.3% accuracy on original test set
- GossipCop dataset: ~76.1% accuracy on original test set

Performance on adversarial test sets shows the model's resilience to stylistic variations, with detailed metrics available in the log files.

## Future Updates

The following improvements are planned for future releases:

### Learning Rate Warmup
**What**: Add a warmup period to the learning rate schedule in the `train_model` function, where the learning rate gradually increases from a small value (e.g., 0 to 2e-5) over the first 10% of training steps, using the existing `get_linear_schedule_with_warmup` from the transformers library.

**Why**: Warmup stabilizes training in the early stages, especially for transformer models like RoBERTa, preventing large gradient updates that could disrupt learning. This can improve convergence and F1 scores, particularly for GossipCop, which has lower baseline performance (~74.45), without changing the model architecture.

### Data Augmentation: Synonym Swap
**What**: In addition to the LLM reframings, randomly pick 1–2 nouns or adjectives in each article and replace them with synonyms via WordNet.

**Why**: This simple "noising" teaches the model to ignore exact keywords and focus on meaning, improving generalization and robustness.

## Contributors

- 
-
-
-
-
-


## Acknowledgments

This project was developed as part of **BITS F464: Machine Learning** course at **Birla Institute of Technology & Science, Pilani**.

## License

