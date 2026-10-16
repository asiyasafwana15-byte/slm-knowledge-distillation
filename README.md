# Evaluating Knowledge Distillation for Resource-Efficient Small Language Models

A controlled comparison of supervised fine-tuning and knowledge distillation on SST-2 sentiment classification.
1. Project Overview

This project evaluates whether knowledge distillation improves the performance of a resource-efficient Small Language Model (SLM) compared with conventional supervised fine-tuning.

The study compares two training approaches using the same compact student architecture on the Stanford Sentiment Treebank (SST-2) binary sentiment classification task. It examines predictive performance, multi-seed robustness, prediction-error overlap, teacher–student agreement, and inference efficiency.

The project focuses on an evidence-based comparison rather than assuming that knowledge distillation necessarily outperforms conventional fine-tuning.

2. Research Question

Under a controlled experimental setup, how does teacher-guided knowledge distillation compare with conventional supervised fine-tuning when training a compact language model for sentiment classification?

3. Models and Dataset

| Component | Configuration |
|---|---|
| Dataset | SST-2 |
| Dataset source | Hugging Face GLUE |
| Dataset identifier | `nyu-mll/glue` |
| Dataset configuration | `sst2` |
| Teacher model | `textattack/bert-base-uncased-SST-2` |
| Student model | `google/bert_uncased_L-2_H-128_A-2` |
| Evaluation split | Validation |
| Validation examples | 872 |

The public test split was not used for metric calculation because usable ground-truth labels were unavailable in the evaluated dataset configuration.

4. Experimental Methodology

Two student-training approaches are compared:

- Supervised Fine-Tuning (SFT): The student is trained using ground-truth sentiment labels.
- Knowledge Distillation (KD): The student is trained using a combination of ground-truth classification loss and teacher-guided distillation loss.

The distillation objective combines cross-entropy loss with temperature-scaled Kullback–Leibler divergence.

The primary KD configuration uses a temperature of 2.0 and a distillation-loss weight of 0.5. Additional configurations use different temperature and loss-weight combinations.

The reported principal training settings are:

- Training epochs: 3
- Learning rate: "2e-5"
- Batch size: 16
- Maximum sequence length: 128
- Random seeds: 42, 43, and 44
- Evaluation metrics: Accuracy and F1-score

The multi-seed results are descriptive and are not presented as evidence of statistical significance.

5. Key Results

### 5.1 Three-Seed Comparison

| Metric | SFT Baseline | Knowledge Distillation |
|---|---:|---:|
| Mean accuracy | 82.49% | 82.11% |
| Mean F1-score | 82.79% | 82.87% |
| Accuracy standard deviation | 0.46 percentage points | 0.40 percentage points |
| F1-score standard deviation | 0.11 percentage points | 0.10 percentage points |

Across the three evaluated seeds, the SFT baseline achieved higher mean accuracy, while knowledge distillation achieved a slightly higher mean F1-score.

These results do not demonstrate a consistent overall performance advantage for knowledge distillation.
### 5.2 Model Size

| Model | Approximate Number of Parameters |
|---|---:|
| Teacher model | 109,483,778 |
| Student model | 4,386,178 |

The teacher model contains approximately 109.48 million parameters, whereas the student model contains approximately 4.39 million parameters. The student architecture therefore has approximately 95.99% fewer parameters than the teacher architecture. This difference reflects the model architectures and should not be interpreted as a parameter reduction caused by knowledge distillation compared with supervised fine-tuning of the same student model.
### 5.3 Inference Efficiency

The reported inference benchmark was conducted on an NVIDIA Tesla T4 GPU using a batch size of 16, ten warm-up iterations, and 50 measured iterations.

| Metric | SFT Baseline | Knowledge Distillation |
|---|---:|---:|
| Latency per batch | 2.58 ms | 3.19 ms |
| Throughput | 6,199.65 samples/second | 5,019.28 samples/second |
| Reported peak GPU memory | 29.34 MB | 29.34 MB |

Under the reported benchmark conditions, the baseline student had lower latency and higher throughput. These measurements do not establish that knowledge distillation makes inference faster.

6. Repository Structure

The repository contains the following experimental notebooks, arranged in the intended project sequence:

Day 1 — Baseline Training

"SLM_KD_Day1_Baseline.ipynb"

Establishes the conventional student-training baseline and its evaluation results.

Day 2 — Knowledge Distillation

"SLM_KD_Day2_Knowledge_Distillation.ipynb"

Documents the teacher-guided student-training approach.

Day 3 — Research Analysis

"SLM_KD_Day3_Reaserch_Analysis.ipynb"

Contains the comparative analysis of the experimental results.

Day 4 — Robustness Analysis

"SLM_KD_Day4_Robustness.ipynb"

Documents the multi-seed comparison and robustness analysis.

The notebooks are intended to document the workflow and results. Their execution requirements and dependencies should be checked within each notebook before reproducing the experiments.

7. Technologies

The project uses the following tools and libraries:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy
- scikit-learn
- Jupyter Notebooks

8. Limitations

The findings should be interpreted in the context of the following limitations:

- The evaluation focuses on one sentiment-classification dataset.
- Only one teacher–student model pair is considered.
- The evaluation uses the validation split because usable test labels were unavailable in the evaluated dataset configuration.
- The multi-seed comparison uses three random seeds and does not establish statistical significance.
- Temperature and distillation-loss weight vary together across the additional KD configurations, preventing their individual effects from being isolated.
- The inference benchmark covers the specified hardware and measurement procedure, not a comprehensive deployment or energy-efficiency evaluation.

9. Conclusion

In this experimental setting, knowledge distillation did not consistently outperform conventional supervised fine-tuning across the reported metrics.

The SFT baseline achieved higher mean accuracy and better measured inference throughput, whereas the KD model achieved a slightly higher mean F1-score.

The results highlight the importance of evaluating predictive performance and inference behaviour separately when studying knowledge distillation for resource-efficient language models.

10. Reproducibility

The notebooks provide the project’s experimental workflow and analysis. Reproducing the reported results requires checking the model configurations, data-processing steps, random seeds, library versions, and evaluation procedures recorded in the notebooks.

The dataset and pretrained models are referenced by their Hugging Face identifiers above. Their availability and access requirements may depend on the corresponding repositories.

11. Project Status

This repository documents an experimental learning and evaluation project on resource-efficient Small Language Models using knowledge distillation.

The reported findings are limited to the models, dataset, configurations, and evaluation procedures described in this repository.

12. Citation

If a citable version of the associated research paper becomes available, please use its published citation when referencing this work.

For questions or reproducibility discussions, please use the repository's GitHub Issues section.


