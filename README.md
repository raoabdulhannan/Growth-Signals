# Growth Signals x CMU Capstone: Sparse Autoencoder

[![Commit activity](https://img.shields.io/github/commit-activity/m/raoabdulhannan/Growth-Signals)](https://img.shields.io/github/commit-activity/m/raoabdulhannan/Growth-Signals)
[![License](https://img.shields.io/github/license/raoabdulhannan/Growth-Signals)](https://img.shields.io/github/license/raoabdulhannan/Growth-Signals)

This project trains a Sparse Autoencoder (SAE) on paragraph-level embeddings from the Cohere Wikipedia 22-12 dataset. The goal is to extract interpretable latent features that correspond to distinct semantic concepts. These features can be automatically interpreted using a large language model (LLM). The repository includes modules for training, hyperparameter tuning, visualization, and model interpretation.

## Features

- Sparse Autoencoder implemented in PyTorch with L1 or reinforcement learning-based sparsity
- Automatic neuron interpretation using max and zero-activating examples with an LLM
- Visualization tools for inspecting latent dimensions, activation patterns, and similarities
- Interactive exploratory data analysis dashboard using Streamlit
- Hyperparameter tuning via Bayesian optimization
- Uses 768-dimensional Cohere multilingual-22-12 paragraph embeddings

## Project Structure

| File | Description |
|------|-------------|
| `growth_signals/sae.py` | SAE architecture and sparsity-based loss/reward functions |
| `growth_signals/training.py` | Full training loop with checkpointing and interpretability data export |
| `growth_signals/llm_score.py` | Language model-based interpretation and prediction of neuron behavior |
| `growth_signals/plot_latent_space.py` | Visualizations of re-ranked latent vectors and heatmaps |
| `growth_signals/custom_dataset.py` | Loads and batches data from the Cohere Wikipedia embeddings |
| `growth_signals/constants.py` | Global hyperparameter definitions |
| `growth_signals/hyperparameter_tuning.py` | Hyperparameter optimization using `skopt` |
| `growth_signals/dashboard.py` | Streamlit-based interactive exploration of Wikipedia embedding space |

## Training

To train the SAE:

```bash
python training.py
```

This script:
- Loads the dataset (default: 100,000 samples)
- Trains the model for 10 epochs
- Applies either L1 or reward-based sparsity depending on the USE_REWARD flag
- Saves interpretability data (top-k, zero-k, and random-k activations)

## Interpretation with LLM

To generate and evaluate interpretations of the learned features:

```bash
python llm_score.py --sae_data_dir ./sae_data --model_name meta-llama/Llama-2-7b-chat
```

This will:
- Load top-activating and zero-activating examples for each neuron
- Prompt an LLM to extract human-readable interpretations
- Evaluate interpretability quality via F1 and correlation metrics

## Visualization

- `plot_latent_space.py` generates re-ranked vector plots and activation similarity heatmaps
- `dashboard.py` provides a browser-based interface for exploratory analysis using PCA, t-SNE, UMAP, and clustering

To launch the dashboard:

```bash
streamlit run dashboard.py
```

## Hyperparameter Tuning

To run Bayesian optimization for tuning learning rate, sparsity strength, hidden size, and decoder activation:

```bash
python hyperparameter_tuning.py
```

This uses skopt.gp_minimize to identify the best configuration based on reconstruction loss.

## Dependencies

Required packages include:
- torch, transformers, datasets, numpy, matplotlib, scikit-learn, streamlit, scipy, plotly, skopt, umap-learn, tqdm
- Optional: visdom for live training visualizations

## License

This project is released under the MIT License. See LICENSE for details.



---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
