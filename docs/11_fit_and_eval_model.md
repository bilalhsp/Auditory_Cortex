## 🧮 Fitting and Evaluating the Model

This guide explains how to use this repository to evaluate pretrained DNNs as encoding models of auditory cortex.  
It outlines the pipeline for mapping DNN activations to neural responses and analyzing model performance.

Before following this guide, ensure the following are complete:

1. ✅ A feature extractor has been implemented for a pretrained DNN-based speech recognition model as per the [DNN feature extractor guide](./2_dnn_features.md)  
2. ✅ A neural dataset has been implemented and registered as per the [Neural Data guide](1_neural_data.md)

---

### 📈 Workflow Overview

Once both the DNN and neural dataset are integrated into the repository, this guide helps you:

- Fit linear models from hidden layer activations to neural responses.
- Save and aggregate correlation results.
- Use normalizer distributions to correct correlation scores.
- Identify tuned neurons and generate plots for model evaluation.

Since each of these steps can be computationally intensive, the repository allows you to execute them independently and save intermediate results.

---

### 🧩 Step-by-Step Instructions

#### 1. **Compute and Save Normalizer Distributions**

Before evaluating model fits, generate null distributions (e.g., from shuffled data or unrelated stimuli).  
These distributions help assess the significance of correlation values by acting as baselines.

#### 2. **Fit the Linear Model (Per Layer)**

For each layer of the DNN, fit a linear model (e.g., ridge regression) to predict neural responses from hidden activations.  
Save the correlation results independently for each layer to allow flexible later combination.

#### 3. **Aggregate Correlations and Add Normalizers**

Once all layers have been processed:

- Merge the correlation results into a single file.
- Attach normalizer statistics (e.g., percentiles, thresholds).
- Save the merged result for each DNN model.

This enables downstream analysis and comparisons across architectures or recording sites.

#### 4. **Analyze Results and Generate Plots**

Use built-in utilities (or write your own analysis scripts) to:

- Visualize correlation scores across layers.
- Identify significantly tuned neurons.
- Compare encoding performance across models or datasets.

---

Each of these steps is modular and scriptable, supporting large-scale experiments across multiple DNNs and datasets.

Let me know if you want a ready-to-use code scaffold for any of these stages!
