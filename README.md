<img src="https://github.com/user-attachments/assets/6857cfe5-8b43-4a7c-aeea-e1e55eb04c73" width="600"  text-align="center">

# ProtRL: Direct Preference Optimization for Protein Language Models

This is the repository for the paper [*Guiding Generative Protein Language Models with Reinforcement Learning*](https://arxiv.org/abs/2412.12979). ProtRL is a Reinforcement Learning (RL) framework for autoregressive protein Language Models (pLMs). In this repository, you will find the scripts used for the experiments found on the paper (`Experiments`) and a basic implementation of ProtRL ready to work with ZymCTRL and optimize sequences to a desired length. This implementation is minimal and easily amenable to other pLMs and custom reward functions. We futher provide the application of DPO for ZymCTRL, ProtGPT2 and REXzyme

### Table of Content
- [About ProtRL](#about-dpo_plm)
- [Installation](#installation)
- [Example](#example)
- [General Usage](#generalusage)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Citation](#citation)

## About ProtRL

`DPO_pLM` is the main Python script for our project. It supports the following flags:

- `--beta`: Specify the beta value.
- `--model_directory`: Specify the local model or the HF name (e.g, AI4PD/ZymCTRL or /your/path/ZymCTRL)
- `--mode`: Choose the mode for experiments. Available options:
  - `paired`: the loss function will take in an ordered pair of sequence responses with different rewards and train the model to give preference to the sequence with the better reward score. 
  - `ranked`: the loss function will take in an ordered set of sequences responses by their reward score and train the model to give preference to the sequences in the ranking order.
  - `weighted`: the loss function will take in a set of sequences with their corresponding reward score and will train the model’s distribution of likelihood over the sequences to match the relative distribution of reward of the sequences derived from the softmax of their scalar labels.

This 3 different loss functions were adapted from the firsts described in [Widatalla et al., 2024](https://www.biorxiv.org/content/10.1101/2024.05.20.595026v1.abstract). You can find detailed explanations for each loss function and its changes in formulation in the Methods section of the [paper](). We recommend using the weighted mode, as it has been extensively tested during our experiments and has demonstrated superior performance in most cases.

Note: Weights are treated as "the higher, the better." If your scoring function is designed to be minimized, please multiply it by -1 before adding it to the weights.

## Installation

The software needed to run ProtRL can be found in `requeriments.txt`. To set up the environment, execute the following command inside your desired working environment:

```bash
git clone https://github.com/AI4PDLab/ProtRL.git
cd ProtRL
pip install -r requirements.txt
```
This work has been developed and tested on Python 3.10.4.

## Example 

ProtRL is reported as a very simple script with the objective of decreasing the length over the different iterations to reach a length of 60 amino acids. In the `Experiments` folder, you can find the scripts for experiments that implement more complex scoring functions such as protein folds, functional annotation of enzymes, and experimental data. If you are interested in optimizing for other protein features, you can use `DPO_pLM.py` as a template for your custom RL experiments.

First of all, you will need to set up ZymCTRL or the pLM of your choice. In our case, we downloaded the [HuggingFace's ZymCTRL](https://huggingface.co/AI4PD/ZymCTRL) repository locally or used it directly from the repo, taking advantage of Huggingface's `transformers` API (AI4PD/ZymCTRL). 

With this simple task, we observe that the three modes achieve the desired goal within just a few iterations. While the paired and ranked modes reach the objectives more quickly, they are more prone to catastrophic forgetting compared to the weighted mode. The weighted mode proves to be more stable, particularly in low-data scenarios. It is likely that, with a more complex scoring function and additional data, the ranked and paired algorithms could demonstrate improved performance and behavior.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/e7e89f70-dad0-4731-afbf-37970718e92a" />


To reproduce the experiments of our paper, you can find all the scripts in the `Experiments` folder. Given the size and computational needs of pLMs, each one of the experiments were executed in one H100 GPU, with differing times of execution. All the parameters and external data used in the experiments can be found in this repo. The `.sh` scripts can be executed from the same folder to conduct each experiment, they have been built to work on a SLURM based cluster, given the need of GPU-intensive computing. To reproduce the results run: 

```bash
bash experiment_name.sh
```
or 
```bash 
sbatch experiment_name.sh
```
Replace `experiment_name` with the desired experiment script path. Each experiment will produce, fold and calculate statistics for each considered feature.

## General Usage
To reinforce your desired feature, you can define and compute a custom reward function within following these steps:

  1. Add Your Custom Functions: Create your own reward function tailored to the feature you want to optimize.
  2. Calculate the Reward: Use your custom function to compute the reward based on your criteria.
  3. Update the DPO weight: Add the computed reward to the data["weights"] column.

Note: Ensure the correct sign of the reward based on your optimization goal: 
  - Use positive values to maximize the scored value.
  - Use negative values to minimize the scored value.
    
In case your are planning to use CLEAN, you will need to clone and set it up as explained in the official [CLEAN repository](https://github.com/tttianhao/CLEAN), and indicate the path in your code. 

## Troubleshooting

Please take a look at the documentation for more details on how to configure and run your experiments.

Feel free to contribute or raise issues if you encounter any problems! We are working to make it more accessible and detailed
## Work in Progress

We are currently working on a more user-friendly version (HF implementation). Additionaly we are working on a LoRA for more computing efficency and a revised form of the ranked and paired form. Stay tuned!
## References

- ESM1v: "Language models enable zero-shot prediction of the effects of mutations on protein function" Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, Alexander Rives; doi: https://doi.org/10.1101/2021.07.09.450648. Computed using https://github.com/seanrjohnson/protein_gibbs_sampler/
- ProteinMPNN: "Robust deep learning–based protein sequence design using ProteinMPNN", J. Dauparas et al. Science378,49-56(2022).DOI:10.1126/science.add2187
- CLEAN: "Enzyme function prediction using contrastive learning". Science379,1358-1363(2023). DOI:10.1126/science.adf2465, GitHub: "https://github.com/tttianhao/CLEAN?tab=readme-ov-file"

## Citation 

If you use ProtRL, please cite our [preprint](https://arxiv.org/abs/2412.12979):

```
@misc{stocco2024guidinggenerativeproteinlanguage,
      title={Guiding Generative Protein Language Models with Reinforcement Learning}, 
      author={Filippo Stocco and Maria Artigues-Lleixa and Andrea Hunklinger and Talal Widatalla and Marc Guell and Noelia Ferruz},
      year={2024},
      eprint={2412.12979},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2412.12979}, 
}
```

 


