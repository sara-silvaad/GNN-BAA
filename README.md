# GNN-BAA

This repository contains the codebase developed for the Final Year Project by Sara Silva and Martín Schmidt: **"Clasificación de Conectomas Basado
en el Análisis Mediante Redes Neuronales en Grafos."**

## Environment Setup

To ensure compatibility, it is recommended to create the Anaconda environment provided in the `environment.yml` file:

```bash
conda env create -f environment.yml

```

## Repository Structure

* **GCL:** Implementation of the Contrastive Learning framework.
* **GRL_FC_TEMPORAL:** Code for Functional Connectivity (FC) classification focused on predicting subsequent time points.
* **GRL_pytorch:** Contains the Encoder-Decoder and Classifier-Encoder architectures used for baselines and signal analysis, as well as Fully Connected (FC) networks.
* **hcp-download-script:** Utilities for downloading functional data from the Human Connectome Project (HCP), processing them into time series, and constructing correlation matrices.

## Data Access and AWS Configuration

Due to data privacy restrictions, the `data` folder in this repository is empty. To utilize the download and processing scripts, users must:

1. Register and request access at [ConnectomeDB](https://db.humanconnectome.org/).
2. Configure AWS credentials within your environment. Since AWS CLI is included in the provided environment, you must set your `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` according to the credentials provided by ConnectomeDB.

## Comprehensive Research Scope

This repository serves as a complete archive of all experimentation and code developed during the thesis project. For the specific implementation and refined code associated with our published paper, please refer to the dedicated repository: [Connectome_GCL](https://github.com/sara-silvaad/Connectome_GCL).

## Citation

If you find this work useful for your research, please cite our paper:

> Silva, S., Schmidt, M. (2026). *Connectome Classification Based on Graph Neural Network Analysis.* [Include Journal/Conference Name if applicable].
