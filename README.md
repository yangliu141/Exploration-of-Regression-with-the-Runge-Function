# FYS-STK3155_Project1
---

## Description

This project explores the Runge function using polynomial regression with OLS, Ridge and Lasso. We compare colse-form solutions with gradient descent optimisers (Momentum, ADAgrad, ADAM and RMSprop) to study performance across model complexities. We use MSE and R2 for evaluating model accuracy, while bootstrapping and cross-validation are applied to analyse the bias-variance trade-off. The results highlight how regularisation and optimisation methods help reduce overfitting in high-degree polynomials.

Here you will find instructions on how to clone, install and run neccessary files, a structured overview of the folders and files used, as well as in-depth descriptions on some of the specific elements. 


## Table of Contents
- [Project prerequisites](#project-prerequisites)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Description of Notebooks](#description-of-notebooks)
- [Workflow](#workflow)

## Project Prerequisites
Before installation, make sure you have:
- Python 3.10 or higher
- pip 22+
- Git
- Jupyter Notebook or Jupyterlab (for `.ipynb` notebooks)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.uio.no/yangliu/FYS-STK3155_Project1.git
   cd FYS-STK3155_Project1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project (see [workflow](#workflow)):
   ```bash
   python file_name.ipynb
   ```


## Folder Structure `Code/`

### `Functions/`
- **Purpose:** Contains all core Python `.py` functions used across the project.
- **You can add:** New `.py` files with reusable functions.
- **Do not:** Modify existing files unless you're updating shared logic used in notebooks.

---

### `Showcase Notebooks/`
- **Purpose:** Jupyter notebooks showcasing different analyses or results using the IMDb dataset and features.
- **You can add:** New notebooks for experiments, visualizations, or analyses.
- **Do not:** Delete or rename existing notebooks without updating references in other files.
---

### `README.md`
- **Purpose:** This file — provides project context and documentation.
- **You can add:** Setup instructions, usage examples, or contribution guidelines.
- **Do not:** Leave it outdated when major changes are made.

### `Exploration of Regression with the Runge Function.pdf`
- **Purpose:** The written report based on this repository
- **You can:** Read it for a detailed explanation of the implementation and findings
- **Do not:** Use the contents of this file without citing it properly
---

## Description of Nootebooks
`GD_learningrate.ipynb` — Gradient Descent Learning Rate Analysis  
`bias-variance-tradeoff.ipynb` — Bias–Variance Trade-off Exploration  
`bias_variance.ipynb` — Bias and Variance Decomposition  
`corssvalidation.ipynb` — Cross-Validation Implementation  
`experiments.ipynb` — Experimental Results and Comparisons  
`resampling_validation.ipynb` — Resampling and Validation Techniques  
`simpleRidge.ipynb` — Ridge Regression Example

## Workflow
If you want to explore results interactively, open any notebook in `Showcase Notebooks/` with Jupyter. They can be run as is, from top to bottom.


## Authors
William Schjerve Moe, Hallvard Hareide, Max Følstad-Andresen & Yang Liu
