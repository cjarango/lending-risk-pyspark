# Lending Club Default Prediction: Scikit-Learn vs. PySpark
**A Comparative Analysis on Financial Risk & Model Interpretability**

**Authors:**
* **Paula Andrea Gómez Vargas** (apaulag@uninorte.edu.co)
* **Juan Camilo Mendoza Arango** (jcamilo01mendoza@gmail.com)
* **Miguel Ángel Pérez Vargas** (miguelangelpv0920@gmail.com)

---

# Project Overview

This project implements a supervised machine learning pipeline to predict loan defaults using the Lending Club dataset (containing over 2 million records). The core objective is to benchmark the performance and computational efficiency of **Scikit-Learn** (local processing) against **PySpark** (distributed processing). 

Furthermore, the project integrates **LIME** (Local Interpretable Model-agnostic Explanations) to provide transparency for individual risk predictions, ensuring the models are not just accurate but explainable in a financial context.

Both authors contributed equally to the research, data engineering, and model development phases of this project.

## Data Source

This dataset contains the full LendingClub data available from their site. There are separate files for accepted and rejected loans. The accepted loans also include the FICO scores, which can only be downloaded when you are signed in to LendingClub and download the data.  
Source: [Lending Club Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club?resource=download)

# Project Structure

To ensure modularity and reproducibility, the project follows this structure. Custom logic and repetitive functions are stored in the `src/` directory to keep notebooks clean and focused on insights.

```text
lending-risk-pyspark/
├── data/               # Raw and processed datasets (Git ignored)
├── notebooks/          # Modular Experimentation & Reports
│   ├── 01_eda_limpieza.ipynb
│   ├── 02_sklearn_model.ipynb
│   ├── 03_pyspark_model.ipynb
│   └── 04_interpretacion_lime.ipynb
├── src/                # Modular source code (Manual functions)
│   ├── preprocessing.py
│   └── evaluation.py
├── models/             # Serialized model artifacts
├── .gitignore          # Environment & Data exclusion rules
├── README.md           # Project documentation
└── environment.yml     # Conda environment configuration
```

## Objectives

The goal is to construct a **classification model** to predict whether a loan will result in **Default (1)** or be **Fully Paid (0)**.

### Target Definition

The target variable `default` is derived from the `loan_status` column:

$$
default =
\begin{cases}
1 & \text{if } loan\_status = \text{'Charged Off'} \\
0 & \text{if } loan\_status = \text{'Fully Paid'}
\end{cases}
$$

---

## Methodology & Deliverables

This project is presented as a **documented Jupyter Book** covering the following components:

- **Exploratory Data Analysis (EDA):**  
  Visualization of class distributions, missing values, and feature correlations.

- **Preprocessing:**  
  Comparative implementation of feature scaling and encoding using both **Scikit-Learn pipelines** and **PySpark ML transformers**.

- **Modeling & Tuning:**  
  Hyperparameter optimization for `RandomForestClassifier` across both frameworks.

- **Evaluation:**  
  Performance metrics including **Accuracy, Precision, Recall, F1-Score, and ROC AUC**, along with **training time benchmarks**.

- **Interpretability:**  
  Utilizing **LIME** to explain misclassified instances and identify key risk drivers.

---

## Critical Reflection

Upon completion, this report addresses the following key questions:

1. Which environment offered the best trade-off between **speed and accuracy**?
2. How does **data volume** impact the necessity of a distributed framework like **PySpark**?
3. How does **LIME** improve trust in automated **financial decision-making**?

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USER/lending-risk-pyspark.git
```

Setup the environment:

```bash
conda env create -f environment.yml
conda activate lending-risk
```
