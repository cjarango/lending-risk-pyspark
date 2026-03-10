# Lending Club Default Prediction: Scikit-Learn vs. PySpark
**A Comparative Analysis on Financial Risk & Model Interpretability**

**Authors:**
* **Paula Andrea Gómez Vargas** (apaulag@uninorte.edu.co)
* **Juan Camilo Mendoza Arango** (jcamilo01mendoza@gmail.com)
* **Miguel Ángel Pérez Vargas** (miguelangelpv0920@gmail.com)

---

# Project Overview

This research focuses on a technical benchmark between local and distributed computing architectures for financial risk modeling. Utilizing the Lending Club dataset—comprising over 1 million records—the project evaluates the scalability, memory management, and execution latency of Scikit-Learn (single-node processing) against Apache PySpark (distributed cluster processing).

Rather than focusing solely on predictive accuracy, this analysis explores the computational trade-offs required to handle large-scale data in a production-grade environment. To ensure these models meet the transparency standards required in the fintech industry, we integrate **LIME** (Local Interpretable Model-agnostic Explanations), providing a framework for auditable and explainable risk predictions across both platforms.

All three authors contributed equally to the research design, data engineering, benchmarking methodology, and model development phases of this project.

## Data Source

This dataset contains the full LendingClub data available from their site. There are separate files for accepted and rejected loans. The accepted loans also include the FICO scores, which can only be downloaded when you are signed in to LendingClub and download the data.  
Source: [Lending Club Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club?resource=download)

# Project Structure

To ensure modularity and reproducibility, the project follows this structure. Custom logic and repetitive functions are stored in the `src/` directory to keep notebooks clean and focused on insights.

```text
lending-risk-pyspark/
├── data/
│   ├── accepted_2007_to_2018Q4.csv.gz
│   └── lending_club_fase1_curated.parquet
├── notebooks/
│   └── lending_risk_pyspark.ipynb
├── src/
│   ├── cv_optimizers.py
│   ├── plots.py
│   ├── preprocessing.py
│   ├── risk_analytics_utils.py
│   └── scoring_engine_tools.py
├── .gitignore
├── README.md
└── environment.yml
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
