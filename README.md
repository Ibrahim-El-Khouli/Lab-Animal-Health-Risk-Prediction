# Laboratory Animal Health Risk Prediction Pipeline

## Overview
The **Laboratory Animal Health Risk Prediction Pipeline** implements a **reproducible and modular computational workflow** to quantify laboratory animal health.  
It integrates multiple measurable indicators into a **composite Health Score (0–100)** and classifies animals into **Low, Medium, or High Risk** categories.

**Core Design Goals:**
- **Species-Adaptable (with retraining):** Pipeline is designed to be retrained for mice, rats, rabbits, guinea pigs, or non-human primates using appropriate datasets.  
- **Veterinary Perspective:** Provides actionable insights to guide animal care, health monitoring, and colony management.  
- **Reproducibility & Scalability:** Applicable to both synthetic and real datasets, supporting translational and laboratory animal research.

---

## Key Features
- **Health Data Automation:** Converts raw logs (age, weight, activity, food/water intake, strain, sex) into interpretable Health Scores and Risk Categories.  
- **Veterinary Monitoring:** Identifies at-risk individuals and generates reports and visualizations to prioritize interventions.  
- **Temporal / Trend Analysis:** Tracks changes in health over time to detect early signs of deterioration.  
- **Strain & Sex Flexibility:** Dynamically handles previously unseen strains and sex categories to support robust predictions.

---

## Modeling Strategy

### Initial Approach: Supervised Learning
- Explored regression/classification models including Linear Regression, Logistic Regression, and Random Forests.  
- Encountered **class imbalance**, with few medium/high-risk samples.  
- Synthetic oversampling produced biologically implausible results, limiting reliability.

### Shift to Unsupervised & Anomaly Detection
- Adopted **clustering and anomaly detection** to uncover latent structure without labeled outcomes.  
- Produced **biologically plausible risk distributions** while reducing overfitting.  

> **Note:** Unsupervised ML projects are ideally applied to **real-world data**. Here, synthetic data was carefully designed to mimic real populations, including **missing values, variability, and anomalies**, to ensure the pipeline is robust and biologically relevant.

---

## Synthetic Dataset Construction
- Realistic distributions for: Age, Weight, Activity, Food/Water Intake, Strain, Sex.  
- Introduced **missing values**, addressed with imputation to maintain robust preprocessing.  
- Added **biological noise and anomalies** to simulate at-risk individuals.  
- Ensures **relevance for veterinary interpretation** without compromising ethical standards.

---

## Pipeline Components
| Component          | Description                                  |
|-------------------|----------------------------------------------|
| `le_strain`       | LabelEncoder for strain                       |
| `le_sex`          | LabelEncoder for sex                          |
| `scaler_numeric`  | MinMaxScaler for numeric features            |
| `pca`             | PCA for dimensionality reduction (2 components) |
| `umap`            | UMAP 2D embedding for visualization          |
| `kmeans`          | KMeans clustering (3 clusters)               |
| `iso_forest`      | Isolation Forest for anomaly detection       |

**Outputs:**
- Health Score (0–100)  
- Risk Category (Low, Medium, High)  
- Cluster assignment & anomaly flags to support veterinary review

---

## Biological & Ethical Significance
- **Veterinary Relevance:** Provides interpretable outputs for veterinarians to assess individual and colony health, identify at-risk animals, and optimize care strategies.  
- **3Rs Alignment:**  
  - **Refinement:** Early detection supports humane endpoints and reduces animal stress.  
  - **Reduction:** Improves data quality and reduces the number of animals needed for statistically valid studies.  
  - **Replacement (partial):** Synthetic and computational approaches reduce unnecessary repetition of animal use.  
- **Scientific Validity:** Minimizes inter-animal variability, enhancing reproducibility and statistical power.  
- **Operational Efficiency:** Automates monitoring to optimize veterinary resources and intervention timing.  
- **Regulatory Support:** Provides quantifiable metrics for IACUC/AAALAC audits and inspections.

---

## Initial Model Species
- **Species:** Rats  
- **Strains:** Long Evans (LE), Sprague Dawley (SD), Fischer 344 (F344), Brown Norway (BN)  
- **Reasoning:** Commonly used in toxicology, pharmacology, and biomedical research. Provides physiological and behavioral diversity for testing generalizability and veterinary interpretation.

---

## Previous Work & Continuity
- Builds on the [**Lab Animal Growth Prediction**](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Growth-Prediction) framework (murine growth) and [**LECI - Lab Environmental Comfort Index**](https://github.com/Ibrahim-El-Khouli/LECI-Lab-Environmental-Comfort-Index.git).  
- Introduces:  
  - Realistic synthetic datasets  
  - Expanded modeling toolkit: Linear, Ridge, Random Forest, Gradient Boosted Trees  
- **Gradient Boosted Trees** emerged as the **most effective and precise**, reproducible for deployment.

---

## Continuity & Advancements
Building on growth prediction and environmental comfort assessment, this pipeline advances to **health risk prediction**:  

| Feature                     | Growth Prediction         | Health Risk Pipeline                          |
|-------------------------------|------------------------|---------------------------------------------|
| Target                        | Body weight            | Health Score / Risk Category                 |
| Approach                       | Supervised regression  | Unsupervised clustering + anomaly detection |
| Visualization                  | Growth curves, residuals | Risk distributions, UMAP embeddings, temporal trends |
| Species Flexibility            | Mice (template)        | Rats, adaptable to other species with retraining |
| Complexity                     | Moderate               | Advanced: dashboards, temporal tracking, dynamic strain/sex handling |

> This pipeline represents a **step forward in laboratory animal medicine**, integrating machine learning approaches with actionable veterinary insights. It offers tools for **objective, reproducible, and interpretable health assessment**.

---

## Summary
This pipeline:  
- Builds on previous growth and environmental frameworks  
- Introduces **unsupervised sophistication** and anomaly detection  
- Provides actionable **veterinary insights for laboratory animal health**  
- Is **modular, reproducible, and adaptable to multiple species with retraining**  

It is a **next-generation computational veterinary toolkit**, designed to support translational research, enhance animal welfare, and advance the practice of laboratory animal medicine.
