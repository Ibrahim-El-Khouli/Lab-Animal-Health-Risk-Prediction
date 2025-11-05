# Lab Animal Health Risk Prediction

## Phase 1 — Project Setup

### 1. Project Objectives

#### Research Problem
The goal of this project is to develop a **reproducible, modular pipeline** for quantifying laboratory animal health by integrating multiple measurable indicators into a composite **Health Score (0–100)**. Each animal is classified into **Low**, **Medium**, or **High** health-risk categories. The project uses [synthetic data](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Health-Risk-Prediction/blob/main/data/data_cleaned_realistic.csv) and a fully documented [notebook](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Health-Risk-Prediction/blob/main/lab-animal-health-risk-prediction.ipynb) for model development and evaluation.

**Design Principle — Future Species Adaptability**

- Currently trained on rats, but the pipeline can be retrained for other species (mice, rabbits, guinea pigs) with species-specific datasets.  
- Enables potential translational applicability in biomedical and preclinical research once retrained.

#### Key Benefits

- **Health Data Automation:** Converts raw logs (age, weight, activity, food/water intake, strain, sex) into interpretable Health Scores.  
- **Veterinary Monitoring Automation:** Identifies at-risk animals, generates reports, and visualizes population-wide risk distributions.

---

### 2. Rationale for Initial Species Choice — Why Rats?
Rats were chosen as the initial model species due to their extensive use in **toxicology**, **pharmacology**, and **biomedical research**.  
Commonly studied strains include **Long Evans (LE)**, **Sprague Dawley (SD)**, **Fischer 344 (F344)**, and **Brown Norway (BN)**.  
Their well-documented physiological and behavioral diversity provides an ideal testbed for evaluating the **generalizability** of the model.

---

### 3. Evolution of Modeling Strategy

#### Initial Approach — Supervised Learning
- Models: Linear Regression, Logistic Regression, Random Forests.  
- Challenge: Severe class imbalance; synthetic oversampling produced **biologically implausible distributions**.

#### Pivot — Unsupervised & Anomaly Detection
- Methods: **KMeans Clustering** and **Isolation Forest**.  
- Advantages:  
  - Generates biologically plausible risk distributions.  
  - Reduces overfitting.  
  - Captures rare, clinically relevant events.  

> **Note:** Although unsupervised methods are ideally applied to real datasets, this project uses **synthetic data** carefully designed to mimic realistic distributions, including randomness, missingness, and anomalies, to approximate true laboratory populations.

---

### 4. Final Model Components

| Component | Method | Purpose |
|------------|---------|----------|
| **Encoders** | `LabelEncoder` for strain and sex | Encode categorical features |
| **Scaler** | `MinMaxScaler` | Normalize numeric features |
| **Dimensionality Reduction** | `PCA` (2D), `UMAP` (2D embedding) | Capture structure and reduce noise |
| **Clustering** | `KMeans` (3 clusters) | Unsupervised health profiling |
| **Anomaly Detection** | `Isolation Forest` | Identify atypical health cases |

All models are versioned and saved in `models/encoders/scalers/` to ensure **reproducibility**.

---

### 5. Significance of Health Risk Prediction

#### Ethical & Welfare Impact
- Supports **humane endpoints** and promotes refined care in alignment with the **3Rs (Replacement, Reduction, Refinement)**.

#### Scientific Validity
- Reduces inter-animal variability, enhancing **reproducibility** and **statistical power**.

#### Operational Efficiency
- Automates prioritization for **veterinary interventions**.

#### Regulatory Compliance
- Provides **quantifiable welfare metrics** suitable for **IACUC** and **AAALAC** oversight and reporting.

---

### 6. Project Continuity
This work builds upon prior projects:
- **Lab Animal Growth Prediction**  
- **LECI — Lab Environmental Comfort Index**

The current pipeline advances toward **health risk prediction** through:
- Unsupervised learning and anomaly detection.  
- Automated reporting and dashboard integration.  
- Increased biological realism, interpretability, and translational potential.

---

## Environment Setup & Library Imports

A reproducible computational environment is fundamental for **transparency and scientific rigor**. This section establishes the software foundation used in the project, emphasizing unsupervised learning, anomaly detection, and reproducibility.

### 1. Standard Libraries
- **os:** File and directory management.  
- **random**, **numpy.random:** Controlled randomness for reproducibility.  
- **sys:** Environment configuration and path management.  
- **warnings:** Suppression of non-critical alerts for cleaner logs.

### 2. Numerical and Data Handling
- **NumPy:** High-performance numerical operations.  
- **Pandas:** DataFrame-based manipulation, feature engineering, and ML integration.

### 3. Visualization Tools
- **Matplotlib**, **Seaborn:** Publication-quality static plots.  
- **Missingno:** Visualization of missing data patterns.  
- **Plotly Express:** Interactive visualizations and dashboard-ready figures.  
- **pandas.plotting.parallel_coordinates:** Multivariate visualization for cluster analysis.  

A consistent Seaborn aesthetic ensures visual standardization across analyses.

### 4. Machine Learning Methods

| Function | Library | Purpose |
|-----------|----------|----------|
| **Clustering** | `KMeans` | Latent health state partitioning |
| **Anomaly Detection** | `IsolationForest`, `OneClassSVM` | Identify biologically atypical subjects |
| **Dimensionality Reduction** | `PCA`, `UMAP` | Structural simplification while preserving variance |
| **Preprocessing** | `SimpleImputer`, `MinMaxScaler`, `LabelEncoder` | Standardization and imputation |
| **Pipelines** | `sklearn.pipeline.Pipeline` | Modular and reproducible ML workflow |
| **Evaluation** | `silhouette_score` | Cluster separation and cohesion metrics |

### 5. Utilities
- **joblib:** Model serialization for persistent storage.  
- **warnings:** Filter non-critical notices from UMAP and scikit-learn.  

### 6. Reproducibility Measures
All random seeds are globally fixed (Python, NumPy, environment hashing) to ensure **deterministic outputs**, meeting biomedical research requirements for **traceability** and **repeatability**.

---

### Research Rationale
This methodological configuration reflects a deliberate transition from **supervised** to **unsupervised** modeling, enabling biologically grounded categorization without reliance on biased or imbalanced labels. The environment supports both **experimental exploration** and **scalable deployment** in research or vivarium settings.

---

## Phase 2 — Data Creation & Exploration

### Step 1: Synthetic Dataset Generation

#### Biological / Veterinary Rationale
Laboratory animal datasets typically include **physiological and behavioral features** such as:
- Weight, age, activity level, food intake, and water intake.

These variables exhibit predictable biological relationships:
- **Age vs. Weight:** Older rodents are heavier but less active.  
- **Strain Differences:** Variations in growth and metabolism across strains (e.g., SD vs. BN).  
- **Sex Differences:** Distinct feeding behavior and body composition patterns.  
- **Anomalies:** Deviations due to illness, stress, or genetic variability indicate potential clinical concern.

To approximate real-world conditions, synthetic data are **generated iteratively** to mirror realistic biological distributions and correlations.

#### Machine Learning Rationale
The synthetic dataset (n=1000) supports:
- Realistic multicollinearity between features.  
- Controlled noise and anomalies for testing model robustness.  
- Balanced categorical representation across sex and strain.  
- Transparent validation for unsupervised and anomaly detection models.

---

### Step 2: Visualization of Synthetic Dataset

#### Biological / Veterinary Rationale
Visual inspection ensures:
- Biological plausibility (e.g., sex-related weight differences).  
- Detection of anomalies (e.g., reduced activity or intake).  
- Face-validity prior to modeling.

#### Machine Learning Rationale
Visualizations verify:
- Balanced feature distributions and absence of bias.  
- Clear separability between clusters.  
- Robust foundation for exploratory data analysis (EDA).

#### Visualization Plan
- **Histograms / KDEs:** Feature distributions.  
- **Pairplots:** Relationships between key variables (e.g., age–weight).  

These checks confirm that the synthetic dataset reflects realistic biological variance and is suitable for machine learning applications.

---

### Step 3: Data Quality Checks

#### Veterinary / Biological Rationale
Real-world research data often contain **missing or incomplete records**.  
Simulating and repairing missingness prepares the model for operational datasets with imperfect integrity.

#### Machine Learning Rationale
Machine learning algorithms cannot process missing values directly.  
Therefore:
- **Continuous variables:** Imputed using mean values.  
- **Categorical variables:** Imputed using the mode.  
- **Visualization:** Missingness patterns are assessed to detect potential bias.

#### Implementation
- Artificially introduce ~5% missingness.  
- Visualize with **Missingno**.  
- Apply simple imputations and validate dataset completeness.

---

### Step 4: Exploratory Data Analysis (EDA)
EDA evaluates the **biological realism** of the dataset before modeling.

#### Goals
- Compute descriptive statistics (mean, std, median).  
- Compare distributions across **strain** and **sex**.  
- Assess inter-feature relationships.  
- Visualize patterns using boxplots, violin plots, scatterplots, correlation heatmaps, and grouped barplots.

This step ensures the dataset aligns with expected physiological and behavioral variability.

---

## Phase 3 — Feature Engineering

### Step 1: Feature Encoding

#### Biological / Veterinary Rationale
Strain and sex influence physiology and metabolism.  
Encoding these features numerically allows their integration into distance-based clustering and anomaly detection models.

#### Machine Learning Rationale
- **Label Encoding:** Converts categories into integers without imposing artificial order.  
- Simplifies integration with models like KMeans and Isolation Forest.

---

### Step 2: Derived Features

#### Biological / Veterinary Rationale
Derived variables capture **functional relationships** that reveal subtle health differences:
- **Weight/Age Ratio:** Growth efficiency and maturity indicator.  
- **Food/Weight Ratio:** Metabolic intake relative to mass (malnutrition or illness marker).  
- **Activity per Gram:** Detects lethargy or hypoactivity normalized by body size.

#### Machine Learning Rationale
Derived features:
- Emphasize **biologically relevant deviations**.  
- Reduce dependence on absolute values.  
- Improve sensitivity to physiological anomalies.

#### Visualization
- **Histograms:** Assess distributions.  
- **Boxplots by strain/sex:** Identify group-level trends.  
- **Outlier analysis:** Detect individuals with atypical patterns suggestive of risk.

*This pipeline forms the foundation for the subsequent phases of modeling, evaluation, and interpretation, ensuring both biological plausibility and computational rigor.*

---

## Phase 4 — Unsupervised Learning & Anomaly Detection

This phase transitions from supervised prediction to the identification of **latent structures and hidden patterns** within the dataset, without reliance on predefined labels. The objective is to discover biologically meaningful clusters, detect anomalies, and derive a composite health score that reflects overall wellness.

---

### Biological / Veterinary Rationale
Laboratory animals naturally exhibit **physiological and behavioral variability**. Unsupervised learning captures this variability to identify:
- Subpopulations or health profiles corresponding to normal, borderline, or at-risk conditions.  
- Individual anomalies that deviate significantly from expected physiological norms.  
- Strain- and sex-specific trends in health metrics, informing **evidence-based veterinary interventions**.

### Machine Learning Rationale
- **Scaling** ensures distance-based algorithms (e.g., KMeans, DBSCAN) treat all features equitably.  
- **Dimensionality reduction** (PCA, UMAP) uncovers structure and supports visualization.  
- **Clustering** reveals intrinsic data organization, while **anomaly detection** quantifies deviation.  
- Combined, these methods enable a **continuous health scoring system**, categorized into Low, Medium, or High Risk.

### Outcome
- Data-driven identification of health profiles and outliers.  
- Early warning for animals requiring clinical attention.  
- Enhanced interpretability via feature-level and cluster-level insights.

---

### Step 1 — Feature Scaling

#### Biological / Veterinary Rationale
Physiological features vary in **unit and biological range**:
- **Age (days):** Inversely related to health beyond maturity.  
- **Weight (grams):** Optimal range corresponds to balanced metabolic status.  
- **Activity, Food, Water Intake:** Higher normalized values generally indicate wellness.

Scaling aligns all features to comparable ranges, ensuring that no variable (e.g., weight) dominates the clustering process due to magnitude differences.  
Non-linear biological relationships are modeled where appropriate (e.g., Gaussian scaling for weight).

#### Machine Learning Rationale
- Ensures balanced influence of all variables in distance-based models.  
- Improves numerical stability and interpretability for algorithms sensitive to magnitude.

**Outcome:**  
A biologically standardized and numerically stable dataset suitable for dimensionality reduction, clustering, and anomaly detection.

---

### Step 2 — Dimensionality Reduction for Visualization

#### Biological / Veterinary Rationale
Dimensionality reduction allows visualization of **multivariate health data** to uncover structure:
- **PCA:** Identifies dominant axes of physiological variance (e.g., strain or sex effects).  
- **UMAP:** Preserves local relationships, revealing subtle subgroupings or potential risk clusters.  
Veterinarians can interpret resulting visualizations as distinct cohorts—healthy, intermediate, or at-risk populations.

#### Machine Learning Rationale
- Reduces dimensionality while retaining variance and structure.  
- Scree plots determine the number of principal components explaining major variability.  
- 2D and 3D projections (PCA, UMAP) support intuitive visualization of cluster separation and anomalies.

**Outcome:**  
Low-dimensional projections reveal underlying biological structure, highlighting both typical and atypical animal profiles.

---

### Step 3 — Clustering to Identify Health Profiles

#### Biological / Veterinary Rationale
Clustering partitions animals into biologically interpretable cohorts:
- **Healthy clusters** show balanced physiological metrics.  
- **Intermediate clusters** may indicate mild deviation or adaptation.  
- **At-risk clusters** show consistent deviations in weight, intake, or activity.

Cluster centroids summarize **mean feature profiles**, providing reference points for veterinary interpretation.  
Visual summaries (radar plots, boxplots) illustrate inter-cluster contrasts.

#### Machine Learning Rationale
- **KMeans** segments animals by similarity in feature space.  
- **Silhouette score** and **elbow method** determine optimal cluster number.  
- Clusters are visualized in PCA and UMAP space for interpretability.

**Outcome:**  
Biologically meaningful health-state clusters, each characterized by unique feature distributions.

---

### Step 4 — Anomaly Detection with Isolation Forest

#### Biological / Veterinary Rationale
Outlier detection identifies individuals deviating from expected health norms, which may represent:
- Early indicators of disease or distress.  
- Exceptionally robust individuals influencing study variability.  
- Measurement errors or data artifacts requiring review.

Detecting these anomalies enables **targeted clinical follow-up** and improves overall data reliability.

#### Machine Learning Rationale
The **Isolation Forest** algorithm isolates rare observations by recursive partitioning:
- Does not assume feature normality.  
- Resistant to irrelevant variables.  
- Scalable for large vivarium populations.

Each animal receives an **anomaly score**, which is integrated into the final health scoring framework.

#### Planned Visualizations
- PCA and UMAP scatterplots with anomalies highlighted.  
- Parallel coordinates plots illustrating multi-feature deviation patterns.

**Outcome:**  
Systematic identification of atypical cases for veterinary review and health monitoring.

---

### Step 5 — Composite Health Score and Risk Categories

#### Biological / Veterinary Rationale
Integrates multiple metrics (age, weight, activity, food/water intake, anomaly score) into a **single continuous measure** of wellness:
- **High scores:** Reflect optimal physiological balance.  
- **Low scores:** Indicate deviation or potential welfare concern.

**Risk Classification:**
- **Low Risk:** Healthy baseline.  
- **Medium Risk:** Moderate deviation.  
- **High Risk:** Significant or flagged anomalies.

#### Machine Learning Rationale
- Weighted sum of scaled features forms the **Health_Score**.  
- Percentile thresholds define categorical risk bands.  
- Enables downstream analyses, ranking, and automated alerts.

**Outcome:**  
An interpretable composite health metric integrating unsupervised structure and anomaly detection into a unified risk model.

---

### Step 6 — Feature Interpretation

#### Biological / Veterinary Rationale
Feature-level interpretation reveals which physiological parameters drive health differentiation:
- Cluster centroids clarify dominant health patterns.  
- Comparative analysis highlights distinguishing metrics between healthy and at-risk cohorts.

#### Machine Learning Rationale
- **Permutation importance** or **SHAP values** quantify individual feature contributions.  
- **Heatmaps** and **barplots** visualize per-cluster feature influence.

**Outcome:**  
Transparent, interpretable insight into the physiological basis of the model’s decisions—critical for both ethical validation and clinical translation.

---

### Step 7 — Temporal / Trend Analysis

#### Biological / Veterinary Rationale
For studies with longitudinal data, tracking health dynamics enables:
- Early identification of decline or recovery trajectories.  
- Continuous welfare monitoring over study duration.  
- Retrospective linkage between interventions and outcomes.

#### Machine Learning Rationale
- **Line plots:** Track Health_Score evolution over time.  
- **Sankey diagrams:** Visualize transitions between clusters or risk categories.  
- **Trend modeling:** Predict progression toward adverse outcomes.

**Outcome:**  
A temporal framework that transforms static health classification into **dynamic health trajectory modeling**, enhancing predictive welfare surveillance.

*This phase culminates in the construction of a biologically informed, data-driven model capable of identifying, monitoring, and interpreting laboratory animal health profiles in an ethical and reproducible manner.*

---

## Phase 5 — Reporting & Dashboard

The final phase focuses on **translating analytical results into interpretable, interactive insights** through dashboards and scientific visualization. This enables veterinarians, researchers, and oversight committees to explore risk predictions and health trends transparently and efficiently.

### Key Visual Summaries

#### 1. Risk Distribution per Strain and Sex
- Bar or pie charts illustrate the proportion of animals across **Low**, **Medium**, and **High** risk categories.
- Facilitates rapid identification of population-level health disparities and strain- or sex-specific susceptibilities.

#### 2. Cluster Scatterplots with Risk Overlays
- **PCA** or **UMAP** projections colored by health risk or cluster membership.
- Provides an intuitive visualization of how animals group based on multidimensional health features.

#### 3. Highlighted Anomalies
- Points flagged by **Isolation Forest** or other anomaly detection models are visually emphasized.
- Allows early identification of potentially deteriorating animals or measurement irregularities.

### Tools and Implementation
- **Plotly** for interactive 2D/3D visualizations (scatterplots, bar charts, pie charts).
- **Dash** or **Streamlit** for dashboard deployment, enabling filtering by strain, sex, or cluster.
- Real-time updates supported through modular model loading for new animal records.

### Purpose
- Facilitate **rapid, transparent health profiling** across cohorts.
- Support **data-driven welfare monitoring** aligned with institutional animal care standards.
- Deliver **publication-ready visualizations** suitable for research dissemination and internal quality control.

---

## Phase 6 — Model Preservation and Predictive Function

### Step 1 — Model Saving and Versioning

#### Biological / Veterinary Rationale
Preserving preprocessing objects and trained models ensures **consistent and objective scoring** for new animals, enabling:
- Early identification of at-risk individuals for **humane interventions**.
- Standardized, reproducible health assessments across studies, strains, and facilities.
- Improved **translational reliability** across species through harmonized data handling.

#### Machine Learning Rationale
- Encoders, scalers, and model objects (PCA, UMAP, clustering, anomaly detection) are serialized using `joblib` or `pickle`.
- Modular design allows version control and future updates without retraining.
- Ensures reproducibility, scalability, and auditability across datasets and institutions.

**Objects Saved**
- Encoders: Strain, sex.
- Scalers: Feature normalization (e.g., MinMaxScaler).
- Dimensionality Reduction: PCA / UMAP.
- Clustering: KMeans or equivalent.
- Anomaly Detection: Isolation Forest or Autoencoder.

#### Outcome
A **versioned and reusable pipeline** supporting:
- Standardized Health Score computation.
- Cluster and anomaly assignment for new subjects.
- Cross-species adaptability through configuration-based adjustments.

---

### Step 2 — Prediction Function

#### Biological / Veterinary Rationale
Provides a standardized framework for **real-time health assessment**, allowing veterinarians to:
- Monitor individual animals’ well-being.
- Detect deviations from expected health norms early.
- Contextualize health status through **multi-feature composite scoring**.

#### Machine Learning Rationale
The function encapsulates preprocessing, transformation, and inference steps to ensure **end-to-end reproducibility**:
1. Encode categorical features (strain, sex).
2. Scale numeric variables.
3. Apply PCA/UMAP transformation.
4. Assign cluster membership.
5. Compute anomaly score.
6. Derive composite **Health Score (0–100)**.
7. Categorize into **Low**, **Medium**, or **High Risk**.

**Outputs**
- Health Score and risk category.
- Visual projection on PCA/UMAP scatterplots.
- Feature deviation plots comparing the subject to cluster medians.

---

### Example Output — Veterinary Health Assessment

**Patient Info**  
Strain: *Sprague Dawley*  
Sex: *Male*  
Age: *400 days*  
Weight: *320 g*  
Activity: *6*  
Food Intake: *18 g/day*  
Water Intake: *40 ml/day*

**Model Prediction**  
- Cluster: *2* — grouped with similar health profiles  
- Anomaly Score: *0.133* (Normal)  
- Health Score: *63.3 → Medium Risk*

**Feature Deviations**
- Weight: Slightly above median → mild overweight  
- Activity: Slightly below median → reduced mobility  
- Food Intake: Normal  
- Water Intake: Slightly below median → monitor hydration  
- Age: Within expected range  

**Interpretation**  
The animal appears generally healthy but exhibits moderate deviations in weight, activity, and hydration, warranting observation and mild husbandry adjustments.

**Visual Aid**  
Feature deviation bar charts allow intuitive recognition of physiological imbalances at a glance.

---

## Conclusion and Ethical Framework

### Key Findings
- The pipeline integrates **unsupervised learning and anomaly detection** to generate interpretable **Health Scores** for laboratory animals.  
- Multi-feature integration (age, weight, activity, intake) revealed consistent trends:
  - Younger, active animals with mid-range weights cluster as healthiest.
  - Low intake and activity correlate strongly with higher risk categories.
- The model demonstrates robust generalization across strains and sexes, highlighting the value of **data-driven welfare monitoring**.

### Biological and Translational Implications
- Enables early detection of deteriorating health states, improving animal welfare and research validity.  
- Standardized scoring enhances **reproducibility and inter-laboratory comparability**.  
- Framework adaptable across species and institutions, advancing **evidence-based refinement** of animal care.

### Ethical and Regulatory Alignment
This analytical framework supports and operationalizes international ethical standards in animal research:
- **IACUC & AAALAC Compliance:** Provides quantifiable, objective welfare metrics to inform ethical oversight and refinement actions.  
- **3Rs Principle (Replacement, Reduction, Refinement):**  
  - *Replacement:* Enhances synthetic data simulation, reducing reliance on live experimentation.  
  - *Reduction:* Enables early detection and intervention, lowering morbidity and attrition rates.  
  - *Refinement:* Promotes continuous welfare monitoring and humane care adjustments through data-driven insights.

### Overall Impact
The **Lab-Animal-Health-Risk-Prediction** pipeline establishes a **quantitative and ethically aligned framework** for continuous welfare assessment, bridging the gap between computational intelligence and humane veterinary science.

---

### Acknowledgments
This work integrates principles from **Laboratory Animal Medicine**, **Machine Learning**, and **Ethical Biomedical Research**.  
The project concept and structure were informed by international welfare standards including **AAALAC International**, **IACUC** oversight principles, and the **3Rs (Replacement, Reduction, Refinement)** framework.  
Gratitude is extended to the broader community of **veterinary scientists and biomedical data researchers** whose work continues to advance both scientific rigor and humane animal care.

---

## **GitHub Repositories for Other Work**

- [PostOpPainGuard™](https://github.com/Ibrahim-El-Khouli/PostOpPainGuard.git)
- [LECI - Lab Environmental Comfort Index](https://github.com/Ibrahim-El-Khouli/LECI-Lab-Environmental-Comfort-Index.git)  
- [Lab Animal Health Risk Prediction](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Health-Risk-Prediction.git)  
- [Lab Animal Growth Prediction](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Growth-Prediction.git)

---

## **License**

**Lab Animal Health Risk Prediction** is released under the **MIT License** — free for academic, research, and non-commercial use.
