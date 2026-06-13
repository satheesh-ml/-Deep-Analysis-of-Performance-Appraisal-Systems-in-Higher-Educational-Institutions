# -Deep-Analysis-of-Performance-Appraisal-Systems-in-Higher-Educational-Institutions

3.1 Dataset 1: IBM HR Analytics Employee Attrition & Performance
Attribute	Details
Source	Kaggle / IBM Data Scientists
URL	https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
Records	1,470 employee records
Features	35 attributes (PerformanceRating, JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, TrainingTimesLastYear, YearsSinceLastPromotion, etc.)
Format	CSV
Relevance	Contains structured Likert-scale appraisal-related fields directly measuring performance ratings, satisfaction dimensions, and motivational proxies.

The IBM HR dataset simulates a realistic organisational survey environment with 35 features capturing employee demographics, job-role characteristics, satisfaction levels, performance ratings, and attrition status. Key fields such as PerformanceRating (1–4 scale), JobInvolvement, JobSatisfaction, RelationshipSatisfaction, and EnvironmentSatisfaction directly reflect appraisal-related constructs central to this study's objectives.
3.2 Dataset 2: Higher Education Students Performance Evaluation (UCI)
Attribute	Details
Source	UCI Machine Learning Repository
URL	https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation
Records	145 student/faculty feedback records
Features	33 features including personal, family, academic habit, and feedback questionnaire items
Format	XLSX / CSV
Relevance	Questionnaire-based dataset from Faculty of Engineering and Educational Sciences, directly representing HEI performance evaluation feedback structures.

Collected from the Faculty of Engineering and Faculty of Educational Sciences in 2019, this UCI dataset consists of questionnaire responses designed to predict end-of-term academic performance. Items 1–10 address personal attributes, 11–16 capture family background, and the remainder reflect educational habits and performance feedback — making it structurally analogous to faculty appraisal questionnaires used in HEIs.
3.3 Dataset 3: Employee Performance and Productivity Data
Attribute	Details
Source	Kaggle (mexwell)
URL	https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data
Records	100,000 employee-level records
Features	20+ features including performance scores, feedback scores, promotions, training hours, work-life balance ratings, and motivation indicators
Format	CSV
Relevance	Large-scale dataset with explicit performance feedback scores and motivation-linked features, enabling deep learning model training at scale.

This large-scale productivity dataset (100,000 records) provides the volume required for deep learning model training and validation. Its inclusion of direct feedback scores, training hours, promotion indicators, and self-reported work-life balance ratings aligns precisely with the three specific objectives of this study.
3.4 Dataset Fusion Strategy
Given heterogeneity across the three datasets, a standardised fusion pipeline is applied: (1) feature harmonisation via semantic alignment of conceptually equivalent fields; (2) normalisation of Likert-scale ratings to a uniform [0,1] range; (3) one-hot encoding of categorical variables; and (4) stratified oversampling (SMOTE) to address class imbalances in performance and motivation categories. The fused dataset yields approximately 101,760 records with 28 harmonised features.

4. Proposed Methodology: HALTE-G Framework
The proposed Hybrid Attention-LSTM with Transformer Encoder and Graph Attention Network (HALTE-G) is an extended novel deep learning architecture specifically designed for structured HR questionnaire and appraisal relational data. It combines the sequential modelling strengths of LSTM networks, the global dependency capture of Transformer encoder blocks, multi-head self-attention fusion, and — as the novel extension — a Graph Attention Network (GAT) module that explicitly models the relational interdependencies between appraisal system dimensions as a learnable graph structure. This five-module hybrid architecture advances the original HALTE by incorporating relational graph-based reasoning into the performance appraisal analysis pipeline.
4.1 Overview of the HALTE-G Architecture
HALTE-G comprises five principal modules:
•	Input Embedding Module: Transforms the 28-feature fused input vector into dense embeddings using learnable linear projections, with positional encoding to preserve feature-ordering semantics.
•	GAT Relational Module (NEW): A 3-layer Graph Attention Network constructs a feature-relation graph where each PAS dimension is a node and edges represent learned relational dependencies. Multi-head graph attention aggregates neighbourhood information to enrich each feature embedding with relational context before sequential processing.
•	LSTM Sequential Module: A 3-layer bidirectional LSTM (hidden size = 256) captures temporal and sequential dependencies within longitudinal appraisal feedback sequences, operating on GAT-enriched embeddings.
•	Transformer Encoder Module: A 4-layer Transformer encoder (8 attention heads, feed-forward dimension = 512) processes the full feature context to capture global, non-local relationships among appraisal dimensions.
•	Hybrid Fusion and Classification Head: Multi-head cross-attention fuses GAT, LSTM, and Transformer representations; a fully connected layer with dropout (p = 0.3) and softmax/linear activation produces the final classification or regression output.
4.2 Graph Attention Network (GAT) Module
The GAT module is the core novel contribution introduced in this extended HALTE-G framework. Unlike conventional feature-processing layers that treat appraisal dimensions as independent inputs, GAT explicitly models the relational structure among PAS features by constructing a directed feature-relation graph G = (V, E), where:
•	V = {v₁, v₂, ..., v₂₈} represents the 28 harmonised PAS feature nodes (e.g., Feedback Quality, Job Involvement, Environment Satisfaction).
•	E represents learnable directed edges encoding the conditional dependency between pairs of appraisal features (e.g., Feedback Quality → Job Involvement → Motivation).
•	Edge weights are learned end-to-end during training, enabling the model to automatically discover which appraisal relational pathways are most predictive of performance and motivation outcomes.
The GAT attention coefficient between node i and its neighbour j is computed as:
eᵢⱼ = LeakyReLU( aᵀ [W·hᵢ || W·hⱼ] )
αᵢⱼ = softmax_j(eᵢⱼ) = exp(eᵢⱼ) / Σₖ∈Nᵢ exp(eᵢₖ)
where hᵢ and hⱼ are feature embeddings of nodes i and j respectively, W is a shared learnable weight matrix, a is the attention vector, and || denotes concatenation. The normalised attention coefficients αᵢⱼ determine how much each neighbouring appraisal feature contributes to the updated representation of node i. The updated node representation is then:
h'ᵢ = σ( Σⱼ∈Nᵢ αᵢⱼ · W · hⱼ )
where σ is the ELU activation function. Multi-head GAT (K=8 heads) is employed to stabilise learning and capture diverse relational patterns across appraisal dimensions simultaneously. The final GAT output concatenates all K head representations: h'ᵢ = ||ₖ₌₁ᴷ σ( Σⱼ∈Nᵢ αᵢⱼᵏ · Wᵏ · hⱼ ).
4.2.1 PAS Feature Graph Construction
The PAS feature graph is constructed using a hybrid strategy: (a) domain-knowledge-driven initial edges based on established HRM theory (e.g., Training → Performance, Feedback → Motivation); and (b) data-driven edge refinement via mutual information scores computed on the training set. Edges with mutual information below a threshold τ = 0.05 are pruned. The resulting graph has 28 nodes and 94 directed edges, with an average node degree of 3.36. This sparse graph structure prevents over-smoothing while preserving the most meaningful inter-feature appraisal relationships.
4.2.2 Why GAT for PAS Analysis in HEIs
The application of GAT to performance appraisal data is motivated by three HEI-specific considerations: (1) appraisal dimensions are inherently interdependent — a faculty member's feedback quality perception is shaped by their environment satisfaction and relationship with their appraiser; (2) conventional deep learning models treat features as independent, missing these relational pathways; and (3) GAT's attention coefficients are directly interpretable as relational impact weights, providing an additional layer of explainability beyond SHAP — specifically revealing which appraisal feature relationships drive outcomes. This constitutes a novel analytical lens not previously applied in the HEI PAS literature.
4.3 Multi-Head Self-Attention Mechanism
The self-attention operation for the i-th head in the Transformer encoder is defined as:
Attention(Q, K, V) = softmax( QKᵀ / √dk ) · V
where Q, K, V are query, key, and value matrices respectively, and dk is the key dimension. Multi-head attention enables the model to jointly attend to information from different appraisal feature subspaces simultaneously, a critical capability for capturing the multidimensional nature of employee motivation and performance constructs. In HALTE-G, the Transformer encoder operates on GAT-enriched and LSTM-processed representations, giving it richer relational and sequential context compared to the original HALTE.
4.4 Loss Function and Optimisation
For performance classification tasks (multi-class: Low / Average / High / Exceptional), categorical cross-entropy loss is used. For motivation intensity regression (continuous score), Mean Squared Error (MSE) loss is employed. The combined HALTE-G loss is: L_total = λ₁·L_CE + λ₂·L_MSE + λ₃·L_GAT, where L_GAT is a graph regularisation term penalising inconsistent attention weights across similar node pairs (λ₁=0.6, λ₂=0.3, λ₃=0.1). The model is optimised using AdamW with a cosine annealing learning rate schedule (initial lr = 1e-4, weight decay = 0.01).
4.5 Explainability: SHAP and GAT Attention Integration
HALTE-G delivers a dual-layer explainability framework: (1) SHAP (SHapley Additive exPlanations) values via DeepSHAP quantify each individual feature's marginal contribution to the model's prediction; and (2) GAT attention coefficients (αᵢⱼ) reveal the relational pathways — specifically, which pairs of appraisal features jointly drive performance and motivation outcomes. Together, SHAP identifies 'what matters' (feature-level) while GAT attention identifies 'how it connects' (relation-level), providing HEI administrators with both diagnostic and structural insights for appraisal system redesign.
4.5 Impact Analysis Framework
To directly address the primary objective — examining the impact of PAS in HEIs — three complementary analytical methods are integrated into the HALTE framework:
4.5.1 Pearson Correlation Analysis
Pearson correlation coefficients are computed between all 28 harmonised appraisal features and the target outcome variables (performance score, motivation index). This provides an initial linear impact mapping to identify which appraisal dimensions have the strongest direct associations with employee outcomes. Features with |r| > 0.3 are flagged as high-impact appraisal indicators. A correlation heatmap is generated to visualise inter-feature relationships and multicollinearity patterns across all three datasets.
4.5.2 Multiple Regression Analysis (Impact Quantification)
A Multiple Linear Regression (MLR) model is fitted on the fused dataset to quantify the independent contribution of each PAS dimension to the performance outcome variable, controlling for demographic and job-role covariates. Standardised beta coefficients (β) are reported to enable direct comparison of impact magnitudes across features measured on different scales. The regression model also serves as an interpretable reference benchmark against which the HALTE deep learning predictions are compared, validating that the deep model captures non-linear effects beyond linear impact.
4.5.3 SHAP-Based Impact Score Analysis
SHAP (SHapley Additive exPlanations) values computed from the trained HALTE model provide a non-linear, model-aware impact score for each appraisal feature. Unlike correlation and regression, SHAP impact scores account for feature interactions and capture the conditional, context-dependent influence of each PAS dimension. Mean absolute SHAP values (|SHAP|) are aggregated across the test set to produce a ranked PAS Impact Index — a novel metric introduced in this study to quantify the overall impact magnitude of individual appraisal system components on employee outcomes in HEIs.
