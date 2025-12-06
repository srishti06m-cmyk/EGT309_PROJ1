## Group member names + email

Name: Srishti Malapuram
Email: 233106B@mymail.nyp.edu.sg

Name: Andrew Lau Yaoneng
Email: 233449m@mymail.nyp.edu.sg

Name: Amirul Haziq Bin Nor Isman
Email: 232206S@mymail.nyp.edu.sg

## Project Overview 

This project focuses on the deployment of an end-to-end machine learning pipeline to predict weather or not a banking customer will subscribe to a term deposit. The given dataset contains customer details such as demographic information, financial history, age and subscription details. The goal of this project is to:

1. Clean and process raw data from a SQLite database
2. Derive insights and visualization through EDA
3. Perform Data processing and feature engineering based on EDA findings 
4. Build a reliable ML pipeline using kedro 
5. Select and Train most suitable models.
6. Report findings and results

## How to run this project

## Description of logical steps/flow of the pipeline
- Method 1: Running on VS code

Step 1: Clone repositary from Git hub using this "gh repo clone srishti06m-cmyk/EGT309_PROJ1" command in VS code terminal
Step 2: Open the project locally using this "code EGT309_PROJ1" command in the VS code terminal
Step 3: Run the kedro pipeline using "kedro run" in the VS code terminal

- Method 2: Running the DockerFile

There are two DockerFiles. One for the Kedro pipeline (Dockerfile.Main) and the other for the EDA file (Dockerfile.run). to run the kedro pipeline Dockerfile (Dockerfile.Main):

Step 1: Ensure Docker Desktop is open before running these commands 
Step 2: Change the project directory on either the VS code terminal, Powershell or Windows Terminal to be inside the project folder containing the Dockerfile.Main
Step 3: Run this "docker build -f Dockerfile.Main -t egt309-pipeline ." command on the terminal
Step 4: Run this "docker run --rm egt309-pipeline" command to run the kedro pipeline

To run the Dockerfile for EDA (Dockerfile.run):

Step 1: Ensure Docker Desktop is open before running these commands
Step 2: Change the project directory on either the VS code terminal, Powershell or Windows Terminal to be inside the project folder containing the Dockerfile.run
Step 3: Run the development container which will bring us into container terminal using this "docker run -it -p 8888:8888 egt309-dev" command
Step 4: Start Jupyter Lab by clicking on the localhost to go to Jupyter Page (jupyter lab --ip=0.0.0.0 --no-browser --allow-root)
Step 5: Run the Necessary notebooks or file (ect. EDA)

- Method 3: Running the run.sh file

Step 1: Navigate to the project directory within a bash terminal (Git Bach, VS code Bash, Ubuntu)
Step 2: Create virtual environment (if it does not exist) using this "python3 -m venv venv" command
Step 3: Activate the virtual environment using this "source venv/bin/activate" command
Step 4: Ensure dependencies (from the requirements.txt) are installed using this "pip install -r requirements.txt" command
Step 5: Run the pipeline using "./run.sh"

## Description of logical steps/flow of the pipeline

1. Data loading - This project follows the standard kedro pipeline that processed data from raw ingestion, passes the cleaned database to machine learning models and finally provides a reporting output. The pipeline paths are defined in catalog.yml.

01_raw (The raw database is uploaded here.)
   ↓
02_intermediate (cleaning)
   ↓
03_primary (cleaned and structured data)
   ↓
04_feature (engineered features in the table)
   ↓
05_model_input (final ML-ready table)
   ↓
06_models (trained ML models)
   ↓
07_model_output (evaluation)
   ↓
08_reporting (final reports)

2. Data Cleaning & Standardization - The nodes.py file in the data_processing folder contains all the logic for cleaning and feature engineering. The pipelines.py file provides configuration values and parameters (from parameters_data_processing.yml) which prevents hard-coding and ensures modular, reusable codes.

3. Machine Learning Models - Machine learning model training is done inside the data_science pipeline. Similarly to data_processing the main ML logic resides within the nodes.py file, while model input, output, parameters and configuration are defined in pipelines.py, catalog.yml and parameters_data_science.yml. Four machine learning models are trained and evaluated.

4. Reporting & Outcomes - Final reporting and model summaries are handled under the reporting folder. This stage is used to organise the model results into readable outputs stored in the 08_reporting directory.

5. Routing & Logic - All dataset paths, storage locations, and parameter settings are managed through the catalog.yml file and the othet files under conf/base/. These files contain code defining how data flows between nodes and how configurations are applied across the entire pipeline.

## Overview of Key Findings from the EDA

The EDA revealed found several data quality issues. Age values contained text and unrealistic outliers which required cleaning and upper-bound capping. Categorical feilds showed inconsistent formatting and multiple versions of the same label. the Value 999 im "previous contact days" was confirmed to represent "no prior contact". Finally, EDA also revealed negative values under campaign calls which needed to be removed. Subscription rates were found to be imbalanced and were affected by customer attributes and factors such as age group, occupation, previous contact history and Contact method. Additional insights showed that excessive campaign calls reduced subscription likelihood while customers contacted via cellular channels were more responsive. These insights directly guided the data processing and feature engineering steps to train the ML models.

## How the features in the dataset were processed

1. Age processing 
    - Removed text attributes such as "years" in the column 
    - Convert values to numeric and set invalid entries to NaN 
    - Caps max age to 95 and removes outliers such as 150\
    - Enables both numeric age and binned age groups to be used for modeling.

2. Categorical Standerdization 
    - Converts all categorical feilds to lowercase 
    - removes spaces and inconsistencies 
    - Ensures consistent formatting across all columns 

3. Previous contact days
    - Converts columns to numeric 
    - identifies 999 as "no previous contact"
    - Creates a new indicator feature: Had Previous Contact (1/0).
    - Replaces 999 with NaN in order to avoid misleading values.

4. Encoding
    - Converts yes/no to to 0/1 values 
    - applied to columns such as Housing Loan, Credit Default, etc.

5. Feature Engineering
    - Age_Group: Assigns customers to bins for easier training during ML (Youth, Young Adult, Adult, Middle Aged, Senior).
    - Job_Type: Maps occupations into broader job categories/clusters.
    - Loan_Count: Combines housing and personal loan indicators to show loan activity.

6. Cleaning Invalid Numerical Values
    - Ensures fields such as Campaign Calls do not contain negative values.
    - Invalid records are corrected/set to NaN.

7. Droping redundant fields
    - Client ID
    - Previous Contact Days (replaced by cleaned fields)
    - Occupation (replaced by Job_Type)

## Explanation for choice of machine Learning Models

1. Random Forest Classifier	
- Identifies non-linear patterns in marketing practices
- Manages categorical variables and missing values.
- Provides feature importance (e.g., duration, past outcome) 

2. Gradient Boosting Classifier	
- High accuracy and increased robustness for imbalanced classes (no > yes).
- Handles intricate interactions among several features. 
- Perform well on structured datasets, similar to the bank_marketing

3. Logistic Regression
- Suitable for binary classifications (yes or no)
- Encodes and interprets mixed category and numerical data, 
- Highlighting key drivers such as age, job, and duration.
	
4. XGBoost Classifier	
- One of the strongest boosting algorithms
- Handles imbalanced, structured datasets well
- Captures complex non-linear behaviours that simpler models cannot capture

5. LightGBM Classifier	
- Efficient and Lightweight
- Strong performance and is able to handle large datasets accurately
- Trains quickly, suitable for quick testing

6. CatBoost Classifier	
- Designed specifically to handle categorial data
- Able to capture more complex feature interactions
- Helps reduce overfitting, suitable for the database

## Evaluations of Machine Learning Models

1. Numeric Metrics
 An evaluation node for each model computes the following metrics and stores them in a results dictionary, which is then translated into a DataFrame and plotted:
- Accuracy
 . Proportion of correct predictions.  
 . This provides a general sense of performance, but should be taken with caution due to the dataset's imbalance (more "no" than "yes").
- Precision (weighted)
 . How many of the samples predicted for a certain class are actually in that class.
 . Higher precision for the "yes" class means that fewer customers are incorrectly identified as possible subscribers.
- Weighted recall 
 . This measures the number of correctly classified samples in a class.
 . Higher recall for the "yes" class shows that the model is identifying more true subscribers.
- F1 score (weighted)
 . The harmonic mean of precision and recall.
 . This gives a single score that weighs capturing actual subscribers (recall) against avoiding too many false positives (accuracy), while enabling class imbalance.

2. Visual Diagnostics.
 In addition to scalar measurements, the reporting nodes generate many plots:
- Confusion Matrix
 . A heatmap for the best performing model was created by combining the true labels and predictions from the test dataset.
 . It displays true positives, true negatives, false positives, and false negatives, allowing you to identify the most prevalent sorts of errors.
- ROC Curves and AUC
 . The ROC curve and Area Under the Curve (AUC) are calculated for each model using the anticipated probabilities.
 . ROC-AUC assesses how well the model distinguishes between positive ("yes") and negative ("no") classes at various probability thresholds; a higher AUC indicates greater discrimination.

## Other considerations for deploying the models

- Consistent Data Pipeline
 . All models require the same preprocessing steps as training (encoding, column sorting, and managing missing variables).
 . Ensures that each model receives properly structured and consistent input.

- Probability Threshold Tuning
 . The default criterion of 0.5 may not provide an effective balance of precision and recall.
 . Thresholds can be set across all models or tweaked individually based on business requirements (for example, decreasing false positives versus catching more actual subscribers).

- Monitoring and Model Drift.
 . Customer behaviour varies with time, and each model may decline differently.
 . Regular monitoring with accuracy, precision, recall, F1-score, and AUC aids in detecting drift.
 . Significant declines should result in retraining or threshold adjustments.

- Multimodel Deployment Strategy
 . Choose whether to employ a single best model, an ensemble (e.g., voting or probability averaging), or different models for each customer category.
 . Ensures that the deployment meets performance, resource, and operational requirements.
 
- Interpretability and Explainability
 . Logistic regression yields visible coefficients.
 . Tree-based models (Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost) can be explained using feature importance or SHAP values.
 . Ensures predictions are understandable.

## Analysis of models' performance

By looking at the metrics below:

'catboost': accuracy': 0.8810238996996212
              'best_threshold': 0.65
              'f1': 0.363382250174703
              'precision': 0.44905008635578586
              'recall': 0.3051643192488263
              'roc_auc': 0.7219075658950026
 'gradient_boosting': 'accuracy': 0.8816768969570328
                       'best_threshold': 0.2
                       'f1': 0.3565340909090909
                       'precision': 0.45143884892086333
                       'recall': 0.29460093896713613
                       'roc_auc': 0.7179144546436098
 'lightgbm': 'accuracy': 0.8798485046362805
              'best_threshold': 0.65
              'f1': 0.36464088397790057
              'precision': 0.4429530201342282
              'recall': 0.30985915492957744
              'roc_auc': 0.7149915658536079
 'log_reg':'accuracy': 0.8865090766618781
             'best_threshold': 0.65
             'f1': 0.35772357723577236
             'precision': 0.48303393213572854
             'recall': 0.284037558685446
             'roc_auc': 0.7201380336882919
 'random_forest': 'accuracy': 0.8823298942144443
                   'best_threshold': 0.65
                   'f1': 0.34662799129804206
                   'precision': 0.45351043643263755
                   'recall': 0.2805164319248826
                   'roc_auc': 0.7114500005174322
 'xgboost': 'accuracy': 0.8825910931174089
             'best_threshold': 0.65
             'f1': 0.3637650389242746
             'precision': 0.45811051693404636
             'recall': 0.30164319248826293
             'roc_auc': 0.7183190004587899

We can see that all models achieved a high accuracy. However, this metric is not helpful in our case due to the extreme class imbalance. This results in the models being able to achieve high accuracies by simply predicting the "no" class for majority of the cases. Hence, we mainly used F1 score and ROC-AUC graphs.

Across our 6 models, LightGBM achieved the highest F1 score after the optimisation, which shows how it has the best balance between precision and recall for our "yes" class. The other gradient-boost models also performed decently well, having a performance just below the LightGBM model. This shows how gradient-boosted models are well-suited for handling datasets such as this bmarket.db. These models are different from the rest, as they have an ability to capture non-linear relationships and can handle categorial varaiables well.

Other models such as the Logistic regression, performed slightly worse than the gradient-boost models. This could possibly be due to it relying on linear boundaries and is unable to capture the more complex patterns in the dataset. Random forest was also weaker than the gradient-boost models, likely due to boosting models being able to perform better than bagging models when there is data that has weak predictors.

The threshold tuning we used helped to evaluate a wider range from 0.1 to 0.9 and improved the F1 scores of all the models we used. This shows how threshold tuning is important as it could lead to substantial improvements in the results.

Our ROC-AUC values across all the models wer similar at around 0.71 to 0.72. This shows how our model is considerably decent, as even though there was a class imbalance, it still managed to reach a score above 0.7. 

We came to the conclusion that LightGBM is the best model after looking through our results and evaluating them. It has higher scores across all the metrics, which shows that it captured the patterns more effectively than the others. However, all of the models did face limitations to their performance due to the apparent class imbalance and the limited predictive strength of the features provided.

## Possible Follow-Up Actions

- Improve class imbalance handling

- Collect additional predictive features

- Campaign Targeting
