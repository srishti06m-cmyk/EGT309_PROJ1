## Group member names + email

Name: Srishti Malapuram
Email: 233106B@mymail.nyp.edu.sg

Name:
Email:

Name: 
Email:

## Project Overview 

This project focuses on the deployment of an end-to-end machine learning pipeline to predict weather or not a banking customer will subscribe to a term deposit. The given dataset contains customer details such as demographic information, financial history, age and subscription details. The goal of this project is to:

1. Clean and process raw data from a SQLite database
2. Derive insights and visualization through EDA
3. Perform Data processing and feature engineering based on EDA findings 
4. Build a reliable ML pipeline using kedro 
5. Select and Train most suitable models.
6. Report findings and results

## How to run this project

## Descrip/on of logical steps/flow of the pipeline

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

