# DASC6020-Final-Project
Final project for DASC 6020 at ECU

# KNN Regression Model for Landings Data

## Project Overview
This project aims to build a K-Nearest Neighbors (KNN) regression model to predict annual landings based on various environmental features. The model is trained and evaluated using data filtered by specific regulation areas.

## Data
The dataset used in this project is `iphc_fulldataset.csv`, which contains the following features:
- `avg_temp`: Average Temperature
- `avg_oxy`: Average Oxygen
- `avg_chloro`: Average Chlorophyll
- `avg_salin`: Average Salinity
- 'year' : Year
- `annual_landings`: Annual Landings (target variable)
- `RegArea`: Regulation Area
- `stnno`: Station Number

## Requirements
- MATLAB R2024a or later
- Statistics and Machine Learning Toolbox

## Project Structure
- `Moore_FinalProject_Dasc6020.m`: Main script to run the KNN regression model.
- `iphc_fulldataset_full_depth.csv`: Dataset file.

## Usage
1. **Load the Data**: The script reads the dataset from `iphc_fulldataset.csv`.
2. **Filter by Regulation Area**: The user is prompted to enter a regulation area to filter the data.
3. **Preprocess the Data**: The script handles missing values and converts categorical variables to numeric.
4. **Train and Evaluate the Model**: The script splits the data into training and testing sets, standardizes the features, and trains a KNN model. It also performs cross-validation to evaluate the model.
5. **User Input for Prediction**: The user can input values for the features to get a prediction for annual landings.

## How to Run
1. Open MATLAB and navigate to the project directory.
2. Run the `Moore_FinalProject_Dasc6020.m` script.
3. Follow the prompts to enter the regulation area and feature values for prediction.
