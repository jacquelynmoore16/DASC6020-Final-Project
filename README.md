# CSCI6020-Final-Project
Final project for CSCI 6020 at ECU

# KNN Regression Model for Landings Data

## Project Overview
This project aims to build a K-Nearest Neighbors (KNN) regression model to predict annual fisheries landings in Alaska based on various oceanographic features. The model is trained and evaluated using data filtered by specific regulation areas.

## Data
The dataset used in this project is `iphc_fulldataset_full_depth.csv`, which contains the following features:
- `avg_temp`: Average Temperature
- `avg_oxy`: Average Oxygen
- `avg_chloro`: Average Chlorophyll
- `avg_salin`: Average Salinity
- `year`: Year
- `annual_landings`: Annual Landings (target variable)
- `RegArea`: Regulation Area
- `stnno`: Station Number

## Requirements
- MATLAB R2021a or later
- Statistics and Machine Learning Toolbox
- R and RStudio (for R Markdown)

## Project Structure
- `Moore_Miller_FinalProject_Preprocessing_CSCI6020.Rmd`: R Markdown file for initial data analysis and visualization.
- `Moore_Miller_FinalProject_CSCI6020.m`: Main script to run the KNN regression model.
- `iphc_fulldataset_full_depth.csv`: Dataset file.
- `iphc2009.csv` : 2009 Data File
- `iphc2010.csv` : 2010 Data File
- `iphc2011.csv` : 2011 Data File
- `iphc2012.csv` : 2012 Data File
- `iphc2013.csv` : 2013 Data File
- `iphc2014.csv` : 2014 Data File
- `iphc2015.csv` : 2015 Data File
- `iphc2016.csv` : 2016 Data File
- `iphc2017.csv` : 2017 Data File
- `iphc2018.csv` : 2018 Data File


## Usage
### R Markdown
1. **Open `Moore_Miller_FinalProject_Preprocessing_CSCI6020.Rmd`**: Open the R Markdown file in RStudio.
2. **Run the Chunks**: Execute the code chunks to perform data analysis and visualization.
3. **Generate Report**: Knit the document to produce a comprehensive report in HTML, PDF, or Word format.
4. 
### MATLAB
1. **Open `Moore_Miller_FinalProject_CSCI6020.m`**: Open the R Markdown file in RStudio.
2. **Load the Data**: The script reads the dataset from `iphc_fulldataset_full_depth.csv`.
3. **Filter by Regulation Area**: The user is prompted to enter a regulation area to filter the data.
4. **Preprocess the Data**: The script handles missing values and converts categorical variables to numeric.
5. **Train and Evaluate the Model**: The script splits the data into training and testing sets, standardizes the features, and trains a KNN model. It also performs cross-validation to evaluate the model.
6. **User Input for Prediction**: The user can input values for the features to get a prediction for annual landings.

