
# Data Preprocessing with Python

This repository contains a data preprocessing pipeline using Python, designed to prepare datasets for further analysis or modeling. The datasets used are train.csv and test.csv, which undergo cleaning, encoding, and transformation before being exported as processed data.






## Features

- Data Loading: Load train and test datasets from CSV files.
- Data Inspection: Check the shape and structure of the data, along with missing values.
- Target Encoding: Convert the target variable 'y' from categorical (yes/no) to numeric (1/0).
- Feature Preprocessing:
    - Standardize numerical features.
    - Apply one-hot encoding to categorical features.
- Pipeline Automation: Use ColumnTransformer to streamline preprocessing steps for both numerical and categorical data.
- Export Processed Data: Save the processed train and test datasets as CSV files for further use.


## Requirements
To run the preprocessing pipeline, ensure that the following Python libraries are installed:
- pip install pandas scikit-learn

## Project Structure
- dataset/: Folder containing the original train.csv and test.csv files.
- preprocessing.py: Python script that handles the preprocessing steps.
- README.md: This documentation file.
## Dataset
The datasets (train.csv and test.csv) should be placed in the dataset/ directory. Each dataset must contain features and the target variable 'y' which is transformed during the preprocessing step.
## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License. Feel free to modify and use it as per your needs.

