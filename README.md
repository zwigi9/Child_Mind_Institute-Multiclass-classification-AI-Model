# 🏆📖*Predicting Problematic Internet Usage Based on Physical Activity*

Welcome to our GitHub repository for the "Predicting Problematic Internet Usage" competition! This project aims to address a growing concern in today’s digital age—problematic internet use among children and adolescents. By leveraging physical activity and fitness data, we strive to build a predictive model capable of identifying early signs of problematic internet behavior.

## Competition Overview

The goal of this competition is to analyze children’s physical activity data and predict their level of problematic internet usage. This initiative is critical in promoting early interventions that encourage healthier digital habits and prevent mental health issues such as anxiety and depression.

## Why This Matters

Traditional methods of measuring problematic internet use often involve complex clinical assessments, which can be inaccessible due to cultural, linguistic, or logistical barriers. On the other hand, physical activity and fitness data are widely available and require minimal intervention, making them ideal proxies for identifying problematic internet use. Excessive technology use often manifests through changes in physical behavior, such as reduced activity levels and poorer posture. By harnessing these indicators, we aim to create scalable solutions to this growing problem.

## Competition Details

- **Start Date:** September 19, 2024
- **Entry Deadline:** December 12, 2024
- **Team Merger Deadline:** December 12, 2024
- **Final Submission Deadline:** December 19, 2024

### Prizes
- **1st Place:** $15,000
- **2nd Place:** $10,000
- **3rd Place:** $8,000
- **4th - 8th Places:** $5,000 each

## Data Source

This competition utilizes data provided by the **Healthy Brain Network**, a mental health study conducted by the Child Mind Institute. This initiative is supported by the California Department of Health Care Services.
The dataset contains data from around 5,000 participants aged 5-22 years, focusing on identifying biological markers for mental health and learning disorders. This competition uses two data types:  
1. **Physical Activity Data**: Wrist-worn accelerometer readings, fitness assessments, and activity questionnaires.  
2. **Internet Usage Data**: Behavioral data related to internet use.  

The goal is to predict the **Severity Impairment Index (SII)**, which measures problematic internet use on a scale from 0 (None) to 3 (Severe).  

### Data Structure  
- **Accelerometer Data**: Stored in Parquet files (`series_train.parquet` and `series_test.parquet`), capturing time-series data for each participant over multiple days.  
- **Tabular Data**: Stored in CSV files (`train.csv` and `test.csv`) and includes demographic, fitness, health, and internet usage data. Field descriptions are in `data_dictionary.csv`.  

### Key Features  
- **Accelerometer Fields**:  
  - `X`, `Y`, `Z`: Acceleration along each axis.  
  - `enmo`: Euclidean Norm Minus One, representing overall movement.  
  - `anglez`: Arm angle relative to the horizontal plane.  
  - `non-wear_flag`: Indicates if the device was worn (0: watch is being worn, 1: the watch is not worn).
  - `light`: Ambient light in lux.  
  - `time_of_day`, `weekday`, `quarter`: Contextual time details.  
  - `relative_date_PCIAT`: Days since the PCIAT test.  

- **Instruments Summary**:

  - `Demographics`: Age and sex of participants.  
  - `Internet Use`: Daily hours spent on computers or the internet.  
  - `Children's Global Assessment Scale`: Rates general functioning of youths under 18.  
  - `Physical Measures`: Includes blood pressure, heart rate, height, weight, waist, and hip measurements.  
  - `FitnessGram Vitals & Treadmill`: Cardiovascular fitness via NHANES treadmill protocol.  
  - `FitnessGram Child`: Assesses aerobic capacity, muscular strength, endurance, flexibility, and body composition.  
  - `Bio-electric Impedance Analysis`: Measures BMI, fat, muscle, and water content.  
  - `Physical Activity Questionnaire`: Tracks participation in vigorous activities over the past week.  
  - `Sleep Disturbance Scale`: Categorizes sleep disorders in children.  
  - `Parent-Child Internet Addiction Test (PCIAT)`: Evaluates compulsive internet behaviors like escapism and dependency.  

### Additional Details  
- **Missing Data**: Many fields are missing for some participants, including the target SII in parts of the training set.  
- **Test Set**: The test set (hidden during the competition) contains around 3,800 instances, with complete SII values.  

## Acknowledgments

Special thanks to our sponsors, the **Child Mind Institute**, **Dell Technologies**, and **NVIDIA**, for their support in making this competition possible. By contributing to this challenge, you are helping pave the way for healthier digital habits and a brighter future for children and adolescents worldwide.


# 📊🔍*Exploratory Data Analysis*
<p align="center">
  <img src="https://github.com/user-attachments/assets/7f56ae4f-ca94-4a85-a51f-7eeb20c88d7a">
</p>

**It appears that the enrollment is nearly the same across all seasons. After testing, we came to the conclusion that it is better to not use season specific columns.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/6a8b4bc1-979d-4dd7-a68d-601fbd1a8662">
</p>

**In our dataset, the majority of individuals use the internet for less than an hour, while those using it for more than 3 hours are quite few.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/020cc65e-b6a7-481a-b308-f3bc73455d59">
</p>

**The pie chart above clearly shows that only 9.9% of people use the internet excessively, while the rest of the users either spend around 2 hours or less online.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/92eab349-809b-4651-ad63-61fbaae5d9b0">
</p>

**The pie chart clearly shows that the majority of people, 58.3%, have no problems, while 26.7% experience only mild issues. This means that 85% of individuals in our dataset have either no problems or mild ones. In contrast, 15% of the population falls into more serious categories, with 13.8% having moderate problems and only 1.2% displaying severe issues. Therefore, in our dataset, only 1.2% of people have a severe problem, which is a very small percentage compared to the other three groups.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/801531fa-da07-4acd-aa48-9f4997acd1b6">
</p>

**Here, we have a scatter plot for our target variable, SII, and the number of hours of internet usage. While the plot is interesting, it is also somewhat perplexing. It shows that all possible SII values are present across all ranges of internet usage hours per day, making it difficult to discern how the SII value changes with varying internet usage time. A potential solution to better understand this relationship would be to include a count of computer/internet users along with the SII and internet usage hours per day. This could help clarify the distribution and trend.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/c98d11c5-1112-4b82-8573-50778307f71e">
</p>

**In the above chart, we have the SII, PreInt_EduHx-computerinternet_hoursday, and the count of PreInt_EduHx-computerinternet_hoursday. These three variables together provide a more comprehensive understanding of the data compared to the previous chart, which only used two features. By incorporating the count, we can better visualize and interpret the relationship between internet usage hours, educational history, and the severity of impairment.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/6a5e95dd-7353-44bc-ac34-5b2753dd96bc">
</p>

**The majority of people in this dataset are male, although there is also a significant number of females represented.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/e79bca87-a373-4b64-bd1d-0da20fa17515">
</p>

**The majority of people in our dataset are young children or teenagers, with no elderly individuals present. It's important to note that we are tasked with determining the Severity Impairment Index (SII) for people in the future.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/48419bb8-ac72-4462-a4f5-4c04ecce41f2">
</p>

**The scatter plot above shows that the Sleep Disturbance Scale values for individuals with SII scores of 0, 1, and 2 fall within a similar range, with only a few outliers. For SII 3, we observe fewer entries, which can explain the smaller number of marks/dots in that section of the plot. However, even though there are fewer dots for SII 3, they still lie within the same range as those for SII 0, 1, and 2. The range for SII 3 is smaller compared to SII 0, 1, and 2, but I refer to it as the "same range" because there aren't a significant number of points outside the range of the other SII scores. A surprising finding is that for SII 3, we don't observe very high or very low sleep disturbance values. Ideally, we would expect SII 3 to correspond to higher sleep disturbance scores, but this isn't reflected in the data.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/a1efdbec-6298-48e6-836e-9ad1416c8139">
</p>

**The violin plot illustrates the distribution of data points, showing that the majority of points fall within the sleep disturbance scale range of 30 to 50. This distribution appears quite similar for SII values of 0, 1, and 2. However, for SII value 3, the data points seem slightly fewer compared to those for SII values 0, 1, and 2.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/ca32c59e-1689-44ea-8b8d-8a57c5aecbdb">
</p>

**The recorded values for BIA BMI and Physical BMI appear to be slightly different. There are two possible reasons for this:**

**1. There may be some discrepancy or error in the measurement of BMI.**

**2. The two BMIs could have been calculated at different ages for the same individual.**

# Our First Attempt

Our first target was to make it work, just to have some reference for the future. 

## Handling missing features

We started with dropping some unnecessery columns, which were: `id`, `sii`(target column) and `Physical-Waist_Circumference` (because it was mostly empty, more than 60%). Then we split the rest into numerical and categorical columns. After that we used SimpleImputer to handle missing data. For the numerical columns it was imputed with `median` value. And for the categorical columns we used `most_frequent` value. 

```python
# Step 1.1: Split the training data into features (X) and target (y)
X_train = train.drop(columns=['id', 'Physical-Waist_Circumference', 'sii'])  # Omit 'id', 'Physical-Waist_Circumference', 'sii'
y_train = train['sii']  # Our target is 'sii'

# Step 1.2: Split the testing data into features (X)
X_test = test.drop(columns=['id', 'Physical-Waist_Circumference']) # Omit 'id', 'Physical-Waist_Circumference',

# Step 2: Split the features into numerical and categorical based on data type
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Step 2.1:Impute numerical columns with median value
numerical_imputer = SimpleImputer(strategy='median')
train[numeric_cols] = numerical_imputer.fit_transform(train[numeric_cols])

# Step 2.2:Impute categorical columns with the most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
train[categorical_cols] = categorical_imputer.fit_transform(train[categorical_cols])
```
## Handling missing classes in target column with regressor

**We found out that there is a lot of missing classes in target column, around 30%.**

<p align="center">
  <img src="https://github.com/user-attachments/assets/c9c942f3-17f4-4bea-b1eb-50887183bab3">
</p>

So, we decided on predicting missing classes with regressor. First, we separated rows with and without missing `sii`. Then we defined a preprocessor to encode categorical features using `One-Hot Encoder` (OHE) before training. We used this encoder instead of a simpler `Label Encoder` (LE), because LE can introduce unintended ordinal relationships for nominal data. In our case where every categorical feature is connected to the four seasons we didn't want it to have some sort of a hierarchical order. On the other hand, OHE creates binary columns for each category (no inherent order).

After that, we split the `train_no_missing` into features (X) and target (y). Next, we preprocessed the data and trained the Random Forest Regressor. Subsequently, we predicted the missing `sii` values and imputed these predictions into our main Data Frame. Lastly, we checked if all values where imputed correctly and did some basic model evaluation using `Mean Squared Error` (MSE).

```python
# Step 3: Separate rows with missing 'sii' and not missing 'sii'
train_no_missing_sii = train[train['sii'].notnull()]
train_missing_sii = train[train['sii'].isnull()]

# Step 4: Preprocess categorical features with one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer', categorical_imputer),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ])

# Step 5: Split the train_no_missing into features (X) and target (y)
X_train_no_missing = train_no_missing_sii[X_train.columns]
y_train_no_missing = train_no_missing_sii['sii']

# Preprocess the data
X_train_no_missing_processed = preprocessor.fit_transform(X_train_no_missing)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_no_missing_processed, y_train_no_missing)

# Step 6: Predict missing 'sii' values
X_train_missing = train_missing_sii[X_train.columns]
X_train_missing_processed = preprocessor.transform(X_train_missing)
predicted_sii = model.predict(X_train_missing_processed)

# Step 7: Impute the missing 'sii' values with the predictions
train.loc[train['sii'].isnull(), 'sii'] = predicted_sii

# Step 8: Check missing values after imputation
print("\nMissing values after imputation:")
print(train.isnull().sum())

# Evaluate the model performance using Mean Squared Error (MSE)
X_train_no_missing_processed = preprocessor.transform(X_train_no_missing)
y_pred = model.predict(X_train_no_missing_processed)
mse = mean_squared_error(y_train_no_missing, y_pred)
print(f"\nMean Squared Error on training set: {mse}")
```
**Console output:**
```bash
Missing values after imputation:
id                                        0
Basic_Demos-Enroll_Season                 0
Basic_Demos-Age                           0
Basic_Demos-Sex                           0
CGAS-Season                               0
                                         ..
SDS-SDS_Total_Raw                         0
SDS-SDS_Total_T                           0
PreInt_EduHx-Season                       0
PreInt_EduHx-computerinternet_hoursday    0
sii                                       0
Length: 82, dtype: int64

Mean Squared Error on training set: 1.7543859649122841e-06
```
We started by identifying common columns in train and test data sets, because test data set didn't have all of them. Then we updated the `numeric_cols` and `categorical_cols` variables so they only contained the common columns. After that, to avoid unnecessary errors, we had to define preprocessor one more time. Then, we preprocessed both the training and test data and trained the model with a simple `RandomForestClassifier`. Lastly, we predicted using the trained model on the preprocessed test data and save the results to `submission.csv`.

```python
# Step 9: Identify common columns
common_cols = X_train.columns.intersection(X_test.columns)

# Step 10: Handle numeric and categorical columns separately
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.intersection(common_cols)
categorical_cols = X_train.select_dtypes(include=['object']).columns.intersection(common_cols)

# For categorical features, apply one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer', categorical_imputer),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
        ]), categorical_cols)
    ])

# Step 11: Preprocess both the training and test data
X_train_preprocessed = preprocessor.fit_transform(X_train[common_cols])
X_test_preprocessed = preprocessor.transform(X_test[common_cols])

# Step 12: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_preprocessed, y_train)

# Step 13: Predict using the trained model on the preprocessed test data
predictions = model.predict(X_test_preprocessed)

# Step 24: Prepare submission file
submission = pd.DataFrame({
    'id': test['id'],  # Assuming 'id' is the column in test.csv
    'sii': predictions.astype(int)  # Ensure predictions are integers (for classification)
})

# Step 25: Save to submission.csv
submission.to_csv('submission.csv', index=False)
print('Saved to submission.csv')
```

<table style="border-collapse: collapse; border: 1px solid black; border-radius: 15px; background-color: #cce7ff; padding: 5px;">
  <tr>
    <td style="text-align: center; font-family: Arial, sans-serif; font-size: 50;">
      <img src="https://github.com/user-attachments/assets/7e141746-d572-4e52-96cf-31a1f01f3a97" alt="Icon" style="width: 20px; vertical-align: down; margin-right: 10px;">
      <b>Current Kaggle Score: 0.186</b>
    </td>
  </tr>
</table>


# Further Testing

## Feature Importance Anal(ysis)
After our successful first attempt we wanted to test if we need to use all the columns, so we created a feature importance plot to determine which ones are most important for our model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/eae8fe31-d202-457b-a598-3620afa86f1e">
</p>

According to this plot all of the categorical data (Seasonal data) was irrelevant. For a long time we'd used top 15 features from this plot. But when we used feature importance plot again, this time on our best performing model, the results quite differed.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bd21c9f3-eef9-4e44-b97f-534adbda267d">
</p>

Using the last plot as reference we finally ended up using following columns: `Physical-Height`, `Physical-Weight`, `Basic_Demos-Age`, `SDS-SDS_Total_Raw`, `Physical-BMI`, `SDS-SDS_Total_T`, `Physical-Systolic_BP`, `Physical-HeartRate`, `CGAS-CGAS_Score`, `PreInt_EduHx-computerinternet_hoursday`, `Basic_Demos-Sex`, `Physical-Diastolic_BP`, `PAQ_Total`.

## How to Handle 'sii' - Handling missing classes in target column. Part 2
Initially, in our first attempt, we used a `RandomForestRegressor` to impute missing values in the target column `sii` ([here](https://github.com/zwigi9/Child_Mind_Institute-Multiclass-classification-AI-Model/blob/main/README.md#handling-missing-classes-in-target-column-with-regressor)). Later, we considered an alternative approach: simply removing rows with missing `sii` values. After conducting several tests using the same model but handling the missing `sii` differently, we found that the best-performing method was to delete these rows. This approach proved effective, and we stuck to it for the remainder of the project.

## Parquet Files Adventures
### First Method
At first, we struggled with loading the Parquet files because the basic pandas functions couldn’t handle their large size within Kaggle’s runtime limits. After building a few simple models, we revisited the problem and searched online for solutions. We found a code snippet that appeared in multiple competition notebooks, offering a better way to process the time series data in Parquet format.

The snippet introduces a function, `load_time_series`, designed to efficiently handle Parquet files in a directory. It uses `os.listdir` to get the file names and `ThreadPoolExecutor` to process multiple files at the same time. Each file is passed to a helper function, `process_file`, which reads the Parquet file, removes the `step` column, and calculates summary statistics using `df.describe()`. These statistics include useful details about the data, such as how many values are present (count), the average (mean), how spread out the data is (standard deviation), and key percentiles (minimum, 25%, 50%, 75%, and maximum). These numbers give a quick but detailed picture of the data in each file. The statistics are flattened into a single row, and the file ID is extracted from the file name and included with the results.

The `load_time_series` function combines these rows into a DataFrame, where each column represents one of the statistics, and an `id` column identifies which file the data came from. This approach makes it much faster and easier to work with the time series data, as it turns large, complex datasets into compact summaries that are ready for modeling.

The resulting DataFrames are then merged with the main datasets (`train.csv`, `test.csv`) using the `id` column to match them. Once merged, the `id` column is no longer needed and is dropped, leaving the data clean and ready for analysis.

```python
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]

def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
            
    stats, indexes = zip(*results)

    df = pd.DataFrame(stats, columns=[f"Stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes

    return df

train_parquet_path = "series_train.parquet"
test_parquet_path = "series_test.parquet"

train_ts = load_time_series(train_parquet_path)
test_ts = load_time_series(test_parquet_path)

time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")

train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')
```
<table style="border-collapse: collapse; border: 1px solid black; border-radius: 15px; background-color: #cce7ff; padding: 5px;">
  <tr>
    <td style="text-align: center; font-family: Arial, sans-serif; font-size: 50;">
      <img src="https://github.com/user-attachments/assets/7e141746-d572-4e52-96cf-31a1f01f3a97" alt="Icon" style="width: 20px; vertical-align: down; margin-right: 10px;">
      <b>Current Kaggle Score: 0.162</b>
    </td>
  </tr>
</table>

**This version uses code from the [First Attempt](https://github.com/zwigi9/Child_Mind_Institute-Multiclass-classification-AI-Model/edit/main/README.md#our-first-attempt) + First Method (this one) of parquet handling.**

### Second Method TBC

<table style="border-collapse: collapse; border: 1px solid black; border-radius: 15px; background-color: #cce7ff; padding: 5px;">
  <tr>
    <td style="text-align: center; font-family: Arial, sans-serif; font-size: 50;">
      <img src="https://github.com/user-attachments/assets/7e141746-d572-4e52-96cf-31a1f01f3a97" alt="Icon" style="width: 20px; vertical-align: down; margin-right: 10px;">
      <b>Current Kaggle Score: NaN</b>
    </td>
  </tr>
</table>

**This version uses code from the [First Attempt](https://github.com/zwigi9/Child_Mind_Institute-Multiclass-classification-AI-Model/edit/main/README.md#our-first-attempt) + Second Method (this one) of parquet handling.**

### Third Method TBC

<table style="border-collapse: collapse; border: 1px solid black; border-radius: 15px; background-color: #cce7ff; padding: 5px;">
  <tr>
    <td style="text-align: center; font-family: Arial, sans-serif; font-size: 50;">
      <img src="https://github.com/user-attachments/assets/7e141746-d572-4e52-96cf-31a1f01f3a97" alt="Icon" style="width: 20px; vertical-align: down; margin-right: 10px;">
      <b>Current Kaggle Score: NaN</b>
    </td>
  </tr>
</table>

**This version uses code from the [First Attempt](https://github.com/zwigi9/Child_Mind_Institute-Multiclass-classification-AI-Model/edit/main/README.md#our-first-attempt) + Third Method (this one) of parquet handling.**

# Our Best and Final Attempt TBC

<table style="border-collapse: collapse; border: 1px solid black; border-radius: 15px; background-color: #cce7ff; padding: 5px;">
  <tr>
    <td style="text-align: center; font-family: Arial, sans-serif; font-size: 50;">
      <img src="https://github.com/user-attachments/assets/7e141746-d572-4e52-96cf-31a1f01f3a97" alt="Icon" style="width: 20px; vertical-align: down; margin-right: 10px;">
      <b>Final Kaggle Score: 0.402</b>
    </td>
  </tr>
</table>
