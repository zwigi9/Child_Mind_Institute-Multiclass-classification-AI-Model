# 🏆📖*Machine Learning Challenges: Addressing Heterogeneous Data Formats, Missing Labels, and Unbalanced Data*
**This is a review of our approach to solve Child's Mind Institue competition on Kaggle.**

*https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use*

*Authors: Adam Tokarski, Maksymilian Wiciński*
### ⭐ Introduction

This project focused on solving a multiclass classification problem with an imbalanced dataset, predicting the Severity Impairment Index (SII) (0-3) for individuals based on physical activity and fitness data.

- 🎯 Goal: Predict SII scores to assess problematic internet usage impacts.
- 📊 Data Handling: Utilized mixed-format datasets (`CSV` and `Parquet`), performed feature engineering, and addressed missing values and class imbalance using resampling (`SMOTE-Tomek`).
- 🧠 Modeling:
  - Developed a custom ensemble approach, training `Random Forest`, `XGBoost`, and `LightGBM` classifiers.
  - Applied class-specific model selection for optimal performance on individual SII categories.
  - Evaluated models using a validation set split from the resampled training data, calculating `Quadratic Weighted Kappa (QWK)` scores for each class to ensure model performance across all classes.
- 🛠 Tools, Libraries & Techniques:
  - Designed an end-to-end data preprocessing pipeline using `scikit-learn`, `imbalanced-learn`, `pandas`, and `numpy`.
  - Leveraged time-series data preprocessing and feature selection tailored to the problem.
  - Created data visualizations and feature importance plots using `matplotlib` and `seaborn` to gain insights and enhance interpretability.
  - Worked within a Kaggle competition framework, meeting runtime constraints.

### 🎓 This project enhanced our skills in:

- Building, testing and picking the best ensemble models for multiclass classification tasks.
- Tackling imbalanced classification problems and effectively applying resampling techniques.
- Analyzing model performance through custom metrics and cross-validation strategies.
- Designing versatile data preprocessing pipelines for mixed-format datasets.
- Visualizing data and interpreting feature importance to make informed decisions.
- Applying critical thinking and creativity to solve real-world ML problems.

These experiences have honed our expertise in analyzing machine learning behaviors and developing effective solutions in competitive settings.

# 🥇🗿Our Best and Final Attempt

<table style="border-collapse: collapse; border: 1px solid black; border-radius: 15px; background-color: #cce7ff; padding: 5px;">
  <tr>
    <td style="text-align: center; font-family: Arial, sans-serif; font-size: 50;">
      <img src="https://github.com/user-attachments/assets/7e141746-d572-4e52-96cf-31a1f01f3a97" alt="Icon" style="width: 20px; vertical-align: down; margin-right: 10px;">
      <b>Our Final Kaggle Score: 0.423</b>
    </td>
  </tr>
</table>

<table style="border-collapse: collapse; border: 1px solid black; border-radius: 15px; background-color: #cce7ff; padding: 5px;">
  <tr>
    <td style="text-align: center; font-family: Arial, sans-serif; font-size: 50;">
      <img src="https://github.com/user-attachments/assets/7e141746-d572-4e52-96cf-31a1f01f3a97" alt="Icon" style="width: 20px; vertical-align: down; margin-right: 10px;">
      <b>1<sup>st</sup> Place Kaggle Score: 0.482</b>
    </td>
  </tr>
</table>

## 🔨 Data Wrangling and Feature Engineering

- **Dataset Loading**

- **Feature Selection**: 
  - Removed unnecessary columns such as `id` and `Physical-Waist_Circumference` (70% data missing) from both the training and testing datasets. 
  - Dropped rows in the training dataset with missing values in the target column `sii`, as they couldn't contribute to model training.

- **Splitting into `X_train` (features) and `y_train` (target)**

- **Feature Engineering**:  
  - Merged two related columns (`PAQ_A-PAQ_A_Total` and `PAQ_C-PAQ_C_Total`) into a single composite feature `PAQ_Total`. This step consolidated similar data to reduce dimensionality and improve interpretability.  
  - Handled missing values in these columns by substituting them with zeros before merging.  
  - Removed the original columns post-merge to maintain a clean and concise feature set.  

This systematic approach to data wrangling and feature engineering optimized the dataset for robust model training, a critical skill in machine learning workflows.

```python
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train = train.drop(columns=['id', 'Physical-Waist_Circumference'])  # Drop unnecessary columns
train = train.dropna(subset=['sii'])  # Remove rows with missing 'sii' values

# Split the cleaned training data into features (X) and target (y)
X_train = train.drop(columns=['sii'])
y_train = train['sii']

# Prepare test set
X_test = test.drop(columns=['id', 'Physical-Waist_Circumference'])  # Drop unnecessary columns

# Merge PAQ_A-PAQ_A_Total and PAQ_C-PAQ_C_Total in X_train
if 'PAQ_A-PAQ_A_Total' in X_train.columns and 'PAQ_C-PAQ_C_Total' in X_train.columns:
    X_train['PAQ_Total'] = X_train['PAQ_A-PAQ_A_Total'].fillna(0) + X_train['PAQ_C-PAQ_C_Total'].fillna(0)
    X_train = X_train.drop(columns=['PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total'])  # Drop the original columns

# Merge PAQ_A-PAQ_A_Total and PAQ_C-PAQ_C_Total in X_test
if 'PAQ_A-PAQ_A_Total' in X_test.columns and 'PAQ_C-PAQ_C_Total' in X_test.columns:
    X_test['PAQ_Total'] = X_test['PAQ_A-PAQ_A_Total'].fillna(0) + X_test['PAQ_C-PAQ_C_Total'].fillna(0)
    X_test = X_test.drop(columns=['PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total'])  # Drop the original columns
```
## ⚙ Numerical Data Preprocessing

- **Column Identification**:  
  - Determined common columns between the training and testing datasets using `.intersection()`, ensuring consistent feature usage.  
  - Identified numerical columns based on data types (`int64` and `float64`) and predefined relevant features, removing any mismatched columns for a unified preprocessing pipeline.
  - Used columns highlighted on the Feature Importance Plot.

- **Imputation Strategy**:  
  - Implemented the `SimpleImputer` with a median strategy to handle missing values in numerical columns, effectively reducing bias from outliers.  

- **Pipeline Configuration**:  
  - Utilized `ColumnTransformer` to apply the imputation only to the selected numerical columns.  
  - Transformed the training data and restructured it into a new DataFrame with proper column names for clarity and downstream processing.  

- **Test Set Integration**:  
  - Adjusted the imputer strategy to the mean and applied the preprocessor to both the training and testing datasets, ensuring consistency in handling missing values across datasets.  

This approach highlights our proficiency in preparing datasets for machine learning by seamlessly integrating feature selection, imputation, and column transformations, all while maintaining compatibility between training and testing data.

```python
common_cols = X_train.columns.intersection(X_test.columns)
numeric_cols_not_list = X_train.select_dtypes(include=['int64', 'float64']).columns.intersection(X_test.columns)
numerical_imputer = SimpleImputer(strategy='median')

numeric_cols = [
    "Physical-Height",
    "Physical-Weight",
    "Basic_Demos-Age",
    "SDS-SDS_Total_Raw",
    "Physical-BMI",
    "SDS-SDS_Total_T",
    "Physical-Systolic_BP",
    "Physical-HeartRate",
    "CGAS-CGAS_Score",
    "PreInt_EduHx-computerinternet_hoursday",
    "Basic_Demos-Sex",
    "Physical-Diastolic_BP",
    'PAQ_Total'
]

for col in numeric_cols:
    if col not in numeric_cols_not_list:
        numeric_cols.remove(col)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numeric_cols),
    ])
subset_x_train = X_train[numeric_cols]

X_train = preprocessor.fit_transform(subset_x_train)

# Create the DataFrame with proper column names
X_train_preprocessed_df = pd.DataFrame(X_train, columns=numeric_cols)

numerical_imputer = SimpleImputer(strategy='mean')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numeric_cols),
    ])

preprocessor.fit(X_train_preprocessed_df)
# Transform both training and test sets
common_cols = X_train_preprocessed_df.columns.intersection(X_test.columns)
X_train_preprocessed = preprocessor.transform(X_train_preprocessed_df)
X_test_preprocessed = preprocessor.transform(X_test[numeric_cols])
```
## 🚀 Ensemble Modeling and Prediction
These are the steps we took to balance the dataset, train multiple machine learning models, and make class-specific predictions using an ensemble approach.
### Balancing the Dataset
To address class imbalance in the training data, SMOTETomek was applied:

```python
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_preprocessed, y_train)
```
This technique combines SMOTE (Synthetic Minority Over-sampling Technique) and Tomek links to oversample minority classes and clean noisy data.

### Model Training
We used three machine learning models:

  - 🌲 Random Forest

  - 📈 XGBoost

  - ⚡ LightGBM

Each model was trained on the balanced dataset:
```python
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
lgb = LGBMClassifier(random_state=42)

rf.fit(X_train_resampled, y_train_resampled)
xgb.fit(X_train_resampled, y_train_resampled)
lgb.fit(X_train_resampled, y_train_resampled)
```

### Custom Prediction Function Using Ensemble Modeling
A custom prediction function was implemented to select the best-performing model for each class:
```python
def class_specific_predict(X, models, best_models_per_class):
    class_probabilities = np.zeros((X.shape[0], len(best_models_per_class)))
    for class_label in best_models_per_class.index:
        model_name = best_models_per_class[class_label]
        class_probabilities[:, int(class_label)] = models[model_name].predict_proba(X)[:, int(class_label)]
    return np.argmax(class_probabilities, axis=1)
```
The `best_models_per_class` dictionary specifies the optimal model for each class based on prior evaluation:
```python
best_models_per_class = pd.Series({
    0: 'Random Forest',
    1: 'Random Forest',
    2: 'Random Forest',
    3: 'LightGBM'
})
```
### Making Predictions and Submission
The custom prediction function was used to generate predictions for the test set:
```python
y_test_pred = class_specific_predict(X_test_preprocessed, models, best_models_per_class)
```
Finally, predictions were saved in the required submission format:
```python
submission_df = pd.DataFrame({
    'id': test['id'],  # Correct IDs from the test data
    'sii': y_test_pred
})
submission_df['sii'] = submission_df['sii'].astype(int)
submission_df.to_csv('submission.csv', index=False)
print("Saved to submission.csv")
```
This pipeline demonstrates the application of advanced ensemble modeling techniques, addressing class imbalance, and creating a robust prediction workflow for a multi-class classification task.

## 🏆 Model Evaluation and Selection
This section outlines the evaluation of machine learning models using Quadratic Weighted Kappa (QWK), a metric that measures agreement between predicted and actual class labels (the evaluation method used in this Kaggle competition) , and the subsequent selection of the best model for each class.

### Validation Split
The resampled training data was split into training and validation sets to ensure robust evaluation:
```python
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_resampled, y_train_resampled, test_size=0.2, random_state=42
)
```
### QWK Performance Evaluation
Each model was trained on the final training set, and predictions were compared against validation labels to compute QWK scores for each class:
```python
for model_name, model in models.items():
    model.fit(X_train_final, y_train_final)
    y_pred = model.predict(X_val)
    for class_label in np.unique(y_val):
        qwk = cohen_kappa_score(
            (y_val == class_label).astype(int),
            (y_pred == class_label).astype(int),
            weights="quadratic"
        )
        performance_per_class_qwk[model_name][class_label] = qwk
```
The QWK scores were organized into a DataFrame for better visualization:
```python
performance_df_qwk = pd.DataFrame(performance_per_class_qwk).T
performance_df_qwk = performance_df_qwk.T
```
### Visualizing Performance
To compare model performance across classes, a grouped column chart was created:
```python
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
x = np.arange(len(performance_df_qwk.index))  # Class positions

# Plot each model's performance
for i, (model_name, color) in enumerate(model_colors.items()):
    ax.bar(
        x + i * bar_width,
        performance_df_qwk[model_name],
        bar_width,
        label=model_name,
        color=color,
        alpha=0.8
    )
```
The chart provides a clear visual representation of how each model performs for different classes.
<p align="center">
  <img src="https://github.com/user-attachments/assets/f8af306b-2d1c-452f-8315-71efed7249a3">
</p>

### Selecting the Best Models
The best-performing model for each class was identified based on QWK scores:
```python
best_models_per_class_qwk = performance_df_qwk.idxmax(axis=1)
```
The results were printed for reference:
```python
print("Best model for each class based on QWK:")
for class_label, model_name in best_models_per_class_qwk.items():
    print(f"Class {class_label}: {model_name}")
```

Console output:

```
Best model for each class based on QWK:
Class 0.0: Random Forest
Class 1.0: Random Forest
Class 2.0: Random Forest
Class 3.0: LightGBM
```

This process ensured that the most suitable model was selected for each class, optimizing predictions for the multi-class classification task. The combination of robust evaluation metrics and intuitive visualization demonstrates a thorough and professional approach to model selection.


# 📊🔍*Data Overview*
## Kaggle Data Source

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

## **Data visualization and analysis using matplotlib**

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
  <img src="https://github.com/user-attachments/assets/ca32c59e-1689-44ea-8b8d-8a57c5aecbdb">
</p>

**The recorded values for BIA BMI and Physical BMI appear to be slightly different. There are two possible reasons for this:**

**1. There may be some discrepancy or error in the measurement of BMI.**

**2. The two BMIs could have been calculated at different ages for the same individual.**

# 1️⃣⚒Our First Attempt 

Our first target was to make it work, just to have some reference for the future. 

## Handling missing features with sklearn's Simple Imputer

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
## Handling missing classes in target column with Random Forest Regressor

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


# 🧪🔬Further Testing using seaborn

## Feature Importance Analysis
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
### First Method Using ThreadPoolExecutor
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
*This version uses code from the [First Attempt](https://github.com/zwigi9/Child_Mind_Institute-Multiclass-classification-AI-Model/tree/main#1%EF%B8%8F%E2%83%A3our-first-attempt) + First Method (this one) of parquet handling.*

<table style="border-collapse: collapse; border: 1px solid black; border-radius: 15px; background-color: #cce7ff; padding: 5px;">
  <tr>
    <td style="text-align: center; font-family: Arial, sans-serif; font-size: 50;">
      <img src="https://github.com/user-attachments/assets/7e141746-d572-4e52-96cf-31a1f01f3a97" alt="Icon" style="width: 20px; vertical-align: down; margin-right: 10px;">
      <b>Current Kaggle Score: 0.162</b>
    </td>
  </tr>
</table>


### Second Method Using scipy.stats

This method focuses on extracting descriptive statistics, trends, and higher-order moments like skewness and kurtosis, which provide deeper insights into the data's distribution.

We started by defining a set of numeric columns, such as `X`, `Y`, `Z`, `enmo`, `anglez`, and `light`, from which we wanted to extract features. These columns were grouped by the time_of_day field, enabling us to calculate detailed metrics for each time segment. For each group, we computed summary statistics such as the mean, standard deviation, minimum, maximum, skewness, and kurtosis, ensuring a rich feature set for our models.

To enhance computational efficiency, we processed all desired statistics in a single group-by operation. Additional features like the trend of the light column were calculated using linear regression when applicable, capturing patterns over time. To ensure that the extracted features were compact and meaningful, we filtered the results to include only metrics with specific suffixes, such as `skew`, `kurtosis`, `mean_over_times`, `std_over_times`, `min_over_times`, `max_over_times`, `trend`.

Our implementation also tackled the challenge of processing multiple Parquet files within Kaggle's runtime constraints. Using a combination of `os.walk` and `tqdm`, we iterated through all files in a directory, applying our feature extraction pipeline to each one. The extracted features for each patient were aggregated into a DataFrame, saved incrementally to prevent memory issues, and merged with the primary datasets (`train.csv`, `test.csv`) for further analysis.

The result was a compact, yet informative, dataset ready for machine learning models. By including advanced metrics like skewness and kurtosis, this method provided a deeper understanding of the data, contributing to better model performance.

```python
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# Define the numeric columns used for feature extraction
numeric_columns = ["X", "Y", "Z", "enmo", "anglez", "light"]

# Desired suffixes for feature selection
desired_suffixes = [
    "skew", "kurtosis", "mean_over_times", "std_over_times", 
    "min_over_times", "max_over_times", "trend"
]

def compute_patient_features(patient_data, patient_id):
    patient_features = {"id": patient_id}

    # Groupby operation performed only once
    daily_data = patient_data.groupby("time_of_day")[numeric_columns].agg(['mean', 'std', 'min', 'max'])
    daily_data.columns = ['_'.join(col).strip() for col in daily_data.columns.values]

    # Efficient computation of skew and kurtosis
    for col in numeric_columns:
        patient_features[f"{col}_skew"] = skew(patient_data[col], nan_policy="omit")
        patient_features[f"{col}_kurtosis"] = kurtosis(patient_data[col], nan_policy="omit")

    # Aggregating features over times without repeating groupby
    for col in daily_data.columns:
        patient_features[f"{col}_mean_over_times"] = daily_data[col].mean()
        patient_features[f"{col}_std_over_times"] = daily_data[col].std()
        patient_features[f"{col}_min_over_times"] = daily_data[col].min()
        patient_features[f"{col}_max_over_times"] = daily_data[col].max()

    # Only compute light trend if light column exists
    if "light" in patient_data.columns:
        light_means_per_time = patient_data.groupby("time_of_day")["light"].mean()
        patient_features["light_trend"] = np.polyfit(range(len(light_means_per_time)), light_means_per_time, 1)[0] if len(light_means_per_time) > 1 else 0

    # Filter to only desired features
    patient_features = {key: value for key, value in patient_features.items() if key == "id" or any(key.endswith(suffix) for suffix in desired_suffixes)}

    return patient_features

def process_and_save_file(file_path, output_file):
    # Read the data
    data = pd.read_parquet(file_path)
    patient_id = file_path.split('/')[-2].split('=')[-1]
    data['id'] = patient_id
    data = data[data["non-wear_flag"] == 0]
    
    # Simplify the time_of_day calculation
    data['time_of_day'] = ((data['time_of_day'] // 60000000000) % 1440).astype(int)  # Get minutes of the day

    # Process patients
    patient_ids = data["id"].unique()
    patient_aggregations = []
    
    with tqdm(total=len(patient_ids), desc=f"Processing {os.path.basename(file_path)}", unit="patient") as pbar:
        for patient_id in patient_ids:
            patient_data = data[data["id"] == patient_id]
            patient_features = compute_patient_features(patient_data, patient_id)
            patient_aggregations.append(patient_features)
            pbar.update(1)
    
    aggregated_features = pd.DataFrame(patient_aggregations)
    aggregated_features.fillna(0, inplace=True)
    
    # Append results to the CSV file
    aggregated_features.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

def load_and_process_directory(parquet_dir, output_file):
    # Iterate over files in the directory, processing each one and appending the results to the CSV file
    with tqdm(total=len(os.listdir(parquet_dir)), desc="Reading Parquet Files", unit="file") as pbar:
        for root, dirs, files in os.walk(parquet_dir):
            for file in files:
                if file.endswith('.parquet'):
                    file_path = os.path.join(root, file)
                    process_and_save_file(file_path, output_file)
                    pbar.update(1)

# Paths
train_parquet_dir = "/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet"
test_parquet_dir = "/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet"
output_file = "/kaggle/working/results.csv"  # Path to save the results

# Process train and test files
print("Processing train files...")
load_and_process_directory(train_parquet_dir, output_file)

print("Processing test files...")
load_and_process_directory(test_parquet_dir, output_file)

print("Processing complete!")
```

*This version uses code from the [First Attempt](https://github.com/zwigi9/Child_Mind_Institute-Multiclass-classification-AI-Model/tree/main#1%EF%B8%8F%E2%83%A3our-first-attempt) + Second Method (this one) of parquet handling.*

<table style="border-collapse: collapse; border: 1px solid black; border-radius: 15px; background-color: #cce7ff; padding: 5px;">
  <tr>
    <td style="text-align: center; font-family: Arial, sans-serif; font-size: 50;">
      <img src="https://github.com/user-attachments/assets/7e141746-d572-4e52-96cf-31a1f01f3a97" alt="Icon" style="width: 20px; vertical-align: down; margin-right: 10px;">
      <b>Current Kaggle Score: 0.180</b>
    </td>
  </tr>
</table>

This way of handling parque files gave us better result, but when used with our [best attempt](https://github.com/zwigi9/Child_Mind_Institute-Multiclass-classification-AI-Model/tree/main#our-best-and-final-attempt) code  it scored significantly lower, so we decided not to use them.



