# Child_Mind_Institute-Multiclassification
nigas
in paris
'''python
# Step 15: Identify common columns
common_cols = X_train.columns.intersection(X_test.columns)

# Step 16: Update your preprocessor to only transform available columns in the test set
# Handle numeric and categorical columns separately
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.intersection(common_cols)
categorical_cols = X_train.select_dtypes(include=['object']).columns.intersection(common_cols)

# For categorical features, apply one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer', categorical_imputer),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # powinno dać się wyjebać tego preprocessora ale błąd wypierdala pandas???
        ]), categorical_cols)
    ])

# Step 17: Preprocess both the training and test data
X_train_preprocessed = preprocessor.fit_transform(X_train[common_cols])
X_test_preprocessed = preprocessor.transform(X_test[common_cols])

# Step 18: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_preprocessed, y_train)

# Step 19: Predict using the trained model on the preprocessed test data
predictions = model.predict(X_test_preprocessed)

# Step 20: Prepare submission file
submission = pd.DataFrame({
    'id': test['id'],  # Assuming 'id' is the column in test.csv
    'sii': predictions.astype(int)  # Ensure predictions are integers (for classification)
})

# Step 21: Save to submission.csv
submission.to_csv('submission.csv', index=False)
print('Saved to submission.csv"')
'''
