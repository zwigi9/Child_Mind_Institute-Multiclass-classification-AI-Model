# Child_Mind_Institute-Multiclassification
nigas
in paris
'''python

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer', categorical_imputer),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # powinno dać się wyjebać tego preprocessora ale błąd wypierdala pandas???
        ]), categorical_cols)
    ])

'''
