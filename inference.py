
import pandas as pd
import pickle
import os
from helper_functions import select_features


# Load the test data / new data to predict (here we are using the whole dataset as test data)
new_df = pd.read_csv('data/UCI_Credit_Card.csv')
X_test = new_df.drop(columns=['default.payment.next.month', 'ID'])
y_test = new_df['default.payment.next.month']


# THESE STEPS SHOULD BE DONE BEFORE LOADING THE MODELS AND IN THE DATA ENGINEERING PIPELINE
# THIS CODE IS NOT NECESARY WHEN THE DATA IS CLEANED AND PREPROCESSED PROPERLY
# Replace values in the EDUCATION column
X_test['EDUCATION'] = X_test['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
# Replace values in the MARRIAGE column
X_test['MARRIAGE'] = X_test['MARRIAGE'].replace({0: 3})
# Shift the values in the PAY_[digit] columns to match the data description
pay_columns = X_test[[col for col in X_test.columns if col.startswith('PAY_') and col[4:].isdigit()]].columns
X_test[pay_columns] = X_test[pay_columns] + 1


# Load the saved PCA model from the pickle file
with open(os.path.join('./models/pca_model.pickle'), 'rb') as handle:
    pca = pickle.load(handle)
# Load the saved Scaler model from the pickle file
with open(os.path.join('./models/scaler_model.pickle'), 'rb') as handle:
    scaler = pickle.load(handle)
# Load the saved XGBoost model from the pickle file
with open(os.path.join('./models/xgboost.pickle'), 'rb') as handle:
    model = pickle.load(handle)

# Feature selection and transformation
# Select only the 'BILL_AMT' columns
bill_df_test = X_test.filter(like='BILL_AMT')  
# scale data using the same scaler used for training data
scaled_data_test = scaler.transform(bill_df_test)
# Fit PCA on the new test data 
principal_components_test = pca.transform(scaled_data_test)
pca_df_test = pd.DataFrame(
    data=principal_components_test,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)]
)
# remove the BILL_AMT columns and add the PCA components
X_test = select_features(X_test, pca_df_test[['PC1', 'PC2']])
# remove highly correlated features and the ID column
X_test.drop(columns=['PAY_5'], inplace=True)
# X_test.info()

# Make predictions
predictions = model.predict(X_test)

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['predicted_label'])

# Print the DataFrame
print(predictions_df)

