# import streamlit as st
# import pandas as pd
# import pickle
# from io import BytesIO
# from sklearn.preprocessing import MinMaxScaler

# # Load your pre-trained model
# with open('best_model_GradientBoostingRegressor.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# # Set up the Streamlit app
# st.set_page_config(page_title='Loan Default Prediction', layout='centered', initial_sidebar_state='expanded')

# st.title('Loan Default Prediction')

# # Sidebar for user input method (Manual or CSV)
# input_method = st.sidebar.radio("Select Input Method", ("Manual Input", "Upload CSV/Excel"))

# scaler = MinMaxScaler()
# input_df = None

# # Define required columns for input
# required_columns = ['ID', 'Age', 'Income', 'Home', 'Emp_Length', 'Intent', 'Amount', 'Rate', 'Status', 
#                     'Percent_Income', 'Default', 'Cred_Length']

# # Define the mapping dictionary
# mapping_dict = {
#     'Home': {
#         'OWN': 0,
#         'RENT': 1,
#         'MORTGAGE': 2,
#         'OTHER': 3
#     },
#     'Intent': {
#         'PERSONAL': 0,
#         'EDUCATION': 1,
#         'MEDICAL': 2,
#         'VENTURE': 3,
#         'HOMEIMPROVEMENT': 4,
#         'DEBTCONSOLIDATION': 5
#     }
# }

# # Function to map categorical features
# def map_categorical_features(data, mappings):
#     for column, mapping in mappings.items():
#         if column in data.columns:
#             try:
#                 data[column] = data[column].replace(mapping)
#             except Exception as e:
#                 st.error(f"An error occurred when mapping column '{column}': {e}")
#         else:
#             st.warning(f"The column '{column}' is not found in the DataFrame")
#     return data

# if input_method == "Upload CSV/Excel":
#     # File upload form for CSV/Excel files
#     uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
#     if uploaded_file:
#         if uploaded_file.name.endswith('.csv'):
#             input_df = pd.read_csv(uploaded_file)
#         elif uploaded_file.name.endswith('.xlsx'):
#             input_df = pd.read_excel(uploaded_file)
        
#         # Ensure all required columns are present
#         if not all(col in input_df.columns for col in required_columns):
#             st.error("Uploaded file does not contain all required columns.")
#             input_df = None
#         else:
#             st.write(input_df.head())  # Display the first few rows of the file

# else:
#     # Manual input form for loan applicant data
#     with st.sidebar.form(key='manual_input_form'):
#         st.header('Enter Applicant Data')
#         age = st.number_input('Age', min_value=18, max_value=100, step=1)
#         income = st.number_input('Income', min_value=0, step=1000)
#         home = st.selectbox('Home Ownership', ['Own', 'Mortgage', 'Rent'])
#         emp_length = st.number_input('Employment Length (years)', min_value=0, max_value=40, step=1)
#         intent = st.selectbox('Loan Intent', ['Education', 'Home Improvement', 'Debt Consolidation'])
#         amount = st.number_input('Loan Amount', min_value=500, step=500)
#         rate = st.number_input('Interest Rate (%)', min_value=0.0, step=0.01)
#         status = st.selectbox('Loan Status', ['Fully Paid', 'Charged Off', 'Current'])
#         percent_income = st.number_input('Percent of Income Loan Amount', min_value=0.0, step=0.1)
#         cred_length = st.number_input('Credit History Length (years)', min_value=0, max_value=50, step=1)
        
#         submit_button = st.form_submit_button(label='Predict')

#     # Create a dictionary of input data
#     if submit_button:
#         input_data = {
#             'Age': age,
#             'Income': income,
#             'Home': home,
#             'Emp_Length': emp_length,
#             'Intent': intent,
#             'Amount': amount,
#             'Rate': rate,
#             'Status': status,
#             'Percent_Income': percent_income,
#             'Cred_Length': cred_length,
#         }
#         input_df = pd.DataFrame([input_data])

# # Preprocessing and prediction
# def preprocess_input(input_df):
#     # Ensuring the DataFrame has the required columns
#     input_df = input_df.reindex(columns=required_columns, fill_value=0)
    
#     # Scaling the data
#     input_df_scaled = pd.DataFrame(scaler.fit_transform(input_df), columns=required_columns)
#     return input_df_scaled

# if input_df is not None:
#     # Preprocess and predict
#     processed_data = preprocess_input(input_df)
#     predictions = model.predict(processed_data)
    
#     st.write("Prediction results:")
#     input_df['Prediction'] = predictions
#     st.write(input_df)

#     # Allow download of results as Excel
#     def to_excel(df):
#         output = BytesIO()
#         writer = pd.ExcelWriter(output, engine='xlsxwriter')
#         df.to_excel(writer, index=False, sheet_name='Predictions')
#         writer.close()
#         return output.getvalue()

#     df_xlsx = to_excel(input_df)
#     st.download_button(label='📥 Download Predictions as Excel', data=df_xlsx, file_name='loan_default_predictions.xlsx')


# ===========================================================================================

import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler

# Load your pre-trained model
with open('best_model_GradientBoostingRegressor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Set up the Streamlit app
st.set_page_config(page_title='Loan Default Prediction', layout='centered', initial_sidebar_state='expanded')

st.title('Loan Default Prediction')

# Sidebar for user input method (Manual or CSV)
input_method = st.sidebar.radio("Select Input Method", ("Manual Input", "Upload CSV/Excel"))

scaler = MinMaxScaler()
input_df = None

# Define required columns for input
required_columns = ['ID', 'Age', 'Income', 'Home', 'Emp_Length', 'Intent', 'Amount', 'Rate', 'Status', 
                    'Percent_Income', 'Default', 'Cred_Length']

# Define the mapping dictionary
mapping_dict = {
    'Home': {
        'Own': 0,
        'Rent': 1,
        'Mortgage': 2
    },
    'Intent': {
        'Personal': 0,
        'Education': 1,
        'Medical': 2,
        'Venture': 3,
        'Home Improvement': 4,
        'Debt Consolidation': 5
    }
}

# Function to map categorical features
def map_categorical_features(data, mappings):
    for column, mapping in mappings.items():
        if column in data.columns:
            try:
                data[column] = data[column].replace(mapping)
            except Exception as e:
                st.error(f"An error occurred when mapping column '{column}': {e}")
        else:
            st.warning(f"The column '{column}' is not found in the DataFrame")
    return data

if input_method == "Upload CSV/Excel":
    # File upload form for CSV/Excel files
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            input_df = pd.read_excel(uploaded_file)
        
        # Ensure all required columns are present
        if not all(col in input_df.columns for col in required_columns):
            st.error("Uploaded file does not contain all required columns.")
            input_df = None
        else:
            st.write(input_df.head())  # Display the first few rows of the file

else:
    # Manual input form for loan applicant data
    with st.sidebar.form(key='manual_input_form'):
        st.header('Enter Applicant Data')
        age = st.number_input('Age', min_value=18, max_value=100, step=1)
        income = st.number_input('Income', min_value=0, step=1000)
        home = st.selectbox('Home Ownership', ['Own', 'Mortgage', 'Rent'])
        emp_length = st.number_input('Employment Length (years)', min_value=0, max_value=40, step=1)
        intent = st.selectbox('Loan Intent', ['Personal','Education', 'Medical', 'Venture', 'Home Improvement', 'Debt Consolidation'])
        amount = st.number_input('Loan Amount', min_value=500, step=500)
        rate = st.number_input('Interest Rate (%)', min_value=0.0, step=0.01)
        status = st.selectbox('Loan Status', ['Fully Paid', 'Charged Off', 'Current'])
        percent_income = st.number_input('Percent of Income Loan Amount', min_value=0.0, step=0.1)
        cred_length = st.number_input('Credit History Length (years)', min_value=0, max_value=50, step=1)
        
        submit_button = st.form_submit_button(label='Predict')

    # Create a dictionary of input data
    if submit_button:
        input_data = {
            'Age': age,
            'Income': income,
            'Home': home,
            'Emp_Length': emp_length,
            'Intent': intent,
            'Amount': amount,
            'Rate': rate,
            'Status': status,
            'Percent_Income': percent_income,
            'Cred_Length': cred_length,
        }
        input_df = pd.DataFrame([input_data])

# Preprocessing and prediction
def preprocess_input(input_df):
    # Ensure DataFrame has the required columns
    input_df = input_df.reindex(columns=required_columns, fill_value=0)

    # Apply mapping to categorical features
    input_df = map_categorical_features(input_df, mapping_dict)
    
    # Scaling the data
    input_df_scaled = pd.DataFrame(scaler.fit_transform(input_df), columns=required_columns)
    return input_df_scaled

if input_df is not None:
    # Preprocess and predict
    processed_data = preprocess_input(input_df)
    predictions = model.predict(processed_data)
    
    st.write("Prediction results:")
    input_df['Prediction'] = predictions
    st.write(input_df)

    # Allow download of results as Excel
    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Predictions')
        writer.close()
        return output.getvalue()

    df_xlsx = to_excel(input_df)
    st.download_button(label='📥 Download Predictions as Excel', data=df_xlsx, file_name='loan_default_predictions.xlsx')
