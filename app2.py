import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Loading the model
with open('best_model_GradientBoostingRegressor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Setting the Streamlit app
st.set_page_config(page_title='Loan Defualt Prdictor', page_icon='bitcoin.svg',
                   layout='centered', initial_sidebar_state='expanded')

# Custom CSS to add a background image
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: black;
#         background-size: cover;
#         background-position: center;
#     }
#     .sidebar .sidebar-content {
#         background: rgba(255, 255, 255, 0.8);
#     }
#     .st-bm {
#         background: rgba(255, 255, 255, 0.8);
#         border-radius: 10px;
#         padding: 15px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.title('Loan Things')

# Load and display the image using PIL
# image = Image.open('crypto_img.jpg')
# st.image(image, use_column_width=True)

# User choice for input method
input_method = st.sidebar.radio("Select Input Method", ("Manual Input", "Upload CSV/Excel"))

input_df = None
scaler = StandardScaler()  # Instantiate scaler once

# Required columns list
required_columns = ['Age', 'Income', 'Home', 'Emp_Length', 'Intent', 'Amount', 'Rate', 'Status', 
                    'Percent_Income','Cred_Length']

if input_method == "Upload CSV/Excel":
    # File upload form
    with st.sidebar.form(key='file_upload_form'):
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        submit_file_button = st.form_submit_button(label='Predict (Upload File)')

    if uploaded_file is not None and submit_file_button:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            input_df = pd.read_excel(uploaded_file)
        
        # Ensure the required columns are present
        if not all(col in input_df.columns for col in required_columns):
            st.error("Uploaded file does not contain all required columns.")
            input_df = None

else:
    # Manual input fields using a form
    with st.sidebar.form(key='manual_input_form'):
        st.header('Enter Client details below')

        Age = st.number_input("Age", min_value=18, max_value=100, value=30)
        Income = st.number_input("Income", min_value=0, value=50000)
        Home = st.selectbox("Home Status", options=['OWN', 'RENT', 'MORTGAGE', 'OTHER'])
        Emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
        Intent = st.selectbox("Loan Intent", options=['PERSONAL', 'EDUCATION', 'MEDICAL', 
                                                          'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
        Amount = st.number_input("Loan Amount", min_value=0, value=10000)
        Rate = st.number_input("Interest Rate", min_value=0.0, max_value=100.0, value=5.5)
        Status = st.number_input("Credit Score Status", min_value=300, max_value=850, value=700)
        Percent_income = st.number_input("Percentage of Income", min_value=0.0, max_value=100.0, value=20.0)
        Cred_length = st.number_input("Credit History Length (months)", min_value=0, value=24)
        
        submit_manual_button = st.form_submit_button(label='Predict (Manual Input)')

def preprocess_input(input_data, is_manual=True):
    # Ensure required columns are in the input data
    input_df = pd.DataFrame([input_data]) if is_manual else input_data
    input_df = input_df.reindex(columns=required_columns, fill_value=0)

    # # Scaling the input data
    # input_df_scaled = pd.DataFrame(scaler.fit_transform(input_df), columns=required_columns)
    # return input_df_scaled

if input_method == "Upload CSV/Excel" and input_df is not None:
    # Preprocess and predict
    processed_data = preprocess_input(input_df, is_manual=False)
    predictions = model.predict(processed_data)
    input_df['Predicted Closing Price'] = predictions
    st.write("Predictions made for uploaded data:")
    st.write(input_df)

    # Download button for the prediction results
    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Predictions')
        writer.close()
        processed_data = output.getvalue()
        return processed_data

    df_xlsx = to_excel(input_df)
    st.download_button(label='📥 Download Predictions as Excel', data=df_xlsx, file_name='predictions.xlsx')

elif input_method == "Manual Input" and submit_manual_button:
    # Collect manual input values
    input_data = {

        'Age' = Age,
        'Income' = Income,
        'Home' = Home,
        'Emp_length' = Emp_length,
        'Intent' = Intent,
        'Amount' = Amount,
        'Rate' = Rate,
        'Status' = Status,
        'Percent_income' = Percent_income,
        'Cred_length' = Cred_length
    }

    # Preprocess the manual input data
    input_df_scaled = preprocess_input(input_data, is_manual=True)
    
    # Make predictions
    predictions = model.predict(input_df_scaled)
    st.write("Predictions:", predictions)
