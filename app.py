import streamlit as st
import joblib
import numpy as np

# Load the model (ensure this file is in the same directory)
model = joblib.load('voting_classifier_model.pkl')

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Predictor")
st.markdown("Use this app to predict **customer churn** based on customer details using a Voting Classifier model.")

st.header("👤 Enter Customer Details")

# --- UI FOR INPUT COLLECTION ---
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])
with col2:
    senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])

col3, col4 = st.columns(2)
with col3:
    partner = st.selectbox("Partner", ['No', 'Yes'])
with col4:
    dependents = st.selectbox("Dependents", ['No', 'Yes'])

# Account Information
st.subheader("Account Information")
tenure = st.slider("Tenure (months)", min_value=1, max_value=72, value=12)
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox("Paperless Billing", ['No', 'Yes'])

col5, col6 = st.columns(2)
with col5:
    # Assuming the numerical inputs were scaled/standardized during training, 
    # but input is taken as raw values for simplicity here.
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=50.0)
with col6:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

# Services
st.header("📱 Services & Features")

col7, col8 = st.columns(2)
with col7:
    phone_service = st.selectbox("Phone Service", ['No', 'Yes'])
with col8:
    multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])

internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

st.subheader("Internet Services")
col9, col10 = st.columns(2)
with col9:
    online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
with col10:
    tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])

# Payment Method
payment_method = st.selectbox("Payment Method", [
    'Electronic check', 
    'Mailed check', 
    'Bank transfer (automatic)', 
    'Credit card (automatic)'
])

def encode_inputs(gender, senior_citizen, partner, dependents, phone_service, multiple_lines, 
                  internet_service, online_security, online_backup, device_protection, 
                  tech_support, streaming_tv, streaming_movies, contract, paperless_billing, 
                  payment_method):
    
    # 1. Binary Features (0 or 1)
    encoded = []
    
    # Assuming 'Female' = 1, 'Male' = 0
    encoded.append(1 if gender == 'Female' else 0)
    
    # 'Yes' = 1, 'No' = 0 for other binary columns
    encoded.append(1 if senior_citizen == 'Yes' else 0)
    encoded.append(1 if partner == 'Yes' else 0)
    encoded.append(1 if dependents == 'Yes' else 0)
    encoded.append(1 if phone_service == 'Yes' else 0)
    encoded.append(1 if paperless_billing == 'Yes' else 0)
    
    # 2. Multi-Category Features (Label Encoding: 0, 1, 2, 3...)
    
    # MultipleLines (3 categories: [1 0 2] in your data)
    # We map the UI string to the expected integer value
    if multiple_lines == 'No':
        encoded.append(1) # Assuming 'No' maps to 1
    elif multiple_lines == 'Yes':
        encoded.append(0) # Assuming 'Yes' maps to 0
    else: # 'No phone service'
        encoded.append(2) # Assuming 'No phone service' maps to 2

    # InternetService (3 categories: [0 1 2] in your data)
    if internet_service == 'DSL':
        encoded.append(0) # Assuming 'DSL' maps to 0
    elif internet_service == 'Fiber optic':
        encoded.append(1) # Assuming 'Fiber optic' maps to 1
    else: # 'No'
        encoded.append(2) # Assuming 'No' maps to 2

    # Internet-related services (6 features, 3 categories each: [0 2 1] in your data)
    # We use a consistent mapping for No/Yes/No Internet Service
    service_mapping = {
        'No': 0, # Assuming 'No' maps to 0
        'Yes': 2, # Assuming 'Yes' maps to 2
        'No internet service': 1 # Assuming 'No internet service' maps to 1
    }

    services = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    for service in services:
        encoded.append(service_mapping[service])

    # Contract (3 categories: [0 1 2] in your data)
    if contract == 'Month-to-month':
        encoded.append(0) # Assuming 'Month-to-month' maps to 0
    elif contract == 'One year':
        encoded.append(1) # Assuming 'One year' maps to 1
    else: # 'Two year'
        encoded.append(2) # Assuming 'Two year' maps to 2
    
    # PaymentMethod (4 categories: [2 3 0 1] in your data)
    payment_mapping = {
        'Electronic check': 2,
        'Mailed check': 3,
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 1
    }
    encoded.append(payment_mapping[payment_method])
    
    # The final encoded list contains 3 (numerical) + 16 (categorical) = 19 features. 
    # This assumes your model input vector is [3 numerical features] + [16 single-integer categorical features].
    return encoded

if st.button("🔍 Predict Customer Churn"):
    # 1. Numerical features (Tenure, MonthlyCharges, TotalCharges)
    numerical_features = [
        tenure,
        monthly_charges,
        total_charges
    ]
    
    # 2. Categorical features (Encoded to single integers)
    categorical_features = encode_inputs(
        gender, senior_citizen, partner, dependents, phone_service, multiple_lines,
        internet_service, online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies, contract, paperless_billing,
        payment_method
    )
    
    # Combine all features into a single input vector
    input_vector = np.array([numerical_features + categorical_features])
    
    # Make prediction
    try:
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)[0]
        
        # Display results
        if prediction == 1:
            st.error(f"🚨 High Churn Risk: **{prediction_proba[1]*100:.1f}%** probability of churn")
            st.warning("This customer is likely to churn. Consider retention strategies.")
        else:
            st.success(f"✅ Low Churn Risk: **{prediction_proba[0]*100:.1f}%** probability of staying")
            st.info("This customer is likely to remain with the service.")
            
        # Show probability breakdown
        st.subheader("Probability Breakdown")
        col_left, col_right = st.columns(2)
        with col_left:
            st.metric("Probability of Staying", f"{prediction_proba[0]*100:.1f}%")
        with col_right:
            st.metric("Probability of Churning", f"{prediction_proba[1]*100:.1f}%")

    except ValueError as e:
        st.error("Error: The input vector size is incorrect.")
        st.error(f"Expected input size mismatch. The array should have been shaped like the input data to your model. Current input size: {input_vector.shape}")
        st.caption("Double-check the mappings in the `encode_inputs` function.")
