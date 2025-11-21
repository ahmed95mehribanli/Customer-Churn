import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('voting_classifier_model.pkl')

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Predictor")
st.markdown("Use this app to predict **customer churn** based on customer details using a Voting Classifier model.")

st.header("👤 Enter Customer Details")

# Personal Information
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
    
    # Encode binary features
    encoded = []
    encoded.append(1 if gender == 'Female' else 0)  # gender
    encoded.append(1 if senior_citizen == 'Yes' else 0)  # SeniorCitizen
    encoded.append(1 if partner == 'Yes' else 0)  # Partner
    encoded.append(1 if dependents == 'Yes' else 0)  # Dependents
    encoded.append(1 if phone_service == 'Yes' else 0)  # PhoneService
    
    # Encode MultipleLines (3 categories)
    if multiple_lines == 'No':
        encoded.extend([1, 0, 0])
    elif multiple_lines == 'Yes':
        encoded.extend([0, 1, 0])
    else:  # No phone service
        encoded.extend([0, 0, 1])
    
    # Encode InternetService (3 categories)
    if internet_service == 'DSL':
        encoded.extend([1, 0, 0])
    elif internet_service == 'Fiber optic':
        encoded.extend([0, 1, 0])
    else:  # No
        encoded.extend([0, 0, 1])
    
    # Encode internet-related services (3 categories each)
    services = [online_security, online_backup, device_protection, 
                tech_support, streaming_tv, streaming_movies]
    
    for service in services:
        if service == 'No':
            encoded.extend([1, 0, 0])
        elif service == 'Yes':
            encoded.extend([0, 1, 0])
        else:  # No internet service
            encoded.extend([0, 0, 1])
    
    # Encode Contract (3 categories)
    if contract == 'Month-to-month':
        encoded.extend([1, 0, 0])
    elif contract == 'One year':
        encoded.extend([0, 1, 0])
    else:  # Two year
        encoded.extend([0, 0, 1])
    
    # Encode PaperlessBilling
    encoded.append(1 if paperless_billing == 'Yes' else 0)
    
    # Encode PaymentMethod (4 categories)
    if payment_method == 'Electronic check':
        encoded.extend([1, 0, 0, 0])
    elif payment_method == 'Mailed check':
        encoded.extend([0, 1, 0, 0])
    elif payment_method == 'Bank transfer (automatic)':
        encoded.extend([0, 0, 1, 0])
    else:  # Credit card (automatic)
        encoded.extend([0, 0, 0, 1])
    
    return encoded

if st.button("🔍 Predict Customer Churn"):
    # Numerical features
    numerical_features = [
        tenure,
        monthly_charges,
        total_charges
    ]
    
    # Categorical features
    categorical_features = encode_inputs(
        gender, senior_citizen, partner, dependents, phone_service, multiple_lines,
        internet_service, online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies, contract, paperless_billing,
        payment_method
    )
    
    # Combine all features
    input_vector = np.array([numerical_features + categorical_features])
    
    # Make prediction
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
