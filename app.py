import pandas as pd
from datetime import datetime
import joblib
import numpy as np
from pycaret.classification import load_model, predict_model
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load the model
loaded_model = load_model('cybersquatting_rf_model')

# Define a function to process the uploaded data
def process_data(data):
    data = data[data['owned_by_org'] == 'No'].drop(columns=['owned_by_org'])

    # Load the registrars and countries files from user uploads
    registrars_df = pd.read_csv(registrars_file)
    countries_df = pd.read_csv(countries_file)

    # Initialize 'Threat score' column with 0
    data['Threat score'] = 0

    # Update threat score based on country and registrar
    countries_list = countries_df['Countries'].tolist()
    for country in countries_list:
        data.loc[data['country'].str.contains(country, case=False, na=False, regex=False), 'Threat score'] += 4.0

    registrars_list = registrars_df['Registrar'].tolist()
    for registrar in registrars_list:
        data.loc[data['registrar'].str.contains(registrar, case=False, na=False, regex=False), 'Threat score'] += 4.0    

    # Domain Age Check
    data['creation_date'] = pd.to_datetime(data['creation_date'], errors='coerce')
    current_date = datetime.now()
    one_year_ago = current_date - pd.DateOffset(years=1)
    three_years_ago = current_date - pd.DateOffset(years=3)

    data.loc[data['creation_date'] >= one_year_ago, 'Threat score'] += 8.0  # High-Risk (Less than 1 year old)
    data.loc[(data['creation_date'] < one_year_ago) & (data['creation_date'] >= three_years_ago), 'Threat score'] += 4.0  # Medium-Risk

    # Expiration Date Check
    data['expiration_date'] = pd.to_datetime(data['expiration_date'], errors='coerce')
    one_month_later = current_date + pd.DateOffset(months=1)
    three_months_later = current_date + pd.DateOffset(months=3)
    six_months_later = current_date + pd.DateOffset(months=6)

    data.loc[(data['expiration_date'] >= one_month_later) & (data['expiration_date'] < three_months_later), 'Threat score'] += 6.0  # High-Risk (1-3 months)
    data.loc[(data['expiration_date'] >= three_months_later) & (data['expiration_date'] < six_months_later), 'Threat score'] += 3.0  # Medium-Risk

    # DNSSEC Check
    data.loc[data['dnssec'] != 'signed', 'Threat score'] += 4.0

    # Security Features Check
    data.loc[data['DMARC'] == 'Not Enabled', 'Threat score'] += 6.0  # DMARC
    data.loc[data['CAA'] == 'Not Enabled', 'Threat score'] += 3.0   # CAA
    data.loc[data['DKIM'] == 'Not Enabled', 'Threat score'] += 6.0   # DKIM
    data.loc[data['SPF'] == 'Not Enabled', 'Threat score'] += 3.0   # SPF

    # DNS Records Check
    data.loc[data['A'] == 'Not Enabled', 'Threat score'] += 3.0  # A record
    data.loc[data['AAAA'] == 'Not Enabled', 'Threat score'] += 5.0  # AAAA record
    data.loc[data['NS'] == 'Not Enabled', 'Threat score'] += 3.0  # NS record
    data.loc[data['SOA'] == 'Not Enabled', 'Threat score'] += 3.0  # SOA record

    # Normalize the Threat Score
    max_score = data['Threat score'].max()
    data['Threat score (Scaled)'] = round((data['Threat score'] / 58) * 10, 2)

    # Step 1: Categorize the Threat score (Scaled) based on CVSS-like categories
    def categorize_risk(score):
        if score >= 9.0:
            return 'Critical'
        elif 7.0 <= score < 9.0:
            return 'High'
        elif 4.0 <= score < 7.0:
            return 'Medium'
        else:
            return 'Low'

    data['Risk Category'] = data['Threat score (Scaled)'].apply(categorize_risk)

    # Define feature weightage
    feature_weightage = {
        # Define feature weightage here (same as in your original code)
        'DMARC': {'weight': round((6.0 / 58) * 10, 2), 'importance': 'DMARC provides email protection, preventing phishing and spoofing attacks.'},
        'DKIM': {'weight': round((6.0 / 58) * 10, 2), 'importance': 'DKIM verifies the sender’s domain, ensuring email integrity and authenticity.'},
        'SPF': {'weight': round((3.0 / 58) * 10, 2), 'importance': 'SPF helps prevent unauthorized use of your domain in email sending.'},
        'CAA': {'weight': round((3.0 / 58) * 10, 2), 'importance': 'CAA restricts certificate authorities from issuing certificates for your domain, reducing the risk of man-in-the-middle attacks.'},
        'dnssec': {'weight': round((4.0 / 58) * 10, 2), 'importance': 'DNSSEC protects against DNS spoofing, ensuring users reach the correct website.'},
        'A': {'weight': round((3.0 / 58) * 10, 2), 'importance': 'A record directs domain traffic to the correct IP address; its absence can lead to resolution failures.'},
        'AAAA': {'weight': round((5.0 / 58) * 10, 2), 'importance': 'AAAA record enables IPv6 support; its absence can limit accessibility.'},
        'NS': {'weight': round((3.0 / 58) * 10, 2), 'importance': 'NS records specify authoritative name servers, essential for proper domain resolution.'},
        'SOA': {'weight': round((3.0 / 58) * 10, 2), 'importance': 'SOA record defines the zone\'s authoritative details, crucial for DNS management.'},
        'creation_date': {'weight': round((4.0 / 58) * 10, 2), 'importance': 'Domains less than 1 year old are generally considered higher risk due to lack of history.'},
        'expiration_date': {'weight': round((4.0 / 58) * 10, 2), 'importance': 'Domains expiring soon can pose a risk, as they may be abandoned or sold off quickly.'},
        'country': {'weight': round((5.0 / 58) * 10, 2), 'importance': 'Domains registered in high-risk countries may be more susceptible to malicious activities.'},
        'registrar': {'weight': round((5.0 / 58) * 10, 2), 'importance': 'Domains registered with risky registrars can indicate potential security vulnerabilities.'}
    }

    # Update the generate_reason function
    def generate_reason(row):
        reasons = []
        # Check security features
        for feature, details in feature_weightage.items():
            if row[feature] == 'Not Enabled':
                reasons.append(f"{feature} not enabled (weight: {details['weight']}) - {details['importance']}")

        # Evaluate domain creation date
        if row['creation_date'] >= one_year_ago:
            reasons.append(f"Domain less than 1 year old (weight: {feature_weightage['creation_date']['weight']}) - {feature_weightage['creation_date']['importance']}")
        elif (one_year_ago <= row['creation_date'] < three_years_ago):
            reasons.append("Domain between 1-3 years old (moderate risk)")

        # Evaluate domain expiration date
        if row['expiration_date'] < three_months_later:
            reasons.append(f"Domain expiring soon (weight: {feature_weightage['expiration_date']['weight']}) - {feature_weightage['expiration_date']['importance']}")
        elif (three_months_later <= row['expiration_date'] < six_months_later):
            reasons.append("Domain expiring within 3-6 months (medium risk)")

        # Evaluate country risk
        if row['country'] in countries_list:
            reasons.append(f"Domain registered in a high-risk country (weight: {feature_weightage['country']['weight']}) - {feature_weightage['country']['importance']}")

        # Evaluate registrar risk
        if row['registrar'] in registrars_list:
            reasons.append(f"Registrar considered risky (weight: {feature_weightage['registrar']['weight']}) - {feature_weightage['registrar']['importance']}")

        return "; ".join(reasons)

    # Apply the generate_reason function
    data['Risk Reasons'] = data.apply(generate_reason, axis=1)

    # Sort the DataFrame by Threat Score (Scaled)
    data = data.sort_values(by='Threat score (Scaled)', ascending=False)

    # Step 2: Drop the 'domain' column (optional, if you don't want to use domain names)
    data_cleaned = data.drop(columns=['domain'])

    # Step 3: Encoding the categorical columns with Label Encoding
    label_encoder = LabelEncoder()

    # Encode 'registrar' and 'country' (Label Encoding)
    if 'registrar' in data_cleaned.columns:
        data_cleaned.loc[:, 'registrar'] = label_encoder.fit_transform(data_cleaned['registrar'])

    if 'country' in data_cleaned.columns:
        data_cleaned.loc[:, 'country'] = label_encoder.fit_transform(data_cleaned['country'])

    # Step 4: Convert date columns into numeric features
    # Calculate domain age in days (current date minus creation date)
    data_cleaned['domain_age_days'] = (datetime.now() - data_cleaned['creation_date']).dt.days

    # Calculate days until expiration
    data_cleaned['days_until_expiration'] = (data_cleaned['expiration_date'] - datetime.now()).dt.days

    # Drop the original date columns if you don't need them anymore
    data_cleaned = data_cleaned.drop(columns=['creation_date', 'expiration_date'])

    # Step 5: Encode binary columns (Enabled/Not Enabled, Signed/Unsigned)
    binary_columns = ['dnssec', 'DMARC', 'CAA', 'DKIM', 'SPF', 'A', 'AAAA', 'NS', 'MX', 'SOA']
    for column in binary_columns:
        if column in data_cleaned.columns:
            data_cleaned.loc[:, column] = data_cleaned[column].map({'Enabled': 1, 'Not Enabled': 0, 'Signed': 1, 'Unsigned': 0})

    # Predict using the loaded model
    predictions = predict_model(loaded_model, data=data_cleaned)

    # Add domain to predictions
    predictions['domain'] = data['domain'].values 

    # Filter predictions for prediction_label == 1 if that column exists
    if 'prediction_label' in predictions.columns:
        predictions = predictions[predictions['prediction_label'] == 1]

    # Add Cybersquatting Risk column
    predictions['Cybersquatting_Risk'] = predictions['prediction_label'].apply(
        lambda x: 'Susceptible to Cybersquatting' if x == 1 else 'No Risk of Cybersquatting'
    )

    return predictions

# Streamlit UI
st.title("Cybersquatting Risk Assessment Tool")

# File upload for new data
st.subheader("Upload Domain Data CSV")
uploaded_file = st.file_uploader("Choose a CSV file...", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # File upload for registrars data
    st.subheader("Upload Registrars Data CSV")
    registrars_file = st.file_uploader("Choose a CSV file for Registrars...", type="csv")
    if registrars_file:
        # File upload for countries data
        st.subheader("Upload Countries Data CSV")
        countries_file = st.file_uploader("Choose a CSV file for Countries...", type="csv")
        if countries_file:
            predictions = process_data(data)

            # Display predictions
            st.subheader("Predictions")
            st.write(predictions[['domain', 'Cybersquatting_Risk', 'Threat score (Scaled)', 'Risk Category', 'Risk Reasons']])

            # Save Results Button
            if st.button("Save Results"):
                predictions.to_csv('MEGA_RFC_Result_with_reasons.csv', index=False)
                st.success("Results saved successfully!")
