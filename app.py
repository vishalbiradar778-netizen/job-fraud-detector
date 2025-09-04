import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Page config for title, icon, and layout
st.set_page_config(
    page_title="Job Fraud Detector",
    page_icon="🕵️‍♂️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Function to load and prepare data
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('fake_job_postings.csv')
    df = df.fillna('')
    df['text'] = df['title'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits']
    return df

# Function to train model
@st.cache_resource(show_spinner=False)
def train_model(df):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['text'])
    y = df['fraudulent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, tfidf, report

# Main app
def main():
    st.title("🕵️‍♂️ Job Fraud Detector")
    st.markdown("""
    Detect if a job posting is **fraudulent or real** based on the job description and details.
    """)

    # Sidebar info
    with st.sidebar:
        st.header("About this App")
        st.write("""
        - Built with Python, Streamlit, and scikit-learn.
        - Detects fake job postings using Random Forest and TF-IDF.
        - Upload dataset and train model included.
        """)
        st.write("Made by Your Name")

    with st.spinner("Loading and training model..."):
        df = load_data()
        model, tfidf, report = train_model(df)
    st.success("Model trained successfully!")

    # Display evaluation metrics
    with st.expander("See model evaluation metrics"):
        st.write(f"**Accuracy:** {report['accuracy']:.4f}")
        st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Test a Job Posting")

    user_input = st.text_area("Paste job title, description, requirements, benefits here:")

    if st.button("Predict Fraud"):
        if not user_input.strip():
            st.warning("Please enter job posting text to analyze.")
        else:
            vect_input = tfidf.transform([user_input])
            prediction = model.predict(vect_input)[0]
            if prediction == 0:
                st.success("✅ This job posting looks REAL.")
            else:
                st.error("🚨 This job posting looks FAKE.")

    # Footer
    st.markdown("---")
    st.markdown("© 2025 Your Name | [GitHub](https://github.com/yourusername)")

if __name__ == "__main__":
    main()
