import streamlit as st
import pickle
import os
from sms_preprocessing.preprocess import text_processing


# Define the path to the saved model and vectorizer files
# Path to the model file
model_path = os.path.join('saved_models', 'model.pkl')
# Path to the vectorizer file
vectorizer_path = os.path.join('saved_models', 'vectorizer.pkl')

# Apply custom CSS for the Streamlit app to improve the UI appearance
st.markdown(
    """
    <style>
        /* Center the app content vertically */
        .main {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        /* Customize the header title */
        h1 {
            color: 	#ffffff;
            font-size: 2.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    # Load the pre-trained model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)  # Load the classifier model

    # Load the vectorizer used to transform input data
    with open(vectorizer_path, 'rb') as vectorizer_file:
        # Load the vectorizer (e.g., TF-IDF)
        tfv = pickle.load(vectorizer_file)

    # Set the title of the app
    st.title("SMS SPAM CLASSIFIER")

    # Input box to collect SMS text from the user
    sms = st.text_input('Enter the message to check if it is spam or ham...')

    # Button to trigger the classification
    if st.button('Check', type='primary'):
      

        # Preprocess the SMS message using the custom text processing function
        preprocessed_msg = text_processing(sms)

        # Vectorize the preprocessed message using the TF-IDF vectorizer
        vectorized_msg = tfv.transform([preprocessed_msg])

        # Use the model to predict the class (Spam or Ham)
        # Get the predicted label (0 for Ham, 1 for Spam)
        predict = model.predict(vectorized_msg)[0]

        # Display the prediction result
        if predict == 1:
            st.header("Spam SMS")  # If the prediction is 1, it's spam
        else:
            st.header("Ham SMS")  # If the prediction is 0, it's ham

except Exception as e:
    # Handle exceptions and print error messages
    st.error(f"An error occurred: {e}")
