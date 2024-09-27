import nltk
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

def text_processing(text):
    """
    Function to preprocess text by performing the following steps:
    1. Convert text to lowercase.
    2. Tokenize the text into individual words.
    3. Lemmatize the tokens to their base form.
    4. Remove stopwords (common words like 'is', 'the').
    5. Remove non-alphanumeric characters (like punctuation).

    Parameters:
    -----------
    text : str
        Input text to be preprocessed.

    Returns:
    --------
    str
        A cleaned and preprocessed version of the input text, with lemmatized tokens, no stopwords, and no special characters.
    """

    # Step 1: Convert the text to lowercase to ensure consistency
    text_lower = text.lower()

    # Step 2: Tokenize the text into individual words (list of words)
    text_list = word_tokenize(text_lower)

    # Step 3: Initialize the WordNetLemmatizer to reduce words to their base form
    lemmatizer = WordNetLemmatizer()

    # Step 4: Lemmatize each word if it's not a stopword
    # Only retain words that are not in the stopwords list
    text_list = [lemmatizer.lemmatize(word) for word in text_list if word not in stopwords.words('english')]

    # Step 5: Remove non-alphanumeric tokens (special characters, punctuation, etc.)
    text_list = [word for word in text_list if word.isalnum()]

    # Step 6: Join the list of words back into a single string
    # Return the cleaned, processed text
    return ' '.join(text_list)