import streamlit as st
import re
import nltk

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------
# Download NLTK Resources
# ---------------------------------------------------

try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet")

# ---------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Information Retrieval System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Information Retrieval System")
st.subheader("BITS IR Assignment - Section A + B")

# ---------------------------------------------------
# Sidebar Options
# ---------------------------------------------------

st.sidebar.header("Preprocessing Options")

lowercase = st.sidebar.checkbox(
    "Lowercase",
    value=True
)

remove_stopwords = st.sidebar.checkbox(
    "Remove Stopwords",
    value=True
)

hyphen_handling = st.sidebar.checkbox(
    "Hyphen Handling",
    value=True
)

stemming = st.sidebar.checkbox(
    "Stemming"
)

lemmatization = st.sidebar.checkbox(
    "Lemmatization"
)

if stemming and lemmatization:
    st.sidebar.warning(
        "Select either Stemming OR Lemmatization"
    )

# ---------------------------------------------------
# Preprocessing Function
# ---------------------------------------------------

def preprocess(text, method="none"):

    if hyphen_handling:
        text = text.replace("-", " ")

    if lowercase:
        text = text.lower()

    # Tokenization
    tokens = re.findall(r"\b\w+\b", text)

    # Stopword Removal
    if remove_stopwords:

        stop_words = set(
            stopwords.words("english")
        )

        tokens = [
            token
            for token in tokens
            if token not in stop_words
        ]

    # Stemming
    if method == "stemming":

        stemmer = PorterStemmer()

        tokens = [
            stemmer.stem(token)
            for token in tokens
        ]

    # Lemmatization
    elif method == "lemmatization":

        lemmatizer = WordNetLemmatizer()

        tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
        ]

    return tokens

# ---------------------------------------------------
# Inverted Index
# ---------------------------------------------------

def create_inverted_index(documents):

    inverted_index = defaultdict(list)

    for doc in documents:

        tokens = preprocess(doc["content"])

        for token in set(tokens):

            inverted_index[token].append(
                doc["name"]
            )

    return inverted_index

# ---------------------------------------------------
# Upload Documents
# ---------------------------------------------------

st.header("1. Upload Document Collection")

uploaded_files = st.file_uploader(
    "Upload TXT Files",
    type=["txt"],
    accept_multiple_files=True
)

documents = []

if uploaded_files:

    for file in uploaded_files:

        try:

            content = file.read().decode("utf-8")

            documents.append(
                {
                    "name": file.name,
                    "content": content
                }
            )

        except Exception as e:

            st.error(
                f"Error reading {file.name}: {e}"
            )

# ---------------------------------------------------
# View Uploaded Documents
# ---------------------------------------------------

if documents:

    st.header("2. Uploaded Documents")

    for doc in documents:

        with st.expander(doc["name"]):

            st.text_area(
                "Content",
                doc["content"],
                height=200
            )

# ---------------------------------------------------
# Section B : Preprocessing
# ---------------------------------------------------

if documents:

    st.header("3. Text Preprocessing")

    for doc in documents:

        st.subheader(doc["name"])

        original_text = doc["content"]

        st.write("### Original Text")
        st.write(original_text)

        # Hyphen Handling
        hyphen_text = original_text.replace(
            "-",
            " "
        )

        st.write("### After Hyphen Handling")
        st.write(hyphen_text)

        # Lowercase
        lower_text = hyphen_text.lower()

        st.write("### After Lowercasing")
        st.write(lower_text)

        # Tokenization
        tokens = re.findall(
            r"\b\w+\b",
            lower_text
        )

        st.write("### Tokens")
        st.write(tokens)

        # Stopword Removal
        stop_words = set(
            stopwords.words("english")
        )

        filtered_tokens = [
            token
            for token in tokens
            if token not in stop_words
        ]

        st.write("### After Stopword Removal")
        st.write(filtered_tokens)

        # Stemming
        stemmer = PorterStemmer()

        stemmed_tokens = [
            stemmer.stem(token)
            for token in filtered_tokens
        ]

        st.write("### Stemmed Tokens")
        st.write(stemmed_tokens)

        # Lemmatization
        lemmatizer = WordNetLemmatizer()

        lemmatized_tokens = [
            lemmatizer.lemmatize(token)
            for token in filtered_tokens
        ]

        st.write("### Lemmatized Tokens")
        st.write(lemmatized_tokens)

        st.markdown("---")

# ---------------------------------------------------
# Inverted Index
# ---------------------------------------------------

if documents:

    st.header("4. Inverted Index")

    inverted_index = create_inverted_index(
        documents
    )

    for term in sorted(
        inverted_index.keys()
    ):

        st.write(
            f"**{term}** → {inverted_index[term]}"
        )

# ---------------------------------------------------
# Stemming vs Lemmatization Comparison
# ---------------------------------------------------

if documents:

    st.header(
        "5. Stemming vs Lemmatization Comparison"
    )

    comparison_data = []

    for doc in documents:

        normal_tokens = preprocess(
            doc["content"]
        )

        stemmed_tokens = preprocess(
            doc["content"],
            method="stemming"
        )

        lemmatized_tokens = preprocess(
            doc["content"],
            method="lemmatization"
        )

        comparison_data.append(
            {
                "Document":
                    doc["name"],

                "Original Terms":
                    len(set(normal_tokens)),

                "Stemmed Terms":
                    len(set(stemmed_tokens)),

                "Lemmatized Terms":
                    len(set(lemmatized_tokens))
            }
        )

    st.dataframe(
        comparison_data,
        use_container_width=True
    )

    st.info(
        """
        Observation:

        • Stemming aggressively reduces words
        to root forms.

        • Lemmatization preserves actual
        dictionary words.

        • Lemmatization generally provides
        better semantic meaning for retrieval.
        """
    )

# ---------------------------------------------------
# Search Interface
# ---------------------------------------------------

st.header("6. Search Documents")

query = st.text_input(
    "Enter Search Query"
)

if query and documents:

    query_tokens = preprocess(query)

    results = []

    for doc in documents:

        doc_tokens = preprocess(
            doc["content"]
        )

        if any(
            token in doc_tokens
            for token in query_tokens
        ):

            results.append(doc)

    st.header("7. Search Results")

    if results:

        st.success(
            f"{len(results)} matching document(s)"
        )

        for result in results:

            with st.expander(
                result["name"]
            ):

                st.write(
                    result["content"]
                )

    else:

        st.error(
            "No matching documents found."
        )