<<<<<<< HEAD
# IR – Group G31

### Setup & Run

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet punkt
streamlit run app.py
```

The app runs at http://localhost:8501

### Features
- **Section A** – Upload multiple TXT files (falls back to built-in dataset)
- **Section B** – Tokenization, lowercasing, stopword removal, hyphen handling, stemming vs lemmatization comparison with inverted index
- **Section C** – Phrase query using Biword Index and Positional Index with false-positive analysis
- **Section D** – Dictionary search via custom BST and B-Tree with performance benchmarks
- **Section E** – Tolerant retrieval: wildcard queries, k-gram index, edit distance, spell correction, Soundex
- **Section G** – Full inference and discussion report

### Dataset
`dataset.txt` – 25 IR/NLP-domain sentences (used when no files are uploaded)
=======
# IR_G31_Assignment
Information retrival assignment 1 
>>>>>>> e767ed716a61e720f0320ae5aab106a32edc4866
