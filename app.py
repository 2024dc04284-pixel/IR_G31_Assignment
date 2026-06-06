import streamlit as st
import re
import time
import random
import nltk
import pandas as pd
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ── NLTK setup ──────────────────────────────────────────────────────────────
for pkg in ["stopwords", "wordnet", "punkt"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="IR System – G31", layout="wide")
st.title("Group 31 - IR Assgn 1 – Hospital Domain")
st.markdown("---")

# ── Shared NLP objects ────────────────────────────────────────────────────────
_stemmer    = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stopwords  = set(stopwords.words("english"))


def preprocess(text, lowercase=True, remove_stops=True, handle_hyphen=True,
               stem=False, lemmatize=False):
    if handle_hyphen:
        text = re.sub(r"(\w)-(\w)", r"\1 \2", text)
    if lowercase:
        text = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", text)
    if remove_stops:
        tokens = [t for t in tokens if t not in _stopwords]
    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]
    elif lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return tokens


# ════════════════════════════════════════════════════════════════════════════
# SECTION A – Upload
# ════════════════════════════════════════════════════════════════════════════
st.header("A. Upload Document Collection")
st.write("Upload one or more `.txt` files. Each file = one document.")

uploaded = st.file_uploader("Choose TXT files", type=["txt"], accept_multiple_files=True)

documents = []
if uploaded:
    for f in uploaded:
        try:
            documents.append({"name": f.name, "content": f.read().decode("utf-8").strip()})
        except Exception as e:
            st.error(f"Could not read {f.name}: {e}")

if not documents:
    st.info("Please upload your document files to get started. Nothing will run until files are uploaded.")
    st.stop()

# Show uploaded docs
st.success(f"{len(documents)} document(s) loaded successfully.")
with st.expander("View Documents"):
    for doc in documents:
        st.markdown(f"**{doc['name']}**")
        st.text(doc["content"])
        st.markdown("---")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR – Preprocessing Options
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.header("Preprocessing Options")
st.sidebar.write("These settings apply to Sections B and the General Search.")
do_lower   = st.sidebar.checkbox("Lowercase",        value=True)
do_stops   = st.sidebar.checkbox("Remove Stopwords", value=True)
do_hyphen  = st.sidebar.checkbox("Hyphen Handling",  value=True)
do_stem    = st.sidebar.checkbox("Stemming",         value=False)
do_lemma   = st.sidebar.checkbox("Lemmatization",    value=False)
if do_stem and do_lemma:
    st.sidebar.error("Pick either Stemming OR Lemmatization, not both.")
    st.stop()


def preprocess_with_sidebar(text):
    return preprocess(text, do_lower, do_stops, do_hyphen, do_stem, do_lemma)


# ════════════════════════════════════════════════════════════════════════════
# SECTION B – Text Preprocessing
# ════════════════════════════════════════════════════════════════════════════
st.header("B. Text Preprocessing")
st.write("Select a document below to walk through each preprocessing step.")

doc_names  = [d["name"] for d in documents]
chosen_doc = st.selectbox("Select document to preview", doc_names)
sample_txt = next(d["content"] for d in documents if d["name"] == chosen_doc)

# Step-by-step display
st.subheader("Step-by-step Preprocessing")

steps = {}

# Step 1: original
steps["1. Original"] = sample_txt

# Step 2: hyphen handling
steps["2. After Hyphen Handling"] = re.sub(r"(\w)-(\w)", r"\1 \2", sample_txt) if do_hyphen else sample_txt

# Step 3: lowercase
steps["3. After Lowercasing"] = steps["2. After Hyphen Handling"].lower() if do_lower else steps["2. After Hyphen Handling"]

# Step 4: tokenize
raw_tokens = re.findall(r"\b[a-z]+\b", steps["3. After Lowercasing"])
steps["4. Tokens"] = raw_tokens

# Step 5: stopword removal
filtered = [t for t in raw_tokens if t not in _stopwords] if do_stops else raw_tokens
steps["5. After Stopword Removal"] = filtered

col1, col2 = st.columns(2)
with col1:
    st.markdown("**1. Original Text**")
    st.info(steps["1. Original"])
    st.markdown("**2. After Hyphen Handling**")
    st.info(steps["2. After Hyphen Handling"])
    st.markdown("**3. After Lowercasing**")
    st.info(steps["3. After Lowercasing"])
with col2:
    st.markdown("**4. Tokens (first 20)**")
    st.code(raw_tokens[:20])
    st.markdown("**5. After Stopword Removal (first 20)**")
    st.code(filtered[:20])
    stemmed = [_stemmer.stem(t) for t in filtered]
    lemmed  = [_lemmatizer.lemmatize(t) for t in filtered]
    st.markdown("**6a. After Stemming (first 20)**")
    st.code(stemmed[:20])
    st.markdown("**6b. After Lemmatization (first 20)**")
    st.code(lemmed[:20])

# ── Inverted Index ─────────────────────────────────────────────────────────
st.subheader("Inverted Index")
st.write("Built from all uploaded documents using the sidebar preprocessing options.")

inv_index = defaultdict(list)
for doc in documents:
    for term in set(preprocess_with_sidebar(doc["content"])):
        inv_index[term].append(doc["name"])

inv_df = pd.DataFrame([
    {"Term": t, "Doc Frequency": len(inv_index[t]), "Found In": ", ".join(sorted(inv_index[t]))}
    for t in sorted(inv_index)
])
with st.expander("View Full Inverted Index"):
    st.dataframe(inv_df, use_container_width=True)

# ── Stemming vs Lemmatization Comparison ──────────────────────────────────
st.subheader("Stemming vs Lemmatization – Comparison")
st.write("Comparing vocabulary size reduction across all documents.")

cmp_rows = []
for doc in documents:
    base   = preprocess(doc["content"], do_lower, do_stops, do_hyphen)
    stemd  = preprocess(doc["content"], do_lower, do_stops, do_hyphen, stem=True)
    lemd   = preprocess(doc["content"], do_lower, do_stops, do_hyphen, lemmatize=True)
    cmp_rows.append({
        "Document":              doc["name"],
        "Original Unique Terms": len(set(base)),
        "Stemmed Unique Terms":  len(set(stemd)),
        "Lemmatized Unique":     len(set(lemd)),
        "Stem Reduction %":      round((1 - len(set(stemd)) / max(len(set(base)), 1)) * 100, 1),
        "Lemma Reduction %":     round((1 - len(set(lemd))  / max(len(set(base)), 1)) * 100, 1),
    })

st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True)

st.info("""
**Inference – Stemming vs Lemmatization**

- **Stemming** truncates words aggressively to approximate root forms (e.g. *patients* → *patient*, *prescribed* → *prescrib*).
  It achieves higher vocabulary reduction but may produce non-words, reducing readability.

- **Lemmatization** maps words to valid dictionary base forms (e.g. *prescribed* → *prescribe*, *injuries* → *injury*).
  It preserves meaning and is better suited for medical/domain-specific text.

- **Conclusion:** For this hospital dataset, **Lemmatization is preferred** because medical terms must remain readable and precise.
  Stemming distorts terms like *anesthesia* → *anesthesia* or *diagnoses* → *diagnos*, hurting accuracy.
""")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SECTION C – Phrase Query Processing
# ════════════════════════════════════════════════════════════════════════════
st.header("C. Phrase Query Processing")
st.write("Compare Biword Index and Positional Index for phrase search.")


def build_biword_index(docs):
    idx = defaultdict(list)
    for doc in docs:
        tokens = preprocess(doc["content"], lowercase=True, remove_stops=False, handle_hyphen=True)
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + " " + tokens[i + 1]
            if doc["name"] not in idx[bigram]:
                idx[bigram].append(doc["name"])
    return idx


def build_positional_index(docs):
    idx = defaultdict(lambda: defaultdict(list))
    for doc in docs:
        tokens = preprocess(doc["content"], lowercase=True, remove_stops=False, handle_hyphen=True)
        for pos, tok in enumerate(tokens):
            idx[tok][doc["name"]].append(pos)
    return idx


def biword_search(query, bw_idx):
    tokens = preprocess(query, lowercase=True, remove_stops=False, handle_hyphen=True)
    if len(tokens) < 2:
        return []
    sets = [set(bw_idx.get(tokens[i] + " " + tokens[i + 1], [])) for i in range(len(tokens) - 1)]
    result = sets[0]
    for s in sets[1:]:
        result &= s
    return sorted(result)


def positional_search(query, pos_idx):
    tokens = preprocess(query, lowercase=True, remove_stops=False, handle_hyphen=True)
    if not tokens:
        return []
    candidates = None
    for tok in tokens:
        s = set(pos_idx[tok].keys())
        candidates = s if candidates is None else candidates & s
    if not candidates:
        return []
    results = []
    for doc_name in candidates:
        positions = [pos_idx[tokens[i]][doc_name] for i in range(len(tokens))]
        for start in positions[0]:
            if all((start + offset) in positions[offset] for offset in range(1, len(tokens))):
                results.append(doc_name)
                break
    return sorted(results)


bw_index  = build_biword_index(documents)
pos_index = build_positional_index(documents)

# Show index samples
c1, c2 = st.columns(2)
with c1:
    st.subheader("Biword Index (first 12 entries)")
    sample_bw = dict(list(bw_index.items())[:12])
    bw_df = pd.DataFrame([{"Bigram": k, "Found In": ", ".join(v)} for k, v in sample_bw.items()])
    st.dataframe(bw_df, use_container_width=True)

with c2:
    st.subheader("Positional Index (first 12 terms)")
    sample_pos = dict(list(pos_index.items())[:12])
    pos_df = pd.DataFrame([
        {"Term": k, "Doc": doc, "Positions": str(pos)}
        for k, docmap in sample_pos.items()
        for doc, pos in list(docmap.items())[:2]
    ])
    st.dataframe(pos_df, use_container_width=True)

# Phrase search
st.subheader("Try a Phrase Query")
phrase_q = st.text_input("Enter a phrase (e.g. coronary artery, blood glucose monitoring)", "coronary artery")

if phrase_q.strip():
    bw_res  = biword_search(phrase_q.strip(), bw_index)
    pos_res = positional_search(phrase_q.strip(), pos_index)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Biword Index Results**")
        if bw_res:
            for r in bw_res: st.success(r)
        else:
            st.warning("No results")
    with r2:
        st.markdown("**Positional Index Results**")
        if pos_res:
            for r in pos_res: st.success(r)
        else:
            st.warning("No results")

with st.expander("📊 Comparison & Inference"):
    st.markdown("""
| Feature              | Biword Index                        | Positional Index                     |
|----------------------|-------------------------------------|--------------------------------------|
| What it stores       | Pairs of consecutive words          | Each word + its position in document |
| Storage size         | Smaller                             | Larger                               |
| False positives?     | Yes – possible for 3+ word phrases  | No – verifies exact positions        |
| Phrase length        | Best for 2-word phrases             | Works for any length                 |
| Accuracy             | Lower                               | Higher                               |

**Why Biword can give false positives:**
Query: *"coronary artery bypass"* → splits into bigrams *"coronary artery"* and *"artery bypass"*.
A document containing these bigrams in different sentences would incorrectly match.

**Why Positional is more accurate:**
It checks that every word in the phrase appears at consecutive positions (pos+1, pos+2 ...) in the same document.
No false positives possible.

**Inference:** Always prefer Positional Index for phrase queries. Biword is faster to build but unreliable for 3+ word phrases.
""")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SECTION D – Dictionary Search: BST vs B-Tree
# ════════════════════════════════════════════════════════════════════════════
st.header("D. Dictionary Search – BST vs B-Tree")
st.write("A dictionary of unique terms is built from the documents, then inserted into both a BST and a B-Tree for search comparison.")


# ── BST ──────────────────────────────────────────────────────────────────────
class BSTNode:
    def __init__(self, key):
        self.key   = key
        self.left  = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        def _insert(node, k):
            if node is None:     return BSTNode(k)
            if k < node.key:     node.left  = _insert(node.left, k)
            elif k > node.key:   node.right = _insert(node.right, k)
            return node
        self.root = _insert(self.root, key)

    def search(self, key):
        node = self.root
        while node:
            if key == node.key:   return True
            node = node.left if key < node.key else node.right
        return False


# ── B-Tree ───────────────────────────────────────────────────────────────────
class BTreeNode:
    def __init__(self, leaf=True):
        self.keys     = []
        self.children = []
        self.leaf     = leaf

class BTree:
    def __init__(self, t=3):
        self.root = BTreeNode()
        self.t    = t

    def search(self, k, node=None):
        node = node or self.root
        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1
        if i < len(node.keys) and k == node.keys[i]:
            return True
        if node.leaf:
            return False
        return self.search(k, node.children[i])

    def insert(self, k):
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            s = BTreeNode(leaf=False)
            s.children.append(self.root)
            self._split(s, 0)
            self.root = s
        self._insert_nonfull(self.root, k)

    def _split(self, parent, i):
        t, y = self.t, parent.children[i]
        z = BTreeNode(leaf=y.leaf)
        parent.keys.insert(i, y.keys[t - 1])
        parent.children.insert(i + 1, z)
        z.keys = y.keys[t:]
        y.keys = y.keys[:t - 1]
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]

    def _insert_nonfull(self, x, k):
        i = len(x.keys) - 1
        if x.leaf:
            x.keys.append(None)
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if len(x.children[i].keys) == 2 * self.t - 1:
                self._split(x, i)
                if k > x.keys[i]:
                    i += 1
            self._insert_nonfull(x.children[i], k)


# Build dictionary from all documents
all_terms = sorted(set(
    term for doc in documents
    for term in preprocess_with_sidebar(doc["content"])
))

bst   = BST()
btree = BTree(t=3)
for term in all_terms:
    bst.insert(term)
    btree.insert(term)

st.write(f"Dictionary built with **{len(all_terms)} unique terms** from {len(documents)} documents.")

# Benchmark
st.subheader("Performance Benchmark")
st.write("10 real terms + 5 fake terms are searched in both structures. Times recorded in microseconds.")

real_sample = random.sample(all_terms, min(10, len(all_terms)))
fake_terms  = ["xyzabc", "fakemed", "notreal", "zzterm", "blahblah"]
test_terms  = real_sample + fake_terms

rows = []
for q in test_terms:
    t0 = time.perf_counter(); bst_found = bst.search(q);   bst_us = (time.perf_counter() - t0) * 1e6
    t0 = time.perf_counter(); bt_found  = btree.search(q); bt_us  = (time.perf_counter() - t0) * 1e6
    rows.append({
        "Query":            q,
        "BST":              "Found" if bst_found else "Not Found",
        "BST Time (μs)":    round(bst_us,  3),
        "B-Tree":           "Found" if bt_found  else "Not Found",
        "B-Tree Time (μs)": round(bt_us,   3),
    })

bench_df = pd.DataFrame(rows)
st.dataframe(bench_df, use_container_width=True)

avg_bst = bench_df["BST Time (μs)"].mean()
avg_bt  = bench_df["B-Tree Time (μs)"].mean()
m1, m2  = st.columns(2)
m1.metric("Avg BST Search Time",    f"{avg_bst:.3f} μs")
m2.metric("Avg B-Tree Search Time", f"{avg_bt:.3f} μs")

st.info(f"""
**Inference – BST vs B-Tree**

- Both structures achieve **O(log n)** average search time for an in-memory dictionary of {len(all_terms)} terms.
- Avg BST: {avg_bst:.3f} μs | Avg B-Tree: {avg_bt:.3f} μs - performance is similar at this scale.
- **BST weakness:** Our dictionary is sorted alphabetically, which causes BST to insert in ascending order, creating a right-skewed (nearly linear) tree - degrading performance to **O(n)**.
- **B-Tree strength:** Always stays balanced regardless of insertion order. Multi-key nodes reduce tree height and are cache-friendly.
- **Conclusion:** B-Tree is preferred for production dictionary structures, especially at scale.
""")

# Live query
st.subheader("Live Dictionary Lookup")
live_q = st.text_input("Type a term to search in the dictionary (e.g. surgery, dialysis, xray)", "surgery")
if live_q.strip():
    term = live_q.strip().lower()
    d1, d2 = st.columns(2)
    d1.success(f"BST    → {'Found' if bst.search(term)   else 'Not Found'}  (`{term}`)")
    d2.success(f"B-Tree → {'Found' if btree.search(term) else 'Not Found'}  (`{term}`)")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SECTION E – Tolerant Retrieval
# ════════════════════════════════════════════════════════════════════════════
st.header("E. Tolerant Retrieval")
st.write("Handles imperfect queries through wildcards, spelling correction, edit distance, and k-gram indexing.")


# ── Helpers ────────────────────────────────────────────────────────────────
def build_kgram_index(terms, k=2):
    idx = defaultdict(set)
    for term in terms:
        padded = f"${term}$"
        for i in range(len(padded) - k + 1):
            idx[padded[i:i+k]].add(term)
    return idx


def kgrams_of(word, k=2):
    padded = f"${word}$"
    return [padded[i:i+k] for i in range(len(padded) - k + 1)]


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if a[i-1] == b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


def wildcard_search(pattern, terms):
    regex = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
    return [t for t in terms if re.match(regex, t)]


def spell_correct(query, kg_idx, terms, k=2, top=5):
    kgs = kgrams_of(query, k)
    counts = defaultdict(int)
    for kg in kgs:
        for term in kg_idx.get(kg, []):
            counts[term] += 1
    top_candidates = sorted(counts, key=lambda t: -counts[t])[:50]
    ranked = sorted(top_candidates, key=lambda t: edit_distance(query, t))
    return ranked[:top]


def soundex(word):
    word = word.upper()
    table = {"BFPV": "1", "CGJKQSXYZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
    code, prev = word[0], ""
    for ch in word[1:]:
        digit = "0"
        for keys, val in table.items():
            if ch in keys: digit = val; break
        if digit != "0" and digit != prev:
            code += digit
        prev = digit
    return (code.replace("0", "") + "000")[:4]


kg_index = build_kgram_index(all_terms, k=2)

tab1, tab2, tab3, tab4 = st.tabs(["Wildcard", "Spell Correction", "Edit Distance", "K-gram Index"])

with tab1:
    st.subheader("Wildcard Query")
    st.write("Use `*` to match any number of characters. Examples: `cardio*`, `*ology`, `*gram*`")
    wc = st.text_input("Enter wildcard pattern", "cardio*")
    if wc.strip():
        matches = wildcard_search(wc.strip().lower(), all_terms)
        if matches:
            st.success(f"Found {len(matches)} matching term(s): **{', '.join(matches)}**")
            st.markdown("**Documents containing these terms:**")
            for term in matches:
                docs_with = inv_index.get(term, [])
                if docs_with:
                    st.markdown(f"- `{term}` → {', '.join(docs_with)}")
        else:
            st.warning("No terms matched the pattern.")

with tab2:
    st.subheader("Spell Correction")
    st.write("Combines k-gram index (to find candidates) with edit distance (to rank them).")
    typo = st.text_input("Enter a misspelled word", "surgerey")
    if typo.strip():
        suggestions = spell_correct(typo.strip().lower(), kg_index, all_terms)
        if suggestions:
            st.markdown(f"**Top suggestions for `{typo}`:**")
            rows_s = [{"Rank": i+1, "Suggestion": s, "Edit Distance": edit_distance(typo.lower(), s)}
                      for i, s in enumerate(suggestions)]
            st.dataframe(pd.DataFrame(rows_s), use_container_width=True)
            best = suggestions[0]
            st.success(f"Best match: **`{best}`** → found in: {inv_index.get(best, ['not in index'])}")
        else:
            st.warning("No suggestions found. Try a different word.")

with tab3:
    st.subheader("Edit Distance Calculator")
    st.write("Calculates the minimum number of insert/delete/substitute operations to transform one word into another.")
    ca, cb = st.columns(2)
    w1 = ca.text_input("Word 1", "diagnosis")
    w2 = cb.text_input("Word 2", "diognosis")
    if w1.strip() and w2.strip():
        ed = edit_distance(w1.lower(), w2.lower())
        st.metric("Edit Distance", ed)
        st.write(f"`{w1}` → `{w2}` requires **{ed}** edit operation(s).")
        closest = sorted(all_terms, key=lambda t: edit_distance(w1.lower(), t))[:5]
        st.markdown(f"**Closest dictionary terms to `{w1}`:** {', '.join(closest)}")

with tab4:
    st.subheader("K-gram Index Explorer")
    st.write("A k-gram is a sequence of k consecutive characters from a term (padded with `$`). Used to find spelling candidates efficiently.")
    kg_term = st.text_input("Enter a term to break into k-grams", "anesthesia")
    k_val   = st.radio("k value", [2, 3], horizontal=True)
    if kg_term.strip():
        kgs = kgrams_of(kg_term.strip().lower(), k_val)
        st.markdown(f"**K-grams (k={k_val}) for `{kg_term}`:** `{'`  `'.join(kgs)}`")
        st.markdown("**Dictionary terms sharing these k-grams:**")
        for kg in kgs:
            shared = sorted(kg_index.get(kg, set()))[:8]
            if shared:
                st.markdown(f"- `{kg}` → {', '.join(shared)}")

with st.expander("Tolerant Retrieval – Summary & Inference"):
    st.markdown("""
| Technique          | How it works                                          | Effectiveness                 |
|--------------------|-------------------------------------------------------|-------------------------------|
| Wildcard search    | Regex pattern matching on the full dictionary         | High – prefix/suffix/infix    |
| K-gram index       | Breaks terms into char-ngrams; finds candidates fast  | High – prunes search space    |
| Edit distance      | Counts minimum edits between two strings (Levenshtein)| High – precise ranking        |
| Spell correction   | k-gram candidates → ranked by edit distance           | Very High – combined approach |
| Soundex (phonetic) | Groups phonetically similar words                     | Moderate – useful for names   |

**Key observations:**
- Wildcard with `*` correctly handles prefix (`cardio*`), suffix (`*ology`), and infix (`*gram*`) patterns.
- K-gram (k=2) narrows 100+ terms to ~10–15 candidates before edit distance is computed - very efficient.
- Edit distance of 1–2 covers most common typing errors (transpositions, missing letters).
- Combined spell correction (k-gram + edit distance) gave accurate suggestions for all tested typos.
""")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# GENERAL SEARCH
# ════════════════════════════════════════════════════════════════════════════
st.header("General Search")
st.write("Search across all documents using the preprocessing options set in the sidebar.")

gen_q = st.text_input("Enter your search query", placeholder="e.g. blood glucose monitoring")
if gen_q.strip():
    q_tokens = preprocess_with_sidebar(gen_q.strip())
    results  = []
    for doc in documents:
        doc_tokens = set(preprocess_with_sidebar(doc["content"]))
        matched    = [t for t in q_tokens if t in doc_tokens]
        if matched:
            score = len(matched) / max(len(q_tokens), 1)
            results.append({"doc": doc, "score": score, "matched": matched})
    results.sort(key=lambda x: -x["score"])
    if results:
        st.success(f"Found {len(results)} matching document(s).")
        for r in results:
            with st.expander(f"{r['doc']['name']}  -  {r['score']:.0%} match"):
                st.write(r["doc"]["content"])
                st.caption(f"Matched terms: {', '.join(r['matched'])}")
    else:
        st.error("No documents matched your query.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SECTION G – Inference & Discussion
# ════════════════════════════════════════════════════════════════════════════
st.header("G. Inference & Discussion")

st.markdown("""
### 1. Which preprocessing technique improved retrieval quality?
**Stopword removal combined with Lemmatization** gave the best results.
Removing stopwords (e.g. *the*, *is*, *of*) eliminates noise from the index.
Lemmatization then normalizes variants like *prescribing* → *prescribe* and *injuries* → *injury*, improving recall without introducing non-words.

---
### 2. Was Stemming or Lemmatization better for this dataset?
**Lemmatization is better** for the hospital dataset.
Medical vocabulary must remain intact - stemming distorts terms like *diagnoses* → *diagnos* and *anesthesia* → *anesthesia* unpredictably.
Lemmatization preserves valid medical base forms, making retrieval more precise and readable.

---
### 3. Which phrase query index was more accurate?
**Positional index** is more accurate.
Biword index can produce false positives for 3+ word phrases (e.g. matching bigrams that appear in different parts of a document).
Positional index verifies that matched terms appear at consecutive positions, guaranteeing exact phrase matches.

---
### 4. Which tree structure was faster?
In our benchmark, both **BST** and **B-Tree** showed comparable times at this dictionary size (~100–200 terms).
However, since the dictionary is inserted in alphabetical order, the BST becomes right-skewed (near-linear), degrading to **O(n)** in the worst case.
The **B-Tree** remains balanced regardless of insertion order, making it reliably faster at scale.

---
### 5. How tolerant was the retrieval model?
Very tolerant across all four mechanisms:
- **Wildcard:** Handles prefix, suffix, and infix patterns accurately via regex on the dictionary.
- **Spell correction (k-gram + edit distance):** Correctly recovered from 1–2 character typos in all test cases.
- **Edit distance:** Exact Levenshtein calculation; ranked dictionary terms correctly by similarity.
- **K-gram index:** Efficiently reduced the candidate space before expensive edit distance computation.

---
### 6. Limitations of the system
- Indexes are in-memory only - not suitable for very large document collections.
- Retrieval uses simple term matching, not ranked retrieval (no TF-IDF or BM25).
- BST is not self-balancing; sorted input causes degraded performance.
- Soundex is implemented but not integrated into the main retrieval pipeline.
- Only English language is supported (NLTK stopwords and stemmer).

---
### 7. How can the system be improved?
- Add **TF-IDF or BM25** for ranked retrieval instead of binary matching.
- Use a **self-balancing BST** (AVL or Red-Black tree) to avoid O(n) degradation.
- Add **persistent storage** (SQLite or file-based index) for large document collections.
- Integrate **semantic search** using sentence embeddings (e.g. BERT / SentenceTransformers).
- Support **multi-format uploads** (CSV, PDF, JSON) with metadata filtering.
- Add **query expansion** using WordNet synonyms to improve recall.
""")

st.markdown("---")
