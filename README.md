# IR System – Group G31

## Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet punkt
streamlit run app.py
```

## Usage
1. Upload the `.txt` files from the `docs/` folder via the app's upload widget
2. The app processes only uploaded files — nothing runs without uploads
3. Use the sidebar to toggle preprocessing options

## Suggested query terms per section

**B – Preprocessing:** `blood`, `patients`, `prescribed`, `infections`
**C – Phrase Queries:** `coronary artery`, `blood glucose monitoring`, `intensive care unit`
**D – Dictionary:** `surgery`, `dialysis`, `biopsy`, `ventilator`, `xray`
**E – Tolerant Retrieval:**
  - Wildcard: `cardio*`, `*ology`, `*gram*`
  - Spell: `surgerey`, `anasthesia`, `diognosis`
  - Edit distance: `phamracy` vs `pharmacy`
  - K-gram: `anti` (k=2)
