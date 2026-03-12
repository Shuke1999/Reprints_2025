# Seminar presentation (Streamlit)

Reprint detection seminar: from naive signals to LLM-assisted discovery.

## Run

From the **repository root**:

```bash
streamlit run seminar/app.py
```

Or with custom data root:

```bash
REPRINTS_DATA_DIR=/path/to/data streamlit run seminar/app.py
```

## Structure

- `app.py` – Entry point; sidebar navigation only.
- `data.py` – Paths (see plan §10), constants, feature comparison and threshold tables.
- `hero.py`, `overview.py`, `naive_signals.py`, `naive_results.py`, `restructuring.py`, `llm_discovery.py`, `next_steps.py` – One page each.
- `data_search.py` – Data & Search: GT preview, top-k by feature, derived data pointers.

All paths in `data.py` are relative to `DATA_DIR` (default: `repo/data`). Replace placeholders (mermaid / captions) with your own figures or `st.image` as needed.
