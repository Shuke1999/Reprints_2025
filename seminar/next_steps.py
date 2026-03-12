"""Next-stage proposal: normalize, validate, hard-case, framework."""
import streamlit as st


def render():
    st.header("Next-stage Proposal", divider="rainbow")

    st.markdown("### Immediate next step")
    st.markdown("Normalize the first-round discovered signals into a cleaner inventory (e.g. lexical-syntactic fidelity, discourse completeness, paragraph preservation, minimal contextual integration, attribution markers, artifact-related).")

    st.markdown("### Next experiments")
    st.markdown(
        "1. **Signal normalization** – Merge repeated signal names into a cleaner taxonomy.  \n"
        "2. **Signal validation** – Test whether these signals hold on the main evaluation set.  \n"
        "3. **Hard-case analysis** – Check whether the signals explain Group B reprints and hard negatives.  \n"
        "4. **Detection framework construction** – Convert validated signals into a practical reprint detection framework."
    )

