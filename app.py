#streamlit app for annotation tool
import streamlit as st
import pandas as pd
import tempfile
from pyvis.network import Network
import networkx as nx
from NER_RE_MCN_2025 import MedicalPipeline  # Your pipeline import

@st.cache_resource
def load_pipeline():
    return MedicalPipeline()

def visualize_graph(entities_df, relations_df, height_px=600):
    G = nx.Graph()

    # Add nodes: use candidate_conceptId as node id, label with entity_text
    for _, row in entities_df.iterrows():
        node_id = row.get('candidate_conceptId')
        label = row.get('entity_text', 'Unknown')
        if node_id:
            G.add_node(node_id, label=label)

    # Add edges with relations between subject_text and object_text nodes
    for _, row in relations_df.iterrows():
        source = row.get('subject_text')
        target = row.get('object_text')
        relation = row.get('matched_relations', '')
        if source and target:
            G.add_edge(source, target, label=relation)

    net = Network(height=f"{height_px}px", width="100%", notebook=False)

    net.from_nx(G)

    # Add edge labels for better interaction
    for edge in net.edges:
        edge['title'] = edge.get('label', '')
        edge['label'] = edge.get('label', '')

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(temp_file.name)

    return temp_file.name

def read_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        else:
            st.warning("Unsupported file type. Please upload a .txt or .pdf file.")
            return None
    return None

def main():
    st.title("Medical Concept Annotation Pipeline with Graph Visualization")

    pipeline = load_pipeline()

    input_mode = st.radio("Select input mode:", options=["Text Input", "Upload Document"])

    clinical_text = ""

    if input_mode == "Text Input":
        clinical_text = st.text_area("Enter clinical narrative text here:", height=250)
    else:
        uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=["txt", "pdf"])
        if uploaded_file is not None:
            clinical_text = read_uploaded_file(uploaded_file)
            if clinical_text:
                st.text_area("Document content preview:", clinical_text, height=250, disabled=True)

    if st.button("Annotate"):
        if not clinical_text.strip():
            st.warning("Please enter or upload clinical text to annotate.")
            return

        with st.spinner("Running annotation pipeline..."):
            entities_df, relations_df = pipeline.run_pipeline(clinical_text)

        st.subheader("Normalized Entities")
        if entities_df.empty:
            st.write("No entities detected.")
        else:
            st.dataframe(entities_df)

        st.subheader("Extracted Relations")
        if relations_df.empty:
            st.write("No relations found.")
        else:
            st.dataframe(relations_df)

        height_px = 700
        width_px = 700
        st.subheader("Entity-Relation Graph")
        graph_file = visualize_graph(entities_df, relations_df, height_px=height_px)
        with open(graph_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=height_px, width=width_px)

if __name__ == "__main__":
    main()
