import streamlit as st
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


st.set_page_config(page_title="AI Document Intelligence Assistant", layout="wide")


model = SentenceTransformer("all-MiniLM-L6-v2")


st.sidebar.title("Navigation")
st.sidebar.info("Agentic AI Powered Document Assistant")


st.title("📄 AI Document Intelligence Assistant")
st.write("Analyze Research Papers, Resume and Reports using AI")

document_type = st.sidebar.selectbox(
    "Select Document Type",
    ["Research Paper", "Resume", "Report"]
)


if document_type == "Research Paper":
    feature = st.sidebar.selectbox(
        "Select Feature",
        ["Ask Questions", "Research Gap Finder"]
    )
elif document_type == "Resume":
    feature = st.sidebar.selectbox(
        "Select Feature",
        ["Resume Analyzer"]
    )
else:
    feature = st.sidebar.selectbox(
        "Select Feature",
        ["Summary", "Key Findings"]
    )


uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    st.success("PDF uploaded successfully!")

    pdf = PdfReader(uploaded_file)

    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

   
    if document_type == "Research Paper":

        if feature == "Ask Questions":

            if "messages" not in st.session_state:
                st.session_state.messages = []

            if st.sidebar.button("Clear Chat"):
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            question = st.chat_input("Ask question about paper...")

            if question:
                with st.chat_message("user"):
                    st.markdown(question)

                query_embedding = model.encode([question])

                distances, indices = index.search(
                    np.array(query_embedding, dtype=np.float32), k=1
                )

                context = chunks[indices[0][0]]

                prompt = f"""
Answer based on this research paper context:

{context}

Question: {question}
"""

                with st.spinner("Thinking..."):
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}]
                    )

                answer = response.choices[0].message.content

                with st.chat_message("assistant"):
                    st.markdown(answer)

        elif feature == "Research Gap Finder":

            if st.button("Find Research Gaps"):

                prompt = f"""
Analyze this research paper and identify:

1. Research gaps
2. Limitations
3. Future scope
4. Improvement opportunities

Paper Content:
{text[:2000]}
"""

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )

                st.subheader("Analysis Result")
                st.write(response.choices[0].message.content)

  
    elif document_type == "Resume":

        if st.button("Analyze Resume"):

            prompt = f"""
Analyze this resume and provide:

1. ATS Score
2. Missing Skills
3. Best Matching Roles
4. Improvement Suggestions

Resume Content:
{text[:2000]}
"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )

            st.subheader("Analysis Result")
            st.write(response.choices[0].message.content)

   
    elif document_type == "Report":

        if feature == "Summary":

            if st.button("Generate Summary"):

                prompt = f"""
Summarize this report:

{text[:2000]}
"""

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )

                st.subheader("Analysis Result")
                st.write(response.choices[0].message.content)

        elif feature == "Key Findings":

            if st.button("Find Key Insights"):

                prompt = f"""
Analyze this report and provide:

1. Key Findings
2. Important Insights
3. Recommendations

Report Content:
{text[:2000]}
"""

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )

                st.subheader("Analysis Result")
                st.write(response.choices[0].message.content)