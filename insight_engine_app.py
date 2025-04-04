# Streamlit App: Flexible Insight Engine for Unstructured Market Data

import streamlit as st
import pandas as pd
from openai import OpenAI
import io
import matplotlib.pyplot as plt
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from fpdf import FPDF
import base64
import fitz  # PyMuPDF
from PIL import Image
import requests

# --- OpenAI Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- UI Setup ---
st.title("📊 Market Insight Engine: Multi-Source Analyzer")
st.markdown(
    "Upload up to 5 files: CSV, Excel, PDF (text-based), or images (JPG/PNG). "
    "You can also paste your own notes. When you're ready, click 'Analyze' to generate actionable, informal business insights."
)

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload Data Files (CSV, Excel, PDF, JPG, PNG)",
    type=["csv", "xlsx", "pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

text_input = st.text_area("📋 Paste notes, observations, or qualitative data you want included in the analysis")

# --- Trigger Button ---
analyze_button = st.button("🔍 Analyze")

# --- Helper to Read Files ---
def read_file(file):
    if file.name.endswith("csv"):
        return pd.read_csv(file), None
    elif file.name.endswith("xlsx"):
        return pd.read_excel(file), None
    elif file.name.endswith("pdf"):
        text = ""
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        return None, text
    elif file.name.endswith(("jpg", "jpeg", "png")):
        image_bytes = file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an OCR assistant that extracts clean, structured text from images."},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]}
            ]
        )
        extracted = response.choices[0].message.content
        return None, f"GPT-4o Vision Extracted from {file.name}:\n{extracted}"
    else:
        return None, None

# --- Helper to summarize DataFrames ---
def summarize_dataframe(df, name):
    summary = f"Dataset: {name}\nColumns: {', '.join(df.columns)}\nPreview:\n{df.head(3).to_csv(index=False)}"
    return summary

# --- Email helper ---
def send_email_report(recipient, subject, body, attachment_bytes, filename):
    msg = MIMEMultipart()
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = recipient
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    part = MIMEApplication(attachment_bytes, Name=filename)
    part['Content-Disposition'] = f'attachment; filename="{filename}"'
    msg.attach(part)

    with smtplib.SMTP(st.secrets["EMAIL_SMTP"], st.secrets["EMAIL_PORT"]) as server:
        server.starttls()
        server.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
        server.send_message(msg)

# --- PDF Export Helper ---
def create_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)

# --- Process & Analyze Files ---
if analyze_button and (uploaded_files or text_input):
    summaries = []
    dataframes = []
    extra_texts = []

    for file in uploaded_files:
        df, extracted_text = read_file(file)
        if df is not None:
            dataframes.append(df)
            summaries.append(summarize_dataframe(df, file.name))
        elif extracted_text:
            extra_texts.append(f"File: {file.name}\n{extracted_text}")

    st.subheader("🗂 Dataset Summaries")
    for i, summary in enumerate(summaries):
        with st.expander(f"Dataset {i+1} Summary"):
            st.text(summary)

    combined_summary = "\n\n".join(summaries + extra_texts)
    full_context = combined_summary + "\n\nUser Notes:\n" + text_input

    st.subheader("🧠 AI-Generated Digestible Insights")
    prompt = f"""
    You are an informal but sharp business analyst with a talent for pulling useful, digestible insights out of raw and messy data. Below are summaries of several datasets, observations, and text from reports or screenshots. Topics might include city-level data, pet services, population, dog ownership, or related business metrics.

    Based on this information:
    1. Briefly summarize what each dataset or text block is about
    2. Identify any key patterns, anomalies, or missed opportunities
    3. Calculate any interesting derived metrics (e.g., ratios, gaps, mismatches)
    4. Write 2–4 punchy, actionable business insights — keep the tone casual but smart, like you're giving advice to a founder or investor
    5. If anything is unclear, mention what extra info would help

    Here's the uploaded content:
    {full_context}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an informal but intelligent business analyst who specializes in cross-dataset strategy and storytelling."},
            {"role": "user", "content": prompt}
        ]
    )

    insights = response.choices[0].message.content
    st.markdown(insights)

    st.subheader("📈 Visual Exploration of Uploaded Datasets")
    for i, df in enumerate(dataframes):
        st.markdown(f"**Dataset {i+1} Charts**")
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots()
            ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.set_title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
            st.pyplot(fig)
        else:
            st.info("Not enough numeric data to visualize.")

    st.subheader("📄 Export Report")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        create_pdf(insights, f.name)
        f.flush()
        with open(f.name, "rb") as file_bytes:
            st.download_button("Download Insight Report (PDF)", file_bytes, file_name="insight_report.pdf")

    st.subheader("📬 Email This Report")
    recipient_email = st.text_input("Enter your email to receive this report")
    if st.button("Send Email") and recipient_email:
        with open(f.name, "rb") as fbytes:
            send_email_report(
                recipient=recipient_email,
                subject="Your Market Insight Report",
                body="Attached is your AI-generated insight report based on the uploaded content.",
                attachment_bytes=fbytes.read(),
                filename="insight_report.pdf"
            )
        st.success("📤 Email sent!")
