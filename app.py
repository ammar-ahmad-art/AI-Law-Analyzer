import os
import io
import re
import json
import zipfile
import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import fitz
import faiss
import pytesseract

from PIL import Image
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuMind",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_NAME = "DocuMind"
MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

GROQ_API_KEY = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY", "")
)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "page": "Home",
    "language": "English",
    "filename": None,
    "document_text": "",
    "chunks": [],
    "faiss_index": None,
    "analysis": None,
    "chat_history": [],
    "file_hash": None
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CUSTOM CSS
# ============================================================
#
# IMPORTANT:
# This is CSS only.
#
# The actual UI below uses Streamlit components.
#
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       MAIN APP
       ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at 75% 10%,
                rgba(45, 87, 180, 0.16),
                transparent 30%
            ),
            radial-gradient(
                circle at 15% 85%,
                rgba(105, 55, 190, 0.12),
                transparent 28%
            ),
            #020b18;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 28px;
        padding-bottom: 50px;
    }

    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #06172b 0%,
                #03101e 100%
            );

        border-right:
            1px solid
            rgba(74, 135, 224, 0.20);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 25px;
    }

    /* =========================
       TEXT
       ========================= */

    h1, h2, h3 {
        color: #eef5ff !important;
    }

    p, label {
        color: #a8c3e8 !important;
    }

    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        width: 100%;
        min-height: 42px;

        border-radius: 10px;

        border:
            1px solid
            rgba(61, 139, 255, 0.55);

        background:
            linear-gradient(
                135deg,
                #147fe8,
                #575ce4
            );

        color: white;

        font-weight: 700;

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);

        box-shadow:
            0 8px 25px
            rgba(40, 105, 230, 0.25);
    }

    /* =========================
       FILE UPLOADER
       ========================= */

    div[data-testid="stFileUploader"] {
        border-radius: 15px;
    }

    div[data-testid="stFileUploader"] section {
        background:
            rgba(5, 24, 47, 0.72);

        border:
            1.5px dashed
            rgba(55, 139, 255, 0.55);

        border-radius: 16px;

        padding: 20px;
    }

    /* =========================
       INPUTS
       ========================= */

    .stTextInput input,
    .stTextArea textarea {

        background:
            #061a32 !important;

        color:
            #edf5ff !important;

        border:
            1px solid
            rgba(68, 133, 220, 0.35) !important;

        border-radius:
            10px !important;
    }

    /* =========================
       SELECTBOX
       ========================= */

    div[data-baseweb="select"] > div {
        background: #071a32 !important;

        border:
            1px solid
            rgba(70, 130, 220, 0.35) !important;

        color: white !important;
    }

    /* =========================
       METRICS
       ========================= */

    div[data-testid="stMetric"] {

        background:
            linear-gradient(
                145deg,
                rgba(8, 31, 61, 0.92),
                rgba(3, 17, 34, 0.92)
            );

        border:
            1px solid
            rgba(62, 126, 218, 0.25);

        border-radius: 15px;

        padding: 15px;
    }

    /* =========================
       TABS
       ========================= */

    button[data-baseweb="tab"] {
        color: #9bb9df !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #67aaff !important;
    }

    /* =========================
       EXPANDERS
       ========================= */

    details {
        background:
            rgba(7, 27, 52, 0.75);

        border:
            1px solid
            rgba(61, 128, 220, 0.22);

        border-radius: 12px;

        margin-bottom: 10px;
    }

    /* =========================
       ALERTS
       ========================= */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# GROQ
# ============================================================

@st.cache_resource
def get_groq():

    if not GROQ_API_KEY:
        return None

    return Groq(
        api_key=GROQ_API_KEY
    )


def ask_groq(
    system_prompt,
    user_prompt,
    max_tokens=6000
):

    client = get_groq()

    if client is None:

        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.1,

        max_tokens=max_tokens,

        response_format={
            "type": "json_object"
        }
    )

    return response.choices[0].message.content


# ============================================================
# LEGAL KNOWLEDGE BASE
# ============================================================

LEGAL_SOURCES = [

    {
        "id": "PAK-CA-10",
        "title": "Contract Act, 1872 — Section 10",
        "law": "Contract Act, 1872",
        "description":
            "Concerns requirements relating to agreements becoming contracts.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-CA-15",
        "title": "Contract Act, 1872 — Section 15",
        "law": "Contract Act, 1872",
        "description":
            "Concerns coercion and its relevance to consent.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-CA-16",
        "title": "Contract Act, 1872 — Section 16",
        "law": "Contract Act, 1872",
        "description":
            "Concerns undue influence.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-CA-17",
        "title": "Contract Act, 1872 — Section 17",
        "law": "Contract Act, 1872",
        "description":
            "Concerns fraud.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-CA-18",
        "title": "Contract Act, 1872 — Section 18",
        "law": "Contract Act, 1872",
        "description":
            "Concerns misrepresentation.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-CA-19",
        "title": "Contract Act, 1872 — Section 19",
        "law": "Contract Act, 1872",
        "description":
            "Concerns agreements where consent has been affected by coercion, fraud or misrepresentation.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-CA-23",
        "title": "Contract Act, 1872 — Section 23",
        "law": "Contract Act, 1872",
        "description":
            "Concerns lawful consideration and lawful object.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-CA-28",
        "title": "Contract Act, 1872 — Section 28",
        "law": "Contract Act, 1872",
        "description":
            "Concerns agreements in restraint of legal proceedings, subject to statutory exceptions.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-CA-73",
        "title": "Contract Act, 1872 — Section 73",
        "law": "Contract Act, 1872",
        "description":
            "Concerns compensation for loss or damage caused by breach of contract.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-CA-74",
        "title": "Contract Act, 1872 — Section 74",
        "law": "Contract Act, 1872",
        "description":
            "Concerns compensation where a contract specifies a sum payable or contains a stipulation by way of penalty.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-CONST-25",
        "title": "Constitution of Pakistan — Article 25",
        "law":
            "Constitution of the Islamic Republic of Pakistan",
        "description":
            "Provides constitutional equality before law and equal protection of law.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-CONST-10A",
        "title": "Constitution of Pakistan — Article 10A",
        "law":
            "Constitution of the Islamic Republic of Pakistan",
        "description":
            "Provides for fair trial and due process in specified matters.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-ETO-2002",
        "title": "Electronic Transactions Ordinance, 2002",
        "law":
            "Electronic Transactions Ordinance, 2002",
        "description":
            "Provides a legal framework concerning electronic documents, communications and signatures.",
        "url":
            "https://www.pakistancode.gov.pk/"
    },

    {
        "id": "PAK-PECA",
        "title": "Prevention of Electronic Crimes Act, 2016",
        "law":
            "Prevention of Electronic Crimes Act, 2016",
        "description":
            "Provides the statutory framework concerning specified electronic and information-system offences.",
        "url":
            "https://www.pakistancode.gov.pk/"
    }
]


# ============================================================
# TEXT EXTRACTION
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def extract_pdf(data):

    output = []

    document = fitz.open(
        stream=data,
        filetype="pdf"
    )

    for page_number, page in enumerate(document):

        text = page.get_text(
            "text"
        )

        if text.strip():

            output.append(
                f"[Page {page_number + 1}]\n{text}"
            )

        else:

            try:

                pix = page.get_pixmap(
                    matrix=fitz.Matrix(
                        1.5,
                        1.5
                    )
                )

                image = Image.frombytes(
                    "RGB",
                    [
                        pix.width,
                        pix.height
                    ],
                    pix.samples
                )

                ocr = pytesseract.image_to_string(
                    image,
                    lang="eng+urd"
                )

                if ocr.strip():

                    output.append(
                        f"[Page {page_number + 1} OCR]\n{ocr}"
                    )

            except Exception:
                pass

    document.close()

    return clean_text(
        "\n\n".join(output)
    )


def extract_docx(data):

    with zipfile.ZipFile(
        io.BytesIO(data)
    ) as archive:

        xml = archive.read(
            "word/document.xml"
        ).decode(
            "utf-8",
            errors="ignore"
        )

    xml = re.sub(
        r"</w:p>",
        "\n",
        xml
    )

    xml = re.sub(
        r"<[^>]+>",
        "",
        xml
    )

    return clean_text(
        xml
    )


def extract_xlsx(data):

    output = []

    with zipfile.ZipFile(
        io.BytesIO(data)
    ) as archive:

        files = archive.namelist()

        shared_strings = []

        if "xl/sharedStrings.xml" in files:

            xml = archive.read(
                "xl/sharedStrings.xml"
            ).decode(
                "utf-8",
                errors="ignore"
            )

            strings = re.findall(
                r"<t[^>]*>(.*?)</t>",
                xml,
                re.S
            )

            shared_strings = strings

        sheets = [
            x for x in files
            if re.match(
                r"xl/worksheets/sheet\d+\.xml",
                x
            )
        ]

        for sheet in sheets:

            xml = archive.read(
                sheet
            ).decode(
                "utf-8",
                errors="ignore"
            )

            rows = re.findall(
                r"<row[^>]*>(.*?)</row>",
                xml,
                re.S
            )

            for row in rows:

                values = []

                cells = re.findall(
                    r"<c[^>]*>(.*?)</c>",
                    row,
                    re.S
                )

                for cell in cells:

                    match = re.search(
                        r"<v>(.*?)</v>",
                        cell,
                        re.S
                    )

                    if match:

                        value = match.group(1)

                        values.append(
                            value
                        )

                if values:

                    output.append(
                        " | ".join(values)
                    )

    return clean_text(
        "\n".join(output)
    )


def extract_image(data):

    image = Image.open(
        io.BytesIO(data)
    )

    image = image.convert(
        "RGB"
    )

    try:

        text = pytesseract.image_to_string(
            image,
            lang="eng+urd"
        )

    except Exception:

        text = pytesseract.image_to_string(
            image
        )

    return clean_text(
        text
    )


def extract_document(
    filename,
    data
):

    extension = Path(
        filename
    ).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(data)

    if extension == ".docx":
        return extract_docx(data)

    if extension == ".xlsx":
        return extract_xlsx(data)

    if extension in [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff"
    ]:
        return extract_image(data)

    if extension in [
        ".txt",
        ".md"
    ]:

        return clean_text(
            data.decode(
                "utf-8",
                errors="ignore"
            )
        )

    raise ValueError(
        "Unsupported file format."
    )


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = clean_text(
        text
    )

    paragraphs = text.split(
        "\n"
    )

    chunks = []

    current = ""

    counter = 1

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current) + len(paragraph) < chunk_size:

            if current:
                current += "\n"

            current += paragraph

        else:

            if current:

                chunks.append(
                    {
                        "id":
                            f"DOC-{counter}",
                        "text":
                            current
                    }
                )

                counter += 1

            overlap_text = current[
                -overlap:
            ]

            current = (
                overlap_text
                + "\n"
                + paragraph
            )

    if current:

        chunks.append(
            {
                "id":
                    f"DOC-{counter}",
                "text":
                    current
            }
        )

    return chunks


# ============================================================
# FAISS
# ============================================================

def create_faiss_index(
    chunks
):

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return index


def search_document(
    query,
    chunks,
    index,
    top_k=TOP_K
):

    if not chunks or index is None:
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    k = min(
        top_k,
        len(chunks)
    )

    scores, ids = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        if idx < 0:
            continue

        results.append(
            {
                "id":
                    chunks[idx]["id"],
                "text":
                    chunks[idx]["text"],
                "score":
                    float(score)
            }
        )

    return results


# ============================================================
# LEGAL SOURCE RETRIEVAL
# ============================================================

def search_legal_sources(
    query,
    top_k=5
):

    model = load_embedding_model()

    texts = [
        (
            source["title"]
            + " "
            + source["description"]
        )
        for source in LEGAL_SOURCES
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    k = min(
        top_k,
        len(LEGAL_SOURCES)
    )

    scores, ids = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        source = dict(
            LEGAL_SOURCES[idx]
        )

        source["score"] = float(
            score
        )

        results.append(
            source
        )

    return results


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text):

    try:
        return json.loads(
            text
        )

    except Exception:
        pass

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.I
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if start >= 0 and end >= 0:

        try:

            return json.loads(
                text[start:end + 1]
            )

        except Exception:
            pass

    return {}


# ============================================================
# ANALYSIS
# ============================================================

def analyze_document(
    chunks,
    language
):

    document_context = "\n\n".join(

        f"[{chunk['id']}]\n{chunk['text']}"

        for chunk in chunks[:35]
    )

    legal_sources = search_legal_sources(
        "contract rights obligations penalties termination consent liability unfair clauses",
        6
    )

    legal_context = "\n\n".join(

        f"""
        [{source['id']}]
        {source['title']}
        {source['law']}
        {source['description']}
        """

        for source in legal_sources
    )

    system_prompt = """
You are DocuMind, a Pakistan-focused legal document
analysis assistant.

You provide informational analysis only.

VERY IMPORTANT:

Never invent:
- Pakistani laws
- section numbers
- cases
- legal citations
- legal authorities

Only use legal sources explicitly provided in the prompt.

Document evidence is identified by DOC-X.

Legal sources are identified by PAK-X.

If a legal conclusion is not supported by the provided
legal sources, say that the issue requires professional
legal review.

Do not say that something is definitely illegal or void
unless the supplied evidence directly supports that.

Use cautious language:
- potentially risky
- may warrant review
- could create uncertainty
- appears unusual

Return JSON only.
"""

    user_prompt = f"""
Analyze this document.

Language:
{language}

DOCUMENT:

{document_context}

VERIFIED LEGAL SOURCES:

{legal_context}

Return:

{{
    "summary": "...",
    "overall_assessment": "...",

    "risk_counts": {{
        "high": 0,
        "medium": 0,
        "low": 0
    }},

    "risks": [
        {{
            "title": "...",
            "level": "High",
            "clause": "...",
            "why_risky": "...",
            "potential_impact": "...",
            "recommendation": "...",
            "document_citations": ["DOC-1"],
            "legal_citations": ["PAK-CA-74"]
        }}
    ],

    "rights": [
        {{
            "title": "...",
            "explanation": "...",
            "document_citations": ["DOC-2"],
            "legal_citations": ["PAK-CONST-25"]
        }}
    ],

    "important_clauses": [
        {{
            "title": "...",
            "explanation": "...",
            "document_citations": ["DOC-3"]
        }}
    ],

    "missing_clauses": [
        {{
            "title": "...",
            "explanation": "...",
            "certainty": "Possible"
        }}
    ]
}}

Every document claim should have a DOC citation.

Every legal claim should have a PAK citation.

Do not fabricate citations.
"""

    result = ask_groq(
        system_prompt,
        user_prompt,
        7000
    )

    return parse_json(
        result
    )


# ============================================================
# Q&A
# ============================================================

def answer_question(
    question,
    chunks,
    index,
    language
):

    document_results = search_document(
        question,
        chunks,
        index,
        6
    )

    legal_results = search_legal_sources(
        question,
        5
    )

    document_context = "\n\n".join(

        f"[{item['id']}]\n{item['text']}"

        for item in document_results
    )

    legal_context = "\n\n".join(

        f"""
        [{item['id']}]
        {item['title']}
        {item['description']}
        """

        for item in legal_results
    )

    system_prompt = """
You are DocuMind.

Answer questions about the uploaded document.

Use the document as the primary source.

Use legal sources only for legal context.

Never invent laws, sections, cases or citations.

If evidence is insufficient, say so.

Return JSON only.
"""

    user_prompt = f"""
Language:
{language}

USER QUESTION:
{question}

DOCUMENT EVIDENCE:

{document_context}

LEGAL SOURCES:

{legal_context}

Return:

{{
    "answer": "...",
    "confidence": "High|Medium|Low",
    "document_citations": ["DOC-1"],
    "legal_citations": ["PAK-CA-10"],
    "limitations": "..."
}}
"""

    result = ask_groq(
        system_prompt,
        user_prompt,
        3000
    )

    return parse_json(
        result
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚖️ DocuMind")

    st.markdown(
        "**Pakistan Legal Risk AI**"
    )

    st.caption(
        "Pakistan-Focused AI Legal Risk & Rights Analyzer"
    )

    st.divider()

    navigation = st.radio(
        "Navigation",
        [
            "Home",
            "Upload & Analyze",
            "Scan & Analyze",
            "Risks Found",
            "Ask DocuMind",
            "Legal Sources"
        ],
        key="navigation"
    )

    st.session_state.page = navigation

    st.divider()

    st.markdown(
        "### Language"
    )

    st.session_state.language = st.selectbox(
        "Response language",
        [
            "English",
            "Urdu"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.info(
        "🛡️ Documents are processed during "
        "the current Streamlit session."
    )

    st.markdown(
        "### 💡 Smarter Documents"
    )

    st.caption(
        "Safer decisions through AI-assisted document understanding."
    )


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    hero_left, hero_right = st.columns(
        [2.4, 1]
    )

    with hero_left:

        st.markdown(
            "# DocuMind"
        )

        st.markdown(
            "## Pakistan-Focused AI Legal Risk & Rights Analyzer"
        )

        st.write(
            "Find potentially hidden risks, important clauses, "
            "relevant rights and legal context inside your documents."
        )

        st.info(
            "Upload a contract, agreement, legal document, "
            "spreadsheet or image to get started."
        )

    with hero_right:

        st.markdown(
            "# ⚖️"
        )

        st.markdown(
            "### AI + Pakistani Legal Context"
        )

        st.caption(
            "Document RAG + Legal Knowledge Base + Groq"
        )

    st.divider()

    st.subheader(
        "What DocuMind can do"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            "### 🛡️"
        )

        st.markdown(
            "**Find Hidden Clauses**"
        )

        st.caption(
            "Detect potentially unfair or risky language."
        )

    with c2:

        st.markdown(
            "### 📜"
        )

        st.markdown(
            "**Legal Context**"
        )

        st.caption(
            "Connect findings with a curated Pakistani legal KB."
        )

    with c3:

        st.markdown(
            "### ⚠️"
        )

        st.markdown(
            "**Assess Risk**"
        )

        st.caption(
            "Classify potential issues as High, Medium or Low."
        )

    with c4:

        st.markdown(
            "### 💡"
        )

        st.markdown(
            "**Simple Insights**"
        )

        st.caption(
            "Explain complicated legal language simply."
        )

    st.divider()

    st.subheader(
        "🚀 Quick Start"
    )

    s1, s2, s3 = st.columns(3)

    with s1:

        st.markdown(
            "### 1"
        )

        st.write(
            "Upload your document"
        )

    with s2:

        st.markdown(
            "### 2"
        )

        st.write(
            "Run AI analysis"
        )

    with s3:

        st.markdown(
            "### 3"
        )

        st.write(
            "Review risks, rights and citations"
        )

    st.divider()

    st.subheader(
        "📄 Try the Sample Contract"
    )

    st.write(
        "You can test the complete RAG pipeline without uploading a file."
    )

    if st.button(
        "Use Sample Contract"
    ):

        sample = """
EMPLOYMENT AGREEMENT

This Employment Agreement is made between ABC Technologies
(Pvt.) Ltd. and the Employee.

1. SALARY

The employee shall receive a monthly salary of PKR 100,000.

2. WORKING HOURS

The employee shall work from 9:00 AM to 6:00 PM,
Monday to Saturday.

3. TERMINATION

The company may terminate the employee immediately and
without notice for any reason.

4. CONFIDENTIALITY

The employee shall not disclose confidential company
information during or after employment.

5. PENALTY

If the employee leaves the company before completing
two years, the employee shall pay PKR 1,000,000.

6. DISPUTES

Any dispute shall be resolved according to applicable
laws of Pakistan.

7. MODIFICATION

The company may modify the terms of this agreement
without obtaining the employee's consent.
"""

        chunks = create_chunks(
            sample
        )

        with st.spinner(
            "Building document index..."
        ):

            index = create_faiss_index(
                chunks
            )

        st.session_state.document_text = sample
        st.session_state.chunks = chunks
        st.session_state.faiss_index = index
        st.session_state.filename = "sample_contract.txt"
        st.session_state.file_hash = hashlib.sha256(
            sample.encode()
        ).hexdigest()

        st.success(
            "Sample contract loaded."
        )


# ============================================================
# UPLOAD & ANALYZE
# ============================================================

elif st.session_state.page == "Upload & Analyze":

    st.title(
        "📄 Upload & Analyze"
    )

    st.write(
        "Upload a document and DocuMind will extract, "
        "chunk, embed and analyze it."
    )

    uploaded = st.file_uploader(
        "Choose a document",
        type=[
            "pdf",
            "docx",
            "xlsx",
            "txt",
            "md",
            "png",
            "jpg",
            "jpeg",
            "webp",
            "bmp",
            "tiff"
        ]
    )

    if uploaded:

        data = uploaded.getvalue()

        file_hash = hashlib.sha256(
            data
        ).hexdigest()

        if file_hash != st.session_state.file_hash:

            with st.spinner(
                "Extracting document..."
            ):

                try:

                    text = extract_document(
                        uploaded.name,
                        data
                    )

                    if not text:

                        st.error(
                            "No readable text was found."
                        )

                    else:

                        chunks = create_chunks(
                            text
                        )

                        with st.spinner(
                            "Creating semantic search index..."
                        ):

                            index = create_faiss_index(
                                chunks
                            )

                        st.session_state.document_text = text
                        st.session_state.chunks = chunks
                        st.session_state.faiss_index = index
                        st.session_state.filename = uploaded.name
                        st.session_state.file_hash = file_hash
                        st.session_state.analysis = None

                        st.success(
                            "Document successfully processed."
                        )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )

    if st.session_state.chunks:

        st.divider()

        st.subheader(
            "Document Ready"
        )

        a, b, c = st.columns(3)

        with a:

            st.metric(
                "File",
                st.session_state.filename
            )

        with b:

            st.metric(
                "Chunks",
                len(
                    st.session_state.chunks
                )
            )

        with c:

            st.metric(
                "Language",
                st.session_state.language
            )

        st.divider()

        if st.button(
            "🔎 Analyze Document"
        ):

            with st.spinner(
                "Analyzing document and retrieving Pakistani legal context..."
            ):

                try:

                    result = analyze_document(
                        st.session_state.chunks,
                        st.session_state.language
                    )

                    st.session_state.analysis = result

                    st.success(
                        "Analysis completed."
                    )

                except Exception as e:

                    st.error(
                        f"Analysis failed: {e}"
                    )


# ============================================================
# SCAN
# ============================================================

elif st.session_state.page == "Scan & Analyze":

    st.title(
        "🔍 Scan & Analyze"
    )

    if not st.session_state.chunks:

        st.warning(
            "Upload a document first."
        )

    else:

        st.write(
            f"Current document: **{st.session_state.filename}**"
        )

        query = st.text_input(
            "What would you like to scan for?",
            placeholder=
                "termination, penalty, liability, privacy, payment..."
        )

        if st.button(
            "Scan Document"
        ):

            if not query:

                st.warning(
                    "Enter a scan topic."
                )

            else:

                with st.spinner(
                    "Searching document..."
                ):

                    results = search_document(
                        query,
                        st.session_state.chunks,
                        st.session_state.faiss_index
                    )

                st.subheader(
                    "Relevant Document Sections"
                )

                for result in results:

                    with st.expander(
                        f"{result['id']}  •  Similarity {result['score']:.2f}"
                    ):

                        st.write(
                            result["text"]
                        )

                st.subheader(
                    "Relevant Pakistani Legal Sources"
                )

                legal_results = search_legal_sources(
                    query
                )

                for source in legal_results:

                    with st.expander(
                        f"{source['id']} — {source['title']}"
                    ):

                        st.write(
                            source["description"]
                        )

                        st.link_button(
                            "Open official source",
                            source["url"]
                        )


# ============================================================
# RISKS
# ============================================================

elif st.session_state.page == "Risks Found":

    st.title(
        "⚠️ Risks Found"
    )

    analysis = st.session_state.analysis

    if not analysis:

        st.info(
            "Run a document analysis first."
        )

    else:

        counts = analysis.get(
            "risk_counts",
            {}
        )

        a, b, c = st.columns(3)

        with a:

            st.metric(
                "🔴 High Risk",
                counts.get(
                    "high",
                    0
                )
            )

        with b:

            st.metric(
                "🟠 Medium Risk",
                counts.get(
                    "medium",
                    0
                )
            )

        with c:

            st.metric(
                "🟢 Low Risk",
                counts.get(
                    "low",
                    0
                )
            )

        st.divider()

        risks = analysis.get(
            "risks",
            []
        )

        if not risks:

            st.success(
                "No specific potential risks were identified."
            )

        for risk in risks:

            level = risk.get(
                "level",
                "Medium"
            )

            if level == "High":

                st.error(
                    f"🔴 HIGH RISK — {risk.get('title', '')}"
                )

            elif level == "Medium":

                st.warning(
                    f"🟠 MEDIUM RISK — {risk.get('title', '')}"
                )

            else:

                st.success(
                    f"🟢 LOW RISK — {risk.get('title', '')}"
                )

            with st.expander(
                "View finding"
            ):

                st.markdown(
                    "**Clause**"
                )

                st.write(
                    risk.get(
                        "clause",
                        ""
                    )
                )

                st.markdown(
                    "**Why this may be risky**"
                )

                st.write(
                    risk.get(
                        "why_risky",
                        ""
                    )
                )

                st.markdown(
                    "**Potential impact**"
                )

                st.write(
                    risk.get(
                        "potential_impact",
                        ""
                    )
                )

                st.markdown(
                    "**Recommendation**"
                )

                st.write(
                    risk.get(
                        "recommendation",
                        ""
                    )
                )

                docs = risk.get(
                    "document_citations",
                    []
                )

                laws = risk.get(
                    "legal_citations",
                    []
                )

                if docs:

                    st.markdown(
                        "**Document evidence:**"
                    )

                    st.caption(
                        ", ".join(docs)
                    )

                if laws:

                    st.markdown(
                        "**Legal references:**"
                    )

                    for law_id in laws:

                        source = next(
                            (
                                x for x in LEGAL_SOURCES
                                if x["id"] == law_id
                            ),
                            None
                        )

                        if source:

                            st.link_button(
                                source["title"],
                                source["url"]
                            )


# ============================================================
# ASK DOCUMIND
# ============================================================

elif st.session_state.page == "Ask DocuMind":

    st.title(
        "💬 Ask DocuMind"
    )

    st.write(
        "Ask questions about your uploaded document."
    )

    if not st.session_state.chunks:

        st.warning(
            "Upload a document first."
        )

    else:

        question = st.text_area(
            "Your question",
            placeholder=
                "Can the company terminate this agreement without notice?",
            height=100
        )

        if st.button(
            "Ask DocuMind"
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Searching document and Pakistani legal sources..."
                ):

                    try:

                        answer = answer_question(
                            question,
                            st.session_state.chunks,
                            st.session_state.faiss_index,
                            st.session_state.language
                        )

                        st.session_state.chat_history.append(
                            {
                                "question":
                                    question,
                                "answer":
                                    answer
                            }
                        )

                    except Exception as e:

                        st.error(
                            f"Error: {e}"
                        )

        for chat in reversed(
            st.session_state.chat_history
        ):

            st.markdown(
                "### You"
            )

            st.info(
                chat["question"]
            )

            st.markdown(
                "### ⚖️ DocuMind"
            )

            answer = chat["answer"]

            st.write(
                answer.get(
                    "answer",
                    "No answer generated."
                )
            )

            confidence = answer.get(
                "confidence"
            )

            if confidence:

                st.caption(
                    f"Confidence: {confidence}"
                )

            document_citations = answer.get(
                "document_citations",
                []
            )

            legal_citations = answer.get(
                "legal_citations",
                []
            )

            if document_citations:

                st.caption(
                    "Document evidence: "
                    + ", ".join(
                        document_citations
                    )
                )

            if legal_citations:

                st.markdown(
                    "**Legal sources:**"
                )

                for law_id in legal_citations:

                    source = next(
                        (
                            x
                            for x in LEGAL_SOURCES
                            if x["id"] == law_id
                        ),
                        None
                    )

                    if source:

                        st.link_button(
                            source["title"],
                            source["url"]
                        )

            if answer.get(
                "limitations"
            ):

                st.caption(
                    "⚠️ "
                    + answer["limitations"]
                )

            st.divider()


# ============================================================
# LEGAL SOURCES
# ============================================================

elif st.session_state.page == "Legal Sources":

    st.title(
        "📚 Pakistani Legal Sources"
    )

    st.write(
        "DocuMind's legal citations are restricted to "
        "sources included in this curated knowledge base."
    )

    st.warning(
        "For a production legal product, this knowledge base "
        "should be expanded and reviewed against current official "
        "legislation and Gazette publications."
    )

    for source in LEGAL_SOURCES:

        with st.expander(
            f"{source['id']} — {source['title']}"
        ):

            st.write(
                source["description"]
            )

            st.caption(
                source["law"]
            )

            st.link_button(
                "Open official source",
                source["url"]
            )


# ============================================================
# ANALYSIS SUMMARY
# ============================================================

if (
    st.session_state.analysis
    and st.session_state.page
    in [
        "Home",
        "Upload & Analyze"
    ]
):

    analysis = st.session_state.analysis

    st.divider()

    st.header(
        "📊 Analysis Results"
    )

    tabs = st.tabs(
        [
            "Summary",
            "Risks",
            "Rights",
            "Important Clauses",
            "Missing Clauses"
        ]
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    with tabs[0]:

        st.subheader(
            "Simple Summary"
        )

        st.write(
            analysis.get(
                "summary",
                "No summary available."
            )
        )

        st.subheader(
            "Overall Assessment"
        )

        st.info(
            analysis.get(
                "overall_assessment",
                "No overall assessment available."
            )
        )

    # --------------------------------------------------------
    # RISKS
    # --------------------------------------------------------

    with tabs[1]:

        risks = analysis.get(
            "risks",
            []
        )

        for risk in risks:

            level = risk.get(
                "level",
                "Medium"
            )

            if level == "High":

                st.error(
                    f"🔴 {risk.get('title', '')}"
                )

            elif level == "Medium":

                st.warning(
                    f"🟠 {risk.get('title', '')}"
                )

            else:

                st.success(
                    f"🟢 {risk.get('title', '')}"
                )

            st.write(
                risk.get(
                    "why_risky",
                    ""
                )
            )

    # --------------------------------------------------------
    # RIGHTS
    # --------------------------------------------------------

    with tabs[2]:

        rights = analysis.get(
            "rights",
            []
        )

        if not rights:

            st.info(
                "No potentially relevant rights were identified."
            )

        for right in rights:

            st.subheader(
                right.get(
                    "title",
                    "Potential Right"
                )
            )

            st.write(
                right.get(
                    "explanation",
                    ""
                )
            )

            citations = right.get(
                "legal_citations",
                []
            )

            for citation in citations:

                source = next(
                    (
                        x
                        for x in LEGAL_SOURCES
                        if x["id"] == citation
                    ),
                    None
                )

                if source:

                    st.link_button(
                        source["title"],
                        source["url"]
                    )

    # --------------------------------------------------------
    # IMPORTANT CLAUSES
    # --------------------------------------------------------

    with tabs[3]:

        clauses = analysis.get(
            "important_clauses",
            []
        )

        for clause in clauses:

            st.subheader(
                clause.get(
                    "title",
                    "Important Clause"
                )
            )

            st.write(
                clause.get(
                    "explanation",
                    ""
                )
            )

            citations = clause.get(
                "document_citations",
                []
            )

            if citations:

                st.caption(
                    "Evidence: "
                    + ", ".join(
                        citations
                    )
                )

    # --------------------------------------------------------
    # MISSING CLAUSES
    # --------------------------------------------------------

    with tabs[4]:

        missing = analysis.get(
            "missing_clauses",
            []
        )

        if not missing:

            st.success(
                "No obvious potentially missing clauses were identified."
            )

        for item in missing:

            st.subheader(
                item.get(
                    "title",
                    "Potentially Missing Clause"
                )
            )

            st.write(
                item.get(
                    "explanation",
                    ""
                )
            )

            st.caption(
                "Certainty: "
                + item.get(
                    "certainty",
                    "Possible"
                )
            )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    """
⚠️ Legal Disclaimer: DocuMind is an AI-powered informational
and educational tool. It does not provide professional legal
advice and does not create an attorney-client relationship.
Risk classifications are indicators for further review, not
definitive legal conclusions. Pakistani law may depend on
jurisdiction, facts, amendments and the current official text.
For important matters, consult a qualified Pakistani lawyer.
"""
)
