
import os
import io
import re
import json
import textwrap
import shutil
from typing import Any, Dict, List, Tuple

import streamlit as st
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pakistan AI Legal Risk & Rights Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Pakistan AI Legal Risk & Rights Analyzer"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# A current Groq model can be overridden with the GROQ_MODEL
# environment variable / Streamlit secret.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

SUPPORTED_EXTENSIONS = ["pdf", "png", "jpg", "jpeg"]


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .app-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .app-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.25rem;
    }

    .risk-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 14px;
        background: #ffffff;
    }

    .risk-high {
        border-left: 6px solid #dc2626;
    }

    .risk-medium {
        border-left: 6px solid #f59e0b;
    }

    .risk-low {
        border-left: 6px solid #16a34a;
    }

    .metric-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        background: #ffffff;
        text-align: center;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.85rem;
    }

    .disclaimer {
        border: 1px solid #fbbf24;
        background: #fffbeb;
        padding: 14px 16px;
        border-radius: 12px;
        color: #713f12;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "pages": [],
    "chunks": [],
    "doc_index": None,
    "legal_index": None,
    "analysis": None,
    "sources": [],
    "uploaded_name": None,
    "chat_history": [],
    "text_ready": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

def configure_tesseract() -> Tuple[bool, str]:
    """
    Detect/configure Tesseract.

    Streamlit Cloud/Linux:
        packages.txt should install tesseract-ocr.

    Windows:
        common executable locations are checked automatically.
        TESSERACT_CMD can also be supplied as an environment variable.
    """
    candidates = []

    env_path = os.getenv("TESSERACT_CMD")
    if env_path:
        candidates.append(env_path)

    candidates.extend(
        [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]
    )

    existing = shutil.which("tesseract")
    if existing:
        candidates.insert(0, existing)

    for path in candidates:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                version = str(pytesseract.get_tesseract_version())
                return True, f"Tesseract detected: {version.splitlines()[0]}"
            except Exception:
                return True, f"Tesseract detected at {path}"

    return False, (
        "Tesseract OCR was not found. Install Tesseract and make sure "
        "the executable is available in PATH or set TESSERACT_CMD."
    )


TESSERACT_AVAILABLE, TESSERACT_STATUS = configure_tesseract()


# ============================================================
# API KEY / GROQ
# ============================================================

def get_secret(name: str, default: str = "") -> str:
    """Read a value from Streamlit secrets first, then environment."""
    try:
        value = st.secrets.get(name, None)
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv(name, default)


def get_groq_client() -> Groq:
    api_key = get_secret("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets or "
            "your environment variables."
        )

    return Groq(api_key=api_key)


def get_groq_model() -> str:
    return get_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL)


# ============================================================
# LEGAL KNOWLEDGE BASE
# ============================================================
#
# IMPORTANT:
# This is deliberately a SMALL MVP knowledge base.
# It contains source records and short legal-context notes that
# should be reviewed/expanded before production use.
#
# The application never invents citations. Every source shown to
# the model comes from this list.
# ============================================================

LEGAL_SOURCES = [
    {
        "source_id": "constitution-pakistan",
        "title": "Constitution of the Islamic Republic of Pakistan",
        "jurisdiction": "Pakistan",
        "section": "Constitution — official text",
        "url": "https://pakistancode.gov.pk/",
        "text": (
            "The Constitution of the Islamic Republic of Pakistan is the "
            "country's supreme constitutional framework. For legal analysis, "
            "the official Pakistan Code source should be consulted for the "
            "current authoritative text and amendments."
        ),
    },
    {
        "source_id": "pakistan-code",
        "title": "Pakistan Code",
        "jurisdiction": "Pakistan",
        "section": "Official federal legislation portal",
        "url": "https://pakistancode.gov.pk/",
        "text": (
            "Pakistan Code is the official federal legislation portal used "
            "to access federal laws and legal instruments. The current text "
            "of an Act, Ordinance, rule, or regulation should be verified "
            "against the official portal before relying on it."
        ),
    },
    {
        "source_id": "sindh-laws",
        "title": "Sindh Laws",
        "jurisdiction": "Sindh, Pakistan",
        "section": "Government of Sindh legal portal",
        "url": "https://www.sindhlaws.gov.pk/",
        "text": (
            "Sindh Laws is the Government of Sindh legal portal. Sindh-specific "
            "legislation and amendments should be checked against the current "
            "official text on this portal."
        ),
    },
    {
        "source_id": "constitution-article-10a",
        "title": "Constitution of Pakistan — Article 10A",
        "jurisdiction": "Pakistan",
        "section": "Article 10A — Right to fair trial",
        "url": "https://pakistancode.gov.pk/",
        "text": (
            "Article 10A of the Constitution provides a constitutional "
            "right to a fair trial and due process for determination of "
            "civil rights and obligations or in a criminal charge. Verify "
            "the current official constitutional text before relying on it."
        ),
    },
    {
        "source_id": "constitution-article-18",
        "title": "Constitution of Pakistan — Article 18",
        "jurisdiction": "Pakistan",
        "section": "Article 18 — Freedom of trade, business or profession",
        "url": "https://pakistancode.gov.pk/",
        "text": (
            "Article 18 concerns freedom to enter lawful professions, "
            "occupations, trades or businesses, subject to the constitutional "
            "conditions and restrictions stated in the Article. Verify the "
            "current official text and applicable law."
        ),
    },
    {
        "source_id": "constitution-article-23",
        "title": "Constitution of Pakistan — Article 23",
        "jurisdiction": "Pakistan",
        "section": "Article 23 — Provision as to property",
        "url": "https://pakistancode.gov.pk/",
        "text": (
            "Article 23 addresses the right of citizens to acquire, hold "
            "and dispose of property, subject to the Constitution and law. "
            "The current official text should be checked for legal decisions."
        ),
    },
    {
        "source_id": "constitution-article-24",
        "title": "Constitution of Pakistan — Article 24",
        "jurisdiction": "Pakistan",
        "section": "Article 24 — Protection of property rights",
        "url": "https://pakistancode.gov.pk/",
        "text": (
            "Article 24 provides constitutional protection relating to "
            "property, subject to the qualifications and conditions stated "
            "in the Constitution. Verify the current official text."
        ),
    },
]


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def ocr_image(image: Image.Image) -> str:
    """OCR a PIL image using English + Urdu where available."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "Tesseract OCR is not installed or is not available in PATH."
        )

    image = image.convert("RGB")

    # Try English + Urdu first.
    try:
        text = pytesseract.image_to_string(
            image,
            lang="eng+urd",
            config="--psm 6",
        )
        if text.strip():
            return text.strip()
    except Exception:
        pass

    # English fallback.
    try:
        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--psm 6",
        )
        return text.strip()
    except Exception as exc:
        raise RuntimeError(f"OCR failed: {exc}") from exc


def extract_image(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text from JPG/JPEG/PNG.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()

        text = ocr_image(image)

        return [
            {
                "page": 1,
                "text": text,
                "method": "OCR",
            }
        ]

    except Exception as exc:
        raise RuntimeError(f"Could not read image: {exc}") from exc


def extract_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text from both text-based and scanned PDFs.

    Text-based page:
        page.get_text("text")

    Scanned page:
        render page -> PIL image -> Tesseract OCR
    """
    pages = []

    try:
        document = fitz.open(
            stream=file_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF: {exc}") from exc

    try:
        total_pages = len(document)

        progress = st.progress(0)
        status = st.empty()

        for page_number, page in enumerate(document, start=1):
            status.write(
                f"Extracting page {page_number} of {total_pages}..."
            )

            # First attempt: selectable PDF text.
            text = page.get_text("text").strip()

            if text:
                method = "PDF text"
            else:
                # Second attempt: OCR the rendered page.
                if not TESSERACT_AVAILABLE:
                    text = ""
                    method = "OCR unavailable"
                else:
                    try:
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(2, 2),
                            alpha=False,
                        )

                        image = Image.frombytes(
                            "RGB",
                            [pix.width, pix.height],
                            pix.samples,
                        )

                        text = ocr_image(image)
                        method = "OCR"
                    except Exception as exc:
                        text = ""
                        method = f"OCR failed: {exc}"

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                    "method": method,
                }
            )

            progress.progress(page_number / max(total_pages, 1))

        progress.empty()
        status.empty()

    finally:
        document.close()

    return pages


def extract_document(
    filename: str,
    file_bytes: bytes,
) -> List[Dict[str, Any]]:
    """Route the uploaded file to the correct extraction method."""
    extension = filename.lower().rsplit(".", 1)[-1]

    if extension == "pdf":
        return extract_pdf(file_bytes)

    if extension in {"png", "jpg", "jpeg"}:
        return extract_image(file_bytes)

    raise RuntimeError(
        "Unsupported file type. Please upload PDF, PNG, JPG or JPEG."
    )


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_page_text(
    page_number: int,
    text: str,
    chunk_size: int = 1200,
    overlap: int = 180,
) -> List[Dict[str, Any]]:
    """
    Character-based chunking with overlap.
    Page numbers are preserved for citations.
    """
    text = normalize_text(text)

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Prefer breaking at a sentence/space.
        if end < text_length:
            candidates = [
                text.rfind(". ", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end),
            ]
            best = max(candidates)

            if best > start + int(chunk_size * 0.55):
                end = best + 1

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "page": page_number,
                }
            )

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_document_chunks(
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chunks = []

    for page in pages:
        page_chunks = chunk_page_text(
            page_number=int(page["page"]),
            text=page.get("text", ""),
        )
        chunks.extend(page_chunks)

    return chunks


# ============================================================
# FAISS VECTOR STORE
# ============================================================

def build_faiss_index(
    records: List[Dict[str, Any]],
    model: SentenceTransformer,
) -> Tuple[Any, np.ndarray]:
    if not records:
        raise ValueError("Cannot build a vector store from empty records.")

    texts = [record["text"] for record in records]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def search_faiss(
    index: Any,
    records: List[Dict[str, Any]],
    model: SentenceTransformer,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if index is None or not records:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(records))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(records):
            continue

        record = dict(records[idx])
        record["score"] = float(score)
        results.append(record)

    return results


# ============================================================
# LEGAL VECTOR STORE
# ============================================================

@st.cache_resource(show_spinner=False)
def build_legal_store():
    model = load_embedding_model()
    index, _ = build_faiss_index(LEGAL_SOURCES, model)
    return index


def retrieve_legal_context(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        model = load_embedding_model()
        index = build_legal_store()
        return search_faiss(
            index=index,
            records=LEGAL_SOURCES,
            model=model,
            query=query,
            top_k=top_k,
        )
    except Exception:
        return []


# ============================================================
# GROQ HELPERS
# ============================================================

def call_groq(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
) -> str:
    client = get_groq_client()

    response = client.chat.completions.create(
        model=get_groq_model(),
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("The AI returned an empty response.")

    return content.strip()


def clean_json_response(text: str) -> str:
    """
    Remove common Markdown code fences around JSON.
    """
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    # If the model included surrounding text, isolate the
    # outermost JSON object when possible.
    first = cleaned.find("{")
    last = cleaned.rfind("}")

    if first != -1 and last != -1 and last > first:
        cleaned = cleaned[first:last + 1]

    return cleaned.strip()


def parse_ai_json(text: str) -> Dict[str, Any]:
    cleaned = clean_json_response(text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The AI returned invalid JSON. Please try the analysis again."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("The AI response was not a JSON object.")

    return data


# ============================================================
# PROMPTS
# ============================================================

LEGAL_SYSTEM_PROMPT = """
You are a careful legal-document analysis assistant for a Pakistan-focused
hackathon application.

Your task is INFORMATIONAL ANALYSIS, not legal representation.

STRICT EVIDENCE RULES:
1. Use only the uploaded-document evidence and the supplied legal-source
   context.
2. Never invent a law, section, article, case, court decision, citation,
   legal requirement, or URL.
3. If the supplied legal context is insufficient, explicitly say that it
   is insufficient.
4. Do not call a clause "illegal" unless the supplied evidence actually
   establishes that conclusion.
5. Prefer cautious wording such as "potentially risky", "may be
   unfavorable", "requires legal review", or "appears relevant".
6. Separate:
   - Document Evidence
   - Legal Context
   - AI Assessment
7. Page numbers must come from the supplied document chunks.
8. Never fabricate a quotation from the document.
9. Do not claim that a clause is legally required unless the supplied
   legal-source context supports that conclusion.
10. The legal-source database in this MVP is limited. Make that limitation
    clear when appropriate.

Return valid JSON only.
"""


def build_analysis_prompt(
    document_chunks: List[Dict[str, Any]],
    legal_results: List[Dict[str, Any]],
    language: str,
) -> str:
    # Limit context so requests remain manageable.
    selected_doc = document_chunks[:30]

    document_context = "\n\n".join(
        [
            f"[DOCUMENT PAGE {item['page']}]\n{item['text']}"
            for item in selected_doc
        ]
    )

    legal_context = "\n\n".join(
        [
            (
                f"[LEGAL SOURCE {item['source_id']}]\n"
                f"Title: {item['title']}\n"
                f"Section: {item.get('section', '')}\n"
                f"Jurisdiction: {item.get('jurisdiction', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Context: {item['text']}"
            )
            for item in legal_results
        ]
    )

    if not legal_context:
        legal_context = "No sufficiently relevant legal-source context was retrieved."

    return f"""
Analyze the following uploaded Pakistani legal document.

RESPONSE LANGUAGE:
{language}

Return EXACTLY this JSON structure:

{{
  "summary": "string",
  "document_type": "string",
  "risks": [
    {{
      "title": "string",
      "severity": "HIGH|MEDIUM|LOW",
      "why_it_matters": "string",
      "document_evidence": "string",
      "page": 0,
      "legal_context": "string",
      "source_ids": []
    }}
  ],
  "rights": [
    {{
      "title": "string",
      "explanation": "string",
      "document_evidence": "string",
      "legal_context": "string",
      "source_ids": []
    }}
  ],
  "missing_clauses": [
    {{
      "title": "string",
      "reason": "string",
      "status": "RECOMMENDED|POTENTIALLY_REQUIRED",
      "source_ids": []
    }}
  ],
  "important_clauses": [
    {{
      "title": "string",
      "explanation": "string",
      "document_evidence": "string",
      "page": 0
    }}
  ],
  "sources": [
    {{
      "source_id": "string",
      "title": "string",
      "section": "string",
      "url": "string"
    }}
  ]
}}

Rules:
- Use page 0 when a page number genuinely cannot be established.
- Risks should be limited to meaningful issues, not trivial wording.
- If no reliable legal context supports a point, say so rather than inventing one.
- source_ids must be chosen ONLY from the legal sources supplied below.
- The "sources" array must ONLY contain sources supplied below.
- Do not invent source IDs or URLs.
- Do not quote document text unless it is actually present in the document context.
- Keep the summary understandable to a non-lawyer.

================ DOCUMENT CONTEXT ================

{document_context}

================ LEGAL CONTEXT ================

{legal_context}
"""


# ============================================================
# SOURCE VALIDATION
# ============================================================

def validate_sources(data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        source["source_id"]: source
        for source in LEGAL_SOURCES
    }

    def valid_source_ids(items: Any) -> List[str]:
        if not isinstance(items, list):
            return []

        return [
            str(source_id)
            for source_id in items
            if str(source_id) in allowed
        ]

    for risk in data.get("risks", []) or []:
        risk["source_ids"] = valid_source_ids(
            risk.get("source_ids", [])
        )

    for right in data.get("rights", []) or []:
        right["source_ids"] = valid_source_ids(
            right.get("source_ids", [])
        )

    for missing in data.get("missing_clauses", []) or []:
        missing["source_ids"] = valid_source_ids(
            missing.get("source_ids", [])
        )

    valid_sources = []

    for source in data.get("sources", []) or []:
        source_id = str(source.get("source_id", ""))

        if source_id in allowed:
            original = allowed[source_id]
            valid_sources.append(
                {
                    "source_id": source_id,
                    "title": original["title"],
                    "section": original.get("section", ""),
                    "url": original["url"],
                }
            )

    data["sources"] = valid_sources

    return data


# ============================================================
# DOCUMENT ANALYSIS
# ============================================================

def analyze_document(
    pages: List[Dict[str, Any]],
    language: str,
) -> Dict[str, Any]:
    chunks = build_document_chunks(pages)

    if not chunks:
        raise RuntimeError(
            "No readable text was extracted from this document."
        )

    model = load_embedding_model()

    with st.spinner("Building document vector index..."):
        doc_index, _ = build_faiss_index(chunks, model)

    # Build a representative query from the first chunks.
    representative_text = " ".join(
        chunk["text"]
        for chunk in chunks[:8]
    )

    representative_text = representative_text[:5000]

    with st.spinner("Retrieving relevant Pakistani legal context..."):
        legal_results = retrieve_legal_context(
            representative_text,
            top_k=6,
        )

    with st.spinner("AI is analyzing the document..."):
        prompt = build_analysis_prompt(
            document_chunks=chunks,
            legal_results=legal_results,
            language=language,
        )

        raw = call_groq(
            system_prompt=LEGAL_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.1,
        )

    data = parse_ai_json(raw)
    data = validate_sources(data)

    st.session_state["chunks"] = chunks
    st.session_state["doc_index"] = doc_index
    st.session_state["analysis"] = data
    st.session_state["sources"] = legal_results
    st.session_state["text_ready"] = True

    return data


# ============================================================
# ASK AI
# ============================================================

def answer_document_question(
    question: str,
    language: str,
) -> str:
    chunks = st.session_state.get("chunks", [])
    doc_index = st.session_state.get("doc_index")

    if not chunks or doc_index is None:
        raise RuntimeError(
            "Please analyze a document before asking questions."
        )

    model = load_embedding_model()

    document_results = search_faiss(
        index=doc_index,
        records=chunks,
        model=model,
        query=question,
        top_k=6,
    )

    legal_results = retrieve_legal_context(
        question,
        top_k=5,
    )

    document_context = "\n\n".join(
        [
            f"[DOCUMENT PAGE {item['page']}]\n{item['text']}"
            for item in document_results
        ]
    )

    legal_context = "\n\n".join(
        [
            (
                f"[LEGAL SOURCE {item['source_id']}]\n"
                f"{item['title']}\n"
                f"{item['section']}\n"
                f"{item['text']}\n"
                f"URL: {item['url']}"
            )
            for item in legal_results
        ]
    )

    if not legal_context:
        legal_context = "No sufficiently relevant legal source was retrieved."

    system_prompt = f"""
You are an informational legal-document assistant focused on Pakistan.

Answer in {language}.

Rules:
- Answer using ONLY the supplied document context and legal context.
- Never invent laws, sections, cases, URLs, or legal requirements.
- If the evidence is insufficient, say so.
- Distinguish what the document says from legal context.
- Give page numbers only when provided.
- Be concise but useful.
- This is not a substitute for a qualified lawyer.
"""

    user_prompt = f"""
USER QUESTION:
{question}

DOCUMENT CONTEXT:
{document_context}

LEGAL CONTEXT:
{legal_context}

Answer the question directly.
"""

    return call_groq(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def safe_list(value: Any) -> List:
    return value if isinstance(value, list) else []


def render_sources(source_ids: List[str]) -> None:
    if not source_ids:
        return

    source_map = {
        source["source_id"]: source
        for source in LEGAL_SOURCES
    }

    for source_id in source_ids:
        source = source_map.get(source_id)

        if source:
            st.markdown(
                f"**{source['title']}** — "
                f"{source.get('section', '')}"
            )
            st.markdown(
                f"[Official source]({source['url']})"
            )


def render_risks(risks: List[Dict[str, Any]]) -> None:
    if not risks:
        st.info("No specific risks were identified from the available evidence.")
        return

    for risk in risks:
        severity = str(
            risk.get("severity", "MEDIUM")
        ).upper()

        if severity not in {"HIGH", "MEDIUM", "LOW"}:
            severity = "MEDIUM"

        css_class = f"risk-{severity.lower()}"

        st.markdown(
            f"""
            <div class="risk-card {css_class}">
                <strong>{severity} RISK</strong>
                <h4>{risk.get("title", "Unnamed risk")}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"**Why it matters:** "
            f"{risk.get('why_it_matters', 'Not provided.')}"
        )

        evidence = risk.get("document_evidence", "")
        if evidence:
            st.markdown(
                f"**Document evidence:** {evidence}"
            )

        page = risk.get("page", 0)
        if page:
            st.markdown(f"**Page:** {page}")

        legal_context = risk.get("legal_context", "")
        if legal_context:
            st.markdown(
                f"**Legal context:** {legal_context}"
            )

        source_ids = risk.get("source_ids", [])
        if source_ids:
            st.markdown("**Supporting sources:**")
            render_sources(source_ids)

        st.divider()


def render_rights(rights: List[Dict[str, Any]]) -> None:
    if not rights:
        st.info(
            "No specific rights were identified from the available evidence."
        )
        return

    for right in rights:
        with st.expander(
            right.get("title", "Potentially relevant right"),
            expanded=False,
        ):
            st.write(
                right.get(
                    "explanation",
                    "No explanation provided.",
                )
            )

            evidence = right.get("document_evidence", "")
            if evidence:
                st.markdown(
                    f"**Document evidence:** {evidence}"
                )

            legal_context = right.get("legal_context", "")
            if legal_context:
                st.markdown(
                    f"**Legal context:** {legal_context}"
                )

            source_ids = right.get("source_ids", [])
            if source_ids:
                st.markdown("**Sources:**")
                render_sources(source_ids)


def render_missing_clauses(
    clauses: List[Dict[str, Any]],
) -> None:
    if not clauses:
        st.info("No potentially important missing clauses were identified.")
        return

    for clause in clauses:
        status = clause.get(
            "status",
            "RECOMMENDED",
        )

        with st.expander(
            f"{clause.get('title', 'Missing clause')} — {status}"
        ):
            st.write(
                clause.get(
                    "reason",
                    "No reason provided.",
                )
            )

            if status == "POTENTIALLY_REQUIRED":
                st.warning(
                    "The AI marked this as potentially required. "
                    "Verify the applicable law before relying on this conclusion."
                )

            source_ids = clause.get("source_ids", [])
            if source_ids:
                render_sources(source_ids)


def render_important_clauses(
    clauses: List[Dict[str, Any]],
) -> None:
    if not clauses:
        st.info("No important clauses were identified.")
        return

    for clause in clauses:
        with st.expander(
            clause.get("title", "Important clause")
        ):
            st.write(
                clause.get(
                    "explanation",
                    "No explanation provided.",
                )
            )

            evidence = clause.get("document_evidence", "")
            if evidence:
                st.markdown(
                    f"**Document evidence:** {evidence}"
                )

            page = clause.get("page", 0)
            if page:
                st.markdown(f"**Page:** {page}")


def render_extracted_text(
    pages: List[Dict[str, Any]],
) -> None:
    for page in pages:
        text = page.get("text", "").strip()

        if not text:
            continue

        method = page.get("method", "Unknown")

        with st.expander(
            f"Page {page['page']} — {method}"
        ):
            st.text(text)


# ============================================================
# SIDEBAR
# ============================================================

# ============================================================
# DOCUMIND UI
# ============================================================

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown("""
<style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #030a16;
        color: #f8fafc;
    }

    .block-container {
        padding: 1.5rem 2rem 3rem 2rem !important;
        max-width: 1500px !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #040d1a;
        border-right: 1px solid #111e38;
    }

    section[data-testid="stSidebar"] > div {
        padding: 25px 18px;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }

    .sidebar-logo-icon {
        width: 45px;
        height: 45px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 14px;

        background:
            linear-gradient(
                135deg,
                #6366f1 0%,
                #3b82f6 100%
            );

        font-size: 22px;

        box-shadow:
            0 0 18px rgba(99,102,241,0.35);
    }

    .sidebar-logo-text {
        color: white;
        font-size: 20px;
        font-weight: 800;
        line-height: 1.1;
    }

    .sidebar-logo-text span {
        display: block;
        color: #60a5fa;
        font-size: 14px;
        margin-top: 2px;
    }

    .sidebar-tagline {
        color: #64748b;
        font-size: 12px;
        line-height: 1.5;
        margin: 8px 0 25px 2px;
    }

    .sidebar-heading {
        color: #475569;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 25px 0 10px 2px;
    }

    .sidebar-item {
        color: #94a3b8;
        font-size: 13px;
        padding: 10px 12px;
        border-radius: 9px;
        margin-bottom: 4px;
    }

    .sidebar-item.active {
        background:
            linear-gradient(
                90deg,
                #1d3557 0%,
                #0c192e 100%
            );

        color: #38bdf8;

        border-left: 3px solid #38bdf8;
    }

    .sidebar-status {
        background: #06152d;
        border: 1px solid #162a4a;
        border-radius: 12px;
        padding: 13px;
        margin-top: 18px;
    }

    .sidebar-status-title {
        color: #38bdf8;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sidebar-status-text {
        color: #64748b;
        font-size: 11px;
        line-height: 1.5;
    }


    /* ========================================================
       TOP HEADER
       ======================================================== */

    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }

    .top-brand {
        color: #64748b;
        font-size: 12px;
    }

    .top-user {
        color: #cbd5e1;
        font-size: 13px;
        font-weight: 500;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        background:
            linear-gradient(
                135deg,
                #06152d 0%,
                #040e1e 100%
            );

        border: 1px solid #162a4a;

        border-radius: 20px;

        padding: 30px 32px;

        display: flex;
        justify-content: space-between;
        align-items: center;

        min-height: 210px;

        overflow: hidden;

        box-shadow:
            0 12px 35px rgba(0,0,0,0.35);

        margin-bottom: 20px;
    }

    .hero-content {
        max-width: 720px;
    }

    .hero-title-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 16px;
    }

    .hero-icon {
        width: 50px;
        height: 50px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 14px;

        background:
            rgba(56,189,248,0.1);

        border:
            1px solid rgba(56,189,248,0.3);

        font-size: 25px;

        box-shadow:
            0 0 18px rgba(56,189,248,0.15);
    }

    .hero-title {
        margin: 0;
        color: white;
        font-size: 31px;
        font-weight: 800;
        line-height: 1;
    }

    .hero-title span {
        color: #60a5fa;
    }

    .hero-subtitle {
        margin: 0 0 8px 0;
        color: white;
        font-size: 16px;
        font-weight: 700;
    }

    .hero-description {
        margin: 0;
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.6;
    }

    .hero-document {
        width: 105px;
        height: 120px;

        position: relative;

        background:
            linear-gradient(
                180deg,
                #93c5fd 0%,
                #60a5fa 100%
            );

        border-radius: 13px;

        padding: 14px;

        box-shadow:
            0 10px 30px rgba(96,165,250,0.25);
    }

    .hero-document-line {
        height: 4px;
        background: white;
        border-radius: 3px;
        opacity: 0.8;
        margin-bottom: 10px;
    }

    .hero-alert {
        position: absolute;

        width: 46px;
        height: 46px;

        right: -16px;
        bottom: -12px;

        border-radius: 50%;

        display: flex;
        align-items: center;
        justify-content: center;

        background: #06152d;

        border: 3px solid #38bdf8;

        font-size: 20px;
    }


    /* ========================================================
       FEATURE CARDS
       ======================================================== */

    .feature-card {
        background: #06152d;
        border: 1px solid #162a4a;
        border-radius: 16px;
        padding: 19px 17px;
        min-height: 138px;
    }

    .feature-icon {
        width: 38px;
        height: 38px;

        border-radius: 50%;

        background:
            rgba(56,189,248,0.1);

        border:
            1px solid rgba(56,189,248,0.2);

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 17px;

        margin-bottom: 13px;
    }

    .feature-card h4 {
        color: white;
        font-size: 14px;
        margin: 0 0 6px 0;
        font-weight: 700;
    }

    .feature-card p {
        color: #64748b;
        font-size: 11px;
        line-height: 1.5;
        margin: 0;
    }


    /* ========================================================
       MAIN CARDS
       ======================================================== */

    .main-card {
        background: #06152d;
        border: 1px solid #162a4a;
        border-radius: 18px;
        padding: 24px;
        margin-top: 20px;
    }

    .section-heading {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 18px;
    }

    .section-heading-icon {
        width: 34px;
        height: 34px;

        border-radius: 9px;

        display: flex;
        align-items: center;
        justify-content: center;

        background:
            rgba(56,189,248,0.1);

        color: #38bdf8;

        font-size: 16px;
    }

    .section-heading h3 {
        color: white;
        font-size: 16px;
        margin: 0;
    }

    .section-heading p {
        color: #64748b;
        font-size: 11px;
        margin: 2px 0 0 0;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        width: 100%;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #030c1c !important;

        border:
            2px dashed #1d4ed8 !important;

        border-radius: 14px !important;

        min-height: 170px;

        transition: 0.2s;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #38bdf8 !important;
        background: rgba(56,189,248,0.02) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #94a3b8 !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 10px;

        background: #2563eb;

        border: 1px solid #2563eb;

        color: white;

        font-size: 13px;

        font-weight: 600;

        padding: 10px 16px;

        transition: 0.2s;

        box-shadow:
            0 5px 18px rgba(37,99,235,0.18);
    }

    .stButton > button:hover {
        background: #3b82f6;
        border-color: #3b82f6;
        color: white;
        transform: translateY(-1px);
    }


    /* ========================================================
       TIP
       ======================================================== */

    .tip {
        background: #041226;
        border: 1px solid #1d4ed8;
        border-radius: 10px;
        padding: 12px 15px;
        color: #94a3b8;
        font-size: 11px;
        line-height: 1.5;
        margin-top: 15px;
    }

    .tip strong {
        color: #38bdf8;
    }


    /* ========================================================
       RIGHT SIDE
       ======================================================== */

    .side-card {
        background: #06152d;
        border: 1px solid #162a4a;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .side-title {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 17px;
    }

    .side-title-icon {
        color: #38bdf8;
        font-size: 18px;
    }

    .side-title h3 {
        color: white;
        font-size: 15px;
        margin: 0;
    }

    .step {
        display: flex;
        align-items: center;
        gap: 11px;
        color: #cbd5e1;
        font-size: 12px;
        margin-bottom: 14px;
    }

    .step:last-child {
        margin-bottom: 0;
    }

    .step-number {
        width: 24px;
        height: 24px;

        border-radius: 50%;

        background: #2563eb;

        color: white;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 11px;
        font-weight: 700;

        flex-shrink: 0;
    }

    .side-description {
        color: #64748b;
        font-size: 11px;
        line-height: 1.5;
        margin-bottom: 15px;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: #06152d;
        border: 1px solid #162a4a;
        border-radius: 13px;
        padding: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    div[data-testid="stMetricValue"] {
        color: white !important;
    }


    /* ========================================================
       TABS
       ======================================================== */

    button[data-baseweb="tab"] {
        color: #64748b !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
    }

    div[data-baseweb="tab-highlight"] {
        background: #38bdf8 !important;
    }


    /* ========================================================
       DISCLAIMER
       ======================================================== */

    .disclaimer {
        background: #071426;
        border: 1px solid #162a4a;
        border-radius: 11px;

        padding: 12px 15px;

        color: #64748b;

        font-size: 11px;

        line-height: 1.5;

        margin-bottom: 20px;
    }

    .disclaimer strong {
        color: #fbbf24;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 900px) {

        .hero-document {
            display: none;
        }

        .hero-title {
            font-size: 25px;
        }
    }

    @media (max-width: 650px) {

        .block-container {
            padding: 1rem !important;
        }

        .hero {
            padding: 22px;
        }

        .hero-title {
            font-size: 22px;
        }

        .hero-subtitle {
            font-size: 14px;
        }

    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">

        <div class="sidebar-logo-icon">
            🧠
        </div>

        <div class="sidebar-logo-text">
            DocuMind
            <span>Legal AI</span>
        </div>

    </div>

    <div class="sidebar-tagline">
        Pakistan Legal & Contract<br>
        Risk Analyzer
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div class="sidebar-heading">
        Navigation
    </div>

    <div class="sidebar-item active">
        🏠 &nbsp; Home
    </div>

    <div class="sidebar-item">
        📄 &nbsp; Upload Document
    </div>

    <div class="sidebar-item">
        🔍 &nbsp; Scan & Analyze
    </div>

    <div class="sidebar-item">
        ⚠️ &nbsp; Risks & Rights
    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Existing language selector
    # --------------------------------------------------------

    st.markdown("""
    <div class="sidebar-heading">
        AI Settings
    </div>
    """, unsafe_allow_html=True)

    language = st.selectbox(
        "Response language",
        ["English", "Urdu"],
        index=0,
        label_visibility="collapsed",
    )


    # --------------------------------------------------------
    # System status
    # --------------------------------------------------------

    st.markdown("""
    <div class="sidebar-heading">
        System Status
    </div>
    """, unsafe_allow_html=True)

    if TESSERACT_AVAILABLE:
        st.success("🟢 OCR: Ready")
    else:
        st.error("🔴 OCR: Not available")
        st.caption(TESSERACT_STATUS)

    if get_secret("GROQ_API_KEY"):
        st.success("🟢 Groq API: Connected")
    else:
        st.warning("🟡 Groq API: Missing")


    st.markdown("""
    <div class="sidebar-status">

        <div class="sidebar-status-title">
            🛡️ Privacy First
        </div>

        <div class="sidebar-status-text">
            Documents are processed for AI-assisted
            analysis. Do not upload documents containing
            information you are not authorized to process.
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TOP HEADER
# ============================================================

st.markdown("""
<div class="top-header">

    <div class="top-brand">
        Pakistan Legal AI • Hackathon MVP
    </div>

    <div class="top-user">
        🛡️ AI Legal Assistant
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-content">

        <div class="hero-title-row">

            <div class="hero-icon">
                🛡️
            </div>

            <h1 class="hero-title">
                DocuMind
                <span>Legal AI</span>
            </h1>

        </div>

        <h3 class="hero-subtitle">
            Find hidden risks in your Pakistani legal documents.
        </h3>

        <p class="hero-description">
            Upload a legal document and let AI analyze potentially
            risky clauses, important rights, missing provisions and
            relevant Pakistani legal sources — in simple language.
        </p>

    </div>


    <div class="hero-document">

        <div class="hero-document-line"
             style="width:70%;">
        </div>

        <div class="hero-document-line"
             style="width:100%;">
        </div>

        <div class="hero-document-line"
             style="width:90%;">
        </div>

        <div class="hero-document-line"
             style="width:60%;">
        </div>

        <div class="hero-alert">
            ⚠️
        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN GRID
# ============================================================

left_column, right_column = st.columns(
    [3.2, 1.15],
    gap="large",
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_column:

    # --------------------------------------------------------
    # FEATURE CARDS
    # --------------------------------------------------------

    feature1, feature2, feature3, feature4 = st.columns(4)

    with feature1:
        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                🛡️
            </div>

            <h4>
                Find Hidden Clauses
            </h4>

            <p>
                Detect potentially unfair or risky
                language in your document.
            </p>

        </div>
        """, unsafe_allow_html=True)


    with feature2:
        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                📄
            </div>

            <h4>
                Check Compliance
            </h4>

            <p>
                Compare relevant provisions with
                Pakistani legal sources.
            </p>

        </div>
        """, unsafe_allow_html=True)


    with feature3:
        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                ⚠️
            </div>

            <h4>
                Assess Legal Risks
            </h4>

            <p>
                Identify potential liabilities and
                problematic provisions.
            </p>

        </div>
        """, unsafe_allow_html=True)


    with feature4:
        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                💡
            </div>

            <h4>
                Get Simple Insights
            </h4>

            <p>
                Understand complex legal language
                more easily.
            </p>

        </div>
        """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # UPLOAD CARD
    # --------------------------------------------------------

    st.markdown("""
    <div class="main-card">

        <div class="section-heading">

            <div class="section-heading-icon">
                📄
            </div>

            <div>

                <h3>
                    Upload Your Legal Document
                </h3>

                <p>
                    Choose a PDF or image and click Analyze.
                </p>

            </div>

        </div>

    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # REAL STREAMLIT FILE UPLOADER
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Choose a Pakistani legal document",
        type=SUPPORTED_EXTENSIONS,
        help="PDF, JPG, JPEG and PNG are supported.",
        label_visibility="collapsed",
    )


    st.markdown("""
        <div class="tip">

            <strong>💡 Tip:</strong>
            Scanned PDFs and images require OCR.
            Make sure the OCR status in the sidebar shows
            <strong>Ready</strong> before analyzing scanned documents.

        </div>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # FILE INFORMATION
    # --------------------------------------------------------

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        file_col1, file_col2, file_col3 = st.columns(3)

        with file_col1:
            st.metric(
                "File",
                uploaded_file.name,
            )

        with file_col2:
            st.metric(
                "Size",
                f"{len(file_bytes) / 1024:.1f} KB",
            )

        with file_col3:

            extension = (
                uploaded_file.name
                .rsplit(".", 1)[-1]
                .upper()
            )

            st.metric(
                "Type",
                extension,
            )


        # ----------------------------------------------------
        # ANALYZE BUTTON
        # ----------------------------------------------------

        analyze_clicked = st.button(
            "🔍  Extract & Analyze Document",
            type="primary",
            use_container_width=True,
        )


        if analyze_clicked:

            # Clear old analysis
            st.session_state["pages"] = []
            st.session_state["chunks"] = []
            st.session_state["doc_index"] = None
            st.session_state["legal_index"] = None
            st.session_state["analysis"] = None
            st.session_state["sources"] = []
            st.session_state["chat_history"] = []
            st.session_state["uploaded_name"] = uploaded_file.name
            st.session_state["text_ready"] = False

            try:

                # ------------------------------------------------
                # EXTRACTION
                # ------------------------------------------------

                with st.spinner(
                    "📄 Extracting document and running OCR if needed..."
                ):

                    pages = extract_document(
                        uploaded_file.name,
                        file_bytes,
                    )


                total_text = sum(
                    len(page.get("text", ""))
                    for page in pages
                )


                if not pages or total_text == 0:

                    st.error(
                        "No readable text was found."
                    )

                    if not TESSERACT_AVAILABLE:

                        st.info(
                            "OCR is currently unavailable. "
                            "Install/configure Tesseract before "
                            "processing scanned documents."
                        )

                    st.stop()


                st.session_state["pages"] = pages
                st.session_state["text_ready"] = True


                st.success(
                    f"Extraction complete: "
                    f"{len(pages)} page(s), "
                    f"{total_text:,} characters."
                )


                # ------------------------------------------------
                # AI ANALYSIS
                # ------------------------------------------------

                with st.spinner(
                    "🤖 AI is analyzing the document..."
                ):

                    analysis = analyze_document(
                        pages=pages,
                        language=language,
                    )


                st.success(
                    "✅ Analysis completed successfully."
                )


            except Exception as exc:

                st.error(
                    f"Analysis failed: {exc}"
                )

                with st.expander(
                    "Technical details"
                ):
                    st.exception(exc)


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_column:

    # --------------------------------------------------------
    # QUICK START
    # --------------------------------------------------------

    st.markdown("""
    <div class="side-card">

        <div class="side-title">

            <div class="side-title-icon">
                🚀
            </div>

            <h3>
                Quick Start
            </h3>

        </div>

        <div class="step">

            <div class="step-number">
                1
            </div>

            <span>
                Upload a legal document
            </span>

        </div>

        <div class="step">

            <div class="step-number">
                2
            </div>

            <span>
                Click Analyze
            </span>

        </div>

        <div class="step">

            <div class="step-number">
                3
            </div>

            <span>
                View risks and insights
            </span>

        </div>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # SUPPORTED FILES
    # --------------------------------------------------------

    st.markdown("""
    <div class="side-card">

        <div class="side-title">

            <div class="side-title-icon">
                📁
            </div>

            <h3>
                Supported Documents
            </h3>

        </div>

        <p class="side-description">
            Upload contracts, agreements and other
            Pakistani legal documents as PDF or image files.
        </p>

        <p style="
            color:#38bdf8;
            font-size:12px;
            margin:0;
        ">
            PDF • JPG • JPEG • PNG
        </p>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # AI DISCLAIMER
    # --------------------------------------------------------

    st.markdown("""
    <div class="side-card"
         style="text-align:center;">

        <div style="
            font-size:40px;
            margin-bottom:8px;
        ">
            🛡️
        </div>

        <h3 style="
            color:white;
            font-size:16px;
            margin:0 0 7px 0;
        ">
            AI Legal Assistant
        </h3>

        <p class="side-description"
           style="margin-bottom:0;">

            This application provides AI-assisted
            informational analysis and does not replace
            advice from a qualified lawyer.

        </p>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RESULTS
# ============================================================

analysis = st.session_state.get(
    "analysis"
)

pages = st.session_state.get(
    "pages",
    []
)


if analysis:

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="main-card">

        <div class="section-heading">

            <div class="section-heading-icon">
                📊
            </div>

            <div>

                <h3>
                    Analysis Results
                </h3>

                <p>
                    AI-assisted analysis of your uploaded document
                </p>

            </div>

        </div>

    """, unsafe_allow_html=True)


    risks = safe_list(
        analysis.get("risks")
    )

    rights = safe_list(
        analysis.get("rights")
    )

    missing = safe_list(
        analysis.get("missing_clauses")
    )

    important = safe_list(
        analysis.get("important_clauses")
    )


    high_risks = sum(
        1
        for risk in risks
        if str(
            risk.get(
                "severity",
                "",
            )
        ).upper() == "HIGH"
    )


    medium_risks = sum(
        1
        for risk in risks
        if str(
            risk.get(
                "severity",
                "",
            )
        ).upper() == "MEDIUM"
    )


    low_risks = sum(
        1
        for risk in risks
        if str(
            risk.get(
                "severity",
                "",
            )
        ).upper() == "LOW"
    )


    # --------------------------------------------------------
    # RESULT METRICS
    # --------------------------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "🔴 High Risks",
            high_risks,
        )

    with metric2:
        st.metric(
            "🟠 Medium Risks",
            medium_risks,
        )

    with metric3:
        st.metric(
            "🟢 Low Risks",
            low_risks,
        )

    with metric4:
        st.metric(
            "📄 Pages",
            len(pages),
        )


    # --------------------------------------------------------
    # RESULT TABS
    # --------------------------------------------------------

    tabs = st.tabs(
        [
            "📝 Summary",
            "🚨 Risks",
            "🛡️ Rights",
            "📌 Missing",
            "⭐ Important",
            "🔗 Sources",
            "💬 Ask AI",
            "📄 Extracted Text",
        ]
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    with tabs[0]:

        st.subheader(
            analysis.get(
                "document_type",
                "Legal Document",
            )
        )

        st.write(
            analysis.get(
                "summary",
                "No summary was returned.",
            )
        )


    # ========================================================
    # RISKS
    # ========================================================

    with tabs[1]:

        render_risks(
            risks
        )


    # ========================================================
    # RIGHTS
    # ========================================================

    with tabs[2]:

        render_rights(
            rights
        )


    # ========================================================
    # MISSING
    # ========================================================

    with tabs[3]:

        render_missing_clauses(
            missing
        )


    # ========================================================
    # IMPORTANT
    # ========================================================

    with tabs[4]:

        render_important_clauses(
            important
        )


    # ========================================================
    # SOURCES
    # ========================================================

    with tabs[5]:

        retrieved_sources = (
            st.session_state.get(
                "sources",
                [],
            )
        )


        if not retrieved_sources:

            st.info(
                "No sufficiently relevant legal source "
                "was retrieved."
            )

        else:

            for source in retrieved_sources:

                st.markdown(
                    f"### {source['title']}"
                )

                st.markdown(
                    f"**Section:** "
                    f"{source.get('section', '')}"
                )

                st.markdown(
                    f"**Jurisdiction:** "
                    f"{source.get('jurisdiction', '')}"
                )

                st.markdown(
                    f"[Open official source]"
                    f"({source['url']})"
                )

                st.write(
                    source.get(
                        "text",
                        "",
                    )
                )

                st.divider()


        st.caption(
            "The current MVP legal knowledge base is limited. "
            "Expand and professionally review the legal corpus "
            "before using this application for real legal decisions."
        )


    # ========================================================
    # ASK AI
    # ========================================================

    with tabs[6]:

        st.subheader(
            "Ask questions about your document"
        )


        for message in st.session_state.get(
            "chat_history",
            [],
        ):

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )


        question = st.chat_input(
            "Ask something about the uploaded document..."
        )


        if question:

            st.session_state[
                "chat_history"
            ].append(
                {
                    "role": "user",
                    "content": question,
                }
            )


            with st.chat_message("user"):

                st.markdown(
                    question
                )


            try:

                with st.chat_message(
                    "assistant"
                ):

                    with st.spinner(
                        "Searching document and legal context..."
                    ):

                        answer = answer_document_question(
                            question=question,
                            language=language,
                        )


                    st.markdown(
                        answer
                    )


                st.session_state[
                    "chat_history"
                ].append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


            except Exception as exc:

                error_message = (
                    f"Could not answer the question: {exc}"
                )


                with st.chat_message(
                    "assistant"
                ):

                    st.error(
                        error_message
                    )


                st.session_state[
                    "chat_history"
                ].append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


    # ========================================================
    # EXTRACTED TEXT
    # ========================================================

    with tabs[7]:

        st.subheader(
            "Extracted document text"
        )


        total_chars = sum(
            len(
                page.get(
                    "text",
                    "",
                )
            )
            for page in pages
        )


        st.caption(
            f"{len(pages)} page(s) • "
            f"{total_chars:,} characters"
        )


        render_extracted_text(
            pages
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="
    border-top:1px solid #111e38;
    margin-top:35px;
    padding-top:18px;
    text-align:center;
    color:#475569;
    font-size:11px;
">
    🛡️ DocuMind Legal AI
    • Pakistan Legal Risk & Rights Analyzer
    • Hackathon MVP
    • Informational use only
</div>
""", unsafe_allow_html=True)
