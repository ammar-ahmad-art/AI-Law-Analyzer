# ============================================================
# Pakistan-Focused AI Legal Risk & Rights Analyzer
# app.py
# ============================================================

import os
import io
import re
import json
import zipfile
import hashlib
import html
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss
import pytesseract

from PIL import Image
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuMind — Pakistan Legal Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CONFIG
# ============================================================

APP_NAME = "DocuMind"
APP_SUBTITLE = "Pakistan-Focused AI Legal Risk & Rights Analyzer"

# IMPORTANT:
# Put your REAL Groq API key in Streamlit Secrets:
#
# .streamlit/secrets.toml
#
# GROQ_API_KEY = "your-real-groq-api-key"
#
# Or configure GROQ_API_KEY as an environment variable.

GROQ_API_KEY = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY", "")
)

# Current GPT-OSS model available through Groq.
# Can be overridden through Streamlit secrets/environment.
GROQ_MODEL = st.secrets.get(
    "GROQ_MODEL",
    os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
)

# Multilingual model is useful because the app supports Urdu.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

MAX_FILE_SIZE_MB = 20
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K_DOCUMENT = 6
TOP_K_LEGAL = 5


# ============================================================
# OFFICIAL PAKISTANI LEGAL KNOWLEDGE BASE
# ============================================================
#
# IMPORTANT DESIGN PRINCIPLE:
#
# The model is NOT allowed to invent legal citations.
#
# Every legal reference that can appear in the application must
# correspond to one of these curated entries.
#
# For production deployment, expand this KB using official
# Gazette / Pakistan Code documents and have every entry reviewed.
#
# Official sources currently used:
# - Pakistan Code
# - Wafaqi Mohtasib
#
# The legal KB contains concise verified summaries, not a claim
# that this is a complete legal database.
# ============================================================

LEGAL_KB = [

    # --------------------------------------------------------
    # CONTRACT ACT
    # --------------------------------------------------------

    {
        "id": "KB-CA-10",
        "title": "Contract Act, 1872 — Section 10",
        "law": "Contract Act, 1872",
        "jurisdiction": "Pakistan",
        "text": (
            "Section 10 concerns when agreements become contracts. "
            "It identifies free consent, competent parties, lawful "
            "consideration and lawful object as important requirements, "
            "subject to the other provisions of the Act."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/english/"
            "UY2FqaJw2-apaUY2Fqa-a50%3D-con-128-sg-jjjjjjjjjjjjj"
        ),
    },

    {
        "id": "KB-CA-15",
        "title": "Contract Act, 1872 — Section 15",
        "law": "Contract Act, 1872",
        "jurisdiction": "Pakistan",
        "text": (
            "Section 15 deals with coercion. The provision is relevant "
            "when assessing whether consent to an agreement may have "
            "been obtained through conduct amounting to coercion."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/english/"
            "UY2FqaJw2-apaUY2Fqa-a50%3D-con-128-sg-jjjjjjjjjjjjj"
        ),
    },

    {
        "id": "KB-CA-16",
        "title": "Contract Act, 1872 — Section 16",
        "law": "Contract Act, 1872",
        "jurisdiction": "Pakistan",
        "text": (
            "Section 16 addresses undue influence. It is relevant when "
            "one party is in a position to dominate the will of another "
            "and the transaction may involve the use of that position."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/english/"
            "UY2FqaJw2-apaUY2Fqa-a50%3D-con-128-sg-jjjjjjjjjjjjj"
        ),
    },

    {
        "id": "KB-CA-17",
        "title": "Contract Act, 1872 — Section 17",
        "law": "Contract Act, 1872",
        "jurisdiction": "Pakistan",
        "text": (
            "Section 17 concerns fraud. It is relevant when a contract "
            "or consent may have been induced through conduct falling "
            "within the statutory concept of fraud."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/english/"
            "UY2FqaJw2-apaUY2Fqa-a50%3D-con-128-sg-jjjjjjjjjjjjj"
        ),
    },

    {
        "id": "KB-CA-18",
        "title": "Contract Act, 1872 — Section 18",
        "law": "Contract Act, 1872",
        "jurisdiction": "Pakistan",
        "text": (
            "Section 18 concerns misrepresentation. It is relevant when "
            "a party may have entered an agreement because of a "
            "representation that falls within the statutory provision."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/english/"
            "UY2FqaJw2-apaUY2Fqa-a50%3D-con-128-sg-jjjjjjjjjjjjj"
        ),
    },

    {
        "id": "KB-CA-19",
        "title": "Contract Act, 1872 — Section 19",
        "law": "Contract Act, 1872",
        "jurisdiction": "Pakistan",
        "text": (
            "Section 19 addresses the consequences of agreements where "
            "consent was caused by coercion, fraud or misrepresentation, "
            "subject to the statutory conditions and exceptions."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/english/"
            "UY2FqaJw2-apaUY2Fqa-a50%3D-con-128-sg-jjjjjjjjjjjjj"
        ),
    },

    {
        "id": "KB-CA-23",
        "title": "Contract Act, 1872 — Section 23",
        "law": "Contract Act, 1872",
        "jurisdiction": "Pakistan",
        "text": (
            "Section 23 concerns lawful consideration and lawful object. "
            "It identifies circumstances in which consideration or the "
            "object of an agreement is not regarded as lawful."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/english/"
            "UY2FqaJw2-apaUY2Fqa-a50%3D-con-128-sg-jjjjjjjjjjjjj"
        ),
    },

    {
        "id": "KB-CA-28",
        "title": "Contract Act, 1872 — Section 28",
        "law": "Contract Act, 1872",
        "jurisdiction": "Pakistan",
        "text": (
            "Section 28 addresses agreements in restraint of legal "
            "proceedings, subject to the statutory wording and exceptions. "
            "It can therefore be relevant to clauses attempting to restrict "
            "a party's ability to pursue legal remedies."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/english/"
            "UY2FqaJw2-apaUY2Fqa-a50%3D-con-128-sg-jjjjjjjjjjjjj"
        ),
    },

    {
        "id": "KB-CA-62",
        "title": "Contract Act, 1872 — Section 62",
        "law": "Contract Act, 1872",
        "jurisdiction": "Pakistan",
        "text": (
            "Section 62 deals with novation, rescission and alteration "
            "of contracts. It can be relevant when the document changes "
            "or replaces an earlier contractual arrangement."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/english/"
            "UY2FqaJw2-apaUY2Fqa-a50%3D-con-128-sg-jjjjjjjjjjjjj"
        ),
    },

    {
        "id": "KB-CA-73",
        "title": "Contract Act, 1872 — Section 73",
        "law": "Contract Act, 1872",
        "jurisdiction": "Pakistan",
        "text": (
            "Section 73 concerns compensation for loss or damage caused "
            "by breach of contract. It is relevant when assessing possible "
            "contractual consequences following breach."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/english/"
            "UY2FqaJw2-apaUY2Fqa-a50%3D-con-128-sg-jjjjjjjjjjjjj"
        ),
    },

    {
        "id": "KB-CA-74",
        "title": "Contract Act, 1872 — Section 74",
        "law": "Contract Act, 1872",
        "jurisdiction": "Pakistan",
        "text": (
            "Section 74 concerns compensation where a contract names "
            "a sum payable on breach or contains another stipulation "
            "by way of penalty, subject to the statutory provision."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/english/"
            "UY2FqaJw2-apaUY2Fqa-a50%3D-con-128-sg-jjjjjjjjjjjjj"
        ),
    },


    # --------------------------------------------------------
    # CONSTITUTION
    # --------------------------------------------------------

    {
        "id": "KB-CONST-4",
        "title": "Constitution of Pakistan — Article 4",
        "law": "Constitution of the Islamic Republic of Pakistan",
        "jurisdiction": "Pakistan",
        "text": (
            "Article 4 concerns the right of individuals to be dealt "
            "with in accordance with law and relevant legal protections. "
            "It may be relevant when a document or dispute raises "
            "questions about treatment under law."
        ),
        "url": (
            "https://pakistancode.gov.pk/pdffiles/"
            "administrator9d8e2ecc414c6d3371ac41114b61a2c4.pdf"
        ),
    },

    {
        "id": "KB-CONST-10A",
        "title": "Constitution of Pakistan — Article 10A",
        "law": "Constitution of the Islamic Republic of Pakistan",
        "jurisdiction": "Pakistan",
        "text": (
            "Article 10A provides for the right to a fair trial and due "
            "process in the determination of civil rights and obligations "
            "or in a criminal charge."
        ),
        "url": (
            "https://pakistancode.gov.pk/pdffiles/"
            "administrator9d8e2ecc414c6d3371ac41114b61a2c4.pdf"
        ),
    },

    {
        "id": "KB-CONST-18",
        "title": "Constitution of Pakistan — Article 18",
        "law": "Constitution of the Islamic Republic of Pakistan",
        "jurisdiction": "Pakistan",
        "text": (
            "Article 18 concerns the freedom to enter lawful professions, "
            "occupations, trades or businesses, subject to the constitutional "
            "conditions and regulation described in the Article."
        ),
        "url": (
            "https://pakistancode.gov.pk/pdffiles/"
            "administrator9d8e2ecc414c6d3371ac41114b61a2c4.pdf"
        ),
    },

    {
        "id": "KB-CONST-23",
        "title": "Constitution of Pakistan — Article 23",
        "law": "Constitution of the Islamic Republic of Pakistan",
        "jurisdiction": "Pakistan",
        "text": (
            "Article 23 concerns the right of citizens to acquire, hold "
            "and dispose of property, subject to the Constitution and law."
        ),
        "url": (
            "https://pakistancode.gov.pk/pdffiles/"
            "administrator9d8e2ecc414c6d3371ac41114b61a2c4.pdf"
        ),
    },

    {
        "id": "KB-CONST-24",
        "title": "Constitution of Pakistan — Article 24",
        "law": "Constitution of the Islamic Republic of Pakistan",
        "jurisdiction": "Pakistan",
        "text": (
            "Article 24 provides constitutional protection concerning "
            "property and sets out circumstances in which property may "
            "be compulsorily acquired or dealt with according to law."
        ),
        "url": (
            "https://pakistancode.gov.pk/pdffiles/"
            "administrator9d8e2ecc414c6d3371ac41114b61a2c4.pdf"
        ),
    },

    {
        "id": "KB-CONST-25",
        "title": "Constitution of Pakistan — Article 25",
        "law": "Constitution of the Islamic Republic of Pakistan",
        "jurisdiction": "Pakistan",
        "text": (
            "Article 25 establishes equality of citizens before law "
            "and equal protection of law, subject to the constitutional "
            "text and permitted measures described there."
        ),
        "url": (
            "https://pakistancode.gov.pk/pdffiles/"
            "administrator9d8e2ecc414c6d3371ac41114b61a2c4.pdf"
        ),
    },


    # --------------------------------------------------------
    # ELECTRONIC TRANSACTIONS
    # --------------------------------------------------------

    {
        "id": "KB-ETO-2002",
        "title": "Electronic Transactions Ordinance, 2002",
        "law": "Electronic Transactions Ordinance, 2002",
        "jurisdiction": "Pakistan",
        "text": (
            "The Electronic Transactions Ordinance, 2002 provides a "
            "legal framework concerning electronic documents, electronic "
            "communications and electronic signatures. It may be relevant "
            "to agreements formed or signed electronically."
        ),
        "url": (
            "https://pakistancode.gov.pk/english/"
            "UY2FqaJw1-apaUY2Fqa-apaUY2Fta5Y%3D-sg-jjjjjjjjjjjjj"
        ),
    },


    # --------------------------------------------------------
    # PECA
    # --------------------------------------------------------

    {
        "id": "KB-PECA-2016",
        "title": "Prevention of Electronic Crimes Act, 2016",
        "law": "Prevention of Electronic Crimes Act, 2016",
        "jurisdiction": "Pakistan",
        "text": (
            "The Prevention of Electronic Crimes Act, 2016 provides "
            "the statutory framework for specified offences involving "
            "information systems, data and electronic activity. Any "
            "specific section should be checked against the current "
            "official text, including amendments."
        ),
        "url": (
            "https://www.pakistancode.gov.pk/pdffiles/"
            "administrator6a061efe0ed5bd153fa8b79b8eb4cba7.pdf"
        ),
    },


    # --------------------------------------------------------
    # WAFaqi MOHTASIB
    # --------------------------------------------------------

    {
        "id": "KB-WMS-COMPLAINT",
        "title": "Wafaqi Mohtasib — Public Complaints",
        "law": "Wafaqi Mohtasib",
        "jurisdiction": "Federal Government",
        "text": (
            "The Wafaqi Mohtasib handles complaints relating to "
            "maladministration by federal government agencies, departments "
            "or officials within its jurisdiction. This source is not a "
            "general substitute for private contractual remedies."
        ),
        "url": (
            "https://www.mohtasib.gov.pk/Detail/"
            "ZjllOWExZmItOGM4YS00YjllLWE5ZDQtOTNiNTY4YzYyYzVk"
        ),
    },
]


# ============================================================
# CSS
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>

        /* -----------------------------
           GLOBAL
        ----------------------------- */

        .stApp {
            background:
                radial-gradient(
                    circle at 70% 10%,
                    rgba(38, 82, 180, 0.18),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 15% 80%,
                    rgba(116, 61, 205, 0.12),
                    transparent 25%
                ),
                #020b18;
            color: #e9f1ff;
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        /* -----------------------------
           SIDEBAR
        ----------------------------- */

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #061529 0%,
                    #03101f 100%
                );
            border-right: 1px solid rgba(64, 132, 235, 0.20);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.2rem;
        }

        .brand {
            padding: 10px 12px 22px 12px;
            border-bottom: 1px solid rgba(90, 143, 220, 0.14);
            margin-bottom: 18px;
        }

        .brand-icon {
            width: 46px;
            height: 46px;
            border-radius: 15px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background:
                linear-gradient(
                    145deg,
                    #1689ff,
                    #7955ff 65%,
                    #ca62f4
                );
            box-shadow:
                0 0 24px rgba(71, 117, 255, 0.35);
            font-size: 25px;
            margin-right: 10px;
            vertical-align: middle;
        }

        .brand-title {
            display: inline-block;
            vertical-align: middle;
        }

        .brand-title h1 {
            font-size: 24px;
            margin: 0;
            font-weight: 800;
            letter-spacing: -0.8px;
        }

        .brand-title p {
            margin: 2px 0 0 0;
            color: #a978ff;
            font-weight: 700;
            font-size: 15px;
        }

        .brand-desc {
            color: #8fb4ea;
            line-height: 1.45;
            margin-top: 15px;
            font-size: 14px;
        }

        .side-note {
            margin-top: 25px;
            padding: 15px;
            border-radius: 14px;
            background: rgba(18, 53, 94, 0.22);
            border: 1px solid rgba(63, 130, 235, 0.18);
            color: #90b5e8;
            font-size: 13px;
        }

        /* -----------------------------
           HERO
        ----------------------------- */

        .hero {
            border: 1px solid rgba(51, 132, 255, 0.30);
            border-radius: 20px;
            padding: 30px;
            min-height: 250px;
            background:
                radial-gradient(
                    circle at 80% 30%,
                    rgba(48, 99, 255, 0.35),
                    transparent 35%
                ),
                radial-gradient(
                    circle at 20% 100%,
                    rgba(125, 55, 235, 0.20),
                    transparent 40%
                ),
                linear-gradient(
                    135deg,
                    rgba(9, 34, 76, 0.96),
                    rgba(3, 17, 38, 0.96)
                );
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.04),
                0 20px 50px rgba(0,0,0,0.18);
        }

        .hero-badge {
            color: #9bc5ff;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .hero h1 {
            margin: 0;
            font-size: 43px;
            line-height: 1.05;
            letter-spacing: -1.5px;
        }

        .gradient-text {
            background:
                linear-gradient(
                    90deg,
                    #27a8ff,
                    #6b72ff,
                    #b66df7
                );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            color: #a9c5ee;
            font-size: 16px;
            line-height: 1.6;
            max-width: 650px;
            margin-top: 15px;
        }

        .hero-visual {
            text-align: center;
            font-size: 92px;
            padding-top: 20px;
            filter:
                drop-shadow(0 0 25px rgba(51, 120, 255, 0.45));
        }

        /* -----------------------------
           CARDS
        ----------------------------- */

        .feature-card,
        .panel,
        .quick-card,
        .source-card,
        .risk-card {
            background:
                linear-gradient(
                    145deg,
                    rgba(7, 29, 57, 0.92),
                    rgba(3, 17, 34, 0.94)
                );
            border: 1px solid rgba(60, 125, 219, 0.22);
            border-radius: 17px;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.025),
                0 12px 35px rgba(0,0,0,0.14);
        }

        .feature-card {
            padding: 18px;
            min-height: 155px;
        }

        .feature-icon {
            width: 43px;
            height: 43px;
            border-radius: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            background: linear-gradient(
                135deg,
                #234dce,
                #7a4fe8
            );
            margin-bottom: 12px;
        }

        .feature-card h3 {
            margin: 0;
            font-size: 16px;
        }

        .feature-card p {
            color: #91b2dc;
            font-size: 13px;
            line-height: 1.5;
        }

        .panel {
            padding: 22px;
            margin-top: 18px;
        }

        .panel-title {
            font-size: 20px;
            font-weight: 750;
            margin-bottom: 5px;
        }

        .panel-subtitle {
            color: #86a8d6;
            font-size: 13px;
            margin-bottom: 18px;
        }

        /* -----------------------------
           UPLOAD BOX
        ----------------------------- */

        .upload-box {
            border: 1.5px dashed rgba(56, 137, 255, 0.70);
            border-radius: 17px;
            min-height: 210px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background:
                radial-gradient(
                    circle at center,
                    rgba(24, 78, 161, 0.13),
                    transparent 55%
                );
            color: #9fc4f4;
            text-align: center;
            padding: 20px;
        }

        .upload-icon {
            font-size: 55px;
            margin-bottom: 10px;
            filter:
                drop-shadow(0 0 15px rgba(35, 151, 255, 0.5));
        }

        /* -----------------------------
           RIGHT CARDS
        ----------------------------- */

        .quick-card {
            padding: 20px;
            margin-bottom: 16px;
        }

        .quick-title {
            font-size: 18px;
            font-weight: 750;
            margin-bottom: 17px;
        }

        .quick-step {
            display: flex;
            gap: 12px;
            margin-bottom: 15px;
            align-items: center;
        }

        .step-number {
            min-width: 32px;
            height: 32px;
            border-radius: 50%;
            background:
                linear-gradient(
                    135deg,
                    #2c8dff,
                    #7258e9
                );
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
        }

        .sample-card {
            padding: 20px;
        }

        .sample-card p {
            color: #8fb0dc;
            font-size: 14px;
            line-height: 1.5;
        }

        /* -----------------------------
           RISK
        ----------------------------- */

        .risk-high {
            border-left: 4px solid #ff5575;
        }

        .risk-medium {
            border-left: 4px solid #ffb84d;
        }

        .risk-low {
            border-left: 4px solid #39d9bd;
        }

        .risk-card {
            padding: 18px;
            margin-bottom: 13px;
        }

        .risk-label {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.5px;
        }

        .risk-label.high {
            background: rgba(255, 69, 103, 0.15);
            color: #ff718c;
        }

        .risk-label.medium {
            background: rgba(255, 184, 77, 0.14);
            color: #ffc66c;
        }

        .risk-label.low {
            background: rgba(57, 217, 189, 0.13);
            color: #54e3ca;
        }

        /* -----------------------------
           CITATIONS
        ----------------------------- */

        .citation {
            background: rgba(20, 54, 96, 0.38);
            border: 1px solid rgba(64, 133, 231, 0.22);
            border-radius: 10px;
            padding: 10px 12px;
            margin: 7px 0;
            font-size: 13px;
        }

        .citation-id {
            color: #65aaff;
            font-weight: 800;
        }

        /* -----------------------------
           DISCLAIMER
        ----------------------------- */

        .disclaimer {
            margin-top: 25px;
            padding: 16px 18px;
            border-radius: 14px;
            background:
                rgba(107, 65, 19, 0.16);
            border: 1px solid rgba(255, 183, 77, 0.25);
            color: #d9b986;
            font-size: 12px;
            line-height: 1.6;
        }

        /* -----------------------------
           STREAMLIT OVERRIDES
        ----------------------------- */

        .stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(53, 137, 255, 0.65);
            background:
                linear-gradient(
                    135deg,
                    #147eea,
                    #5a5ce6
                );
            color: white;
            font-weight: 700;
            min-height: 42px;
            box-shadow:
                0 7px 20px rgba(29, 106, 229, 0.18);
        }

        .stButton > button:hover {
            border-color: #75b8ff;
            transform: translateY(-1px);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] {
            background-color: #061a32 !important;
            border-color: rgba(64, 133, 231, 0.35) !important;
            color: white !important;
        }

        div[data-testid="stFileUploader"] {
            background: transparent;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 7px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
        }

        @media (max-width: 900px) {

            .hero h1 {
                font-size: 32px;
            }

            .hero-visual {
                font-size: 60px;
            }

        }

        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text: str) -> str:
    """Normalize extracted document text."""

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Parse JSON returned by the model.
    Attempts to recover JSON if the model wraps it in markdown.
    """

    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Remove markdown code fences
    text = re.sub(r"^```json\s*", "", text, flags=re.I)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return {}


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf(data: bytes) -> str:
    """Extract text from PDF using PyMuPDF."""

    text_parts = []

    with fitz.open(stream=data, filetype="pdf") as doc:

        for page_number, page in enumerate(doc):

            page_text = page.get_text("text")

            if page_text and page_text.strip():
                text_parts.append(
                    f"\n[Page {page_number + 1}]\n{page_text}"
                )

            else:
                # OCR fallback for scanned PDF pages
                try:
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(1.5, 1.5),
                        alpha=False
                    )

                    img = Image.frombytes(
                        "RGB",
                        [pix.width, pix.height],
                        pix.samples
                    )

                    ocr = pytesseract.image_to_string(
                        img,
                        lang="eng+urd"
                    )

                    if ocr.strip():
                        text_parts.append(
                            f"\n[Page {page_number + 1} - OCR]\n{ocr}"
                        )

                except Exception:
                    pass

    return clean_text("\n".join(text_parts))


def extract_docx(data: bytes) -> str:
    """
    DOCX extraction without python-docx.
    DOCX is a ZIP containing XML.
    """

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:

            xml = z.read("word/document.xml").decode(
                "utf-8",
                errors="ignore"
            )

        # Paragraph breaks
        xml = re.sub(
            r"</w:p>",
            "\n",
            xml,
            flags=re.I
        )

        # Table cells
        xml = re.sub(
            r"</w:tc>",
            "\t",
            xml,
            flags=re.I
        )

        # Remove XML tags
        text = re.sub(
            r"<[^>]+>",
            "",
            xml
        )

        # Decode entities
        text = html.unescape(text)

        return clean_text(text)

    except Exception as e:
        raise RuntimeError(
            f"Could not read DOCX file: {e}"
        )


def extract_xlsx(data: bytes) -> str:
    """
    Lightweight XLSX extractor using stdlib ZIP/XML.
    """

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:

            names = z.namelist()

            shared_strings = []

            if "xl/sharedStrings.xml" in names:

                xml = z.read(
                    "xl/sharedStrings.xml"
                ).decode(
                    "utf-8",
                    errors="ignore"
                )

                strings = re.findall(
                    r"<t[^>]*>(.*?)</t>",
                    xml,
                    flags=re.S
                )

                shared_strings = [
                    html.unescape(re.sub("<[^>]+>", "", s))
                    for s in strings
                ]

            sheets = [
                n for n in names
                if re.match(
                    r"xl/worksheets/sheet\d+\.xml",
                    n
                )
            ]

            output = []

            for sheet_name in sorted(sheets):

                xml = z.read(sheet_name).decode(
                    "utf-8",
                    errors="ignore"
                )

                rows = re.findall(
                    r"<row[^>]*>(.*?)</row>",
                    xml,
                    flags=re.S
                )

                output.append(
                    f"\n[{sheet_name}]\n"
                )

                for row in rows:

                    cells = re.findall(
                        r"<c[^>]*?(?:t=\"([^\"]+)\")?[^>]*>"
                        r"(.*?)</c>",
                        row,
                        flags=re.S
                    )

                    values = []

                    for cell_type, cell_xml in cells:

                        value_match = re.search(
                            r"<v>(.*?)</v>",
                            cell_xml,
                            flags=re.S
                        )

                        if not value_match:
                            inline_match = re.search(
                                r"<t[^>]*>(.*?)</t>",
                                cell_xml,
                                flags=re.S
                            )

                            if inline_match:
                                value = html.unescape(
                                    inline_match.group(1)
                                )
                            else:
                                value = ""

                        else:
                            value = value_match.group(1)

                            if cell_type == "s":
                                try:
                                    value = shared_strings[
                                        int(value)
                                    ]
                                except Exception:
                                    pass

                        values.append(str(value))

                    if values:
                        output.append(
                            " | ".join(values)
                        )

            return clean_text("\n".join(output))

    except Exception as e:
        raise RuntimeError(
            f"Could not read XLSX file: {e}"
        )


def extract_csv(data: bytes) -> str:
    import csv

    text = data.decode(
        "utf-8",
        errors="ignore"
    )

    rows = []

    reader = csv.reader(
        io.StringIO(text)
    )

    for row in reader:
        rows.append(" | ".join(row))

    return clean_text("\n".join(rows))


def extract_image(data: bytes) -> str:
    """OCR images in English + Urdu."""

    image = Image.open(
        io.BytesIO(data)
    )

    image = image.convert("RGB")

    try:
        text = pytesseract.image_to_string(
            image,
            lang="eng+urd"
        )
    except Exception:
        text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

    return clean_text(text)


def extract_document(
    filename: str,
    data: bytes
) -> str:

    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_pdf(data)

    if ext == ".docx":
        return extract_docx(data)

    if ext == ".xlsx":
        return extract_xlsx(data)

    if ext == ".csv":
        return extract_csv(data)

    if ext in [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
    ]:
        return extract_image(data)

    if ext in [
        ".txt",
        ".md",
    ]:
        return clean_text(
            data.decode(
                "utf-8",
                errors="ignore"
            )
        )

    raise ValueError(
        f"Unsupported file format: {ext}"
    )


# ============================================================
# CHUNKING
# ============================================================

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:

    text = clean_text(text)

    if not text:
        return []

    paragraphs = text.split("\n")

    chunks = []
    current = ""

    chunk_id = 1

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current) + len(paragraph) + 1 <= chunk_size:

            current += (
                "\n" + paragraph
            ).strip()

        else:

            if current:
                chunks.append({
                    "id": f"DOC-{chunk_id}",
                    "text": current,
                    "source": "Uploaded document",
                })

                chunk_id += 1

            # overlap from previous chunk
            overlap_text = current[
                -overlap:
            ] if current else ""

            current = (
                overlap_text
                + "\n"
                + paragraph
            ).strip()

    if current:
        chunks.append({
            "id": f"DOC-{chunk_id}",
            "text": current,
            "source": "Uploaded document",
        })

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


def embed_texts(
    texts: List[str]
) -> np.ndarray:

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=16
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )


def build_faiss_index(
    chunks: List[Dict[str, Any]]
):

    texts = [
        c["text"]
        for c in chunks
    ]

    embeddings = embed_texts(
        texts
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index, embeddings


def search_index(
    query: str,
    chunks: List[Dict[str, Any]],
    index,
    top_k: int = 5
):

    if not chunks or index is None:
        return []

    query_embedding = embed_texts(
        [query]
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

        item = dict(
            chunks[idx]
        )

        item["score"] = float(
            score
        )

        results.append(item)

    return results


# ============================================================
# LEGAL KNOWLEDGE BASE INDEX
# ============================================================

@st.cache_resource(show_spinner=False)
def build_legal_index():

    texts = [
        (
            f"{item['title']}\n"
            f"{item['law']}\n"
            f"{item['text']}"
        )
        for item in LEGAL_KB
    ]

    embeddings = embed_texts(
        texts
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return index


def search_legal_kb(
    query: str,
    top_k: int = TOP_K_LEGAL
):

    index = build_legal_index()

    embeddings = embed_texts(
        [query]
    )

    k = min(
        top_k,
        len(LEGAL_KB)
    )

    scores, ids = index.search(
        embeddings,
        k
    )

    results = []

    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        if idx < 0:
            continue

        item = dict(
            LEGAL_KB[idx]
        )

        item["score"] = float(
            score
        )

        results.append(item)

    return results


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_groq_client():

    if not GROQ_API_KEY:
        return None

    return Groq(
        api_key=GROQ_API_KEY
    )


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 7000
) -> str:

    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
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
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={
            "type": "json_object"
        },
    )

    return response.choices[0].message.content


# ============================================================
# PROMPTS
# ============================================================

LEGAL_SYSTEM_PROMPT = """
You are DocuMind, a Pakistan-focused legal document analysis assistant.

You are NOT a lawyer.

Your job is to analyze documents for informational purposes only.

CRITICAL LEGAL ACCURACY RULES:

1. NEVER invent Pakistani laws.
2. NEVER invent section numbers.
3. NEVER invent court cases.
4. NEVER invent legal authorities.
5. NEVER create fake citations.
6. Only cite legal sources explicitly supplied in the LEGAL SOURCES
   section of the prompt.
7. If the supplied legal sources do not establish something, say:
   "The supplied legal knowledge base does not establish this."
8. Distinguish between:
   - what the document actually says,
   - a possible legal concern,
   - and a confirmed legal rule.
9. Never say a clause is definitely "illegal", "void", "criminal",
   or "unenforceable" unless the supplied source directly supports
   that conclusion.
10. Prefer language such as:
   "potentially risky",
   "may warrant legal review",
   "could raise a concern",
   "appears unusual".
11. If jurisdiction matters, explicitly state that provincial/federal
    differences may apply.
12. Never treat an outdated legal summary as current law.
13. Specific legal sections must only be cited when the supplied
    knowledge base contains that section.

DOCUMENT CITATION RULE:

Document chunks are labelled DOC-1, DOC-2, etc.

Whenever making a claim about the uploaded document, cite one or
more relevant document chunk IDs such as [DOC-3].

LEGAL CITATION RULE:

Whenever making a legal claim based on the legal knowledge base,
cite the supplied legal source ID such as [KB-CA-10].

If there is insufficient evidence, do not fabricate a citation.

LANGUAGE:

Answer in the requested language:
- English
- Urdu

If Urdu is selected, use clear Pakistani Urdu.
Keep legal act names and section numbers in English where useful.

Return JSON only.
"""


# ============================================================
# DOCUMENT ANALYSIS
# ============================================================

def analyze_document(
    document_chunks: List[Dict[str, Any]],
    language: str,
    jurisdiction: str
) -> Dict[str, Any]:

    # Retrieve representative document chunks.
    all_text = "\n\n".join(
        f"[{c['id']}]\n{c['text']}"
        for c in document_chunks[:30]
    )

    legal_results = search_legal_kb(
        "contract unfair clauses consent breach liability rights "
        "legal remedies electronic agreement",
        top_k=TOP_K_LEGAL
    )

    legal_context = "\n\n".join(
        (
            f"[{x['id']}]\n"
            f"Title: {x['title']}\n"
            f"Law: {x['law']}\n"
            f"Content: {x['text']}"
        )
        for x in legal_results
    )

    prompt = f"""
Analyze the uploaded document.

Requested response language:
{language}

Jurisdiction:
{jurisdiction}

DOCUMENT:

{all_text}

VERIFIED LEGAL SOURCES:

{legal_context}

Return this exact JSON structure:

{{
  "summary": "Simple-language summary.",
  "document_type": "Likely document type.",
  "overall_assessment": "Short overall assessment.",
  "risk_counts": {{
      "high": 0,
      "medium": 0,
      "low": 0
  }},
  "risks": [
    {{
      "title": "Risk title",
      "risk_level": "High|Medium|Low",
      "clause": "Short quote or faithful excerpt from document",
      "explanation": "Why this may be risky",
      "impact": "Potential impact on the user",
      "recommendation": "What the user should consider reviewing",
      "document_citations": ["DOC-1"],
      "legal_citations": ["KB-CA-10"]
    }}
  ],
  "rights": [
    {{
      "right": "Potentially relevant right",
      "explanation": "Simple explanation",
      "document_citations": ["DOC-2"],
      "legal_citations": ["KB-CONST-25"]
    }}
  ],
  "important_clauses": [
    {{
      "title": "Clause name",
      "importance": "Why it matters",
      "document_citations": ["DOC-4"],
      "legal_citations": []
    }}
  ],
  "missing_or_unclear_clauses": [
    {{
      "title": "Potentially missing clause",
      "why_it_matters": "Why users may want to check this",
      "certainty": "Possible|Likely|Unclear"
    }}
  ],
  "key_obligations": [
    {{
      "party": "Party name or side",
      "obligation": "Obligation",
      "document_citations": ["DOC-5"]
    }}
  ],
  "questions_to_consider": [
    "Question the user may want to ask a lawyer or the other party"
  ]
}}

IMPORTANT:

Do not manufacture missing clauses.

A "missing clause" means the document does not appear to contain
an obvious provision that could reasonably matter for the type of
document. Mark uncertainty appropriately.

Every document-based claim should have document_citations.

Every legal claim should have legal_citations.

If no legal source supports a claim, leave legal_citations empty.
"""

    raw = call_llm(
        LEGAL_SYSTEM_PROMPT,
        prompt,
        temperature=0.05,
        max_tokens=8000
    )

    result = safe_json_loads(
        raw
    )

    if not result:
        return {
            "summary": raw,
            "risks": [],
            "rights": [],
            "important_clauses": [],
            "missing_or_unclear_clauses": [],
            "key_obligations": [],
            "questions_to_consider": [],
        }

    return result


# ============================================================
# Q&A
# ============================================================

def answer_document_question(
    question: str,
    document_chunks: List[Dict[str, Any]],
    document_index,
    language: str,
    jurisdiction: str
):

    doc_results = search_index(
        question,
        document_chunks,
        document_index,
        TOP_K_DOCUMENT
    )

    legal_results = search_legal_kb(
        question,
        TOP_K_LEGAL
    )

    document_context = "\n\n".join(
        (
            f"[{x['id']}]\n"
            f"{x['text']}"
        )
        for x in doc_results
    )

    legal_context = "\n\n".join(
        (
            f"[{x['id']}]\n"
            f"{x['title']}\n"
            f"{x['text']}"
        )
        for x in legal_results
    )

    prompt = f"""
Answer the user's question about the uploaded document.

Question:
{question}

Language:
{language}

Jurisdiction:
{jurisdiction}

RELEVANT DOCUMENT EXCERPTS:

{document_context}

RELEVANT VERIFIED LEGAL SOURCES:

{legal_context}

Return JSON:

{{
  "answer": "Clear answer",
  "confidence": "High|Medium|Low",
  "document_citations": ["DOC-1"],
  "legal_citations": ["KB-CA-10"],
  "limitations": "Explain any uncertainty"
}}

Rules:

- Use the document as the primary source for what the document says.
- Use the legal knowledge base only for legal context.
- Never invent legal citations.
- If the answer cannot be established from the supplied context,
  explicitly say so.
- Do not give definitive legal advice.
- Do not claim a clause is definitely illegal unless supported.
"""

    raw = call_llm(
        LEGAL_SYSTEM_PROMPT,
        prompt,
        temperature=0.05,
        max_tokens=3000
    )

    return safe_json_loads(
        raw
    )


# ============================================================
# CITATION RENDERING
# ============================================================

def render_citations(
    document_ids=None,
    legal_ids=None
):

    document_ids = document_ids or []
    legal_ids = legal_ids or []

    if not document_ids and not legal_ids:
        return

    st.markdown(
        "<div style='margin-top:10px;'>"
        "<b style='color:#8fb8ec;'>Sources</b>"
        "</div>",
        unsafe_allow_html=True
    )

    for doc_id in document_ids:

        st.markdown(
            f"""
            <div class="citation">
                <span class="citation-id">{doc_id}</span>
                — Uploaded document evidence
            </div>
            """,
            unsafe_allow_html=True
        )

    legal_map = {
        x["id"]: x
        for x in LEGAL_KB
    }

    for legal_id in legal_ids:

        source = legal_map.get(
            legal_id
        )

        if not source:
            continue

        st.markdown(
            f"""
            <div class="citation">
                <span class="citation-id">
                    {html.escape(source["id"])}
                </span>
                —
                {html.escape(source["title"])}
                <br>
                <a href="{html.escape(source["url"])}"
                   target="_blank"
                   style="color:#62aaff;">
                    Official source
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">

            <div>
                <span class="brand-icon">⚖</span>

                <span class="brand-title">
                    <h1>DocuMind</h1>
                    <p>Legal Risk AI</p>
                </span>
            </div>

            <div class="brand-desc">
                Pakistan-Focused AI Legal Risk &
                Rights Analyzer
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📄 Upload & Analyze",
            "🔍 Scan & Analyze",
            "⚠️ Risks Found",
            "💬 Ask DocuMind",
            "📚 Legal Sources",
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class="side-note">
            🛡️ <b>Privacy-first demo</b><br><br>
            Uploaded documents are processed for the
            current Streamlit session and are not intentionally
            saved as a permanent document database by this app.
        </div>

        <div class="side-note">
            💡 <b>Smarter Documents.<br>
            Safer Decisions.</b>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "document_text": "",
    "document_chunks": [],
    "document_index": None,
    "filename": "",
    "analysis": None,
    "history": [],
    "document_hash": "",
    "language": "English",
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# TOP BAR
# ============================================================

top_left, top_right = st.columns(
    [5, 1]
)

with top_right:

    language = st.selectbox(
        "Language",
        [
            "English",
            "Urdu",
        ],
        index=0 if st.session_state.language == "English" else 1,
        label_visibility="collapsed"
    )

    st.session_state.language = language


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    left, right = st.columns(
        [3.1, 1]
    )

    with left:

        st.markdown(
            """
            <div class="hero">

                <div class="hero-badge">
                    ⚖️ PAKISTAN-FOCUSED LEGAL AI
                </div>

                <h1>
                    Docu<span class="gradient-text">Mind</span>
                </h1>

                <h2 style="
                    margin-top:3px;
                    font-size:25px;
                    color:#c4d8f7;
                ">
                    Legal Risk & Rights Analyzer
                </h2>

                <p>
                    Upload a contract, agreement, legal document,
                    spreadsheet or image and let AI identify
                    potentially risky clauses, important rights,
                    missing provisions and legal references.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<div style='height:18px'></div>",
            unsafe_allow_html=True
        )

        feature_columns = st.columns(4)

        features = [
            (
                "🛡️",
                "Find Hidden Clauses",
                "Detect potentially unfair or risky language."
            ),
            (
                "📜",
                "Check Legal Context",
                "Ground findings in a curated Pakistani legal KB."
            ),
            (
                "⚠️",
                "Assess Legal Risks",
                "Prioritize High, Medium and Low risk findings."
            ),
            (
                "💡",
                "Simple Insights",
                "Explain complex documents in plain language."
            ),
        ]

        for col, item in zip(
            feature_columns,
            features
        ):

            with col:

                icon, title, description = item

                st.markdown(
                    f"""
                    <div class="feature-card">

                        <div class="feature-icon">
                            {icon}
                        </div>

                        <h3>{title}</h3>

                        <p>{description}</p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(
            """
            <div class="panel">

                <div class="panel-title">
                    📄 Upload Your Document
                </div>

                <div class="panel-subtitle">
                    PDF, DOCX, XLSX, CSV, TXT or image
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose a document",
            type=[
                "pdf",
                "docx",
                "xlsx",
                "csv",
                "txt",
                "png",
                "jpg",
                "jpeg",
                "webp",
                "bmp",
                "tiff",
            ],
            label_visibility="collapsed"
        )

        if uploaded_file:

            if uploaded_file.size > (
                MAX_FILE_SIZE_MB * 1024 * 1024
            ):

                st.error(
                    f"File is larger than "
                    f"{MAX_FILE_SIZE_MB} MB."
                )

            else:

                file_data = uploaded_file.getvalue()

                file_hash = hash_bytes(
                    file_data
                )

                if (
                    file_hash
                    != st.session_state.document_hash
                ):

                    with st.spinner(
                        "Extracting document text..."
                    ):

                        try:

                            text = extract_document(
                                uploaded_file.name,
                                file_data
                            )

                            chunks = split_text(
                                text
                            )

                            if not chunks:
                                st.error(
                                    "No readable text was found."
                                )

                            else:

                                with st.spinner(
                                    "Building semantic search index..."
                                ):

                                    index, _ = build_faiss_index(
                                        chunks
                                    )

                                st.session_state.document_text = text
                                st.session_state.document_chunks = chunks
                                st.session_state.document_index = index
                                st.session_state.filename = uploaded_file.name
                                st.session_state.document_hash = file_hash
                                st.session_state.analysis = None

                                st.success(
                                    f"Loaded {uploaded_file.name} "
                                    f"— {len(chunks)} searchable chunks."
                                )

                        except Exception as e:

                            st.error(
                                f"Could not process document: {e}"
                            )

        if st.session_state.document_chunks:

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )

            if st.button(
                "🔎 Analyze Document",
                use_container_width=True
            ):

                with st.spinner(
                    "DocuMind is analyzing the document..."
                ):

                    try:

                        st.session_state.analysis = (
                            analyze_document(
                                st.session_state.document_chunks,
                                st.session_state.language,
                                "Pakistan"
                            )
                        )

                        st.success(
                            "Analysis completed."
                        )

                    except Exception as e:

                        st.error(
                            f"Analysis failed: {e}"
                        )

    with right:

        st.markdown(
            """
            <div class="quick-card">

                <div class="quick-title">
                    🚀 Quick Start
                </div>

                <div class="quick-step">
                    <div class="step-number">1</div>
                    <div>Upload a document</div>
                </div>

                <div class="quick-step">
                    <div class="step-number">2</div>
                    <div>Click Analyze</div>
                </div>

                <div class="quick-step">
                    <div class="step-number">3</div>
                    <div>Review risks & rights</div>
                </div>

                <div class="quick-step">
                    <div class="step-number">4</div>
                    <div>Ask questions</div>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="quick-card sample-card">

                <div class="quick-title">
                    📄 Sample Document
                </div>

                <p>
                    Test the application using a small
                    example agreement.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "📄 Use Sample Contract",
            use_container_width=True
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
two years, the employee shall pay PKR 1,000,000 to the company.

6. DISPUTES
Any dispute shall be resolved according to applicable
laws of Pakistan.

7. MODIFICATION
The company may modify the terms of this agreement
at any time without obtaining the employee's consent.
"""

            chunks = split_text(
                sample
            )

            index, _ = build_faiss_index(
                chunks
            )

            st.session_state.document_text = sample
            st.session_state.document_chunks = chunks
            st.session_state.document_index = index
            st.session_state.filename = "sample_contract.txt"
            st.session_state.document_hash = hash_bytes(
                sample.encode()
            )
            st.session_state.analysis = None

            st.success(
                "Sample contract loaded."
            )


# ============================================================
# UPLOAD & ANALYZE
# ============================================================

elif page == "📄 Upload & Analyze":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                DOCUMENT ANALYSIS
            </div>

            <h1>
                Understand your
                <span class="gradient-text">
                    document
                </span>
            </h1>

            <p>
                Upload a legal document and DocuMind will
                extract its contents, create semantic embeddings,
                retrieve relevant Pakistani legal sources and
                produce an evidence-grounded analysis.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload document",
        type=[
            "pdf",
            "docx",
            "xlsx",
            "csv",
            "txt",
            "png",
            "jpg",
            "jpeg",
            "webp",
            "bmp",
            "tiff",
        ]
    )

    if uploaded_file:

        if uploaded_file.size > (
            MAX_FILE_SIZE_MB * 1024 * 1024
        ):

            st.error(
                f"Maximum file size is "
                f"{MAX_FILE_SIZE_MB} MB."
            )

        else:

            data = uploaded_file.getvalue()

            with st.spinner(
                "Extracting text..."
            ):

                try:

                    text = extract_document(
                        uploaded_file.name,
                        data
                    )

                    chunks = split_text(
                        text
                    )

                    if not chunks:

                        st.error(
                            "No readable text found."
                        )

                    else:

                        with st.spinner(
                            "Creating semantic index..."
                        ):

                            index, _ = build_faiss_index(
                                chunks
                            )

                        st.session_state.document_text = text
                        st.session_state.document_chunks = chunks
                        st.session_state.document_index = index
                        st.session_state.filename = uploaded_file.name
                        st.session_state.document_hash = hash_bytes(data)
                        st.session_state.analysis = None

                        st.success(
                            f"{uploaded_file.name} processed successfully."
                        )

                except Exception as e:

                    st.error(
                        f"Processing error: {e}"
                    )

    if st.session_state.document_chunks:

        st.markdown(
            f"""
            <div class="panel">

                <div class="panel-title">
                    📄 {html.escape(
                        st.session_state.filename
                    )}
                </div>

                <div class="panel-subtitle">
                    {len(
                        st.session_state.document_chunks
                    )} semantic chunks indexed
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "⚡ Run Full Legal Analysis",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing clauses, rights, risks and legal context..."
            ):

                try:

                    st.session_state.analysis = (
                        analyze_document(
                            st.session_state.document_chunks,
                            st.session_state.language,
                            "Pakistan"
                        )
                    )

                    st.success(
                        "Full analysis completed."
                    )

                except Exception as e:

                    st.error(
                        f"Analysis error: {e}"
                    )


# ============================================================
# SCAN & ANALYZE
# ============================================================

elif page == "🔍 Scan & Analyze":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                AI DOCUMENT SCANNER
            </div>

            <h1>
                Scan for
                <span class="gradient-text">
                    legal risk
                </span>
            </h1>

            <p>
                DocuMind uses semantic retrieval to compare
                document clauses against relevant entries in
                its curated Pakistani legal knowledge base.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.document_chunks:

        st.info(
            "Upload a document first from "
            "'Upload & Analyze'."
        )

    else:

        st.write(
            f"Current document: "
            f"**{st.session_state.filename}**"
        )

        query = st.text_input(
            "What should DocuMind scan for?",
            placeholder=(
                "e.g. termination, penalties, "
                "unfair obligations, liability, consent..."
            )
        )

        if st.button(
            "🔍 Scan",
            use_container_width=True
        ):

            if not query.strip():

                st.warning(
                    "Enter something to scan for."
                )

            else:

                document_results = search_index(
                    query,
                    st.session_state.document_chunks,
                    st.session_state.document_index,
                    TOP_K_DOCUMENT
                )

                legal_results = search_legal_kb(
                    query,
                    TOP_K_LEGAL
                )

                st.markdown(
                    "### Document Evidence"
                )

                for result in document_results:

                    score = result["score"]

                    st.markdown(
                        f"""
                        <div class="risk-card">

                            <b>{result["id"]}</b>

                            <span style="
                                float:right;
                                color:#70adff;
                            ">
                                Similarity:
                                {score:.2f}
                            </span>

                            <p style="
                                color:#a4bee2;
                                margin-top:10px;
                            ">
                                {html.escape(
                                    result["text"]
                                )}
                            </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    "### Relevant Pakistani Legal Sources"
                )

                for source in legal_results:

                    st.markdown(
                        f"""
                        <div class="source-card"
                             style="padding:16px;margin-bottom:10px;">

                            <b>
                                {html.escape(
                                    source["title"]
                                )}
                            </b>

                            <p style="
                                color:#9cb9df;
                                font-size:13px;
                            ">
                                {html.escape(
                                    source["text"]
                                )}
                            </p>

                            <a href="{html.escape(source["url"])}"
                               target="_blank"
                               style="color:#64aaff;">
                                View official source →
                            </a>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# RISKS FOUND
# ============================================================

elif page == "⚠️ Risks Found":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                RISK DASHBOARD
            </div>

            <h1>
                Potential
                <span class="gradient-text">
                    risks found
                </span>
            </h1>

            <p>
                These are AI-generated risk indicators,
                not definitive legal conclusions.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    analysis = st.session_state.analysis

    if not analysis:

        st.info(
            "Run an analysis first."
        )

    else:

        counts = analysis.get(
            "risk_counts",
            {}
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "🔴 High",
                counts.get("high", 0)
            )

        with c2:
            st.metric(
                "🟠 Medium",
                counts.get("medium", 0)
            )

        with c3:
            st.metric(
                "🟢 Low",
                counts.get("low", 0)
            )

        st.markdown(
            "## Risk Findings"
        )

        risks = analysis.get(
            "risks",
            []
        )

        if not risks:

            st.success(
                "No specific risks were identified "
                "from the supplied context."
            )

        for risk in risks:

            level = str(
                risk.get(
                    "risk_level",
                    "Medium"
                )
            ).lower()

            if level not in [
                "high",
                "medium",
                "low"
            ]:
                level = "medium"

            st.markdown(
                f"""
                <div class="risk-card risk-{level}">

                    <span class="risk-label {level}">
                        {level.upper()}
                    </span>

                    <h3>
                        {html.escape(
                            risk.get("title", "Risk")
                        )}
                    </h3>

                    <p>
                        <b>Clause:</b><br>
                        <span style="
                            color:#b8ccef;
                        ">
                            {html.escape(
                                risk.get(
                                    "clause",
                                    ""
                                )
                            )}
                        </span>
                    </p>

                    <p style="color:#9db8dd;">
                        <b>Why it matters:</b>
                        {html.escape(
                            risk.get(
                                "explanation",
                                ""
                            )
                        )}
                    </p>

                    <p style="color:#9db8dd;">
                        <b>Potential impact:</b>
                        {html.escape(
                            risk.get(
                                "impact",
                                ""
                            )
                        )}
                    </p>

                    <p style="color:#9db8dd;">
                        <b>Recommendation:</b>
                        {html.escape(
                            risk.get(
                                "recommendation",
                                ""
                            )
                        )}
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            render_citations(
                risk.get(
                    "document_citations",
                    []
                ),
                risk.get(
                    "legal_citations",
                    []
                )
            )


# ============================================================
# ASK DOCUMIND
# ============================================================

elif page == "💬 Ask DocuMind":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                DOCUMENT Q&A
            </div>

            <h1>
                Ask your
                <span class="gradient-text">
                    document
                </span>
            </h1>

            <p>
                Ask questions about the uploaded document.
                Answers are generated from retrieved document
                passages and the curated Pakistani legal KB.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.document_chunks:

        st.info(
            "Upload and analyze a document first."
        )

    else:

        question = st.text_area(
            "Ask a question",
            placeholder=(
                "Example: Can the company terminate this "
                "agreement without notice?"
            ),
            height=120
        )

        if st.button(
            "💬 Ask DocuMind",
            use_container_width=True
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Searching the document and legal knowledge base..."
                ):

                    try:

                        answer = answer_document_question(
                            question,
                            st.session_state.document_chunks,
                            st.session_state.document_index,
                            st.session_state.language,
                            "Pakistan"
                        )

                        st.session_state.history.append(
                            {
                                "question": question,
                                "answer": answer
                            }
                        )

                    except Exception as e:

                        st.error(
                            f"Q&A error: {e}"
                        )

        # Show history
        for item in reversed(
            st.session_state.history
        ):

            st.markdown(
                f"""
                <div class="panel">

                    <div style="
                        color:#69acff;
                        font-weight:750;
                    ">
                        You
                    </div>

                    <div style="
                        margin-top:8px;
                        color:#dceaff;
                    ">
                        {html.escape(
                            item["question"]
                        )}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            answer = item["answer"]

            st.markdown(
                f"""
                <div class="panel">

                    <div style="
                        color:#a56eff;
                        font-weight:750;
                    ">
                        ⚖️ DocuMind
                    </div>

                    <div style="
                        margin-top:8px;
                        color:#c0d5f2;
                        line-height:1.7;
                    ">
                        {html.escape(
                            answer.get(
                                "answer",
                                "No answer available."
                            )
                        )}
                    </div>

                    <div style="
                        margin-top:14px;
                        color:#819fc8;
                        font-size:12px;
                    ">
                        Confidence:
                        {html.escape(
                            answer.get(
                                "confidence",
                                "Unknown"
                            )
                        )}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            render_citations(
                answer.get(
                    "document_citations",
                    []
                ),
                answer.get(
                    "legal_citations",
                    []
                )
            )

            if answer.get(
                "limitations"
            ):

                st.caption(
                    "⚠️ "
                    + answer["limitations"]
                )


# ============================================================
# LEGAL SOURCES
# ============================================================

elif page == "📚 Legal Sources":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                VERIFIED SOURCE REGISTRY
            </div>

            <h1>
                Pakistani Legal
                <span class="gradient-text">
                    Sources
                </span>
            </h1>

            <p>
                DocuMind only uses legal references that are
                explicitly present in this curated source registry.
                This prevents the model from inventing citations.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    for source in LEGAL_KB:

        st.markdown(
            f"""
            <div class="source-card"
                 style="
                    padding:20px;
                    margin-bottom:12px;
                 ">

                <div style="
                    color:#6dafff;
                    font-size:12px;
                    font-weight:800;
                ">
                    {html.escape(source["id"])}
                </div>

                <h3 style="
                    margin:5px 0 7px 0;
                ">
                    {html.escape(source["title"])}
                </h3>

                <div style="
                    color:#89a9d4;
                    font-size:13px;
                    margin-bottom:10px;
                ">
                    {html.escape(source["law"])}
                </div>

                <p style="
                    color:#b1c7e6;
                    line-height:1.6;
                ">
                    {html.escape(source["text"])}
                </p>

                <a href="{html.escape(source["url"])}"
                   target="_blank"
                   style="
                        color:#61aaff;
                        font-weight:700;
                   ">
                    Open official source →
                </a>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# ANALYSIS OVERVIEW
# ============================================================

# Display this underneath most pages if analysis exists.

if (
    st.session_state.analysis
    and page in [
        "🏠 Home",
        "📄 Upload & Analyze",
    ]
):

    analysis = st.session_state.analysis

    st.markdown(
        "<br><br>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero"
             style="min-height:auto;">

            <div class="hero-badge">
                ANALYSIS COMPLETE
            </div>

            <h1 style="font-size:32px;">
                Your document
                <span class="gradient-text">
                    insights
                </span>
            </h1>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">
                📝 Simple Summary
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        analysis.get(
            "summary",
            "No summary available."
        )
    )

    # --------------------------------------------------------
    # OVERALL ASSESSMENT
    # --------------------------------------------------------

    overall = analysis.get(
        "overall_assessment"
    )

    if overall:

        st.info(
            overall
        )

    # --------------------------------------------------------
    # RIGHTS
    # --------------------------------------------------------

    rights = analysis.get(
        "rights",
        []
    )

    if rights:

        st.markdown(
            "## 🛡️ Potentially Relevant Rights"
        )

        for right in rights:

            st.markdown(
                f"""
                <div class="panel">

                    <h3>
                        {html.escape(
                            right.get(
                                "right",
                                ""
                            )
                        )}
                    </h3>

                    <p style="
                        color:#a3bfdf;
                        line-height:1.6;
                    ">
                        {html.escape(
                            right.get(
                                "explanation",
                                ""
                            )
                        )}
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            render_citations(
                right.get(
                    "document_citations",
                    []
                ),
                right.get(
                    "legal_citations",
                    []
                )
            )

    # --------------------------------------------------------
    # IMPORTANT CLAUSES
    # --------------------------------------------------------

    important = analysis.get(
        "important_clauses",
        []
    )

    if important:

        st.markdown(
            "## 📌 Important Clauses"
        )

        for item in important:

            st.markdown(
                f"""
                <div class="panel">

                    <h3>
                        {html.escape(
                            item.get(
                                "title",
                                ""
                            )
                        )}
                    </h3>

                    <p style="
                        color:#a2bfdf;
                    ">
                        {html.escape(
                            item.get(
                                "importance",
                                ""
                            )
                        )}
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            render_citations(
                item.get(
                    "document_citations",
                    []
                ),
                item.get(
                    "legal_citations",
                    []
                )
            )

    # --------------------------------------------------------
    # MISSING / UNCLEAR
    # --------------------------------------------------------

    missing = analysis.get(
        "missing_or_unclear_clauses",
        []
    )

    if missing:

        st.markdown(
            "## 🔎 Missing or Unclear Clauses"
        )

        for item in missing:

            st.markdown(
                f"""
                <div class="risk-card">

                    <span class="risk-label medium">
                        {html.escape(
                            item.get(
                                "certainty",
                                "Possible"
                            )
                        ).upper()}
                    </span>

                    <h3>
                        {html.escape(
                            item.get(
                                "title",
                                ""
                            )
                        )}
                    </h3>

                    <p style="
                        color:#a2bfdf;
                    ">
                        {html.escape(
                            item.get(
                                "why_it_matters",
                                ""
                            )
                        )}
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # OBLIGATIONS
    # --------------------------------------------------------

    obligations = analysis.get(
        "key_obligations",
        []
    )

    if obligations:

        st.markdown(
            "## 📋 Key Obligations"
        )

        for item in obligations:

            st.markdown(
                f"""
                <div class="panel">

                    <b>
                        {html.escape(
                            item.get(
                                "party",
                                "Party"
                            )
                        )}
                    </b>

                    <p style="
                        color:#a2bfdf;
                    ">
                        {html.escape(
                            item.get(
                                "obligation",
                                ""
                            )
                        )}
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            render_citations(
                item.get(
                    "document_citations",
                    []
                ),
                []
            )


# ============================================================
# FOOTER DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

        ⚠️ <b>Legal Disclaimer:</b>
        DocuMind is an AI-powered informational and educational
        tool. It does not provide professional legal advice,
        establish an attorney-client relationship, or guarantee
        that a document is valid, invalid, enforceable or
        unenforceable. AI-generated risk classifications are
        indicators for further review, not legal conclusions.

        <br><br>

        Pakistani laws can differ by jurisdiction, province,
        subject matter and subsequent amendments. Always verify
        important legal questions against the current official
        legislation/Gazette and consult a qualified Pakistani
        lawyer when appropriate.

    </div>
    """,
    unsafe_allow_html=True
)
