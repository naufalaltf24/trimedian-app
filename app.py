import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
from PIL import Image 

# =========================
# PAGE CONFIG
# =========================
try:
    icon_image = Image.open("figure/logo.png") 
except FileNotFoundError:
    icon_image = "🎯"

st.set_page_config(
    page_title="Recruitment Offer Acceptance Predictor",
    page_icon=icon_image,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .feature-value {
        background-color: #e9ecef;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Load Model & Artifacts
# =========================
@st.cache_resource
def load_artifacts():
    """Load model, encoders, scaler, and metadata"""
    try:
        model = joblib.load('model_artifacts/recruitment_model.joblib')
        encoders = joblib.load('model_artifacts/label_encoders.joblib')
        scaler = joblib.load('model_artifacts/scaler.joblib')
        metadata = joblib.load('model_artifacts/metadata.joblib')
        return model, encoders, scaler, metadata
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {str(e)}")
        st.info("Pastikan file berikut ada di folder 'model_artifacts/':")
        st.code("""
        - recruitment_model.joblib
        - label_encoders.joblib
        - scaler.joblib
        - metadata.joblib
        """)
        st.stop()

model, encoders, scaler, metadata = load_artifacts()

# =========================
# Department -> Job Title Filter (ANTI-GAGAL)
# =========================
# Tujuan:
# - Saat user memilih 1 department, opsi Job Title hanya menampilkan title yang memang ada di department tsb.
# - Tidak error walaupun mapping belum tersedia (fallback: tampilkan semua job title).
# - Mapping akan dicari dari:
#   1) metadata.joblib (jika di dalamnya ada mapping/pairs)
#   2) file dataset lokal (CSV/XLSX/Parquet) yang mengandung kolom department & job_title (atau variasinya)

import os
import re
from pathlib import Path

def _norm_str(x) -> str:
    """Normalize string for robust matching."""
    if x is None:
        return ""
    return str(x).strip().lower()

def _norm_col(col: str) -> str:
    """Normalize column names (department / job_title detection)."""
    col = str(col).strip().lower()
    col = re.sub(r"[\s\-\/]+", "_", col)
    col = re.sub(r"[^a-z0-9_]+", "", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col

_DEPT_COL_CANDIDATES = [
    "department", "dept", "division", "function", "team", "org", "organisation", "organization",
    "business_unit", "businessunit", "unit"
]
_TITLE_COL_CANDIDATES = [
    "job_title", "jobtitle", "title", "position", "role", "job_role", "jobrole", "job_position", "jobposition"
]

def _detect_pair_columns(df: pd.DataFrame):
    """Detect department & job title column names from a dataframe."""
    # Note: dataframe boleh kosong (0 baris) — kita hanya butuh kolomnya.
    if df is None or getattr(df, "columns", None) is None or len(df.columns) == 0:
        return None, None

    norm_to_actual = {_norm_col(c): c for c in df.columns}

    dept_col = None
    for cand in _DEPT_COL_CANDIDATES:
        if cand in norm_to_actual:
            dept_col = norm_to_actual[cand]
            break

    title_col = None
    for cand in _TITLE_COL_CANDIDATES:
        if cand in norm_to_actual:
            title_col = norm_to_actual[cand]
            break

    return dept_col, title_col

def _build_dept_job_map_from_df(
    df_pairs: pd.DataFrame,
    dept_norm_to_label: dict,
    title_norm_to_label: dict
) -> dict:
    """Build mapping {department_label_in_encoder: [job_title_label_in_encoder, ...]} from a dataframe."""
    if df_pairs is None or df_pairs.empty:
        return {}

    dept_col, title_col = _detect_pair_columns(df_pairs)
    if not dept_col or not title_col:
        # Maybe it is already standardized
        if {"department", "job_title"}.issubset(df_pairs.columns):
            dept_col, title_col = "department", "job_title"
        else:
            return {}

    tmp = df_pairs[[dept_col, title_col]].copy()
    tmp.columns = ["department", "job_title"]
    tmp["department"] = tmp["department"].map(_norm_str)
    tmp["job_title"] = tmp["job_title"].map(_norm_str)
    tmp = tmp.dropna()

    mapping = {}
    for dept_norm, grp in tmp.groupby("department", dropna=True):
        if dept_norm not in dept_norm_to_label:
            continue
        dept_label = dept_norm_to_label[dept_norm]

        titles_norm = grp["job_title"].dropna().unique().tolist()
        titles = [
            title_norm_to_label[t]
            for t in titles_norm
            if t in title_norm_to_label
        ]
        if titles:
            mapping[dept_label] = sorted(set(titles))

    return mapping

def _try_extract_from_metadata(dept_norm_to_label: dict, title_norm_to_label: dict):
    """Try to extract mapping from metadata.joblib (supports many possible structures)."""
    if metadata is None:
        return {}, None

    # Helper to read dict-like or attribute-like metadata
    def _meta_get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # 1) Direct mapping dict: {department: [job_title,...]}
    possible_map_keys = [
        "dept_job_title_map",
        "department_job_title_map",
        "job_titles_by_department",
        "department_to_job_titles",
        "department_job_titles",
        "job_title_by_department",
        "dept_to_job_titles",
    ]
    for k in possible_map_keys:
        m = _meta_get(metadata, k, None)
        if isinstance(m, dict) and m:
            try:
                records = []
                for dept_k, titles in m.items():
                    if isinstance(titles, (list, tuple, set, pd.Series, np.ndarray)):
                        for t in list(titles):
                            records.append({"department": dept_k, "job_title": t})
                    else:
                        records.append({"department": dept_k, "job_title": titles})
                df_pairs = pd.DataFrame(records)
                mapping = _build_dept_job_map_from_df(df_pairs, dept_norm_to_label, title_norm_to_label)
                if mapping:
                    return mapping, f"metadata['{k}']"
            except Exception:
                pass

    # 2) DataFrame-like object stored in metadata (dict of lists / list of dicts / df)
    possible_df_keys = ["data", "train_data", "training_data", "df", "dataset", "raw_data", "pairs_df"]
    for k in possible_df_keys:
        d = _meta_get(metadata, k, None)
        if d is None:
            continue
        try:
            df_pairs = pd.DataFrame(d)
            mapping = _build_dept_job_map_from_df(df_pairs, dept_norm_to_label, title_norm_to_label)
            if mapping:
                return mapping, f"metadata['{k}']"
        except Exception:
            pass

    # 3) Pairs list/records in metadata
    possible_pairs_keys = ["dept_job_title_pairs", "department_job_title_pairs", "pairs", "pair_records"]
    for k in possible_pairs_keys:
        d = _meta_get(metadata, k, None)
        if d is None:
            continue
        try:
            df_pairs = pd.DataFrame(d)
            mapping = _build_dept_job_map_from_df(df_pairs, dept_norm_to_label, title_norm_to_label)
            if mapping:
                return mapping, f"metadata['{k}']"
        except Exception:
            pass

    return {}, None

def _try_extract_from_local_files(dept_norm_to_label: dict, title_norm_to_label: dict):
    """Try to find a local dataset file that contains department & job_title columns and build mapping."""
    # Base directory of the app (works on streamlit)
    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

    # Candidate folders (common in DS projects)
    candidate_dirs = [
        base_dir,
        base_dir / "data",
        base_dir / "dataset",
        base_dir / "datasets",
        base_dir / "input",
        base_dir / "assets",
        base_dir / "model_artifacts",
    ]

    patterns = ["*.csv", "*.xlsx", "*.xls", "*.parquet"]

    # Collect candidates
    files = []
    for d in candidate_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for pat in patterns:
            files.extend(list(d.glob(pat)))

    if not files:
        return {}, None

    # Prefer files with these keywords in filename
    preferred_keywords = ["recruit", "offer", "accept", "train", "dataset", "data", "clean", "hr"]
    def score(p: Path):
        name = p.name.lower()
        return 0 if any(k in name for k in preferred_keywords) else 1

    files = sorted(files, key=score)

    # Try each file until we find usable cols
    for p in files:
        try:
            if p.suffix.lower() == ".csv":
                # read header only first
                cols = pd.read_csv(p, nrows=0).columns.tolist()
                tmp_df = pd.DataFrame(columns=cols)
                dept_col, title_col = _detect_pair_columns(tmp_df)
                if not dept_col or not title_col:
                    continue
                df_pairs = pd.read_csv(p, usecols=[dept_col, title_col], dtype=str)
            elif p.suffix.lower() in [".xlsx", ".xls"]:
                df_full = pd.read_excel(p, dtype=str)
                dept_col, title_col = _detect_pair_columns(df_full)
                if not dept_col or not title_col:
                    continue
                df_pairs = df_full[[dept_col, title_col]].copy()
            elif p.suffix.lower() == ".parquet":
                df_full = pd.read_parquet(p)
                dept_col, title_col = _detect_pair_columns(df_full)
                if not dept_col or not title_col:
                    continue
                df_pairs = df_full[[dept_col, title_col]].copy()
            else:
                continue

            mapping = _build_dept_job_map_from_df(df_pairs, dept_norm_to_label, title_norm_to_label)
            if mapping:
                return mapping, str(p)
        except Exception:
            continue

    return {}, None

@st.cache_resource(show_spinner=False)
def _load_dept_jobtitle_map():
    """Load mapping and return (mapping_dict, source_info). Cached for performance."""
    dept_classes = list(encoders["department"].classes_)
    title_classes = list(encoders["job_title"].classes_)

    dept_norm_to_label = {_norm_str(d): d for d in dept_classes}
    title_norm_to_label = {_norm_str(t): t for t in title_classes}

    # 1) metadata
    mapping, source = _try_extract_from_metadata(dept_norm_to_label, title_norm_to_label)
    if mapping:
        return mapping, f"metadata ({source})"

    # 2) local files
    mapping, source = _try_extract_from_local_files(dept_norm_to_label, title_norm_to_label)
    if mapping:
        return mapping, f"local file ({source})"

    # 3) fallback
    return {}, "fallback (no mapping found)"

DEPT_JOBTITLE_MAP, DEPT_JOBTITLE_MAP_SOURCE = _load_dept_jobtitle_map()

def get_job_title_options(selected_department: str):
    """Return (job_title_options, source_flag) for selected department.

    - If mapping exists: return filtered titles for that department.
    - If mapping missing / department not found: return all job titles (so app never crash).
    """
    all_titles = sorted(list(encoders["job_title"].classes_))

    # Allow optional override from session_state (if you want to set it elsewhere)
    override = st.session_state.get("dept_jobtitle_map_override")
    mapping = override if isinstance(override, dict) and override else DEPT_JOBTITLE_MAP

    # Direct key lookup (most common)
    if selected_department in mapping and mapping[selected_department]:
        return sorted(list(mapping[selected_department])), "filtered"

    # Try normalized key matching
    dept_norm = _norm_str(selected_department)
    for k, v in mapping.items():
        if _norm_str(k) == dept_norm and v:
            return sorted(list(v)), "filtered"

    return all_titles, "fallback_all"


# =========================
# Acceptance Rate Heuristic
# =========================
def estimate_acceptance_rate(source, time_to_hire_days, cost_per_hire):
    """
    Heuristic untuk estimasi acceptance rate
    """
    rate = 0.65  # base expected acceptance

    # Source effect
    src = str(source).strip().lower()
    if "referral" in src:
        rate += 0.10
    elif "linkedin" in src:
        rate += 0.03
    else:
        rate -= 0.05

    # Time pressure effect
    if time_to_hire_days < 30:
        rate += 0.05
    elif time_to_hire_days > 60:
        rate -= 0.05

    # Cost pressure effect
    if cost_per_hire < 3000:
        rate += 0.03
    elif cost_per_hire > 6000:
        rate -= 0.03

    return float(np.clip(rate, 0.3, 0.9))

# =========================
# Feature Engineering - SINGLE
# =========================
def prepare_input_data(user_inputs, encoders, scaler):
    """
    Persiapan data untuk Single Prediction
    """
    df = pd.DataFrame([user_inputs])

    # 1. Feature Engineering
    df['applicant_pressure_index'] = df['num_applicants'] / df['time_to_hire_days']
    df['cost_efficiency_daily'] = df['cost_per_hire'] / df['time_to_hire_days']
    df['cost_per_applicant'] = df['cost_per_hire'] / df['num_applicants']
    df['hire_days_per_applicant'] = df['time_to_hire_days'] / df['num_applicants']
    
    # Difficulty index
    df['difficulty_index'] = df['num_applicants'] * df['cost_per_hire'] * df['time_to_hire_days']
    df['difficulty_index_log'] = np.log(df['difficulty_index'])

    # Acceptance pressures
    df['acceptance_cost_pressure'] = df['cost_per_hire'] * (1 - df['offer_acceptance_rate'])
    df['acceptance_time_pressure'] = df['time_to_hire_days'] * (1 - df['offer_acceptance_rate'])

    # 2. Categorical mappings
    time_map = {'Fast': 0, 'Medium': 1, 'Slow': 2}
    cost_map = {'Low Cost': 0, 'Medium Cost': 1, 'High Cost': 2}

    df['time_to_hire_category'] = time_map[user_inputs['time_to_hire_category']]
    df['cost_bucket'] = cost_map[user_inputs['cost_bucket']]

    # 3. Label encoding
    for col in ['department', 'job_title', 'source']:
        df[col] = df[col].str.strip().str.lower()
        df[col] = encoders[col].transform(df[col])

    # 4. Select features
    feature_cols = [
        'department', 'job_title', 'num_applicants', 'source',
        'time_to_hire_category', 'cost_bucket',
        'applicant_pressure_index', 'cost_efficiency_daily',
        'cost_per_applicant', 'hire_days_per_applicant',
        'difficulty_index_log',
        'acceptance_cost_pressure', 'acceptance_time_pressure'
    ]

    X = df[feature_cols].copy()

    # 5. Scaling
    num_cols_to_scale = [
        "num_applicants", "applicant_pressure_index", "cost_efficiency_daily",
        "cost_per_applicant", "hire_days_per_applicant", "difficulty_index_log",
        "acceptance_cost_pressure", "acceptance_time_pressure"
    ]
    
    X[num_cols_to_scale] = scaler.transform(X[num_cols_to_scale])

    return X, df

# =========================
# Feature Engineering - BATCH
# =========================
def prepare_batch_data(df_input, encoders, scaler):
    """
    Memproses DataFrame input batch agar siap untuk prediksi
    """
    df = df_input.copy()
    
    # 0. Basic Validation & Pre-calculation
    # Menghitung offer_acceptance_rate heuristic untuk setiap baris
    df['offer_acceptance_rate'] = df.apply(
        lambda x: estimate_acceptance_rate(x['source'], x['time_to_hire_days'], x['cost_per_hire']), 
        axis=1
    )
    
    # Generate categories
    df['time_to_hire_category_label'] = df['time_to_hire_days'].apply(get_time_category)
    df['cost_bucket_label'] = df['cost_per_hire'].apply(get_cost_category)

    # 1. Feature Engineering
    df['applicant_pressure_index'] = df['num_applicants'] / df['time_to_hire_days']
    df['cost_efficiency_daily'] = df['cost_per_hire'] / df['time_to_hire_days']
    df['cost_per_applicant'] = df['cost_per_hire'] / df['num_applicants']
    df['hire_days_per_applicant'] = df['time_to_hire_days'] / df['num_applicants']
    
    df['difficulty_index'] = df['num_applicants'] * df['cost_per_hire'] * df['time_to_hire_days']
    df['difficulty_index_log'] = np.log(df['difficulty_index'])

    df['acceptance_cost_pressure'] = df['cost_per_hire'] * (1 - df['offer_acceptance_rate'])
    df['acceptance_time_pressure'] = df['time_to_hire_days'] * (1 - df['offer_acceptance_rate'])

    # 2. Categorical Mappings
    time_map = {'Fast': 0, 'Medium': 1, 'Slow': 2}
    cost_map = {'Low Cost': 0, 'Medium Cost': 1, 'High Cost': 2}
    
    df['time_to_hire_category'] = df['time_to_hire_category_label'].map(time_map)
    df['cost_bucket'] = df['cost_bucket_label'].map(cost_map)

    # 3. Label Encoding (Safe Transform)
    # Kita menggunakan loop untuk transformasi yang aman (handle unseen labels jika perlu)
    for col in ['department', 'job_title', 'source']:
        # Pastikan input format string dan lowercase
        df[col + '_encoded'] = df[col].astype(str).str.strip().str.lower()
        
        # Check if values exist in encoder, otherwise use a default or handle error
        known_labels = set(encoders[col].classes_)
        
        # Simple transform assuming valid input from template
        # Dalam produksi, sebaiknya handle 'unknown' values
        df[col + '_encoded'] = df[col + '_encoded'].map(
            lambda x: encoders[col].transform([x])[0] if x in known_labels else encoders[col].transform([encoders[col].classes_[0]])[0]
        )

    # 4. Prepare X matrix
    # Mapping encoded columns back to expected feature names
    X = pd.DataFrame()
    X['department'] = df['department_encoded']
    X['job_title'] = df['job_title_encoded']
    X['num_applicants'] = df['num_applicants']
    X['source'] = df['source_encoded']
    X['time_to_hire_category'] = df['time_to_hire_category']
    X['cost_bucket'] = df['cost_bucket']
    X['applicant_pressure_index'] = df['applicant_pressure_index']
    X['cost_efficiency_daily'] = df['cost_efficiency_daily']
    X['cost_per_applicant'] = df['cost_per_applicant']
    X['hire_days_per_applicant'] = df['hire_days_per_applicant']
    X['difficulty_index_log'] = df['difficulty_index_log']
    X['acceptance_cost_pressure'] = df['acceptance_cost_pressure']
    X['acceptance_time_pressure'] = df['acceptance_time_pressure']

    # 5. Scaling
    num_cols_to_scale = [
        "num_applicants", "applicant_pressure_index", "cost_efficiency_daily",
        "cost_per_applicant", "hire_days_per_applicant", "difficulty_index_log",
        "acceptance_cost_pressure", "acceptance_time_pressure"
    ]
    
    X[num_cols_to_scale] = scaler.transform(X[num_cols_to_scale])
    
    return X, df

# =========================
# Helper Functions
# =========================
def get_time_category(days):
    if days < 30:
        return "Fast"
    elif days < 60:
        return "Medium"
    return "Slow"

def get_cost_category(cost):
    if cost < 3000:
        return "Low Cost"
    elif cost < 6000:
        return "Medium Cost"
    return "High Cost"

def get_prediction_explanation(pred, prob, user_inputs):
    """Generate detailed explanation for the prediction"""
    class_names = ['Likely Reject', 'Uncertain', 'Likely Accept']
    predicted_class = class_names[pred]
    confidence = prob[pred]
    
    explanations = []
    
    # Source analysis
    source = user_inputs['source'].lower()
    if 'referral' in source:
        explanations.append("✅ **Referral** source typically has higher acceptance rates")
    elif 'linkedin' in source:
        explanations.append("💼 **LinkedIn** source shows moderate acceptance rates")
    else:
        explanations.append("📊 **Job Portal** source may have lower acceptance rates")
    
    # Time analysis
    time_cat = user_inputs['time_to_hire_category']
    if time_cat == 'Fast':
        explanations.append("⚡ **Fast hiring process** (<30 days) positively impacts acceptance")
    elif time_cat == 'Slow':
        explanations.append("⏰ **Slow hiring process** (>60 days) may reduce acceptance rate")
    
    # Cost analysis
    cost_cat = user_inputs['cost_bucket']
    if cost_cat == 'High Cost':
        explanations.append("💰 **High cost per hire** indicates competitive offer package")
    elif cost_cat == 'Low Cost':
        explanations.append("💵 **Low cost per hire** may indicate less competitive package")
    
    # Applicant pressure
    pressure = user_inputs['num_applicants'] / user_inputs['time_to_hire_days']
    if pressure > 5:
        explanations.append("⚠️ **High applicant pressure** - many candidates in short time")
    elif pressure < 2:
        explanations.append("✅ **Low applicant pressure** - selective hiring process")
    
    return predicted_class, confidence, explanations


def generate_what_if_scenarios(user_inputs, encoders, scaler, model):
    """Generate exploratory what-if scenarios (non-goal-seeking).

    Dipakai terutama ketika hasil sudah "Likely Accept" untuk melihat sensitivitas strategi.
    """
    scenarios = []

    # Scenario 1: Faster hiring
    if user_inputs.get('time_to_hire_days', 0) > 30:
        scenario1 = user_inputs.copy()
        scenario1['time_to_hire_days'] = max(7, int(user_inputs['time_to_hire_days'] - 15))
        scenario1['time_to_hire_category'] = get_time_category(scenario1['time_to_hire_days'])
        scenario1['offer_acceptance_rate'] = estimate_acceptance_rate(
            scenario1['source'], scenario1['time_to_hire_days'], scenario1['cost_per_hire']
        )
        X1, _ = prepare_input_data(scenario1, encoders, scaler)
        pred1 = int(model.predict(X1)[0])
        prob1 = model.predict_proba(X1)[0]
        scenarios.append({
            'name': f"⚡ Faster Hiring ({scenario1['time_to_hire_days']} days)",
            'prediction': pred1,
            'confidence': float(prob1[pred1]),
            'accept_prob': float(prob1[2]) if len(prob1) > 2 else np.nan,
            'change': f"-{int(user_inputs['time_to_hire_days'] - scenario1['time_to_hire_days'])} days"
        })

    # Scenario 2: Higher cost (better package)
    if float(user_inputs.get('cost_per_hire', 0)) < 8000:
        scenario2 = user_inputs.copy()
        scenario2['cost_per_hire'] = float(min(10000, float(user_inputs['cost_per_hire']) * 1.3))
        scenario2['cost_bucket'] = get_cost_category(scenario2['cost_per_hire'])
        scenario2['offer_acceptance_rate'] = estimate_acceptance_rate(
            scenario2['source'], scenario2['time_to_hire_days'], scenario2['cost_per_hire']
        )
        X2, _ = prepare_input_data(scenario2, encoders, scaler)
        pred2 = int(model.predict(X2)[0])
        prob2 = model.predict_proba(X2)[0]
        scenarios.append({
            'name': f"💰 Better Package (${scenario2['cost_per_hire']:.0f})",
            'prediction': pred2,
            'confidence': float(prob2[pred2]),
            'accept_prob': float(prob2[2]) if len(prob2) > 2 else np.nan,
            'change': f"+${scenario2['cost_per_hire'] - float(user_inputs['cost_per_hire']):.0f}"
        })

    # Scenario 3: Referral source (if not already)
    try:
        cur_source = str(user_inputs.get('source', '')).lower()
    except Exception:
        cur_source = ''
    if 'referral' not in cur_source and 'source' in encoders:
        scenario3 = user_inputs.copy()
        referral_options = [c for c in encoders['source'].classes_ if 'referral' in str(c).lower()]
        if referral_options:
            scenario3['source'] = referral_options[0]
            scenario3['offer_acceptance_rate'] = estimate_acceptance_rate(
                scenario3['source'], scenario3['time_to_hire_days'], scenario3['cost_per_hire']
            )
            X3, _ = prepare_input_data(scenario3, encoders, scaler)
            pred3 = int(model.predict(X3)[0])
            prob3 = model.predict_proba(X3)[0]
            scenarios.append({
                'name': "🤝 Switch to Referral",
                'prediction': pred3,
                'confidence': float(prob3[pred3]),
                'accept_prob': float(prob3[2]) if len(prob3) > 2 else np.nan,
                'change': "Source change"
            })

    return scenarios


def _apply_scenario_changes(base_inputs, source=None, time_to_hire_days=None, cost_per_hire=None):
    """Copy base inputs and apply changes safely + recompute derived fields."""
    s = dict(base_inputs)

    if source is not None:
        s["source"] = source
    if time_to_hire_days is not None:
        # keep within reasonable UI range
        try:
            s["time_to_hire_days"] = int(max(1, min(120, round(float(time_to_hire_days)))))
        except Exception:
            pass
    if cost_per_hire is not None:
        try:
            s["cost_per_hire"] = float(max(100.0, min(15000.0, float(cost_per_hire))))
        except Exception:
            pass

    # Derived fields (wajib untuk feature engineering)
    s["time_to_hire_category"] = get_time_category(s["time_to_hire_days"])
    s["cost_bucket"] = get_cost_category(s["cost_per_hire"])
    s["offer_acceptance_rate"] = estimate_acceptance_rate(
        s["source"], s["time_to_hire_days"], s["cost_per_hire"]
    )
    return s


def _predict_scenario(user_inputs, encoders, scaler, model):
    """Return (pred_int, prob_array). Safe wrapper."""
    X, _ = prepare_input_data(user_inputs, encoders, scaler)
    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0]
    return pred, prob


def _candidate_time_values(base_time):
    """Candidate time-to-hire values (bias: faster is better)."""
    try:
        t = int(round(float(base_time)))
    except Exception:
        t = 30

    # Only suggest equal or faster (<= baseline), because it's actionable and usually preferred.
    candidates = {t}
    for delta in (5, 10, 15, 20, 30):
        candidates.add(max(7, t - delta))

    # Add a few meaningful fixed points if below baseline
    for fixed in (30, 25, 20, 15, 10):
        if fixed <= t:
            candidates.add(fixed)

    # Sort ascending (fastest first) but keep list small
    vals = sorted(candidates)
    # Keep up to 5 values, biased to faster options (smallest)
    return vals[:5]


def _candidate_cost_values(base_cost):
    """Candidate cost-per-hire values (bias: higher budget can improve acceptance)."""
    try:
        c = float(base_cost)
    except Exception:
        c = 3000.0

    candidates = {c}
    for mult in (1.10, 1.25, 1.50):
        candidates.add(min(15000.0, c * mult))

    # Add some common benchmarks above baseline
    for fixed in (5000.0, 8000.0, 10000.0, 12000.0, 15000.0):
        if fixed >= c:
            candidates.add(fixed)

    bigger = sorted([x for x in candidates if x >= c])
    # Baseline + up to 4 next bigger options
    out = [bigger[0]]
    out.extend(bigger[1:5])
    # unique & sorted
    out = sorted(set(out))
    return out


def generate_goal_seeking_scenarios(user_inputs, encoders, scaler, model, target_class=2, max_scenarios=3):
    """Goal-seeking what-if scenarios.

    Jika prediksi awal bukan "Likely Accept", kita cari beberapa perubahan yang paling 'masuk akal'
    untuk mendorong model ke target_class (default: 2 = Likely Accept).

    Output:
      list of scenarios, masing-masing berisi:
        - name, prediction, confidence (of predicted class)
        - accept_prob (probability target class)
        - delta_accept_pp (perubahan dalam percentage points vs baseline)
        - change (ringkasan perubahan)
        - meets_target (bool)
    """
    # Baseline
    base_pred, base_prob = _predict_scenario(user_inputs, encoders, scaler, model)
    base_accept_prob = float(base_prob[target_class]) if len(base_prob) > target_class else float("nan")

    base_time = int(user_inputs.get("time_to_hire_days", 30))
    base_cost = float(user_inputs.get("cost_per_hire", 3000.0))
    base_source = user_inputs.get("source", "")

    time_candidates = _candidate_time_values(base_time)
    cost_candidates = _candidate_cost_values(base_cost)

    # Rank sources by their target probability (with baseline time & cost) and take top-N.
    sources = list(encoders["source"].classes_) if "source" in encoders else [base_source]
    src_rank = []
    for src in sources:
        try:
            s_tmp = _apply_scenario_changes(user_inputs, source=src, time_to_hire_days=base_time, cost_per_hire=base_cost)
            pred_s, prob_s = _predict_scenario(s_tmp, encoders, scaler, model)
            accept_p = float(prob_s[target_class]) if len(prob_s) > target_class else 0.0
            src_rank.append((accept_p, src))
        except Exception:
            continue

    src_rank.sort(key=lambda x: x[0], reverse=True)
    top_sources = [s for _, s in src_rank[:5]] if src_rank else [base_source]
    if base_source and base_source not in top_sources:
        top_sources = [base_source] + top_sources
    # de-duplicate
    seen = set()
    top_sources = [s for s in top_sources if not (s in seen or seen.add(s))]

    # Evaluate grid of candidates
    candidates = []
    seen_keys = set()

    def _change_cost_metric(t, c, src):
        # lower = smaller change
        t0 = max(1, base_time)
        c0 = max(1.0, base_cost)
        return abs(t0 - t) / t0 + abs(c0 - c) / c0 + (0.25 if src != base_source else 0.0)

    for src in top_sources:
        for t in time_candidates:
            for c in cost_candidates:
                # Skip baseline
                if src == base_source and int(t) == int(base_time) and float(c) == float(base_cost):
                    continue

                key = (str(src), int(t), round(float(c), 2))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                try:
                    s = _apply_scenario_changes(user_inputs, source=src, time_to_hire_days=t, cost_per_hire=c)
                    pred, prob = _predict_scenario(s, encoders, scaler, model)

                    accept_p = float(prob[target_class]) if len(prob) > target_class else float("nan")
                    pred_conf = float(prob[pred]) if len(prob) > pred else float("nan")

                    meets_target = (pred == target_class)

                    # Build a readable change string
                    changes = []
                    if src != base_source:
                        changes.append(f"Source: {base_source} → {src}")
                    if int(t) != int(base_time):
                        changes.append(f"Time: {base_time}d → {int(t)}d")
                    if float(c) != float(base_cost):
                        changes.append(f"Cost: ${base_cost:.0f} → ${float(c):.0f}")

                    change_str = " | ".join(changes) if changes else "No change"
                    name = "🎯 " + (" + ".join([ch.split(':')[0] for ch in changes]) if changes else "Alternative")

                    candidates.append({
                        "name": name,
                        "prediction": pred,
                        "confidence": pred_conf,
                        "accept_prob": accept_p,
                        "delta_accept_pp": (accept_p - base_accept_prob) * 100.0 if (accept_p == accept_p and base_accept_prob == base_accept_prob) else float("nan"),
                        "change": change_str,
                        "meets_target": meets_target,
                        "change_cost": _change_cost_metric(int(t), float(c), src),
                    })
                except Exception:
                    continue

    if not candidates:
        return []

    # Sorting: prefer those that meet target, then highest accept_prob, then smallest change
    candidates.sort(
        key=lambda s: (
            0 if s["meets_target"] else 1,
            - (s["accept_prob"] if s["accept_prob"] == s["accept_prob"] else -1.0),
            s["change_cost"],
        )
    )

    # If none meets target, still return best "closest" improvements (by accept_prob, then smallest change)
    if not any(s["meets_target"] for s in candidates):
        candidates.sort(
            key=lambda s: (
                - (s["accept_prob"] if s["accept_prob"] == s["accept_prob"] else -1.0),
                s["change_cost"],
            )
        )

    # Trim
    return candidates[:max_scenarios]


def render_scenario_cards(scenarios, base_accept_prob=None):
    """Small UI helper: render scenario cards consistently."""
    if not scenarios:
        st.info("Tidak ada skenario yang bisa dihitung. Coba ubah input atau pastikan artifacts lengkap.")
        return

    class_names = ['Likely Reject', 'Uncertain', 'Likely Accept']

    cols = st.columns(len(scenarios))
    for col, scenario in zip(cols, scenarios):
        with col:
            scenario_class = class_names[scenario['prediction']] if scenario.get('prediction') in [0, 1, 2] else str(scenario.get('prediction'))
            pred_idx = scenario.get('prediction', 0)
            color = '#28a745' if pred_idx == 2 else '#ffc107' if pred_idx == 1 else '#dc3545'

            accept_prob = scenario.get("accept_prob", np.nan)
            delta_pp = scenario.get("delta_accept_pp", np.nan)

            extra_lines = []
            if accept_prob == accept_prob:
                extra_lines.append(f"Likely Accept: <b>{accept_prob*100:.1f}%</b>")
            if delta_pp == delta_pp:
                sign = "+" if delta_pp >= 0 else ""
                extra_lines.append(f"Δ vs base: <b>{sign}{delta_pp:.1f} pp</b>")

            extra_html = "<br/>".join(extra_lines)

            badge = ""
            if scenario.get("meets_target", False):
                badge = "<span style='font-size:0.8rem; background:#28a745; color:white; padding:0.15rem 0.5rem; border-radius:999px; margin-left:0.5rem;'>TARGET</span>"

            st.markdown(f"""
            <div style="background-color: {color}22; border: 2px solid {color}; border-radius: 10px; padding: 1rem;">
                <h4 style="color: {color}; margin: 0;">{scenario['name']}{badge}</h4>
                <p style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">{scenario_class}</p>
                <p style="font-size: 1.0rem; margin: 0;">Confidence: {scenario.get('confidence', 0)*100:.1f}%</p>
                <p style="font-size: 0.95rem; margin: 0.5rem 0 0 0;">{extra_html}</p>
                <p style="font-size: 0.85rem; color: #6c757d; margin: 0.75rem 0 0 0;">{scenario.get('change','')}</p>
            </div>
            """, unsafe_allow_html=True)
# =========================
# MAIN UI
# =========================
st.markdown('<h1 class="main-header">🎯 Recruitment Offer Acceptance Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Candidate Acceptance Prediction | 95% Accuracy Model</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])

# =========================
# TAB 1: SINGLE PREDICTION
# =========================
with tab1:
    # Sidebar is primarily for Single Prediction inputs
    with st.sidebar:
        st.image("figure/logo.png", width=80)
        st.title("Parameters Input")
        st.caption("Use this for Single Predictions")
        st.markdown("---")
        
        # Department
        department = st.selectbox(
            "🏢 Department",
            options=sorted(encoders['department'].classes_),
            key="department",
            help="Select the hiring department"
        )
        
        # Job Title (Auto-filtered by selected Department) — ANTI-GAGAL
        job_title_options, _jt_flag = get_job_title_options(department)

        # Pastikan options tidak kosong & tidak bikin Streamlit error saat department berubah
        if not job_title_options:
            job_title_options = sorted(list(encoders["job_title"].classes_))

        # Jika sebelumnya user memilih job title yang tidak ada di department baru,
        # reset ke opsi pertama agar tidak error.
        if "job_title" not in st.session_state or st.session_state["job_title"] not in job_title_options:
            st.session_state["job_title"] = job_title_options[0]

        job_title = st.selectbox(
            "💼 Job Title",
            options=job_title_options,
            key="job_title",
            help="Select the job position (filtered by Department)"
        )
        
        # Source
        source = st.selectbox(
            "📢 Recruitment Source",
            options=sorted(encoders['source'].classes_),
            help="Where the candidate was sourced from"
        )
        
        st.markdown("---")
        
        # Numerical inputs
        num_applicants = st.number_input(
            "👥 Number of Applicants",
            min_value=1,
            max_value=1000,
            value=50,
            help="Total number of applicants for this position"
        )
        
        time_to_hire_days = st.number_input(
            "⏱️ Time to Hire (days)",
            min_value=1,
            max_value=180,
            value=30,
            help="Expected time to complete hiring process"
        )
        
        cost_per_hire = st.number_input(
            "💵 Cost per Hire ($)",
            min_value=100.0,
            max_value=20000.0,
            value=3000.0,
            step=100.0,
            help="Total recruitment cost per hire"
        )
        
        st.markdown("---")
        predict_button = st.button("🚀 Predict Acceptance", use_container_width=True, type="primary")

    if predict_button:
        with st.spinner("🔮 Analyzing recruitment scenario..."):
            # Prepare data
            expected_rate = estimate_acceptance_rate(source, time_to_hire_days, cost_per_hire)
            
            user_inputs = {
                "department": department,
                "job_title": job_title,
                "source": source,
                "num_applicants": num_applicants,
                "time_to_hire_days": time_to_hire_days,
                "cost_per_hire": cost_per_hire,
                "time_to_hire_category": get_time_category(time_to_hire_days),
                "cost_bucket": get_cost_category(cost_per_hire),
                "offer_acceptance_rate": expected_rate
            }

            # Make prediction
            X, df_features = prepare_input_data(user_inputs, encoders, scaler)
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            
            # Get explanation
            predicted_class, confidence, explanations = get_prediction_explanation(pred, prob, user_inputs)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Prediction</p>
                    <p class="metric-value">{"✅" if pred == 2 else "⚠️" if pred == 1 else "❌"}</p>
                    <p class="metric-label">{predicted_class}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Confidence</p>
                    <p class="metric-value">{confidence*100:.1f}%</p>
                    <p class="metric-label">Model Certainty</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Expected Rate</p>
                    <p class="metric-value">{expected_rate*100:.0f}%</p>
                    <p class="metric-label">Acceptance Heuristic</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Probability Distribution
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Prediction Confidence Distribution")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Likely Reject', 'Uncertain', 'Likely Accept'],
                        y=prob * 100,
                        marker_color=['#dc3545', '#ffc107', '#28a745'],
                        text=[f'{p*100:.1f}%' for p in prob],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    yaxis_title="Probability (%)",
                    height=350,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Key Metrics")
                st.metric("Applicant Pressure", f"{df_features['applicant_pressure_index'].values[0]:.2f}", 
                          help="Applicants per day")
                st.metric("Cost Efficiency", f"${df_features['cost_efficiency_daily'].values[0]:.2f}", 
                          help="Daily cost")
                st.metric("Difficulty Index", f"{df_features['difficulty_index_log'].values[0]:.2f}", 
                          help="Log-scaled complexity")
            
            st.markdown("---")
            
            # Explanation
            st.subheader("💡 Why This Prediction?")
            
            # Choose box color based on prediction
            if pred == 2:
                box_class = "success-box"
            elif pred == 1:
                box_class = "warning-box"
            else:
                box_class = "danger-box"
            
            st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
            st.markdown(f"**Predicted Outcome:** {predicted_class} with **{confidence*100:.1f}%** confidence")
            st.markdown("</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Contributing Factors")
                for exp in explanations:
                    st.markdown(exp)
            
            with col2:
                st.markdown("### 🔍 Feature Insights")
                st.markdown(f"""
                - **Time Category:** {user_inputs['time_to_hire_category']}
                - **Cost Category:** {user_inputs['cost_bucket']}
                - **Acceptance Cost Pressure:** ${df_features['acceptance_cost_pressure'].values[0]:.2f}
                - **Acceptance Time Pressure:** {df_features['acceptance_time_pressure'].values[0]:.2f} days
                """)
            
            st.markdown("---")
            
            
            # Goal-Seeking / What-If Analysis
            if pred != 2:
                st.subheader("🎯 Goal-Seeking What-If: menuju 'Likely Accept'")
                st.markdown(
                    "Karena hasil saat ini **bukan Likely Accept**, aplikasi akan mencoba beberapa perubahan strategi "
                    "yang paling masuk akal untuk mendorong prediksi menjadi **Likely Accept**. "
                    "Ini berbasis model (decision support), bukan jaminan di dunia nyata."
                )

                base_accept_prob = float(prob[2]) if len(prob) > 2 else np.nan
                if base_accept_prob == base_accept_prob:
                    st.caption(f"Baseline Likely Accept probability: {base_accept_prob*100:.1f}%")

                goal_scenarios = generate_goal_seeking_scenarios(
                    user_inputs, encoders, scaler, model, target_class=2, max_scenarios=3
                )

                if goal_scenarios:
                    render_scenario_cards(goal_scenarios, base_accept_prob=base_accept_prob)

                    if not any(s.get("meets_target") for s in goal_scenarios):
                        st.warning(
                            "Dalam rentang perubahan yang dicoba, belum ada skenario yang membuat prediksi menjadi "
                            "**Likely Accept**. Yang ditampilkan adalah opsi dengan peningkatan probabilitas terbesar."
                        )
                else:
                    st.info("Belum bisa membuat skenario rekomendasi. Coba ubah input atau cek kembali artifacts.")

            else:
                st.subheader("🔮 What-If Scenario Analysis")
                st.markdown(
                    "Hasil sudah **Likely Accept**. Berikut uji sensitivitas: apa yang terjadi jika strategi diubah?"
                )

                scenarios = generate_what_if_scenarios(user_inputs, encoders, scaler, model)
                if scenarios:
                    render_scenario_cards(scenarios, base_accept_prob=float(prob[2]) if len(prob) > 2 else np.nan)
                else:
                    st.info("Konfigurasi saat ini sudah cukup optimal — tidak ada skenario eksplorasi yang relevan.")
# Recommendations
            st.subheader("💼 Strategic Recommendations")
            
            if pred == 0:  # Likely Reject
                st.error("""
                **⚠️ High Risk of Rejection - Immediate Action Required:**
                
                1. 🎯 **Improve Offer Package:** Consider increasing compensation or benefits
                2. ⚡ **Speed Up Process:** Reduce time-to-hire to show commitment
                3. 🤝 **Leverage Referrals:** Switch to referral-based sourcing if possible
                4. 📞 **Enhanced Engagement:** Increase candidate touchpoints throughout process
                5. 🔄 **Review Strategy:** Re-evaluate job requirements and market positioning
                """)
                
            elif pred == 1:  # Uncertain
                st.warning("""
                **⚠️ Uncertain Outcome - Optimization Recommended:**
                
                1. 💰 **Enhance Package:** Small improvements to offer could swing outcome
                2. ⏱️ **Optimize Timeline:** Balance speed with thoroughness
                3. 📢 **Communication:** Increase engagement and transparency with candidate
                4. 🎁 **Add Value:** Highlight growth opportunities, culture, benefits
                5. 📊 **Monitor Closely:** Track candidate sentiment throughout process
                """)
                
            else:  # Likely Accept
                st.success("""
                **✅ High Probability of Acceptance - Stay the Course:**
                
                1. ✨ **Maintain Momentum:** Keep the positive candidate experience going
                2. 📋 **Prepare Onboarding:** Start planning for smooth transition
                3. 🤝 **Stay Engaged:** Continue regular communication until offer acceptance
                4. 🎯 **Close Efficiently:** Don't delay - move quickly to secure acceptance
                5. 📈 **Document Success:** Record what worked for future hires
                """)
    else:
        # Welcome screen for Single Tab
        st.info("👈 **Enter recruitment parameters in the sidebar and click 'Predict' to get started**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 95% Accuracy
            Our ML model achieves 95% prediction accuracy, significantly outperforming traditional methods
            """)
        
        with col2:
            st.markdown("""
            ### ⚡ Instant Insights
            Get real-time predictions and actionable recommendations in seconds
            """)
        
        with col3:
            st.markdown("""
            ### 💡 Data-Driven
            Make informed hiring decisions based on historical patterns and advanced analytics
            """)

# =========================
# TAB 2: BATCH PREDICTION
# =========================
with tab2:
    st.header("📂 Batch Prediction")
    st.markdown("Upload file Excel (.xlsx) berisi data kandidat untuk melakukan prediksi massal.")

    # 1. Template Downloader (XLSX)
    with st.expander("ℹ️ Format File Excel (Download Template)"):
        st.markdown("Pastikan file Excel Anda mengikuti struktur berikut:")
        
        # Create dummy template dataframe
        template_data = {
            'department': ['Sales', 'IT', 'Finance'],
            'job_title': ['Sales Manager', 'Developer', 'Analyst'],
            'source': ['LinkedIn', 'Referral', 'Job Portal'],
            'num_applicants': [50, 30, 100],
            'time_to_hire_days': [30, 45, 20],
            'cost_per_hire': [3000, 5000, 2500]
        }
        df_template = pd.DataFrame(template_data)
        
        st.dataframe(df_template, use_container_width=True)
        
        # Convert to Excel for download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_template.to_excel(writer, index=False, sheet_name='Sheet1')
            
        download_data = buffer.getvalue()
        
        # Tambahkan key="template_dl" agar unik
        st.download_button(
            label="📥 Download Template Excel (.xlsx)",
            data=download_data,
            file_name="recruitment_batch_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Klik untuk mengunduh contoh file Excel.",
            key="template_dl_button" 
        )

    # 2. File Uploader (XLSX Only)
    # PERBAIKAN: Tambahkan key="batch_uploader" di sini
    uploaded_file = st.file_uploader(
        "Upload file Excel Anda", 
        type=["xlsx"], 
        key="batch_uploader_key"
    )

    if uploaded_file is not None:
        try:
            # Read Excel
            input_df = pd.read_excel(uploaded_file)
            st.success(f"File berhasil diupload: {input_df.shape[0]} baris data.")
            
            # Preview Input
            st.subheader("Preview Data")
            st.dataframe(input_df.head(), use_container_width=True)
            
            # Validasi Kolom
            required_cols = ['department', 'job_title', 'source', 'num_applicants', 'time_to_hire_days', 'cost_per_hire']
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"❌ Kolom berikut hilang dari file Excel: {', '.join(missing_cols)}")
            else:
                # Tambahkan key juga di tombol proses
                if st.button("🚀 Proses Batch Prediction", type="primary", key="process_batch_btn"):
                    with st.spinner("Memproses prediksi massal..."):
                        # Prepare data for batch
                        X_batch, df_processed = prepare_batch_data(input_df, encoders, scaler)
                        
                        # Predict
                        predictions = model.predict(X_batch)
                        probs = model.predict_proba(X_batch)
                        
                        # Map predictions to labels
                        class_names = {0: 'Likely Reject', 1: 'Uncertain', 2: 'Likely Accept'}
                        
                        # Create Result DataFrame
                        results_df = input_df.copy()
                        results_df['Prediction_Label'] = [class_names[p] for p in predictions]
                        results_df['Confidence_Score'] = [max(probs[i]) for i in range(len(probs))]
                        
                        # Visualization for Batch
                        st.markdown("---")
                        st.subheader("📊 Ringkasan Hasil Batch")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart of predictions
                            pred_counts = results_df['Prediction_Label'].value_counts()
                            fig_pie = px.pie(
                                values=pred_counts.values,
                                names=pred_counts.index,
                                title="Distribusi Prediksi",
                                color=pred_counts.index,
                                color_discrete_map={
                                    'Likely Reject': '#dc3545',
                                    'Uncertain': '#ffc107',
                                    'Likely Accept': '#28a745'
                                }
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                        with col2:
                            # Average metrics
                            avg_conf = results_df['Confidence_Score'].mean()
                            accept_rate = (results_df['Prediction_Label'] == 'Likely Accept').mean()
                            
                            st.metric("Rata-rata Confidence", f"{avg_conf*100:.1f}%")
                            st.metric("Prediksi Acceptance Rate", f"{accept_rate*100:.1f}%")
                        
                        # Detailed Table
                        st.subheader("📋 Detail Hasil")
                        
                        # Color coding function
                        def color_coding(val):
                            color = 'black'
                            if val == 'Likely Accept': color = '#28a745'
                            elif val == 'Uncertain': color = '#ffc107'
                            elif val == 'Likely Reject': color = '#dc3545'
                            return f'color: {color}; font-weight: bold'

                        st.dataframe(
                            results_df.style.map(color_coding, subset=['Prediction_Label'])
                            .format({'Confidence_Score': "{:.2%}", 'cost_per_hire': "${:,.0f}"}),
                            use_container_width=True
                        )
                        
                        # Download Results as XLSX
                        buffer_res = io.BytesIO()
                        with pd.ExcelWriter(buffer_res, engine='xlsxwriter') as writer:
                            results_df.to_excel(writer, index=False, sheet_name='Predictions')
                            
                        download_res = buffer_res.getvalue()

                        # Tambahkan key di tombol download hasil
                        st.download_button(
                            label="📥 Download Hasil Prediksi (.xlsx)",
                            data=download_res,
                            file_name="recruitment_predictions_result.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary",
                            key="download_result_btn"
                        )

        except Exception as e:
            st.error(f"Error memproses file: {str(e)}")
            st.info("Pastikan file yang diupload adalah format Excel (.xlsx) yang valid.")


# Footer (Common)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem 0;">
    <p><strong>Recruitment Offer Acceptance Predictor</strong> | Model Accuracy: 95%</p>
    <p><strong>by TriMedian</strong> | Rakamain Data Science </p>
</div>
""", unsafe_allow_html=True)
