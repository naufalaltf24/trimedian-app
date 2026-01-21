import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
from PIL import Image 

# =========================
# PAGE CONFIG
# =========================
try:
    icon_image = Image.open("figure/logo(2).png") 
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
    """Generate what-if analysis"""
    scenarios = []
    
    # Scenario 1: Faster hiring
    if user_inputs['time_to_hire_days'] > 30:
        scenario1 = user_inputs.copy()
        scenario1['time_to_hire_days'] = max(20, user_inputs['time_to_hire_days'] - 15)
        scenario1['time_to_hire_category'] = get_time_category(scenario1['time_to_hire_days'])
        scenario1['offer_acceptance_rate'] = estimate_acceptance_rate(
            scenario1['source'], scenario1['time_to_hire_days'], scenario1['cost_per_hire']
        )
        X1, _ = prepare_input_data(scenario1, encoders, scaler)
        pred1 = model.predict(X1)[0]
        prob1 = model.predict_proba(X1)[0]
        scenarios.append({
            'name': f"⚡ Faster Hiring ({scenario1['time_to_hire_days']} days)",
            'prediction': pred1,
            'confidence': prob1[pred1],
            'change': f"-{user_inputs['time_to_hire_days'] - scenario1['time_to_hire_days']} days"
        })
    
    # Scenario 2: Higher cost (better package)
    if user_inputs['cost_per_hire'] < 8000:
        scenario2 = user_inputs.copy()
        scenario2['cost_per_hire'] = min(10000, user_inputs['cost_per_hire'] * 1.3)
        scenario2['cost_bucket'] = get_cost_category(scenario2['cost_per_hire'])
        scenario2['offer_acceptance_rate'] = estimate_acceptance_rate(
            scenario2['source'], scenario2['time_to_hire_days'], scenario2['cost_per_hire']
        )
        X2, _ = prepare_input_data(scenario2, encoders, scaler)
        pred2 = model.predict(X2)[0]
        prob2 = model.predict_proba(X2)[0]
        scenarios.append({
            'name': f"💰 Better Package (${scenario2['cost_per_hire']:.0f})",
            'prediction': pred2,
            'confidence': prob2[pred2],
            'change': f"+${scenario2['cost_per_hire'] - user_inputs['cost_per_hire']:.0f}"
        })
    
    # Scenario 3: Referral source (if not already)
    if 'referral' not in user_inputs['source'].lower():
        scenario3 = user_inputs.copy()
        # Find referral in encoder classes
        referral_options = [c for c in encoders['source'].classes_ if 'referral' in c.lower()]
        if referral_options:
            scenario3['source'] = referral_options[0]
            scenario3['offer_acceptance_rate'] = estimate_acceptance_rate(
                scenario3['source'], scenario3['time_to_hire_days'], scenario3['cost_per_hire']
            )
            X3, _ = prepare_input_data(scenario3, encoders, scaler)
            pred3 = model.predict(X3)[0]
            prob3 = model.predict_proba(X3)[0]
            scenarios.append({
                'name': "🤝 Switch to Referral",
                'prediction': pred3,
                'confidence': prob3[pred3],
                'change': "Source change"
            })
    
    return scenarios

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
        st.image("figure/logo(2).png", width=80)
        st.title("Parameters Input")
        st.caption("Use this for Single Predictions")
        st.markdown("---")
        
        # Department
        department = st.selectbox(
            "🏢 Department",
            options=sorted(encoders['department'].classes_),
            help="Select the hiring department"
        )
        
        # Job Title
        job_title = st.selectbox(
            "💼 Job Title",
            options=sorted(encoders['job_title'].classes_),
            help="Select the job position"
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
            max_value=500,
            value=50,
            help="Total number of applicants for this position"
        )
        
        time_to_hire_days = st.number_input(
            "⏱️ Time to Hire (days)",
            min_value=1,
            max_value=120,
            value=30,
            help="Expected time to complete hiring process"
        )
        
        cost_per_hire = st.number_input(
            "💵 Cost per Hire ($)",
            min_value=100.0,
            max_value=15000.0,
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
            
            # What-If Analysis
            st.subheader("🔮 What-If Scenario Analysis")
            st.markdown("See how changes to recruitment strategy might affect the outcome:")
            
            scenarios = generate_what_if_scenarios(user_inputs, encoders, scaler, model)
            
            if scenarios:
                cols = st.columns(len(scenarios))
                class_names = ['Likely Reject', 'Uncertain', 'Likely Accept']
                
                for i, (col, scenario) in enumerate(zip(cols, scenarios)):
                    with col:
                        scenario_class = class_names[scenario['prediction']]
                        color = '#28a745' if scenario['prediction'] == 2 else '#ffc107' if scenario['prediction'] == 1 else '#dc3545'
                        
                        st.markdown(f"""
                        <div style="background-color: {color}22; border: 2px solid {color}; border-radius: 10px; padding: 1rem;">
                            <h4 style="color: {color}; margin: 0;">{scenario['name']}</h4>
                            <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{scenario_class}</p>
                            <p style="font-size: 1.2rem; margin: 0;">{scenario['confidence']*100:.1f}%</p>
                            <p style="font-size: 0.9rem; color: #6c757d; margin: 0.5rem 0 0 0;">{scenario['change']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Current configuration is already optimized!")
            
            st.markdown("---")
            
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
    st.markdown("Upload a CSV file containing multiple candidate records to generate predictions in bulk.")

    # 1. Template Downloader
    with st.expander("ℹ️ How to format your CSV (Download Template)"):
        st.markdown("Please ensure your CSV file follows exactly this structure:")
        
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
        
        # Convert to CSV for download
        csv_buffer = io.BytesIO()
        df_template.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_bytes,
            file_name="recruitment_batch_template.csv",
            mime="text/csv",
            help="Click to download a sample CSV file to fill out."
        )

    # 2. File Uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully: {input_df.shape[0]} rows loaded.")
            
            # Preview Input
            st.subheader("Preview Data")
            st.dataframe(input_df.head(), use_container_width=True)
            
            # Validasi Kolom
            required_cols = ['department', 'job_title', 'source', 'num_applicants', 'time_to_hire_days', 'cost_per_hire']
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns in CSV: {', '.join(missing_cols)}")
            else:
                if st.button("🚀 Process Batch Prediction", type="primary"):
                    with st.spinner("Processing batch predictions..."):
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
                        st.subheader("📊 Batch Results Summary")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart of predictions
                            pred_counts = results_df['Prediction_Label'].value_counts()
                            fig_pie = px.pie(
                                values=pred_counts.values,
                                names=pred_counts.index,
                                title="Prediction Distribution",
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
                            
                            st.metric("Average Confidence", f"{avg_conf*100:.1f}%")
                            st.metric("Predicted Acceptance Rate", f"{accept_rate*100:.1f}%")
                        
                        # Detailed Table
                        st.subheader("📋 Detailed Results")
                        
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
                        
                        # Download Results
                        csv_result = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv_result,
                            file_name="recruitment_predictions_result.csv",
                            mime="text/csv",
                            type="primary"
                        )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


# Footer (Common)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem 0;">
    <p><strong>Recruitment Offer Acceptance Predictor</strong> | Model Accuracy: 95%</p>
    <p><strong>by TriMedian</strong> | Rakamain Data Science </p>
</div>
""", unsafe_allow_html=True)
