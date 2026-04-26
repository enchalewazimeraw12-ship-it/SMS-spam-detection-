import os
import streamlit as st
import joblib
import re
import plotly.graph_objects as go
import pandas as pd
import io

st.set_page_config(page_title="Spam Detector", page_icon="", layout="wide")
st.markdown("""
    <style>
    /* የ GitHub እና Fork ምልክቶችን ብቻ ለሁሉም ሰው ይደብቃል */
    header[data-testid="stHeader"] a {
        display: none !important;
    }
    
    /* የ Deploy በተኑን ይደብቃል */
    .stAppDeployButton {
        display: none !important;
    }

    /* ከላይ በቀኝ በኩል ያለውን የሶስት ነጥብ ሜኑ (Settings) ግን ለአንተ እንዲታይ ይተወዋል */
    </style>
""", unsafe_allow_html=True)
# Custom CSS to reduce padding from 5rem to 1rem, enabling Wide Mode
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* Ensure main content area uses full width */
    .stApp > div {
        max-width: 100% !important;
    }
    /* Reduce sidebar padding */
    .stSidebar {
        padding: 10px;
        line-height: 1.5;
    }
    .stSidebar .stExpander {
        margin-bottom: 10px;
    }
    /* Persistent footer for Group 11 academic branding */
    .academic-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #94a3b8;
        text-align: center;
        padding: 12px 20px;
        font-size: 0.9rem;
        border-top: 2px solid #334155;
        z-index: 9999;
    }
    .academic-footer span {
        color: #38bdf8;
        font-weight: 600;
    }
    /* Add padding to prevent footer overlap */
    .stApp {
        padding-bottom: 60px !important;
    }
</style>
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .dynamic-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        margin-bottom: 30px;
        margin-top: 10px;
        border-bottom: 2px solid #334155;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 8px;
    }
    .dynamic-header h1 {
        margin: 0;
        padding: 10px 0;
        color: #f8fafc;
    }
    @media (max-width: 768px) {
        .dynamic-header {
            font-size: 1.8rem;
            padding: 15px;
            margin-bottom: 20px;
        }
    }
    @media (max-width: 480px) {
        .dynamic-header {
            font-size: 1.4rem;
            padding: 10px;
            margin-bottom: 15px;
        }
    }
    </style>    
    <div class="dynamic-header"><h1>
        ✉️ SMS Spam Detector</h1>
    </div>
    """, 
    unsafe_allow_html=True
)
st.markdown("""
<style>
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        padding: 15px 20px;
        margin-bottom: 25px;
        border-radius: 6px;
        border-left: 4px solid #38bdf8;
    }
    @media (max-width: 768px) {
        .subtitle {
            font-size: 1rem;
            padding: 12px 15px;
            margin-bottom: 20px;
        }
    }
    @media (max-width: 480px) {
        .subtitle {
            font-size: 0.9rem;
            padding: 10px 12px;
            margin-bottom: 15px;
        }
    }
</style>
<h5 class="subtitle">Enter your message below and click Predict to classify it as spam or not spam.</h5>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    vectorizer = joblib.load('models/vectorizer.joblib')
    model = joblib.load('models/spam_model.joblib')
    return vectorizer, model


def normalize_message(text):
    normalized = re.sub(r'\s+', ' ', str(text).strip().lower())
    return normalized


def expert_spam_analysis(text):
    """
    Expert cybersecurity analysis for SMS/Email spam classification.
    Returns: (classification, confidence_score, reason)
    """
    text_lower = text.lower()
    confidence = 0.5  # Base confidence
    reasons = []

    # SPAM Indicators with regex patterns
    urgency_patterns = [
        r'\bact\s+now\b',
        r'\bonly\s+\d+\s+hours?\s+left\b',
        r'\bimmediate\s+action\s+required\b',
        r'\burgent\b',
        r'\bdo not wait\b',
        r'\bright\s+now\b',
    ]

    obfuscated_patterns = [
        r'w[\W_]*i[\W_]*n[\W_]*n[\W_]*e[\W_]*r',
        r'f[\W_]*r[\W_]*e[\W_]*e',
        r'p[\W_]*r[\W_]*i[\W_]*z[\W_]*[e3]',
        r'c[\W_]*l[\W_]*a[\W_]*i[\W_]*m',
        r'b[\W_]*o[\W_]*n[\W_]*u[\W_]*s',
    ]

    url_patterns = [
        r'\bhttps?://[^\s]+\b',
        r'\bbit\.ly/[^\s]+\b',
        r'\btinyurl\.com/[^\s]+\b',
        r'\bt\.co/[^\s]+\b',
        r'\b[\w.-]+\.[a-z]{2,3}/[\w\-_%&=\?]+\b',
    ]

    phone_patterns = [
        r'\+\d{1,3}[\s-]?(?:\(\d+\)|\d+)(?:[\s-]?\d{1,4}){1,4}',
        r'\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b',
        r'\b\d{2,4}[\s-]\d{2,4}[\s-]\d{2,4}\b',
    ]

    # Detect urgency using regex
    urgency_matches = sum(1 for pattern in urgency_patterns if re.search(pattern, text_lower))
    if urgency_matches > 0:
        confidence += 0.15 * urgency_matches
        reasons.append("Contains urgent or time-sensitive language")

    # Detect obfuscated spam words
    obfuscated_matches = sum(1 for pattern in obfuscated_patterns if re.search(pattern, text_lower))
    if obfuscated_matches > 0:
        confidence += 0.2 * obfuscated_matches
        reasons.append("Contains obfuscated spam wording")

    # Detect suspicious URLs or short links
    url_matches = sum(1 for pattern in url_patterns if re.search(pattern, text_lower))
    if url_matches > 0:
        confidence += 0.3
        reasons.append("Contains suspicious or shortened URL")

    # Detect phone numbers in international/scam formats
    phone_matches = sum(1 for pattern in phone_patterns if re.search(pattern, text))
    if phone_matches > 0:
        confidence += 0.15 * phone_matches
        reasons.append("Contains suspicious phone number formatting")

    # Check for promises of money/prizes
    money_keywords = ['win', 'prize', 'cash', 'money', 'free entry', 'claim your prize']
    money_count = sum(1 for word in money_keywords if word in text_lower)
    if money_count > 0:
        confidence += 0.25 * money_count
        reasons.append("Promises money or prizes")

    # Check for poor grammar (multiple exclamation marks, all caps words)
    exclamation_count = text.count('!')
    if exclamation_count > 3:
        confidence += 0.1
        reasons.append("Excessive punctuation")

    all_caps_words = [word for word in text.split() if word.isupper() and len(word) > 3]
    if len(all_caps_words) > 2:
        confidence += 0.1
        reasons.append("Excessive capitalization")

    # HAM Indicators
    ham_indicators = ['hey', 'hi', 'hello', 'thanks', 'see you', 'meeting', 'dinner', 'lunch', 'call me', 'text me']
    personal_count = sum(1 for word in ham_indicators if word in text_lower)
    if personal_count > 0:
        confidence -= 0.2 * personal_count
        reasons.append("Personal/conversational tone")

    # Check for clear sender identity (names, relationships)
    sender_indicators = ['mom', 'dad', 'brother', 'sister', 'friend', 'boss', 'john', 'mary', 'dr.', 'professor']
    sender_count = sum(1 for word in sender_indicators if word in text_lower)
    if sender_count > 0:
        confidence -= 0.15 * sender_count
        reasons.append("Clear sender identity")

    # Check for phishing keywords
    phishing_keywords = ['verify', 'account', 'login', 'password', 'bank', 'secure', 'click', 'confirm', 'credentials', 'identity', 'urgent']
    phishing_count = sum(1 for word in phishing_keywords if word in text_lower)
    if phishing_count > 0:
        confidence += 0.15 * phishing_count
        reasons.append("Contains phishing keyword")
    
    # Tweak for short messages without links
    if len(text) < 20 and url_matches == 0:
        confidence -= 0.2
        reasons.append("Short message without links")

    # Normalize confidence
    confidence = max(0.0, min(1.0, confidence))
    
    # Determine classification (binary: SPAM or HAM)
    if confidence >= 0.3:
        classification = "SPAM"
        reason = "Warning: Potential spam indicators or suspicious language patterns found."
    else:
        classification = "HAM"
        reason = "Safe: No suspicious patterns or links detected."
    
    return classification, confidence, reason


def create_spam_detection_plots(ml_score: float, expert_score: float):
    """Return a Plotly gauge and bar figure for spam detection scores."""
    # Compute combined score for gauge using the 75/25 hybrid logic
    combined_score = ml_score * 0.75 + expert_score * 0.25
    
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=combined_score * 100,
            number={"suffix": "%", "font": {"color": "#ffffff"}},
            delta={
                "reference": ml_score * 100,
                "valueformat": ".1f",
                "increasing": {"color": "#22c55e"},
                "decreasing": {"color": "#f97316"},
                "font": {"color": "#ffffff"},
            },
            title={"text": "Unified Probability Score", "font": {"color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "#0f172a",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#0f172a"},
                    {"range": [25, 65], "color": "#1e293b"},
                    {"range": [65, 100], "color": "#110e2f"},
                ],
                "threshold": {
                    "line": {"color": "#f97316", "width": 4},
                    "thickness": 0.75,
                    "value": combined_score * 100,
                },
            },
        )
    )
    gauge_fig.update_layout(
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "#e2e8f0", "family": "Inter, sans-serif"},
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
    )

    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=["Statistical Patterns", "Security Indicators"],
                y=[ml_score * 100, expert_score * 100],
                marker={
                    "color": ["#38bdf8", "#f97316"],
                    "line": {"color": "#334155", "width": 1},
                },
                text=[f"{ml_score * 100:.1f}%", f"{expert_score * 100:.1f}%"],
                textposition="auto",
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            )
        ]
    )
    bar_fig.update_layout(
        title={"text": "Components of Unified Probability Score", "font": {"color": "#f8fafc"}},
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        xaxis={"tickfont": {"color": "#cbd5e1"}, "showgrid": False},
        yaxis={"tickfont": {"color": "#cbd5e1"}, "gridcolor": "#1e293b", "zerolinecolor": "#334155"},
        font={"color": "#e2e8f0", "family": "Inter, sans-serif"},
        margin={"t": 48, "b": 20, "l": 20, "r": 20},
    )

    return gauge_fig, bar_fig


def determine_final_verdict(ml_prob: float, expert_conf: float, expert_reason: str):
    """Compute a weighted combined spam score and final verdict with risk levels."""
    combined_score = ml_prob * 0.75 + expert_conf * 0.25

    if combined_score > 0.65:
        risk_level = "High Risk (SPAM)"
        classification = "SPAM"
        explanation = (
            f"Combined spam score of {combined_score:.1%} indicates high risk. "
            "The statistical patterns and security indicators detected strong spam patterns. "
            "This message should be treated as spam."
        )
    elif combined_score >= 0.25:
        risk_level = "Medium Risk (REVIEW)"
        classification = "HAM"
        explanation = (
            f"Combined spam score of {combined_score:.1%} indicates medium risk. "
            "The statistical patterns provide the primary signal, supported by security indicators. "
            "Review the message carefully before proceeding."
        )
    else:
        risk_level = "Low Risk (SAFE)"
        classification = "HAM"
        explanation = (
            f"Combined spam score of {combined_score:.1%} indicates low risk. "
            "The statistical patterns and security indicators found no significant spam patterns. "
            "The message appears safe."
        )

    return classification, combined_score, risk_level, explanation


@st.cache_data
def load_batch_file(file_bytes, filename):
    if filename.lower().endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data
def detect_message_columns(df):
    candidate_columns = df.select_dtypes(include=['object']).columns.tolist()
    label_column_pattern = re.compile(r'(^|[_\s\-])(label|target|class|category|type|result)([_\s\-]|$)', re.I)
    label_like_columns = [col for col in candidate_columns if label_column_pattern.search(col) or re.fullmatch(r'(spam|ham|prediction|predicted|actual|true|groundtruth)s?', col, re.I)]
    filtered_columns = [col for col in candidate_columns if col not in label_like_columns]
    return filtered_columns if filtered_columns else candidate_columns


@st.cache_data
def compute_batch_predictions(messages_tuple):
    messages = list(messages_tuple)
    X_input = vectorizer.transform(messages)
    predictions = model.predict(X_input)
    probabilities = model.predict_proba(X_input)
    spam_probabilities = probabilities[:, 1]
    return predictions.tolist(), spam_probabilities.tolist()


def create_performance_dashboard(hour_labels, counts):
    bar_fig = go.Figure(
        go.Bar(
            x=hour_labels,
            y=counts,
            marker_color="#38bdf8",
            marker_line_color="#64748b",
            marker_line_width=1.5,
            hovertemplate="%{x}: %{y} messages<extra></extra>",
        )
    )
    bar_fig.update_layout(
        title={"text": "Messages Processed per Hour", "font": {"color": "#f8fafc"}},
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        xaxis={"tickfont": {"color": "#cbd5e1"}, "gridcolor": "#1e293b", "showgrid": False},
        yaxis={"tickfont": {"color": "#cbd5e1"}, "gridcolor": "#1e293b", "zerolinecolor": "#334155"},
        font={"color": "#e2e8f0", "family": "Inter, sans-serif"},
        margin={"t": 40, "b": 20, "l": 20, "r": 20},
    )
    return bar_fig
try:
    vectorizer, model = load_model()
    st.success(" Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model not found. Please run 'train_model.py' first.\n\nError: {e}")
    st.stop()

with st.sidebar:
    with st.expander("ℹ️ About this Project", expanded=False):
        st.write(
            "This app uses both machine learning (TF-IDF + Logistic Regression) and expert cybersecurity analysis "
            "to classify SMS text as spam or ham. The ML model is trained on the SMS Spam Collection dataset."
        )
    
    with st.expander("🛡️ Analysis Methods", expanded=False):
        st.write("**🤖 Machine Learning:** Statistical pattern recognition")
        st.write("**🛡️ Expert Analysis:** Rule-based cybersecurity indicators")
    
    with st.expander("🚀 How to Use", expanded=False):
        st.write("1. Enter or paste an SMS message.\n2. Click Predict.\n3. Review both ML and expert results.")
    
    with st.expander("📊 Security Monitoring", expanded=True):
        # Example dashboard data for the last 24 hours
        now = pd.Timestamp.now().floor('h')
        hours = [(now - pd.Timedelta(hours=i)).strftime('%H:%M') for i in reversed(range(24))]
        messages_processed = [18, 22, 24, 20, 17, 15, 12, 14, 18, 21, 26, 30, 32, 29, 27, 31, 35, 40, 38, 34, 28, 24, 20, 19]

        col1, col2 = st.columns(2)
        col1.metric("Total Spam Blocked", "1,248")
        col2.metric("Average Confidence", "87.4%")

        st.plotly_chart(create_performance_dashboard(hours, messages_processed), width='stretch')

examples = [
    "Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)",
    "Hey, are we still meeting for dinner tonight?",
    "URGENT! Your account has been suspended. Call 09061701461 immediately."
]

with st.expander("Example messages"):
    for example in examples:
        st.code(example)

user_input = st.text_area("📝 Enter your SMS message here", height=170)

if st.button("🔍 Predict", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message before predicting.")
    else:
        # Machine Learning Prediction
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0]
        spam_score = probability[1]

        st.subheader("🤖 Machine Learning Analysis")
        if prediction == 1:
            st.error(f"🚨 SPAM - Likely spam with confidence {spam_score:.2%}")
        else:
            st.success(f" NOT SPAM - Likely safe with confidence {1 - spam_score:.2%}")

        st.write("**ML Prediction details:**")
        st.write(f"- Spam probability: {spam_score:.2%}")
        st.write(f"- Ham probability: {1 - spam_score:.2%}")

        st.markdown("---")
        
        # Expert Cybersecurity Analysis
        st.subheader("🛡️ Expert Cybersecurity Analysis")
        expert_class, expert_conf, expert_reason = expert_spam_analysis(user_input)
        
        gauge_fig, bar_fig = create_spam_detection_plots(spam_score, expert_conf)
        st.plotly_chart(gauge_fig, width='stretch')
        st.plotly_chart(bar_fig, width='stretch')
        
        if expert_class == "SPAM":
            st.error(f"🚨 {expert_class} - Expert confidence: {expert_conf:.2%}")
        else:
            st.success(f"✅ {expert_class} - Expert confidence: {1 - expert_conf:.2%}")
        
        st.write(f"**Reason:** {expert_reason}")
        st.write(f"**Expert confidence score:** {expert_conf:.2f}")

        final_label, final_score, final_risk, final_explanation = determine_final_verdict(
            spam_score,
            expert_conf,
            expert_reason,
        )
        st.markdown("---")
        st.subheader("🧾 Final Verdict")
        st.write(f"**Combined Spam Score:** {final_score:.1%}")
        st.write(f"**Risk Level:** {final_risk}")
        if "High Risk" in final_risk:
            st.error(f"🚨 {final_risk} - {final_explanation}")
        elif "Medium Risk" in final_risk:
            st.warning(f"⚠️ {final_risk} - {final_explanation}")
        else:
            st.success(f"✅ {final_risk} - {final_explanation}")

st.markdown("---")

# Batch Prediction Feature
st.header(" Batch Prediction")
st.markdown("Upload a CSV or Excel file with a 'message' column to classify multiple messages at once.")

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'], key="batch_upload")

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        df = load_batch_file(file_bytes, uploaded_file.name)

        string_columns = detect_message_columns(df)

        if not string_columns:
            st.error("The file must contain at least one text column.")
        else:
            if len(string_columns) == 1:
                message_column = string_columns[0]
                st.info(f"Detected message column: '{message_column}'")
            else:
                message_column = st.selectbox("Select the column containing messages:", string_columns, key="message_column_select")

            if message_column:
                st.markdown("**Preview of first 5 rows:**")
                st.dataframe(df[[message_column]].head(5))
                st.caption("Review the preview and click Process Batch to analyze the full file.")

                process_button = st.button("Process Batch", key="process_batch")
                if process_button:
                    messages = df[message_column].fillna('').astype(str)
                    preview_count = min(5, len(messages))
                    st.info(f"Previewed {preview_count} rows. Processing {len(messages)} total messages...")

                    progress = st.progress(0)
                    status_text = st.empty()

                    # Chunked processing for progress updates
                    chunk_size = max(100, len(messages) // 20)
                    all_predictions = []
                    all_probabilities = []
                    all_expert_confidences = []
                    all_expert_reasons = []
                    all_combined_scores = []
                    all_combined_predictions = []

                    for i in range(0, len(messages), chunk_size):
                        batch_messages = messages[i:i + chunk_size].tolist()
                        _, spam_probs = compute_batch_predictions(tuple(batch_messages))
                        spam_probs = [float(p) for p in spam_probs]
                        expert_results = [expert_spam_analysis(text) for text in batch_messages]
                        expert_confidences = [float(result[1]) for result in expert_results]
                        expert_reasons = [result[2] for result in expert_results]

                        combined_scores = [ml_prob * 0.75 + expert_conf * 0.25 for ml_prob, expert_conf in zip(spam_probs, expert_confidences)]
                        combined_predictions = ['SPAM' if score >= 0.5 else 'HAM' for score in combined_scores]

                        all_predictions.extend(combined_predictions)
                        all_probabilities.extend(spam_probs)
                        all_expert_confidences.extend(expert_confidences)
                        all_expert_reasons.extend(expert_reasons)
                        all_combined_scores.extend(combined_scores)
                        all_combined_predictions.extend(combined_predictions)

                        progress_percent = int(min(100, (i + len(batch_messages)) / len(messages) * 100))
                        progress.progress(progress_percent)
                        status_text.text(f"Processed {min(i + len(batch_messages), len(messages))} / {len(messages)} messages")

                    progress.progress(100)
                    status_text.text("Batch processing complete.")

                    df['combined_score'] = pd.Series(all_combined_scores, index=df.index)
                    df['prediction'] = df['combined_score'].apply(lambda score: 'SPAM' if float(score) >= 0.5 else 'HAM')
                    df['ml_spam_probability'] = pd.Series(all_probabilities, index=df.index)
                    df['expert_confidence'] = pd.Series(all_expert_confidences, index=df.index)
                    df['expert_reason'] = pd.Series(all_expert_reasons, index=df.index)

                    spam_count = sum(1 for label in all_combined_predictions if label == 'SPAM')
                    ham_count = sum(1 for label in all_combined_predictions if label == 'HAM')

                    pie_fig = go.Figure(
                        data=[go.Pie(
                            labels=['SPAM', 'HAM'],
                            values=[spam_count, ham_count],
                            marker_colors=['#f97316', '#22c55e'],
                            textinfo='label+percent',
                            insidetextorientation='radial'
                        )]
                    )
                    pie_fig.update_layout(
                        title={"text": "Spam vs Ham Distribution", "font": {"color": "#f8fafc"}},
                        paper_bgcolor="#020617",
                        plot_bgcolor="#020617",
                        font={"color": "#e2e8f0", "family": "Inter, sans-serif"},
                        margin={"t": 48, "b": 20, "l": 20, "r": 20},
                    )

                    st.plotly_chart(pie_fig, width='stretch')
                    st.success(f"Processed {len(messages)} messages. Spam: {spam_count}, Ham: {ham_count}")

                    st.markdown("**Batch Results Preview:**")
                    st.dataframe(
                        df[[message_column, 'prediction', 'combined_score', 'ml_spam_probability', 'expert_confidence']].head(20),
                        column_config=None,
                        hide_index=False,
                        use_container_width=False
                    )

                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="spam_predictions.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
    except Exception as e:
        st.error(f"Error processing file: {e}")

st.markdown("---")
st.caption("Spam detector system | Academic project | Group 11")

# Persistent footer for academic accountability
st.markdown("""
<div class="academic-footer">
    📊 <span>by new Technology</span> — SMS Spam Detection System | Academic Project
</div>
""", unsafe_allow_html=True)
