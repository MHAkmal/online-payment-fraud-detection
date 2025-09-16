import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import base64

# --- Page layout configuration ---
st.set_page_config(
    page_title="Fraud Detection App",
    layout="wide"
)

# --- Custom CSS for styling ---
st.markdown(
    """
    <style>
    /* Sidebar width */
    [data-testid="stSidebar"] {
        width: 300px !important;
    }

    /* Container to help with alignment */
    .connect-box {
        display: flex;
        flex-direction: column;
        align-items: center; /* Center alignment */
        justify-content: center;
        height: 100%;
    }

    /* Style for connect links */
    .connect-container {
        display: flex;
        flex-direction: column;
        gap: 10px; /* Space between links */
    }
    .connect-link {
        display: flex;
        align-items: center;
        gap: 8px; /* Space between icon and text */
        text-decoration: none;
        font-weight: bold;
        color: #FFFFFF !important; /* White text for links */
    }
    .connect-link:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SVGs for Icons (Base64 Encoded) ---
def get_svg_as_b64(svg_raw):
    """Encodes a raw SVG string into a Base64 string."""
    return base64.b64encode(svg_raw.encode('utf-8')).decode()

linkedin_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#0077B5" stroke="currentColor" stroke-width="0" stroke-linecap="round" stroke-linejoin="round"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>')
github_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#FFFFFF" stroke="currentColor" stroke-width="0" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>')
x_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 16 16" fill="#FFFFFF"><path d="M12.6.75h2.454l-5.36 6.142L16 15.25h-4.937l-3.867-5.07-4.425 5.07H.316l5.733-6.57L0 .75h5.063l3.495 4.633L12.602.75Zm-1.283 12.95h1.46l-7.48-10.74h-1.55l7.57 10.74Z"/></svg>')
threads_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a10 10 0 1 1 0-20 10 10 0 0 1 0 20Z"></path><path d="M16.5 8.5c-.7-1-1.8-1.5-3-1.5s-2.3.5-3 1.5"></path><path d="M16.5 15.5c-.7 1-1.8 1.5-3 1.5s-2.3-.5-3-1.5"></path><path d="M8.5 12a5.5 5.5 0 1 0 7 0 5.5 5.5 0 0 0-7 0Z"></path></svg>')

# --- Caching the Model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained fraud detection model."""
    try:
        # The saved file might contain a tuple: (model, y_pred, X_test, y_test)
        loaded_artifacts = joblib.load('online-transaction-fraud-detection-dtree.pkl')
        
        # Check if the loaded object is a tuple, and extract the first element
        if isinstance(loaded_artifacts, tuple):
            model_pipeline = loaded_artifacts[0]
        else:
            model_pipeline = loaded_artifacts # Assume it was saved correctly as just the model
        
        return model_pipeline
        
    except FileNotFoundError:
        st.error("Model file 'online-transaction-fraud-detection-dtree.pkl' not found. Please ensure the model is in the same directory.")
        return None
    except (IndexError, TypeError):
         st.error("The model file seems to be in an incorrect format. It should contain the model pipeline as the first element. Please re-save your model correctly.")
         return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load the model
model = load_model()

# --- Initialize session state for prediction history ---
if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []

# --- Sidebar Navigation ---
with st.sidebar:
    selection = option_menu(
        menu_title="Main Menu",
        options=["Home", "Fraud Detection", "Notebook"],
        icons=["house", "shield-check", "book"],
        menu_icon="cast",
        default_index=0,
    )

# --- Home Page ---
if selection == "Home":
    top_col1, top_col2 = st.columns([0.75, 0.25])

    with top_col1:
        st.title("💳 Online Transaction Fraud Detection")
        st.markdown("""
        Welcome to the Fraud Detection app! This app simulate an online transaction fraud detection based on various attributes.
        """)
        st.warning("Navigate to the **Fraud Detection** tab from the sidebar to check a transaction!", icon="👈")

    with top_col2:
        st.markdown('<div class="connect-box">', unsafe_allow_html=True)
        st.subheader("🔗 Connect With Me")
        linkedin_link = f'<a href="https://www.linkedin.com/in/mhakmal/" class="connect-link"><img src="data:image/svg+xml;base64,{linkedin_svg}" width="24"><span>MHAkmal</span></a>'
        github_link = f'<a href="https://github.com/MHAkmal" class="connect-link"><img src="data:image/svg+xml;base64,{github_svg}" width="24"><span>MHAkmal</span></a>'
        x_link = f'<a href="https://x.com/akmal621" class="connect-link"><img src="data:image/svg+xml;base64,{x_svg}" width="24"><span>MHAkmal</span></a>'
        threads_link = f'<a href="https://www.threads.com/@akmal621?__pwa=1" class="connect-link"><img src="data:image/svg+xml;base64,{threads_svg}" width="24"><span>MHAkmal</span></a>'
        st.markdown(f'<div class="connect-container">{linkedin_link}{github_link}{x_link}{threads_link}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    bp_col1, obj_col2 = st.columns(2)
    with bp_col1:
        st.header("Business Problem")
        st.markdown("""
        - How to reduce fraud transaction in financial company?
        - How to reduce revenue loss because of fraud transaction?
        - How to automatically detect a fraudulent transaction?
        """)

    with obj_col2:
        st.header("Objective")
        st.write("Build a classification machine learning model to predict whether the transaction is fraud or not.")

    st.divider()

    st.header("Business Impact")
    st.subheader("Model Performance (Model: Decision Tree)")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Accuracy", value="99%")
    with m_col2:
        st.metric(label="**Recall**", value="87%")
    with m_col3:
        st.metric(label="Precision", value="89%")
    with m_col4:
        st.metric(label="F1-Score", value="88%")

    st.info("**Recall is the key metric**: It means our model successfully identifies 87% of all actual fraudulent transactions.", icon="🎯")
    
    st.subheader("Impact Simulation (per 100 transactions, assuming 5 are fraudulent)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Before Modeling")
        st.markdown("""
        - The company doesn't know which transaction will be fraud; there is no fraud mitigation at all.
        """)
        st.metric(label="Fraudulent Transactions Occurring", value="5")
        st.metric(label="Estimated Financial Loss (at $1M per transaction)", value="$5,000,000")

    with col2:
        st.markdown("#### After Modeling")
        st.markdown(f"""
        - The model flags transactions that are likely fraudulent.
        - **Fraud Cases Caught**: 5 * 87% (Recall) = 4.35 ≈ 4
        - **Uncaught Fraud**: 5 - 4 = 1
        """)
        st.metric(label="Uncaught Fraudulent Transactions", value="1", delta="-4 transactions", delta_color="inverse")
        st.metric(label="Estimated Financial Loss", value="$1,000,000", delta="-$4,000,000", delta_color="inverse")
    
    st.success("By implementing this model, the company could prevent approximately **4 out of 5 fraudulent transactions**, saving an estimated **$4,000,000** for every 100 transactions under this scenario.")

# --- Prediction Page ---
if selection == "Fraud Detection" and model is not None:
    st.title("Check a Transaction for Fraud")
    st.info("Enter the transaction details in the form below and click 'Assess Transaction' to get a prediction.", icon="ℹ️")
    
    with st.form("transaction_form"):
        st.header("Transaction Details")
        
        c1, c2 = st.columns(2)
        with c1:
            type = st.selectbox('Transaction Type', ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'], key='type')
        with c2:
            amount = st.number_input('Amount', min_value=0, value=181, format="%d", key='amount')

        col1, col2 = st.columns(2)
        with col1:
            oldbalanceOrg = st.number_input('Origin: Old Balance', min_value=0, value=181, format="%d", key='oldbalanceOrg')
            oldbalanceDest = st.number_input('Destination: Old Balance', min_value=0, value=0, format="%d", key='oldbalanceDest')

        with col2:
            newbalanceOrig = st.number_input('Origin: New Balance', min_value=0, value=0, format="%d", key='newbalanceOrig')
            newbalanceDest = st.number_input('Destination: New Balance', min_value=0, value=0, format="%d", key='newbalanceDest')
        
        # --- Live Automatic Input ---
        balance_change_orig = newbalanceOrig - oldbalanceOrg
        balance_change_dest = newbalanceDest - oldbalanceDest
        st.subheader("Automatic Input")
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Origin Balance Change", f"{balance_change_orig:,.0f}")
        mcol2.metric("Destination Balance Change", f"{balance_change_dest:,.0f}")

        st.divider()

        # --- Form Submission and Example ---
        btn_col, example_col = st.columns([1, 2])
        with btn_col:
            submitted = st.form_submit_button('Assess Transaction', type="primary", use_container_width=True)
        
        with example_col:
            with st.expander("Click to see examples of fraudulent transactions"):
                st.info("Note: In this dataset, fraudulent transactions only occur with 'TRANSFER' and 'CASH_OUT' types.", icon="ℹ️")
                
                fraud_examples_data = {
                    'Type': ['TRANSFER', 'CASH_OUT', 'TRANSFER', 'CASH_OUT', 'TRANSFER', 'CASH_OUT', 'TRANSFER', 'CASH_OUT', 'CASH_OUT', 'CASH_OUT'],
                    'Amount': [20128, 20128, 1277212, 1277212, 181, 181, 25071, 25071, 132842, 416001],
                    'Origin Old Balance': [20128, 20128, 1277212, 1277212, 181, 181, 25071, 25071, 4499, 0],
                    'Origin New Balance': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    'Dest. Old Balance': [0, 6268, 0, 0, 0, 21182, 0, 9083, 0, 102],
                    'Dest. New Balance': [0, 12146, 0, 2444985, 0, 0, 0, 34155, 132842, 9291619]
                }
                fraud_df = pd.DataFrame(fraud_examples_data)
                st.dataframe(fraud_df.style.format('{:,.0f}', subset=fraud_df.columns.drop('Type')), use_container_width=True)

    if submitted:
        # Prepare input for the model
        current_input = {
            'type': type, 'amount': float(amount), 'oldbalanceOrg': float(oldbalanceOrg),
            'newbalanceOrig': float(newbalanceOrig), 'oldbalanceDest': float(oldbalanceDest),
            'newbalanceDest': float(newbalanceDest), 'balance_change_orig': float(balance_change_orig),
            'balance_change_dest': float(balance_change_dest),
        }
        
        input_df = pd.DataFrame([current_input])
        
        # Ensure column order matches the model's training data
        feature_order = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'balance_change_orig', 'balance_change_dest']
        input_df = input_df[feature_order]

        prediction = model.predict(input_df)[0]

        st.header("Prediction Result")
        if prediction == 1:
            st.error(f"**High Risk: This transaction is likely FRAUDULENT.**", icon="🚨")
        else:
            st.success(f"**Low Risk: This transaction is likely LEGITIMATE.**", icon="✅")
        
        # Log the prediction to session state
        log_entry = {'input': current_input, 'prediction': 'Fraudulent' if prediction == 1 else 'Legitimate'}
        st.session_state.prediction_log.insert(0, log_entry)

    # --- Display Prediction History ---
    st.divider()
    st.header("Assessment History")
    if st.session_state.prediction_log:
        history_list = []
        for entry in st.session_state.prediction_log:
            display_entry = {
                'Type': entry['input']['type'],
                'Amount': entry['input']['amount'],
                'Origin Balance Change': f"{entry['input']['oldbalanceOrg']:,.0f} -> {entry['input']['newbalanceOrig']:,.0f}",
                'Dest Balance Change': f"{entry['input']['oldbalanceDest']:,.0f} -> {entry['input']['newbalanceDest']:,.0f}",
                'Prediction': entry['prediction']
            }
            history_list.append(display_entry)
        
        history_df = pd.DataFrame(history_list)
        st.dataframe(history_df.style.format({'Amount': '{:,.0f}'}), use_container_width=True)
    else:
        st.info("No transactions have been assessed in this session yet.")

elif selection == "Fraud Detection" and model is None:
    st.warning("The fraud detection model is not available. Please check for error messages when the app started.")

# --- Notebook Page ---
if selection == "Notebook":
    st.title("Fraud Detection Model Notebook")
    st.write("This section displays the Jupyter Notebook used for data exploration, cleaning, and model building.")
    st.info("The notebook below is a static HTML file and is not interactive.")

    notebook_filename = "online-transaction-fraud-detection-ml-model.html"
    try:
        with open(notebook_filename, "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, width=None, height=800, scrolling=True)
    except FileNotFoundError:

        st.error(f"File not found: '{notebook_filename}'. Please ensure the notebook has been exported to HTML and is in the same directory as this script.")

