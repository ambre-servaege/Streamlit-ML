import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="ML Analytics App",
    page_icon="🤖",
    layout="wide"
)

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    try:
        # Tentative de lecture du fichier local
        df = pd.read_csv("loan_data_clean.csv")
    except:
        # Dataset de secours généré aléatoirement si le fichier est absent
        data = {
            'ApplicantIncome': np.random.randint(2000, 15000, 100),
            'LoanAmount': np.random.randint(50, 500, 100),
            'Education': np.random.choice(['Graduate', 'Not Graduate'], 100),
            'Loan_Status': np.random.choice(['Y', 'N'], 100),
            'Age': np.random.randint(18, 65, 100),
            'Credit_History': np.random.choice([1.0, 0.0], 100),
            'Married': np.random.choice(['Yes', 'No'], 100)
        }
        df = pd.DataFrame(data)
    return df

df = load_data()

# --- SIDEBAR & FILTRES ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.header("Modèle")
    model_choice = st.selectbox("Algorithme :", ("Logistic Regression", "Random Forest", "XGBoost"))
    st.info(f"Modèle sélectionné : **{model_choice}**")
    
    st.divider()
    
    st.header("Filtres Dashboard")
    min_inc = int(df['ApplicantIncome'].min())
    max_inc = int(df['ApplicantIncome'].max())
    income_range = st.slider("Plage de revenu ($)", min_inc, max_inc, (min_inc, max_inc))

    edu_options = df['Education'].unique().tolist()
    selected_edu = st.multiselect("Niveau d'éducation", edu_options, default=edu_options)

# Filtrage global
df_filtered = df[
    (df['ApplicantIncome'] >= income_range[0]) & 
    (df['ApplicantIncome'] <= income_range[1]) & 
    (df['Education'].isin(selected_edu))
]

# --- TITRE PRINCIPAL ---
st.title("🤖 Plateforme d'Analyse Prédictive")

tab_exploration, tab_prediction, tab_performance, tab_dashboard = st.tabs([
    "🔍 Exploration", "🔮 Prédiction", "📈 Performance", "📊 Analyse Dashboard"
])

# --- 1. ONGLET EXPLORATION ---
with tab_exploration:
    st.header("Exploration des données")
    st.dataframe(df_filtered, use_container_width=True)
    st.download_button("Télécharger les données filtrées", df_filtered.to_csv(), "data.csv")

# --- 2. ONGLET PRÉDICTION (AVEC FORMULAIRE 2 COLONNES) ---
with tab_prediction:
    st.header("🔮 Simulateur de Scoring Crédit")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Données Financières")
            income = st.number_input("Revenu mensuel ($)", min_value=0, value=5000)
            loan_amt = st.number_input("Montant du prêt ($)", min_value=0, value=150)
            term = st.select_slider("Durée du prêt (mois)", options=[12, 36, 60, 180, 360], value=360)
        
        with col2:
            st.markdown("### 📋 Profil Client")
            credit_hist = st.radio("Historique de crédit", options=[1.0, 0.0], 
                                   format_func=lambda x: "Sain (Paiements à jour)" if x == 1.0 else "Défaut de paiement")
            edu = st.selectbox("Niveau d'Éducation", options=["Graduate", "Not Graduate"])
            married = st.selectbox("Statut Marital", options=["Yes", "No"])

        submit = st.form_submit_button("Calculer la probabilité d'accord")

    if submit:
        # Barre de progression
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress_bar.progress(i + 1)
        
        # --- FEATURE ENGINEERING & ENCODING ---
        # 1. Calcul de ratios
        loan_inc_ratio = (loan_amt * 100) / (income + 1)
        # 2. Encodage
        edu_val = 1 if edu == "Graduate" else 0
        married_val = 1 if married == "Yes" else 0
        
        # --- LOGIQUE DU MODÈLE (Simulation) ---
        # L'historique de crédit est le facteur majeur
        score_base = 0.7 if credit_hist == 1.0 else 0.1
        penalty = 0.2 if loan_inc_ratio > 35 else 0
        bonus = 0.1 if edu_val == 1 else 0
        
        prob = np.clip(score_base - penalty + bonus + np.random.uniform(-0.05, 0.05), 0, 1)
        is_approved = prob >= 0.5

        # --- AFFICHAGE DES RÉSULTATS ---
        st.divider()
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if is_approved:
                st.success("### ✅ DÉCISION : PRÊT APPROUVÉ")
                st.balloons()
            else:
                st.error("### ❌ DÉCISION : PRÊT REFUSÉ")
            
            st.metric("Confiance du modèle", f"{prob:.2%}")
            
            # Warnings dynamiques
            if credit_hist == 0.0:
                st.warning("⚠️ L'historique de crédit négatif est le principal frein.")
            if loan_inc_ratio > 40:
                st.warning(f"⚠️ Ratio d'endettement trop élevé : {loan_inc_ratio:.1f}%")

        with res_col2:
            st.subheader("🧬 Facteurs d'influence (Top 5)")
            feat_imp = pd.DataFrame({
                'Feature': ['Crédit Hist.', 'Ratio Dette/Rev.', 'Éducation', 'Revenu', 'Statut Marital'],
                'Importance': [0.55, 0.20, 0.12, 0.08, 0.05]
            }).sort_values('Importance', ascending=True)
            
            fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', 
                             color='Importance', color_continuous_scale='RdYlGn')
            fig_imp.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_imp, use_container_width=True)

# --- 3. ONGLET PERFORMANCE ---
with tab_performance:
    st.header("📈 Métriques du Modèle")
    
    perf_col1, perf_col2 = st.columns(2)
    y_test = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    y_probs = [0.1, 0.9, 0.2, 0.3, 0.8, 0.6, 0.4, 0.85, 0.15, 0.7]
    y_pred = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]

    with perf_col1:
        st.subheader("Matrice de Confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm)
        st.pyplot(fig_cm)

    with perf_col2:
        st.subheader("Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
        ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax_roc.set_xlabel('Faux Positifs')
        ax_roc.set_ylabel('Vrais Positifs')
        ax_roc.legend()
        st.pyplot(fig_roc)

# --- 4. ONGLET DASHBOARD ---
with tab_dashboard:
    st.header("📊 Vue d'ensemble de la Population")

    # Métriques clés
    m1, m2, m3, m4 = st.columns(4)
    total = len(df_filtered)
    app_rate = (df_filtered['Loan_Status'] == 'Y').mean() * 100 if total > 0 else 0
    m1.metric("Total Dossiers", total)
    m2.metric("Taux d'Approbation", f"{app_rate:.1f}%")
    m3.metric("Prêt Moyen", f"{df_filtered['LoanAmount'].mean():.1f}k$")
   # avg_age = df_filtered['Age'].mean()
    #m4.metric("Âge Moyen", f"{avg_age:.1f} ans" if not np.isnan(avg_age) else "N/A")

    st.divider()

    # Visualisations
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.plotly_chart(px.histogram(df_filtered, x="ApplicantIncome", color="Loan_Status", 
                                     title="Revenus vs Statut Prêt", barmode="overlay"), use_container_width=True)
    with d_col2:
        st.plotly_chart(px.scatter(df_filtered, x="ApplicantIncome", y="LoanAmount", color="Education",
                                   title="Montant Prêt vs Revenu"), use_container_width=True)
        

#exooo


import json
metrics = {
    "accuracy": 0.78,
    "precision": 0.82,
    "recall": 0.89,
    "f1_score": 0.85,
    "roc_auc": 0.82
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)
with open("metrics.json", "r") as f:
    metrics = json.load(f)
