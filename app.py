import streamlit as st
import pandas as pd

# 1. TOUJOURS en premier : Configuration de la page
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Chargement des données (avec gestion d'erreur optionnelle)
df = pd.read_csv("loan_data_clean.csv")

# --- TITRE ET EN-TÊTE ---
st.title("💰 Loan Approval Predictor")
st.header("Prédisez l'approbation des demandes de prêt avec des modèles de machine learning")
st.subheader("Par Ambre Servaege")
st.divider() # Plus propre que st.text("___")

# --- SECTION À PROPOS ---
st.markdown("## À propos de l'application")
st.caption("Cette application utilise des modèles de machine learning pour prédire l'approbation des demandes de prêt en fonction des caractéristiques du demandeur.")
st.code("Version 1.0.0", language="text")

# --- CONTENU ---
st.write("""
Bienvenue dans l'application Loan Approval Predictor ! 
Cette application utilise des modèles de machine learning pour prédire l'approbation des demandes de prêt 
en fonction des caractéristiques du demandeur.
""")

# Affichage des données
if not df.empty:
    st.subheader("Aperçu des données")
    st.dataframe(df.head(10)) # .head(10) pour ne pas charger des milliers de lignes inutilement
    
    st.subheader("Statistiques rapides")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Précision du Modèle", "78%", "+5%")
    with col2:
        st.metric("Total des lignes", len(df))

# --- MÉDIAS ---
st.json({"Statut du projet": "Opérationnel", "Auteur": "Ambre Servaege"})

# Note : Les liens ci-dessous doivent être valides pour s'afficher
st.image("https://via.placeholder.com/800x400.png?text=Visualisation+des+donnees", caption="Visualisation des données de prêt")
st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ") # Rickroll détecté ! ;)

#imput user 
if st.button("Cliquez ici"):
    st.write("Bouton cliqué !")

#Slider 
age = st.slider("Sélectionnez votre âge", max_value=100, min_value=30)

#input numérique 
income = st.number_input("Entrez votre revenu annuel", min_value=0, max_value=1000000)

#selectbox
education = st.selectbox("Niveau d'éducation", ["Graduate", "Not Graduate"])

#checkbox 
agree = st.checkbox("J'accepte les termes et conditions")

#radio button
married = st.radio("Êtes-vous marié ?", ["Oui", "Non"])

upload_file = st.file_uploader("Téléchargez votre propre fichier de données CSV") 

#colonne
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Colonne 1")
    st.text("Contenu de la colonne 1")
with col2:
    st.header("Colonne 2")
    st.text("Contenu de la colonne 2")
with col3:
    st.header("Colonne 3")
    st.text("Contenu de la colonne 3")


#sidebar
st.sidebar.title("options")
st.sidebar.selectbox("Choisissez une option", ["Option 1", "Option 2", "Option 3"])
st.sidebar.slider("Sélectionnez une valeur", 0, 100, 50)

with st.expander("Voir plus d'options"):
    st.write("Contenu supplémentaire dans l'expander.") 
    st.dataframe(df.describe())


st.balloons()  # Pour célébrer la fin du chargement de l'application !