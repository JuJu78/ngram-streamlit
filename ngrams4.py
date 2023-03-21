# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
from io import BytesIO
import base64
import time
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import chardet
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import spacy
from langdetect import detect

# Chargez les modèles français et anglais
nlp_fr = spacy.load("fr_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

# Chargement du fichier
st.title("Analyseur de n-grams")
uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=["csv", "xlsx"])

# Décodage du texte
def decode_text(text):
    # Détecte l'encodage du texte
    detected_encoding = chardet.detect(text)['encoding']
    
    # Décode le texte en utilisant l'encodage détecté
    decoded_text = text.decode(detected_encoding)
    
    return decoded_text

# Nettoyage du texte
def clean_text(raw_text):
    # Décode le texte en utilisant l'encodage détecté
    text = decode_text(raw_text)

    # Vérifie si le texte est vide ou ne contient que des caractères spéciaux
    if not re.search(r'\w+', text):
        return text
    
    # Suppression des caractères accentués
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')

    # Détection de la langue
    lang = detect(text) if re.search(r'\w+', text) else "fr"

    # Choix du modèle de langue en fonction de la langue détectée
    nlp = nlp_fr if lang == "fr" else nlp_en

    # Création d'un objet Doc pour le texte
    doc = nlp(text)

    # Suppression des stopwords en anglais et en français
    stop_words = set(stopwords.words('english')) | set(stopwords.words('french'))

    # Ajout des stopwords supplémentaires
    additional_stopwords = {"de", "de ", " de", " de "}
    stop_words |= additional_stopwords

    # Lemmatisation en utilisant spaCy et suppression des stopwords
    text = ' '.join([token.lemma_ for token in doc if token.text.lower() not in stop_words])

    # Suppression explicite des mots à exclure
    excluded_words = {"de", "ce", "puis"}
    for word in excluded_words:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)

    return text

@st.cache_data
def analyze_ngrams(data, column, ngrams_list):
    # Nettoyage du texte
    data[column] = data[column].apply(lambda x: clean_text(x.encode('utf-8')))

    # Extraction des n-grams pour chaque taille de n-gram sélectionnée
    results_list = []
    for ngrams in ngrams_list:
        vectorizer = CountVectorizer(ngram_range=(ngrams, ngrams))
        X = vectorizer.fit_transform(data[column])
        features = vectorizer.get_feature_names_out()
        counts = X.toarray().sum(axis=0)
        results = pd.DataFrame({'ngram': features, 'occurrence': counts})
        # Supprimer l'index
        results.reset_index(drop=True, inplace=True)
        results['taille du ngram'] = ngrams
        results_list.append(results)

    # Concaténation des DataFrames pour chaque taille de n-gram
    df = pd.concat(results_list).sort_values(by='occurrence', ascending=False)

    return df

def to_excel_download_link(df, filename="ngrams_results.xlsx"):
    # Générer un lien pour télécharger un DataFrame au format Excel
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Télécharger les résultats en tant que fichier Excel</a>'
    return href

if uploaded_file:
    # Détecte l'encodage du fichier
    file_content = uploaded_file.read()
    detected_encoding = chardet.detect(file_content)['encoding']
    # Lecture du fichier
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        data = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        uploaded_file.seek(0)  # Rembobine le fichier à son début
        data = pd.read_csv(uploaded_file, encoding=detected_encoding, error_bad_lines=False)

    # Sélection de la colonne
    column = st.selectbox("Sélectionnez la colonne contenant les chaînes de caractères", data.columns)

    # Sélection des tailles de n-grams
    ngrams_list = st.multiselect("Choisissez les tailles des n-grams", options=[1, 2, 3, 4, 5, 6], default=[1, 2, 3, 4, 5])

    if ngrams_list:
        # Analyse des n-grams
        results = analyze_ngrams(data, column, ngrams_list)

        # Définition du style de la largeur des colonnes
        column_width_style = [
            {
                'selector': 'th',
                'props': [('width', '500px')]
            }
        ]

        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text=progress_text)

        # Affichage du DataFrame sans index et avec une largeur de colonne personnalisée
        st.write(results.style.set_table_styles(column_width_style))

        # Lien pour télécharger le DataFrame sous la forme d'un fichier Excel
        st.markdown(to_excel_download_link(results), unsafe_allow_html=True)

        # Affichage du graphique en colonne
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(x='ngram', y='occurrence', data=results.head(50), ax=ax)
        plt.xticks(rotation=90)

        # Ajout des libellés de données sur chaque barre
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline', fontsize=9, color='black', xytext=(0, 3),
                        textcoords='offset points')

        st.pyplot(fig)

        # Affichage du graphique en camembert
        fig, ax = plt.subplots(figsize=(12, 5))
        top_10_ngrams = results.head(10)
        total_ngrams = results['occurrence'].sum()
        proportions = top_10_ngrams['occurrence']
        labels = top_10_ngrams['ngram']
        plt.pie(proportions, labels=labels, autopct='%1.1f%%')
        plt.axis('equal')
        plt.title("Pourcentage de chacun des 10 n-grams les plus fréquents par rapport à l'ensemble des n-grams",
          fontsize=16, fontweight='bold')  # Ajout du titre avec une taille de police et un style spécifiques
        
        st.pyplot(fig)
    
    else:
        st.warning("Veuillez sélectionner au moins une taille de n-gram.")
