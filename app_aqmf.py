# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:28:16 2025

@author: felima
"""
# -*- coding: utf-8 -*-
"""
Analyse AQMF - Version corrigée avec gestion PDF
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import functions
from fpdf import FPDF
from pathlib import Path
import io
from datetime import date
import tempfile
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Analyse AQMF", layout="wide")
st.title("Analyse AQMF - Rapport Mouvement Facial")

# Fonction pour sauvegarder les figures en mémoire
def save_fig_to_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# Charger la référence
try:
    with open("list_hr_mean_norm_9.pkl", "rb") as f:
        list_hr_m = pickle.load(f)
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier de référence: {str(e)}")
    st.stop()

# Collecte des informations patient
with st.sidebar:
    st.header("Informations du patient")
    nom = st.text_input("Nom")
    prenom = st.text_input("Prénom")
    
    # Paramètres de date étendus
    min_date = date(1900, 1, 1)
    max_date = date.today()
    
    date_naissance = st.date_input(
        "Date de naissance",
        value=date(1980, 1, 1),
        min_value=min_date,
        max_value=max_date
    )
    
    date_examen = st.date_input(
        "Date de l'examen",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

# Upload des fichiers
uploaded_files = st.file_uploader("Importer les fichiers CSV", type="csv", accept_multiple_files=True)

if uploaded_files and nom and prenom:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Sauvegarde des fichiers
        for i, file in enumerate(uploaded_files):
            file_path = tmpdir_path / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Traitement des données...")
        
        # Traitement par catégorie
        categories = [f"M{i}" for i in range(1, 10)]
        results = {
            'anomalies': {},
            'faciogrammes': {},
            'mouvements': None
        }
        
        # ... (le reste du traitement des données reste inchangé jusqu'à la génération des graphiques)
        
        # Génération du PDF corrigée
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Page 1 - En-tête et anomalies
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, f"Rapport AQMF - {prenom} {nom}", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f"Date de naissance: {date_naissance.strftime('%d/%m/%Y')}", ln=True)
            pdf.cell(0, 10, f"Date d'examen: {date_examen.strftime('%d/%m/%Y')}", ln=True)
            pdf.ln(10)
            
            # Graphique des anomalies
            if hasattr(results, 'fig_anomalies'):
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Anomalies par catégorie", ln=True)
                
                # Conversion correcte de l'image
                img = Image.open(results.fig_anomalies)
                img_path = tmpdir_path / "anomalies.png"
                img.save(img_path)
                pdf.image(str(img_path), x=10, w=180)
            
            # Page 2 - Faciogrammes
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Faciogrammes", ln=True)
            
            for i, (cat, buf) in enumerate(results['faciogrammes'].items()):
                if i % 2 == 0:
                    pdf.ln(5)
                
                # Conversion correcte de chaque faciogramme
                img = Image.open(buf)
                img_path = tmpdir_path / f"facio_{cat}.png"
                img.save(img_path)
                
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"Mouvement {cat}", ln=True)
                pdf.image(str(img_path), x=10 + (i % 2) * 100, w=90)
            
            # Sauvegarde finale du PDF
            pdf_output_path = tmpdir_path / f"rapport_{nom}_{prenom}.pdf"
            pdf.output(str(pdf_output_path))
            
            # Lecture du PDF généré
            with open(pdf_output_path, "rb") as f:
                pdf_bytes = f.read()
            
            # Bouton de téléchargement
            st.download_button(
                label="📄 Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name=f"rapport_AQMF_{nom}_{prenom}.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.error(f"Erreur lors de la génération du PDF: {str(e)}")
        
        progress_bar.empty()
        status_text.text("Traitement terminé!")