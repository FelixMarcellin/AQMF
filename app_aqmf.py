# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:28:16 2025

@author: felima
"""

# -*- coding: utf-8 -*-
"""
Analyse AQMF - Version finale avec corrections d'affichage
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import functions
import visualization
from fpdf import FPDF
from pathlib import Path
import io
from datetime import date
import tempfile
from PIL import Image
import os

# Configuration de la page
st.set_page_config(page_title="Analyse AQMF", layout="wide")
st.title("Analyse AQMF - Rapport Mouvement Facial")

def generate_faciogram(mean_hr, mean_px, category, tmpdir_path):
    """Génère un faciogramme et le sauvegarde en PNG"""
    try:
        # Création d'une figure avec fond blanc
        fig = plt.figure(figsize=(10, 8), facecolor='white')
        
        # Appel à la fonction de visualisation originale
        visualization.faciograph_px(mean_hr, mean_px)
        
        # Conversion en image PNG
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        
        # Sauvegarde temporaire
        img_path = tmpdir_path / f"{category}.png"
        with open(img_path, "wb") as f:
            f.write(buf.getbuffer())
            
        plt.close()
        return buf, img_path
    except Exception as e:
        st.error(f"Erreur génération faciogramme {category}: {str(e)}")
        return None, None

# Chargement des données de référence
try:
    with open("list_hr_mean_norm_9.pkl", "rb") as f:
        list_hr_m = pickle.load(f)
except Exception as e:
    st.error(f"Erreur chargement fichier référence: {str(e)}")
    st.stop()

# Formulaire patient
with st.sidebar:
    st.header("Informations patient")
    nom = st.text_input("Nom")
    prenom = st.text_input("Prénom")
    date_naissance = st.date_input("Date naissance", 
                                 value=date(1980, 1, 1),
                                 min_value=date(1900, 1, 1),
                                 max_value=date.today())
    date_examen = st.date_input("Date examen", value=date.today())

# Upload des fichiers CSV
uploaded_files = st.file_uploader("Importer fichiers CSV", 
                                type="csv", 
                                accept_multiple_files=True)

if uploaded_files and nom and prenom:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Sauvegarde des fichiers uploadés
        for i, file in enumerate(uploaded_files):
            file_path = tmpdir_path / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())
            progress_bar.progress((i + 1)/len(uploaded_files))
        
        status_text.text("Analyse des données...")
        
        # Traitement par catégorie
        categories = [f"M{i}" for i in range(1, 10)]
        results = {
            'anomalies': {'means': [], 'stds': [], 'cats': []},
            'faciogrammes': {},
            'mouvements': None
        }
        
        list_px_m_01 = []
        
        for i, cat in enumerate(categories):
            files = [f for f in tmpdir_path.glob("*.csv") if cat in f.name]
            if not files:
                st.warning(f"Pas de fichiers pour {cat}")
                continue
            
            try:
                # Pipeline de traitement
                ref = functions.create_list_ref_1_9([str(f) for f in files])
                ds = functions.create_list_dataset(ref)
                interp = functions.interpolate_list(ds)
                fixed = functions.create_fixed_duration_dataset(interp, 500)
                dental = functions.dental_frame(fixed)
                disp = functions.displacement_list(dental, norm=True)
                
                list_px_m_01.append(disp)
                
                # Calcul des anomalies
                mean_px = np.nanmean([np.nanmean(d[0], axis=0)[3:] for d in disp], axis=0)
                mean_hr = np.nanmean(list_hr_m[i], axis=0)[3:]
                anomaly = np.abs(mean_hr - mean_px)
                
                results['anomalies']['means'].append(np.nanmean(anomaly))
                results['anomalies']['stds'].append(np.nanstd(anomaly))
                results['anomalies']['cats'].append(cat)
                
                # Génération faciogramme
                facio_buf, facio_path = generate_faciogram(mean_hr, mean_px, cat, tmpdir_path)
                if facio_buf:
                    results['faciogrammes'][cat] = {'buf': facio_buf, 'path': facio_path}
                    
                    # Affichage dans Streamlit
                    with st.expander(f"Faciogramme {cat}"):
                        st.image(facio_path, use_column_width=True)
                
            except Exception as e:
                st.error(f"Erreur {cat}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1)/len(categories))
        
        # Graphique anomalies
        if results['anomalies']['means']:
            st.header("Résultats analyse")
            
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
            ax.bar(results['anomalies']['cats'],
                  results['anomalies']['means'],
                  yerr=results['anomalies']['stds'],
                  color='skyblue')
            ax.axhline(np.mean(results['anomalies']['means']),
                     color='red', linestyle='--')
            ax.set_title("Anomalies par catégorie")
            st.pyplot(fig)
            
            # Sauvegarde pour PDF
            anomaly_path = tmpdir_path / "anomalies.png"
            fig.savefig(anomaly_path, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        # Génération PDF
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(True, 15)
            
            # Page 1 - Entête
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, f"Rapport AQMF - {prenom} {nom}", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f"Date naissance: {date_naissance.strftime('%d/%m/%Y')}", ln=True)
            pdf.cell(0, 10, f"Date examen: {date_examen.strftime('%d/%m/%Y')}", ln=True)
            
            # Graphique anomalies
            if os.path.exists(anomaly_path):
                pdf.ln(10)
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Anomalies par catégorie", ln=True)
                pdf.image(str(anomaly_path), w=180)
            
            # Page 2 - Faciogrammes
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Faciogrammes", ln=True)
            
            for i, (cat, data) in enumerate(results['faciogrammes'].items()):
                if i % 2 == 0:
                    pdf.ln(5)
                
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"Mouvement {cat}", ln=True)
                pdf.image(str(data['path']), w=90, x=10 + (i % 2) * 100)
            
            # Sauvegarde PDF
            pdf_path = tmpdir_path / "rapport.pdf"
            pdf.output(str(pdf_path))
            
            # Bouton téléchargement
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📄 Télécharger rapport PDF",
                    data=f,
                    file_name=f"rapport_AQMF_{nom}_{prenom}.pdf",
                    mime="application/pdf"
                )
            
        except Exception as e:
            st.error(f"Erreur génération PDF: {str(e)}")
        
        progress_bar.empty()
        status_text.success("Analyse terminée!")
        st.balloons()
else:
    st.info("Veuillez importer des fichiers CSV et compléter les informations patient")