# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:28:16 2025

@author: felima
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

# Fonction pour générer les faciogrammes
def generate_faciogram(mean_hr, mean_px, category):
    try:
        fig = visualization.faciograph_px(mean_hr, mean_px, show=False)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        st.error(f"Erreur génération faciogramme {category}: {str(e)}")
        return None

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
            'anomalies': {'means': [], 'stds': [], 'cats': []},
            'faciogrammes': {},
            'mouvements': None
        }
        
        list_px_m_01 = []
        
        for i, cat in enumerate(categories):
            files = [f for f in tmpdir_path.glob("*.csv") if cat in f.name]
            if not files:
                st.warning(f"Aucun fichier trouvé pour {cat}")
                continue
            
            try:
                # Traitement des données
                ref = functions.create_list_ref_1_9([str(f) for f in files])
                ds = functions.create_list_dataset(ref)
                interp = functions.interpolate_list(ds)
                fixed = functions.create_fixed_duration_dataset(interp, fixed_duration=500)
                dental = functions.dental_frame(fixed)
                disp = functions.displacement_list(dental, norm=True)
                
                list_px_m_01.append(disp)
                
                # Calcul des moyennes
                mean_px = np.nanmean([np.nanmean(d[0], axis=0)[3:] for d in disp], axis=0)
                mean_hr = np.nanmean(list_hr_m[i], axis=0)[3:]
                anomaly = np.abs(mean_hr - mean_px)
                
                results['anomalies']['means'].append(np.nanmean(anomaly))
                results['anomalies']['stds'].append(np.nanstd(anomaly))
                results['anomalies']['cats'].append(cat)
                
                # Génération du faciogramme
                facio_buf = generate_faciogram(mean_hr, mean_px, cat)
                if facio_buf:
                    results['faciogrammes'][cat] = facio_buf
                    
                    # Affichage dans Streamlit
                    with st.expander(f"Faciograph {cat}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(facio_buf, caption="Format standard")
                        with col2:
                            jpg_path = tmpdir_path / f"{cat}.jpg"
                            Image.open(facio_buf).save(jpg_path, "JPEG")
                            st.image(str(jpg_path), caption="Format JPG")
                
            except Exception as e:
                st.error(f"Erreur avec {cat}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(categories))
        
        # Graphique des anomalies
        if results['anomalies']['means']:
            st.header("Résultats d'analyse")
            
            fig_anom, ax = plt.subplots(figsize=(10, 5))
            ax.bar(results['anomalies']['cats'], 
                  results['anomalies']['means'],
                  yerr=results['anomalies']['stds'],
                  color='skyblue')
            ax.axhline(y=np.mean(results['anomalies']['means']), 
                      color='red', linestyle='--', label='Moyenne')
            ax.set_title("Anomalies par catégorie")
            ax.set_ylabel("Différence moyenne (mm)")
            ax.legend()
            st.pyplot(fig_anom)
            results['fig_anomalies'] = io.BytesIO()
            fig_anom.savefig(results['fig_anomalies'], format='png', bbox_inches='tight')
            plt.close(fig_anom)
        
        # Génération du PDF
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Page 1 - En-tête
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, f"Rapport AQMF - {prenom} {nom}", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f"Date de naissance: {date_naissance.strftime('%d/%m/%Y')}", ln=True)
            pdf.cell(0, 10, f"Date d'examen: {date_examen.strftime('%d/%m/%Y')}", ln=True)
            pdf.ln(10)
            
            # Graphique des anomalies
            if 'fig_anomalies' in results:
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Anomalies par catégorie", ln=True)
                img_path = tmpdir_path / "anomalies.png"
                with open(img_path, "wb") as f:
                    f.write(results['fig_anomalies'].getvalue())
                pdf.image(str(img_path), x=10, w=180)
            
            # Faciogrammes
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Faciographs", ln=True)
            
            for i, (cat, buf) in enumerate(results['faciographs'].items()):
                if i % 2 == 0:
                    pdf.ln(5)
                
                # Sauvegarde temporaire en JPG pour le PDF
                img_path = tmpdir_path / f"{cat}.jpg"
                with open(img_path, "wb") as f:
                    f.write(buf.getvalue())
                
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"Mouvement {cat}", ln=True)
                pdf.image(str(img_path), x=10 + (i % 2) * 100, w=90)
            
            # Sauvegarde finale
            pdf_path = tmpdir_path / "rapport_final.pdf"
            pdf.output(str(pdf_path))
            
            # Bouton de téléchargement
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Télécharger le rapport complet (PDF)",
                    data=f,
                    file_name=f"rapport_AQMF_{nom}_{prenom}.pdf",
                    mime="application/pdf"
                )
            
        except Exception as e:
            st.error(f"Erreur lors de la génération du PDF: {str(e)}")
        
        progress_bar.empty()
        status_text.text("Analyse terminée avec succès!")
        st.balloons()
else:
    st.info("Veuillez importer les fichiers CSV et remplir les informations patient.")