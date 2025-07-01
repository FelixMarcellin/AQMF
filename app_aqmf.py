# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:28:16 2025

@author: felima
"""
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
from fpdf import FPDF
from pathlib import Path
import io
from datetime import date
import tempfile

st.set_page_config(page_title="Analyse AQMF", layout="wide")
st.title("Analyse AQMF - Rapport Mouvement Facial")

# Charger la référence
try:
    with open("list_hr_mean_norm_9.pkl", "rb") as f:
        list_hr_m = pickle.load(f)
except FileNotFoundError:
    st.error("Fichier de référence list_hr_mean_norm_9.pkl introuvable")
    st.stop()

# Collecte des informations patient
st.sidebar.header("Informations du patient")
nom = st.sidebar.text_input("Nom")
prenom = st.sidebar.text_input("Prénom")

# Gestion des dates avec possibilité de dates avant 2010
min_date = date(1900, 1, 1)
max_date = date.today()

date_naissance = st.sidebar.date_input(
    "Date de naissance",
    value=date(1980, 1, 1),
    min_value=min_date,
    max_value=max_date
)

date_examen = st.sidebar.date_input(
    "Date de l'examen",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

# Upload des fichiers CSV
uploaded_files = st.file_uploader("Importer les fichiers CSV", type="csv", accept_multiple_files=True)

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
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Traitement des données...")
        
        # Regroupement par catégorie
        categories = [f"M{i}" for i in range(1, 10)]
        files_by_category = {cat: [] for cat in categories}
        
        for file in tmpdir_path.glob("*.csv"):
            for cat in categories:
                if cat in file.name:
                    files_by_category[cat].append(str(file))
                    break
        
        # Traitement des données
        list_px_m_01 = []
        mean_anomalies = {}
        sd_mean_anomalies = {}
        faciograph_buffers = {}
        
        for i, cat in enumerate(categories):
            files = files_by_category.get(cat, [])
            if not files:
                st.warning(f"Aucun fichier trouvé pour la catégorie {cat}")
                continue
            
            try:
                # Traitement des données
                ref = functions.create_list_ref_1_9(files)
                ds = functions.create_list_dataset(ref)
                interp = functions.interpolate_list(ds)
                fixed = functions.create_fixed_duration_dataset(interp, fixed_duration=500)
                dental = functions.dental_frame(fixed)
                disp = functions.displacement_list(dental, norm=True)
                
                list_px_m_01.append(disp)
                
                # Calcul des anomalies
                mean_px = np.nanmean([np.nanmean(d[0], axis=0)[3:] for d in disp], axis=0)
                mean_hr = np.nanmean(list_hr_m[i], axis=0)[3:]
                anomaly = np.abs(mean_hr - mean_px)
                mean_anomalies[cat] = np.nanmean(anomaly)
                sd_mean_anomalies[cat] = np.nanstd(anomaly)
                
                # Génération des faciogrammes avec gestion d'erreur
                try:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(mean_hr, label='Référence saine')
                    ax.plot(mean_px, label='Patient')
                    ax.set_title(f"Faciogramme {cat}")
                    ax.set_xlabel("Frames")
                    ax.set_ylabel("Déplacement normalisé")
                    ax.legend()
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    faciograph_buffers[cat] = buf
                    plt.close(fig)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la génération du faciogramme {cat}: {str(e)}")
                    continue
                    
            except Exception as e:
                st.error(f"Erreur lors du traitement de la catégorie {cat}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(categories))
        
        status_text.text("Affichage des résultats...")
        
        # Affichage des résultats
        if not mean_anomalies:
            st.error("Aucune donnée valide n'a pu être traitée")
            st.stop()
        
        # Section Anomalies
        st.header("Analyse des anomalies")
        
        fig_anom, ax = plt.subplots(figsize=(10, 5))
        means = [mean_anomalies[c] for c in categories if c in mean_anomalies]
        valid_cats = [c for c in categories if c in mean_anomalies]
        
        if sd_mean_anomalies:
            stds = [sd_mean_anomalies[c] for c in valid_cats]
            ax.bar(valid_cats, means, yerr=stds, color='skyblue')
        else:
            ax.bar(valid_cats, means, color='skyblue')
        
        ax.axhline(y=np.nanmean(means), color='red', linestyle='--', label='Moyenne globale')
        ax.set_ylabel("Différences moyennes")
        ax.set_title("Anomalies par catégorie")
        ax.legend()
        st.pyplot(fig_anom)
        
        # Section Faciogrammes
        st.header("Faciogrammes par mouvement")
        cols = st.columns(3)
        
        for i, cat in enumerate(categories):
            if cat in faciograph_buffers:
                with cols[i % 3]:
                    st.image(faciograph_buffers[cat], caption=f"Faciogramme {cat}", use_column_width=True)
        
        # Génération du PDF
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # En-tête
            pdf.cell(0, 10, f"Rapport AQMF - {prenom} {nom}", ln=True)
            pdf.cell(0, 10, f"Date de naissance: {date_naissance.strftime('%d/%m/%Y')}", ln=True)
            pdf.cell(0, 10, f"Date d'examen: {date_examen.strftime('%d/%m/%Y')}", ln=True)
            pdf.ln(10)
            
            # Graphique des anomalies
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Anomalies par catégorie", ln=True)
            
            anomaly_buf = io.BytesIO()
            fig_anom.savefig(anomaly_buf, format="png", bbox_inches="tight", dpi=300)
            anomaly_buf.seek(0)
            pdf.image(anomaly_buf, x=10, w=180)
            plt.close(fig_anom)
            
            # Faciogrammes
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Faciogrammes", ln=True)
            
            for i, cat in enumerate(categories):
                if cat in faciograph_buffers:
                    if i % 2 == 0:
                        pdf.ln(5)
                    
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, f"Mouvement {cat}", ln=True)
                    pdf.image(faciograph_buffers[cat], x=10 + (i % 2) * 100, w=90)
            
            # Génération du PDF en mémoire
            pdf_output = io.BytesIO()
            pdf_output.write(pdf.output(dest='S').encode('latin-1'))
            pdf_output.seek(0)
            
            # Bouton de téléchargement
            st.download_button(
                label="📄 Télécharger le rapport PDF",
                data=pdf_output,
                file_name=f"rapport_AQMF_{nom}_{prenom}.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.error(f"Erreur lors de la génération du PDF: {str(e)}")
        
        progress_bar.empty()
        status_text.text("Traitement terminé!")
        
else:
    st.info("Veuillez importer les fichiers CSV et remplir les informations patient.")