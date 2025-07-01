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
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import pickle
import functions
import visualization
from fpdf import FPDF
from pathlib import Path
import io
from PIL import Image
from datetime import datetime, date

st.set_page_config(page_title="Analyse AQMF", layout="wide")
st.title("Analyse AQMF - Rapport Mouvement Facial")

# Charger la référence
with open("list_hr_mean_norm_9.pkl", "rb") as f:
    list_hr_m = pickle.load(f)

# Collecte des informations patient
st.sidebar.header("Informations du patient")
nom = st.sidebar.text_input("Nom")
prenom = st.sidebar.text_input("Prénom")

# Gestion des dates avec possibilité de dates avant 2010
today = date.today()
min_date = date(1900, 1, 1)  # Date minimale très ancienne
max_date = today

date_naissance = st.sidebar.date_input(
    "Date de naissance",
    value=date(1980, 1, 1),  # Valeur par défaut
    min_value=min_date,
    max_value=max_date
)

date_examen = st.sidebar.date_input(
    "Date de l'examen",
    value=today,
    min_value=min_date,
    max_value=max_date
)

# Upload des fichiers CSV
uploaded_files = st.file_uploader("Importer les fichiers CSV", type="csv", accept_multiple_files=True)

if uploaded_files and nom and prenom:
    st.success("Traitement en cours...")
    
    # Section pour afficher les résultats
    results = st.container()
    
    # Sauvegarde temporaire des fichiers
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for file in uploaded_files:
            file_path = tmpdir_path / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())

        # Regrouper par catégorie
        categories = [f"M{i}" for i in range(1, 10)]
        files_by_category = {cat: [] for cat in categories}
        for file in tmpdir_path.glob("*.csv"):
            for cat in categories:
                if cat in file.name:
                    files_by_category[cat].append(str(file))

        list_px_m_01 = []
        mean_anomalies = {}
        sd_mean_anomalies = {}
        faciograph_images = {}

        # Traitement par catégorie
        for i, cat in enumerate(categories):
            files = files_by_category.get(cat, [])
            if not files:
                continue

            ref = functions.create_list_ref_1_9(files)
            ds = functions.create_list_dataset(ref)
            interp = functions.interpolate_list(ds)
            fixed = functions.create_fixed_duration_dataset(interp, fixed_duration=500)
            dental = functions.dental_frame(fixed)
            disp = functions.displacement_list(dental, norm=True)

            list_px_m_01.append(disp)

            mean_px = np.nanmean([np.nanmean(d[0], axis=0)[3:] for d in disp], axis=0)
            mean_hr = np.nanmean(list_hr_m[i], axis=0)[3:]
            anomaly = np.abs(mean_hr - mean_px)
            mean_anomalies[cat] = np.nanmean(anomaly)
            sd_mean_anomalies[cat] = np.nanstd(anomaly)
            
            # Génération des graphiques faciographiques
            fig_facio = visualization.faciograph_px(mean_hr, mean_px, show=False)
            facio_buf = io.BytesIO()
            fig_facio.savefig(facio_buf, format="JPEG", bbox_inches="tight")
            facio_buf.seek(0)
            faciograph_images[cat] = facio_buf
            plt.close(fig_facio)

        # Affichage des résultats dans Streamlit
        with results:
            st.header("Résultats de l'analyse")
            
            # Graphique d'anomalies
            st.subheader("Anomalies par catégorie")
            fig_anomalies, ax = plt.subplots(figsize=(10, 4))
            means = [mean_anomalies[c] for c in categories]
            stds = [sd_mean_anomalies[c] for c in categories]
            ax.bar(categories, means, yerr=stds, color='lightgray')
            ax.axhline(y=np.nanmean(means), color='red', linestyle='--', label='Moyenne globale')
            ax.set_ylabel("Différences moyennes")
            ax.legend()
            st.pyplot(fig_anomalies)
            
            # Sauvegarde en mémoire pour le PDF
            anomaly_buf = io.BytesIO()
            fig_anomalies.savefig(anomaly_buf, format="JPEG", bbox_inches="tight")
            anomaly_buf.seek(0)
            plt.close(fig_anomalies)
            
            # Affichage des faciogrammes
            st.subheader("Faciogrammes par mouvement")
            cols = st.columns(3)
            for i, cat in enumerate(categories):
                with cols[i % 3]:
                    st.image(faciograph_images[cat], caption=f"Faciogramme {cat}", use_column_width=True)
            
            # Graphique de tous les mouvements
            st.subheader("Tous les mouvements")
            fig_mouvements, axes = plt.subplots(9, 1, figsize=(10, 20))
            concat_px_by_category = {}
            
            for i, category in enumerate(categories):
                concat_data = np.concatenate([data[0] for data in list_px_m_01[i]], axis=0)
                concat_px_by_category[category] = concat_data
                
                axes[i].plot(concat_data)
                axes[i].set_title(f'Déplacements pendant {category}')
                axes[i].set_ylabel('Déplacement normalisé')
                
                num_frames = len(concat_data)
                for frame in range(500, num_frames, 500):
                    axes[i].axvline(x=frame, color='k', linestyle='--', linewidth=1)

            plt.tight_layout()
            st.pyplot(fig_mouvements)
            
            # Sauvegarde en mémoire pour le PDF
            mouvements_buf = io.BytesIO()
            fig_mouvements.savefig(mouvements_buf, format="JPEG", bbox_inches="tight")
            mouvements_buf.seek(0)
            plt.close(fig_mouvements)
            
            # Calcul et affichage de la répétabilité
            st.subheader("Répétabilité des mesures")
            
            def calculate_measure_repeatability(data_list):
                means = [np.nanmean(data[0]) for data in data_list]
                overall_mean = np.nanmean(means)
                std_deviation = np.nanstd(means)
                return overall_mean, std_deviation
            
            repeatability_measure_by_category = {}
            for i, category in enumerate(categories):
                category_data_px = list_px_m_01[i]
                overall_mean, std_deviation = calculate_measure_repeatability(category_data_px)
                repeatability_measure_by_category[category] = (overall_mean, std_deviation)
            
            # Graphique de répétabilité
            categories_r = list(repeatability_measure_by_category.keys())
            means_r = [repeatability_measure_by_category[cat][0] for cat in categories_r]
            std_devs = [repeatability_measure_by_category[cat][1] for cat in categories_r]
            
            fig_repet, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(categories_r, means_r, yerr=std_devs, capsize=5, color='blue', edgecolor='black', alpha=0.7)
            ax.set_ylabel('Moyenne de la répétabilité')
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig_repet)
            
            # Sauvegarde en mémoire pour le PDF
            repetabilite_buf = io.BytesIO()
            fig_repet.savefig(repetabilite_buf, format="JPEG", bbox_inches="tight")
            repetabilite_buf.seek(0)
            plt.close(fig_repet)

        # --- Génération du PDF ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Times", size=12)

        # En-tête
        pdf.cell(0, 10, f"Rapport AQMF - {prenom} {nom}", ln=True)
        pdf.cell(0, 10, f"Date de naissance: {date_naissance.strftime('%d/%m/%Y')}", ln=True)
        pdf.cell(0, 10, f"Date d'examen: {date_examen.strftime('%d/%m/%Y')}", ln=True)
        pdf.ln(10)

        # Page 1 - Anomalies
        pdf.set_font("Times", 'B', 16)
        pdf.cell(0, 10, "Anomalies par catégorie", ln=True)
        pdf.image(anomaly_buf, x=10, y=pdf.get_y(), w=180)
        pdf.ln(85)
        
        # Page 2 - Faciogrammes
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(0, 10, "Faciogrammes", ln=True)
        
        # Organisation en 3 colonnes
        x_positions = [10, 70, 130]
        y_position = 30
        img_width = 50
        
        for i, cat in enumerate(categories):
            x = x_positions[i % 3]
            if i % 3 == 0 and i != 0:
                y_position += 60
            
            pdf.set_xy(x, y_position)
            pdf.set_font("Times", 'B', 12)
            pdf.cell(0, 10, cat, ln=True)
            
            pdf.image(faciograph_images[cat], x=x, y=y_position + 5, w=img_width)
            
            if i == 8:  # Dernier élément
                pdf.ln(60)
        
        # Page 3 - Tous les mouvements
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(0, 10, "Tous les mouvements", ln=True)
        pdf.image(mouvements_buf, x=10, y=pdf.get_y(), w=190)
        
        # Page 4 - Répétabilité
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(0, 10, "Répétabilité des mesures", ln=True)
        pdf.image(repetabilite_buf, x=30, y=pdf.get_y(), w=150)

        # Export PDF en mémoire
        pdf_output = io.BytesIO()
        pdf_output.write(pdf.output(dest='S').encode('latin-1'))
        pdf_output.seek(0)

        # Bouton de téléchargement
        st.download_button(
            label="🔗 Télécharger le rapport PDF complet",
            data=pdf_output,
            file_name=f"rapport_AQMF_{nom}_{prenom}.pdf",
            mime="application/pdf"
        )

else:
    st.info("Veuillez importer les fichiers et remplir les informations patient.")