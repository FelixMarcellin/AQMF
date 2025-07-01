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

st.set_page_config(page_title="Analyse AQMF", layout="wide")
st.title("Analyse AQMF - Rapport Mouvement Facial")

# Charger la référence
with open("list_hr_mean_norm_9.pkl", "rb") as f:
    list_hr_m = pickle.load(f)

# Collecte des informations patient
st.sidebar.header("Informations du patient")
nom = st.sidebar.text_input("Nom")
prenom = st.sidebar.text_input("Prénom")
date_naissance = st.sidebar.date_input("Date de naissance")
date_examen = st.sidebar.date_input("Date de l'examen")

# Upload des fichiers CSV
uploaded_files = st.file_uploader("Importer les fichiers CSV", type="csv", accept_multiple_files=True)

if uploaded_files and nom and prenom:
    st.success("Traitement en cours...")

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

        # Création du graphique d'anomalies
        fig, ax = plt.subplots(figsize=(10, 4))
        means = [mean_anomalies[c] for c in categories]
        stds = [sd_mean_anomalies[c] for c in categories]
        ax.bar(categories, means, yerr=stds, color='lightgray')
        ax.axhline(y=np.nanmean(means), color='red', linestyle='--', label='Moyenne globale')
        ax.set_ylabel("Différences moyennes")
        ax.set_title("Anomalies par catégorie")
        ax.legend()

        # Affichage et sauvegarde en mémoire
        st.pyplot(fig)
        anomaly_buf = io.BytesIO()
        fig.savefig(anomaly_buf, format="JPEG", bbox_inches="tight")
        anomaly_buf.seek(0)

        # --- PDF ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Times", size=12)

        pdf.cell(0, 10, f"Rapport AQMF - {prenom} {nom}", ln=True)
        pdf.cell(0, 10, f"Date de naissance: {date_naissance.strftime('%d/%m/%Y')}", ln=True)
        pdf.cell(0, 10, f"Date d'examen: {date_examen.strftime('%d/%m/%Y')}", ln=True)
        pdf.ln(10)

        # Sauvegarde temporaire de l'image
        temp_img_path = tmpdir_path / "anomalies.jpg"
        with open(temp_img_path, "wb") as f:
            f.write(anomaly_buf.read())

        pdf.image(str(temp_img_path), w=180)

        # Export PDF en mémoire
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)

        # Bouton de téléchargement
        st.download_button(
            label="🔗 Télécharger le rapport PDF",
            data=pdf_output,
            file_name=f"rapport_AQMF_{nom}_{prenom}.pdf",
            mime="application/pdf"
        )

else:
    st.info("Veuillez importer les fichiers et remplir les informations patient.")
