# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:55:26 2024

@author: felima
"""

import os
import pathlib
import numpy as np  # Import NumPy for numerical operations
import tkinter as tk
from tkinter import filedialog
import functions  # Assurez-vous que les fonctions nécessaires sont importées
import numpy as np
import pickle

#%% Sélection du dossier contenant les fichiers CSV
def select_directory():
    root = tk.Tk()
    root.withdraw()  # Cacher la fenêtre principale.
    directory_path = filedialog.askdirectory()  # Afficher la boîte de dialogue pour choisir un dossier.
    folder_name = directory_path.split('/')[-1]  # Extraire le nom du dossier.
    print(f"Selected Folder: {folder_name}")
    return directory_path

filepath = select_directory()
print(filepath)

#%% Ouverture des fichiers CSV
ref_csv = sorted(str(p) for p in pathlib.Path(filepath).glob("*.csv"))

# #%% Regroupement des fichiers par catégorie
# categories = ['M6', 'M7', 'M8', 'M9']
# files_by_category = {category: [] for category in categories}

# # Ajouter chaque fichier à la bonne catégorie en fonction du nom de fichier
# for file in ref_csv:
#     for category in categories:
#         if file.endswith(f"{category}.csv"):  # Vérifier si la catégorie est à la fin du nom du fichier
#             files_by_category[category].append(file)
#             break  # Sortir de la boucle une fois que la catégorie correspondante est trouvée

#%% Regroupement des fichiers par catégorie
categories = ['M6', 'M7', 'M8', 'M9']
files_by_category = {category: [] for category in categories}

# Ajouter chaque fichier à la bonne catégorie en fonction du nom de fichier
for file in ref_csv:
    for category in categories:
        if category in os.path.basename(file):  # Vérifier si la catégorie est dans le nom du fichier
            files_by_category[category].append(file)
            break  # Sortir de la boucle une fois que la catégorie correspondante est trouvée

#%% Traitement des fichiers par catégorie
list_px_m_01 = []  # Initialisation de la liste finale

# Boucle sur les catégories (M6, M7, M8, M9)
for i, category in enumerate(categories):
    print(f"Processing category: {category}")
    
    files = files_by_category[category]  # Obtenir les fichiers de la catégorie actuelle
    
    # Pour chaque catégorie, traiter les fichiers correspondants
    list_ref = functions.create_list_ref_6_9(files)  # Créer une référence à partir des fichiers
    list_dataset = functions.create_list_dataset(list_ref)  # Créer un dataset
    
    # Interpolation et nettoyage
    list_ds_interpolated = functions.interpolate_list(list_dataset)
    # Ajuster à une durée fixe de 500 frames
    list_ds_int_fix_dataset = functions.create_fixed_duration_dataset(list_ds_interpolated, fixed_duration=500)
    # Transformation au repère dentaire
    list_ds_int_fix_dental = functions.dental_frame(list_ds_int_fix_dataset)
    # Calcul du déplacement, si applicable
    list_ds_int_fix_dental_disp = functions.displacement_list(list_ds_int_fix_dental, norm=True)
    
    # Stocker les données traitées pour chaque catégorie dans list_px_m_01
    list_px_m_01.append(list_ds_int_fix_dental_disp)  # Ajout d'une liste par catégorie dans list_px_m_01

# À ce stade, `list_px_m_01` est une liste de listes, chaque sous-liste contenant 5 fichiers pour une catégorie



#%% Initialiser une nouvelle liste pour stocker les sous-listes sélectionnées
selected_sub_lists = []

# Accéder à la première sous-liste de la première catégorie
if len(list_px_m_01[0]) > 0:  # Vérifier qu'il y a au moins une sous-liste dans la première liste
    selected_sub_lists.append(list_px_m_01[0][0])  # Ajouter la première sous-liste de la première catégorie

# Accéder à la deuxième sous-liste de la deuxième catégorie
if len(list_px_m_01[1]) > 1:  # Vérifier qu'il y a au moins deux sous-listes dans la deuxième liste
    selected_sub_lists.append(list_px_m_01[1][1])  # Ajouter la deuxième sous-liste de la deuxième catégorie

# Accéder à la troisième sous-liste de la troisième catégorie
if len(list_px_m_01[2]) > 2:  # Vérifier qu'il y a au moins trois sous-listes dans la troisième liste
    selected_sub_lists.append(list_px_m_01[2][2])  # Ajouter la troisième sous-liste de la troisième catégorie

# Accéder à la quatrième sous-liste de la quatrième catégorie
if len(list_px_m_01[3]) > 3:  # Vérifier qu'il y a au moins quatre sous-listes dans la quatrième liste
    selected_sub_lists.append(list_px_m_01[3][3])  # Ajouter la quatrième sous-liste de la quatrième catégorie

# Afficher la nouvelle liste
print("Selected sub-lists:", selected_sub_lists)


#%%

# Initialiser une liste pour stocker les moyennes
mean_arrays = []

# Calculer la moyenne pour chaque sous-liste
for sublist in selected_sub_lists:
    # Convertir la sous-liste en un array 3D de forme (10, 500, 108)
    sublist_array = np.array(sublist)
    
    # Calculer la moyenne sur le premier axe (celui qui correspond aux 10 matrices)
    mean_array = np.nanmean(sublist_array, axis=0)
    
    # Ajouter l'array moyen à la liste
    mean_arrays.append(mean_array)

# Maintenant mean_arrays contient 4 arrays, chacun de dimension (500, 108)


#%% Ouverture fichier sain et ajout des 4 mouvements supplémentaires
with open('list_hr_mean_norm.pkl', 'rb') as file:
    list_hr_m = pickle.load(file)  # Chargement de la référence saine, un tableau par catégorie

list_hr_m.extend(mean_arrays)

#%% enregistrement de la nouvelle ref saine de 9 mouvement

# Nom du fichier pickle
filename = "list_hr_mean_norm_9.pkl"

# Enregistrer la liste dans un fichier pickle
with open(filename, 'wb') as f:
    pickle.dump(list_hr_m, f)

print(f"La liste a été enregistrée sous le nom {filename}")


