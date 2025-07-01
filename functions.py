import numpy as np
from numpy import linalg as LA
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
import matplotlib.pyplot as plt

def create_list_ref(ref_csv):
    list_M = [[] for _ in range(5)]
    for elem in ref_csv:
        id_M = elem[-5]
        list_M[int(id_M)-1].append(elem)
    return list_M



def create_list_ref_1_9(ref_csv):
    list_M = [[] for _ in range(9)]  # 9 mouvements avec chacun 5 fichiers
    counts = [0] * 9  # Un compteur pour chaque mouvement

    for elem in ref_csv:
        if len(elem) >= 5 and elem[-5].isdigit():  # Vérifie que l'élément a au moins 5 caractères et que le 5e dernier est un chiffre
            id_M = int(elem[-5])
            if 1 <= id_M <= 9:  # Vérifie que l'ID est bien entre 1 et 9
                # Ajoute l'élément seulement si la sous-liste contient moins de 5 fichiers
                if counts[id_M - 1] < 5:
                    list_M[id_M - 1].append(elem)
                    counts[id_M - 1] += 1  # Incrémente le compteur pour ce mouvement
    
    # Ne garder que les sous-listes remplies (qui contiennent au moins 1 élément)
    list_M_filled = [sublist for sublist in list_M if sublist]
    
    return list_M_filled




def create_list_ref_6_9(ref_csv):
    # Initialiser 4 sous-listes vides pour M6, M7, M8 et M9
    list_M = [[] for _ in range(4)]
    
    for elem in ref_csv:
        try:
            # Imprimer l'élément pour vérifier sa structure
            print(f"Processing element: {elem}")
            
            # Extraire id_M du cinquième élément en partant de la fin (ou ajustez selon votre structure)
            id_M = int(elem[-5])  # Assurez-vous que l'index correspond bien à la position de id_M
            
            # Vérifiez si id_M est dans la plage de 6 à 9
            if 6 <= id_M <= 9:
                list_M[id_M - 6].append(elem)  # Ajuster l'index pour correspondre à la liste de 0 à 3
            else:
                print(f"Warning: id_M {id_M} out of expected range (6-9) for element {elem}")
        except (ValueError, IndexError) as e:
            print(f"Error processing element {elem}: {e}")
    
    return list_M


def create_dict_sample(df_sample):
    dict_sample = {}
    for i in range(108):
        x = []
        y = []
        z = []
        #marker_list = []
        df_x = df_sample.iloc[:, 3*i]
        x = df_x.values.tolist()
        df_y = df_sample.iloc[:, 3*i+1]
        y = df_y.values.tolist()
        df_z = df_sample.iloc[:, 3*i+2]
        z = df_z.values.tolist()
        marker_dict = {'x':np.array(x), 'y':np.array(y), 'z':np.array(z)}
        dict_sample[i] = marker_dict 
    return dict_sample

def get_all_markers(): # different file
    pass

def get_marker(list_markers): # TODO
    selected_markers = []
    all_markers = get_all_markers() # TODO
    for elem in list_markers:
        index = all_markers.index(elem)
        selected_markers.append(index)
    return selected_markers

def create_fixed_duration_dataset(list_dataset, fixed_duration=500):
    list_fixed_duration_dataset = []
    for list_M in list_dataset:
        list_M_fixed = []
        for csv_sample in list_M:
            np_sample = csv_sample.to_numpy()
            np_fixed = np.zeros((fixed_duration, 324))
            for m in range(324):
                np_fixed[:, m] = interpolate_signal(np_sample[:, m], np_sample[:, m].shape[0], fixed_duration)
            list_M_fixed.append(np_fixed)
        list_fixed_duration_dataset.append(list_M_fixed)
    return list_fixed_duration_dataset
    
def interpolate_signal(insignal, len1, len2):
    x1 = np.linspace(0, len2-1, len1)
    f = interp1d(x1, insignal, axis=0, kind='cubic')
    x2 = np.linspace(0, len2-1, len2)
    return f(x2)
    
def list_to_interpolate(input_list, fixed_duration):
    inter_list = []
    for elem in input_list:
        np_int = interpolate_signal(elem, elem.shape[0], fixed_duration)
        inter_list.append(np_int)
    return inter_list

def create_list_dataset(list_ref):
    list_dataset = []
    for sub_list in list_ref:
        list_M = []
        for sample_path in sub_list:
            #print(sub_sub_list)
            # this is a matrix
            df_sample = pd.read_csv(sample_path, skiprows=5, header=None, usecols=[i+2 for i in range(324)])
            # this is the dict of a sample defined in x, y, z and t
            df_cleaned = clean_dataframe(df_sample)
            list_M.append(df_cleaned)
        list_dataset.append(list_M)
    return list_dataset

def clean_dataframe(df_sample):
    np_sample = df_sample.to_numpy()
    np_cleaned = np.copy(np_sample)
    t_norm = len(np_cleaned)
    t_n_ctr = int(np.floor(t_norm/2))
    for m in range(324):
        if np.isnan(np_cleaned[0:10, m]).any(): # nan for the beginning
            np_cleaned[:, m] = np.nan
        if np.isnan(np_cleaned[t_n_ctr - 50:t_n_ctr + 50, m]).any(): # nan for the middle
            np_cleaned[:, m] = np.nan           
        if np.isnan(np_cleaned[-50:-1, m]).any(): # nan for the end
            np_cleaned[:, m] = np.nan
    df_cleaned = pd.DataFrame(np_cleaned)         
    return df_cleaned

def interpolate_list(list_dataset):
    list_ds_interpolated = []
    for sub_list in list_dataset:
        list_M_interpolated = []
        for elem in sub_list:
            df_interpolated = elem.interpolate('spline', order=2)
            list_M_interpolated.append(df_interpolated)
        list_ds_interpolated.append(list_M_interpolated)
    return list_ds_interpolated

def dental_frame(list_ds):
    list_dental_frame = []
    for list_M in list_ds:
        list_np_dental_samples = []
        for np_sample in list_M:
            # m1, m2 and m3 are the vectors that defines the dental frame and o is its origin
            m1 = np.array(np_sample[:, 0:3])
            m2 = np.array(np_sample[:, 3:6])
            m3 = np.array(np_sample[:, 6:9])
            o = (m1[0] + m2[0] + m3[0])/3
            
            # compute the reference frame
            x_mag = LA.norm(m1[0]-o)
            x = (m1[0] - o)/x_mag
            y_mag = LA.norm(np.cross(m2[0] - m3[0], x))
            y = np.cross(m2[0] - m3[0], x) / y_mag
            z = np.cross(x, y)
            M = np.column_stack((x, y, z))
            M = [x, y, z] 
            Mo = np.column_stack((M, o))
            Mo_h = np.concatenate((Mo, [[0,0,0,1]])) #this is the homogeneous matrix
            
            np_sample_dental = np.zeros((len(np_sample) ,324))
            # "s" stands for step
            for i in range(len(np_sample)): 
                s_0 = np_sample[i] # take a row
                for j in range(3, 108):
                    s_1 = s_0[3*j : 3*j+ 3] # select a 3D point at timestep j
                    s_2 = np.append(s_1, 1) # convert it to homogeneous vector
                    s_3 = np.reshape(s_2, (4,1)) # convert it as a column vector
                    s_4 = np.matmul(Mo_h, s_3) # compute the transformation as homogeneous vector                  
                    np_sample_dental[i][j*3] = s_4[0]
                    np_sample_dental[i][j*3 + 1] = s_4[1]
                    np_sample_dental[i][j*3 + 2] = s_4[2]
            list_np_dental_samples.append(np_sample_dental)
        list_dental_frame.append(list_np_dental_samples)
    return list_dental_frame

def displacement_list(list_ds_dental, norm = False): #experimental
    list_ds_displacement = []
    for list_M in list_ds_dental:
        list_M_disp = []
        for np_sample in list_M:
            np_sample_disp = np.zeros((len(np_sample), 108))
            for i in range(len(np_sample)):
                for j in range(108):
                    X0 = np.array([np_sample[0][3*j], np_sample[0][3*j + 1], np_sample[0][3*j + 2]])
                    Xt = np.array([np_sample[i][3*j], np_sample[i][3*j + 1], np_sample[i][3*j + 2]])
                    dt = np.linalg.norm(X0 - Xt)
                    np_sample_disp[i][j] = dt
            if norm:
                list_M_disp.append(np_sample_disp / np.nanmax(np_sample_disp))
            else:
                list_M_disp.append(np_sample_disp)
        list_ds_displacement.append(list_M_disp)
    return list_ds_displacement

def compute_mean(list_ds, fixed_len = 500):
    list_means_ds = []
    for list_M in list_ds:
        ndarray_M = np.array(list_M)
        nda_M_mean = np.nanmean(ndarray_M, axis=0)
        list_means_ds.append(nda_M_mean)
    return list_means_ds
    
def compute_std(list_ds, fixed_len = 500):
    list_std_ds = []
    for list_M in list_ds:
        ndarray_M = np.array(list_M)
        nda_M_mean = np.nanstd(ndarray_M, axis=0)
        list_std_ds.append(nda_M_mean)
    return list_std_ds    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    