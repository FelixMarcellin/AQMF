#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:21:33 2022

@author: rodriguez
"""

import numpy as np
import matplotlib.pyplot as plt


def faciograph(center_np, health_np, up_np, down_np, save=True, image_name = "faciograph.svg"):#, anomaly_c_np = None):

    center_np_order1=np.array([[center_np[6]],[center_np[5]],[center_np[4]],[center_np[17]],[center_np[16]],[center_np[15]],[center_np[14]],[center_np[13]],[center_np[31]],[center_np[30]],
    [center_np[29]],[center_np[28]],[center_np[27]],[center_np[103]],[center_np[104]],[center_np[34]],[center_np[35]],[center_np[36]],[center_np[37]],[center_np[38]],
    [center_np[39]],[center_np[33]],[center_np[32]],[center_np[92]],[center_np[91]],[center_np[90]],[center_np[89]],[center_np[88]],                                                                                                                                          
    [center_np[83]],[center_np[84]],[center_np[85]],[center_np[86]],[center_np[87]],[center_np[82]],[center_np[81]],[center_np[80]],[center_np[79]],[center_np[78]],
    [center_np[77]],[center_np[40]],[center_np[41]],[center_np[42]],[center_np[67]],[center_np[74]],[center_np[75]],[center_np[76]],[center_np[73]],[center_np[72]]])                                                                                                                                    
    
    center_np_order2=np.array([[center_np[68]],[center_np[69]],[center_np[58]],[center_np[57]],[center_np[56]],[center_np[65]],[center_np[43]],[center_np[44]],[center_np[45]],[center_np[59]],
    [center_np[60]],[center_np[61]],[center_np[62]],[center_np[63]],[center_np[64]],[center_np[55]],[center_np[54]],[center_np[53]],[center_np[52]],[center_np[51]],
    [center_np[46]],[center_np[47]],[center_np[48]],[center_np[49]],[center_np[50]],[center_np[100]],[center_np[99]],[center_np[93]],[center_np[98]],[center_np[97]],                                                                                                                                          
    [center_np[96]],[center_np[95]],[center_np[94]],[center_np[102]],[center_np[101]],[center_np[26]],[center_np[25]],[center_np[24]],[center_np[23]],[center_np[22]],
    [center_np[11]],[center_np[10]],[center_np[9]],[center_np[8]],[center_np[7]],[center_np[2]],[center_np[1]],[center_np[0]]])                                                                                                                                    
    center_np_reorder = np.concatenate((center_np_order1,center_np_order2),axis=None)

    health_np_order1=np.array([[health_np[6]],[health_np[5]],[health_np[4]],[health_np[17]],[health_np[16]],[health_np[15]],[health_np[14]],[health_np[13]],[health_np[31]],[health_np[30]],
    [health_np[29]],[health_np[28]],[health_np[27]],[health_np[103]],[health_np[104]],[health_np[34]],[health_np[35]],[health_np[36]],[health_np[37]],[health_np[38]],
    [health_np[39]],[health_np[33]],[health_np[32]],[health_np[92]],[health_np[91]],[health_np[90]],[health_np[89]],[health_np[88]],                                                                                                                                          
    [health_np[83]],[health_np[84]],[health_np[85]],[health_np[86]],[health_np[87]],[health_np[82]],[health_np[81]],[health_np[80]],[health_np[79]],[health_np[78]],
    [health_np[77]],[health_np[40]],[health_np[41]],[health_np[42]],[health_np[67]],[health_np[74]],[health_np[75]],[health_np[76]],[health_np[73]],[health_np[72]]])                                                                                                                                    
    
    health_np_order2=np.array([[health_np[68]],[health_np[69]],[health_np[58]],[health_np[57]],[health_np[56]],[health_np[65]],[health_np[43]],[health_np[44]],[health_np[45]],[health_np[59]],
    [health_np[60]],[health_np[61]],[health_np[62]],[health_np[63]],[health_np[64]],[health_np[55]],[health_np[54]],[health_np[53]],[health_np[52]],[health_np[51]],
    [health_np[46]],[health_np[47]],[health_np[48]],[health_np[49]],[health_np[50]],[health_np[100]],[health_np[99]],[health_np[93]],[health_np[98]],[health_np[97]],                                                                                                                                          
    [health_np[96]],[health_np[95]],[health_np[94]],[health_np[102]],[health_np[101]],[health_np[26]],[health_np[25]],[health_np[24]],[health_np[23]],[health_np[22]],
    [health_np[11]],[health_np[10]],[health_np[9]],[health_np[8]],[health_np[7]],[health_np[2]],[health_np[1]],[health_np[0]]])                                                                                                                                    
    health_np_reorder = np.concatenate((health_np_order1,health_np_order2),axis=None)

    up_np_order1=np.array([[up_np[6]],[up_np[5]],[up_np[4]],[up_np[17]],[up_np[16]],[up_np[15]],[up_np[14]],[up_np[13]],[up_np[31]],[up_np[30]],
    [up_np[29]],[up_np[28]],[up_np[27]],[up_np[103]],[up_np[104]],[up_np[34]],[up_np[35]],[up_np[36]],[up_np[37]],[up_np[38]],
    [up_np[39]],[up_np[33]],[up_np[32]],[up_np[92]],[up_np[91]],[up_np[90]],[up_np[89]],[up_np[88]],                                                                                                                                          
    [up_np[83]],[up_np[84]],[up_np[85]],[up_np[86]],[up_np[87]],[up_np[82]],[up_np[81]],[up_np[80]],[up_np[79]],[up_np[78]],
    [up_np[77]],[up_np[40]],[up_np[41]],[up_np[42]],[up_np[67]],[up_np[74]],[up_np[75]],[up_np[76]],[up_np[73]],[up_np[72]]])                                                                                                                                    
    
    up_np_order2=np.array([[up_np[68]],[up_np[69]],[up_np[58]],[up_np[57]],[up_np[56]],[up_np[65]],[up_np[43]],[up_np[44]],[up_np[45]],[up_np[59]],
    [up_np[60]],[up_np[61]],[up_np[62]],[up_np[63]],[up_np[64]],[up_np[55]],[up_np[54]],[up_np[53]],[up_np[52]],[up_np[51]],
    [up_np[46]],[up_np[47]],[up_np[48]],[up_np[49]],[up_np[50]],[up_np[100]],[up_np[99]],[up_np[93]],[up_np[98]],[up_np[97]],                                                                                                                                          
    [up_np[96]],[up_np[95]],[up_np[94]],[up_np[102]],[up_np[101]],[up_np[26]],[up_np[25]],[up_np[24]],[up_np[23]],[up_np[22]],
    [up_np[11]],[up_np[10]],[up_np[9]],[up_np[8]],[up_np[7]],[up_np[2]],[up_np[1]],[up_np[0]]])                                                                                                                                    
    up_np_reorder = np.concatenate((up_np_order1,up_np_order2),axis=None)
    
    down_np_order1=np.array([[down_np[6]],[down_np[5]],[down_np[4]],[down_np[17]],[down_np[16]],[down_np[15]],[down_np[14]],[down_np[13]],[down_np[31]],[down_np[30]],
    [down_np[29]],[down_np[28]],[down_np[27]],[down_np[103]],[down_np[104]],[down_np[34]],[down_np[35]],[down_np[36]],[down_np[37]],[down_np[38]],
    [down_np[39]],[down_np[33]],[down_np[32]],[down_np[92]],[down_np[91]],[down_np[90]],[down_np[89]],[down_np[88]],                                                                                                                                          
    [down_np[83]],[down_np[84]],[down_np[85]],[down_np[86]],[down_np[87]],[down_np[82]],[down_np[81]],[down_np[80]],[down_np[79]],[down_np[78]],
    [down_np[77]],[down_np[40]],[down_np[41]],[down_np[42]],[down_np[67]],[down_np[74]],[down_np[75]],[down_np[76]],[down_np[73]],[down_np[72]]])                                                                                                                                    
    
    down_np_order2=np.array([[down_np[68]],[down_np[69]],[down_np[58]],[down_np[57]],[down_np[56]],[down_np[65]],[down_np[43]],[down_np[44]],[down_np[45]],[down_np[59]],
    [down_np[60]],[down_np[61]],[down_np[62]],[down_np[63]],[down_np[64]],[down_np[55]],[down_np[54]],[down_np[53]],[down_np[52]],[down_np[51]],
    [down_np[46]],[down_np[47]],[down_np[48]],[down_np[49]],[down_np[50]],[down_np[100]],[down_np[99]],[down_np[93]],[down_np[98]],[down_np[97]],                                                                                                                                          
    [down_np[96]],[down_np[95]],[down_np[94]],[down_np[102]],[down_np[101]],[down_np[26]],[down_np[25]],[down_np[24]],[down_np[23]],[down_np[22]],
    [down_np[11]],[down_np[10]],[down_np[9]],[down_np[8]],[down_np[7]],[down_np[2]],[down_np[1]],[down_np[0]]])                                                                                                                                    
    down_np_reorder = np.concatenate((down_np_order1,down_np_order2),axis=None)
    
    
    # if anomaly_c_np is not None:
    #     anomaly_c_np_order1=np.array([[anomaly_c_np[6]],[anomaly_c_np[5]],[anomaly_c_np[4]],[anomaly_c_np[17]],[anomaly_c_np[16]],[anomaly_c_np[15]],[anomaly_c_np[14]],[anomaly_c_np[13]],[anomaly_c_np[31]],[anomaly_c_np[30]],
    #     [anomaly_c_np[29]],[anomaly_c_np[28]],[anomaly_c_np[27]],[anomaly_c_np[103]],[anomaly_c_np[104]],[anomaly_c_np[34]],[anomaly_c_np[35]],[anomaly_c_np[36]],[anomaly_c_np[37]],[anomaly_c_np[38]],
    #     [anomaly_c_np[39]],[anomaly_c_np[33]],[anomaly_c_np[32]],[anomaly_c_np[92]],[anomaly_c_np[91]],[anomaly_c_np[90]],[anomaly_c_np[89]],[anomaly_c_np[88]],                                                                                                                                          
    #     [anomaly_c_np[83]],[anomaly_c_np[84]],[anomaly_c_np[85]],[anomaly_c_np[86]],[anomaly_c_np[87]],[anomaly_c_np[82]],[anomaly_c_np[81]],[anomaly_c_np[80]],[anomaly_c_np[79]],[anomaly_c_np[78]],
    #     [anomaly_c_np[77]],[anomaly_c_np[40]],[anomaly_c_np[41]],[anomaly_c_np[42]],[anomaly_c_np[67]],[anomaly_c_np[74]],[anomaly_c_np[75]],[anomaly_c_np[76]],[anomaly_c_np[73]],[anomaly_c_np[72]]])                                                                                                                                    
        
    #     anomaly_c_np_order2=np.array([[anomaly_c_np[68]],[anomaly_c_np[69]],[anomaly_c_np[58]],[anomaly_c_np[57]],[anomaly_c_np[56]],[anomaly_c_np[65]],[anomaly_c_np[43]],[anomaly_c_np[44]],[anomaly_c_np[45]],[anomaly_c_np[59]],
    #     [anomaly_c_np[60]],[anomaly_c_np[61]],[anomaly_c_np[62]],[anomaly_c_np[63]],[anomaly_c_np[64]],[anomaly_c_np[55]],[anomaly_c_np[54]],[anomaly_c_np[53]],[anomaly_c_np[52]],[anomaly_c_np[51]],
    #     [anomaly_c_np[46]],[anomaly_c_np[47]],[anomaly_c_np[48]],[anomaly_c_np[49]],[anomaly_c_np[50]],[anomaly_c_np[100]],[anomaly_c_np[99]],[anomaly_c_np[93]],[anomaly_c_np[98]],[anomaly_c_np[97]],                                                                                                                                          
    #     [anomaly_c_np[96]],[anomaly_c_np[95]],[anomaly_c_np[94]],[anomaly_c_np[102]],[anomaly_c_np[101]],[anomaly_c_np[26]],[anomaly_c_np[25]],[anomaly_c_np[24]],[anomaly_c_np[23]],[anomaly_c_np[22]],
    #     [anomaly_c_np[11]],[anomaly_c_np[10]],[anomaly_c_np[9]],[anomaly_c_np[8]],[anomaly_c_np[7]],[anomaly_c_np[2]],[anomaly_c_np[1]],[anomaly_c_np[0]]])                                                                                                                                    
    #     anomaly_c_np_reorder = np.concatenate((anomaly_c_np_order1,anomaly_c_np_order2),axis=None)
    
    # labels=('Forehead', ' ', ' ', ' ', ' ', 'Eyebrow', ' ', ' ',
    #'External Canthus', ' ', 'Suborbital', ' ', 'Internal Canthus ', ' ',
    #'Eyelid ', ' ', ' ', 'Paranasal ', ' ', ' ', ' ', 'Nostril', ' ',
    #' ', ' ','Zygomatic', ' ', ' ', ' ', ' ', 'Cheek', ' ', ' ',
    #' ', ' ', 'Jawline', ' ', ' ', ' ', 'Labial commissure', 'Upper Lip',
    #' ', 'Lower Lip', ' ', ' ', ' ', ' ',' ', 'Mentalis', ' ',
    #' ', ' ', ' ', ' ','Lower lip', ' ', 'Upper lip', 'Labial commissure', ' ', ' ',
    # ' ', 'Jawline', ' ', ' ', ' ', ' ',
    # 'Cheek', ' ', ' ', ' ', ' ', 'Zygomatic', ' ', ' ', ' ',
    # 'Nostril', ' ', ' ', ' ', 'Paranasal', ' ', ' ', 'Eyelid', ' ',
    # 'Internal Canthus', ' ', 'Suborbital', ' ', 'External Canthus', ' ', ' ',
    # 'Eyebrow',' ', ' ',' ',' ')

    labels=('Front', ' ', ' ', ' ', ' ', 'Sourcils', ' ', ' ',
    'Canthus Externe', ' ', 'Suborbital', ' ', 'Canthus Interne ', ' ',
    'Paupières ', ' ', ' ', 'Paranasal ', ' ', ' ', ' ', 'Nez', ' ',
    ' ', ' ','Zygomatic', ' ', ' ', ' ', ' ', 'Joue', ' ', ' ',
    ' ', ' ', 'Machoire', ' ', ' ', ' ', 'Commissure Labiale', 'Lèvre supérieure',
    ' ', 'Lèvre inférieure', ' ', ' ', ' ', ' ',' ', 'Menton', ' ',
    ' ', ' ', ' ', ' ','Lèvre inférieure', ' ', 'Lèvre supérieure', 'Commissure Labiale', ' ', ' ',
      ' ', 'Machoire', ' ', ' ', ' ', ' ',
    'Joue', ' ', ' ', ' ', ' ', 'Zygomatic', ' ', ' ', ' ',
    'Nez', ' ', ' ', ' ', 'Paranasal', ' ', ' ', 'Paupières', ' ',
    'Canthus Interne', ' ', 'Suborbital', ' ', 'Canthus Externe', ' ', ' ',
    'Sourcils',' ', ' ',' ',' ')
        
    label_loc = np.linspace(start=0, stop=2*np.pi, num=len(center_np_reorder))
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(label_loc, center_np_reorder, color='red')
    ax.plot(label_loc, health_np_reorder, color='green')
    #if anomaly_c_np is not None:
        #ax.plot(label_loc, anomaly_c_np_reorder, color='red')
    #ax.set_ylim(0., 20.)
    #if anomaly_c_np is None:
    plt.fill_between(label_loc, up_np_reorder, down_np_reorder, color='green', alpha=0.15)
    plt.fill_between(label_loc, center_np_reorder, health_np_reorder, color='red', alpha=0.30)
    # else:
    #     plt.fill_between(label_loc, center_np_reorder, anomaly_c_np_reorder, color='red', alpha=0.15)
                
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=labels)
    plt.show()
    
    if save:
        fig.savefig(image_name)
        print(image_name + " saved")
    
    return

def faciograph_px(hr_np, px_np, save=True, image_name = "faciograph.svg"):#, anomaly_c_np = None):

    hr_np_order1=np.array([[hr_np[6]],[hr_np[5]],[hr_np[4]],[hr_np[17]],[hr_np[16]],[hr_np[15]],[hr_np[14]],[hr_np[13]],[hr_np[31]],[hr_np[30]],
    [hr_np[29]],[hr_np[28]],[hr_np[27]],[hr_np[103]],[hr_np[104]],[hr_np[34]],[hr_np[35]],[hr_np[36]],[hr_np[37]],[hr_np[38]],
    [hr_np[39]],[hr_np[33]],[hr_np[32]],[hr_np[92]],[hr_np[91]],[hr_np[90]],[hr_np[89]],[hr_np[88]],                                                                                                                                          
    [hr_np[83]],[hr_np[84]],[hr_np[85]],[hr_np[86]],[hr_np[87]],[hr_np[82]],[hr_np[81]],[hr_np[80]],[hr_np[79]],[hr_np[78]],
    [hr_np[77]],[hr_np[40]],[hr_np[41]],[hr_np[42]],[hr_np[67]],[hr_np[74]],[hr_np[75]],[hr_np[76]],[hr_np[73]],[hr_np[72]]])                                                                                                                                    
    
    hr_np_order2=np.array([[hr_np[68]],[hr_np[69]],[hr_np[58]],[hr_np[57]],[hr_np[56]],[hr_np[65]],[hr_np[43]],[hr_np[44]],[hr_np[45]],[hr_np[59]],
    [hr_np[60]],[hr_np[61]],[hr_np[62]],[hr_np[63]],[hr_np[64]],[hr_np[55]],[hr_np[54]],[hr_np[53]],[hr_np[52]],[hr_np[51]],
    [hr_np[46]],[hr_np[47]],[hr_np[48]],[hr_np[49]],[hr_np[50]],[hr_np[100]],[hr_np[99]],[hr_np[93]],[hr_np[98]],[hr_np[97]],                                                                                                                                          
    [hr_np[96]],[hr_np[95]],[hr_np[94]],[hr_np[102]],[hr_np[101]],[hr_np[26]],[hr_np[25]],[hr_np[24]],[hr_np[23]],[hr_np[22]],
    [hr_np[11]],[hr_np[10]],[hr_np[9]],[hr_np[8]],[hr_np[7]],[hr_np[2]],[hr_np[1]],[hr_np[0]]])                                                                                                                                    
    hr_np_reorder = np.concatenate((hr_np_order1,hr_np_order2),axis=None)
    
    px_np_order1=np.array([[px_np[6]],[px_np[5]],[px_np[4]],[px_np[17]],[px_np[16]],[px_np[15]],[px_np[14]],[px_np[13]],[px_np[31]],[px_np[30]],
    [px_np[29]],[px_np[28]],[px_np[27]],[px_np[103]],[px_np[104]],[px_np[34]],[px_np[35]],[px_np[36]],[px_np[37]],[px_np[38]],
    [px_np[39]],[px_np[33]],[px_np[32]],[px_np[92]],[px_np[91]],[px_np[90]],[px_np[89]],[px_np[88]],                                                                                                                                          
    [px_np[83]],[px_np[84]],[px_np[85]],[px_np[86]],[px_np[87]],[px_np[82]],[px_np[81]],[px_np[80]],[px_np[79]],[px_np[78]],
    [px_np[77]],[px_np[40]],[px_np[41]],[px_np[42]],[px_np[67]],[px_np[74]],[px_np[75]],[px_np[76]],[px_np[73]],[px_np[72]]])                                                                                                                                    
    
    px_np_order2=np.array([[px_np[68]],[px_np[69]],[px_np[58]],[px_np[57]],[px_np[56]],[px_np[65]],[px_np[43]],[px_np[44]],[px_np[45]],[px_np[59]],
    [px_np[60]],[px_np[61]],[px_np[62]],[px_np[63]],[px_np[64]],[px_np[55]],[px_np[54]],[px_np[53]],[px_np[52]],[px_np[51]],
    [px_np[46]],[px_np[47]],[px_np[48]],[px_np[49]],[px_np[50]],[px_np[100]],[px_np[99]],[px_np[93]],[px_np[98]],[px_np[97]],                                                                                                                                          
    [px_np[96]],[px_np[95]],[px_np[94]],[px_np[102]],[px_np[101]],[px_np[26]],[px_np[25]],[px_np[24]],[px_np[23]],[px_np[22]],
    [px_np[11]],[px_np[10]],[px_np[9]],[px_np[8]],[px_np[7]],[px_np[2]],[px_np[1]],[px_np[0]]])                                                                                                                                    
    px_np_reorder = np.concatenate((px_np_order1,px_np_order2),axis=None)

    #labels=('Forehead', ' ', ' ', ' ', ' ', 'Eyebrow', ' ', ' ',
    #'External Canthus', ' ', 'Suborbital', ' ', 'Internal Canthus ', ' ',
    #'Eyelid ', ' ', ' ', 'Paranasal ', ' ', ' ', ' ', 'Nostril', ' ',
    #' ', ' ','Zygomatic', ' ', ' ', ' ', ' ', 'Cheek', ' ', ' ',
    #' ', ' ', 'Jawline', ' ', ' ', ' ', 'Labial commissure', 'Upper Lip',
    #' ', 'Lower Lip', ' ', ' ', ' ', ' ',' ', 'Mentalis', ' ',
    #' ', ' ', ' ', ' ','Lower lip', ' ', 'Upper lip', 'Labial commissure', ' ', ' ',
    # ' ', 'Jawline', ' ', ' ', ' ', ' ',
    # 'Cheek', ' ', ' ', ' ', ' ', 'Zygomatic', ' ', ' ', ' ',
    # 'Nostril', ' ', ' ', ' ', 'Paranasal', ' ', ' ', 'Eyelid', ' ',
    # 'Internal Canthus', ' ', 'Suborbital', ' ', 'External Canthus', ' ', ' ',
    # 'Eyebrow',' ', ' ',' ',' ')

    labels=('Front', ' ', ' ', ' ', ' ', 'Sourcils', ' ', ' ',
    'Canthus Externe', ' ', 'Suborbital', ' ', 'Canthus Interne ', ' ',
    'Paupières ', ' ', ' ', 'Paranasal ', ' ', ' ', ' ', 'Nez', ' ',
    ' ', ' ','Zygomatic', ' ', ' ', ' ', ' ', 'Joue', ' ', ' ',
    ' ', ' ', 'Machoire', ' ', ' ', ' ', 'Commissure Labiale', 'Lèvre supérieure',
    ' ', 'Lèvre inférieure', ' ', ' ', ' ', ' ',' ', 'Menton', ' ',
    ' ', ' ', ' ', ' ','Lèvre inférieure', ' ', 'Lèvre supérieure', 'Commissure Labiale', ' ', ' ',
      ' ', 'Machoire', ' ', ' ', ' ', ' ',
    'Joue', ' ', ' ', ' ', ' ', 'Zygomatic', ' ', ' ', ' ',
    'Nez', ' ', ' ', ' ', 'Paranasal', ' ', ' ', 'Paupières', ' ',
    'Canthus Interne', ' ', 'Suborbital', ' ', 'Canthus Externe', ' ', ' ',
    'Sourcils',' ', ' ',' ',' ')
        
    label_loc = np.linspace(start=0, stop=2*np.pi, num=len(hr_np_reorder))
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(label_loc, hr_np_reorder,color='black')
    ax.plot(label_loc, px_np_reorder, color='red')
    plt.fill_between(label_loc, hr_np_reorder, px_np_reorder, color='red', alpha=0.15)


    lines, labels = plt.thetagrids(np.degrees(label_loc), fontsize=20,labels=labels)
    plt.show()
    
    # Remplacez la fin par:
    if save:
        plt.savefig(image_name, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()
    return