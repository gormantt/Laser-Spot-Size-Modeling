# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:20:02 2019

@author: tgorman
The goal of this script is to analyze data that is exported from Zemax.  It will have two capabilities.  One is to extract average intensities from the spatial distribution.
The second capability  is to assess the effect of water absorption on power densities for some of our projects.
v3 updates: v3 removes the "flooded or not flooded" questoin.  This is because Dietrich and I have updated
our optic studio interface to include water absorption.  To be clear this requires water to be modeled with the LSPT
custom material "H2O".
"""

import numpy as np
import scipy.optimize as sciOpt
import matplotlib.pyplot as plt
import os
import re
import tkinter as tk
from tkinter import filedialog


#plt.close('all')

root = tk.Tk()
root.withdraw()

ellipse_sample_min_abs=0.98 #GW/cm^2
ellipse_sample_max_abs=1.02 #GW/cm^2

def main():
    
    data = []
    #main_target()
    files = filedialog.askopenfilename(parent=root,title='Choose a file',multiple=True)
    files =root.tk.splitlist(files)
    
    #need to take from data frame to numpy array 1 at a time.
    for j in range(len(files)):
        file_name=os.path.split(files[j])[1]
        print(file_name)
        data, x_data, y_data, width, height = get_det_data_Nseq(files[j])
        #data_mean = np.mean(data[data>ellipse_sample_min_abs])
        pts_ellipse_x, pts_ellipse_y, val_x0, val_r1, val_y0, val_r2, val_theta = get_ellipse_data(data, width, height, ellipse_sample_min_abs, ellipse_sample_max_abs)
        data_mean = np.mean(data[(((x_data[None,:]-val_x0)*np.cos(val_theta)-(y_data[:,None]-val_y0)*np.sin(val_theta))/val_r1)**2
                                 +(((x_data[None,:]-val_x0)*np.sin(val_theta)+(y_data[:,None]-val_y0)*np.cos(val_theta))/val_r2)**2<=1])
        #x_model, y_model, r_model, theta_model=elliptic_fourier_fitting(pts_ellipse_x, pts_ellipse_y)
        
        min_rad= min(val_r1,val_r2)
        max_rad= max(val_r1,val_r2)
        
        
        dataT = np.linspace(0, 2*np.pi, num=100)
        min_rad_x=[val_x0,val_x0+min_rad*np.cos(val_theta)]
        min_rad_y=[val_y0,val_y0+min_rad*np.sin(val_theta)]
        max_rad_x=[val_x0,val_x0+max_rad*np.cos(val_theta+np.pi/2)]
        max_rad_y=[val_y0,val_y0+max_rad*np.sin(val_theta+np.pi/2)]
        
        #Begin Plotting
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        if not np.isnan(val_x0) and not (min(val_r1, val_r2)*2 > min(width, height) and max(val_r1, val_r2)*2 > max(width, height)):
            fit_ellipseX, fit_ellipseY = plotEllipse(dataT, val_x0, val_r1, val_y0, val_r2, val_theta)
            ax.plot(fit_ellipseX, fit_ellipseY, "--b",label = "Min. Intensity Ellipse")
        #print(val_r1)
        #print(val_theta)
        
        file_name=os.path.split(files[j])[1]
        c=ax.pcolorfast(x_data,y_data,data,cmap='magma')
        #ax.scatter(pts_ellipse_x, pts_ellipse_y,s=3)
        ax.plot(min_rad_x,min_rad_y, 'r', label="Min. Radius = "+str(round(min_rad,3))+"mm")
        ax.plot(max_rad_x,max_rad_y, 'g',label="Max. Radius = "+str(round(max_rad,3))+"mm")
        #ax.scatter(x_model,y_model)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(file_name+"\nMean Power Density="+str(round(data_mean,2))+'GW/c$\mathregular{m^2}$')#+"Min. Radius = "+str(round(min_rad,3))+"mm"+
                     #"\nMax Radius = "+str(round(max_rad,3))+"mm")
        cbar=fig.colorbar(c)
        cbar.ax.get_yaxis().labelpad=15
        cbar.ax.set_ylabel('GW/c$\mathregular{m^2}$',rotation=0,y=1.05, labelpad=-20)
        ax.legend(prop={"size":14})
        ax.set_aspect(1)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(labelsize= 14)
        ax.title.set_fontsize(15)
        plt.tight_layout()
        plt.show()

def elliptic_fourier_fitting(pts_ellipse_x, pts_ellipse_y):
    #intialize x, y, and radius data arrays
    xa=pts_ellipse_x # x data in millimeters
    ya=pts_ellipse_y # y data in millimeters
    ra=np.sqrt(np.square(xa)+np.square(ya)) # r data
    
    # calculating theta angles from (x,y) data
    thetas=np.arctan2(ya,xa)
    thetas[thetas<0]+=2*np.pi
    
    arg_sort=np.argsort(thetas)
    xa=xa[arg_sort]
    ya=ya[arg_sort]
    thetas=thetas[arg_sort]
    
    xa_subset= xa
    ya_subset= ya
    theta_subset = thetas
    #print(thetas)
    
    num_points = len(xa_subset)
    
    n_max = 0 # initializing the max number of terms in the Fourier Series
    
    
    #defining the max number of  terms in the Fourier Series, i.e. the max frequency
    biggest_division= np.linalg.norm((int)(2*np.pi/np.amax(np.diff(theta_subset[:]))))
    if biggest_division > np.linalg.norm((int)(2*np.pi/((theta_subset[-1]-theta_subset[1]+np.pi)%(2*np.pi)-np.pi))):
        biggest_division = np.linalg.norm((int)(2*np.pi/((theta_subset[-1]-theta_subset[1]+np.pi)%(2*np.pi)-np.pi)))
    if np.mod(biggest_division,2)==0: #num_points is even
        n_max=(biggest_division)/2-1
    elif np.mod(biggest_division,2)==1: #num_points is odd
        n_max=(biggest_division-1)/2
    
    
    n_max=(int)(n_max) # converting to integer
    
    #edge cases of algorithm.  Making sure that the answers don't blow up
    if biggest_division == 2:
        n_max=1
    elif biggest_division ==1:
        n_max = 0

    if n_max > 300:
        n_max=300
    #intializing the matrix that contains all of the cos(theta) and sin(theta) terms
    theta_matrix=np.zeros(shape=(num_points,n_max*2+1))
    # n=0 terms
    theta_matrix[:,0]=1
    
  
    #filling out theta matrix
    for pp in range(num_points):
        for nn in range(1,n_max*2+1):
            if np.mod(nn,2)==1: #cosine terms
                theta_matrix[pp,nn]=np.cos((nn+1)/2*theta_subset[pp])#np.cos((nn+1)/2*thetas[pp])
            if np.mod(nn,2)==0: #sine terms
                theta_matrix[pp,nn]=np.sin(nn/2*theta_subset[pp])#np.cos((nn)/2*thetas[pp])
    # code for finding x and y coeffs
    trans_theta_matrix=np.transpose(theta_matrix[:,:])
    inv_theta_matrix=np.linalg.inv(np.matmul(trans_theta_matrix,theta_matrix[:,:]));
    regression_mat=np.matmul(inv_theta_matrix,trans_theta_matrix);
    x_coeffs=np.matmul(regression_mat,xa_subset[:])
    y_coeffs=np.matmul(regression_mat,ya_subset[:])

# Calculation Model points
    theta_step=360#len(thetas)#100
    theta_model=np.arange(0,2*np.pi,2*np.pi/theta_step)
    x_model=np.zeros(len(theta_model))
    y_model=np.zeros(len(theta_model))
    theta_model_matrix=np.zeros(shape=(theta_step,n_max*2+1))
    theta_model_matrix[:,0]=1;
    for pp in range(theta_step):
        for nn in range(1,n_max*2+1):
            if np.mod(nn,2)==1: #cosine terms
                theta_model_matrix[pp,nn]=np.cos((nn+1)/2*theta_model[pp])#np.cos((nn+1)/2*thetas[pp])
            if np.mod(nn,2)==0: #sine terms
                theta_model_matrix[pp,nn]=np.sin(nn/2*theta_model[pp])#np.cos((nn)/2*thetas[pp])
            
    x_model=np.matmul(theta_model_matrix,x_coeffs)
    y_model=np.matmul(theta_model_matrix,y_coeffs)
    r_model=np.sqrt(x_model**2+y_model**2)
    #ra[:,qq] = np.sqrt(xa[:,qq]**2+ya[:,qq]**2)

    return x_model, y_model, r_model, theta_model

def get_det_data_Nseq(fname):
    fh=open(fname,"r",encoding='utf-16').read()
        
    match = re.findall("Size.*W", fh)
        
    for line in match:
        width=float(re.findall("Size(.*?)W", line)[0])
        height=float(re.findall("X(.*?)H", line)[0])
    #temp_data=pd.read_csv(fname, encoding='utf-16', header=18, sep="\t")
    temp_data = np.loadtxt(fname, encoding = 'utf-16',skiprows = 24)
    #temp_data=temp_data.iloc[:,1:]
    temp_data = temp_data[:,1:]
    
    data=temp_data#.values.tolist()
    data=np.array(data)
    data=np.flipud(data) 
    x_data=np.arange(-width/2,width/2+width/np.size(data,1),width/(np.size(data,1)-1))
    y_data=np.arange(-height/2,height/2+height/np.size(data,0),height/(np.size(data,0)-1))
    
    return data, x_data, y_data, width, height

def get_ellipse_data(data, width, height, min_lvl, max_lvl):
    #data_ellipse = np.zeros(data.shape)
    #data_ellipse[np.logical_and(minLvl < data, data < maxLvl)] = 1
    pts_ellipse = np.where(np.logical_and(min_lvl < data, data < max_lvl))
    pts_ellipse_y, pts_ellipse_x = pts_ellipse
    
    pixX = width/data.shape[1]
    pixY = height/data.shape[0]
    
    pts_ellipse_x = pixX*pts_ellipse_x+pixX/2-width/2
    pts_ellipse_y = pixY*pts_ellipse_y+pixY/2-height/2
    
    def fitEllipse(params):
        #    return (rx*np.cos(t)+x0, ry*np.sin(t+dp)+y0)
        x0, r1, y0, r2, theta = params
        return (  (((pts_ellipse_x-x0)*np.cos(theta)-(pts_ellipse_y-y0)*np.sin(theta))/r1)**2
                + (((pts_ellipse_x-x0)*np.sin(theta)+(pts_ellipse_y-y0)*np.cos(theta))/r2)**2 - 1)
    
    mean_x = np.mean(pts_ellipse_x)
    mean_y = np.mean(pts_ellipse_y)
    guess = (mean_x, np.max(np.abs(pts_ellipse_x))-mean_x, mean_y, np.max(np.abs(pts_ellipse_y))-mean_y, 0)
    solve = sciOpt.least_squares(fitEllipse, guess, xtol=1e-13, ftol=1e-13, gtol=1e-13,
                                 bounds=((-width/2, 0, -height/2, 0, -np.pi/2),
                                         (width/2,width/2,
                                          height/2,height/2, np.pi/2)))
    val_x0, val_r1, val_y0, val_r2, val_theta = solve.x
    if val_r2 < val_r1:
        val_theta = np.pi-val_theta
    return (pts_ellipse_x, pts_ellipse_y, val_x0, val_r1, val_y0, val_r2, val_theta)

# rotated parametric equation
def plotEllipse(t, val_x0, val_r1, val_y0, val_r2, val_theta):
    return((val_r1*np.cos(t) + val_r2*np.tan(val_theta)*np.sin(t))/(np.cos(val_theta)+np.sin(val_theta)**2/np.cos(val_theta))+val_x0,
           (val_r2*np.sin(t) - val_r1*np.tan(val_theta)*np.cos(t))/(np.cos(val_theta)+np.sin(val_theta)**2/np.cos(val_theta))+val_y0)

if __name__ == '__main__':
    main()