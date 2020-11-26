import numpy as np
import pandas as pd
from pylab import *
import matplotlib
import plplot
from scipy import stats
#from io import StringIO
print(pd.__version__)
#AttributeError: 'Series' object has no attribute 'to_numpy'
import re

import math
import astropy as astro
import scipy.ndimage as spimage
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
import matplotlib as mpl
from numpy import std as biweight_midvariance
import matplotlib.cm as cm

from scipy import optimize
from scipy.stats import gaussian_kde
from matplotlib import colors

from collections import Counter


def make_colourmap(ind, red, green, blue, name):
    newInd = range(0, 256)
    r = np.interp(newInd, ind, red, left=None, right=None)
    g = np.interp(newInd, ind, green, left=None, right=None)
    b = np.interp(newInd, ind, blue, left=None, right=None)
    colours = np.transpose(np.asarray((r, g, b)))
    fctab= colours/255.0
    cmap = colors.ListedColormap(fctab, name=name,N=None)
    return cmap

def get_califa_velocity_cmap():
    ind = [1., 35., 90.,125.,160.,220.,255.]
    red = [148., 0., 0., 55.,221.,255.,255.]
    green = [ 0., 0.,191., 55.,160., 0.,165.]
    blue = [211.,128.,255., 55.,221., 0., 0.]
    return make_colourmap(ind, red, green, blue, 'califa_vel')


def get_califa_velocity_cmap_2():
    ind = [0., 1., 35., 90.,125.,160.,220.,255.]
    red = [ 0.,148., 0., 0., 55.,221.,255.,255.]
    green = [ 0., 0., 0.,191., 55.,160., 0.,165.]
    blue = [ 0.,211.,128.,255., 55.,221., 0., 0.]
    return make_colourmap(ind, red, green, blue, 'califa_vel')

def get_califa_intensity_cmap_2():
    ind = [ 0., 1., 50.,100.,150.,200.,255.]
    red = [ 0., 0., 0.,255.,255., 55.,221.]
    green =	[ 0., 0.,191., 0.,165., 55.,160.]
    blue = [ 0.,128.,255., 0., 0., 55.,221.]
    return make_colourmap(ind, red, green, blue, 'califa_int')

def get_califa_intensity_cmap():
    ind = [ 1., 50.,100.,150.,200.,255.]
    red = [ 0., 0.,255.,255., 55.,221.]
    green =	[ 0.,191., 0.,165., 55.,160.]
    blue = [ 128.,255., 0., 0., 55.,221.]
    return make_colourmap(ind, red, green, blue, 'califa_int')

def get_califa_velocity_cmap_r():
    ind = [0., 1., 35., 90.,125.,160.,220.,255.]
    red = [ 0.,148., 0., 0., 55.,221.,255.,255.]
    green = [ 0., 0., 0.,191., 55.,160., 0.,165.]
    blue = [ 0.,211.,128.,255., 55.,221., 0., 0.]
    return make_colourmap(ind, red[::-1], green[::-1], blue[::-1], 'califa_vel_r')


califa_vel = get_califa_velocity_cmap()
califa_vel_r = get_califa_velocity_cmap_r()
califa_int = get_califa_intensity_cmap()

#califa_int=cm.Spectral
#califa_vel=cm.Spectral

def fit_leastsq_pure(p0, datax, datay, function):

    errfunc = lambda p, x, y: function(x,p) - y

    pfit, pcov, infodict, errmsg, success = \
        optimize.leastsq(errfunc, p0, args=(datax, datay), \
                          full_output=1, epsfcn=0.0001)

    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    return pfit_leastsq, perr_leastsq 

def fit_leastsq(p0, datax, datay, function):

    errfunc = lambda p, x, y: function(x,p) - y

    pfit, pcov, infodict, errmsg, success = \
        optimize.leastsq(errfunc, p0, args=(datax, datay), \
                          full_output=1, epsfcn=0.01)
# epsfcn=0.0001)


    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    return pfit_leastsq, pcov 

#
# Binning!
#

def binning_OH(M_OK, OH_Ref_OK, bin1 , min1 , max1 ):
    
    M_bin=[]
    OH_bin=[]
    D_OH_bin=[]
    
    OH_binM    = np.arange(min1,max1,bin1) 
    OH_binM = OH_binM-bin1*0.5
    m_range    = np.zeros(OH_binM.size)
    OH_binD    = np.zeros(OH_binM.size)

    for i, val  in enumerate(OH_binM):
        tmp = (OH_Ref_OK >= val) & (OH_Ref_OK <= val+bin1)
        m_sub=M_OK[tmp]
        n_vals=m_sub.size
        m_range[i]   = np.median(M_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])
        if (n_vals > 10):
            M_bin.append(m_range[i])
            OH_bin.append(OH_binM[i])
            D_OH_bin.append(OH_binD[i])
    m_range=np.array(M_bin)
    OH_binM=np.array(OH_bin)
    OH_binD=np.array(D_OH_bin)
        
    return(m_range, OH_binM, OH_binD)

def binning(M_OK, OH_Ref_OK, bin1 , min1 , max1 ):
    
    
    m_range = np.arange(min1,max1,bin1)
    M_binM    = np.zeros(m_range.size)
    M_binV    = np.zeros(m_range.size)
    OH_binM    = np.zeros(m_range.size)
    OH_binD    = np.zeros(m_range.size)
    n_vals    = np.zeros(m_range.size)

    for i, val  in enumerate(m_range):
        tmp = (M_OK >= val) & (M_OK <= val+bin1)
        OH_binM[i]   = np.median(OH_Ref_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])+0.02
#        tmp = (OH_Ref_OK >= OH_binM[i]-0.125*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+0.125*OH_binD[i]) & (M_OK >= val-4*bin1) & (M_OK <= val+5*bin1)       
        tmp = (OH_Ref_OK >= OH_binM[i]-0.1*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+0.1*OH_binD[i]) & (M_OK >= val-3*bin1) & (M_OK <= val+3*bin1)       
        m_sub=M_OK[tmp]
        n_vals[i]=m_sub.size
#        print('n_val',n_vals,', vals = ',M_OK[tmp])
#        if (n_vals > 2):
        M_binM[i]   = np.median(M_OK[tmp])
        M_binV[i] = val+0.5*bin1        
        if ((np.isnan(M_binM[i])) or (np.isinf(M_binM[i]))):
            M_binM[i]=M_binV[i]
    M_bin_out=0.5*(M_binM+M_binV)
    #print '',M_binM,M_binV,M_bin_out
    mask_val= n_vals>5
    
    return(M_bin_out[mask_val], OH_binM[mask_val], OH_binD[mask_val])

def binning_M(M_OK, OH_Ref_OK, bin1 , min1 , max1 , Nmax, delta_y=0.1, delta_x=3.0):
    
    
    m_range = np.arange(min1,max1,bin1)
    M_binM    = np.zeros(m_range.size)
    M_binV    = np.zeros(m_range.size)
    OH_binM    = np.zeros(m_range.size)
    OH_binD    = np.zeros(m_range.size)
    n_vals    = np.zeros(m_range.size)

    for i, val  in enumerate(m_range):
        tmp = (M_OK >= val) & (M_OK <= val+bin1)
        OH_binM[i]   = np.median(OH_Ref_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])+0.02
        #print('Y_vals =',OH_binM[i],OH_binD[i],val,val+bin1)
        #print('vector =',OH_Ref_OK[tmp])
        tmp = (OH_Ref_OK >= OH_binM[i]-delta_y*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+delta_y*OH_binD[i]) & (M_OK >= val-delta_x*bin1) & (M_OK <= val+delta_x*bin1)       
        m_sub=M_OK[tmp]
        n_vals[i]=m_sub.size
#        print('n_val',n_vals,', vals = ',M_OK[tmp])
#        if (n_vals > 2):
        M_binM[i]   = np.median(M_OK[tmp])
        M_binV[i] = val+0.5*bin1        
        if ((np.isnan(M_binM[i])) or (np.isinf(M_binM[i]))):
            M_binM[i]=M_binV[i]
    M_bin_out=0.5*(M_binM+M_binV)
    #print '',M_binM,M_binV,M_bin_out
    #print('# ',n_vals,Nmax)
    mask_val= n_vals>Nmax
    
    return(M_bin_out[mask_val], OH_binM[mask_val], OH_binD[mask_val])


def binning_M2(M_OK, OH_Ref_OK, bin1 , min1 , max1 , Nmax, delta_y=0.1, delta_x=3.0):
    
    
    m_range = np.arange(min1,max1,bin1)
    M_binM    = np.zeros(m_range.size)
    M_binV    = np.zeros(m_range.size)
    OH_binM    = np.zeros(m_range.size)
    OH_binD    = np.zeros(m_range.size)
    n_vals    = np.zeros(m_range.size)

    for i, val  in enumerate(m_range):
        tmp = (M_OK >= val) & (M_OK <= val+bin1)
        OH_binM[i]   = np.median(OH_Ref_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])+0.02
        #print('Y_vals =',OH_binM[i],OH_binD[i],val,val+bin1)
        #print('vector =',OH_Ref_OK[tmp])
        tmp = (OH_Ref_OK >= OH_binM[i]-delta_y*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+delta_y*OH_binD[i]) & (M_OK >= val-delta_x*bin1) & (M_OK <= val+delta_x*bin1)       
        m_sub=M_OK[tmp]
        n_vals[i]=m_sub.size
        
        print('n_val',i,', vals = ',M_OK[tmp], OH_Ref_OK[tmp])
#        if (n_vals > 2):
        M_binM[i]   = np.median(M_OK[tmp])
        M_binV[i] = val+0.5*bin1        
        if ((np.isnan(M_binM[i])) or (np.isinf(M_binM[i]))):
            M_binM[i]=M_binV[i]
    M_bin_out=0.5*(M_binM+M_binV)
    #print '',M_binM,M_binV,M_bin_out
    #print('# ',n_vals,Nmax)
    mask_val= n_vals>Nmax
    
    return(M_bin_out[mask_val], OH_binM[mask_val], OH_binD[mask_val])



def binning2(M_OK, OH_Ref_OK, bin1 , min1 , max1 ):
    
    
    m_range = np.arange(min1,max1,bin1)
    M_binM    = np.zeros(m_range.size)
    M_binV    = np.zeros(m_range.size)
    OH_binM    = np.zeros(m_range.size)
    OH_binD    = np.zeros(m_range.size)

    for i, val  in enumerate(m_range):
        tmp = (M_OK >= val) & (M_OK <= val+bin1)
        OH_binM[i]   = np.median(OH_Ref_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])+0.02
#        tmp = (OH_Ref_OK >= OH_binM[i]-0.125*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+0.125*OH_binD[i]) & (M_OK >= val-4*bin1) & (M_OK <= val+5*bin1)       
        tmp = (OH_Ref_OK >= OH_binM[i]-0.1*OH_binD[i]) & (OH_Ref_OK <= OH_binM[i]+0.1*OH_binD[i]) & (M_OK >= val-3*bin1) & (M_OK <= val+3*bin1)       

        M_binM[i]   = np.median(M_OK[tmp])
        M_binV[i] = val+0.5*bin1        
        if ((np.isnan(M_binM[i])) or (np.isinf(M_binM[i]))):
            M_binM[i]=M_binV[i]
    M_bin_out=0.5*(M_binM+M_binV)
    #print '',M_binM,M_binV,M_bin_out
    mask = np.logical_not(np.isnan(M_bin_out)) & np.logical_not(np.isnan(OH_binM)) & np.logical_not(np.isnan(OH_binD))
    M_bin_out = M_bin_out[mask]
    OH_binM = OH_binM[mask]
    OH_binD = OH_binD[mask]
    return(M_bin_out, OH_binM, OH_binD)

def binning_old(M_OK, OH_Ref_OK, bin1 , min1 , max1 ):
    
    
    m_range = np.arange(min1,max1,bin1)
    OH_binM    = np.zeros(m_range.size)
    OH_binD    = np.zeros(m_range.size)

    for i, val  in enumerate(m_range):
        tmp = (M_OK >= val) & (M_OK <= val+bin1)
        OH_binM[i]   = np.median(OH_Ref_OK[tmp])
        OH_binD[i]   = np.std(OH_Ref_OK[tmp])
            
    return(m_range, OH_binM, OH_binD)

def make_cont(x ,y, min2s,max2s,min1s,max1s,bin1s,bin2s, frac):

    m1s      = math.floor((max1s-min1s)/bin1s) + 1
    m2s      = math.floor((max2s-min2s)/bin2s) + 1
    
    vals, xedges, yedges = np.histogram2d(x, y, bins=[m1s,m2s])
    
    xbins = 0.5 * (xedges[:-1] + xedges[1:])
    ybins = 0.5 * (yedges[:-1] + yedges[1:])
    
    L = (1-frac)*(np.max(vals) - np.min(vals))+ np.min(vals)
    return(xbins, ybins, vals.T, L)





#
# Pandas reading columns
#
def header_columns_old_pd(filename,column):
    COMMENT_CHAR = '#'
    col_NAME = []
    with open(filename, 'r') as td:
        for line in td:
            if line[0] == COMMENT_CHAR:
                info = re.split(' +', line.rstrip('\n'))
                col_NAME.append(info[column])
    return col_NAME

def header_columns_formatted(filename,column):
    COMMENT_CHAR = '#'
    col_NAME = []
    with open(filename, 'r') as td:
        for line in td:
            if (line[0] == COMMENT_CHAR) and (line.find("COLUMN")>-1):
                start_info = re.split(',+', line.rstrip('\n'))
                info = re.split(' +', start_info[0])
                col_NAME.append(info[column])
    counts = {k:v for k,v in Counter(col_NAME).items() if v > 1}
    col_NAME_NEW = col_NAME[:]
    for i in reversed(range(len(col_NAME))):
        item = col_NAME[i]
        if item in counts and counts[item]:
            if (counts[item]>1):
                col_NAME_NEW[i] += str(counts[item]-1)
            counts[item]-=1                
    return col_NAME_NEW

def header_columns(filename,column):
    COMMENT_CHAR = '#'
    col_NAME = []
    with open(filename, 'r') as td:
        for line in td:
            if line[0] == COMMENT_CHAR:
                info = re.split(' +', line.rstrip('\n'))
                col_NAME.append(info[column])
    counts = {k:v for k,v in Counter(col_NAME).items() if v > 1}
    col_NAME_NEW = col_NAME[:]
    for i in reversed(range(len(col_NAME))):
        item = col_NAME[i]
        if item in counts and counts[item]:
            if (counts[item]>1):
                col_NAME_NEW[i] += str(counts[item]-1)
            counts[item]-=1                
    return col_NAME_NEW


