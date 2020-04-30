from matplotlib import pyplot as plt
from astropy.io import fits
import scipy.stats as st
import math
from scipy import signal
from astropy import wcs
import sys
from scipy import interpolate
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import scipy.stats as st
from scipy import signal
from astropy import wcs
import sys
from scipy import interpolate
from numpy import inf
from astropy.io.fits import getheader
from astropy.utils.data import get_pkg_data_filename
from reproject import reproject_interp
from reproject import reproject_exact
import time
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import scipy.interpolate as spi
import pandas as pd


mf = pd.read_csv('/Users/nikhil/code/Newtext/Matchtxt/spec.txt', comment='#', header=None, delim_whitespace=True)
Indices = np.array(mf[2])
H_Blue = np.array(mf[9])
L_Blue = np.array(mf[8])
H_Red = np.array(mf[13])
L_Red = np.array(mf[12])
H_Index = np.array(mf[5])
L_Index = np.array(mf[4])
Unit_indx = np.array(mf[16])
KJ = np.where(Unit_indx == 'mag')
KL = Unit_indx*0
KL[KJ] = 1
Rmid = (H_Red + L_Red)/2
Bmid = (H_Blue + L_Blue)/2
Imid = (H_Index + L_Index)/2
Delt_In = H_Index - L_Index
Delt_R = H_Red - L_Red
Delt_B = H_Blue - L_Blue

start_time = time.time()
with open('/Users/Nikhil/code/Newtext/Matchtxt/W2W1M2TOT.txt') as f:
   Line = [line.rstrip('\n') for line in open('/Users/Nikhil/code/Newtext/Matchtxt/W2W1M2TOT.txt')]


q=0
KT = 1
for q in range (0,np.shape(Line)[0]):
    one = 1
#path to drpall
    drpall = fits.open('/Users/Nikhil/Data/MaNGAPipe3D/Newmanga/drpall-v2_3_1.fits')
    tbdata = drpall[1].data
    ind = np.where(tbdata['mangaid'] == Line[q])
    objectra = tbdata['objra'][ind][0]
    objectdec = tbdata['objdec'][ind][0]
    redshift = tbdata['nsa_z'][ind][0]
    plate = tbdata['plate'][ind][0]
    ifu = tbdata['ifudsgn'][ind][0]
    sloan = tbdata['nsa_iauname'][ind][0]
    axs = tbdata['nsa_elpetro_ba'][ind][0]
    pa = tbdata['nsa_elpetro_phi'][ind][0]
    Ref = tbdata['NSA_ELPETRO_TH50_R'][ind][0]
    
#Define psfs for Swift filter convolution
    FWHMar = np.array([2.92/2.355,2.45/2.355,2.37/2.355])
    SwftSigmaw2 = FWHMar[0]
    SwftSigmam2 = FWHMar[1]
    SwftSigmaw1 = FWHMar[2]
    
#Read spectra from DRP Logcube
    hdu = fits.open("/Volumes/Nikhil/Data/LOGCUBE/manga"+"-" + str(plate) + "-" + str(ifu)+"-"+"LOGCUBE.fits")
    Good_bit = 2**10
    Flux_spectra = (hdu['FLUX'].data)/(0.25) #Read spectra and convert it from /spaxel^2 to /arsec^2
    Mask_spectra = hdu['MASK'].data #Read Mask
    Index_S = np.where(Mask_spectra>Good_bit)
    Index_S_N = np.where(Mask_spectra<=Good_bit)
    Mask_spectra[Index_S] = 1
    Mask_spectra[Index_S_N] = 0
    Variance_spectra = 1/hdu['IVAR'].data #Read variance
    Variance_spectra[np.isnan(Variance_spectra)]=0
    Variance_spectra[Variance_spectra == -inf] = 0
    Variance_spectra[Variance_spectra == inf] = 0
    RSeeing = hdu[0].header['RFWHM'] #Read seeing FWHM
    SigmaM = RSeeing/2.355 #Convert to sigma
    Predisp = hdu['PREDISP'].data #Read instrumental resolution of MaNGA
    Waveh = hdu['WAVE'].data
#Read velocities, dispersions, spec index masks and emission line EWs and fluxes and its vars and masks from DAP Maps.
    hdu = fits.open("/Volumes/Nikhil/HYB10-GAU-MILESHC/"+str(plate)+"/"+str(ifu)+"/"+"manga"+"-"+ str(plate) + "-" + str(ifu)+"-MAPS-HYB10-GAU-MILESHC.fits.gz")
    SP_MASK = hdu['SPECINDEX_MASK'].data
    SP_D = hdu['SPECINDEX'].data
    SP_S = np.sqrt(1/hdu['SPECINDEX_IVAR'].data)
    I_SP = np.where(SP_MASK>Good_bit)
    I_SP_N = np.where(SP_MASK<=Good_bit)
    SP_MASK[I_SP]=1
    SP_MASK[I_SP_N]=0
    H_manga = hdu['STELLAR_VEL'].header
    Vel = hdu['STELLAR_VEL'].data #Stellar velocity
    Shift = 1 + redshift + Vel/299792 #Define shift
    Vel_sigma_corr = hdu['STELLAR_SIGMACORR'].data
    Vel_sigma = hdu['STELLAR_SIGMA'].data
    Vel_sigma_err = 1/hdu['STELLAR_SIGMA_IVAR'].data
    Vel_sigma_mask = hdu['STELLAR_SIGMA_MASK'].data
    Vel_index = np.where(Vel_sigma_mask>Good_bit)
    Vel_index_N = np.where(Vel_sigma_mask<=Good_bit)
    Vel_sigma_mask[Vel_index] = 1
    Vel_sigma_mask[Vel_index_N] = 0
    Astro_sigma_mnga = np.ma.array((Vel_sigma**2 - Vel_sigma_corr**2),mask=Vel_sigma_mask)
#Read Emission Line fluxes
    GFLUX = hdu['EMLINE_GFLUX'].data
    GEW = hdu['EMLINE_GEW'].data
    GFLUX_VAR = 1/hdu['EMLINE_GFLUX_IVAR'].data
    GFLUX_VAR[np.isnan(GFLUX_VAR)] = 0
    GFLUX_VAR[GFLUX_VAR == inf] = 0
    GFLUX_VAR[GFLUX_VAR == -inf] = 0
    GEW_VAR = 1/hdu['EMLINE_GEW_IVAR'].data
    GEW_VAR[np.isnan(GEW_VAR)] = 0
    GEW_VAR[GEW_VAR == inf] = 0
    GEW_VAR[GEW_VAR == -inf] = 0
    GFMASK = hdu['EMLINE_GFLUX_MASK'].data
    GEWMASK = hdu['EMLINE_GEW_MASK'].data
    Index_F = np.where(GFMASK>Good_bit)
    Index_E = np.where(GFMASK>Good_bit)
    Index_F_N = np.where(GFMASK<=Good_bit)
    Index_E_N = np.where(GFMASK<=Good_bit)
    GFMASK[Index_F] = 1
    GEWMASK[Index_E] = 1
    GFMASK[Index_F_N] = 0
    GEWMASK[Index_E_N] = 0
    GFLUX_Clip = np.zeros(hdu['EMLINE_GFLUX'].data.shape)
    GEW_Clip = np.zeros(hdu['EMLINE_GEW'].data.shape)
    p=0
    for p in range (0,np.shape(GFLUX)[0]):
           GFLUX_Clip[p,:,:] = np.clip(GFLUX[p,:,:],0,np.max(GFLUX[p,:,:]))
           GEW_Clip[p,:,:] = np.clip(GEW[p,:,:],0,np.max(GEW[p,:,:]))

#Derive emission continuum from EW and fluxes
    GCONT_Clip = GFLUX_Clip/GEW_Clip
    GCONT_Clip[np.isnan(GCONT_Clip)] = 0
    GCONT_Clip[GCONT_Clip == inf] = 0
    GCONT_Clip[GCONT_Clip == -inf] = 0
    
#Read best fit emission spectra from DAP Logcube
    hdu = fits.open("/Volumes/Nikhil/Data/LOGCUBE/manga"+"-" + str(plate) + "-" + str(ifu)+"-LOGCUBE-HYB10-GAU-MILESHC.fits")
    Emission = hdu['EMLINE'].data
    Emission_base = hdu['EMLINE_BASE'].data
    Continuum_Flux = Flux_spectra - Emission - Emission_base
    Continuum_Flux_Rest = np.zeros(Continuum_Flux.shape)
    Predisp_Rest = np.zeros(Continuum_Flux.shape)

#Get spectra and pre disp to rest frame
    i=0
    for i in range (0,np.shape(Continuum_Flux)[1]):
           j = 0
           for j in range (0,np.shape(Continuum_Flux)[1]):
              Waven = Waveh/Shift[i,j]
              Y_naught = Continuum_Flux[:,i,j]*Shift[i,j]
              Y_PD = (Predisp[:,i,j]/Waveh)*(299792)
              X_naught = Waven
              Z = spi.interp1d(X_naught,Y_naught,fill_value="extrapolate")
              K_predisp = spi.interp1d(X_naught,Y_PD,fill_value="extrapolate")
              Continuum_Flux_Rest[:,i,j] = Z(Waveh)
              Predisp_Rest[:,i,j] = K_predisp(Waveh)
    
    Rest_Flux = np.ma.array(Continuum_Flux_Rest,mask=Mask_spectra)
    Rest_Predisp = np.ma.array(Predisp_Rest,mask=Mask_spectra)
    Spec_variance = np.ma.array(Variance_spectra,mask=Mask_spectra)


#Calculate Dn4000
    cvel = 2.997*math.pow(10,10)
    #Red window
    H1_R = np.where(Waveh < 4000)
    L1_R = np.where(Waveh > 4100)
    H10_R = np.max(H1_R)
    L10_R = np.min(L1_R)
    ar1_R = Waveh[H10_R:L10_R]
    Numerator = Rest_Flux[H10_R:L10_R,:,:]
    NUM = Numerator*(0)
    i=0
    for i in range (0,len(Numerator)):
        NUM[i,:,:] = ((ar1_R[i])**2)*Numerator[i,:,:]/cvel
    #Blue window
    H2_B = np.where(Waveh < 3850)
    L2_B = np.where(Waveh > 3950)
    H20_B = np.max(H2_B)
    L20_B = np.min(L2_B)
    ar2_B = Waveh[H20_B:L20_B]
    Denominator = Rest_Flux[H20_B:L20_B,:,:]
    DENO = Denominator*(0)
    i=0
    for i in range (0,len(Denominator)):
        DENO[i,:,:] = ((ar2_B[i])**2)*Denominator[i,:,:]/cvel
    NNUM= np.sum(NUM,axis=0)
    NDENO = np.sum(DENO,axis=0)
    
    N_VAR = Spec_variance[H10_R:L10_R,:,:]
    D_VAR = Spec_variance[H20_B:L20_B,:,:]
    N_ER = np.zeros(np.shape(N_VAR))
    D_ER = np.zeros(np.shape(D_VAR))
    i=0
    for i in range (0,len(N_VAR)):
        N_ER[i,:,:] = (ar1_R[i]**2)*np.sqrt(N_VAR[i,:,:])/cvel
    i=0
    for i in range (0,len(D_VAR)):
        D_ER[i,:,:] = (ar2_B[i]**2)*np.sqrt(D_VAR[i,:,:])/cvel
    N_ERS = np.sum(N_ER**2,axis=0)
    D_ERS = np.sum(D_ER**2,axis=0)
    
 #Calculate Spectral Indices and flux weighted dispersion (sigma^2*F_I)
    SPEC_Index_Mask = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    SPEC_Index_Flux = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    SPEC_Index_Cont = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    SPEC_Index_Flux_sigma = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    SPEC_Index_Cont_sigma = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    COMB_AVG = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    COMB_AVG_MASK = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    COMB_AVG_SIGMA = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    l =0
    for l in range (0,np.shape(mf)[0]):
       High_Blue = np.where(Waveh < L_Blue[l])
       Low_Blue = np.where(Waveh > H_Blue[l])
       HB0 = np.max(High_Blue)
       LB0 = np.min(Low_Blue)
       Wave_Blue = Waveh[HB0:LB0]
       Dlambda_Blue = Waveh[LB0] - Waveh[HB0]
       mid_Blue = np.int(np.shape(Wave_Blue)[0]/2)
       MB = np.where(Waveh == Wave_Blue[mid_Blue] )
       Flux_Blue = Rest_Flux[HB0:LB0]
       Cntm_Blue = np.trapz(Flux_Blue,x=Wave_Blue,axis=0)/Delt_B[l]
       Var_Blue = np.sum(Spec_variance[HB0:LB0,:,:],axis=0)
       # # Red band selection
       High_Red = np.where(Waveh < L_Red[l])
       Low_Red = np.where(Waveh > H_Red[l])
       HR0 = np.max(High_Red)
       LR0 = np.min(Low_Red)
       Wave_Red = Waveh[HR0:LR0]
       Dlambda_Red = (Waveh[LR0] - Waveh[HR0])
       mid_Red = np.int(np.shape(Wave_Red)[0]/2)
       MR = np.where(Waveh == Wave_Red[mid_Red] )
       Flux_Red = Rest_Flux[HR0:LR0]
       Cntm_Red = np.trapz(Flux_Red,x=Wave_Red,axis=0)/Delt_R[l]
       Slope = (Cntm_Red - Cntm_Blue)/(Rmid[l]-Bmid[l])
       Var_Red = np.sum(Spec_variance[HR0:LR0,:,:],axis=0)
       High_Index = np.where(Waveh < L_Index[l])
       Low_Index = np.where(Waveh > H_Index[l])
       HI0 = np.max(High_Index)
       LI0 = np.min(Low_Index)
       Wave_Index = Waveh[HI0:LI0]
       Dlambda_Index = Waveh[LI0] - Waveh[HI0]
       mid_Index = np.int(np.shape(Wave_Index)[0]/2)
       IndexFlux = Rest_Flux[HI0:LI0,:,:]
       Dlambda = Wave_Index.max() - Wave_Index.min()
       Var_Index = np.sum(Spec_variance[HI0:LI0,:,:],axis=0)
       Continua = Slope*(Imid[l]-Bmid[l]) + Cntm_Blue
       K_lambda = (Imid[l] - Bmid[l])/(Rmid[l]-Bmid[l])
       Var_Conti = (Var_Red)*(K_lambda**2) + (Var_Blue)*((K_lambda-1)**2)
       LGCB_EW = Delt_In[l] - np.trapz(IndexFlux/Continua,x=Wave_Index,axis=0)
       SPEC_Index_Flux[l,:,:] = np.trapz(IndexFlux,x=Wave_Index,axis=0)
       SPEC_Index_Cont[l,:,:] = Continua
       SPEC_Index_Flux_sigma[l,:,:] = Var_Index
       SPEC_Index_Cont_sigma[l,:,:] = Var_Conti
       SPEC_Index_Mask[l,:,:] = SP_MASK[l]
       
       IPK = np.min(np.where(Waveh>Imid[l]))
       C_DISP = Rest_Predisp[IPK,:,:]**2 + Astro_sigma_mnga
       COMB_AVG[l] = C_DISP*SPEC_Index_Flux[l]
       COMB_AVG_SIGMA[l] = np.sqrt(((Imid[l]*Vel_sigma_err*Vel_sigma)**2)/((2.54*300000)**2 + (Imid[l]*Vel_sigma)**2))
       COMB_AVG_MASK[l] = np.logical_or(SP_MASK[l],Vel_sigma_mask)

#Define MaNGA psf convolution kernel
    lmnga = math.ceil(3*SigmaM)
    dmnga = 0.5
    lw1 = math.ceil(SwftSigmaw1)
    dw1=1
    lm2 = math.ceil(SwftSigmam2)
    dm2=1
    
    sm = np.sqrt(SwftSigmaw2**2 - SigmaM**2) #Manga - uvw2 kernel
    x_mnga = np.arange((-lmnga)*dmnga,(lmnga*dmnga)+dmnga,step=dmnga)
    XM,YM = np.meshgrid(x_mnga,x_mnga)
    KM = np.exp(-(XM ** 2 + YM ** 2) / (2 * sm ** 2))
    Ga = KM/np.sum(KM)
    
#Define Swift psf convolution kernels fro uvw1 and uvm2
    sw1 = np.sqrt(SwftSigmaw2**2 - SwftSigmaw1**2) #uvw1 - uvw2 kernel
    x_swft_W1 = np.arange(-lw1,lw1+dw1,dw1)
    XW1,YW1 = np.meshgrid(x_swft_W1,x_swft_W1)
    KW1 = np.exp(-(XW1 ** 2 + YW1 ** 2) / (2 * sw1 ** 2))
    Ga1 = KW1/np.sum(KW1)
    
    sm2 = np.sqrt(SwftSigmaw2**2 - SwftSigmam2**2) #uvm2 - uvw2 kernel
    x_swft_M2 = np.arange(-lm2,lm2+dm2,dm2)
    XM2,YM2 = np.meshgrid(x_swft_M2,x_swft_M2)
    KM2 = np.exp(-(XM2 ** 2 + YM2 ** 2) / (2 * sm2 ** 2))
    Ga2 = KM2/np.sum(KM2)
    
#Locate the object in Swift's W2 filter and produce a cutout image of UVW2 with size sufficiently large to include
#everything in manga IFU.
    radec = np.array([objectra,objectdec])
    hdu = fits.open("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVW2_flx.fits")
    hdu_Flux = fits.open(get_pkg_data_filename("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVW2_flx.fits"))[0]
    convfact_W2 = hdu[0].header['FLMBDA']
    abz_W2 = hdu[0].header['ABMAGZP']
    skycnts_W2 = hdu[0].header['SKYC']
    eskycnts_W2 = hdu[0].header['ESKYC']
    wcs = WCS(hdu_Flux.header)
    P1,P2 = wcs.wcs_world2pix((objectra),(objectdec),1)
#Generate cutout and update the new header as wcs_N
    cutout = Cutout2D(hdu_Flux.data, position=(P1,P2), size=np.shape(GFLUX[0]), wcs=wcs)
    Flux_W2 = Cutout2D(hdu[0].data, position=(P1,P2), size=np.shape(GFLUX[0]), wcs=wcs)
    Flux_W2_sigma = Cutout2D(hdu[1].data, position=(P1,P2), size=np.shape(GFLUX[0]), wcs=wcs)
    Cnts_W2 = Cutout2D(hdu[2].data, position=(P1,P2), size=np.shape(GFLUX[0]), wcs=wcs)
    Ecnts_W2 = Cutout2D(hdu[3].data, position=(P1,P2), size=np.shape(GFLUX[0]), wcs=wcs)
    Exp_W2 = Cutout2D(hdu[4].data, position=(P1,P2), size=np.shape(GFLUX[0]), wcs=wcs)
    Mask_W2 = Cutout2D(hdu[5].data, position=(P1,P2), size=np.shape(GFLUX[0]), wcs=wcs)
    hdu_Flux.data = cutout.data
    hdu_Flux.header.update(cutout.wcs.to_header())
    wcs_N = WCS(hdu_Flux.header)
    
#Convolution & Reprojection of D4000 HDUS (Fluxes/Vars/Masks)

    CONV_D4R = signal.convolve2d(NNUM,Ga,boundary='symm',mode='same')
    CONV_D4B = signal.convolve2d(NDENO, Ga,boundary='symm',mode='same')
    CONV_D4R_SIG = signal.convolve2d(N_ERS,Ga,boundary='symm',mode='same')
    CONV_D4B_SIG = signal.convolve2d(D_ERS,Ga,boundary='symm',mode='same')
    CONV_D4_MASK = signal.convolve2d(SP_MASK[44],Ga,boundary='symm',mode='same')
    ND4_R = reproject_exact((CONV_D4R,H_manga), hdu_Flux.header)[0]
    ND4_B = reproject_exact((CONV_D4B,H_manga), hdu_Flux.header)[0]
    ND4_R_SIG = np.sqrt(reproject_exact((CONV_D4R_SIG,H_manga), hdu_Flux.header))[0]
    ND4_B_SIG = np.sqrt(reproject_exact((CONV_D4B_SIG,H_manga), hdu_Flux.header))[0]
    ND4_MASK = reproject_exact((CONV_D4_MASK,H_manga), hdu_Flux.header)[0]
#Cut out and Slice the exact overlap that cover MaNGA feild of view
    Slice = np.where(ND4_R==np.isnan(ND4_R))
#Small math done if the cutout reproject doesnot turn up a square map
    A1=Slice[0][0]
    A2=Slice[0][-1]
    B1=Slice[1][0]
    B2=Slice[1][-1]
    if (A2-A1)/(B2-B1) > 1:
        Slice[0][-1] = Slice[0][-1] - 1
    if (A2-A1)/(B2-B1) < 1:
        Slice[1][-1] = Slice[1][-1] - 1
 #Define new sliced D4000 values
    ND4_R = ND4_R[Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    ND4_B = ND4_B[Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    ND4_R_SIG = ND4_R_SIG[Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    ND4_B_SIG = ND4_B_SIG[Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    ND4_MASK = ND4_MASK[Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    
 #Define corner pixel RA,DEC for crval1,crval2   
    Ref_pix = wcs_N.wcs_pix2world(Slice[0][0],Slice[1][0],1)


#Convolution and reprojection of Emission lines continua and fluxes (& their variances and masks)
    CONV_EFLUX = np.zeros(np.shape(GFLUX_Clip))
    CONV_ECONT = np.zeros(np.shape(GFLUX_Clip))
    CONV_MASK_EF = np.zeros(np.shape(GFLUX_Clip))
    CONV_MASK_EE = np.zeros(np.shape(GFLUX_Clip))
    CONV_EFLUX_SIG = np.zeros(np.shape(GFLUX_Clip))
    CONV_EEW_SIG = np.zeros(np.shape(GFLUX_Clip))
    NEW_EFLUX = np.zeros((np.shape(GFLUX_Clip)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_ECONT = np.zeros((np.shape(GFLUX_Clip)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_MASK_EF = np.zeros((np.shape(GFLUX_Clip)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_MASK_EE = np.zeros((np.shape(GFLUX_Clip)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_EFLUX_SIG = np.zeros((np.shape(GFLUX_Clip)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_EEW_SIG = np.zeros((np.shape(GFLUX_Clip)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    p=0
    for p in range (0,np.shape(GFLUX_Clip)[0]):
        CONV_EFLUX[p,:,:] = signal.convolve2d(GFLUX_Clip[p,:,:]/(0.25),Ga,boundary='symm',mode='same')
        CONV_ECONT[p,:,:] = signal.convolve2d(GCONT_Clip[p,:,:]/(0.25),Ga,boundary='symm',mode='same')
        CONV_MASK_EF[p,:,:] = signal.convolve2d(GFMASK[p,:,:],Ga,boundary='symm',mode='same')
        CONV_MASK_EE[p,:,:] = signal.convolve2d(GEWMASK[p,:,:],Ga,boundary='symm',mode='same')
        CONV_EFLUX_SIG[p,:,:] = signal.convolve2d(GFLUX_VAR[p,:,:],Ga,boundary='symm',mode='same')
        CONV_EEW_SIG[p,:,:] = signal.convolve2d(GEW_VAR[p,:,:],Ga,boundary='symm',mode='same')
        NEW_EFLUX[p,:,:] = reproject_exact((CONV_EFLUX[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_ECONT[p,:,:] = reproject_exact((CONV_ECONT[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_MASK_EF[p,:,:] = reproject_exact((CONV_MASK_EF[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_MASK_EE[p,:,:] = reproject_exact((CONV_MASK_EE[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_EFLUX_SIG[p,:,:] = np.sqrt(reproject_exact((CONV_EFLUX_SIG[p,:,:],H_manga), hdu_Flux.header)[0])
        NEW_EEW_SIG[p,:,:] = np.sqrt(reproject_exact((CONV_EEW_SIG[p,:,:],H_manga), hdu_Flux.header)[0])

    CONV_IFLUX = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_ICONT = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_MASK_SI = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_IFLUX_SIG = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_ICONT_SIG = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_COMB_AVG = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_COMB_AVG_SIGMA = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_COMB_AVG_MASK = np.zeros(np.shape(SPEC_Index_Flux))

#Convolution and reprojection of Spectral Indices and flux weighted dispersions (& their variances and masks)
    NEW_IFLUX = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_ICONT = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_MASK_SI = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_IFLUX_SIG = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_ICONT_SIG = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_COMB_AVG = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_COMB_AVG_SIGMA = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_COMB_AVG_MASK = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    p=0
    for p in range (0,np.shape(SPEC_Index_Flux)[0]):
        CONV_IFLUX[p,:,:] = signal.convolve2d(SPEC_Index_Flux[p,:,:],Ga,boundary='symm',mode='same')
        CONV_ICONT[p,:,:] = signal.convolve2d(SPEC_Index_Cont[p,:,:],Ga,boundary='symm',mode='same')
        CONV_MASK_SI[p,:,:] = signal.convolve2d(SPEC_Index_Mask[p,:,:],Ga,boundary='symm',mode='same')
        CONV_IFLUX_SIG[p,:,:] = signal.convolve2d(SPEC_Index_Flux_sigma[p,:,:],Ga,boundary='symm',mode='same')
        CONV_ICONT_SIG[p,:,:] = signal.convolve2d(SPEC_Index_Cont_sigma[p,:,:],Ga,boundary='symm',mode='same')
        CONV_COMB_AVG[p,:,:] = signal.convolve2d(COMB_AVG[p,:,:]*4,Ga,boundary='symm',mode='same')
        CONV_COMB_AVG_SIGMA[p,:,:] = signal.convolve2d(COMB_AVG_SIGMA[p,:,:],Ga,boundary='symm',mode='same')
        CONV_COMB_AVG_MASK[p,:,:] = signal.convolve2d(COMB_AVG_MASK[p,:,:],Ga,boundary='symm',mode='same')
        NEW_IFLUX[p,:,:] = reproject_exact((CONV_IFLUX[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_ICONT[p,:,:] = reproject_exact((CONV_ICONT[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_MASK_SI[p,:,:] = reproject_exact((CONV_MASK_SI[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_IFLUX_SIG[p,:,:] = np.sqrt(reproject_exact((CONV_IFLUX_SIG[p,:,:],H_manga), hdu_Flux.header)[0])
        NEW_ICONT_SIG[p,:,:] = np.sqrt(reproject_exact((CONV_ICONT_SIG[p,:,:],H_manga), hdu_Flux.header)[0])
        NEW_COMB_AVG[p,:,:] = reproject_exact((CONV_COMB_AVG[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_COMB_AVG_SIGMA[p,:,:] = reproject_exact((CONV_COMB_AVG_SIGMA[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_COMB_AVG_MASK[p,:,:] = reproject_exact((CONV_COMB_AVG_MASK[p,:,:],H_manga), hdu_Flux.header)[0]
        
    NEW_EFLUX = NEW_EFLUX[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_ECONT = NEW_ECONT[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_MASK_EF = NEW_MASK_EF[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_MASK_EE = NEW_MASK_EE[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_EFLUX_SIG = NEW_EFLUX_SIG[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_EEW_SIG = NEW_EEW_SIG[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    
    NEW_IFLUX = NEW_IFLUX[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_ICONT = NEW_ICONT[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_MASK_SI = NEW_MASK_SI[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_IFLUX_SIG = NEW_IFLUX_SIG[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_ICONT_SIG = NEW_ICONT_SIG[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_COMB_AVG = NEW_COMB_AVG[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_COMB_AVG_SIGMA = NEW_COMB_AVG_SIGMA[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    NEW_COMB_AVG_MASK = NEW_COMB_AVG_MASK[:,Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    
 #CONVERT ALL FRACTIONAL MASK VALUES (values>0.5 are bad) TO INTEGER
    EFLUX_I1 = np.where(NEW_MASK_EF>0.5)
    ELINE_I1 = np.where(NEW_MASK_EE>0.5)
    SI_I1 = np.where(NEW_MASK_SI>0.5)
    D4_I1 = np.where(ND4_MASK>0.5)
    DISP_I1 = np.where(NEW_COMB_AVG_MASK>0.5)
    
    NEW_MASK_EF[EFLUX_I1]=1
    NEW_MASK_EE[ELINE_I1]=1
    NEW_MASK_SI[SI_I1]=1
    ND4_MASK[D4_I1]=1
    NEW_COMB_AVG_MASK[DISP_I1]=1
    
    EFLUX_I2 = np.where(NEW_MASK_EF<0.5)
    ELINE_I2 = np.where(NEW_MASK_EE<0.5)
    SI_I2 = np.where(NEW_MASK_SI<0.5)
    D4_I2 = np.where(ND4_MASK<0.5)
    DISP_I2 = np.where(NEW_COMB_AVG_MASK<0.5)
    
    NEW_MASK_EF[EFLUX_I2]=0
    NEW_MASK_EE[ELINE_I2]=0
    NEW_MASK_SI[SI_I2]=0
    ND4_MASK[D4_I2]=0
    NEW_COMB_AVG_MASK[DISP_I2]=0
    
#Convolution and reprojection of Swift UVM2
    hdu = fits.open("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVM2_flx.fits")
    uvm2_h = hdu[0].header
    Flux_M2 = signal.convolve2d(hdu[0].data,Ga2,boundary='symm',mode='same')
    Flux_M2_sigma = signal.convolve2d(hdu[1].data**2,Ga2,boundary='symm',mode='same')
    Cnts_M2 = signal.convolve2d(hdu[2].data,Ga2,boundary='symm',mode='same')
    Ecnts_M2 = signal.convolve2d(hdu[3].data**2,Ga2,boundary='symm',mode='same')
    Exp_M2 = signal.convolve2d(hdu[4].data,Ga2,boundary='symm',mode='same')
    Mask_M2 = signal.convolve2d(hdu[5].data,Ga2,boundary='symm',mode='same')
    convfact_M2 = hdu[0].header['FLMBDA']
    abz_M2 = hdu[0].header['ABMAGZP']
    skycnts_M2 = hdu[0].header['SKYC']
    eskycnts_M2 = hdu[0].header['ESKYC']
    
    FM2 = reproject_exact((Flux_M2,uvm2_h), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    FEM2 = np.sqrt(reproject_exact((Flux_M2_sigma,uvm2_h), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]])
    CM2 = reproject_exact((Cnts_M2,uvm2_h), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    EM2 = np.sqrt(reproject_exact((Ecnts_M2,uvm2_h), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]])
    EXM2 = reproject_exact((Exp_M2,uvm2_h), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    MKM2 = reproject_exact((Mask_M2,uvm2_h),hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    IKM2 = np.where(MKM2>0)
    MKM2[IKM2]=1
    
#Convolution and reprojection of Swift UVW1
    hdu = fits.open("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVW1_flx.fits")
    uvw1_h = hdu[0].header
    Flux_W1 = signal.convolve2d(hdu[0].data,Ga1,boundary='symm',mode='same')
    Flux_W1_sigma = signal.convolve2d(hdu[1].data**2,Ga1,boundary='symm',mode='same')
    Cnts_W1 = signal.convolve2d(hdu[2].data,Ga1,boundary='symm',mode='same')
    Ecnts_W1 = signal.convolve2d(hdu[3].data**2,Ga1,boundary='symm',mode='same')
    Exp_W1 = signal.convolve2d(hdu[4].data,Ga1,boundary='symm',mode='same')
    Mask_W1 = signal.convolve2d(hdu[5].data,Ga1,boundary='symm',mode='same')
    convfact_W1 = hdu[0].header['FLMBDA']
    abz_W1 = hdu[0].header['ABMAGZP']
    skycnts_W1 = hdu[0].header['SKYC']
    eskycnts_W1 = hdu[0].header['ESKYC']
    FW1 = reproject_exact((Flux_W1,uvw1_h), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    FEW1 = np.sqrt(reproject_exact((Flux_W1_sigma,uvw1_h), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]])
    CW1 = reproject_exact((Cnts_W1,uvw1_h), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    EW1 = np.sqrt(reproject_exact((Ecnts_W1,uvw1_h), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]])
    EXW1 = reproject_exact((Exp_W1,uvw1_h), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    MKW1 = reproject_exact((Mask_W1,uvw1_h),hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    IKW1 = np.where(MKW1>0)
    MKW1[IKW1]=1
    
#Generate cutouts on original UVW2 image
    FW2 = Flux_W2.data[Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    FEW2 = Flux_W2_sigma.data[Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    CW2 = Cnts_W2.data[Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    EW2 = Ecnts_W2.data[Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    EXW2 = Exp_W2.data[Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    MKW2 = Mask_W2.data[Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    
    
    
 #Convolution kernel for sloan
    SigmaSloan = (1.4/2.355)
    lsln = math.ceil(4*SigmaSloan)
    dsln = 0.396
    sln = np.sqrt(SwftSigmaw2**2 - SigmaSloan**2)
    x_sln1 = np.arange(0,lsln+1,0.396)
    x_sln = np.zeros(np.shape(x_sln1)[0]+np.shape(x_sln1)[0]-1)
    x_sln[10:21] = x_sln1
    x_sln[0:10] = np.flip(x_sln1[1:21])*(-1)
    XS,YS = np.meshgrid(x_sln,x_sln)
    KS = np.exp(-(XS ** 2 + YS ** 2) / (2 * sln ** 2))
    G = KS/np.sum(KS)
    

    
#Concolution and reprojection of SDSS u image    
    hdu = fits.open("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-u.fits")
    cvu1 = getheader("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-u.fits",0)
    SDSS_u = hdu[0].data/(0.156816) #convert Flux from /spaxel^2 to /arcsec%2
    SDSS_u_var = 1/hdu[1].data
    SDSS_u_var[np.isnan(SDSS_u_var)] = 0
    SDSS_u_var[SDSS_u_var == inf] = 0
    SDSS_u_var[SDSS_u_var == -inf] = 0
    conv_u = signal.convolve2d(SDSS_u, G, boundary='symm', mode='same')
    conv_u_var = signal.convolve2d(SDSS_u_var,G, boundary='symm', mode='same')
    new_u = reproject_exact((conv_u,cvu1), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    new_u_sig = np.sqrt(reproject_exact((conv_u_var,cvu1), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]])
    
#Concolution and reprojection of SDSS g image       
    hdu = fits.open("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-g.fits")
    cvg1 = getheader("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-g.fits",0)
    SDSS_g = hdu[0].data/(0.156816) #convert Flux from /spaxel^2 to /arcsec^2
    SDSS_g_var = 1/hdu[1].data
    SDSS_g_var[np.isnan(SDSS_g_var)] = 0
    SDSS_g_var[SDSS_g_var == inf] = 0
    SDSS_g_var[SDSS_g_var == -inf] = 0
    conv_g = signal.convolve2d(SDSS_g, G, boundary='symm', mode='same')
    conv_g_var = signal.convolve2d(SDSS_g_var,G, boundary='symm', mode='same')
    new_g = reproject_exact((conv_g,cvg1), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    new_g_sig = np.sqrt(reproject_exact((conv_g_var,cvg1), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]])

#Concolution and reprojection of SDSS r image   
    hdu = fits.open("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-r.fits")
    cvr1 = getheader("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-r.fits",0)
    SDSS_r = hdu[0].data/(0.156816) #convert Flux from /spaxel^2 to /arcsec^2
    SDSS_r_var = 1/hdu[1].data
    SDSS_r_var[np.isnan(SDSS_r_var)] = 0
    SDSS_r_var[SDSS_r_var == inf] = 0
    SDSS_r_var[SDSS_r_var == -inf] = 0
    conv_r = signal.convolve2d(SDSS_r,G, boundary='symm', mode='same')
    conv_r_var = signal.convolve2d(SDSS_r_var,G, boundary='symm', mode='same')
    new_r = reproject_exact((conv_r,cvr1), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    new_r_sig = np.sqrt(reproject_exact((conv_r_var,cvr1), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]])
    
 #Concolution and reprojection of SDSS i image      
    hdu = fits.open("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-i.fits")
    cvi1 = getheader("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-i.fits",0)
    SDSS_i = hdu[0].data/(0.156816) #convert Flux from /spaxel^2 to /arcsec^2
    SDSS_i_var = 1/hdu[1].data
    SDSS_i_var[np.isnan(SDSS_i_var)] = 0
    SDSS_i_var[SDSS_i_var == inf] = 0
    SDSS_i_var[SDSS_i_var == -inf] = 0
    conv_i = signal.convolve2d(SDSS_i,G, boundary='symm', mode='same')
    conv_i_var = signal.convolve2d(SDSS_i_var,G, boundary='symm', mode='same')
    new_i = reproject_exact((conv_i,cvi1), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    new_i_sig = np.sqrt(reproject_exact((conv_i_var,cvi1), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]])
    
#Concolution and reprojection of SDSS z image       
    hdu = fits.open("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-z.fits")
    cvz1 = getheader("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-z.fits",0)
    SDSS_z = hdu[0].data/(0.156816) #convert Flux from /spaxel^2 to /arcsec^2
    SDSS_z_var = 1/hdu[1].data
    SDSS_z_var[np.isnan(SDSS_z_var)] = 0
    SDSS_z_var[SDSS_z_var == inf] = 0
    SDSS_z_var[SDSS_z_var == -inf] = 0
    conv_z = signal.convolve2d(SDSS_z, G, boundary='symm', mode='same')
    conv_z_var = signal.convolve2d(SDSS_z_var,G, boundary='symm', mode='same')
    new_z = reproject_exact((conv_z,cvz1), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]]
    new_z_sig = np.sqrt(reproject_exact((conv_z_var,cvz1), hdu_Flux.header)[0][Slice[0][0]:Slice[0][-1],Slice[1][0]:Slice[1][-1]])


#Define D4000 HDU and make sure all nans,infs are replaced with zero
    D4000_HDU = np.zeros((5,np.shape(ND4_R)[0],np.shape(ND4_R)[1]))
    D4000_HDU[0] = ND4_R
    D4000_HDU[1] = ND4_B
    D4000_HDU[2] = ND4_R_SIG
    D4000_HDU[3] = ND4_B_SIG
    D4000_HDU[4] = ND4_MASK
    D4000_HDU[np.isnan(D4000_HDU)] = 0
    D4000_HDU[D4000_HDU== inf] = 0
    D4000_HDU[D4000_HDU == -inf] = 0
#make sure all nans,infs are replaced with zero
    NEW_IFLUX[np.isnan(NEW_IFLUX)]=0
    NEW_IFLUX[NEW_IFLUX== inf] = 0
    NEW_IFLUX[NEW_IFLUX== -inf] = 0
    NEW_IFLUX_SIG[np.isnan(NEW_IFLUX_SIG)]=0
    NEW_IFLUX_SIG[NEW_IFLUX_SIG== inf] = 0
    NEW_IFLUX_SIG[NEW_IFLUX_SIG== -inf] = 0
    NEW_ICONT[np.isnan(NEW_ICONT)]=0
    NEW_ICONT[NEW_ICONT== inf] = 0
    NEW_ICONT[NEW_ICONT== -inf] = 0
    NEW_ICONT_SIG[np.isnan(NEW_ICONT_SIG)]=0
    NEW_ICONT_SIG[NEW_ICONT_SIG== inf] = 0
    NEW_ICONT_SIG[NEW_ICONT_SIG== -inf] = 0
    NEW_DISP = np.sqrt(NEW_COMB_AVG/(NEW_IFLUX))
    NEW_DISP[np.isnan(NEW_DISP)] = 0
    NEW_DISP[NEW_DISP== inf] = 0
    NEW_DISP[NEW_DISP == -inf] = 0
    NEW_COMB_AVG_SIGMA[np.isnan(NEW_COMB_AVG_SIGMA)] = 0
    NEW_COMB_AVG_SIGMA[NEW_COMB_AVG_SIGMA== inf] = 0
    NEW_COMB_AVG_SIGMA[NEW_COMB_AVG_SIGMA == -inf] = 0
    NEW_EFLUX[np.isnan(NEW_EFLUX)]=0
    NEW_EFLUX[NEW_EFLUX== inf] = 0
    NEW_EFLUX[NEW_EFLUX== -inf] = 0
    NEW_EFLUX_SIG[np.isnan(NEW_EFLUX_SIG)]=0
    NEW_EFLUX_SIG[NEW_EFLUX_SIG== inf] = 0
    NEW_EFLUX_SIG[NEW_EFLUX_SIG== -inf] = 0
#Define EW from continua and flux
    NEW_EEW = NEW_EFLUX/NEW_ECONT
    NEW_EEW[np.isnan(NEW_EEW)]=0
    NEW_EEW[NEW_EEW== inf] = 0
    NEW_EEW[NEW_EEW == -inf] = 0
#Define photometric fluxes HDU
    PHOTO = np.zeros((8,np.shape(ND4_R)[0],np.shape(ND4_R)[1]))
    PHOTO_SIG = np.zeros((8,np.shape(ND4_R)[0],np.shape(ND4_R)[1]))
    PHOTO[0,:,:] = FW2
    PHOTO[1,:,:] = FW1
    PHOTO[2,:,:] = FM2
    PHOTO[3,:,:] = new_u
    PHOTO[4,:,:] = new_g
    PHOTO[5,:,:] = new_r
    PHOTO[6,:,:] = new_i
    PHOTO[7,:,:] = new_z
    PHOTO_SIG[0,:,:] = FEW2
    PHOTO_SIG[1,:,:] = FEW1
    PHOTO_SIG[2,:,:] = FEM2
    PHOTO_SIG[3,:,:] = new_u_sig
    PHOTO_SIG[4,:,:] = new_g_sig
    PHOTO_SIG[5,:,:] = new_r_sig
    PHOTO_SIG[6,:,:] = new_i_sig
    PHOTO_SIG[7,:,:] = new_z_sig
    PHOTO[np.isnan(PHOTO)]=0
    PHOTO[PHOTO== inf] = 0
    PHOTO[PHOTO == -inf] = 0
    PHOTO_SIG[np.isnan(PHOTO_SIG)]=0
    PHOTO_SIG[PHOTO_SIG== inf] = 0
    PHOTO_SIG[PHOTO_SIG == -inf] = 0
#Define original UVOT HDU
    UVOT = np.zeros((12,np.shape(ND4_R)[0],np.shape(ND4_R)[1]))
    UVOT[0,:,:] = CW2 + skycnts_W2
    UVOT[1,:,:] = CW1 + skycnts_W1
    UVOT[2,:,:] = CM2 + skycnts_M2
    UVOT[3,:,:] = EW2
    UVOT[4,:,:] = EW1
    UVOT[5,:,:] = EM2
    UVOT[6,:,:] = EXW2
    UVOT[7,:,:] = EXW1
    UVOT[8,:,:] = EXM2
    UVOT[9,:,:] = MKW2
    UVOT[10,:,:] = MKW1
    UVOT[11,:,:] = MKM2
    UVOT[np.isnan(UVOT)]=0
    UVOT[UVOT== inf] = 0
    UVOT[UVOT== -inf] = 0
    
#RA,DEC for (0,0) pixel    
    crval1 = np.asscalar(Ref_pix[0])
    crval2 = np.asscalar(Ref_pix[1])
    
    
    hdu0 = fits.PrimaryHDU(D4000_HDU)
    hdu1 = fits.ImageHDU(NEW_IFLUX)
    hdu2 = fits.ImageHDU(NEW_ICONT)
    hdu3 = fits.ImageHDU(NEW_IFLUX_SIG)
    hdu4 = fits.ImageHDU(NEW_ICONT_SIG)
    hdu5 = fits.ImageHDU(NEW_MASK_SI)
    hdu6 = fits.ImageHDU(NEW_DISP)
    hdu7 = fits.ImageHDU(NEW_COMB_AVG_SIGMA)
    hdu8 = fits.ImageHDU(NEW_COMB_AVG_MASK)
    hdu9 = fits.ImageHDU(NEW_EFLUX)
    hdu10 = fits.ImageHDU(NEW_EFLUX_SIG)
    hdu11 = fits.ImageHDU(NEW_MASK_EF)
    hdu12 = fits.ImageHDU(NEW_EEW)
    hdu13 = fits.ImageHDU(NEW_EEW_SIG)
    hdu14 = fits.ImageHDU(NEW_MASK_EE)
    hdu15 = fits.ImageHDU(PHOTO)
    hdu16 = fits.ImageHDU(PHOTO_SIG)
    hdu17 = fits.ImageHDU(UVOT)
 
 #Convert all MASK HDUs to int format
    hdu5.scale('int32')
    hdu8.scale('int32')
    hdu11.scale('int32')
    hdu14.scale('int32')
    
    new_hdul = fits.HDUList([hdu0,hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7,hdu8,hdu9,hdu10,hdu11,hdu12,hdu13,hdu14,hdu15,hdu16,hdu17])
    new_hdul.writeto("/Volumes/Nikhil/Data/SWIFT/MPL-7_RP/SwiM_"+str(Line[q])+".fits")

    hdu = fits.open("/Volumes/Nikhil/Data/SWIFT/MPL-7_RP/SwiM_"+str(Line[q])+".fits")
    
    hdr0 = hdu[0].header
    hdr0['PLATE'] = plate
    hdr0['IFUDSGN'] = np.int(ifu)
    hdr0['EXTNAME'] = 'D4000'
    hdr0['CTYPE1'] = 'RA---TAN'
    hdr0['CTYPE2'] = 'DEC--TAN'
    hdr0['CDELT1'] = -0.000278888884
    hdr0['CDELT2'] =  0.000278888884
    hdr0['CRPIX1'] = 0
    hdr0['CRPIX2'] = 0
    hdr0['CUNIT1'] = 'deg'
    hdr0['CUNIT2'] = 'deg'
    hdr0['RADECSYS'] = 'FK5'
    hdr0['CRVAL1'] = crval1
    hdr0['CRVAL2'] = crval2
    hdr0['UNIT'] = 'erg/s/cm^2/Hz/arcsec^2'
    hdr0['C0'] = 'Fnu Red'
    hdr0['C1'] = 'Fnu Blue'
    hdr0['C2'] = 'Sigma Red'
    hdr0['C3'] = 'Sigma Blue'
    hdr0['C4'] = 'Mask'

    
    
    hdr1 = hdu[1].header
    hdr1['PLATE'] = plate
    hdr1['IFUDSGN'] = np.int(ifu)
    hdr1['EXTNAME'] = 'SPECINDX_FLUX'
    hdr1['CTYPE1'] = 'RA---TAN'
    hdr1['CTYPE2'] = 'DEC--TAN'
    hdr1['CDELT1'] = -0.000278888884
    hdr1['CDELT2'] =  0.000278888884
    hdr1['CRPIX1'] = 0
    hdr1['CRPIX2'] = 0
    hdr1['CUNIT1'] = 'deg'
    hdr1['CUNIT2'] = 'deg'
    hdr1['RADECSYS'] = 'FK5'
    hdr1['CRVAL1'] = crval1
    hdr1['CRVAL2'] = crval2
    hdr1['UNIT'] = 'erg/s/cm^2/arcsec^2'
    hdr1['C0'] = Indices[0]
    hdr1['C1'] = Indices[1]
    hdr1['C2'] = Indices[2]
    hdr1['C3'] = Indices[3]
    hdr1['C4'] = Indices[4]
    hdr1['C5'] = Indices[5]
    hdr1['C6'] = Indices[6]
    hdr1['C7'] = Indices[7]
    hdr1['C8'] = Indices[8]
    hdr1['C9'] = Indices[9]
    hdr1['C10'] = Indices[10]
    hdr1['C11'] = Indices[11]
    hdr1['C12'] = Indices[12]
    hdr1['C13'] = Indices[13]
    hdr1['C14'] = Indices[14]
    hdr1['C15'] = Indices[15]
    hdr1['C16'] = Indices[16]
    hdr1['C17'] = Indices[17]
    hdr1['C18'] = Indices[18]
    hdr1['C19'] = Indices[19]
    hdr1['C20'] = Indices[20]
    hdr1['C21'] = Indices[21]
    hdr1['C22'] = Indices[22]
    hdr1['C23'] = Indices[23]
    hdr1['C24'] = Indices[24]
    hdr1['C25'] = Indices[25]
    hdr1['C26'] = Indices[26]
    hdr1['C27'] = Indices[27]
    hdr1['C28'] = Indices[28]
    hdr1['C29'] = Indices[29]
    hdr1['C30'] = Indices[30]
    hdr1['C31'] = Indices[31]
    hdr1['C32'] = Indices[32]
    hdr1['C33'] = Indices[33]
    hdr1['C34'] = Indices[34]
    hdr1['C35'] = Indices[35]
    hdr1['C36'] = Indices[36]
    hdr1['C37'] = Indices[37]
    hdr1['C38'] = Indices[38]
    hdr1['C39'] = Indices[39]
    hdr1['C40'] = Indices[40]
    hdr1['C41'] = Indices[41]
    hdr1['C42'] = Indices[42]

    
    hdr2 = hdu[2].header
    hdr2['PLATE'] = plate
    hdr2['IFUDSGN'] = np.int(ifu)
    hdr2['EXTNAME'] = 'SPECINDX_CONT'
    hdr2['CTYPE1'] = 'RA---TAN'
    hdr2['CTYPE2'] = 'DEC--TAN'
    hdr2['CDELT1'] = -0.000278888884
    hdr2['CDELT2'] =  0.000278888884
    hdr2['CRPIX1'] = 0
    hdr2['CRPIX2'] = 0
    hdr2['CUNIT1'] = 'deg'
    hdr2['CUNIT2'] = 'deg'
    hdr2['RADECSYS'] = 'FK5'
    hdr2['CRVAL1'] = crval1
    hdr2['CRVAL2'] = crval2
    hdr2['UNIT'] = 'erg/s/cm^2/A/arcsec^2'
    hdr2['C0'] = Indices[0]
    hdr2['C1'] = Indices[1]
    hdr2['C2'] = Indices[2]
    hdr2['C3'] = Indices[3]
    hdr2['C4'] = Indices[4]
    hdr2['C5'] = Indices[5]
    hdr2['C6'] = Indices[6]
    hdr2['C7'] = Indices[7]
    hdr2['C8'] = Indices[8]
    hdr2['C9'] = Indices[9]
    hdr2['C10'] = Indices[10]
    hdr2['C11'] = Indices[11]
    hdr2['C12'] = Indices[12]
    hdr2['C13'] = Indices[13]
    hdr2['C14'] = Indices[14]
    hdr2['C15'] = Indices[15]
    hdr2['C16'] = Indices[16]
    hdr2['C17'] = Indices[17]
    hdr2['C18'] = Indices[18]
    hdr2['C19'] = Indices[19]
    hdr2['C20'] = Indices[20]
    hdr2['C21'] = Indices[21]
    hdr2['C22'] = Indices[22]
    hdr2['C23'] = Indices[23]
    hdr2['C24'] = Indices[24]
    hdr2['C25'] = Indices[25]
    hdr2['C26'] = Indices[26]
    hdr2['C27'] = Indices[27]
    hdr2['C28'] = Indices[28]
    hdr2['C29'] = Indices[29]
    hdr2['C30'] = Indices[30]
    hdr2['C31'] = Indices[31]
    hdr2['C32'] = Indices[32]
    hdr2['C33'] = Indices[33]
    hdr2['C34'] = Indices[34]
    hdr2['C35'] = Indices[35]
    hdr2['C36'] = Indices[36]
    hdr2['C37'] = Indices[37]
    hdr2['C38'] = Indices[38]
    hdr2['C39'] = Indices[39]
    hdr2['C40'] = Indices[40]
    hdr2['C41'] = Indices[41]
    hdr2['C42'] = Indices[42]

    
    hdr3 = hdu[3].header
    hdr3['PLATE'] = plate
    hdr3['IFUDSGN'] = np.int(ifu)
    hdr3['EXTNAME'] = 'SPECINDX_FLUX_SIGMA'
    hdr3['CTYPE1'] = 'RA---TAN'
    hdr3['CTYPE2'] = 'DEC--TAN'
    hdr3['CDELT1'] = -0.000278888884
    hdr3['CDELT2'] =  0.000278888884
    hdr3['CRPIX1'] = 0
    hdr3['CRPIX2'] = 0
    hdr3['CUNIT1'] = 'deg'
    hdr3['CUNIT2'] = 'deg'
    hdr3['RADECSYS'] = 'FK5'
    hdr3['CRVAL1'] = crval1
    hdr3['CRVAL2'] = crval2
    hdr3['UNIT'] = 'erg/s/cm^2/arcsec^2'
    hdr3['C0'] = Indices[0]
    hdr3['C1'] = Indices[1]
    hdr3['C2'] = Indices[2]
    hdr3['C3'] = Indices[3]
    hdr3['C4'] = Indices[4]
    hdr3['C5'] = Indices[5]
    hdr3['C6'] = Indices[6]
    hdr3['C7'] = Indices[7]
    hdr3['C8'] = Indices[8]
    hdr3['C9'] = Indices[9]
    hdr3['C10'] = Indices[10]
    hdr3['C11'] = Indices[11]
    hdr3['C12'] = Indices[12]
    hdr3['C13'] = Indices[13]
    hdr3['C14'] = Indices[14]
    hdr3['C15'] = Indices[15]
    hdr3['C16'] = Indices[16]
    hdr3['C17'] = Indices[17]
    hdr3['C18'] = Indices[18]
    hdr3['C19'] = Indices[19]
    hdr3['C20'] = Indices[20]
    hdr3['C21'] = Indices[21]
    hdr3['C22'] = Indices[22]
    hdr3['C23'] = Indices[23]
    hdr3['C24'] = Indices[24]
    hdr3['C25'] = Indices[25]
    hdr3['C26'] = Indices[26]
    hdr3['C27'] = Indices[27]
    hdr3['C28'] = Indices[28]
    hdr3['C29'] = Indices[29]
    hdr3['C30'] = Indices[30]
    hdr3['C31'] = Indices[31]
    hdr3['C32'] = Indices[32]
    hdr3['C33'] = Indices[33]
    hdr3['C34'] = Indices[34]
    hdr3['C35'] = Indices[35]
    hdr3['C36'] = Indices[36]
    hdr3['C37'] = Indices[37]
    hdr3['C38'] = Indices[38]
    hdr3['C39'] = Indices[39]
    hdr3['C40'] = Indices[40]
    hdr3['C41'] = Indices[41]
    hdr3['C42'] = Indices[42]


    hdr4 = hdu[4].header
    hdr4['PLATE'] = plate
    hdr4['IFUDSGN'] = np.int(ifu)
    hdr4['EXTNAME'] = 'SPECINDX_CONT_SIGMA'
    hdr4['CTYPE1'] = 'RA---TAN'
    hdr4['CTYPE2'] = 'DEC--TAN'
    hdr4['CDELT1'] = -0.000278888884
    hdr4['CDELT2'] =  0.000278888884
    hdr4['CRPIX1'] = 0
    hdr4['CRPIX2'] = 0
    hdr4['CUNIT1'] = 'deg'
    hdr4['CUNIT2'] = 'deg'
    hdr4['RADECSYS'] = 'FK5'
    hdr4['CRVAL1'] = crval1
    hdr4['CRVAL2'] = crval2
    hdr4['UNIT'] = 'erg/s/cm^2/A/arcsec^2'
    hdr4['C0'] = Indices[0]
    hdr4['C1'] = Indices[1]
    hdr4['C2'] = Indices[2]
    hdr4['C3'] = Indices[3]
    hdr4['C4'] = Indices[4]
    hdr4['C5'] = Indices[5]
    hdr4['C6'] = Indices[6]
    hdr4['C7'] = Indices[7]
    hdr4['C8'] = Indices[8]
    hdr4['C9'] = Indices[9]
    hdr4['C10'] = Indices[10]
    hdr4['C11'] = Indices[11]
    hdr4['C12'] = Indices[12]
    hdr4['C13'] = Indices[13]
    hdr4['C14'] = Indices[14]
    hdr4['C15'] = Indices[15]
    hdr4['C16'] = Indices[16]
    hdr4['C17'] = Indices[17]
    hdr4['C18'] = Indices[18]
    hdr4['C19'] = Indices[19]
    hdr4['C20'] = Indices[20]
    hdr4['C21'] = Indices[21]
    hdr4['C22'] = Indices[22]
    hdr4['C23'] = Indices[23]
    hdr4['C24'] = Indices[24]
    hdr4['C25'] = Indices[25]
    hdr4['C26'] = Indices[26]
    hdr4['C27'] = Indices[27]
    hdr4['C28'] = Indices[28]
    hdr4['C29'] = Indices[29]
    hdr4['C30'] = Indices[30]
    hdr4['C31'] = Indices[31]
    hdr4['C32'] = Indices[32]
    hdr4['C33'] = Indices[33]
    hdr4['C34'] = Indices[34]
    hdr4['C35'] = Indices[35]
    hdr4['C36'] = Indices[36]
    hdr4['C37'] = Indices[37]
    hdr4['C38'] = Indices[38]
    hdr4['C39'] = Indices[39]
    hdr4['C40'] = Indices[40]
    hdr4['C41'] = Indices[41]
    hdr4['C42'] = Indices[42]

    hdr5 = hdu[5].header
    hdr5['PLATE'] = plate
    hdr5['IFUDSGN'] = np.int(ifu)
    hdr5['EXTNAME'] = 'SPECINDX_MASK'
    hdr5['CTYPE1'] = 'RA---TAN'
    hdr5['CTYPE2'] = 'DEC--TAN'
    hdr5['CDELT1'] = -0.000278888884
    hdr5['CDELT2'] =  0.000278888884
    hdr5['CRPIX1'] = 0
    hdr5['CRPIX2'] = 0
    hdr5['CUNIT1'] = 'deg'
    hdr5['CUNIT2'] = 'deg'
    hdr5['RADECSYS'] = 'FK5'
    hdr5['CRVAL1'] = crval1
    hdr5['CRVAL2'] = crval2
    hdr5['C0'] = Indices[0]
    hdr5['C1'] = Indices[1]
    hdr5['C2'] = Indices[2]
    hdr5['C3'] = Indices[3]
    hdr5['C4'] = Indices[4]
    hdr5['C5'] = Indices[5]
    hdr5['C6'] = Indices[6]
    hdr5['C7'] = Indices[7]
    hdr5['C8'] = Indices[8]
    hdr5['C9'] = Indices[9]
    hdr5['C10'] = Indices[10]
    hdr5['C11'] = Indices[11]
    hdr5['C12'] = Indices[12]
    hdr5['C13'] = Indices[13]
    hdr5['C14'] = Indices[14]
    hdr5['C15'] = Indices[15]
    hdr5['C16'] = Indices[16]
    hdr5['C17'] = Indices[17]
    hdr5['C18'] = Indices[18]
    hdr5['C19'] = Indices[19]
    hdr5['C20'] = Indices[20]
    hdr5['C21'] = Indices[21]
    hdr5['C22'] = Indices[22]
    hdr5['C23'] = Indices[23]
    hdr5['C24'] = Indices[24]
    hdr5['C25'] = Indices[25]
    hdr5['C26'] = Indices[26]
    hdr5['C27'] = Indices[27]
    hdr5['C28'] = Indices[28]
    hdr5['C29'] = Indices[29]
    hdr5['C30'] = Indices[30]
    hdr5['C31'] = Indices[31]
    hdr5['C32'] = Indices[32]
    hdr5['C33'] = Indices[33]
    hdr5['C34'] = Indices[34]
    hdr5['C35'] = Indices[35]
    hdr5['C36'] = Indices[36]
    hdr5['C37'] = Indices[37]
    hdr5['C38'] = Indices[38]
    hdr5['C39'] = Indices[39]
    hdr5['C40'] = Indices[40]
    hdr5['C41'] = Indices[41]
    hdr5['C42'] = Indices[42]
    
    hdrp = hdu[6].header
    hdrp['PLATE'] = plate
    hdrp['IFUDSGN'] = np.int(ifu)
    hdrp['EXTNAME'] = 'COMBINED_DISP'
    hdrp['CTYPE1'] = 'RA---TAN'
    hdrp['CTYPE2'] = 'DEC--TAN'
    hdrp['CDELT1'] = -0.000278888884
    hdrp['CDELT2'] =  0.000278888884
    hdrp['CRPIX1'] = 0
    hdrp['CRPIX2'] = 0
    hdrp['CUNIT1'] = 'deg'
    hdrp['CUNIT2'] = 'deg'
    hdrp['RADECSYS'] = 'FK5'
    hdrp['UNIT'] = 'km/s'
    hdrp['CRVAL1'] = crval1
    hdrp['CRVAL2'] = crval2
    hdrp['C0'] = Indices[0]
    hdrp['C1'] = Indices[1]
    hdrp['C2'] = Indices[2]
    hdrp['C3'] = Indices[3]
    hdrp['C4'] = Indices[4]
    hdrp['C5'] = Indices[5]
    hdrp['C6'] = Indices[6]
    hdrp['C7'] = Indices[7]
    hdrp['C8'] = Indices[8]
    hdrp['C9'] = Indices[9]
    hdrp['C10'] = Indices[10]
    hdrp['C11'] = Indices[11]
    hdrp['C12'] = Indices[12]
    hdrp['C13'] = Indices[13]
    hdrp['C14'] = Indices[14]
    hdrp['C15'] = Indices[15]
    hdrp['C16'] = Indices[16]
    hdrp['C17'] = Indices[17]
    hdrp['C18'] = Indices[18]
    hdrp['C19'] = Indices[19]
    hdrp['C20'] = Indices[20]
    hdrp['C21'] = Indices[21]
    hdrp['C22'] = Indices[22]
    hdrp['C23'] = Indices[23]
    hdrp['C24'] = Indices[24]
    hdrp['C25'] = Indices[25]
    hdrp['C26'] = Indices[26]
    hdrp['C27'] = Indices[27]
    hdrp['C28'] = Indices[28]
    hdrp['C29'] = Indices[29]
    hdrp['C30'] = Indices[30]
    hdrp['C31'] = Indices[31]
    hdrp['C32'] = Indices[32]
    hdrp['C33'] = Indices[33]
    hdrp['C34'] = Indices[34]
    hdrp['C35'] = Indices[35]
    hdrp['C36'] = Indices[36]
    hdrp['C37'] = Indices[37]
    hdrp['C38'] = Indices[38]
    hdrp['C39'] = Indices[39]
    hdrp['C40'] = Indices[40]
    hdrp['C41'] = Indices[41]
    hdrp['C42'] = Indices[42]
    
    hdrp1 = hdu[7].header
    hdrp1['PLATE'] = plate
    hdrp1['IFUDSGN'] = np.int(ifu)
    hdrp1['EXTNAME'] = 'COMBINED_DISP_SIGMA'
    hdrp1['CTYPE1'] = 'RA---TAN'
    hdrp1['CTYPE2'] = 'DEC--TAN'
    hdrp1['CDELT1'] = -0.000278888884
    hdrp1['CDELT2'] =  0.000278888884
    hdrp1['CRPIX1'] = 0
    hdrp1['CRPIX2'] = 0
    hdrp1['CUNIT1'] = 'deg'
    hdrp1['CUNIT2'] = 'deg'
    hdrp1['RADECSYS'] = 'FK5'
    hdrp1['UNIT'] = 'km/s'
    hdrp1['CRVAL1'] = crval1
    hdrp1['CRVAL2'] = crval2
    hdrp1['C0'] = Indices[0]
    hdrp1['C1'] = Indices[1]
    hdrp1['C2'] = Indices[2]
    hdrp1['C3'] = Indices[3]
    hdrp1['C4'] = Indices[4]
    hdrp1['C5'] = Indices[5]
    hdrp1['C6'] = Indices[6]
    hdrp1['C7'] = Indices[7]
    hdrp1['C8'] = Indices[8]
    hdrp1['C9'] = Indices[9]
    hdrp1['C10'] = Indices[10]
    hdrp1['C11'] = Indices[11]
    hdrp1['C12'] = Indices[12]
    hdrp1['C13'] = Indices[13]
    hdrp1['C14'] = Indices[14]
    hdrp1['C15'] = Indices[15]
    hdrp1['C16'] = Indices[16]
    hdrp1['C17'] = Indices[17]
    hdrp1['C18'] = Indices[18]
    hdrp1['C19'] = Indices[19]
    hdrp1['C20'] = Indices[20]
    hdrp1['C21'] = Indices[21]
    hdrp1['C22'] = Indices[22]
    hdrp1['C23'] = Indices[23]
    hdrp1['C24'] = Indices[24]
    hdrp1['C25'] = Indices[25]
    hdrp1['C26'] = Indices[26]
    hdrp1['C27'] = Indices[27]
    hdrp1['C28'] = Indices[28]
    hdrp1['C29'] = Indices[29]
    hdrp1['C30'] = Indices[30]
    hdrp1['C31'] = Indices[31]
    hdrp1['C32'] = Indices[32]
    hdrp1['C33'] = Indices[33]
    hdrp1['C34'] = Indices[34]
    hdrp1['C35'] = Indices[35]
    hdrp1['C36'] = Indices[36]
    hdrp1['C37'] = Indices[37]
    hdrp1['C38'] = Indices[38]
    hdrp1['C39'] = Indices[39]
    hdrp1['C40'] = Indices[40]
    hdrp1['C41'] = Indices[41]
    hdrp1['C42'] = Indices[42]
    
    hdrp2 = hdu[8].header
    hdrp2['PLATE'] = plate
    hdrp2['IFUDSGN'] = np.int(ifu)
    hdrp2['EXTNAME'] = 'COMBINED_DISP_MASK'
    hdrp2['CTYPE1'] = 'RA---TAN'
    hdrp2['CTYPE2'] = 'DEC--TAN'
    hdrp2['CDELT1'] = -0.000278888884
    hdrp2['CDELT2'] =  0.000278888884
    hdrp2['CRPIX1'] = 0
    hdrp2['CRPIX2'] = 0
    hdrp2['CUNIT1'] = 'deg'
    hdrp2['CUNIT2'] = 'deg'
    hdrp2['RADECSYS'] = 'FK5'
    hdrp2['CRVAL1'] = crval1
    hdrp2['CRVAL2'] = crval2
    hdrp2['C0'] = Indices[0]
    hdrp2['C1'] = Indices[1]
    hdrp2['C2'] = Indices[2]
    hdrp2['C3'] = Indices[3]
    hdrp2['C4'] = Indices[4]
    hdrp2['C5'] = Indices[5]
    hdrp2['C6'] = Indices[6]
    hdrp2['C7'] = Indices[7]
    hdrp2['C8'] = Indices[8]
    hdrp2['C9'] = Indices[9]
    hdrp2['C10'] = Indices[10]
    hdrp2['C11'] = Indices[11]
    hdrp2['C12'] = Indices[12]
    hdrp2['C13'] = Indices[13]
    hdrp2['C14'] = Indices[14]
    hdrp2['C15'] = Indices[15]
    hdrp2['C16'] = Indices[16]
    hdrp2['C17'] = Indices[17]
    hdrp2['C18'] = Indices[18]
    hdrp2['C19'] = Indices[19]
    hdrp2['C20'] = Indices[20]
    hdrp2['C21'] = Indices[21]
    hdrp2['C22'] = Indices[22]
    hdrp2['C23'] = Indices[23]
    hdrp2['C24'] = Indices[24]
    hdrp2['C25'] = Indices[25]
    hdrp2['C26'] = Indices[26]
    hdrp2['C27'] = Indices[27]
    hdrp2['C28'] = Indices[28]
    hdrp2['C29'] = Indices[29]
    hdrp2['C30'] = Indices[30]
    hdrp2['C31'] = Indices[31]
    hdrp2['C32'] = Indices[32]
    hdrp2['C33'] = Indices[33]
    hdrp2['C34'] = Indices[34]
    hdrp2['C35'] = Indices[35]
    hdrp2['C36'] = Indices[36]
    hdrp2['C37'] = Indices[37]
    hdrp2['C38'] = Indices[38]
    hdrp2['C39'] = Indices[39]
    hdrp2['C40'] = Indices[40]
    hdrp2['C41'] = Indices[41]
    hdrp2['C42'] = Indices[42]

    hdr6 = hdu[9].header
    hdr6['PLATE'] = plate
    hdr6['IFUDSGN'] = np.int(ifu)
    hdr6['EXTNAME'] = 'ELINE_FLUX'
    hdr6['CTYPE1'] = 'RA---TAN'
    hdr6['CTYPE2'] = 'DEC--TAN'
    hdr6['CDELT1'] = -0.000278888884
    hdr6['CDELT2'] =  0.000278888884
    hdr6['CRPIX1'] = 0
    hdr6['CRPIX2'] = 0
    hdr6['CUNIT1'] = 'deg'
    hdr6['CUNIT2'] = 'deg'
    hdr6['RADECSYS'] = 'FK5'
    hdr6['CRVAL1'] = crval1
    hdr6['CRVAL2'] = crval2
    hdr6['Unit'] = '10^(-17) erg/s/cm^2/arsec^2'
    hdr6['C0'] = 'OII-3727'
    hdr6['C1'] = 'OII-3729'
    hdr6['C2'] = 'Hthe-3798'
    hdr6['C3'] = 'Heta-3836'
    hdr6['C4'] = 'NeIII-3869'
    hdr6['C5'] = 'Hzet-3890'
    hdr6['C6'] = 'NeIII-3968'
    hdr6['C7'] = 'Heps-3971'
    hdr6['C8'] = 'Hdel-4102'
    hdr6['C9'] = 'Hgam-4341'
    hdr6['C10'] = 'HeII-4687'
    hdr6['C11'] = 'Hb-4862'
    hdr6['C12'] = 'OIII-4960'
    hdr6['C13'] = 'OIII-5008'
    hdr6['C14'] = 'HeI-5877'
    hdr6['C15'] = 'OI-6302'
    hdr6['C16'] = 'OI-6365'
    hdr6['C17'] = 'NII-6549'
    hdr6['C18'] = 'Ha-6564'
    hdr6['C19'] = 'NII-6585'
    hdr6['C20'] = 'SII-6718'
    hdr6['C21'] = 'SII-6732'
    
    hdr7 = hdu[10].header
    hdr7['PLATE'] = plate
    hdr7['IFUDSGN'] = np.int(ifu)
    hdr7['EXTNAME'] = 'ELINE_FLUX_SIGMA'
    hdr7['CTYPE1'] = 'RA---TAN'
    hdr7['CTYPE2'] = 'DEC--TAN'
    hdr7['CDELT1'] = -0.000278888884
    hdr7['CDELT2'] =  0.000278888884
    hdr7['CRPIX1'] = 0
    hdr7['CRPIX2'] = 0
    hdr7['CUNIT1'] = 'deg'
    hdr7['CUNIT2'] = 'deg'
    hdr7['RADECSYS'] = 'FK5'
    hdr7['CRVAL1'] = crval1
    hdr7['CRVAL2'] = crval2
    hdr7['Unit'] = '10^(-17) erg/s/cm^2/arsec^2'
    hdr7['C0'] = 'OII-3727'
    hdr7['C1'] = 'OII-3729'
    hdr7['C2'] = 'Hthe-3798'
    hdr7['C3'] = 'Heta-3836'
    hdr7['C4'] = 'NeIII-3869'
    hdr7['C5'] = 'Hzet-3890'
    hdr7['C6'] = 'NeIII-3968'
    hdr7['C7'] = 'Heps-3971'
    hdr7['C8'] = 'Hdel-4102'
    hdr7['C9'] = 'Hgam-4341'
    hdr7['C10'] = 'HeII-4687'
    hdr7['C11'] = 'Hb-4862'
    hdr7['C12'] = 'OIII-4960'
    hdr7['C13'] = 'OIII-5008'
    hdr7['C14'] = 'HeI-5877'
    hdr7['C15'] = 'OI-6302'
    hdr7['C16'] = 'OI-6365'
    hdr7['C17'] = 'NII-6549'
    hdr7['C18'] = 'Ha-6564'
    hdr7['C19'] = 'NII-6585'
    hdr7['C20'] = 'SII-6718'
    hdr7['C21'] = 'SII-6732'
    
    hdr8 = hdu[11].header
    hdr8['PLATE'] = plate
    hdr8['IFUDSGN'] = np.int(ifu)
    hdr8['EXTNAME'] = 'ELINE_FLUX_MASK'
    hdr8['CTYPE1'] = 'RA---TAN'
    hdr8['CTYPE2'] = 'DEC--TAN'
    hdr8['CDELT1'] = -0.000278888884
    hdr8['CDELT2'] =  0.000278888884
    hdr8['CRPIX1'] = 0
    hdr8['CRPIX2'] = 0
    hdr8['CUNIT1'] = 'deg'
    hdr8['CUNIT2'] = 'deg'
    hdr8['RADECSYS'] = 'FK5'
    hdr8['CRVAL1'] = crval1
    hdr8['CRVAL2'] = crval2
    hdr8['C0'] = 'OII-3727'
    hdr8['C1'] = 'OII-3729'
    hdr8['C2'] = 'Hthe-3798'
    hdr8['C3'] = 'Heta-3836'
    hdr8['C4'] = 'NeIII-3869'
    hdr8['C5'] = 'Hzet-3890'
    hdr8['C6'] = 'NeIII-3968'
    hdr8['C7'] = 'Heps-3971'
    hdr8['C8'] = 'Hdel-4102'
    hdr8['C9'] = 'Hgam-4341'
    hdr8['C10'] = 'HeII-4687'
    hdr8['C11'] = 'Hb-4862'
    hdr8['C12'] = 'OIII-4960'
    hdr8['C13'] = 'OIII-5008'
    hdr8['C14'] = 'HeI-5877'
    hdr8['C15'] = 'OI-6302'
    hdr8['C16'] = 'OI-6365'
    hdr8['C17'] = 'NII-6549'
    hdr8['C18'] = 'Ha-6564'
    hdr8['C19'] = 'NII-6585'
    hdr8['C20'] = 'SII-6718'
    hdr8['C21'] = 'SII-6732'

    hdr9 = hdu[12].header
    hdr9['PLATE'] = plate
    hdr9['IFUDSGN'] = np.int(ifu)
    hdr9['EXTNAME'] = 'ELINE_EW'
    hdr9['CTYPE1'] = 'RA---TAN'
    hdr9['CTYPE2'] = 'DEC--TAN'
    hdr9['CDELT1'] = -0.000278888884
    hdr9['CDELT2'] =  0.000278888884
    hdr9['CRPIX1'] = 0
    hdr9['CRPIX2'] = 0
    hdr9['CUNIT1'] = 'deg'
    hdr9['CUNIT2'] = 'deg'
    hdr9['RADECSYS'] = 'FK5'
    hdr9['CRVAL1'] = crval1
    hdr9['CRVAL2'] = crval2
    hdr9['Unit'] = 'A'
    hdr9['C0'] = 'OII-3727'
    hdr9['C1'] = 'OII-3729'
    hdr9['C2'] = 'Hthe-3798'
    hdr9['C3'] = 'Heta-3836'
    hdr9['C4'] = 'NeIII-3869'
    hdr9['C5'] = 'Hzet-3890'
    hdr9['C6'] = 'NeIII-3968'
    hdr9['C7'] = 'Heps-3971'
    hdr9['C8'] = 'Hdel-4102'
    hdr9['C9'] = 'Hgam-4341'
    hdr9['C10'] = 'HeII-4687'
    hdr9['C11'] = 'Hb-4862'
    hdr9['C12'] = 'OIII-4960'
    hdr9['C13'] = 'OIII-5008'
    hdr9['C14'] = 'HeI-5877'
    hdr9['C15'] = 'OI-6302'
    hdr9['C16'] = 'OI-6365'
    hdr9['C17'] = 'NII-6549'
    hdr9['C18'] = 'Ha-6564'
    hdr9['C19'] = 'NII-6585'
    hdr9['C20'] = 'SII-6718'
    hdr9['C21'] = 'SII-6732'

    hdr10 = hdu[13].header
    hdr10['PLATE'] = plate
    hdr10['IFUDSGN'] = np.int(ifu)
    hdr10['EXTNAME'] = 'ELINE_EW_SIGMA'
    hdr10['CTYPE1'] = 'RA---TAN'
    hdr10['CTYPE2'] = 'DEC--TAN'
    hdr10['CDELT1'] = -0.000278888884
    hdr10['CDELT2'] =  0.000278888884
    hdr10['CRPIX1'] = 0
    hdr10['CRPIX2'] = 0
    hdr10['CUNIT1'] = 'deg'
    hdr10['CUNIT2'] = 'deg'
    hdr10['RADECSYS'] = 'FK5'
    hdr10['CRVAL1'] = crval1
    hdr10['CRVAL2'] = crval2
    hdr10['Unit'] = 'A'
    hdr10['C0'] = 'OII-3727'
    hdr10['C1'] = 'OII-3729'
    hdr10['C2'] = 'Hthe-3798'
    hdr10['C3'] = 'Heta-3836'
    hdr10['C4'] = 'NeIII-3869'
    hdr10['C5'] = 'Hzet-3890'
    hdr10['C6'] = 'NeIII-3968'
    hdr10['C7'] = 'Heps-3971'
    hdr10['C8'] = 'Hdel-4102'
    hdr10['C9'] = 'Hgam-4341'
    hdr10['C10'] = 'HeII-4687'
    hdr10['C11'] = 'Hb-4862'
    hdr10['C12'] = 'OIII-4960'
    hdr10['C13'] = 'OIII-5008'
    hdr10['C14'] = 'HeI-5877'
    hdr10['C15'] = 'OI-6302'
    hdr10['C16'] = 'OI-6365'
    hdr10['C17'] = 'NII-6549'
    hdr10['C18'] = 'Ha-6564'
    hdr10['C19'] = 'NII-6585'
    hdr10['C20'] = 'SII-6718'
    hdr10['C21'] = 'SII-6732'
    
    hdr11 = hdu[14].header
    hdr11['PLATE'] = plate
    hdr11['IFUDSGN'] = np.int(ifu)
    hdr11['EXTNAME'] = 'ELINE_EW_MASK'
    hdr11['CTYPE1'] = 'RA---TAN'
    hdr11['CTYPE2'] = 'DEC--TAN'
    hdr11['CDELT1'] = -0.000278888884
    hdr11['CDELT2'] =  0.000278888884
    hdr11['CRPIX1'] = 0
    hdr11['CRPIX2'] = 0
    hdr11['CUNIT1'] = 'deg'
    hdr11['CUNIT2'] = 'deg'
    hdr11['RADECSYS'] = 'FK5'
    hdr11['CRVAL1'] = crval1
    hdr11['CRVAL2'] = crval2
    hdr11['C0'] = 'OII-3727'
    hdr11['C1'] = 'OII-3729'
    hdr11['C2'] = 'Hthe-3798'
    hdr11['C3'] = 'Heta-3836'
    hdr11['C4'] = 'NeIII-3869'
    hdr11['C5'] = 'Hzet-3890'
    hdr11['C6'] = 'NeIII-3968'
    hdr11['C7'] = 'Heps-3971'
    hdr11['C8'] = 'Hdel-4102'
    hdr11['C9'] = 'Hgam-4341'
    hdr11['C10'] = 'HeII-4687'
    hdr11['C11'] = 'Hb-4862'
    hdr11['C12'] = 'OIII-4960'
    hdr11['C13'] = 'OIII-5008'
    hdr11['C14'] = 'HeI-5877'
    hdr11['C15'] = 'OI-6302'
    hdr11['C16'] = 'OI-6365'
    hdr11['C17'] = 'NII-6549'
    hdr11['C18'] = 'Ha-6564'
    hdr11['C19'] = 'NII-6585'
    hdr11['C20'] = 'SII-6718'
    hdr11['C21'] = 'SII-6732'

    hdr12 = hdu[15].header
    hdr12['EXTNAME'] = 'SWIFT/SDSS'
    hdr12['CTYPE1'] = 'RA---TAN'
    hdr12['CTYPE2'] = 'DEC--TAN'
    hdr12['CDELT1'] = -0.000278888884
    hdr12['CDELT2'] =  0.000278888884
    hdr12['CRPIX1'] = 0
    hdr12['CRPIX2'] = 0
    hdr12['CUNIT1'] = 'deg'
    hdr12['CUNIT2'] = 'deg'
    hdr12['RADECSYS'] = 'FK5'
    hdr12['CRVAL1'] = crval1
    hdr12['CRVAL2'] = crval2
    hdr12['Unit'] = 'Nanomaggies'
    hdr12['C0'] = 'UVW2'
    hdr12['C1'] = 'UVW1'
    hdr12['C2'] = 'UVM2'
    hdr12['C3'] = 'SDSS u'
    hdr12['C4'] = 'SDSS g'
    hdr12['C5'] = 'SDSS r'
    hdr12['C6'] = 'SDSS i'
    hdr12['C7'] = 'SDSS z'

    hdr13 = hdu[16].header
    hdr13['EXTNAME'] = 'SWIFT/SDSS_SIGMA'
    hdr13['CTYPE1'] = 'RA---TAN'
    hdr13['CTYPE2'] = 'DEC--TAN'
    hdr13['CDELT1'] = -0.000278888884
    hdr13['CDELT2'] =  0.000278888884
    hdr13['CRPIX1'] = 0
    hdr13['CRPIX2'] = 0
    hdr13['CUNIT1'] = 'deg'
    hdr13['CUNIT2'] = 'deg'
    hdr13['RADECSYS'] = 'FK5'
    hdr13['CRVAL1'] = crval1
    hdr13['CRVAL2'] = crval2
    hdr13['Unit'] = 'Nanomaggies'
    hdr13['C0'] = 'UVW2'
    hdr13['C1'] = 'UVW1'
    hdr13['C2'] = 'UVM2'
    hdr13['C3'] = 'SDSS u'
    hdr13['C4'] = 'SDSS g'
    hdr13['C5'] = 'SDSS r'
    hdr13['C6'] = 'SDSS i'
    hdr13['C7'] = 'SDSS z'

    hdr14 = hdu[17].header
    hdr14['EXTNAME'] = 'SWIFT_UVOT'
    hdr14['CTYPE1'] = 'RA---TAN'
    hdr14['CTYPE2'] = 'DEC--TAN'
    hdr14['CDELT1'] = -0.000278888884
    hdr14['CDELT2'] =  0.000278888884
    hdr14['CRPIX1'] = 0
    hdr14['CRPIX2'] = 0
    hdr14['CUNIT1'] = 'deg'
    hdr14['CUNIT2'] = 'deg'
    hdr14['RADECSYS'] = 'FK5'
    hdr14['CRVAL1'] = crval1
    hdr14['CRVAL2'] = crval2
    hdr14['ABZP_W2'] = abz_W2
    hdr14['FLAMBDA_W2'] = convfact_W2
    hdr14['SKY_W2'] = skycnts_W2
    hdr14['ESKY_W2'] = eskycnts_W2
    hdr14['ABZP_W1'] = abz_W1
    hdr14['FLAMBDA_W1'] = convfact_W1
    hdr14['SKY_W1'] = skycnts_W1
    hdr14['ESKY_W1'] = eskycnts_W1
    hdr14['ABZP_M2'] = abz_M2
    hdr14['FLAMBDA_M2'] = convfact_M2
    hdr14['SKY_M2'] = skycnts_W2
    hdr14['ESKY_M2'] = eskycnts_M2
    hdr14['C0'] = 'UVW2 Counts(Not sky-subtracted)'
    hdr14['C1'] = 'UVW1 Counts(Not sky-subtracted)'
    hdr14['C2'] = 'UVM2 Counts(Not sky-subtracted)'
    hdr14['C3'] = 'UVW2 Counts Err'
    hdr14['C4'] = 'UVW1 Counts Err'
    hdr14['C5'] = 'UVM2 Counts Err'
    hdr14['C6'] = 'UVW2 Exposure'
    hdr14['C7'] = 'UVW1 Exposure'
    hdr14['C8'] = 'UVM2 Exposure'
    hdr14['C9'] = 'UVW2 Mask'
    hdr14['C10'] = 'UVW1 Mask'
    hdr14['C11'] = 'UVM2 Mask'

    hdu.writeto("/Volumes/Nikhil/Data/SWIFT/MPL-7_RP/SwiM_"+str(Line[q])+".fits",overwrite=True)

    



