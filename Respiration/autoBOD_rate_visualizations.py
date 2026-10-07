import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### This notebook is for making plots of respiration vs. key variables
    - see autoBOD_rate_correlation_analysis_bootstrapping.ipynb for robust code testing correlations with different data subsets and calculating uncertainty estimates
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import matplotlib.lines as mlines
    from itertools import chain
    import pandas as pd
    import xarray as xr
    import sys
    import os
    import glob
    import datetime
    from scipy import stats
    from sklearn.preprocessing import PowerTransformer
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.stats import pearsonr, spearmanr
    import gsw
    import matplotlib
    import cmocean
    import scipy
    from itertools import groupby
    from numpy import random
    import seaborn as sns
    from scipy.optimize import curve_fit
    from statsmodels.regression.linear_model import WLS
    from scipy.stats import rankdata as rd
    import pingouin as pg

    matplotlib.rcParams.update({'font.size': 15})
    return cmocean, glob, gsw, matplotlib, np, pd, plt, sns, xr


@app.cell
def _(sns):
    sns.set_context("notebook", font_scale=1.5, rc={"figure.figsize": (10, 7.5)})
    sns.set_theme('notebook', style='ticks')
    return


@app.cell
def _():
    std_cutoff=0.25 # this is the QC cutoff we use for rates that are too noisy! make sure to keep this consistent through all analysis!
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AutoBOD analyzed data
    """)
    return


@app.cell
def _(pd):
    # Location of log files from AutoBOD

    wkdir = '/Users/katyabbott/Documents/MIT_WHOI_Joint_Program/Research_Projects/Microbial_respiration_CALYPSO/'
    resp_dir = wkdir + 'Data/Respiration/'
    output_dir = wkdir + 'Analyses/Respiration/Output/'
    hydrg_dir = wkdir + 'Data/Hydrographic/'
    fcm_dir = wkdir + 'Data/Flow cytometry/'
    btl_dir = wkdir + 'Data/Bottle_data/'

    log_dir = resp_dir + 'logs/AutoBOD_logs/'

    resp_df_pooled = pd.read_csv(output_dir + 'CALYPSO_all_respiration_rates_QC_with_environmental_data_and_calibrated_Chl_pooled_060326.csv')
    resp_df_pooled = resp_df_pooled.drop('Unnamed: 0', axis=1)
    resp_df_pooled = resp_df_pooled.rename({'Pooled_mean_rate': 'Mean_rate', 'Pooled_sd': 'Std_rate'}, axis=1)
    return hydrg_dir, resp_df_pooled


@app.cell
def _(resp_df_pooled):
    ### Adding phase to dataframe for pooled data

    resp_df_pooled['Phase'] = ' '

    resp_df_pooled.loc[resp_df_pooled['Cast_number'] < 10, 'Phase'] = 'Phase 1'
    resp_df_pooled.loc[(resp_df_pooled['Cast_number'] > 9) & (resp_df_pooled['Cast_number'] < 36), 'Phase'] = 'Phase 2'
    resp_df_pooled.loc[(resp_df_pooled['Cast_number'] > 35), 'Phase'] = 'Phase 3'
    return


@app.cell
def _(glob, hydrg_dir):
    glob.glob(hydrg_dir + '*')
    return


@app.cell
def _(gsw, hydrg_dir, np, xr):
    ctd_ds = xr.open_dataset(hydrg_dir + 'L3_CTD_Calibrated_with_nitrate.nc')

    pressures = np.tile(ctd_ds.CTD_pres, ctd_ds.dims['n_casts']).reshape(ctd_ds.CTD_SA1.shape)

    lons = np.repeat(ctd_ds.CTD_lon, ctd_ds.dims['n_levels']).values.reshape(ctd_ds.CTD_SA1.shape)
    lats = np.repeat(ctd_ds.CTD_lat, ctd_ds.dims['n_levels']).values.reshape(ctd_ds.CTD_SA1.shape)

    o2_sol = gsw.O2sol(ctd_ds.CTD_SA1.values, ctd_ds.CTD_CT1.values, 
             pressures, lats, lons)

    o2_sol = o2_sol*((ctd_ds.CTD_Sigma01.values + 1000)/1000) #convert from umol/kg to umol/L


    AOU = o2_sol - o2_sol*ctd_ds['OXY_O2sat']/100

    ctd_ds['AOU'] = (('n_casts', 'n_levels'), AOU.data)

    ctd_ds['OXY_O2con_umol_L'] = o2_sol*ctd_ds['OXY_O2sat']/100
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## For these plots, we are using the following QC procedure:
    Keep all data except
    - [x] a) positive rates (oxygen production, not consumption) and
    - [x] b) rates with large standard error, greater than 0.25 — corresponds to a 10-point measurement standard deviation $\ge 1.5$ uM O$_2$, which is too noisy to be meaningful
    """)
    return


@app.cell
def _(resp_df_pooled):
    intr_pooled = resp_df_pooled[(resp_df_pooled['Intrusion_from_anomaly'] == 1)].copy()

    non_intr_pooled = resp_df_pooled[(resp_df_pooled['Feature_type'] == 'Aphotic_background') | (resp_df_pooled['Feature_type'] == 'Photic_zone')].copy()
    return intr_pooled, non_intr_pooled


@app.cell
def _(resp_df_pooled):
    resp_df_pooled.head()
    return


@app.cell
def _(np, plt, resp_df_pooled):
    plt.scatter(resp_df_pooled['Std_rate'], resp_df_pooled['DOXY_UMOL-KG'])

    np.corrcoef(resp_df_pooled['Std_rate'], resp_df_pooled['DOXY_UMOL-KG'])

    plt.show()
    # unsurprisingly, the uncertainty is higher as oxygen concentration increases, which makes sense
    return


@app.cell
def _(sns):
    sns.set_context("notebook", font_scale=1.65, rc={"figure.figsize": (10, 7.5)})
    return


@app.cell
def _(cmocean, matplotlib, np):
    #cmap_str='Purples' #cmocean.cm.deep
    _min_val, _max_val = (0.05, 0.9)
    _n = 10
    _orig_cmap = cmocean.cm.deep
    _colors_list = _orig_cmap(np.linspace(_min_val, _max_val, _n))  #plt.cm.Reds
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', _colors_list)
    return (cmap,)


@app.cell
def _():
    return


@app.cell
def _(cmap, intr_pooled, non_intr_pooled, plt, resp_df_pooled):
    _fig, ((_ax1, _ax2, _ax3), (_ax4, _ax5, _ax6)) = plt.subplots(2, 3, figsize=(15, 10), dpi=250)
    _vmin = 0
    _vmax = 200
    df_1 = resp_df_pooled.copy()
    intr_1 = intr_pooled.copy()
    non_intr_1 = non_intr_pooled.copy()
    _s_diamond = 150
    _s_star = 300
    _s_circ = 125
    _ax1.scatter(df_1[df_1['Cast_number'] == 1]['AOU'], -df_1[df_1['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_1[df_1['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', label='Cast 1', edgecolor='black')
    _ax1.scatter(intr_1['AOU'], -intr_1['Mean_rate'], s=_s_diamond, c=intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='Intrusion')
    _ax1.scatter(non_intr_1['AOU'], -non_intr_1['Mean_rate'], c=non_intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, s=_s_circ, marker='o', clip_on=False, linewidth=1.5, zorder=5, label='Background')
    _ax4.scatter(df_1[df_1['Cast_number'] == 1]['CHLA'], -df_1[df_1['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_1[df_1['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', label='Site 1', edgecolor='black')
    _ax4.scatter(intr_1['CHLA'], -intr_1['Mean_rate'], s=_s_diamond, c=intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='ACM')
    _ax4.scatter(non_intr_1['CHLA'], -non_intr_1['Mean_rate'], c=non_intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, s=_s_circ, marker='o', linewidth=1.5, zorder=5, label='Background')
    _ax5.scatter(df_1[df_1['Cast_number'] == 1]['Phototrophs_cells-mL'], -df_1[df_1['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_1[df_1['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', label='Site 1', edgecolor='black')
    _ax5.scatter(intr_1['Phototrophs_cells-mL'], -intr_1['Mean_rate'], s=_s_diamond, c=intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='ACM')
    _ax5.scatter(non_intr_1['Phototrophs_cells-mL'], -non_intr_1['Mean_rate'], c=non_intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, s=_s_circ, marker='o', linewidth=1.5, zorder=5, label='Background')
    _ax3.scatter(df_1[df_1['Cast_number'] == 1]['C (umol/L)'], -df_1[df_1['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_1[df_1['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', edgecolor='black')
    _ax3.scatter(intr_1['C (umol/L)'], -intr_1['Mean_rate'], s=_s_diamond, c=intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7)
    _ax3.scatter(non_intr_1['C (umol/L)'], -non_intr_1['Mean_rate'], c=non_intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, s=_s_circ, marker='o', linewidth=1.5, zorder=5)
    _ax2.scatter(df_1[df_1['Cast_number'] == 1]['NITRATE'], -df_1[df_1['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_1[df_1['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, zorder=6, marker='*', label='Site 1', edgecolor='black')
    _ax2.scatter(intr_1['NITRATE'], -intr_1['Mean_rate'], s=_s_diamond, c=intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='ACM')
    _ax2.scatter(non_intr_1['NITRATE'], -non_intr_1['Mean_rate'], c=non_intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, s=_s_circ, marker='o', clip_on=False, linewidth=1.5, zorder=5, label='Background')
    cp_1 = _ax6.scatter(df_1[df_1['Cast_number'] == 1]['Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'], -df_1[df_1['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_1[df_1['Cast_number'] == 1]['Depth'], clip_on=False, vmin=_vmin, vmax=_vmax, cmap=cmap, zorder=6, marker='*', label='Site 1', edgecolor='black')
    _ax6.scatter(intr_1['Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'], -intr_1['Mean_rate'], s=_s_diamond, c=intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='ACM')
    _ax6.scatter(non_intr_1['Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'], -non_intr_1['Mean_rate'], c=non_intr_1['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, s=_s_circ, marker='o', clip_on=False, linewidth=1.5, zorder=5, label='Background')
    _ax4.semilogx()
    _ax5.semilogx()
    _ax6.semilogx()
    _ax1.set_xlabel('$\\mu$mol kg$^{-1}$')
    _ax4.set_xlabel('$\\mu$g L$^{-1}$')
    _ax5.set_xlabel('cells mL$^{-1}$')
    _ax3.set_xlabel('$\\mu$mol L$^{-1}$')
    _ax2.set_xlabel('$\\mu$mol L$^{-1}$')
    _ax6.set_xlabel('cells mL$^{-1}$')
    _ax1.set_title('Apparent oxygen utilization')
    _ax4.set_title('Chl')
    _ax5.set_title('Picophytoplankton')
    _ax3.set_title('Particulate organic carbon')
    _ax2.set_title('Nitrate')
    _ax6.set_title('Non-photosynthetic bacteria')
    _ax1.set_ylabel('Respiration rate \n ($\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$)')
    _ax4.set_ylabel('Respiration rate \n ($\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$)')
    _ax1.legend(loc='upper right', fontsize=14, framealpha=1, facecolor='gainsboro')
    _labels = ['A', 'B', 'C', 'D', 'E', 'F']
    _i = 0
    for _ax in (_ax1, _ax2, _ax3, _ax4, _ax5, _ax6):
        _ax.grid(lw=0.5, c='gray', alpha=0.25)
        _ax.patch.set_edgecolor('black')
        _ax.patch.set_linewidth(1.5)
        _ax.text(-0.06, -0.15, _labels[_i], weight='bold', fontsize=24, transform=_ax.transAxes)
        _i = _i + 1
        _ax.set_yticks([0, 0.5, 1, 1.5])
        _ax.set_ylim(0, 1.5)
    for _ax in (_ax2, _ax3, _ax5, _ax6):
        _ax.set_yticklabels([])
    _fig.subplots_adjust(right=0.8)
    _cbar_ax = _fig.add_axes([0.83, 0.12, 0.02, 0.76])
    _fig.colorbar(cp_1, cax=_cbar_ax, label='Depth (m)', extend='max')
    _cbar_ax.invert_yaxis()
    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    plt.show()
    return


@app.cell
def _(cmap, intr_pooled, non_intr_pooled, plt, resp_df_pooled):
    _fig, ((ax7, _ax1, _ax2, _ax3), (ax8, _ax4, _ax5, _ax6)) = plt.subplots(2, 4, figsize=(20, 10), dpi=300)
    _vmin = 0
    _vmax = 200
    df_2 = resp_df_pooled.copy()
    intr_2 = intr_pooled.copy()
    non_intr_2 = non_intr_pooled.copy()
    _s_diamond = 150
    _s_star = 300
    _s_circ = 125
    _ax1.scatter(df_2[df_2['Cast_number'] == 1]['AOU'], -df_2[df_2['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_2[df_2['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', label='Cast 1', edgecolor='black')
    _ax1.scatter(intr_2['AOU'], -intr_2['Mean_rate'], s=_s_diamond, c=intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='Intrusion')
    _ax1.scatter(non_intr_2['AOU'], -non_intr_2['Mean_rate'], c=non_intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, s=_s_circ, marker='o', clip_on=False, linewidth=1.5, zorder=5, label='Background')
    _ax4.scatter(df_2[df_2['Cast_number'] == 1]['CHLA'], -df_2[df_2['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_2[df_2['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', label='Site 1', edgecolor='black')
    _ax4.scatter(intr_2['CHLA'], -intr_2['Mean_rate'], s=_s_diamond, c=intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='ACM')
    _ax4.scatter(non_intr_2['CHLA'], -non_intr_2['Mean_rate'], c=non_intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, s=_s_circ, marker='o', linewidth=1.5, zorder=5, label='Background')
    _ax5.scatter(df_2[df_2['Cast_number'] == 1]['Phototrophs_cells-mL'], -df_2[df_2['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_2[df_2['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', label='Site 1', edgecolor='black')
    _ax5.scatter(intr_2['Phototrophs_cells-mL'], -intr_2['Mean_rate'], s=_s_diamond, c=intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='ACM')
    _ax5.scatter(non_intr_2['Phototrophs_cells-mL'], -non_intr_2['Mean_rate'], c=non_intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, s=_s_circ, marker='o', linewidth=1.5, zorder=5, label='Background')
    _ax3.scatter(df_2[df_2['Cast_number'] == 1]['C (umol/L)'], -df_2[df_2['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_2[df_2['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', edgecolor='black')
    _ax3.scatter(intr_2['C (umol/L)'], -intr_2['Mean_rate'], s=_s_diamond, c=intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7)
    _ax3.scatter(non_intr_2['C (umol/L)'], -non_intr_2['Mean_rate'], c=non_intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, s=_s_circ, marker='o', linewidth=1.5, zorder=5)
    _ax2.scatter(df_2[df_2['Cast_number'] == 1]['NITRATE'], -df_2[df_2['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_2[df_2['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, zorder=6, marker='*', label='Site 1', edgecolor='black')
    _ax2.scatter(intr_2['NITRATE'], -intr_2['Mean_rate'], s=_s_diamond, c=intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='ACM')
    _ax2.scatter(non_intr_2['NITRATE'], -non_intr_2['Mean_rate'], c=non_intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, s=_s_circ, marker='o', clip_on=False, linewidth=1.5, zorder=5, label='Background')
    cp_2 = _ax6.scatter(df_2[df_2['Cast_number'] == 1]['Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'], -df_2[df_2['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_2[df_2['Cast_number'] == 1]['Depth'], clip_on=False, vmin=_vmin, vmax=_vmax, cmap=cmap, zorder=6, marker='*', label='Site 1', edgecolor='black')
    _ax6.scatter(intr_2['Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'], -intr_2['Mean_rate'], s=_s_diamond, c=intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='ACM')
    _ax6.scatter(non_intr_2['Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'], -non_intr_2['Mean_rate'], c=non_intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, s=_s_circ, marker='o', clip_on=False, linewidth=1.5, zorder=5, label='Background')
    ax7.scatter(df_2[df_2['Cast_number'] == 1]['SA'], -df_2[df_2['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_2[df_2['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', label='Cast 1', edgecolor='black')
    ax7.scatter(intr_2['SA'], -intr_2['Mean_rate'], s=_s_diamond, c=intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='Intrusion')
    ax7.scatter(non_intr_2['SA'], -non_intr_2['Mean_rate'], c=non_intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, s=_s_circ, marker='o', clip_on=False, linewidth=1.5, zorder=5, label='Background')
    ax8.scatter(df_2[df_2['Cast_number'] == 1]['CT'], -df_2[df_2['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_2[df_2['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', edgecolor='black')
    ax8.scatter(intr_2['CT'], -intr_2['Mean_rate'], s=_s_diamond, c=intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7)
    ax8.scatter(non_intr_2['CT'], -non_intr_2['Mean_rate'], c=non_intr_2['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, s=_s_circ, marker='o', linewidth=1.5, zorder=5)
    _ax4.semilogx()
    _ax5.semilogx()
    _ax6.semilogx()
    _ax1.set_xlabel('$\\mu$mol kg$^{-1}$')
    _ax4.set_xlabel('$\\mu$g L$^{-1}$')
    _ax5.set_xlabel('cells mL$^{-1}$')
    _ax3.set_xlabel('$\\mu$mol L$^{-1}$')
    _ax2.set_xlabel('$\\mu$mol L$^{-1}$')
    _ax6.set_xlabel('cells mL$^{-1}$')
    ax8.set_xlabel('$^\\circ$C')
    _ax1.set_title('Apparent oxygen utilization')
    _ax4.set_title('Chl')
    _ax5.set_title('Picophytoplankton')
    _ax3.set_title('Particulate organic carbon')
    _ax2.set_title('Nitrate')
    _ax6.set_title('Non-photosynthetic \x08acteria')
    ax7.set_title('Absolute Salinity (SA)')
    ax8.set_title('Conservative Temp (CT)')
    ax7.set_ylabel('Respiration rate \n ($\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$)')
    ax8.set_ylabel('Respiration rate \n ($\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$)')
    _ax1.legend(loc='upper right', fontsize=14, framealpha=1, facecolor='gainsboro')
    _labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    _i = 0
    for _ax in (ax7, _ax1, _ax2, _ax3, ax8, _ax4, _ax5, _ax6):
        _ax.grid(lw=0.5, c='gray', alpha=0.25)
        _ax.patch.set_edgecolor('black')
        _ax.patch.set_linewidth(1.5)
        _ax.text(-0.07, -0.19, _labels[_i], weight='bold', fontsize=24, transform=_ax.transAxes)
        _i = _i + 1
        _ax.set_yticks([0, 0.5, 1, 1.5])
        _ax.set_ylim(0, 1.5)
    for _ax in (_ax2, _ax3, _ax5, _ax6, _ax1, _ax4):
        _ax.set_yticklabels([])
    _fig.subplots_adjust(right=0.8)
    _cbar_ax = _fig.add_axes([0.83, 0.12, 0.02, 0.76])
    _fig.colorbar(cp_2, cax=_cbar_ax, label='Depth (m)', extend='max')
    _cbar_ax.invert_yaxis()
    plt.subplots_adjust(hspace=0.45, wspace=0.1)

    plt.show()
    return (cp_2,)


@app.cell
def _(cmap, cp_2, intr_pooled, non_intr_pooled, plt, resp_df_pooled):
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=250)
    _vmin = 0
    _vmax = 200
    df_3 = resp_df_pooled.copy()
    intr_3 = intr_pooled.copy()
    non_intr_3 = non_intr_pooled.copy()
    _s_diamond = 150
    _s_star = 300
    _s_circ = 125
    _ax1.scatter(df_3[df_3['Cast_number'] == 1]['SA'], -df_3[df_3['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_3[df_3['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', label='Cast 1', edgecolor='black')
    _ax1.scatter(intr_3['SA'], -intr_3['Mean_rate'], s=_s_diamond, c=intr_3['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='Intrusion')
    _ax1.scatter(non_intr_3['SA'], -non_intr_3['Mean_rate'], c=non_intr_3['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, s=_s_circ, marker='o', clip_on=False, linewidth=1.5, zorder=5, label='Background')
    _ax2.scatter(df_3[df_3['Cast_number'] == 1]['CT'], -df_3[df_3['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_3[df_3['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, zorder=6, marker='*', edgecolor='black')
    _ax2.scatter(intr_3['CT'], -intr_3['Mean_rate'], s=_s_diamond, c=intr_3['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7)
    _ax2.scatter(non_intr_3['CT'], -non_intr_3['Mean_rate'], c=non_intr_3['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, clip_on=False, s=_s_circ, marker='o', linewidth=1.5, zorder=5)
    _ax3.scatter(df_3[df_3['Cast_number'] == 1]['SIGMA0'], -df_3[df_3['Cast_number'] == 1]['Mean_rate'], s=_s_star, c=df_3[df_3['Cast_number'] == 1]['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, zorder=6, marker='*', label='Site 1', edgecolor='black')
    _ax3.scatter(intr_3['SIGMA0'], -intr_3['Mean_rate'], s=_s_diamond, c=intr_3['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, marker='d', clip_on=False, edgecolor='black', linewidth=1.5, zorder=7, label='ACM')
    _ax3.scatter(non_intr_3['SIGMA0'], -non_intr_3['Mean_rate'], c=non_intr_3['Depth'], vmin=_vmin, vmax=_vmax, cmap=cmap, s=_s_circ, marker='o', clip_on=False, linewidth=1.5, zorder=5, label='Background')
    _ax1.set_xlabel('')
    _ax2.set_xlabel('$^\\circ$C')
    _ax3.set_xlabel('kg m$^{-3}$')
    _ax1.set_title('Absolute Salinity')
    _ax2.set_title('Conservative Temperature')
    _ax3.set_title('Potential density anomaly')
    _ax1.set_ylabel('Respiration rate \n ($\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$)')
    _ax1.legend(loc='upper right', fontsize=14, framealpha=1, facecolor='gainsboro')
    _labels = ['A', 'B', 'C', 'D', 'E', 'F']
    _i = 0
    for _ax in (_ax1, _ax2, _ax3):
        _ax.grid(lw=0.5, c='gray', alpha=0.25)
        _ax.patch.set_edgecolor('black')
        _ax.patch.set_linewidth(1.5)
        _ax.text(-0.06, -0.15, _labels[_i], weight='bold', fontsize=24, transform=_ax.transAxes)
        _i = _i + 1
        _ax.set_yticks([0, 0.5, 1, 1.5])
        _ax.set_ylim(0, 1.5)
    for _ax in (_ax2, _ax3):
        _ax.set_yticklabels([])
    _fig.subplots_adjust(right=0.8)
    _cbar_ax = _fig.add_axes([0.83, 0.12, 0.02, 0.76])
    _fig.colorbar(cp_2, cax=_cbar_ax, label='Depth (m)', extend='max')
    _cbar_ax.invert_yaxis()
    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    plt.show()
    return


@app.cell
def _(resp_df_pooled):
    resp_df_pooled['CT'].max()
    return


@app.cell
def _(plt, resp_df_pooled):
    plt.hist(100*resp_df_pooled['Pro_cells-mL']/resp_df_pooled['Bacteria_cells-mL'])
    return


@app.cell
def _(matplotlib, np, plt):
    import matplotlib.colors as colors
    _min_val, _max_val = (0.2, 1.0)
    _n = 10
    _orig_cmap = plt.cm.viridis
    _orig_cmap = plt.cm.YlGn
    _colors_list = _orig_cmap(np.linspace(_min_val, _max_val, _n))
    cmap_1 = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', _colors_list)
    return (colors,)


@app.cell
def _(cmocean, colors, intr_pooled, non_intr_pooled, np, plt, resp_df_pooled):
    df_6 = resp_df_pooled.copy()
    intr_6 = intr_pooled.copy()
    non_intr_6 = non_intr_pooled.copy()
    lnorm = colors.LogNorm(vmin=0.02, vmax=1.0)
    _fig, _axes = plt.subplots(1, 2, figsize=(11.66, 8), dpi=300)
    _ax = _axes[0]
    _vmin = 28.75
    _vmax = 29.1
    _ax.scatter(-df_6[df_6['Cast_number'] == 1]['Mean_rate'], df_6[df_6['Cast_number'] == 1]['Depth'], label='Cast 1', marker='*', s=300, edgecolor='black', linewidth=1.3, c=df_6[df_6['Cast_number'] == 1]['SIGMA0'], vmin=_vmin, vmax=_vmax, cmap=cmocean.cm.dense, zorder=5)
    _ax.scatter(-non_intr_6['Mean_rate'], non_intr_6['Depth'], label='Background', s=150, c=non_intr_6['SIGMA0'], cmap=cmocean.cm.dense, vmin=_vmin, vmax=_vmax, zorder=4)
    cp_3 = _ax.scatter(-intr_6['Mean_rate'], intr_6['Depth'], label='Intrusion', edgecolor='black', s=150, c=intr_6['SIGMA0'], cmap=cmocean.cm.dense, vmin=_vmin, vmax=_vmax, marker='d', linewidth=1.5, zorder=6)
    _ax.set_ylim(550, -8)
    _ax.set_xlabel('Respiration rate ($\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$)', fontsize=18)
    _ax.set_ylabel('Depth (m)', fontsize=18)
    _ax.legend(fontsize=16, facecolor='gainsboro', framealpha=1)
    _ax.grid(lw=0.5, c='gray', alpha=0.25)
    _ax.patch.set_edgecolor('black')
    _ax.patch.set_linewidth(1)
    _btm = 260
    _scale_factor = 2

    def _forward(x):
        return np.array([a if a < _btm else a / _scale_factor + _btm / _scale_factor for a in x])

    def _inverse(x):
        return np.array([a if a < _btm else a * _scale_factor + _btm * _scale_factor for a in x])
    _ax.set_yscale('function', functions=(_forward, _inverse))
    _ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    _ax.yaxis.tick_right()
    _ax.yaxis.set_label_position('right')
    _ax.set_xticks([0, 0.5, 1, 1.5], labels=[0, 0.5, 1, 1.5])
    _cbar = _fig.colorbar(cp_3, ax=_ax, extend='max', label='Depth (m)', orientation='horizontal', pad=0.15)
    _cbar.set_label(label='$\\sigma_{\\theta}$ (kg m$^{-3}$)', size=18)
    _vmin = 0
    _vmax = 300
    _ax = _axes[1]
    _ax.scatter(-df_6[df_6['Cast_number'] == 1]['Mean_rate'], df_6[df_6['Cast_number'] == 1]['SIGMA0'], label='Cast 1', marker='*', s=300, edgecolor='black', linewidth=1.3, c=df_6[df_6['Cast_number'] == 1]['Depth'], cmap=cmocean.cm.deep, vmin=_vmin, vmax=_vmax, zorder=5)
    _ax.scatter(-non_intr_6['Mean_rate'], non_intr_6['SIGMA0'], label='Background', s=150, c=non_intr_6['Depth'], cmap=cmocean.cm.deep, vmin=_vmin, vmax=_vmax, linewidth=1.5, zorder=4)
    cp_3 = _ax.scatter(-intr_6['Mean_rate'], intr_6['SIGMA0'], label='Aphotic Chl max (ACM)', edgecolor='black', s=150, c=intr_6['Depth'], cmap=cmocean.cm.deep, vmin=_vmin, vmax=_vmax, linewidth=1.5, marker='d', zorder=6)
    _cbar = plt.colorbar(cp_3, ax=_ax, extend='max', label='Depth (m)', orientation='horizontal', pad=0.15)
    _cbar.set_label(label='Depth (m)', size=18)
    _ax.set_xlabel('Respiration rate ($\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$)', fontsize=18)
    _ax.set_ylabel('Potential density anomaly (kg m$^{-3}$)', fontsize=18)
    _ax.set_xticks([0, 0.5, 1, 1.5], labels=[0, 0.5, 1, 1.5])
    _ax.set_ylim(29.11, 28.72)
    _ax.grid(lw=0.5, c='gray', alpha=0.25)
    _ax.patch.set_edgecolor('black')
    _ax.patch.set_linewidth(1)
    plt.subplots_adjust(wspace=0.65)
    _labels = ['B', 'C']
    for _i, _ax in enumerate(_axes):
        _ax.text(-0.1, -0.1, _labels[_i], weight='bold', fontsize=24, transform=_ax.transAxes)

    plt.show()
    return


@app.cell
def _(intr_pooled, non_intr_pooled, plt):
    intr_13 = intr_pooled.copy()
    non_intr_13 = non_intr_pooled.copy()
    _fig, _ax = plt.subplots(1, figsize=(2.475, 2.7), dpi=300)
    _bplot1 = plt.boxplot(-intr_13[(intr_13['Depth'] >= 50) & (intr_13['Depth'] <= 100)]['Mean_rate'], positions=[11.75], patch_artist=True, widths=1.1)
    _bplot2 = plt.boxplot(-non_intr_13[(non_intr_13['Depth'] >= 50) & (non_intr_13['Depth'] <= 100)]['Mean_rate'], positions=[10.5], patch_artist=True, widths=1.1)
    _bplot3 = plt.boxplot(-intr_13[(intr_13['Depth'] > 100) & (intr_13['Depth'] <= 150)]['Mean_rate'], positions=[8.25], patch_artist=True, widths=1.1)
    _bplot4 = plt.boxplot(-non_intr_13[(non_intr_13['Depth'] >= 100) & (non_intr_13['Depth'] <= 150)]['Mean_rate'], positions=[7], patch_artist=True, widths=1.1)
    _bplot5 = plt.boxplot(-intr_13[(intr_13['Depth'] > 150) & (intr_13['Depth'] <= 200)]['Mean_rate'], positions=[4.75], patch_artist=True, widths=1.1)
    _bplot6 = plt.boxplot(-non_intr_13[(non_intr_13['Depth'] > 150) & (non_intr_13['Depth'] <= 200)]['Mean_rate'], positions=[3], patch_artist=True, widths=1.1)
    _bplot7 = plt.boxplot(-intr_13[(intr_13['Depth'] > 200) & (intr_13['Depth'] <= 250)]['Mean_rate'], positions=[1.25], patch_artist=True, widths=1.1)
    _bplot8 = plt.boxplot(-non_intr_13[(non_intr_13['Depth'] > 200) & (non_intr_13['Depth'] <= 250)]['Mean_rate'], positions=[0], patch_artist=True, widths=1.1)
    _bplot1['boxes'][0].set_facecolor('orange')
    _bplot1['medians'][0].set_color('black')
    _bplot2['boxes'][0].set_facecolor('slateblue')
    _bplot2['medians'][0].set_color('black')
    _bplot3['boxes'][0].set_facecolor('orange')
    _bplot3['medians'][0].set_color('black')
    _bplot4['boxes'][0].set_facecolor('slateblue')
    _bplot4['medians'][0].set_color('black')
    _bplot5['boxes'][0].set_facecolor('orange')
    _bplot5['medians'][0].set_color('black')
    _bplot6['boxes'][0].set_facecolor('slateblue')
    _bplot6['medians'][0].set_color('black')
    _bplot7['boxes'][0].set_facecolor('orange')
    _bplot7['medians'][0].set_color('black')
    _bplot8['boxes'][0].set_facecolor('slateblue')
    _bplot8['medians'][0].set_color('black')
    _ax.set_xticklabels(['', '', '', '', '', '', '', ''])
    _ax.set_yticks([0.0, 0.5, 1.0, 1.5], [0.0, 0.5, 1.0, 1.5], fontsize=16)
    _ax.set_xlim(-0.75, 12.5)
    _ax.set_xlim(12.5, -0.75)
    _ax.spines[['right', 'top']].set_visible(True)
    _ax.set_ylabel('Resp. rate', fontsize=16)
    plt.axvline(2.35, lw=0.75, c='grey')
    plt.axvline(5.9, lw=0.75, c='grey')
    plt.axvline(9.3, lw=0.75, c='grey')
    plt.grid(alpha=0.5, lw=0.25)

    plt.show()
    return intr_13, non_intr_13


@app.cell
def _(intr_13):
    -intr_13[(intr_13['Depth'] > 50) & (intr_13['Depth'] <= 100)]['Mean_rate'].median()
    return


@app.cell
def _(intr_13):
    -intr_13[(intr_13['Depth'] > 100) & (intr_13['Depth'] <= 150)]['Mean_rate'].median()
    return


@app.cell
def _(non_intr_13):
    -non_intr_13[(non_intr_13['Depth'] > 100) & (non_intr_13['Depth'] <= 150)]['Mean_rate'].median()
    return


@app.cell
def _(intr_pooled, non_intr_pooled, plt):
    intr_15 = intr_pooled.copy()
    non_intr_15 = non_intr_pooled.copy()
    _fig, _ax = plt.subplots(1, figsize=(3.375, 3), dpi=300)
    _bplot1 = plt.boxplot(-intr_15[(intr_15['Depth'] > 50) & (intr_15['Depth'] <= 100)]['Mean_rate'], positions=[0], patch_artist=True, widths=1)
    _bplot2 = plt.boxplot(-non_intr_15[(non_intr_15['Depth'] > 50) & (non_intr_15['Depth'] <= 100)]['Mean_rate'], positions=[1.25], patch_artist=True, widths=1)
    _bplot3 = plt.boxplot(-intr_15[(intr_15['Depth'] > 100) & (intr_15['Depth'] <= 150)]['Mean_rate'], positions=[3.5], patch_artist=True, widths=1)
    _bplot4 = plt.boxplot(-non_intr_15[(non_intr_15['Depth'] > 100) & (non_intr_15['Depth'] <= 150)]['Mean_rate'], positions=[4.75], patch_artist=True, widths=1)
    _bplot5 = plt.boxplot(-intr_15[(intr_15['Depth'] > 150) & (intr_15['Depth'] <= 200)]['Mean_rate'], positions=[7], patch_artist=True, widths=1)
    _bplot6 = plt.boxplot(-non_intr_15[(non_intr_15['Depth'] > 150) & (non_intr_15['Depth'] <= 200)]['Mean_rate'], positions=[8.25], patch_artist=True, widths=1)
    _bplot7 = plt.boxplot(-intr_15[(intr_15['Depth'] > 200) & (intr_15['Depth'] <= 250)]['Mean_rate'], positions=[10.5], patch_artist=True, widths=1)
    _bplot8 = plt.boxplot(-non_intr_15[(non_intr_15['Depth'] > 200) & (non_intr_15['Depth'] <= 250)]['Mean_rate'], positions=[11.75], patch_artist=True, widths=1)
    _bplot1['boxes'][0].set_facecolor('limegreen')
    _bplot1['medians'][0].set_color('black')
    _bplot2['boxes'][0].set_facecolor('darkblue')
    _bplot2['medians'][0].set_color('black')
    _bplot3['boxes'][0].set_facecolor('limegreen')
    _bplot3['medians'][0].set_color('black')
    _bplot4['boxes'][0].set_facecolor('darkblue')
    _bplot4['medians'][0].set_color('black')
    _bplot5['boxes'][0].set_facecolor('limegreen')
    _bplot5['medians'][0].set_color('black')
    _bplot6['boxes'][0].set_facecolor('darkblue')
    _bplot6['medians'][0].set_color('black')
    _bplot7['boxes'][0].set_facecolor('limegreen')
    _bplot7['medians'][0].set_color('black')
    _bplot8['boxes'][0].set_facecolor('darkblue')
    _bplot8['medians'][0].set_color('black')
    _ax.set_xticklabels(['', '', '', '', '', '', '', ''])
    _ax.set_xlim(-0.75, 12.5)
    _ax.set_xlim(12.5, -0.75)
    _ax.spines[['right', 'top']].set_visible(False)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Binning data according to depths to accompany deep convection analysis of phototrophs and bacteria
    """)
    return


@app.cell
def _(resp_df_pooled):
    resp_df_noc1 = resp_df_pooled[resp_df_pooled['Cast_number'] != 1]
    resp_df_c1 = resp_df_pooled[resp_df_pooled['Cast_number'] == 1]
    return resp_df_c1, resp_df_noc1


@app.cell
def _(np, resp_df_c1):
    discrete_colors_bright = ['#5E79FD', '#FFA400', '#FF3399']
    colors_1 = np.repeat('lightgray', 4)
    depth_bins = [[0, 25], [25, 50], [50, 100], [100, 150], [150, 200], [200, 300], [300, 550]]
    x1 = [(len(depth_bins) - _i) / 4 for _i in range(len(depth_bins))]
    x2 = np.roll(x1, shift=1)
    bounds = [len(depth_bins) / 4 + 0.15] + list(((x1 + x2) / 2)[1:]) + [0.1]
    _labels = resp_df_c1.Depth.values
    new_labels = []
    for label in _labels:
        for _i, depth_bin in enumerate(depth_bins):
            if (label > depth_bin[0]) & (label <= depth_bin[1]):
                i_db = _i
                db = depth_bin
        bb = bounds[i_db:i_db + 2]
        new_labels.append(-(label - db[0]) / (db[1] - db[0]) * 0.15 + bb[0])
    return depth_bins, new_labels


@app.cell
def _(depth_bins, new_labels, plt, resp_df_c1, resp_df_noc1):
    _fig, _ax = plt.subplots(1, figsize=(5, 8 / 3), dpi=300)
    for _i, dd in enumerate(depth_bins):
        _temp_df = resp_df_noc1[resp_df_noc1['Depth'].isin(range(dd[0], dd[1]))].copy()
        bplot = _ax.boxplot(-_temp_df['Mean_rate'].values, vert=False, positions=[(len(depth_bins) - _i) / 4], patch_artist=True)
        bplot['boxes'][0].set_facecolor('lightgray')
        bplot['medians'][0].set_color('black')
    _ax.set_ylim(0.1, len(depth_bins) / 4 + 0.15)
    _ax.set_yticklabels(['0 - 25 m', '25 - 50 m', '50 - 100 m', ' 100 - 150 m', '150 - 200 m', '200 - 300 m', '300 - 550 m'])
    plt.xlabel('$\\mu$M O$_2$/day')
    #ax.set_xscale('log')
    plt.title('Respiration')
    #ax.set_xlim(3e4, 2e6)
    vals = -resp_df_c1['Mean_rate'].astype('float').values
    _ax.scatter(vals, new_labels, color='red', s=100, marker='*', zorder=10, clip_on=False)
    if _i == 0:
        _ax.scatter([], [], color='red', s=100, marker='*', zorder=10, label='Cast 1')
    #ax.set_title(phototroph_names[phototroph])
    _ax.grid(lw=0.5, c='gray', alpha=0.25)
    _ax.patch.set_edgecolor('black')
    #plt.savefig('figures/Respiration_Cast1_box_plots.png', dpi=300, bbox_inches='tight')
    _ax.patch.set_linewidth(1)

    plt.show()
    return


if __name__ == "__main__":
    app.run()
