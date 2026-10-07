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
    ### Looking at specific hypotheses about how respiration should relate to carbon, oxygen and bacteria

    Cleaned up 12/1/25 to contain only analysis going directly into manuscript
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
    from openpyxl import load_workbook
    import glob
    import datetime
    from scipy import stats
    from sklearn.preprocessing import PowerTransformer
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
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    import scipy.odr
    from statistics import geometric_mean
    from scipy.stats import rankdata as rd
    import pingouin as pg
    matplotlib.rcParams.update({'font.size': 15})
    return (
        curve_fit,
        geometric_mean,
        matplotlib,
        np,
        pd,
        pearsonr,
        plt,
        r2_score,
        scipy,
        spearmanr,
    )


@app.cell
def _():
    std_cutoff=0.25 # this is the QC cutoff we use for rates that are too noisy! make sure to keep this consistent through all analysis!
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AutoBOD analysis of data
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
    # Location of saved unpooled rates dataset
    #resp_df_unpooled = pd.read_csv(output_dir + 'CALYPSO_all_respiration_rates_QC_with_environmental_data_unpooled_060326.csv')

    #resp_ds = xr.open_dataset(output_dir + 'autoBOD_processed_oxygen.nc')
    # resp_df = pd.read_csv(output_dir + 'autoBOD_processed_rates_011725.csv')
    #resp_df_unpooled = resp_df_unpooled.drop('Unnamed: 0', axis=1)


    resp_df_pooled = pd.read_csv(output_dir + 'CALYPSO_all_respiration_rates_QC_with_environmental_data_and_calibrated_Chl_pooled_060326.csv')
    resp_df_pooled = resp_df_pooled.drop('Unnamed: 0', axis=1)
    resp_df_pooled = resp_df_pooled.rename({'Pooled_mean_rate': 'Mean_rate', 'Pooled_sd': 'Std_rate'}, axis=1)
    return (resp_df_pooled,)


@app.cell
def _(np):
    def calculate_weights(df):
        weights = []
        for i, row in df.iterrows():
            n = df[df['Bottle'] == row['Bottle']].shape[0]
            weights.append(1/n)

        return np.array(weights)

    return


@app.function
# def subset_df_by_type(df, euphotic_depth=55):
#     non_intr = df.copy()
#     non_intr = non_intr[non_intr['Cast_number'] != 1] 
#     #non_intr = non_intr[non_intr['Cast'] != 'C6'] 
#     #non_intr = non_intr[non_intr['Cast'] != 'C7'] 
#     non_intr = non_intr[(non_intr['Depth'] >= euphotic_depth) & (non_intr['Intrusion'] == 'N')]

#     intr = df.copy()#[rates_df_filtered['Intrusion'] == 'N'].copy()
#     intr = intr[intr['Cast_number'] != 1] 
#     intr = intr[intr['Intrusion'] == 'Y']

#     intr_euphotic = df.copy()#[rates_df_filtered['Intrusion'] == 'N'].copy()
#     intr_euphotic = intr_euphotic[intr_euphotic['Cast_number'] != 1] 
#     intr_euphotic = intr_euphotic[(intr_euphotic['Depth'] < euphotic_depth) | (intr_euphotic['Intrusion'] == 'Y')]

#     return non_intr, intr, intr_euphotic

def subset_df_by_type(df, threshold='ACA_05'):
    if threshold == 'ACA_025':
        non_intr = df[df['Feature_type'] == 'Aphotic_background'].copy()
        intr = df[(df['Feature_type'] == 'ACA_05') | (df['Feature_type'] == 'ACA_025')].copy()
        intr_euphotic = df[(df['Feature_type'] == 'ACA_05') | (df['Feature_type'] == 'Photic_zone')| (df['Feature_type'] == 'ACA_025')].copy()
    elif threshold == 'ACA_05':
        non_intr = df[(df['Feature_type'] == 'Aphotic_background') | (df['Feature_type'] == 'ACA_025')].copy()
        intr = df[df['Feature_type'] == 'ACA_05'].copy()
        intr_euphotic = df[(df['Feature_type'] == 'ACA_05') | (df['Feature_type'] == 'Photic_zone')].copy()

    return non_intr, intr, intr_euphotic


@app.cell
def _(np, resp_df_pooled):
    len(np.unique(resp_df_pooled['Cast_number']))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pooled analysis
    """)
    return


@app.cell
def _(resp_df_pooled):
    ### Want to repeat this with pooled data for sanity check

    non_intr_p, intr_p, intr_euphotic_p = subset_df_by_type(resp_df_pooled, threshold='ACA_025')
    return (intr_euphotic_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### We can use two different bacterial population estimates:
    - 1 is just the T0 estimate, and 2 is the T0-T48 averaged value

    They don't give significantly different results. There are a handful of casts where we observe phototroph growth in the first 12 hours and then it levels off... in at least one of these, bacterial cell counts might be underestimated by T0 (but it's not the same as bottle effects driving unchecked growth) - so using 2. for now

    Note that I've also subtracted out Pro cell counts from bacterial cell counts
    """)
    return


@app.cell
def _():
    bact_conversion = 20 #fg C/cell, from literature values cited in Freilich et al. (2024)
    pro_conversion = 39 #fg C/cell, from literature values cited in Freilich et al. (2024)
    syn_conversion = 82 #fg C/cell, from literature values cited in Freilich et al. (2024)
    euk_conversion = 530 #fg C/cell, from literature values cited in Freilich et al. (2024)

    # 1 fg = 1 femtogram is 10^-15 grams or 1e-9 ug

    #fcm_df['Microbial_carbon_ug/L'] = 1000*1e-9*(fcm_df['Bacteria_carbon'] + fcm_df['Pro_carbon'] + fcm_df['Syn_carbon'] + fcm_df['Euks_carbon'])
    return bact_conversion, euk_conversion, pro_conversion, syn_conversion


@app.cell
def _():
    #bacteria_key = 'Bacteria_cells-mL'
    #carbon_key = 'Nonbacterial_carbon_ug/L_from_T0'
    bacteria_key = 'Average_bacteria_cells-mL_over_48_hrs'
    bacteria_key = 'Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'
    carbon_key = 'Nonbacterial_carbon_ug/L_averaged_48_hours'
    #carbon_key = 'C (ug/L)'

    #bacteria_key = 'Microbial_cells'
    return bacteria_key, carbon_key


@app.cell
def _(
    bact_conversion,
    bacteria_key,
    euk_conversion,
    pro_conversion,
    resp_df_pooled,
    syn_conversion,
):
    resp_df_pooled['Bacteria_carbon_averaged_48_hours'] = resp_df_pooled['Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro']*bact_conversion
    resp_df_pooled['Bacteria_carbon_from_T0'] = resp_df_pooled['Bacteria_cells-mL_subtr_pro']*bact_conversion

    resp_df_pooled['Pro_carbon'] = resp_df_pooled['Pro_cells-mL']*pro_conversion
    resp_df_pooled['Syn_carbon'] = resp_df_pooled['Syn_cells-mL']*syn_conversion
    resp_df_pooled['Euks_carbon'] = resp_df_pooled['Euks_cells-mL']*euk_conversion

    # resp_df_pooled['Microbial_carbon_ug/L'] = 1000*1e-9*(resp_df_pooled['Bacteria_carbon'] + resp_df_pooled['Pro_carbon'] + resp_df_pooled['Syn_carbon'] + resp_df_pooled['Euks_carbon'])
    # resp_df_pooled['Photosynthetic_carbon_ug/L'] = 1000*1e-9*(resp_df_pooled['Pro_carbon'] + resp_df_pooled['Syn_carbon'] + resp_df_pooled['Euks_carbon'])

    #resp_df_pooled['Nonmicrobial_carbon_ug/L_'] = resp_df_pooled['C (ug/L)'] - resp_df_pooled['Microbial_carbon_ug/L'] 
    resp_df_pooled['Nonbacterial_carbon_ug/L_from_T0'] = resp_df_pooled['C (ug/L)'] - resp_df_pooled['Bacteria_carbon_from_T0']*1000*1e-9
    resp_df_pooled['Nonbacterial_carbon_ug/L_averaged_48_hours'] = resp_df_pooled['C (ug/L)'] - resp_df_pooled['Bacteria_carbon_averaged_48_hours']*1000*1e-9

    df_p_all = resp_df_pooled.dropna(subset=['C (umol/L)', bacteria_key]).copy()

    df_p_all['Microbial_cells'] = df_p_all['Bacteria_cells-mL_subtr_pro'] + df_p_all['Phototrophs_cells-mL']
    return (df_p_all,)


@app.cell
def _(
    bact_conversion,
    bacteria_key,
    euk_conversion,
    intr_euphotic_p,
    pro_conversion,
    syn_conversion,
):
    intr_euphotic_p['Bacteria_carbon_averaged_48_hours'] = intr_euphotic_p['Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro']*bact_conversion
    intr_euphotic_p['Bacteria_carbon_from_T0'] = intr_euphotic_p['Bacteria_cells-mL_subtr_pro']*bact_conversion
    intr_euphotic_p['Pro_carbon'] = intr_euphotic_p['Pro_cells-mL']*pro_conversion
    intr_euphotic_p['Syn_carbon'] = intr_euphotic_p['Syn_cells-mL']*syn_conversion
    intr_euphotic_p['Euks_carbon'] = intr_euphotic_p['Euks_cells-mL']*euk_conversion

    #intr_euphotic_p['Microbial_carbon_ug/L'] = 1000*1e-9*(intr_euphotic_p['Bacteria_carbon'] + intr_euphotic_p['Pro_carbon'] + intr_euphotic_p['Syn_carbon'] + intr_euphotic_p['Euks_carbon'])
    #intr_euphotic_p['Photosynthetic_carbon_ug/L'] = 1000*1e-9*(intr_euphotic_p['Pro_carbon'] + intr_euphotic_p['Syn_carbon'] + intr_euphotic_p['Euks_carbon'])

    #intr_euphotic_p['Nonmicrobial_carbon_ug/L'] = intr_euphotic_p['C (ug/L)'] - intr_euphotic_p['Microbial_carbon_ug/L'] 
    intr_euphotic_p['Nonbacterial_carbon_ug/L_from_T0'] = intr_euphotic_p['C (ug/L)'] - intr_euphotic_p['Bacteria_carbon_from_T0']*1000*1e-9
    intr_euphotic_p['Nonbacterial_carbon_ug/L_averaged_48_hours'] = intr_euphotic_p['C (ug/L)'] - intr_euphotic_p['Bacteria_carbon_averaged_48_hours']*1000*1e-9

    df_p = intr_euphotic_p.dropna(subset=['C (umol/L)', bacteria_key]).copy()

    df_p['Microbial_cells'] = df_p['Bacteria_cells-mL_subtr_pro'] + df_p['Phototrophs_cells-mL']
    return (df_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Organization of this analysis

    We want to look at 2 different scenarios:
    1. An organic carbon-limited respiration rate that behaves like Michaelis-Menten
    2. An organic carbon-limited respiration rate that behaves like Michaelis-Menten and is stimulated by a higher Chl:C ratio
    """)
    return


@app.cell
def _(carbon_key):
    print(carbon_key)
    bacteria_key_1 = 'Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'
    return (bacteria_key_1,)


@app.cell
def _(bacteria_key_1, carbon_key, curve_fit, df_p, np, plt):
    ### First case
    def func(x, C1, k):
        return C1 * x / (x + k)
    monod_df = df_p.copy()
    xdata = monod_df[carbon_key]
    ydata = -monod_df['Mean_rate'] / monod_df[bacteria_key_1]
    cdata = monod_df['CHLA']
    popt, pcov = curve_fit(func, xdata, ydata)
    print(popt)
    pred_resp_per_cell = popt[0] * xdata / (xdata + popt[1])  #)/df_p['C (ug/L)']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    ax = axes[0]
    ax.scatter(-monod_df['Mean_rate'], pred_resp_per_cell * monod_df[bacteria_key_1], s=100, color='cornflowerblue')
    ax.plot(np.arange(0, 1.5, 0.1), np.arange(0, 1.5, 0.1), color='black', linestyle='dashed', linewidth=2, zorder=-1)
    ax.set_xlabel('True respiration rate', fontsize=12)
    ax.set_ylabel('Predicted respiration rate', fontsize=12)
    ax.grid(alpha=0.5)
    ax = axes[1]
    ax.scatter(ydata * 1000000.0, pred_resp_per_cell * 1000000.0, s=100, color='cornflowerblue')
    ax.plot(np.arange(0.2, 1.6, 0.1), np.arange(0.2, 1.6, 0.1), color='black', linestyle='dashed', linewidth=2, zorder=-1)
    ax.set_xlabel('True per-cell respiration rate (pmol O$_2$ cell$^{-1}$ day$^{-1}$)', fontsize=12)
    ax.set_ylabel('Predicted per-cell respiration rate (pmol O$_2$ cell$^{-1}$ day$^{-1}$)', fontsize=12)
    ax.grid(alpha=0.5)
    axes[0].set_title('Predicted vs. actual \n respiration rate', fontsize=12)
    axes[1].set_title('Predicted vs.actual cell-specific \n respiration rate', fontsize=12)
    axes[0].set_xlim(0, 1.5)
    axes[0].set_ylim(0, 1.5)
    axes[1].set_xlim(0.2, 1.4)
    #ax.text(1.1,0.95, f'C={round(popt[0],8)}')
    #ax.text(1.1,0.85, r'K$_m$='+f'{round(popt[1],3)} ug/L')
    axes[1].set_ylim(0.2, 1.4)
    plt.subplots_adjust(wspace=0.4)
    return cdata, func, monod_df, pcov, popt, pred_resp_per_cell, xdata, ydata


@app.cell
def _(bacteria_key_1, carbon_key, df_p, df_p_all, func, np, plt, popt):
    plt.figure(dpi=150)
    plt.grid()
    plt.scatter(-df_p_all['Mean_rate'] / df_p_all[bacteria_key_1] * 1000000.0, 1000000.0 * func(df_p_all[carbon_key].values.T, popt[0], popt[1]), c='orange', zorder=5, label='Aphotic background')
    plt.scatter(-df_p['Mean_rate'] / df_p[bacteria_key_1] * 1000000.0, 1000000.0 * func(df_p[carbon_key].values.T, popt[0], popt[1]), c='orange', edgecolor='black', zorder=6, label='Intrusions + euphotic')
    plt.plot(np.arange(0.3, 1.2, 0.1), np.arange(0.3, 1.2, 0.1), c='black', zorder=2, label='1-1 line')
    plt.legend(fontsize=10, facecolor='gainsboro', framealpha=1)
    plt.xlabel('Measured cell-specific respiration rate \n (pmol O$_2$/cell/day)')
    plt.ylabel('Predicted cell-specific \n respiration rate \n (pmol O$_2$/cell/day)')
    #plt.xlim(0, 2e-6)
    plt.title('Traditional Monod')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculating error statistics for paper

    quick unit check

    respiration rate is in umol/L/day - so 1e-6 mol/L/day
    bacterial cell numbers are cells/mL - so 1e3 cells/L

    respiration rate/cell count -> 1e-9 mol/cell/day or 1e6 fmol/cell/day where a femtomole is 1e-15 moles

    so, we want to multiply our results by 1e6 to get fmol/cell/day
    """)
    return


@app.cell
def _(np, pcov):
    print(np.linalg.cond(pcov))

    print(np.diag(pcov)) # these are pretty different values - indicates the solution is not super well constrained, which isn't that surprising to me given the lack of data
    return


@app.cell
def _(bacteria_key_1, monod_df, np, pred_resp_per_cell, ydata):
    from sklearn.metrics import mean_squared_error
    rms = np.sqrt(mean_squared_error(ydata, pred_resp_per_cell))#, squared=True))
    print('Error of per-cell rate', rms / np.nanmean(ydata))
    rms = np.sqrt(mean_squared_error(-monod_df['Mean_rate'], pred_resp_per_cell * monod_df[bacteria_key_1]))#, squared=True))
    print('Error of total rate', rms / np.nanmean(-monod_df['Mean_rate']))  # getting the normalized squared error
    return (mean_squared_error,)


@app.cell
def _(bacteria_key_1, monod_df, pearsonr, pred_resp_per_cell, ydata):
    print('Per-cell correlation:', pearsonr(pred_resp_per_cell, ydata))
    print('Overall correlation:', pearsonr(-monod_df['Mean_rate'], pred_resp_per_cell * monod_df[bacteria_key_1]))
    return


@app.cell
def _(func):
    # def f(x,b0,b1):
    #     return b0 + (b1 * x)

    def f_wrapper_for_odr(beta, x): # parameter order for odr
        return func(x, *beta)

    return (f_wrapper_for_odr,)


@app.cell
def _(f_wrapper_for_odr, np, popt, scipy, xdata, ydata):
    parameters = popt

    model = scipy.odr.odrpack.Model(f_wrapper_for_odr)
    data = scipy.odr.odrpack.Data(xdata,ydata)
    myodr = scipy.odr.odrpack.ODR(data, model, beta0=parameters,  maxit=0)
    myodr.set_job(fit_type=2)
    parameterStatistics = myodr.run()
    df_e = len(xdata) - len(parameters) # degrees of freedom, error
    cov_beta = parameterStatistics.cov_beta # parameter covariance matrix from ODR
    sd_beta = parameterStatistics.sd_beta * parameterStatistics.sd_beta
    ci = []
    t_df = scipy.stats.t.ppf(0.975, df_e)
    ci = []
    for i in range(len(parameters)):
        ci.append([parameters[i] - t_df * parameterStatistics.sd_beta[i], parameters[i] + t_df * parameterStatistics.sd_beta[i]])

    tstat_beta = parameters / parameterStatistics.sd_beta # coeff t-statistics
    pstat_beta = (1.0 - scipy.stats.t.cdf(np.abs(tstat_beta), df_e)) * 2.0    # coef. p-values

    for i in range(len(parameters)):
        print('parameter:', parameters[i])
        print('   conf interval:', ci[i][0], ci[i][1])
        print('   tstat:', tstat_beta[i])
        print('   pstat:', pstat_beta[i])
        print(ci[i][0] - popt[i])
        print()


    ### Want to multiply parameter1 by 1e6 to get fmol/cell/day units
    ### Want to divide parameter 2 by 12.011 to get umol C/L units from ug/L
    return ci, parameters


@app.cell
def _(ci, parameters, popt):
    i_1 = 1
    print('half-sat is', parameters[i_1] / 12.011)
    print('Error on half sat is', (ci[i_1][0] - popt[i_1]) / 12.011)
    return


@app.cell
def _(ci, parameters, popt):
    i_2 = 0
    print('yc is', parameters[i_2])
    print('Error on yc is', ci[i_2][0] - popt[i_2])
    return


@app.cell
def _():
    ### Also doing some analysis of how robust this correlation is when we take into account measurement uncertainty.
    return


@app.cell
def _(
    bacteria_key_1,
    carbon_key,
    curve_fit,
    func,
    geometric_mean,
    monod_df,
    np,
    pearsonr,
):
    sigma = monod_df['Std_rate'].copy()
    rate = -monod_df['Mean_rate'].copy()
    n = 2000
    xdata_1 = monod_df[carbon_key]
    popts = np.empty((n, 2))
    rho_sim = np.empty(n)
    p_sim = np.empty(n)
    for i_3 in range(n):
        rate_sim = np.array([np.random.normal(r, s, 1) for r, s in zip(rate, sigma)])
        ydata_1 = (rate_sim.T / monod_df[bacteria_key_1].values).flatten()
        popt_1, pcov_1 = curve_fit(func, xdata_1.values, ydata_1)
        pred_resp_per_cell_1 = popt_1[0] * xdata_1 / (xdata_1 + popt_1[1])
        rho_sim[i_3] = pearsonr(pred_resp_per_cell_1, ydata_1).statistic
        p_sim[i_3] = pearsonr(pred_resp_per_cell_1, ydata_1).pvalue
        popts[i_3] = popt_1
    print(np.nanmean(rho_sim))
    print(geometric_mean(p_sim))
    return popt_1, xdata_1, ydata_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One clear takeaway here is that the reason this relationship *seems* to reasonably well when we look at the measured respiration rates

    is because it is so significantly correlated with the number of bacteria and that relationship gives us the right "monotonic" relationships between input and output. However, we really want to ask how well the per-cell respiration rates are predicted by this method. And the answer is... not super well. The Pearson correlation is not significant, and the plot is not super compelling either.
    """)
    return


@app.cell
def _(cdata, matplotlib, np, plt, popt_1, xdata_1, ydata_1):
    import matplotlib.colors as colors
    min_val, max_val = (0.2, 1.0)
    n_1 = 10
    orig_cmap = plt.cm.YlGn
    colors_list = orig_cmap(np.linspace(min_val, max_val, n_1))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', colors_list)
    lnorm = matplotlib.colors.LogNorm(vmin=0.08, vmax=1.0)
    plt.figure(dpi=150)
    plt.scatter(xdata_1 / 12.011, ydata_1 * 1000000.0, s=75, c=cdata, label='Observations', clip_on=False, zorder=10, cmap=cmap, norm=lnorm)
    plt.colorbar(extend='both', label='Chl ($\\mu$g L$^{-1}$)')
    plt.xlabel('Non-bacterial POC ($\\mu$mol L$^{-1}$)')
    plt.ylabel('Cell-specific respiration rate\n' + '(fmol O$_2$/cell/day)')
    plt.grid(alpha=0.5)
    plt.plot(np.arange(0, 50) / 12.011, 1000000.0 * popt_1[0] * np.arange(0, 50) / (np.arange(0, 50) + popt_1[1]), color='orange', lw=2, zorder=-1, label='$\\mathcal{R}$ = $y_{OM} \\frac{OM}{OM + K_m}$')
    plt.legend(bbox_to_anchor=(0.705, 0.86), loc='center', framealpha=1, facecolor='lightgrey', fontsize=12)
    plt.text(55 / 12.011, 2.2, 'y$_{OM}$' + f'={round(popt_1[0] * 1000000.0, 4)}')
    plt.text(55 / 12.011, 1.9, 'K$_m$=' + f'{round(popt_1[1] / 12.011, 3)} umol/L')
    #popt, pcov 
    #plt.xlim(0, 100)
    #plt.savefig('figures/Monod_respiration_fitting_nonbacterial_POC.png', dpi=300, bbox_inches='tight')
    plt.ylim(0, 1.7)  # Want to write these manually with Photoshop
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Proposing an alternative to Monod
    """)
    return


@app.cell
def _(curve_fit, df_p, np, plt):
    ### Hypothesis number 2
    def func_1(X, C1, C2, k):
        x1, x2 = X
        return C1 * (1 + C2 * x2 / x1) * (x1 / (x1 + k))  # x is carbon, y is Chl
    monod_df_1 = df_p.copy()
    bacteria_key_2 = 'Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'
    carbon_key_1 = 'Nonbacterial_carbon_ug/L_averaged_48_hours'
    xdata_2 = (monod_df_1[carbon_key_1], monod_df_1['CHLA'])
    #bacteria_key = 'Microbial_cells' 
    ydata_2 = -monod_df_1['Mean_rate'] / monod_df_1[bacteria_key_2]
    popt_2, pcov_2 = curve_fit(func_1, xdata_2, ydata_2)
    #carbon_key='C (ug/L)'
    print(popt_2)
    pred_resp_per_cell_2 = popt_2[0] * (1 + popt_2[1] * xdata_2[1] / xdata_2[0]) * (xdata_2[0] / (xdata_2[0] + popt_2[2]))
    fig_1, axes_1 = plt.subplots(1, 2, figsize=(10, 4), dpi=150)  #ml to uL
    ax_1 = axes_1[0]
    ax_1.scatter(-monod_df_1['Mean_rate'], pred_resp_per_cell_2 * monod_df_1[bacteria_key_2], s=100, color='cornflowerblue')
    ax_1.plot(np.arange(0, 1.55, 0.1), np.arange(0, 1.55, 0.1), color='black', linestyle='dashed', linewidth=2, zorder=-1)
    ax_1.set_xlabel('True respiration rate', fontsize=12)
    ax_1.set_ylabel('Predicted respiration rate', fontsize=12)
    ax_1.grid(alpha=0.5)
    ax_1 = axes_1[1]
    ax_1.scatter(ydata_2 * 1000000.0, pred_resp_per_cell_2 * 1000000.0, s=100, color='cornflowerblue')
    ax_1.plot(np.arange(0.2, 1.7, 0.1), np.arange(0.2, 1.7, 0.1), color='black', linestyle='dashed', linewidth=2, zorder=-1)
    ax_1.set_xlabel('True per-cell respiration rate (fmol O$_2$/cell/day)', fontsize=12)
    ax_1.set_ylabel('Predicted per-cell respiration rate \n (fmol O$_2$/cell/day)', fontsize=12)
    ax_1.grid(alpha=0.5)
    axes_1[0].set_title('Predicted vs. actual \n respiration rate', fontsize=12)
    # plt.scatter(-df['Mean_rate'], 
    #             popt[0]*df['Bacteria_cells-mL']*df['POC']/(df['POC'] + popt[1]))
    axes_1[1].set_title('Predicted vs.actual cell-specific \n respiration rate', fontsize=12)
    axes_1[0].set_xlim(0, 1.5)
    axes_1[0].set_ylim(0, 1.5)
    axes_1[1].set_xlim(0.2, 1.5)
    axes_1[1].set_ylim(0.2, 1.5)
    # ax.text(1.1,0.95, f'C={round(popt[0],8)}')
    # ax.text(1.1,0.85, r'K$_m$='+f'{round(popt[1],3)} ug/L')
    # axes[1].set_xlim(.2e-6, 1.4e-6)
    # axes[1].set_ylim(.2e-6, 1.4e-6)
    plt.subplots_adjust(wspace=0.5)
    return (
        bacteria_key_2,
        carbon_key_1,
        func_1,
        pcov_2,
        popt_2,
        pred_resp_per_cell_2,
        xdata_2,
        ydata_2,
    )


@app.cell
def _(np, pcov_2):
    print(np.linalg.cond(pcov_2))
    # I actually think it's largely driven by the fact that the C value in front has to be SO small - let's try doing the same thing but changing the units to be bacteria/uL
    print(np.diag(pcov_2))  # these are pretty different values - indicates the solution is not super well constrained, which isn't that surprising to me given the lack of data
    return


@app.cell
def _(parameters_1):
    parameters_1
    return


@app.cell
def _(f_wrapper_for_odr, np, popt_2, scipy, xdata_2, ydata_2):
    parameters_1 = popt_2
    model_1 = scipy.odr.odrpack.Model(f_wrapper_for_odr)
    data_1 = scipy.odr.odrpack.Data(xdata_2, ydata_2)
    myodr_1 = scipy.odr.odrpack.ODR(data_1, model_1, beta0=parameters_1, maxit=0)
    myodr_1.set_job(fit_type=2)
    parameterStatistics_1 = myodr_1.run()
    df_e_1 = len(xdata_2[0]) - len(parameters_1)
    #df_e = len(xdata) - len(parameters) # degrees of freedom, error
    cov_beta_1 = parameterStatistics_1.cov_beta  # degrees of freedom, error
    sd_beta_1 = parameterStatistics_1.sd_beta * parameterStatistics_1.sd_beta  # parameter covariance matrix from ODR
    ci_1 = []
    t_df_1 = scipy.stats.t.ppf(0.975, df_e_1)
    ci_1 = []
    for i_4 in range(len(parameters_1)):
        ci_1.append([parameters_1[i_4] - t_df_1 * parameterStatistics_1.sd_beta[i_4], parameters_1[i_4] + t_df_1 * parameterStatistics_1.sd_beta[i_4]])
    tstat_beta_1 = parameters_1 / parameterStatistics_1.sd_beta
    pstat_beta_1 = (1.0 - scipy.stats.t.cdf(np.abs(tstat_beta_1), df_e_1)) * 2.0
    for i_4 in range(len(parameters_1)):  # coeff t-statistics
        print('parameter:', parameters_1[i_4])  # coef. p-values
        print('   conf interval:', ci_1[i_4][0], ci_1[i_4][1])
        print('   tstat:', tstat_beta_1[i_4])
        print('   pstat:', pstat_beta_1[i_4])
        print(ci_1[i_4][0] - popt_2[i_4])
        print()
    return ci_1, parameters_1


@app.cell
def _(popt_2):
    popt_2
    return


@app.cell
def _(ci_1, parameters_1, popt_2):
    i_5 = 2
    print('half-sat is', parameters_1[i_5] / 12.011)
    print('Error on half sat is', (ci_1[i_5][0] - popt_2[i_5]) / 12.011)
    return


@app.cell
def _(ci_1, parameters_1, popt_2):
    i_6 = 0
    print('yc is', parameters_1[i_6])
    print('Error on yc is', ci_1[i_6][0] - popt_2[i_6])
    return


@app.cell
def _(ci_1, parameters_1, popt_2):
    i_7 = 1
    print('A0 is', parameters_1[i_7])
    print('Error on A0 is', ci_1[i_7][0] - popt_2[i_7])
    return


@app.cell
def _():
    ### Quick check on the units here
    return


@app.cell
def _(bacteria_key_2, df_p, df_p_all):
    # multiply by 1e6 picomol/1 umol -> yc has pmol O2/cell/day
    # A0 is unitless
    df_p['Mean_rate'] / df_p_all[bacteria_key_2] * 1000000.0  # this is in umol/L/day/cell
    return


@app.cell
def _(bacteria_key_2, carbon_key_1, df_p, df_p_all, func_1, np, plt, popt_2):
    plt.figure(dpi=150)
    plt.scatter(-1000000.0 * df_p_all['Mean_rate'] / df_p_all[bacteria_key_2], 1000000.0 * func_1(df_p_all[[carbon_key_1, 'CHLA']].values.T, popt_2[0], popt_2[1], popt_2[2]), c='lightgray', label='Background', zorder=5)
    plt.scatter(-df_p['Mean_rate'] * 1000000.0 / df_p[bacteria_key_2], 1000000.0 * func_1(df_p[[carbon_key_1, 'CHLA']].values.T, popt_2[0], popt_2[1], popt_2[2]), c='orange', edgecolor='black', label='Intrusions + euphotic', zorder=6)
    plt.plot(np.arange(0.3, 1.2, 0.1), np.arange(0.3, 1.2, 0.1), c='black', zorder=2, label='1-1 line')
    plt.legend(fontsize=10, facecolor='gainsboro', framealpha=1)
    plt.grid()
    plt.xlabel('Measured cell-specific respiration rate (fmol O$_2$/cell/day)')
    plt.ylabel('Predicted cell-specific \n respiration rate (fmol O$_2$/cell/day)')
    plt.title('Modified Monod')
    return


@app.cell
def _(bacteria_key_2, carbon_key_1, df_p, df_p_all, func_1, np, plt, popt_2):
    plt.figure(dpi=150)
    plt.scatter(-1000000.0 * df_p_all['Mean_rate'] / df_p_all[bacteria_key_2], 1000000.0 * func_1(df_p_all[[carbon_key_1, 'CHLA']].values.T, popt_2[0], popt_2[1], popt_2[2]), c='lightgray', label='Background', zorder=5)
    plt.scatter(-df_p['Mean_rate'] * 1000000.0 / df_p[bacteria_key_2], 1000000.0 * func_1(df_p[[carbon_key_1, 'CHLA']].values.T, popt_2[0], popt_2[1], popt_2[2]), c='orange', edgecolor='black', label='Intrusions + euphotic', zorder=6)
    plt.plot(np.arange(0.3, 1.2, 0.1), np.arange(0.3, 1.2, 0.1), c='black', zorder=2, label='1-1 line')
    plt.legend(fontsize=10, facecolor='gainsboro', framealpha=1)
    plt.grid()
    plt.xlabel('Measured cell-specific respiration rate (fmol O$_2$/cell/day)')
    plt.ylabel('Predicted cell-specific \n respiration rate (fmol O$_2$/cell/day)')
    plt.title('Modified Monod')
    return


@app.cell
def _(bacteria_key_2, carbon_key_1, df_p, df_p_all, func_1, np, plt, popt_2):
    plt.figure(dpi=150)
    plt.scatter(-df_p_all['Mean_rate'], df_p_all[bacteria_key_2] * func_1(df_p_all[[carbon_key_1, 'CHLA']].values.T, popt_2[0], popt_2[1], popt_2[2]), s=50, c='darkgray', label='Background', zorder=5)
    plt.scatter(-df_p['Mean_rate'], df_p[bacteria_key_2] * func_1(df_p[[carbon_key_1, 'CHLA']].values.T, popt_2[0], popt_2[1], popt_2[2]), c='orange', edgecolor='black', label='Intrusions + euphotic', zorder=6, s=51)
    plt.plot(np.arange(0, 1.5, 0.1), np.arange(0, 1.5, 0.1), c='black', zorder=2, label='1-1 line')
    plt.legend(fontsize=10, facecolor='gainsboro', framealpha=1)
    plt.grid()
    plt.xlabel('Measured respiration rate')
    plt.ylabel('Predicted respiration rate')
    plt.title('Modified Monod')
    return


@app.cell
def _(bacteria_key_2, carbon_key_1, curve_fit, df_p, func_1, np):
    ### Quick proof of concept that it's all in the scaling
    monod_df_2 = df_p.copy()
    xdata_3 = (monod_df_2[carbon_key_1], monod_df_2['CHLA'])
    ydata_scaled = -monod_df_2['Mean_rate'] * 1000 / (monod_df_2[bacteria_key_2] * 1 / 1000 / 10)
    popt_scaled, pcov_scaled = curve_fit(func_1, xdata_3, ydata_scaled)
    print(np.linalg.cond(pcov_scaled))  #ml to nL and umol to nom
    # now these values look much more similar, e.g. same order of magnitude roughly
    # and, the values are identical to what we get before, so it tells me there's no difference in the result
    print(np.diag(pcov_scaled))
    return (monod_df_2,)


@app.cell
def _(
    bacteria_key_2,
    mean_squared_error,
    monod_df_2,
    np,
    pred_resp_per_cell_2,
    ydata_2,
):
    rms_1 = np.sqrt(mean_squared_error(ydata_2, pred_resp_per_cell_2))#, squared=True))
    print(rms_1 / np.nanmean(ydata_2))
    rms_1 = np.sqrt(mean_squared_error(-monod_df_2['Mean_rate'], pred_resp_per_cell_2 * monod_df_2[bacteria_key_2]))# squared=True))
    print(rms_1 / np.nanmean(-monod_df_2['Mean_rate']))  # getting the normalized squared error
    return


@app.cell
def _(
    bacteria_key_2,
    carbon_key_1,
    df_p_all,
    func_1,
    mean_squared_error,
    np,
    popt_2,
):
    rms_2 = np.sqrt(mean_squared_error(-df_p_all['Mean_rate'], func_1(df_p_all[[carbon_key_1, 'CHLA']].values.T, popt_2[0], popt_2[1], popt_2[2]) * df_p_all[bacteria_key_2]))# squared=True))
    # this is for all samples
    print(rms_2 / np.nanmean(-df_p_all['Mean_rate']))  # getting the normalized squared error
    return


@app.cell
def _(bacteria_key_2, monod_df_2, pearsonr, pred_resp_per_cell_2, ydata_2):
    ### Spearman!! 
    print('Per-cell correlation:', pearsonr(pred_resp_per_cell_2, ydata_2))
    print('Overall correlation:', pearsonr(-monod_df_2['Mean_rate'], pred_resp_per_cell_2 * monod_df_2[bacteria_key_2]))
    return


@app.cell
def _(bacteria_key_2, monod_df_2, pred_resp_per_cell_2, spearmanr, ydata_2):
    print('Per-cell correlation:', spearmanr(pred_resp_per_cell_2, ydata_2))
    print('Overall correlation:', spearmanr(-monod_df_2['Mean_rate'], pred_resp_per_cell_2 * monod_df_2[bacteria_key_2]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How do things change a lot if we use all microbial cells to calculate normalized abundance (e.g., assuming all cells contributing to respiration)
    """)
    return


@app.cell
def _(curve_fit, monod_df_2):
    def func2(X, C1, C2, k):
        x1, x2 = X  # x is carbon, y is Chl
        return C1 * (1 + C2 * x2 / x1) * (x1 / (x1 + k))
    carbon_key_2 = 'Nonbacterial_carbon_ug/L_averaged_48_hours'
    ### Now with 48-hour averaged
    bacteria_key_3 = 'Microbial_cells'
    xdata_4 = (monod_df_2[carbon_key_2], monod_df_2['CHLA'])  #'Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'
    # this is using the microbial cells, not bacteria-ish cells
    ydata_3 = -monod_df_2['Mean_rate'] / monod_df_2[bacteria_key_3]
    popt2, pcov_3 = curve_fit(func2, xdata_4, ydata_3)
    pred_resp_per_cell_3 = popt2[0] * (1 + popt2[1] * xdata_4[1] / xdata_4[0]) * (xdata_4[0] / (xdata_4[0] + popt2[2])) * 1000000.0  #ml to uL
    return carbon_key_2, pred_resp_per_cell_3, ydata_3


@app.cell
def _(pred_resp_per_cell_3, r2_score, ydata_3):
    coefficient_of_det = r2_score(ydata_3 * 1000000.0, pred_resp_per_cell_3)
    # not a big change
    coefficient_of_det
    return


@app.cell
def _():
    from scipy.stats import normaltest

    return


@app.cell
def _():
    # QQ Plot
    from numpy.random import seed
    from numpy.random import randn
    from statsmodels.graphics.gofplots import qqplot
    from matplotlib import pyplot
    # seed the random number generator
    seed(1)
    # generate univariate observations
    data_2 = 5 * randn(100) + 50
    # q-q plot
    qqplot(data_2, line='s')
    pyplot.show()
    return (qqplot,)


@app.cell
def _(qqplot, ydata_3):
    qqplot(ydata_3, line='s')
    return


@app.cell
def _(pred_resp_per_cell_3, qqplot):
    qqplot(pred_resp_per_cell_3, line='s')
    return


@app.cell
def _(
    carbon_key_2,
    curve_fit,
    func_1,
    geometric_mean,
    interrupt,
    monod_df_2,
    np,
    pearsonr,
):
    sigma_1 = monod_df_2['Std_rate'].copy()
    rate_1 = -monod_df_2['Mean_rate'].copy()
    n_2 = 2000
    bacteria_key_4 = 'Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'
    xdata_5 = (monod_df_2[carbon_key_2], monod_df_2['CHLA'])
    popts_1 = np.empty((n_2, 3))
    rho_sim_1 = np.empty(n_2)
    p_sim_1 = np.empty(n_2)
    i_8 = 0
    while i_8 < n_2:
        rate_sim_1 = np.array([np.random.normal(r, s, 1) for r, s in zip(rate_1, sigma_1)])
        ydata_4 = (rate_sim_1.T / monod_df_2[bacteria_key_4].values).flatten()
        try:
            popt_3, pcov_4 = curve_fit(func_1, xdata_5, ydata_4)
        except RuntimeError:
            continue
        pred_resp_per_cell_4 = popt_3[0] * (1 + popt_3[1] * xdata_5[1] / xdata_5[0]) * (xdata_5[0] / (xdata_5[0] + popt_3[2]))
        rho_sim_1[i_8] = pearsonr(pred_resp_per_cell_4, ydata_4).statistic
        if np.isinf(pearsonr(pred_resp_per_cell_4, ydata_4).statistic):
            interrupt
        p_sim_1[i_8] = pearsonr(pred_resp_per_cell_4, ydata_4).pvalue
        popts_1[i_8] = popt_3
        i_8 = i_8 + 1
    print(np.nanmean(rho_sim_1))
    # print(np.nanmean(rho_sim))
    # print(geometric_mean(p_sim))
    print(geometric_mean(p_sim_1))  # ok nice nice nice
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### So when we use this modified relationship,

    we do get a better relationship between the predicted and actual per-cell respiration - which is gratifying because this should actually be harder to constrain. It also reduces the RMSE of both the predicted per-cell and predicted total respiration, so also good.

    Caveats are that this excludes data within the aphotic background bc we have some strange per-cell rates (I think simply because the values of both respiration and bacteria are pretty low for some of these deeper measurements). We could also replace the Chl/C with a freshness factor $\gamma$ and find some other way to prescribe or measure it in models etc.

    We can write this expression two ways:

    $r = C_1 (1 + C_2 \frac{Chl}{C})\frac{C}{C + K} \longleftrightarrow r = C_1 (\frac{C + C_2 Chl}{C}) \frac{C}{C + K}$
    """)
    return


@app.cell
def _(carbon_key_2, curve_fit, df_p, monod_df_2):
    def func1(X, C1, k):
        x1 = X  # x is carbon, y is Chl
        return C1 * (x1 / (x1 + k))

    def func2_1(X, C1, C2, k):
        x1, x2 = X  # x is carbon, y is Chl
        return C1 * (1 + C2 * x2 / x1) * (x1 / (x1 + k))
    bacteria_key_5 = 'Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'
    #carbon_key = 'Nonbacterial_carbon_ug/L_from_T0'
    xdata_T0_func1 = df_p[carbon_key_2]
    ydata_T0 = -df_p['Mean_rate'] / df_p[bacteria_key_5]
    popt1_T0, pcov1_T0 = curve_fit(func1, xdata_T0_func1, ydata_T0)
    pred_resp_per_cell_T0_func1 = func1(xdata_T0_func1, popt1_T0[0], popt1_T0[1])
    xdata_T0_func2 = (monod_df_2[carbon_key_2], monod_df_2['CHLA'])
    popt2_T0, pcov2_T0 = curve_fit(func2_1, xdata_T0_func2, ydata_T0)
    pred_resp_per_cell_T0_func2 = func2_1(xdata_T0_func2, popt2_T0[0], popt2_T0[1], popt2_T0[2])
    bacteria_key_5 = 'Average_bacteria_cells-mL_over_48_hrs-mL_subtr_pro'
    xdata_avg_func1 = df_p[carbon_key_2]
    ydata_avg = -df_p['Mean_rate'] / df_p[bacteria_key_5]
    popt1_avg, pcov1_avg = curve_fit(func1, xdata_avg_func1, ydata_avg)
    pred_resp_per_cell_avg_func1 = func1(xdata_avg_func1, popt1_avg[0], popt1_avg[1])
    xdata_avg_func2 = (monod_df_2[carbon_key_2], monod_df_2['CHLA'])
    popt2_avg, pcov2_avg = curve_fit(func2_1, xdata_avg_func2, ydata_avg)
    #carbon_key = 'Nonbacterial_carbon_ug/L_averaged_48_hours'
    pred_resp_per_cell_avg_func2 = func2_1(xdata_avg_func2, popt2_avg[0], popt2_avg[1], popt2_avg[2])
    return (
        bacteria_key_5,
        func1,
        func2_1,
        popt1_avg,
        popt2_avg,
        pred_resp_per_cell_avg_func1,
        pred_resp_per_cell_avg_func2,
        xdata_avg_func1,
        ydata_avg,
    )


@app.cell
def _():
    from matplotlib.ticker import AutoMinorLocator

    return (AutoMinorLocator,)


@app.cell
def _(
    AutoMinorLocator,
    bacteria_key_5,
    carbon_key_2,
    df_p,
    df_p_all,
    func1,
    func2_1,
    np,
    plt,
    popt1_avg,
    popt2_avg,
    pred_resp_per_cell_avg_func1,
    pred_resp_per_cell_avg_func2,
    r2_score,
    xdata_avg_func1,
    ydata_avg,
):
    markersize = 145
    fontsize = 15
    fig_2 = plt.figure(figsize=(11, 18), constrained_layout=True, dpi=300)
    gs = fig_2.add_gridspec(3, 28, height_ratios=(1.0, 1, 1))
    ax1 = fig_2.add_subplot(gs[0, :27])
    cax1 = fig_2.add_subplot(gs[0, 27:28])
    ax3 = fig_2.add_subplot(gs[1, :14])
    ax4 = fig_2.add_subplot(gs[1, 14:])
    #cax2 = fig.add_subplot(gs[1, 2:])
    ax5 = fig_2.add_subplot(gs[2, 7:21])
    from matplotlib.colors import LinearSegmentedColormap
    hex_colors = ['#603607', '#cb9b4e', '#f5eace', '#81bc41', '#2e6e18']
    cmap_1 = LinearSegmentedColormap.from_list('mycmap', hex_colors)
    # min_val, max_val = 0.0,0.9
    # n = 10
    # orig_cmap = plt.cm.BrBG
    # colors_list = orig_cmap(np.linspace(min_val, max_val, n))
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_cmap", colors_list)
    cp = ax1.scatter(xdata_avg_func1, ydata_avg * 1000000.0, s=160, c=df_p['CHLA'] / df_p[carbon_key_2], vmin=0, vmax=0.2, cmap=cmap_1, label='Observations', clip_on=False, zorder=10)
    ax1.set_xlabel('Non-bacterial POC ($\\mu$g L$^{-1}$)', fontsize=fontsize)
    ax1.set_ylabel('Cell-specific resp. rate\n' + '(fmol O$_2$ cell$^{-1}$ day$^{-1}$)', fontsize=fontsize)
    ax1.plot(np.arange(0, 50), 1000000.0 * func1(np.arange(0, 50), popt1_avg[0], popt1_avg[1]), color='orange', lw=6, zorder=-1, label='$\\mathcal{R}$ = $y_{C} ~\\frac{C}{C + K_{C}}$')
    ax1.legend(loc='lower right', framealpha=1, fontsize=14, facecolor='gainsboro')
    cbar = plt.colorbar(cp, cax=cax1, extend='max', label='Chl:C', pad=-20, orientation='vertical')
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(label='Chl:C', fontsize=fontsize)
    ax1.set_xlim(0, 48)
    ax1.set_ylim(0, 1.4)
    ax1.grid(alpha=0.5)
    cp = ax3.scatter(ydata_avg * 1000000.0, pred_resp_per_cell_avg_func1 * 1000000.0, s=markersize, cmap=cmap_1, vmin=0, vmax=0.075, edgecolor='black', c='orange', label='48-hour avg', zorder=10)
    ax3.legend(bbox_to_anchor=(0.425, 0.125), loc='center', framealpha=1, facecolor='gainsboro', fontsize=14)
    cp = ax4.scatter(ydata_avg * 1000000.0, pred_resp_per_cell_avg_func2 * 1000000.0, s=markersize, cmap=cmap_1, vmin=0, vmax=0.075, c='deeppink', edgecolor='black', zorder=10)
    coefficient_of_det_1 = r2_score(ydata_avg, pred_resp_per_cell_avg_func1)
    ax3.text(1.65, 0.325, '$\\mathcal{R}$ $\\sim \\frac{OM}{OM + K_{OM}}$' + '\n\n' + 'r$^2$=' + str(np.round(coefficient_of_det_1, 2)), horizontalalignment='right', fontsize=fontsize, verticalalignment='center', zorder=10, bbox=dict(facecolor='gainsboro', alpha=1))
    # we want a constant Chl to C ratio
    coefficient_of_det_1 = r2_score(ydata_avg, pred_resp_per_cell_avg_func2)
    # ax1.plot(np.arange(0.01, 50), 1e6*func2((np.arange(.01, 50),0*np.arange(.01, 50)), popt2_avg[0], popt2_avg[1], popt2_avg[2]), color='deeppink', 
    #          ls='dashed', lw=3, zorder=-2)
    # ax1.text(40, .625, 'Chl:OM=0',horizontalalignment='center', verticalalignment='center', zorder=10, fontsize=14,
    #          bbox=dict(facecolor='white', edgecolor='deeppink', alpha=1))
    ax4.text(1.65, 0.325, '$\\mathcal{R}$ $\\sim \\left(1 + \\frac{OM}{Chl}\\right)\\frac{OM}{OM + K_{OM}}$' + '\n\n' + 'r$^2$=' + str(np.round(coefficient_of_det_1, 2)) + '*', horizontalalignment='right', fontsize=fontsize, verticalalignment='center', zorder=10, bbox=dict(facecolor='gainsboro', alpha=1))
    pred_resp_all_data = func2_1(df_p_all[[carbon_key_2, 'CHLA']].values.T, popt2_avg[0], popt2_avg[1], popt2_avg[2])
    # ax1.plot(np.arange(0.01, 50), 1e6*func2((np.arange(.01, 50),.06*np.arange(.01, 50)), popt2_avg[0], popt2_avg[1], popt2_avg[2]), color='deeppink', 
    ax5.scatter(-df_p_all['Mean_rate'], pred_resp_all_data * df_p_all[bacteria_key_5], s=markersize, cmap=cmap_1, vmin=0, vmax=0.075, c='darkgray', label='Aphotic background', zorder=9)
    # ax1.text(40, 1.35, 'Chl:OM=0.06',horizontalalignment='center', verticalalignment='center', zorder=10, fontsize=14,
    ax5.scatter(ydata_avg * df_p[bacteria_key_5], pred_resp_per_cell_avg_func2 * df_p[bacteria_key_5], s=markersize, cmap=cmap_1, vmin=0, vmax=0.075, label='Intrusion+euphotic', c='skyblue', edgecolor='black', zorder=10)
    ax5.legend(framealpha=1, facecolor='gainsboro', fontsize=14, loc='upper left')
    # ax1.plot(np.arange(0.01, 50), 1e6*func2((np.arange(.01, 50),.03*np.arange(.01, 50)), popt2_avg[0], popt2_avg[1], popt2_avg[2]), color='deeppink', 
    ax5.grid()
    # ax1.text(40, 1.0, 'Chl:OM=0.03',horizontalalignment='center', verticalalignment='center', zorder=10, fontsize=14,
    ax5.plot(np.arange(0, 1.8, 0.1), np.arange(0, 1.8, 0.1), color='black', linestyle='dashed', linewidth=2, zorder=-1)
    ax5.set_ylim(0, 1.2)
    ax5.set_xlim(0, 1.45)
    ax5.set_xlabel('Measured resp. rate \n ($\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$)', fontsize=fontsize)
    #ax1.plot([], [], ls='dashed', c='deeppink', lw=3,  label=r'$\mathcal{R}$ = $y_{OM} \left(1 + C_0 \frac{Chl}{OM}\right)\frac{OM}{OM + K_{OM}}$')
    ax5.set_ylabel('Pred. resp. rate \n ($\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$)', fontsize=fontsize)
    for ax_2 in (ax3, ax4):
        ax_2.plot(np.arange(0, 1.8, 0.1), np.arange(0, 1.8, 0.1), color='black', linestyle='dashed', linewidth=2, zorder=-2)
        ax_2.set_ylim(0.2, 1.05)
        ax_2.set_xlim(0.2, 1.7)
        ax_2.grid(alpha=0.5)
        if ax_2 != ax3:
            ax_2.set_yticklabels([])
    #cbar.set_ticks([0.05, 0.1, .25, 0.5, 1], labels=[0.05, 0.1, .25, 0.5, 1])
        else:
            ax_2.set_ylabel('Pred. cell-specific resp. rate\n' + '(fmol O$_2$ cell$^{-1}$ day$^{-1}$)', fontsize=fontsize)
        ax_2.set_xlabel('Measured cell-specific resp. rate\n' + '(fmol O$_2$ cell$^{-1}$ day$^{-1}$)', fontsize=fontsize)
    labels = ['A', 'B', 'C', 'D', 'E']
    i_9 = 0
    for ax_2 in (ax1, ax3, ax4, ax5):
        ax_2.text(-0.05, -0.15, labels[i_9], weight='bold', fontsize=20, transform=ax_2.transAxes)
        i_9 = i_9 + 1
    for ax_2 in (ax1, cax1):
    # cp = ax3.scatter(ydata_T0*1e6, pred_resp_per_cell_T0_func1*1e6, s=markersize, 
    #             cmap=cmap, vmin=0, vmax=.075, 
    #             c='white', edgecolor='black',
    #             label='T0',
    #                  zorder=9)
        ax_2.xaxis.set_minor_locator(AutoMinorLocator(5))
    for ax_2 in (ax3, ax4, ax5):
        ax_2.xaxis.set_minor_locator(AutoMinorLocator(2))
    for ax_2 in (ax1, cax1, ax3, ax4, ax5):
        ax_2.yaxis.set_minor_locator(AutoMinorLocator(2))
    # cp = ax4.scatter(ydata_T0*1e6, pred_resp_per_cell_T0_func2*1e6,  s=markersize, 
    #             cmap=cmap, vmin=0, vmax=.075, edgecolor='black',
    #             c='white',
    #                  zorder=10)
    ### Pick locators
    #plt.savefig('figures/Monod_and_revised_Monod_fit_C_Chl_4_panel_figure.png', dpi=300, bbox_inches='tight')
    plt.subplots_adjust(hspace=0.3, wspace=0.45)

    plt.show()
    return (LinearSegmentedColormap,)


@app.cell
def _():
    ug_umol_conversion = 1/12.011
    return (ug_umol_conversion,)


@app.cell
def _(
    AutoMinorLocator,
    LinearSegmentedColormap,
    bacteria_key_5,
    carbon_key_2,
    df_p,
    df_p_all,
    func1,
    func2_1,
    np,
    plt,
    popt1_avg,
    popt2_avg,
    pred_resp_per_cell_avg_func1,
    pred_resp_per_cell_avg_func2,
    r2_score,
    ug_umol_conversion,
    xdata_avg_func1,
    ydata_avg,
):
    markersize_1 = 145
    fontsize_1 = 15
    fig_3 = plt.figure(figsize=(10, 18), dpi=100)
    gs_1 = fig_3.add_gridspec(3, 28, height_ratios=(1.0, 1, 1))
    ax1_1 = fig_3.add_subplot(gs_1[0, :27])
    cax1_1 = fig_3.add_subplot(gs_1[0, 27:28])
    ax3_1 = fig_3.add_subplot(gs_1[1, :14])
    #cax2 = fig.add_subplot(gs[1, 2:])
    ax4_1 = fig_3.add_subplot(gs_1[1, 14:])
    ax5_1 = fig_3.add_subplot(gs_1[2, 7 - 5:21 + 5])
    hex_colors_1 = ['#603607', '#cb9b4e', '#f5eace', '#81bc41', '#2e6e18']
    cmap_2 = LinearSegmentedColormap.from_list('mycmap', hex_colors_1)
    # min_val, max_val = 0.0,0.9
    # n = 10
    # orig_cmap = plt.cm.BrBG
    # colors_list = orig_cmap(np.linspace(min_val, max_val, n))
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_cmap", colors_list)
    cp_1 = ax1_1.scatter(xdata_avg_func1 * ug_umol_conversion, ydata_avg * 1000000.0, s=160, c=df_p['CHLA'] / df_p[carbon_key_2], vmin=0, vmax=0.06, cmap=cmap_2, label='Observations', clip_on=False, zorder=10)
    ax1_1.set_xlabel('Estimated bioavailable organic carbon (POC minus  \n non-photosynthetic bacterial carbon as proxy) ($\\mu$mol C L$^{-1}$)', fontsize=fontsize_1)
    ax1_1.set_ylabel('Abundance-normalized resp. rate\n' + '(pmol O$_2$ cell$^{-1}$ day$^{-1}$)', fontsize=fontsize_1)
    c_x_array = np.arange(0.01, 15, 0.05)
    ax1_1.plot(c_x_array, 1000000.0 * func1(c_x_array / ug_umol_conversion, popt1_avg[0], popt1_avg[1]), color='orange', alpha=1, lw=6, zorder=-1, label='$\\mathcal{R}$ = $y_{C} ~\\frac{C}{C + K_{C}}$')
    ax1_1.plot(c_x_array, 1000000.0 * func2_1((c_x_array / ug_umol_conversion, 0 * c_x_array / ug_umol_conversion), popt2_avg[0], popt2_avg[1], popt2_avg[2]), color='deeppink', ls='solid', lw=3, zorder=-2)
    ax1_1.text(3.5, 0.625, 'Chl:C=0', horizontalalignment='center', verticalalignment='center', zorder=10, fontsize=14, bbox=dict(facecolor='white', edgecolor='deeppink', alpha=1))
    ax1_1.plot(c_x_array, 1000000.0 * func2_1((c_x_array / ug_umol_conversion, 0.06 * c_x_array / ug_umol_conversion), popt2_avg[0], popt2_avg[1], popt2_avg[2]), color='deeppink', ls='solid', lw=3, zorder=-2)
    ax1_1.text(3.5, 1.35, 'Chl:C=0.06', horizontalalignment='center', verticalalignment='center', zorder=10, fontsize=14, bbox=dict(facecolor='white', edgecolor='deeppink', alpha=1))
    ax1_1.plot(c_x_array, 1000000.0 * func2_1((c_x_array / ug_umol_conversion, 0.03 * c_x_array / ug_umol_conversion), popt2_avg[0], popt2_avg[1], popt2_avg[2]), color='deeppink', ls='solid', lw=3, zorder=-2)
    ax1_1.text(3.5, 1.0, 'Chl:C=0.03', horizontalalignment='center', verticalalignment='center', zorder=10, fontsize=14, bbox=dict(facecolor='white', edgecolor='deeppink', alpha=1))
    ax1_1.plot([], [], ls='solid', c='deeppink', lw=3, label='$\\mathcal{R}$ = $y_{C} \\left(1 + A_0 \\frac{Chl}{C}\\right)\\frac{C}{C + K_{C}}$')
    ax1_1.legend(loc='lower right', framealpha=1, fontsize=14, facecolor='gainsboro')
    cbar_1 = plt.colorbar(cp_1, cax=cax1_1, extend='max', label='Chl:C', pad=-20, orientation='vertical')
    cbar_1.ax.tick_params(labelsize=fontsize_1)
    cbar_1.set_label(label='Chl:C', fontsize=fontsize_1)
    ax1_1.set_xlim(0, 4)
    ax1_1.set_ylim(0, 1.4)
    # we want a constant Chl to C ratio
    ax1_1.grid(alpha=0.5, zorder=-10)
    cp_1 = ax3_1.scatter(ydata_avg * 1000000.0, pred_resp_per_cell_avg_func1 * 1000000.0, s=markersize_1, cmap=cmap_2, vmin=0, vmax=0.075, edgecolor='black', c='orange', zorder=10)
    ax3_1.plot([], [], c='k', ls='dashed', lw=3, label='1:1 line')
    ax3_1.legend(bbox_to_anchor=(0.425, 0.125), loc='center', framealpha=1, facecolor='gainsboro', fontsize=14)
    cp_1 = ax4_1.scatter(ydata_avg * 1000000.0, pred_resp_per_cell_avg_func2 * 1000000.0, s=markersize_1, cmap=cmap_2, vmin=0, vmax=0.075, c='deeppink', edgecolor='black', zorder=10)
    coefficient_of_det_2 = r2_score(ydata_avg, pred_resp_per_cell_avg_func1)
    ax3_1.text(1.65, 0.325, '$\\mathcal{R}$ $\\sim \\frac{C}{C + K_{C}}$' + '\n\n' + 'r$^2$=' + str(np.round(coefficient_of_det_2, 2)), horizontalalignment='right', fontsize=fontsize_1, verticalalignment='center', zorder=10, bbox=dict(facecolor='gainsboro', alpha=1))
    coefficient_of_det_2 = r2_score(ydata_avg, pred_resp_per_cell_avg_func2)
    ax4_1.text(1.65, 0.325, '$\\mathcal{R}$ $\\sim \\left(1 + \\frac{Chl}{C}\\right)\\frac{C}{C + K_{C}}$' + '\n\n' + 'r$^2$=' + str(np.round(coefficient_of_det_2, 2)) + '**', horizontalalignment='right', fontsize=fontsize_1, verticalalignment='center', zorder=10, bbox=dict(facecolor='gainsboro', alpha=1))
    pred_resp_all_data_1 = func2_1(df_p_all[[carbon_key_2, 'CHLA']].values.T, popt2_avg[0], popt2_avg[1], popt2_avg[2])
    ax5_1.scatter(-df_p_all['Mean_rate'], pred_resp_all_data_1 * df_p_all[bacteria_key_5], s=markersize_1, cmap=cmap_2, vmin=0, vmax=0.075, c='darkgray', label='Aphotic non-intrusion', zorder=9)
    ax5_1.scatter(ydata_avg * df_p[bacteria_key_5], pred_resp_per_cell_avg_func2 * df_p[bacteria_key_5], s=markersize_1, cmap=cmap_2, vmin=0, vmax=0.075, label='Intrusion+euphotic', c='#ff92cd', edgecolor='black', zorder=10)
    coefficient_of_det_2 = r2_score(ydata_avg * df_p[bacteria_key_5], pred_resp_per_cell_avg_func2 * df_p[bacteria_key_5])
    ax5_1.text(1.4, 0.175, '$\\mathcal{R}_B$ $\\sim B \\left(1 + \\frac{Chl}{C}\\right)\\frac{C}{C + K_{C}}$' + '\n\n' + 'r$^2$=' + str(np.round(coefficient_of_det_2, 2)) + '***', horizontalalignment='right', fontsize=14, verticalalignment='center', zorder=10, bbox=dict(facecolor='gainsboro', alpha=1))
    ax5_1.legend(framealpha=1, facecolor='gainsboro', fontsize=14, loc='upper left')
    ax5_1.grid()
    ax5_1.plot(np.arange(0, 1.8, 0.1), np.arange(0, 1.8, 0.1), color='black', linestyle='dashed', linewidth=2, zorder=-1)
    ax5_1.set_ylim(0, 1.2)
    ax5_1.set_xlim(0, 1.45)
    ax5_1.set_xlabel('Measured resp. rate \n ($\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$)', fontsize=fontsize_1)
    ax5_1.set_ylabel('Pred. resp. rate \n ($\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$)', fontsize=fontsize_1)
    for ax_3 in (ax3_1, ax4_1):
        ax_3.plot(np.arange(0, 1.8, 0.1), np.arange(0, 1.8, 0.1), color='black', linestyle='dashed', linewidth=2, zorder=-2)
        ax_3.set_ylim(0.2, 1.05)
        ax_3.set_xlim(0.2, 1.7)
        ax_3.grid(alpha=0.5)
    #cbar.set_ticks([0.05, 0.1, .25, 0.5, 1], labels=[0.05, 0.1, .25, 0.5, 1])
        if ax_3 != ax3_1:
            ax_3.set_yticklabels([])
        else:
            ax_3.set_ylabel('Pred. abundance-norm. \n' + 'resp. rate ' + '(pmol O$_2$ cell$^{-1}$ day$^{-1}$)', fontsize=fontsize_1)
        ax_3.set_xlabel('Measured abundance-norm. \n' + 'resp. rate ' + '(pmol O$_2$ cell$^{-1}$ day$^{-1}$)', fontsize=fontsize_1)
    labels_1 = ['A', 'B', 'C', 'D', 'E']
    i_10 = 0
    for ax_3 in (ax1_1, ax3_1, ax4_1, ax5_1):
        ax_3.text(-0.07, -0.15, labels_1[i_10], weight='bold', fontsize=20, transform=ax_3.transAxes)
    # cp = ax3.scatter(ydata_T0*1e6, pred_resp_per_cell_T0_func1*1e6, s=markersize, 
    #             cmap=cmap, vmin=0, vmax=.075, 
    #             c='white', edgecolor='black',
    #             label='T0',
    #                  zorder=9)
        i_10 = i_10 + 1
    for ax_3 in (ax1_1, cax1_1):
        ax_3.xaxis.set_minor_locator(AutoMinorLocator(5))
    for ax_3 in (ax3_1, ax4_1, ax5_1):
        ax_3.xaxis.set_minor_locator(AutoMinorLocator(2))
    for ax_3 in (ax1_1, cax1_1, ax3_1, ax4_1, ax5_1):  #label='48-hour avg',
        ax_3.yaxis.set_minor_locator(AutoMinorLocator(2))
    # cp = ax4.scatter(ydata_T0*1e6, pred_resp_per_cell_T0_func2*1e6,  s=markersize, 
    #             cmap=cmap, vmin=0, vmax=.075, edgecolor='black',
    #             c='white',
    #                  zorder=10)
    ### Pick locators
    #plt.savefig('../../Respiration/Notebooks/AutoBOD_rate_analysis/figures/Monod_and_revised_Monod_fit_C_Chl_4_panel_figure_060326.png', dpi=300, bbox_inches='tight')
    plt.subplots_adjust(hspace=0.27, wspace=0.8)

    plt.show()
    return


@app.cell
def _(bacteria_key_5, df_p, pred_resp_per_cell_avg_func2, r2_score, ydata_avg):
    r2_score(ydata_avg * df_p[bacteria_key_5], pred_resp_per_cell_avg_func2 * df_p[bacteria_key_5])
    return


@app.cell
def _(ug_umol_conversion):
    2.42*ug_umol_conversion
    return


@app.cell
def _(ug_umol_conversion):
    7.3*ug_umol_conversion
    return


@app.cell
def _(ug_umol_conversion):
    4.01*ug_umol_conversion
    return


@app.cell
def _(ug_umol_conversion):
    10.14*ug_umol_conversion
    return


@app.cell
def _(bacteria_key_5, df_p, pred_resp_per_cell_avg_func2, r2_score, ydata_avg):
    coefficient_of_det_3 = r2_score(ydata_avg * df_p[bacteria_key_5], pred_resp_per_cell_avg_func2 * df_p[bacteria_key_5])
    coefficient_of_det_3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Want to implement likelihood ratio test, per Andy Solow

    We have two models:

    null model - $r_c = \mu_0 \frac{OM}{OM + k_{OM}}$

    and alternative model = $r_c = \mu_0 (1 + a_0 \frac{Chl}{OM})\frac{OM}{OM + k_{OM}}$

    Fit to get least squares parameter estimation in natural log space - so

    $log (r_c) = log(\mu_0) + log(\frac{OM}{OM + k_{OM}}) + error = f_0(x) + error$

    and $log(r_c) = log(\mu_0) + log(1 + a_0 \frac{Chl}{OM}) + log(\frac{OM}{OM + k_{OM}})+ error = f_1(x) + error$

    looking at [this website](https://nedcharles.com/regression/Nonlinear_Least_Squares_Regression_For_Python.html) for some examples of how to fit nonlinear least squares, which is what we're looking at - essentially, minimizing $\sum_1^n (log(r_c) - f_0(x))^2$ - the difference between the proposed and actual solution. Now my understanding is that nonlinear least squares - doesn't have a closed solution so we iterate to get the right parameters that minimize the difference. Then, $RSS_0 = \sum_1^n (log(r_c) - f_0(x))^2$ and $RSS_1 = \sum_1^n (log(r_c) - f_1(\vec{x}))^2$

    To calculate the likelihood ratio, we write $LR = n log(\frac{RSS_0}{RSS_1})$ - if the null hypothesis is a bad model, the top term should be much larger than the bottom term. Given that we added 1 more free parameter, we can test against the chi squared distribution with 1 degree of freedom - if it exceeds that distribution at the upper 0.05 quantile, we reject the null model.
    """)
    return


@app.cell
def _():
    # from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
    # # Plotting module
    # import matplotlib.pyplot as plt
    from scipy.optimize import least_squares

    return (least_squares,)


@app.cell
def _(np):
    def fcn2minExpCos(x, beta1, beta2):
        return np.exp(-beta1*x) * np.cos(beta2*x)

    return (fcn2minExpCos,)


@app.cell
def _(fcn2minExpCos, np, plt):
    # Simulate data using the same function we will fit to.
    # Generate array of 101 data points from zero to ten in 0.1 increments
    x = np.linspace(0, 10.0, num=101)   
    Beta1 = 0.5                         # First Beta parameter for the exponential decay
    Beta2 = 5                           # Second Beta parameter for the cosine
    NumParams = 2                       # Number of model parameters 
    StdNoise = 0.1                      # Noise Std Dev
    y = fcn2minExpCos(x, Beta1, Beta2)  # Generate the signal values before adding noise

    # Generate random noise sampled from a normal (Gaussian) distribution
    # are added to the original signal
    noiseSamples = np.random.normal(size=len(y), scale=StdNoise)
    yNoisy = y + noiseSamples

    # Plot the original signal and overlay the noisy signal to show the scale of the noise
    plt.plot(x, y, 'b')
    plt.plot(x, yNoisy, 'r')
    plt.xlabel('x')
    plt.ylabel('y (blue) and yNoise (red)')
    plt.show()
    return x, yNoisy


@app.cell
def _(bacteria_key_5, carbon_key_2, curve_fit, df_p, func1):
    ## Testing whether contour plot is better way to represent subplot C
    monod_df_3 = df_p.copy()
    xdata_6 = (monod_df_3[carbon_key_2], monod_df_3['CHLA'])
    ydata_5 = -monod_df_3['Mean_rate'] / monod_df_3[bacteria_key_5]
    # pred_resp_per_cell = func2(xdata, popt[0], popt[1], popt[2])
    # chl_vals = np.linspace(0, 1.3, 30) 
    # poc_vals = np.linspace(1.7, 49, 30)
    # poc_vals, chl_vals = np.meshgrid(poc_vals, chl_vals)
    popt_4, pcov_5 = curve_fit(func1, xdata_6[0], ydata_5)  #ml to uL
    return xdata_6, ydata_5


@app.cell
def _(np):
    def fcn2minExpCosErrFunc(beta, x, y):
        return (y - (np.exp(-beta[0]*x) * np.cos(beta[1]*x)))

    return (fcn2minExpCosErrFunc,)


@app.cell
def _():
    InitialParams = [1., 1.]
    return (InitialParams,)


@app.cell
def _(InitialParams, fcn2minExpCosErrFunc, least_squares, x, yNoisy):
    LSOptimResult = least_squares(fcn2minExpCosErrFunc, InitialParams, method='lm', args=(x, yNoisy))
    return


@app.function
def fnc2minNullModel(beta, x, y):
    return (y - (beta[0]*x/(x + beta[1])))


@app.cell
def _(np):
    def fnc2minNullModel_1(beta, x, y):
        func_val = beta[0] + np.log(x / (x + beta[1]))
        return np.log(y) - func_val

    def fnc2minAlternateModel(beta, x, y):
        func_val = beta[0] + np.log(x[0] / (x[0] + beta[1])) + np.log(1 + beta[2] * x[1] / x[0])
        return np.log(y) - func_val

    return fnc2minAlternateModel, fnc2minNullModel_1


@app.cell
def _(InitialParams, fnc2minNullModel_1, least_squares, xdata_6, ydata_5):
    LSOptimResult_null_model = least_squares(fnc2minNullModel_1, InitialParams, method='lm', args=(xdata_6[0], ydata_5))
    print(LSOptimResult_null_model)
    return (LSOptimResult_null_model,)


@app.cell
def _(LSOptimResult_null_model, fnc2minNullModel_1, np, xdata_6, ydata_5):
    ### calculate RSS_0 
    RSS_0 = np.sum(fnc2minNullModel_1(LSOptimResult_null_model.x, xdata_6[0], ydata_5) ** 2)
    RSS_0
    return (RSS_0,)


@app.cell
def _(fnc2minAlternateModel, least_squares, xdata_6, ydata_5):
    LSOptimResult_AlternateModel = least_squares(fnc2minAlternateModel, [1.0, 1.0, 1.0], method='lm', args=(xdata_6, ydata_5))
    print(LSOptimResult_AlternateModel)
    return (LSOptimResult_AlternateModel,)


@app.cell
def _(
    LSOptimResult_AlternateModel,
    fnc2minAlternateModel,
    np,
    xdata_6,
    ydata_5,
):
    RSS_1 = np.sum(fnc2minAlternateModel(LSOptimResult_AlternateModel.x, xdata_6, ydata_5) ** 2)
    return (RSS_1,)


@app.cell
def _(RSS_0, RSS_1, np, ydata_5):
    LR = len(ydata_5) * np.log(RSS_0 / RSS_1)
    print(LR)
    if LR > 3.84:
        print('We can reject the null hypothesis and conclude that the alternative model is an improvement at the 0.05 significance level')
    else:
        print('We cannot reject the null hypothesis')
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
