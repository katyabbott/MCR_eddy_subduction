import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualizing cell concentrations from incubation experiments on board R/V PQP
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import os
    import glob
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    from scipy.stats.mstats import gmean
    import xarray as xr
    from scipy.interpolate import interp1d
    import gsw
    import scipy.optimize

    return gmean, np, os, pd, plt, sns


@app.cell
def _(os, sns):
    sns.set_theme(style="ticks", font_scale=1.5, font='Verdana')
    os.getcwd()

    wkdir = '/Users/katyabbott/Documents/MIT_WHOI_Joint_Program/Research_Projects/Microbial_respiration_CALYPSO/'
    fcm_dir = 'Data/Flow cytometry/'
    return fcm_dir, wkdir


@app.cell
def _(fcm_dir, pd, wkdir):
    fcm_df = pd.read_csv(wkdir + fcm_dir + 'Processed_data/cleaned_data_for_NGS/Calypso2022_FCM_bacteria.csv')

    cast_fcm_df = fcm_df[fcm_df['Timepoint (h)'] == 0]

    # excluding cast 1
    cast_fcm_df_nocast1 = cast_fcm_df[~cast_fcm_df['Cast'].isin([1, 3, 4])].copy()
    return cast_fcm_df_nocast1, fcm_df


@app.cell
def _(cast_fcm_df_nocast1, plt):
    _fig, _ax = plt.subplots(figsize=(8, 8))
    cast_fcm_df_nocast1.plot.scatter(x='Bacteria_cells-mL', y='Depth', ax=_ax, s=75, color='deeppink')
    plt.semilogx()
    # plt.ylim(400, 0)
    # plt.xlim(10**0, 10**7)
    plt.ylim(300, 0)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Looking at incubation data
    """)
    return


@app.cell
def _(fcm_df):
    inc_fcm = fcm_df[fcm_df['Respiration_msmt']].copy()
    return (inc_fcm,)


@app.cell
def _(inc_fcm):
    inc_fcm['Cast_Niskin_Timepoint'] = inc_fcm['Cast_Niskin'].astype('str') + '_T' + inc_fcm['Timepoint (h)'].astype('str')
    return


@app.cell
def _(inc_fcm, pd):
    cast_ids = pd.unique(inc_fcm['Cast_Niskin'])
    return (cast_ids,)


@app.cell
def _(cast_ids, gmean, inc_fcm, np):
    pct_diffs = np.array([])
    for _id in cast_ids:
        _subset_df = inc_fcm[inc_fcm['Cast_Niskin'] == _id]
        if _subset_df.shape[0] == 1:
            print(_subset_df)
            print('Not enough data')
            continue
        _subset_df = _subset_df[['Cast_Niskin_Timepoint', 'Timepoint (h)', 'Bacteria_cells-mL']]
        _subset_df = _subset_df.groupby('Cast_Niskin_Timepoint').transform(gmean).drop_duplicates()
        _subset_df = _subset_df.sort_values(by='Timepoint (h)')
        _T0_val = _subset_df[_subset_df['Timepoint (h)'] == 0.0]['Bacteria_cells-mL'].values
        _delta_bacteria = _subset_df[_subset_df['Timepoint (h)'] == _subset_df['Timepoint (h)'].max()]['Bacteria_cells-mL'].values - _subset_df[_subset_df['Timepoint (h)'] == 0.0]['Bacteria_cells-mL'].values
        _pct_diff = 100 * _delta_bacteria / _T0_val
        pct_diffs = np.append(pct_diffs, _pct_diff)
    return (pct_diffs,)


@app.cell
def _(cast_ids, gmean, inc_fcm, np):
    pct_diff_24 = np.array([])
    pct_diff_48 = np.array([])
    pct_diff_72 = np.array([])
    for _id in cast_ids:
        _subset_df = inc_fcm[inc_fcm['Cast_Niskin'] == _id]
        if _subset_df.shape[0] == 1:
            print('Not enough data')
            continue
        _subset_df = _subset_df[['Cast_Niskin_Timepoint', 'Timepoint (h)', 'Bacteria_cells-mL']]
        _subset_df = _subset_df.groupby('Cast_Niskin_Timepoint').transform(gmean).drop_duplicates()
        _subset_df['Timepoint (h)'] = np.round(_subset_df['Timepoint (h)'])
        _subset_df = _subset_df.sort_values(by='Timepoint (h)')
        _T0_val = _subset_df[_subset_df['Timepoint (h)'] == 0.0]['Bacteria_cells-mL'].values
        _delta_bacteria = _subset_df[_subset_df['Timepoint (h)'] == 24.0]['Bacteria_cells-mL'].values - _subset_df[_subset_df['Timepoint (h)'] == 0.0]['Bacteria_cells-mL'].values
        if len(_delta_bacteria) != 1:
            # there's no data at t24
            continue
        _pct_diff = 100 * _delta_bacteria / _T0_val
        pct_diff_24 = np.append(pct_diff_24, _pct_diff)
        if _pct_diff > 150:
            print(_id)
        _delta_bacteria = _subset_df[_subset_df['Timepoint (h)'] == 48.0]['Bacteria_cells-mL'].values - _subset_df[_subset_df['Timepoint (h)'] == 0.0]['Bacteria_cells-mL'].values
        _pct_diff = 100 * _delta_bacteria / _T0_val
        pct_diff_48 = np.append(pct_diff_48, _pct_diff)
        _delta_bacteria = _subset_df[_subset_df['Timepoint (h)'] == 72.0]['Bacteria_cells-mL'].values - _subset_df[_subset_df['Timepoint (h)'] == 0.0]['Bacteria_cells-mL'].values
        _pct_diff = 100 * _delta_bacteria / _T0_val
        pct_diff_72 = np.append(pct_diff_72, _pct_diff)
    return pct_diff_24, pct_diff_48, pct_diff_72


@app.cell
def _(pct_diff_24, pct_diff_48, pct_diff_72, plt):
    plt.figure(figsize=(6, 4))
    _bplot1 = plt.boxplot(pct_diff_24, vert=True, positions=[1], widths=[0.2], patch_artist=True)
    _bplot2 = plt.boxplot(pct_diff_48, vert=True, positions=[2], widths=[0.2], patch_artist=True)
    _bplot3 = plt.boxplot(pct_diff_72, vert=True, positions=[3], widths=[0.2], patch_artist=True)
    _bplot1['boxes'][0].set_facecolor('lightblue')
    _bplot1['medians'][0].set_color('black')
    _bplot1['medians'][0].set_lw(3)
    _bplot2['boxes'][0].set_facecolor('lightblue')
    _bplot2['medians'][0].set_color('black')
    _bplot2['medians'][0].set_lw(3)
    _bplot3['boxes'][0].set_facecolor('lightblue')
    _bplot3['medians'][0].set_color('black')
    _bplot3['medians'][0].set_lw(3)
    plt.axhline(0, ls='dashed', c='k', zorder=1, lw=3)
    plt.gca().set_xticklabels(['24 hr', '48 hr', '72 hr'])
    plt.yticks([-25, 0, 25, 50, 75])
    plt.ylabel('% change from T0')
    #plt.savefig('figures/Bacterial_incubations_percent_change.png', dpi=300)
    plt.grid(zorder=-1)

    plt.show()
    return


@app.cell
def _(pct_diffs, plt):
    plt.hist(pct_diffs, bins=50);

    plt.show()
    return


@app.cell
def _(np, pct_diffs):
    np.sum(np.abs(pct_diffs) > 25)/len(pct_diffs)
    return


@app.cell
def _(cast_ids, np, pct_diffs):
    cast_ids[np.abs(pct_diffs) > 25]
    return


@app.cell
def _(inc_fcm):
    cast_niskin = inc_fcm[inc_fcm['Cast_Niskin'] == 'C33N14'].copy()
    return (cast_niskin,)


@app.cell
def _(cast_niskin):
    ### Getting sample stats for one particular case

    t0_val = cast_niskin[cast_niskin['Timepoint (h)'] == 0]['Bacteria_cells-mL'].values
    print(100*(cast_niskin[cast_niskin['Timepoint (h)'] == 24]['Bacteria_cells-mL'] - t0_val)/t0_val)
    print(100*(cast_niskin[cast_niskin['Timepoint (h)'] == 48]['Bacteria_cells-mL'] - t0_val)/t0_val, "%")
    print(100*(cast_niskin[cast_niskin['Timepoint (h)'] == 72]['Bacteria_cells-mL'] - t0_val)/t0_val, "%")
    return


@app.cell
def _(inc_fcm):
    fcm_t0 = inc_fcm[inc_fcm['Timepoint (h)'] == 0][['Cast_Niskin', 'Bacteria_cells-mL']].set_index('Cast_Niskin')
    return (fcm_t0,)


@app.cell
def _(fcm_t0, inc_fcm):
    fcm_subset = inc_fcm[inc_fcm['Timepoint (h)'].isin([24, 48])].copy()
    fcm_subset_mean_conc = fcm_subset[['Cast_Niskin', 'Bacteria_cells-mL']].groupby('Cast_Niskin').aggregate('mean')
    merged_df = fcm_t0.merge(fcm_subset_mean_conc, left_index=True, right_index=True)
    merged_df.head()
    return (merged_df,)


@app.cell
def _(merged_df, np, pct_diff_24, pct_diff_48, pct_diff_72, plt):
    _fig, axes = plt.subplots(2, figsize=(9, 16))
    _ax = axes[0]
    _bplot1 = _ax.boxplot(pct_diff_24, vert=True, positions=[1], widths=[0.2], patch_artist=True)
    _bplot2 = _ax.boxplot(pct_diff_48, vert=True, positions=[2], widths=[0.2], patch_artist=True)
    _bplot3 = _ax.boxplot(pct_diff_72, vert=True, positions=[3], widths=[0.2], patch_artist=True)
    _bplot1['boxes'][0].set_facecolor('lightblue')
    _bplot1['medians'][0].set_color('black')
    _bplot1['medians'][0].set_lw(3)
    _bplot2['boxes'][0].set_facecolor('lightblue')
    _bplot2['medians'][0].set_color('black')
    _bplot2['medians'][0].set_lw(3)
    _bplot3['boxes'][0].set_facecolor('lightblue')
    _bplot3['medians'][0].set_color('black')
    _bplot3['medians'][0].set_lw(3)
    _ax.axhline(0, ls='dashed', c='k', zorder=1, lw=3)
    _ax.set_xticklabels(['24 hr', '48 hr', '72 hr'])
    _ax.set_yticks([-25, 0, 25, 50, 75])
    _ax.set_ylabel('% change from T0')
    _ax.grid(zorder=-1)
    plt.scatter(merged_df['Bacteria_cells-mL_x'], merged_df['Bacteria_cells-mL_y'], s=125, c='lightblue', edgecolor='k')
    plt.plot(np.arange(0, 1300000.0), np.arange(0, 1300000.0), zorder=-2, ls='dashed', c='k')
    plt.xlim(0, 1300000.0)
    plt.ylim(0, 1300000.0)
    plt.xlabel('Non-photo bacterioplankton \n cell counts at T0 (cells mL$^{-1}$)')
    plt.ylabel('Average non-photo \n bacterioplankton cell counts \nover T24 and T48 (cells mL$^{-1}$)')
    plt.grid()
    # plt.savefig(wkdir + 'Manuscript/figures/SupportingInfo/Bacterial_incubations_percent_change_2panel.png', dpi=300, bbox_inches='tight')

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
