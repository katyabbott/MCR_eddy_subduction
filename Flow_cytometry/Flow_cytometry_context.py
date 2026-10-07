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
    ### Generate box plots of FCM cell counts with Cast 1 plotted separately
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import os
    import glob
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    import xarray as xr
    from matplotlib.lines import Line2D
    import random
    import pickle
    import cmocean
    import gsw
    from matplotlib import ticker

    return np, pd, plt, sns


@app.cell
def _(sns):
    sns.set_context("notebook", font_scale=1.25, rc={"figure.figsize": (10, 7.5)})
    sns.set_theme('notebook', style='white')
    return


@app.cell
def _(pd):
    wkdir = '/Users/katyabbott/Documents/MIT_WHOI_Joint_Program/Research_Projects/Microbial_respiration_CALYPSO/'
    fcm_dir = 'Data/Flow cytometry/'

    # Location of log files from AutoBOD

    output_dir = wkdir + 'Analyses/Respiration/Output/'

    resp_df_pooled = pd.read_csv(output_dir + 'CALYPSO_all_respiration_rates_QC_with_environmental_data_pooled.csv')
    resp_df_pooled = resp_df_pooled.drop('Unnamed: 0', axis=1)
    resp_df_pooled = resp_df_pooled.rename({'Pooled_mean_rate': 'Mean_rate', 'Pooled_sd': 'Std_rate'}, axis=1)
    return fcm_dir, resp_df_pooled, wkdir


@app.cell
def _(fcm_dir, pd, wkdir):
    pt_fcm_df = pd.read_csv(wkdir + fcm_dir + 'Processed_data/cleaned_data_for_NGS/Calypso2022_FCM_phototrophs.csv')
    # pt - phototrophs

    bact_fcm_df = pd.read_csv(wkdir + fcm_dir + 'Processed_data/cleaned_data_for_NGS/Calypso2022_FCM_bacteria.csv')


    # # ignore all but T0s
    bact_fcm_df = bact_fcm_df[bact_fcm_df['Timepoint (h)'] == 0].copy() 
    return bact_fcm_df, pt_fcm_df


@app.cell
def _():
    phototroph_names = {'Syn': 'Synechococcus', 'Euks': 'Picoeukaryotes', 'Pro': 'Prochlorococcus'}
    return (phototroph_names,)


@app.cell
def _(bact_fcm_df, pt_fcm_df):
    ### Defining some phases based on known cast distinctions 
    pt_phase1 = pt_fcm_df[pt_fcm_df.Cast.isin(range(1,10))].copy()
    pt_phase2 = pt_fcm_df[pt_fcm_df.Cast.isin(range(10,36))].copy()
    pt_phase3 = pt_fcm_df[pt_fcm_df.Cast.isin(range(36,62))].copy()
    pt_phase2_3 = pt_fcm_df[pt_fcm_df.Cast.isin(range(10,62))].copy()

    bact_phase2 = bact_fcm_df[bact_fcm_df.Cast.isin(range(10,36))].copy()
    bact_phase3 = bact_fcm_df[bact_fcm_df.Cast.isin(range(36,62))].copy()


    # C1 vs. no C1
    pt_cast1 = pt_fcm_df[pt_fcm_df["Cast"] == 1].sort_values(by='Depth').copy()
    bact_cast1 = bact_fcm_df[bact_fcm_df["Cast"] == 1].sort_values(by='Depth').copy()

    pt_noc1 = pt_fcm_df[pt_fcm_df['Cast'] != 1].copy()
    bact_noc1 = bact_fcm_df[bact_fcm_df['Cast'] != 1].copy()

    return bact_cast1, bact_noc1, pt_cast1, pt_noc1


@app.cell
def _(np, pt_cast1):
    _discrete_colors_bright = ['#5E79FD', '#FFA400', '#FF3399']
    colors = np.repeat('lightgray', 4)
    depth_bins = [[0, 25], [25, 50], [50, 100], [100, 150], [150, 200], [200, 300], [300, 550]]
    _x1 = [(len(depth_bins) - _i) / 4 for _i in range(len(depth_bins))]
    _x2 = np.roll(_x1, shift=1)
    _bounds = [len(depth_bins) / 4 + 0.15] + list(((_x1 + _x2) / 2)[1:]) + [0.1]
    _labels = pt_cast1[pt_cast1['Alias'] == 'Syn'].Depth.values
    new_labels = []
    for _label in _labels:
        for _i, _depth_bin in enumerate(depth_bins):
            if (_label > _depth_bin[0]) & (_label <= _depth_bin[1]):
                _i_db = _i
    # converting depth bins to appropriate locations for labels
                _db = _depth_bin
        _bb = _bounds[_i_db:_i_db + 2]
        new_labels.append(-(_label - _db[0]) / (_db[1] - _db[0]) * 0.15 + _bb[0])
    return colors, depth_bins, new_labels


@app.cell
def _(
    bact_cast1,
    bact_noc1,
    colors,
    depth_bins,
    new_labels,
    phototroph_names,
    plt,
    pt_cast1,
    pt_noc1,
):
    _cast1_color = 'dodgerblue'
    _fig, _axes = plt.subplots(4, figsize=(4.5, 11.5), dpi=300)
    for _j, _phototroph in enumerate(['Euks', 'Syn', 'Pro']):
        _ax = _axes[_j]
        for _i, _dd in enumerate(depth_bins):
            _temp_df = pt_noc1[pt_noc1['Depth'].isin(range(_dd[0], _dd[1]))].copy()
            _bplot = _ax.boxplot(_temp_df[_temp_df['Alias'] == _phototroph]['Cells-mL'].values, vert=False, positions=[(len(depth_bins) - _i) / 4], patch_artist=True)
            _bplot['boxes'][0].set_facecolor(colors[_j])
            _bplot['medians'][0].set_color('black')
        _ax.set_ylim(0.1, len(depth_bins) / 4 + 0.15)
        _ax.set_yticklabels(['0 - 25 m', '25 - 50 m', '50 - 100 m', ' 100 - 150 m', '150 - 200 m', '200 - 300 m', '300 - 550 m'])
        _ax.set_title(phototroph_names[_phototroph])
    plt.xlabel('cells mL$^{-1}$')  #ax.set_xscale('log')
    _labels = pt_cast1[pt_cast1['Alias'] == 'Syn'].Depth.values
    for _i, _phototroph in enumerate(['Euks', 'Syn', 'Pro']):  #ax.set_xlim(10**.55, 1e5)
        _ax = _axes[_i]
        _cells = pt_cast1[pt_cast1['Alias'] == _phototroph]['Cells-mL'].astype('float').values
        _ax.scatter(_cells, new_labels, color=_cast1_color, s=100, marker='*', zorder=10, clip_on=False)
        if _i == 0:
            _ax.scatter([], [], color=_cast1_color, s=100, marker='*', zorder=10, label='Cast 1')
    _axes[0].legend(bbox_to_anchor=(0.84, 0.125), loc='center')
    _ax = _axes[3]
    _alias = 'Bacteria'
    for _i, _dd in enumerate(depth_bins):
        _temp_df = bact_noc1[bact_noc1['Depth'].isin(range(_dd[0], _dd[1]))].copy()
        _bplot = _ax.boxplot(_temp_df['Bacteria_cells-mL'].values, vert=False, positions=[(len(depth_bins) - _i) / 4], patch_artist=True)
        _bplot['boxes'][0].set_facecolor(colors[_j])
        _bplot['medians'][0].set_color('black')
    _ax.set_ylim(0.1, len(depth_bins) / 4 + 0.15)
    _ax.set_xlim(30000.0, 1600000.0)
    _ax.set_yticklabels(['0 - 25 m', '25 - 50 m', '50 - 100 m', ' 100 - 150 m', '150 - 200 m', '200 - 300 m', '300 - 550 m'])
    _ax.set_title('Non-photosynthetic bacterioplankton')
    _cells = bact_cast1['Bacteria_cells-mL'].astype('float').values

    _ax.scatter(_cells, new_labels, color=_cast1_color, s=100, marker='*', zorder=10, clip_on=False)
    if _i == 0:
        _ax.scatter([], [], color=_cast1_color, s=100, marker='*', zorder=10, label='Cast 1')
    for _ax in _axes:
        _ax.grid(lw=0.5, c='gray', alpha=0.25)
        _ax.patch.set_edgecolor('black')
        _ax.patch.set_linewidth(1)
    #plt.savefig('figures/FCM_Cast1_all_casts_boxplots.png', dpi=300, bbox_inches='tight')
    plt.subplots_adjust(hspace=0.35)

    plt.show()
    return


@app.cell
def _(
    bact_cast1,
    bact_noc1,
    colors,
    depth_bins,
    new_labels,
    plt,
    pt_cast1,
    pt_noc1,
):
    _cast1_color = 'dodgerblue'
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 2.0), dpi=450)
    _ax = _axes[0]
    for _i, _dd in enumerate(depth_bins):
        for _j, _phototroph in enumerate(['Euks', 'Syn', 'Pro']):
            _temp_df = pt_noc1[pt_noc1['Depth'].isin(range(_dd[0], _dd[1]))].copy()
            if _j == 0:
                _tot_cells = _temp_df[_temp_df['Alias'] == _phototroph]['Cells-mL'].values.copy()
            else:
                _tot_cells = _tot_cells + _temp_df[_temp_df['Alias'] == _phototroph]['Cells-mL'].values
        _bplot = _ax.boxplot(_tot_cells, vert=False, positions=[(len(depth_bins) - _i) / 4], patch_artist=True)
        _bplot['boxes'][0].set_facecolor(colors[_j])
        _bplot['medians'][0].set_color('black')
        _ax.set_ylim(0.1, len(depth_bins) / 4 + 0.15)
    _ax.set_title('Picophytoplankton')
    _labels = pt_cast1[pt_cast1['Alias'] == 'Syn'].Depth.values
    _ax = _axes[0]
    _cells = pt_cast1[pt_cast1['Alias'] == 'Syn']['Cells-mL'].values + pt_cast1[pt_cast1['Alias'] == 'Pro']['Cells-mL'].values + pt_cast1[pt_cast1['Alias'] == 'Euks']['Cells-mL'].values
    _ax.scatter(_cells, new_labels, color=_cast1_color, s=100, marker='*', zorder=10, clip_on=False)
    _ax.scatter([], [], color=_cast1_color, s=100, marker='*', zorder=10, label='Cast 1')
    _ax.legend(bbox_to_anchor=(0.82, 0.125), loc='center', facecolor='white', framealpha=1)
    _ax.ticklabel_format(axis='x', style='scientific', scilimits=[-5, 5])
    _ax.set_xlim(-100)
    _ax = _axes[1]
    _alias = 'Bacteria'
    for _i, _dd in enumerate(depth_bins):
        _temp_df = bact_noc1[bact_noc1['Depth'].isin(range(_dd[0], _dd[1]))].copy()
        _bplot = _ax.boxplot(_temp_df['Bacteria_cells-mL'].values, vert=False, positions=[(len(depth_bins) - _i) / 4], patch_artist=True)
        _bplot['boxes'][0].set_facecolor(colors[_j])
        _bplot['medians'][0].set_color('black')
    _ax.set_ylim(0.1, len(depth_bins) / 4 + 0.15)
    _ax.set_xlim(30000.0, 1600000.0)
    _ax.set_title('Non-photosynthetic bacterioplankton')
    _cells = bact_cast1['Bacteria_cells-mL'].astype('float').values
    _ax.scatter(_cells, new_labels, color=_cast1_color, s=100, marker='*', zorder=10, clip_on=False)
    if _i == 0:
        _ax.scatter([], [], color=_cast1_color, s=100, marker='*', zorder=10, label='Cast 1')
    for _ax in _axes:
        _ax.grid(lw=0.5, c='gray', alpha=0.25)
        _ax.patch.set_edgecolor('black')
        _ax.patch.set_linewidth(1)
        _ax.set_xlabel('cells mL$^{-1}$')
    plt.subplots_adjust(hspace=0.35)
    for _ax in _axes:
        if _ax == _axes[0]:
            _ax.grid(lw=0.5, c='gray', alpha=0.25)
            _ax.patch.set_edgecolor('black')
            _ax.patch.set_linewidth(1)
            _ax.set_yticks([(len(depth_bins) - _i) / 4 for _i in range(len(depth_bins))], labels=['0 - 25 m', '25 - 50 m', '50 - 100 m', ' 100 - 150 m', '150 - 200 m', '200 - 300 m', '300 - 550 m'])
            _ax.tick_params(which='major', axis='y', length=6, width=10, color='black')
        else:
            _ax.set_yticklabels([])
    plt.subplots_adjust(wspace=0.05)

    plt.show()
    return


@app.cell
def _(depth_bins, phototroph_names, plt, pt_cast1, pt_noc1):
    _fig, _axes = plt.subplots(3, figsize=(4.5, 8.5), dpi=300)
    for _j, _phototroph in enumerate(['Euks', 'Syn', 'Pro']):
        _ax = _axes[_j]
        for _i, _dd in enumerate(depth_bins):
            _temp_df = pt_noc1[pt_noc1['Depth'].isin(range(_dd[0], _dd[1]))].copy()
            _bplot = _ax.boxplot(_temp_df[_temp_df['Alias'] == _phototroph]['Cells-mL'].values, vert=False, positions=[(len(depth_bins) - _i) / 4], patch_artist=True)
            _bplot['boxes'][0].set_facecolor('yellowgreen')
            _bplot['medians'][0].set_color('black')
        _ax.set_ylim(0.1, len(depth_bins) / 4 + 0.15)
        _ax.set_yticklabels(['0 - 25 m', '25 - 50 m', '50 - 100 m', ' 100 - 150 m', '150 - 200 m', '200 - 300 m', '300 - 550 m'])
        _ax.set_title(phototroph_names[_phototroph])
    plt.xlabel('cells/mL')
    _labels = pt_cast1[pt_cast1['Alias'] == 'Syn'].Depth.values
    for _ax in _axes:
        _ax.grid(lw=0.5, c='gray', alpha=0.25)
        _ax.patch.set_edgecolor('black')
        _ax.patch.set_linewidth(1)
    #plt.savefig('figures/FCM_Cast1_all_casts_boxplots_phototrophs.png', dpi=300, bbox_inches='tight')
    plt.subplots_adjust(hspace=0.35)

    plt.show()
    return


@app.cell
def _(colors, depth_bins, plt, pt_noc1):
    _fig, _ax = plt.subplots(1, figsize=(4.5, 4), dpi=300)
    for _i, _dd in enumerate(depth_bins):
        for _j, _phototroph in enumerate(['Euks', 'Syn', 'Pro']):
            _temp_df = pt_noc1[pt_noc1['Depth'].isin(range(_dd[0], _dd[1]))].copy()
            if _j == 0:
                _tot_cells = _temp_df[_temp_df['Alias'] == _phototroph]['Cells-mL'].values.copy()
            else:
                _tot_cells = _tot_cells + _temp_df[_temp_df['Alias'] == _phototroph]['Cells-mL'].values
        _bplot = _ax.boxplot(_tot_cells, vert=False, positions=[(len(depth_bins) - _i) / 4], patch_artist=True)
        _bplot['boxes'][0].set_facecolor(colors[_j])
        _bplot['medians'][0].set_color('black')
        _ax.set_ylim(0.1, len(depth_bins) / 4 + 0.15)
    _ax.set_yticklabels(['0 - 25 m', '25 - 50 m', '50 - 100 m', ' 100 - 150 m', '150 - 200 m', '200 - 300 m', '300 - 550 m'])
    _ax.set_title('All cells (depth distribution)')
    plt.xlabel('cells/mL')

    plt.show()
    return


@app.cell
def _():
    ### Plotting respiration data alongside phototroph + bacterial
    return


@app.cell
def _(resp_df_pooled):
    resp_df_noc1 = resp_df_pooled[resp_df_pooled['Cast'] != 'C1']
    resp_df_c1 = resp_df_pooled[resp_df_pooled['Cast'] == 'C1']

    resp_df_c1.sort_values(by='Depth', inplace=True)
    return resp_df_c1, resp_df_noc1


@app.cell
def _(np, pt_cast1):
    _discrete_colors_bright = ['#5E79FD', '#FFA400', '#FF3399']
    colors_1 = np.repeat('lightgray', 4)
    depth_bins_1 = [[0, 25], [25, 50], [50, 100], [100, 150], [150, 200], [200, 300], [300, 550]]
    _x1 = [(len(depth_bins_1) - _i) / 4 for _i in range(len(depth_bins_1))]
    _x2 = np.roll(_x1, shift=1)
    _bounds = [len(depth_bins_1) / 4 + 0.15] + list(((_x1 + _x2) / 2)[1:]) + [0.1]
    new_labels_1 = []
    _labels = pt_cast1[pt_cast1['Alias'] == 'Syn'].Depth.values
    for _label in _labels:
        for _i, _depth_bin in enumerate(depth_bins_1):
            if (_label > _depth_bin[0]) & (_label <= _depth_bin[1]):
                _i_db = _i
                _db = _depth_bin
        _bb = _bounds[_i_db:_i_db + 2]
        new_labels_1.append(-(_label - _db[0]) / (_db[1] - _db[0]) * 0.15 + _bb[0])
    return colors_1, depth_bins_1, new_labels_1


@app.cell
def _(
    bact_noc1,
    colors_1,
    depth_bins_1,
    new_labels_1,
    plt,
    pt_cast1,
    pt_noc1,
    resp_df_c1,
    resp_df_noc1,
):
    color = 'dodgerblue'
    _fig, _axes = plt.subplots(3, figsize=(4.5, 8.5), dpi=300)
    _ax = _axes[0]
    for _i, _dd in enumerate(depth_bins_1):
        for _j, _phototroph in enumerate(['Euks', 'Syn', 'Pro']):
            _temp_df = pt_noc1[pt_noc1['Depth'].isin(range(_dd[0], _dd[1]))].copy()
            if _j == 0:
                _tot_cells = _temp_df[_temp_df['Alias'] == _phototroph]['Cells-mL'].values.copy()
            else:
                _tot_cells = _tot_cells + _temp_df[_temp_df['Alias'] == _phototroph]['Cells-mL'].values
        _bplot = _ax.boxplot(_tot_cells, vert=False, positions=[(len(depth_bins_1) - _i) / 4], patch_artist=True)
        _bplot['boxes'][0].set_facecolor(colors_1[_j])
        _bplot['medians'][0].set_color('black')
        _ax.set_ylim(0.1, len(depth_bins_1) / 4 + 0.15)
    _ax.set_title('Picophytoplankton')
    _labels = pt_cast1[pt_cast1['Alias'] == 'Syn'].Depth.values
    _ax = _axes[0]
    _cells = pt_cast1[pt_cast1['Alias'] == 'Syn']['Cells-mL'].values + pt_cast1[pt_cast1['Alias'] == 'Pro']['Cells-mL'].values + pt_cast1[pt_cast1['Alias'] == 'Euks']['Cells-mL'].values
    _ax.scatter(_cells, new_labels_1, color=color, s=100, marker='*', zorder=10, clip_on=False)
    _ax.scatter([], [], color=color, s=100, marker='*', zorder=10, label='Cast 1')
    _ax.legend(bbox_to_anchor=(0.84, 0.125), loc='center', facecolor='white', framealpha=1)
    _ax.ticklabel_format(axis='x', style='scientific', scilimits=[-5, 5])
    _ax.set_xlim(-100)
    _ax = _axes[1]
    _alias = 'Bacteria'
    for _i, _dd in enumerate(depth_bins_1):
        _temp_df = bact_noc1[bact_noc1['Depth'].isin(range(_dd[0], _dd[1]))].copy()
        _bplot = _ax.boxplot(_temp_df['Bacteria_cells-mL'].values, vert=False, positions=[(len(depth_bins_1) - _i) / 4], patch_artist=True)
        _bplot['boxes'][0].set_facecolor('orange')
        _bplot['medians'][0].set_color('black')
    _ax.set_ylim(0.1, len(depth_bins_1) / 4 + 0.15)
    _ax.set_xlim(30000.0, 1600000.0)
    _ax.set_title('Non-photosynthetic bacteria')
    _ax.set_xlim(0)
    _ax = _axes[2]
    for _i, _dd in enumerate(depth_bins_1):
        _temp_df = resp_df_noc1[resp_df_noc1['Depth'].isin(range(_dd[0], _dd[1]))].copy()
        _bplot = _ax.boxplot(-_temp_df['Mean_rate'].values, vert=False, positions=[(len(depth_bins_1) - _i) / 4], patch_artist=True)
        _bplot['boxes'][0].set_facecolor('lightgray')
        _bplot['medians'][0].set_color('black')
    _ax.set_ylim(0.1, len(depth_bins_1) / 4 + 0.15)
    _ax.set_xlim(0)
    _ax.set_xlabel('$\\mu$mol O$_2$ L$^{-1}$ day$^{-1}$')
    _ax.set_title('Respiration')
    vals = -resp_df_c1['Mean_rate'].astype('float').values
    _ax.scatter(vals, new_labels_1, color=color, s=100, marker='*', zorder=10, clip_on=False)
    if _i == 0:
        _ax.scatter([], [], color=color, s=100, marker='*', zorder=10, label='Cast 1')
    for _ax in _axes:
        _ax.grid(lw=0.5, c='gray', alpha=0.25)
        _ax.patch.set_edgecolor('black')
        _ax.patch.set_linewidth(1)
        _ax.set_yticks([(len(depth_bins_1) - _i) / 4 for _i in range(len(depth_bins_1))], labels=['0 - 25 m', '25 - 50 m', '50 - 100 m', ' 100 - 150 m', '150 - 200 m', '200 - 300 m', '300 - 550 m'])
        _ax.tick_params(which='major', axis='y', length=6, width=2, color='black')
    plt.subplots_adjust(hspace=0.45)

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
