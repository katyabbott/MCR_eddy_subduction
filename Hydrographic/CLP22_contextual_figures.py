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
    # Context figures for 2022 CALYPSO cruise
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    import scipy.interpolate
    import numpy as np
    import gsw
    import xarray as xr
    import glob, os
    import pandas as pd
    #import hvplot.pandas
    #import holoviews as hv
    #import hvplot.xarray
    import matplotlib.ticker as mticker
    import matplotlib as mpl
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import cmocean
    import pickle
    from sklearn import linear_model
    from sklearn.utils import shuffle
    import seaborn as sns
    import cartopy
    import cartopy.crs as ccrs
    import matplotlib.patheffects as pe
    import colorcet as cc
    from matplotlib.colors import LogNorm

    return (
        LogNorm,
        cartopy,
        ccrs,
        cmocean,
        colors,
        glob,
        gsw,
        loadmat,
        make_axes_locatable,
        mpl,
        mticker,
        np,
        pd,
        pe,
        pickle,
        plt,
        scipy,
        shuffle,
        sns,
        xr,
    )


@app.cell
def _():
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']
    return (CB_color_cycle,)


@app.cell
def _(sns):
    sns.set(style="white", font_scale=1.5, font='Verdana')
    return


@app.cell
def _(np):
    def haversine_np(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)

        All args must be of equal length.    

        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km

    return (haversine_np,)


@app.cell
def _():
    wkdir = '/Users/katyabbott/Documents/MIT_WHOI_Joint_Program/Research_Projects/Microbial_respiration_CALYPSO/'
    resp_dir = 'Data/Respiration/'
    hydrg_dir = 'Data/Hydrographic/'
    return hydrg_dir, resp_dir, wkdir


@app.cell
def _(gsw, hydrg_dir, wkdir, xr):
    ecoctd = xr.open_dataset(wkdir + hydrg_dir + 'L3_EcoCTD_Calibrated_with_optical_backscatter.nc')
    _o2_sol = gsw.O2sol(ecoctd.CTD_SA, ecoctd.CTD_CT, ecoctd.pressure, ecoctd.CTD_lon, ecoctd.CTD_lat)
    #pressure = np.tile()
    _o2_sol = _o2_sol * ((ecoctd.CTD_Sigma0.values + 1000) / 1000)
    o2_conc_CTD = ecoctd['OXY_O2sat'].values * _o2_sol / 100
    AOU = _o2_sol - o2_conc_CTD
    ecoctd['AOU'] = AOU
    ecoctd['AOU'].attrs = {'Long_name': 'Apparent oxygen utilization', 'units': 'umol/L'}  #convert from umol/kg to umol/L
    slope_factor = 5.36064586
    deep_offset = 0.0603
    ecoctd['FLS_chl'] = (ecoctd['FLS_chl'] + 0.16) / 2.65
    ### Because I think a correction to our Chl calibration is needed, I'm going to manually do it right here for now
    # Correcting EcoCTD data
    ecoctd['FLS_chl'] = (ecoctd['FLS_chl'] - deep_offset) * slope_factor
    return (ecoctd,)


@app.cell
def _(gsw, hydrg_dir, wkdir, xr):
    ctd = xr.open_dataset(wkdir + hydrg_dir + 'L3_CALYPSO_CTD_Calibrated.nc')
    ctd['DEPTH'] = -gsw.z_from_p(ctd.PRES, ctd.LAT)
    ctd['DOXY_UMOL-KG'] = ctd['DOXY_PSAT'] * gsw.O2sol(ctd.SA, ctd.CT, ctd.PRES, ctd.LON, ctd.LAT) / 100
    _o2_sol = gsw.O2sol(ctd.SA, ctd.CT, ctd.PRES, ctd.LON, ctd.LAT)
    _o2_sol = _o2_sol * ((ctd.SIGMA0.values + 1000) / 1000)
    ctd['AOU'] = _o2_sol - _o2_sol * ctd['DOXY_PSAT'] / 100  #convert from umol/kg to umol/L
    return (ctd,)


@app.cell
def _(ecoctd, np, pd):
    df = pd.DataFrame({'temp': ecoctd.CTD_CT.values.ravel(), 'salt': ecoctd.CTD_SA.values.ravel()})
    df['chl'] = np.log10(ecoctd.FLS_chl.values.ravel())
    df['O2sat'] = ecoctd.OXY_O2sat.values.ravel()
    df['bbp'] = ecoctd.FLS_bbp2.values.ravel()
    return (df,)


@app.cell
def _(ctd, np, pd):
    df_ctd = pd.DataFrame({'temp': ctd.CT.values.ravel(),
                           'salt': ctd.SA.values.ravel()})
    df_ctd['Chl'] = np.log10(ctd.CHLA.values.ravel())
    df_ctd['Nitrate'] = ctd.NITRATE.values.ravel()
    df_ctd['cast'] = np.repeat(ctd.CAST, ctd.dims['n_levels'])
    return (df_ctd,)


@app.cell
def _(df_ctd):
    df_ctd_1 = df_ctd[df_ctd['salt'] > 38]
    return (df_ctd_1,)


@app.cell
def _(df, gsw, np):
    sa_array = np.arange(np.nanmin(df['salt']) - .2, np.nanmax(df['salt']) + .2, .1)
    ct_array = np.arange(np.nanmin(df['temp']) - .2, np.nanmax(df['temp']) + .2, .1)

    sa_grid, ct_grid = np.meshgrid(sa_array, ct_array)

    sigma_grid = gsw.sigma0(sa_grid, ct_grid)
    return ct_array, ct_grid, sa_array, sa_grid, sigma_grid


@app.cell
def _(df, np):
    oxy = df.O2sat.values
    _nan_ind = ~np.isnan(oxy)
    salt = df.salt.values[_nan_ind]
    temp = df.temp.values[_nan_ind]
    oxy = oxy[_nan_ind]
    return oxy, salt, temp


@app.cell
def _(oxy, salt, shuffle, temp):
    S, T, O2 = shuffle(salt, temp, oxy, random_state=0)
    return O2, S, T


@app.cell
def _(
    O2,
    S,
    T,
    cmocean,
    ct_array,
    ct_grid,
    np,
    plt,
    sa_array,
    sa_grid,
    sigma_grid,
):
    plt.figure(figsize=(10, 7))
    cp = plt.scatter(S, T, c=O2, alpha=0.25, s=1, cmap=cmocean.cm.matter)
    cbar = plt.colorbar(cp, label='Dissolved oxygen (% saturation)')
    _CS = plt.contour(sa_grid, ct_grid, sigma_grid, colors='black', zorder=0, linewidths=1, linestyles='dashed')
    _label_levels = np.array([_CS.levels[2], _CS.levels[3], _CS.levels[6]])
    plt.clabel(_CS, _label_levels, fontsize=15, inline_spacing=10, fmt='%2.1f', zorder=-1)  # Label every second level
    plt.ylim(ct_array[0] + 0.05, ct_array[-1] - 0.01)
    plt.xlim(sa_array[0] + 0.23, sa_array[-1] - 0.12)
    plt.gca().tick_params(bottom=True, left=True)
    cbar.solids.set(alpha=1)
    plt.xlabel('Absolute Salinity')
    #plt.savefig('figures/context_figures/TS_plot_oxygen.png', dpi=300, bbox_inches='tight')
    plt.ylabel('Conservative temperature ($\\degree$C)')
    return


@app.cell
def _(
    S,
    T,
    cmocean,
    ct_array,
    ct_grid,
    df_ctd_1,
    np,
    plt,
    sa_array,
    sa_grid,
    sigma_grid,
):
    plt.figure(figsize=(10, 7))
    plt.scatter(S, T, s=0.5, color='gray', alpha=0.5)
    plt.scatter(df_ctd_1.salt, df_ctd_1.temp, c=df_ctd_1.Nitrate, s=10, cmap=cmocean.cm.dense, alpha=0.6, vmin=0)
    cbar_1 = plt.colorbar(label=f'Nitrate ($\\mu$M)')
    _CS = plt.contour(sa_grid, ct_grid, sigma_grid, colors='black', zorder=0, linewidths=1, linestyles='dashed')
    _label_levels = np.array([_CS.levels[2], _CS.levels[3], _CS.levels[6]])
    plt.clabel(_CS, _label_levels, fontsize=15, inline_spacing=10, fmt='%2.1f', zorder=-1)
    plt.ylim(ct_array[0] + 0.05, ct_array[-1] - 0.01)
    plt.xlim(sa_array[0] + 0.23, sa_array[-1] - 0.12)
    plt.gca().tick_params(bottom=True, left=True)
    plt.xlabel('Absolute Salinity')
    plt.ylabel('Conservative temperature ($\\degree$C)')
    cbar_1.solids.set(alpha=1)
    return (cbar_1,)


@app.cell
def _(cbar_1, ct_grid, df_ctd_1, np, plt, sa_grid, sigma_grid):
    df_ctd_1['Phase'] = ' '
    df_ctd_1.loc[df_ctd_1['cast'] < 10, 'Phase'] = 'Phase 1'
    df_ctd_1.loc[(df_ctd_1['cast'] > 9) & (df_ctd_1['cast'] < 36), 'Phase'] = 'Phase 2'
    df_ctd_1.loc[df_ctd_1['cast'] > 35, 'Phase'] = 'Phase 3'
    color_dict = {'Phase 1': '#5E79FD', 'Phase 2': '#FFA400', 'Phase 3': '#FF3399'}
    _fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for p in ['Phase 1', 'Phase 2', 'Phase 3']:
        _df_subset = df_ctd_1[df_ctd_1['Phase'] == p]
        if p == 'Phase 1':
            ax.scatter(_df_subset.salt, _df_subset.temp, c=color_dict[p], s=10, alpha=0.4, zorder=10)
        else:
            ax.scatter(_df_subset.salt, _df_subset.temp, c=color_dict[p], s=10, alpha=0.4)
    for p in ['Phase 1', 'Phase 2', 'Phase 3']:
        _df_subset = df_ctd_1[df_ctd_1['Phase'] == p]
        ax.scatter(np.nan, np.nan, c=color_dict[p], label=p, s=100, alpha=1)
    _CS = plt.contour(sa_grid, ct_grid, sigma_grid, colors='black', zorder=0, linewidths=1, linestyles='dashed')
    _label_levels = np.array([_CS.levels[2], _CS.levels[3], _CS.levels[6]])
    plt.clabel(_CS, _label_levels, fontsize=15, inline_spacing=10, fmt='%2.1f', zorder=-1)
    plt.xlim(38.2, 38.87)
    plt.ylim(13.1, 14.55)
    plt.gca().tick_params(bottom=True, left=True)
    plt.xlabel('Absolute Salinity')
    plt.ylabel('Conservative temperature ($\\degree$C)')
    cbar_1.solids.set(alpha=1)
    return (ax,)


@app.cell
def _(ax, ct_array, ct_grid, df_ctd_1, np, plt, sa_array, sa_grid, sigma_grid):
    df_ctd_1['Phase'] = ' '
    df_ctd_1.loc[df_ctd_1['cast'] < 10, 'Phase'] = 'Phase 1'
    df_ctd_1.loc[(df_ctd_1['cast'] > 5) & (df_ctd_1['cast'] < 8), 'Phase'] = 'Phase 1'
    df_ctd_1.loc[(df_ctd_1['cast'] > 9) & (df_ctd_1['cast'] < 36), 'Phase'] = 'Phase 2'
    df_ctd_1.loc[df_ctd_1['cast'] > 35, 'Phase'] = 'Phase 3'
    color_dict_1 = {'Phase 1': '#5E79FD', 'Phase 2': '#FFA400', 'Phase 3': '#FF3399'}
    _fig, _axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, p_1 in enumerate(['Phase 1', 'Phase 2', 'Phase 3']):
        _df_subset = df_ctd_1[df_ctd_1['Phase'] == p_1]
        _axes[i].scatter(_df_subset.salt, _df_subset.temp, c=color_dict_1[p_1], s=10, alpha=0.4)
    for i, p_1 in enumerate(['Phase 1', 'Phase 2', 'Phase 3']):
        _df_subset = df_ctd_1[df_ctd_1['Phase'] == p_1]
        _axes[i].scatter(np.nan, np.nan, c=color_dict_1[p_1], label=p_1, s=100, alpha=1)
    ax.legend()
    for ax_1 in _axes:
        _CS = ax_1.contour(sa_grid, ct_grid, sigma_grid, colors='black', zorder=0, linewidths=1, linestyles='dashed')
        _label_levels = np.array([_CS.levels[2], _CS.levels[3], _CS.levels[6]])
        plt.clabel(_CS, _label_levels, fontsize=15, inline_spacing=10, fmt='%2.1f', zorder=-1)
        ax_1.set_ylim(ct_array[0] + 0.5, ct_array[-1] - 0.12)
        ax_1.set_xlim(sa_array[0] + 0.3, sa_array[-1] - 0.18)
        ax_1.tick_params(bottom=True, left=True)
        ax_1.set_xlabel('Absolute Salinity')
        if ax_1 != _axes[0]:
            ax_1.set_yticks([])
    _axes[0].set_ylabel('Conservative temperature ($\\degree$C)')
    return (color_dict_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculate spice anomaly:
    - First, we want to construct a reference T and S profile at interpolated density levels (say 0.1 kg/m3), by taking the median T and S at a given density level. Some smoothing can be applied if jagged
    - Then, we want to calculate the spice anomaly for a value by taking $\alpha \Delta T + \beta\Delta S$,where $\Delta T = T - T_{ref}$ and $\Delta S = S - S_{ref}$, with $T_{ref}$ and $S_{ref}$ defined as a function of $\sigma$
    """)
    return


@app.cell
def _(ecoctd, np):
    _spacing = 0.01
    density_levels = np.arange(np.round(ecoctd.CTD_Sigma0.min(), 2), np.round(ecoctd.CTD_Sigma0.max(), 2), _spacing)
    return (density_levels,)


@app.cell
def _(density_levels, ecoctd, np, scipy):
    for i_1 in range(ecoctd.dims['n_casts']):
        _cast = ecoctd.isel(n_casts=i_1)
        _sigma = _cast.CTD_Sigma0.values
        _ct = _cast.CTD_CT.values
        _sa = _cast.CTD_SA.values
        _sa_nan_ind = ~np.isnan(_sigma) & ~np.isnan(_sa)
        _ct_nan_ind = ~np.isnan(_sigma) & ~np.isnan(_sa)
        if np.sum(_ct_nan_ind) == 0:
            _interp_vals = np.zeros(len(density_levels))
            _interp_vals[:] = np.nan
        else:
            _f_temp = scipy.interpolate.interp1d(_sigma[_ct_nan_ind], _ct[_ct_nan_ind], bounds_error=False, fill_value=np.nan)
            _interp_vals = _f_temp(density_levels)
        if i_1 == 0:
            ct_interp = _interp_vals
        else:
            ct_interp = np.vstack([ct_interp, _interp_vals])
        if np.sum(_ct_nan_ind) == 0:
            _interp_vals = np.zeros(len(density_levels))
            _interp_vals[:] = np.nan
        else:
            _f_sa = scipy.interpolate.interp1d(_sigma[_sa_nan_ind], _sa[_sa_nan_ind], bounds_error=False, fill_value=np.nan)
            _interp_vals = _f_sa(density_levels)
        if i_1 == 0:
            sa_interp = _interp_vals
        else:
            sa_interp = np.vstack([sa_interp, _interp_vals])
    return ct_interp, sa_interp


@app.cell
def _(ct_interp, ecoctd, np, plt, sa_interp):
    plt.figure(figsize=(8,8))
    plt.plot(np.nanmedian(sa_interp, axis=0), np.nanmedian(ct_interp, axis=0), linewidth=3,
            linestyle='dashed', color='black')
    plt.scatter(ecoctd.CTD_SA, ecoctd.CTD_CT, s=1)
    return


@app.cell
def _(ct_interp, np, sa_interp):
    median_S = np.nanmedian(sa_interp, axis=0)
    median_T = np.nanmedian(ct_interp, axis=0)
    return median_S, median_T


@app.cell
def _(density_levels, median_S, median_T, np):
    ### Remove NaN for where we don't have enough data to calculate
    density_levels_1 = density_levels[~np.isnan(median_S)]
    median_T_1 = median_T[~np.isnan(median_S)]
    median_S_1 = median_S[~np.isnan(median_S)]
    return density_levels_1, median_S_1, median_T_1


@app.cell
def _(density_levels_1, ecoctd, gsw, median_S_1, median_T_1, np, scipy):
    _f_sa = scipy.interpolate.interp1d(density_levels_1, median_S_1, bounds_error=False, fill_value=np.nan)
    _f_ct = scipy.interpolate.interp1d(density_levels_1, median_T_1, bounds_error=False, fill_value=np.nan)
    spice = ecoctd.CTD_SA.values.copy()
    spice[:] = np.nan
    for i_2 in range(ecoctd.dims['n_casts']):
        _cast = ecoctd.isel(n_casts=i_2)
        _sigma = _cast.CTD_Sigma0.values
        _ct = _cast.CTD_CT.values
        _sa = _cast.CTD_SA.values
        _pressure = _cast.pressure.values
        _sa_nan_ind = ~np.isnan(_sigma) & ~np.isnan(_sa)
        _ct_nan_ind = ~np.isnan(_sigma) & ~np.isnan(_sa)
        _alpha = gsw.alpha(_f_sa(_sigma), _f_ct(_sigma), _pressure)
        _beta = gsw.beta(_f_sa(_sigma), _f_ct(_sigma), _pressure)
        _spice_i = _alpha * (_ct - _f_ct(_sigma)) + _beta * (_sa - _f_sa(_sigma))
        spice[i_2] = _spice_i * (_sigma + 1000)
    return (spice,)


@app.cell
def _(ecoctd, spice):
    ecoctd['Spice_anomaly'] = (('n_casts', 'n_levels'), spice)
    ecoctd['Spice_anomaly'].attrs = {'Full_name': 'Spice anomaly on an isopycnal',
                'method': r'\rho_0 $\alpha\Delta T + \beta \Delta S$ calculated from a reference T-S profile'}
    return


@app.cell
def _(ctd, density_levels_1, gsw, median_S_1, median_T_1, np, scipy):
    _f_sa = scipy.interpolate.interp1d(density_levels_1, median_S_1, bounds_error=False, fill_value=np.nan)
    _f_ct = scipy.interpolate.interp1d(density_levels_1, median_T_1, bounds_error=False, fill_value=np.nan)
    spice_1 = ctd.SA.values.copy()
    spice_1[:] = np.nan
    for i_3 in range(ctd.dims['n_casts']):
        _cast = ctd.isel(n_casts=i_3)
        _sigma = _cast.SIGMA0.values
        _ct = _cast.CT.values
        _sa = _cast.SA.values
        _pressure = _cast.PRES.values
        _sa_nan_ind = ~np.isnan(_sigma) & ~np.isnan(_sa)
        _ct_nan_ind = ~np.isnan(_sigma) & ~np.isnan(_sa)
        _alpha = gsw.alpha(_sa, _ct, _pressure)
        _beta = gsw.beta(_sa, _ct, _pressure)
        _spice_i = _alpha * (_ct - _f_ct(_sigma)) + _beta * (_sa - _f_sa(_sigma))
        spice_1[i_3] = _spice_i * (_sigma + 1000)
    ctd['Spice_anomaly'] = (('n_casts', 'n_levels'), spice_1)
    ctd['Spice_anomaly'].attrs = {'Full_name': 'Spice anomaly on an isopycnal', 'method': '\\rho_0 $\\alpha\\Delta T + \\beta \\Delta S$ calculated from a reference T-S profile'}
    return


@app.cell
def _(ctd):
    ### Plotting individual profiles

    cast_ds = ctd.isel(n_casts = ctd.CAST == 37).copy()

    #cast_ds = ctd.isel(n_casts = ctd.cast == 3).copy()
    return (cast_ds,)


@app.cell
def _(pd, resp_dir, wkdir):
    ### Getting respiration data
    log_sheet = pd.read_csv(wkdir + resp_dir + 'logs/intrusion_respiration_samples_digital_log_sheet.csv')
    # Location of metadata CSV
    log_sheet = log_sheet.rename({'Intrusion': 'Old_intrusion_assignment', 'Checked_intrusion_102524': 'Intrusion'}, axis=1)
    df_1 = log_sheet[['Cast_Niskin', 'Depth', 'Intrusion']]
    df_1 = df_1[df_1['Cast_Niskin'].str.startswith('C')]
    # 042224 -- went through and double checked all intrusions, relabeling them in the Checked_Intrusions column
    df_1 = df_1.drop_duplicates()
    # Lots of duplicates bc we have a new 
    df_1[['Cast', 'Niskin']] = df_1['Cast_Niskin'].str.split('N', expand=True)
    return (df_1,)


@app.cell
def _(CB_color_cycle, cast_ds, df_1, np, plt):
    _fig, ax_2 = plt.subplots(figsize=(5, 6), dpi=300)
    _twin1 = ax_2.twiny()
    chl_color = 'olivedrab'
    aou_color = 'orchid'
    spice_color = CB_color_cycle[5]
    _p0, = ax_2.plot(cast_ds.CHLA.values.flatten(), cast_ds.PRES.values, color=chl_color, linewidth=5, zorder=2)
    ax_2.semilogx()
    ax_2.invert_yaxis()
    _p1, = _twin1.plot(cast_ds.AOU.values.flatten(), cast_ds.PRES.values, color=aou_color, lw=5, zorder=4, linestyle='solid')
    ax_2.xaxis.label.set_color(chl_color)
    ax_2.set_xlabel('Chl ($\\mu$g L$^{-1}$)')
    ax_2.yaxis.get_ticklocs(minor=True)
    ax_2.minorticks_on()
    _twin1.xaxis.label.set_color(aou_color)
    _twin1.set_xlabel('AOU ($\\mu$mol kg$^{-1}$)')
    _twin1.yaxis.get_ticklocs(minor=True)
    _twin1.minorticks_on()
    _twin1.xaxis.set_label_coords(0.5, 1.1)
    _top_sigma = np.interp(28.91, cast_ds.SIGMA0.values[0], cast_ds.PRES)
    _bottom_sigma = np.interp(28.96, cast_ds.SIGMA0.values[0], cast_ds.PRES)
    plt.axhline(_top_sigma, color='red', zorder=0, lw=3)
    plt.axhline(_bottom_sigma, color='red', zorder=0, lw=3)
    _tkw = dict(size=4, width=2)
    ax_2.tick_params(axis='x', colors=chl_color, **_tkw)
    _twin1.tick_params(axis='x', colors=aou_color, **_tkw)
    ax_2.tick_params(axis='y', **_tkw)
    ax_2.set_zorder(ax_2.get_zorder() + 1)
    ax_2.set_frame_on(False)
    ax_2.set_ylabel('Depth (m)')
    ax_2.set_ylim(250, 5)
    ax_2.grid(lw=0.75, c='gray', alpha=0.75)
    ax_2.patch.set_edgecolor('black')
    ax_2.patch.set_linewidth(1)
    ax_2.plot([], [], lw=4, color=chl_color, label='Chl')
    ax_2.plot([], [], lw=4, color=aou_color, label='AOU', ls='solid')
    for i_4, _row in df_1[df_1['Cast'] == 'C37'].iterrows():
        _twin1.scatter(0, _row['Depth'], c='k', zorder=20, clip_on=False, marker='>', s=175)
    _twin1.set_xlim(0, 85)
    ax_2.legend(bbox_to_anchor=(0.47, 0.875), loc='center', fontsize=16, framealpha=1, facecolor='gainsboro')
    return


@app.cell
def _(ctd):
    transect = ctd.isel(n_casts = slice(11, 17))
    return (transect,)


@app.cell
def _(cmocean, colors, np, plt, sns, transect):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    _ax1.tick_params(bottom=True, left=True)
    _spice_transect = transect.Spice_anomaly.T
    _vnorm_spice = np.nanmax(np.abs(_spice_transect))
    _cf = _ax1.pcolormesh(transect.CHLA.T, cmap=cmocean.cm.speed, norm=colors.LogNorm(vmin=10 ** (-2.5), vmax=3.0))
    _ax1.set_ylim([250, 7.5])
    cbar_2 = plt.colorbar(_cf, label='Chl fluorescence (mg/m$^3$)', orientation='horizontal', ax=_ax1, extend='both')
    _ax1.contour(transect.SIGMA0.values.T, levels=np.arange(28.4, 29.1, 0.05), colors='black', linewidths=1)
    _ax1.set_xlabel('Distance along transect (km)')
    _ax1.set_ylabel('Pressure (db)')
    ax2.tick_params(bottom=True, left=True)
    _cf = ax2.pcolormesh(transect.AOU.T, cmap='PuBuGn_r', vmax=70, vmin=0)
    ax2.set_ylim([250, 7.5])
    plt.colorbar(_cf, label='AOU ($\\mu$mol/L)', orientation='horizontal', ax=ax2, extend='both')
    ax2.contour(transect.SIGMA0.values.T, levels=np.arange(28.4, 29.1, 0.05), colors='black', linewidths=1)
    ax2.set_xlabel('Distance along transect (km)')
    plt.tight_layout()

    plt.show()
    return


@app.cell
def _(ecoctd, plt):
    plt.figure(dpi=200)
    plt.plot(ecoctd.CTD_lon,ecoctd.CTD_lat, c='white', lw=2)

    plt.xticks([])
    plt.yticks([])

    plt.show()

    #plt.savefig('shiptrack_with_resp.eps', format='eps',transparent=True)
    return


@app.cell
def _(cast_ds):
    cast_ds.UTC_TIME.values
    return


@app.cell
def _(ctd, np, plt):
    cast_ds_1 = ctd.isel(n_casts=ctd.CAST == 25).copy()
    _fig, ax_3 = plt.subplots(figsize=(7.0, 7), dpi=300)
    _twin1 = ax_3.twiny()
    _twin1.spines.bottom.set_position(('axes', -0.17))
    _p0, = ax_3.plot(cast_ds_1.CHLA.values.flatten(), cast_ds_1.PRES.values, color='olivedrab', linewidth=4, zorder=2)
    ax_3.invert_yaxis()
    _p1, = _twin1.plot(cast_ds_1.AOU.values.flatten(), cast_ds_1.PRES.values, color='steelblue', lw=4, zorder=2, linestyle='dashed')
    ax_3.axhline(30)
    ax_3.xaxis.label.set_color('olivedrab')
    ax_3.set_xlabel('Chl fl (mg/m$^3$)')
    ax_3.yaxis.get_ticklocs(minor=True)
    ax_3.minorticks_on()
    _twin1.xaxis.label.set_color('steelblue')
    _twin1.set_xlabel('AOU ($\\mu$M)')
    _twin1.yaxis.get_ticklocs(minor=True)
    _twin1.minorticks_on()
    _twin1.xaxis.set_label_coords(0.5, -0.28)
    _top_sigma = np.interp(28.91, cast_ds_1.SIGMA0.values[0], cast_ds_1.PRES)
    _bottom_sigma = np.interp(28.96, cast_ds_1.SIGMA0.values[0], cast_ds_1.PRES)
    _tkw = dict(size=4, width=2)
    ax_3.tick_params(axis='x', colors='olivedrab', **_tkw)
    _twin1.tick_params(axis='x', colors='steelblue', **_tkw)
    ax_3.tick_params(axis='y', **_tkw)
    ax_3.set_zorder(ax_3.get_zorder() + 1)
    ax_3.set_frame_on(False)
    ax_3.set_ylabel('Depth (m)')
    ax_3.set_ylim(300, 5)
    ax_3.grid(lw=0.5, c='gray', alpha=0.25)
    ax_3.patch.set_edgecolor('black')
    ax_3.patch.set_linewidth(1)
    ax_3.plot([], [], lw=4, color='olivedrab', label='Chl')
    ax_3.plot([], [], lw=4, color='steelblue', label='AOU', ls='dashed')
    ax_3.legend(bbox_to_anchor=(0.155, 0.56), loc='center', fontsize=17)
    return (cast_ds_1,)


@app.cell
def _(cast_ds_1, plt):
    _fig, ax_4 = plt.subplots(figsize=(7.0, 7), dpi=100)
    _twin1 = ax_4.twiny()
    twin2 = _twin1.twiny()
    twin3 = twin2.twiny()
    _twin1.spines.bottom.set_position(('axes', -0.17))
    twin2.spines.bottom.set_position(('axes', 1.22))
    _p0, = ax_4.plot(cast_ds_1.CHLA.values.flatten(), cast_ds_1.PRES.values, color='olivedrab', linewidth=4, zorder=0, linestyle=(0, (5, 1)))
    ax_4.semilogx()
    ax_4.invert_yaxis()
    _p1, = _twin1.plot(cast_ds_1.AOU.values.flatten(), cast_ds_1.PRES.values, color='steelblue', lw=4, zorder=2, linestyle='dashed')
    p2 = twin2.plot(cast_ds_1.SA.values.flatten(), cast_ds_1.PRES.values, color='orange', lw=4, zorder=2, linestyle='dashdot')
    p3 = twin3.plot(cast_ds_1.NITRATE.values.flatten(), cast_ds_1.PRES.values, color='darkmagenta', lw=5, zorder=3, linestyle='solid')
    ax_4.set_zorder(1)
    _twin1.set_zorder(4)
    twin2.set_zorder(4)
    twin3.set_zorder(5)
    ax_4.set_frame_on(False)
    ax_4.xaxis.label.set_color('olivedrab')
    ax_4.set_xlabel('Chl fl (mg/m$^3$)')
    ax_4.yaxis.get_ticklocs(minor=True)
    ax_4.minorticks_on()
    _twin1.xaxis.label.set_color('steelblue')
    _twin1.set_xlabel('AOU ($\\mu$M)')
    _twin1.yaxis.get_ticklocs(minor=True)
    _twin1.minorticks_on()
    _twin1.xaxis.set_label_coords(0.5, -0.28)
    twin2.xaxis.label.set_color('orange')
    twin2.set_xlabel('Absolute Salinity')
    twin2.yaxis.get_ticklocs(minor=True)
    twin2.minorticks_on()
    twin2.xaxis.set_label_coords(0.5, 1.26)
    twin3.xaxis.label.set_color('darkmagenta')
    twin3.set_xlabel('Nitrate ($\\mu$M)')
    twin3.yaxis.get_ticklocs(minor=True)
    twin3.minorticks_on()
    _tkw = dict(size=4, width=2)
    ax_4.tick_params(axis='x', colors='olivedrab', **_tkw)
    _twin1.tick_params(axis='x', colors='steelblue', **_tkw)
    twin2.tick_params(axis='x', colors='orange', **_tkw)
    twin3.tick_params(axis='x', colors='darkmagenta', **_tkw)
    ax_4.tick_params(axis='y', **_tkw)
    ax_4.set_zorder(ax_4.get_zorder() + 1)
    ax_4.set_frame_on(False)
    ax_4.set_ylabel('Depth (m)')
    ax_4.set_ylim(300, 5)
    ax_4.grid(lw=0.5, c='gray', alpha=0.25)
    ax_4.patch.set_edgecolor('black')
    ax_4.patch.set_linewidth(1)
    ax_4.plot([], [], lw=4, color='olivedrab', label='Chl', ls=(0, (5, 1)))
    ax_4.plot([], [], lw=4, color='steelblue', label='AOU', ls='dashed')
    ax_4.plot([], [], lw=4, color='orange', label='SA', ls='dashdot')
    ax_4.plot([], [], lw=4, color='darkmagenta', label='NO$_3$', ls='solid')
    ax_4.legend(bbox_to_anchor=(0.155, 0.52), loc='center', fontsize=17)
    return


@app.cell
def _(pd, wkdir):
    transects_df = pd.read_csv(wkdir + 'Data/Metadata/EcoCTD_transects.csv')
    transects_df['Initial_Time'] = pd.to_datetime(transects_df.Initial_Time)
    transects_df['Final_Time'] = pd.to_datetime(transects_df.Final_Time)
    transects_df.head(6)
    return (transects_df,)


@app.cell
def _(ecoctd, pd, transects_df):
    transect_i = 58 # we know this is a good one
    time_ind = (pd.to_datetime(ecoctd.UTC_time.values) >= transects_df.iloc[transect_i]['Initial_Time']) & (pd.to_datetime(ecoctd.UTC_time.values) <= transects_df.iloc[transect_i]['Final_Time'])
    return (time_ind,)


@app.cell
def _(np, time_ind):
    sl = np.where(time_ind)[0][0], np.where(time_ind)[0][-1]+1
    return (sl,)


@app.cell
def _(ecoctd, haversine_np, np, sl):
    transect_1 = ecoctd.isel(n_casts=slice(sl[0], sl[1]))
    dates = transect_1.UTC_time.values
    pres_grid = np.repeat(transect_1.pressure.values, transect_1.dims['n_casts']).reshape((transect_1.dims['n_levels'], transect_1.dims['n_casts']))
    _start_lon, _start_lat = (transect_1.isel(n_casts=0)['CTD_lon'].values, transect_1.isel(n_casts=0)['CTD_lat'].values)
    dists = haversine_np(_start_lon, _start_lat, transect_1['CTD_lon'].values[:], transect_1['CTD_lat'].values[:])
    dist_grid = np.tile(dists, pres_grid.shape[0]).reshape(pres_grid.shape)
    return dist_grid, dists, pres_grid, transect_1


@app.cell
def _(cmocean, colors, dist_grid, dists, np, plt, pres_grid, sns, transect_1):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2_1) = plt.subplots(1, 2, figsize=(18, 7))
    _ax1.tick_params(bottom=True, left=True)
    _spice_transect = transect_1.Spice_anomaly.T
    _vnorm_spice = np.nanmax(np.abs(_spice_transect))
    _cf = _ax1.pcolormesh(dist_grid, pres_grid, transect_1.FLS_chl.T, cmap=cmocean.cm.speed, norm=colors.LogNorm(vmin=10 ** (-2.5), vmax=10.0))
    _ax1.set_ylim([200, 7.5])
    _ax1.set_xlim([0, np.nanmax(dists)])
    cbar_3 = plt.colorbar(_cf, label='Chl fluorescence (mg/m$^3$)', orientation='horizontal', ax=_ax1, extend='both')
    _ax1.contour(dist_grid, pres_grid, transect_1.CTD_Sigma0.values.T, levels=np.arange(28.4, 29.1, 0.05), colors='black', linewidths=1)
    _ax1.set_xlabel('Distance along transect (km)')
    _ax1.set_ylabel('Pressure (db)')
    ax2_1.tick_params(bottom=True, left=True)
    _cf = ax2_1.pcolormesh(dist_grid, pres_grid, transect_1.AOU.T, cmap='PuBuGn_r', vmax=70, vmin=0)
    ax2_1.set_ylim([200, 7.5])
    ax2_1.set_xlim([0, np.nanmax(dists)])
    plt.colorbar(_cf, label='AOU ($\\mu$mol/L)', orientation='horizontal', ax=ax2_1, extend='both')
    ax2_1.contour(dist_grid, pres_grid, transect_1.CTD_Sigma0.values.T, levels=np.arange(28.4, 29.1, 0.05), colors='black', linewidths=1)
    ax2_1.set_xlabel('Distance along transect (km)')
    plt.tight_layout()
    return


@app.cell
def _(cmocean, colors, dist_grid, dists, np, plt, pres_grid, sns, transect_1):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2_2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300)
    _ax1.tick_params(bottom=True, left=True)
    _spice_transect = transect_1.Spice_anomaly.T
    _vnorm_spice = np.nanmax(np.abs(_spice_transect))
    _cf = _ax1.contourf(dist_grid, pres_grid, transect_1.FLS_chl.T, cmap=cmocean.cm.speed, norm=colors.LogNorm(vmin=10 ** (-2.5), vmax=10.0), levels=np.logspace(-3.0, 1, 12), extend='both')
    _ax1.set_ylim([200, 7.5])
    _ax1.set_xlim([0, np.nanmax(dists)])
    cbar_4 = plt.colorbar(_cf, label='Chl fluorescence (mg/m$^3$)', orientation='horizontal', ax=_ax1, extend='both')
    _CS = _ax1.contour(dist_grid, pres_grid, transect_1.CTD_Sigma0.values.T, levels=np.arange(28.4, 29.1, 0.05), colors='black', linewidths=1)
    plt.clabel(_CS, _CS.levels[::2], inline=True, fontsize=14)
    _ax1.set_xlabel('Distance along transect (km)')
    _ax1.set_ylabel('Pressure (db)')
    ax2_2.tick_params(bottom=True, left=True)
    _cf = ax2_2.contourf(dist_grid, pres_grid, transect_1.AOU.T, cmap='YlGnBu', vmax=70, vmin=-10, levels=15, extend='both')
    ax2_2.set_ylim([200, 7.5])
    ax2_2.set_xlim([0, np.nanmax(dists)])
    plt.colorbar(_cf, label='AOU ($\\mu$mol/L)', orientation='horizontal', ax=ax2_2, extend='both')
    _CS = ax2_2.contour(dist_grid, pres_grid, transect_1.CTD_Sigma0.values.T, levels=np.arange(28.4, 29.1, 0.05), colors='black', linewidths=1)
    plt.clabel(_CS, _CS.levels[::2], inline=True, fontsize=14)
    ax2_2.set_xlabel('Distance along transect (km)')
    plt.tight_layout()
    return


@app.cell
def _(np, transect_1):
    AOU_interp = np.zeros(transect_1.AOU.shape)
    AOU_interp[:] = np.nan
    for i_5 in range(transect_1.sizes['n_casts']):
        _nan_ind = np.isnan(transect_1.isel(n_casts=i_5).AOU.values)
        AOU_interp[i_5] = np.interp(transect_1.CTD_SeaPress, transect_1.CTD_SeaPress[~_nan_ind], transect_1.isel(n_casts=i_5).AOU.values[~_nan_ind], left=np.nan, right=np.nan)
    return (AOU_interp,)


@app.cell
def _(AOU_interp, dist_grid, dists, np, plt, pres_grid, sns, transect_1):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, _ax1 = plt.subplots(1, figsize=(9, 7), dpi=300)
    _ax1.tick_params(bottom=True, left=True)
    _cf = _ax1.contourf(dist_grid, pres_grid, AOU_interp.T, cmap='YlGnBu', vmax=60, vmin=-5, levels=10, extend='both')
    _ax1.set_ylim([200, 7.5])
    _ax1.set_xlim([0, np.nanmax(dists)])
    plt.colorbar(_cf, label='AOU ($\\mu$mol/L)', orientation='horizontal', ax=_ax1, extend='both')
    _CS = _ax1.contour(dist_grid, pres_grid, transect_1.CTD_Sigma0.values.T, levels=np.arange(28.4, 29.1, 0.075), colors='black', linewidths=1)
    _ax1.set_xlabel('Distance along transect (km)')
    plt.tight_layout()
    return


@app.cell
def _(mpl):
    mpl.rcParams['xtick.labelsize'] = 'medium'
    return


@app.cell
def _():
    ### Just nav-ing thru a lot of satellite data to see if we can get one good image that lines up with Cast 49
    return


@app.cell
def _(glob):
    gdrive_sat = '/Users/katyabbott/Google Drive/Shared drives/CalypsoShare/2022_Experiment/2022_Data_and_Products/Satellite_Data/raw_data/'
    glob.glob(gdrive_sat+'*')
    return (gdrive_sat,)


@app.cell
def _(gdrive_sat, glob):
    chl_files = glob.glob(gdrive_sat + 'CHL-S3/02/20220217*')

    print(chl_files)
    return (chl_files,)


@app.cell
def _(chl_files, np, xr):
    _chl_ds = xr.open_dataset(chl_files[0])
    # here's the metadata for this: https://data.marine.copernicus.eu/product/OCEANCOLOUR_MED_BGC_L3_MY_009_143/description
    chl = _chl_ds.isel(time=0).CHL
    # log_chl = np.log10(chl_ds.isel(time=0).CHL)
    chl = chl.sel(lon=slice(-4, 10), lat=slice(34, 44))
    # log_chl = log_chl.sel(lon = slice(-4, 10), lat=slice(34, 44))
    # log_chl_data = log_chl.values
    lon_grid, lat_grid = np.meshgrid(chl.lon.values, chl.lat.values)
    chl = chl.values
    return chl, lat_grid, lon_grid


@app.cell
def _(xr):
    socib_ds = xr.open_dataset('/Users/katyabbott/Downloads/dep0022_socib-rv_scb-sbe9001_L1_corr_2022-02-16_data_dm.nc')
    return (socib_ds,)


@app.cell
def _(cartopy, ccrs, chl, lat_grid, lon_grid, mticker, plt, socib_ds):
    plt.figure(figsize=(10, 6))
    ax_5 = plt.axes(projection=ccrs.PlateCarree())
    ax_5.coastlines()
    ax_5.add_feature(cartopy.feature.LAND, edgecolor='black', color='lightgray')
    ax_5.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_5.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([2, 3, 4, 5])
    _gl.ylocator = mticker.FixedLocator([39, 40, 41, 42, 43])
    _gl.top_labels = False
    _gl.right_labels = False
    ax_5.set_extent([0, 3, 38, 41])
    plt.pcolormesh(lon_grid, lat_grid, chl, cmap='viridis', zorder=6, vmin=0, vmax=0.3)
    plt.colorbar(label='Chlorophyll', extend='both')
    plt.scatter(socib_ds.LON, socib_ds.LAT, s=20, color='red', zorder=10)
    return


@app.cell
def _(glob, wkdir):
    glob.glob(wkdir + 'Data/Satellite/*')
    return


@app.cell
def _(glob, np, wkdir, xr):
    i_6 = 2
    chl_files_1 = sorted(glob.glob(wkdir + 'Data/Satellite/Chl/For_calibration/Multi/*'))
    _chl_ds = xr.open_dataset(chl_files_1[i_6])
    _log_chl = np.log10(_chl_ds.isel(time=0).CHL)
    _log_chl = _log_chl.sel(lon=slice(-4, 10), lat=slice(34, 44))
    log_chl_data = _log_chl.values
    chl_1 = _chl_ds.isel(time=0).CHL
    chl_1 = chl_1.sel(lon=slice(-4, 10), lat=slice(34, 44))
    chl_1 = chl_1.values
    lon_grid_1, lat_grid_1 = np.meshgrid(_log_chl.lon.values, _log_chl.lat.values)
    print(chl_files_1[i_6])
    return chl_files_1, lat_grid_1, log_chl_data, lon_grid_1


@app.cell
def _(cartopy, ccrs, lat_grid_1, log_chl_data, lon_grid_1, mticker, pe, plt):
    _fig = plt.figure(figsize=(8, 12))
    ax_6 = plt.axes(projection=ccrs.PlateCarree())
    ax_6.coastlines()
    ax_6.add_feature(cartopy.feature.LAND, edgecolor='black', color='tan')
    ax_6.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_6.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.75, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([2, 3, 4, 5])
    _gl.ylocator = mticker.FixedLocator([40.5, 41, 41.5, 42])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_6.set_extent([2.25, 4.5, 40.25, 41.85])
    ax_6.pcolormesh(lon_grid_1, lat_grid_1, log_chl_data, cmap='viridis', zorder=3, vmin=-0.95, vmax=-0.15, alpha=0.9)
    plt.scatter(5.3698, 43.2965, zorder=5, color='red', s=50)
    plt.scatter(2.1686, 41.3874, zorder=5, color='red', s=50)
    ax_6.text(2.302186, 41.7374, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
    for spine in ax_6.spines.values():
        spine.set_edgecolor('red')
    _left, _bottom, _width, _height = [0.27, 0.68, 0.5, 0.25]
    ax2_3 = _fig.add_axes([_left, _bottom, _width, _height], projection=ccrs.PlateCarree())
    ax2_3.coastlines()
    _resol = '50m'
    _bodr = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=_resol, facecolor='none', alpha=0.7)
    _rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=_resol, edgecolor='b', facecolor='none')
    ax2_3.add_feature(_bodr, linestyle='-', edgecolor='k', alpha=1)
    ax2_3.add_feature(_rivers, linewidth=1)
    ax2_3.add_feature(cartopy.feature.LAND, edgecolor='black', color='tan')
    ax2_3.add_feature(cartopy.feature.OCEAN, color='lightblue')
    ax2_3.set_extent([-7, 16, 34, 44.5])
    _gl = ax2_3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.75, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([-5, 0, 5, 10, 15])
    _gl.ylocator = mticker.FixedLocator([35, 40])
    _gl.top_labels = False
    _gl.right_labels = False
    ax2_3.plot([2.25, 4.5, 4.5, 2.25, 2.25], [40.25, 40.25, 41.75, 41.75, 40.25], color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)


    plt.show()
    return


@app.cell
def _(chl_files_1, np, xr):
    _chl_ds = xr.open_dataset(chl_files_1[13])
    print(chl_files_1[13])
    _log_chl = np.log10(_chl_ds.isel(time=0).CHL)
    #_log_chl = _log_chl.sel(lon=slice(-4, 10), lat=slice(34, 44))
    log_chl_data_1 = _log_chl.values
    chl_2 = _chl_ds.isel(time=0).CHL
    #chl_2 = chl_2.sel(lon=slice(-4, 10), lat=slice(34, 44))
    chl_2 = chl_2.values
    lon_grid_2, lat_grid_2 = np.meshgrid(_log_chl.lon.values, _log_chl.lat.values)
    return chl_2, lat_grid_2, log_chl_data_1, lon_grid_2


@app.cell
def _(
    cartopy,
    ccrs,
    ctd,
    ecoctd,
    lat_grid_2,
    log_chl_data_1,
    lon_grid_2,
    mticker,
    pe,
    plt,
):
    plt.figure(figsize=(8, 12))
    ax_7 = plt.axes(projection=ccrs.PlateCarree())
    ax_7.coastlines()
    ax_7.add_feature(cartopy.feature.LAND, edgecolor='black', color='lightgray')
    ax_7.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_7.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([-6, -4, -2, 0, 2, 4, 6, 8, 10])
    _gl.ylocator = mticker.FixedLocator([34, 36, 38, 40, 42, 44])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_7.set_extent([0, 8, 36, 44])
    ax_7.pcolormesh(lon_grid_2, lat_grid_2, log_chl_data_1, cmap='viridis', zorder=3, vmin=-1.25, vmax=0.1)
    ax_7.scatter(ecoctd.CTD_lon, ecoctd.CTD_lat, s=2, color='#FAAF07', zorder=5)
    ax_7.scatter(ctd.LON, ctd.LAT, color='black', s=50, marker='*', zorder=5)
    ax_7.plot([2, 4.5, 4.5, 2, 2], [40, 40, 42, 42, 40], color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    plt.scatter(5.3698, 43.2965, zorder=5, color='red', s=50)
    plt.scatter(2.1686, 41.3874, zorder=5, color='red', s=50)
    ax_7.text(4.4698, 43.4465, 'Marseilles', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], fontsize=18)
    ax_7.text(1.0186, 41.5874, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
    plt.tight_layout()
    return


@app.cell
def _(ctd, pd, wkdir):
    output_dir = wkdir + 'Analyses/Respiration/Output/'
    resp_df_pooled = pd.read_csv(output_dir + 'CALYPSO_all_respiration_rates_QC_with_environmental_data_and_calibrated_Chl_pooled_093025.csv')
    resp_lons = {'Cast1': [], 'ACM': [], 'Background': [], 'Cast37': []}
    resp_lats = {'Cast1': [], 'ACM': [], 'Background': [], 'Cast37': []}
    for _row in resp_df_pooled.iterrows():
        _cast = _row[1]['Cast_number']
        cast_ds_2 = ctd.isel(n_casts=int(_cast) - 1)
        if _cast == 1:
            resp_lons['Cast1'].append(cast_ds_2['LON'].values)
            resp_lats['Cast1'].append(cast_ds_2['LAT'].values)
        elif _row[1]['Intrusion_from_anomaly'] == 1:
            resp_lons['ACM'].append(cast_ds_2['LON'].values)
            resp_lats['ACM'].append(cast_ds_2['LAT'].values)
        elif _row[1]['Intrusion_from_anomaly'] == 0:
            resp_lons['Background'].append(cast_ds_2['LON'].values)
            resp_lats['Background'].append(cast_ds_2['LAT'].values)
    return resp_lats, resp_lons


@app.cell
def _(ecoctd, plt, resp_lats, resp_lons):
    plt.figure(dpi=200)
    plt.plot(ecoctd.CTD_lon,ecoctd.CTD_lat, c='white', lw=2)
    plt.scatter(resp_lons['Background'], 
                resp_lats['Background'], zorder=10, marker='o', c='orange',
                s=200)

    plt.scatter(resp_lons['ACM'], 
                resp_lats['ACM'], zorder=10, marker='d', c='deeppink', edgecolor='k',
                s=250)


    plt.xticks([])
    plt.yticks([])

    #plt.savefig('shiptrack_with_resp.eps', format='eps',transparent=True)
    return


@app.cell
def _():
    # ### Getting respiration data

    # # Location of metadata CSV
    # log_sheet = pd.read_csv(wkdir + resp_dir + 'logs/intrusion_respiration_samples_digital_log_sheet.csv')



    # log_sheet = log_sheet.rename({"Intrusion": "Old_intrusion_assignment", "Checked_intrusion_102524": "Intrusion"}, axis=1)
    # # 042224 -- went through and double checked all intrusions, relabeling them in the Checked_Intrusions column

    # df = log_sheet[['Cast_Niskin', 'Depth', 'Intrusion']]
    # df = df[df['Cast_Niskin'].str.startswith('C')]

    # # Lots of duplicates bc we have a new 
    # df = df.drop_duplicates()

    # df[['Cast', 'Niskin']] = df['Cast_Niskin'].str.split('N', expand=True)

    # ### Linking CTD data

    # #### Keep just the cast + bottle information, the depth (so we can extract it from the CTD data), and whether it came from an intrusion (based on anomaly in Chl only -- not looking at high oxygen low chl anomalies just yet)

    # resp_lons = {'Cast1': [], 'ACM': [], 'Background': [], 'Cast37' : []}
    # resp_lats = {'Cast1': [], 'ACM': [], 'Background': [], 'Cast37': []}


    # for row in df.iterrows():
    #     cast = row[1]['Cast'][1:]
    #     cast_ds = ctd.isel(n_casts = int(cast) - 1)

    #     if cast ==  1:
    #         resp_lons['Cast1'].append(cast_ds['LON'].values)
    #         resp_lats['Cast1'].append(cast_ds['LAT'].values) 
    #     # elif cast ==  '37':
    #     #     resp_lons['Cast37'].append(cast_ds['CTD_lon'].values)
    #     #     resp_lats['Cast37'].append(cast_ds['CTD_lat'].values)   

    #     elif row[1]['Intrusion'] == 'Y':
    #         resp_lons['ACM'].append(cast_ds['LON'].values)
    #         resp_lats['ACM'].append(cast_ds['LAT'].values)  
    #     elif row[1]['Intrusion'] == 'N':
    #         resp_lons['Background'].append(cast_ds['LON'].values)
    #         resp_lats['Background'].append(cast_ds['LAT'].values)
    return


@app.cell
def _(np, resp_lats, resp_lons):
    resp_lons_all = np.concatenate([resp_lons['Cast1'], resp_lons['ACM'], resp_lons['Background']])
    resp_lats_all = np.concatenate([resp_lats['Cast1'], resp_lats['ACM'], resp_lats['Background']])
    return


@app.cell
def _(df_ctd_1):
    df_ctd_1.loc[df_ctd_1['cast'] < 10, 'Phase'] = 'Phase 1'
    df_ctd_1.loc[(df_ctd_1['cast'] > 5) & (df_ctd_1['cast'] < 8), 'Phase'] = 'Phase 1'
    df_ctd_1.loc[(df_ctd_1['cast'] > 9) & (df_ctd_1['cast'] < 36), 'Phase'] = 'Phase 2'
    df_ctd_1.loc[df_ctd_1['cast'] > 35, 'Phase'] = 'Phase 3'
    return


@app.cell
def _(ctd):
    ctd_p1 = ctd.isel(n_casts = ctd.CAST < 10)

    ctd_p2 = ctd.isel(n_casts = (ctd.CAST > 9) & (ctd.CAST < 36))

    ctd_p3 = ctd.isel(n_casts = ctd.CAST > 35)
    return ctd_p1, ctd_p2, ctd_p3


@app.cell
def _(ctd_p1, ctd_p2, ctd_p3):
    epsilon = 0.05 # adjusting bounds of region 1 bc it's less well defined to begin with

    p1_lons = (ctd_p1.LON.min() - epsilon, ctd_p1.LON.max() + epsilon)
    p1_lats = (ctd_p1.LAT.min() - epsilon, ctd_p1.LAT.max() + epsilon)

    p2_lons = (ctd_p2.LON.min()- epsilon, ctd_p2.LON.max()+ epsilon)
    p2_lats = (ctd_p2.LAT.min()- epsilon, ctd_p2.LAT.max()+ epsilon)

    p3_lons = (ctd_p3.LON.min()- epsilon, ctd_p3.LON.max()+ epsilon)
    p3_lats = (ctd_p3.LAT.min()- epsilon, ctd_p3.LAT.max()+ epsilon)
    return p1_lats, p1_lons, p2_lats, p2_lons, p3_lats, p3_lons


@app.cell
def _(ecoctd, pd):
    phase1 = ecoctd.UTC_time < pd.to_datetime('2022-02-21T08:30:00')
    phase2 = ((ecoctd.UTC_time > pd.to_datetime('2022-02-22T07:25:00')) & 
                (ecoctd.UTC_time < pd.to_datetime('2022-03-01T08:30:00')))
    phase3 = ((ecoctd.UTC_time > pd.to_datetime('2022-03-01T08:30:00')) & 
                (ecoctd.UTC_time < pd.to_datetime('2022-03-10T10:40:00')))

    ecoctd_p1 = ecoctd.isel(n_casts = phase1)
    ecoctd_p2 = ecoctd.isel(n_casts = phase2)
    ecoctd_p3 = ecoctd.isel(n_casts = phase3)
    return


@app.cell
def _():
    # # this is from EcoCTD
    # # p1_lons = (ecoctd_p1.CTD_lon.min(), ecoctd_p1.CTD_lon.max())
    # # p1_lats = (ecoctd_p1.CTD_lat.min(), ecoctd_p1.CTD_lat.max())

    # p2_lons = (ecoctd_p2.CTD_lon.min(), ecoctd_p2.CTD_lon.max())
    # p2_lats = (ecoctd_p2.CTD_lat.min(), ecoctd_p2.CTD_lat.max())

    # p3_lons = (ecoctd_p3.CTD_lon.min(), ecoctd_p3.CTD_lon.max())
    # p3_lats = (ecoctd_p3.CTD_lat.min(), ecoctd_p3.CTD_lat.max())
    return


@app.cell
def _(
    cartopy,
    ccrs,
    p1_lats,
    p1_lons,
    p2_lats,
    p2_lons,
    p3_lats,
    p3_lons,
    plt,
):
    _fig = plt.figure(figsize=(8, 12))
    ax_8 = plt.axes(projection=ccrs.PlateCarree())
    ax_8.coastlines()
    ax_8.add_feature(cartopy.feature.LAND, edgecolor='black', color='tan')
    ax_8.add_feature(cartopy.feature.OCEAN, color='white')
    ax_8.set_extent([2.25, 4.5, 40.25, 41.85])
    ax_8.plot([p3_lons[0], p3_lons[1], p3_lons[1], p3_lons[0], p3_lons[0]], [p3_lats[0], p3_lats[0], p3_lats[1], p3_lats[1], p3_lats[0]], color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    ax_8.plot([p1_lons[0], p1_lons[1], p1_lons[1], p1_lons[0], p1_lons[0]], [p1_lats[0], p1_lats[0], p1_lats[1], p1_lats[1], p1_lats[0]], color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    ax_8.plot([p2_lons[0], p2_lons[1], p2_lons[1], p2_lons[0], p2_lons[0]], [p2_lats[0], p2_lats[0], p2_lats[1], p2_lats[1], p2_lats[0]], color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    return


@app.cell
def _():
    from matplotlib import ticker

    return (ticker,)


@app.cell
def _(
    LogNorm,
    cartopy,
    ccrs,
    chl_2,
    lat_grid_2,
    lon_grid_2,
    mticker,
    p1_lats,
    p1_lons,
    p2_lats,
    p2_lons,
    p3_lats,
    p3_lons,
    pe,
    plt,
    resp_lats,
    resp_lons,
    ticker,
):
    _fig = plt.figure(figsize=(8, 12))
    ax_9 = plt.axes(projection=ccrs.PlateCarree())
    ax_9.coastlines()
    ax_9.add_feature(cartopy.feature.LAND, edgecolor='black', color='tan')
    ax_9.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_9.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.75, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([2, 3, 4, 5])
    _gl.ylocator = mticker.FixedLocator([40.5, 41, 41.5, 42])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_9.set_extent([2.1, 4.5, 40.25, 41.85])
    cp_1 = ax_9.pcolormesh(lon_grid_2, lat_grid_2, chl_2, cmap='viridis', norm=LogNorm(vmin=0.1, vmax=0.7), zorder=3, alpha=0.9)
    ax_9.scatter(resp_lons['Cast1'], resp_lats['Cast1'], zorder=12, s=280, marker='*', color='orchid', label='Cast 1', edgecolor='black')
    ax_9.scatter(resp_lons['ACM'], resp_lats['ACM'], zorder=11, s=115, marker='d', color='orange', label='Intrusion', edgecolor='black')
    ax_9.scatter(resp_lons['Background'], resp_lats['Background'], zorder=10, s=95, marker='o', color='slateblue', label='Background', edgecolor='white')
    ax_9.legend(bbox_to_anchor=(0.83, 0.16), loc='center', fontsize=14, framealpha=1.0)
    plt.scatter(2.1686, 41.3874, zorder=5, color='white', marker='^', s=200, edgecolor='black')
    ax_9.text(2.12186, 41.475, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
    _left, _bottom, _width, _height = [0.27, 0.68, 0.5, 0.25]
    ax2_4 = _fig.add_axes([_left, _bottom, _width, _height], projection=ccrs.PlateCarree())
    ax2_4.coastlines()
    _resol = '50m'
    _bodr = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=_resol, facecolor='none', alpha=0.7)
    _rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=_resol, edgecolor='b', facecolor='none')
    ax2_4.add_feature(_bodr, linestyle='-', edgecolor='k', alpha=1)
    ax2_4.add_feature(_rivers, linewidth=1)
    ax2_4.add_feature(cartopy.feature.LAND, edgecolor='black', color='tan')
    ax2_4.add_feature(cartopy.feature.OCEAN, color='lightblue')
    ax2_4.set_extent([-7, 16, 34, 44.5])
    _gl = ax2_4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.75, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([-5, 0, 5, 10, 15])
    _gl.ylocator = mticker.FixedLocator([35, 40])
    _gl.top_labels = False
    _gl.right_labels = False
    ax2_4.plot([2.25, 4.5, 4.5, 2.25, 2.25], [40.25, 40.25, 41.75, 41.75, 40.25], color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    ax_9.plot([p1_lons[0], p1_lons[1], p1_lons[1], p1_lons[0], p1_lons[0]], [p1_lats[0], p1_lats[0], p1_lats[1], p1_lats[1], p1_lats[0]], color='red', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
    ax_9.plot([p2_lons[0], p2_lons[1], p2_lons[1], p2_lons[0], p2_lons[0]], [p2_lats[0], p2_lats[0], p2_lats[1], p2_lats[1], p2_lats[0]], color='red', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
    ax_9.plot([p3_lons[0], p3_lons[1], p3_lons[1], p3_lons[0], p3_lons[0]], [p3_lats[0], p3_lats[0], p3_lats[1], p3_lats[1], p3_lats[0]], color='red', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
    ax_9.text(2.4, 40.6, 'R2', color='red', fontsize=17, weight='bold')
    ax_9.text(2.44, 40.965, 'R3', color='red', fontsize=17, weight='bold')
    ax_9.text(3.71, 41.59, 'R1', color='red', fontsize=17, weight='bold')
    cax = ax_9.inset_axes([1.02, 0.0, 0.03, 1])
    cb = _fig.colorbar(cp_1, cax=cax, orientation='vertical', extend='both', ticks=[0.1, 0.2, 0.3, 0.4, 0.6], label='Chl ($\\mu$g L$^{-1}$)')
    cb.ax.tick_params(labelsize=10)
    cb.formatter = ticker.StrMethodFormatter('{x:,.1f}')

    plt.show()
    # plt.savefig('../../../Manuscript/Figures/regional_map_ecoCTD_CTD_with_Chl_respiration_annotations.png', bbox_inches='tight', dpi=300)
    return


@app.cell
def _(xr):
    monthly_ocean_wkdir = '/Users/katyabbott/Documents/MIT_WHOI_Joint_Program/Research_Projects/Microbial_respiration_CALYPSO/Data/Satellite/Chl/2022_CMEMS_Med_BGC_Monthly_OceanColor/'

    march_chl_ds = xr.open_dataset(monthly_ocean_wkdir + '20220301-20220331_cmems_obs-oc_med_bgc-plankton_my_l4-multi-1km_P1M.nc')

    feb_chl_ds = xr.open_dataset(monthly_ocean_wkdir + '20220201-20220228_cmems_obs-oc_med_bgc-plankton_my_l4-multi-1km_P1M.nc')

    monthly_chl_ds = march_chl_ds.copy() # choosing this month for plotting
    return (monthly_chl_ds,)


@app.cell
def _(
    LogNorm,
    cartopy,
    ccrs,
    chl_2,
    lat_grid_2,
    lon_grid_2,
    monthly_chl_ds,
    mticker,
    p1_lats,
    p1_lons,
    p2_lats,
    p2_lons,
    p3_lats,
    p3_lons,
    pe,
    plt,
    resp_lats,
    resp_lons,
    ticker,
):
    def _():
        _fig = plt.figure(figsize=(8, 12), dpi=300)
        ax_9 = plt.axes(projection=ccrs.PlateCarree())
        ax_9.coastlines()
        ax_9.add_feature(cartopy.feature.LAND, edgecolor='black', color='tan')
        ax_9.add_feature(cartopy.feature.OCEAN, color='white')
        _gl = ax_9.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.75, linestyle='--')
        _gl.xlocator = mticker.FixedLocator([2, 3, 4, 5])
        _gl.ylocator = mticker.FixedLocator([40.5, 41, 41.5, 42])
        _gl.top_labels = False
        _gl.right_labels = False
        _gl.xlabel_style = {'color': 'black'}
        _gl.xlabel_style = {'color': 'black'}
        ax_9.set_extent([2.1, 4.5, 40.25, 41.85])
        cp_1 = ax_9.pcolormesh(lon_grid_2, lat_grid_2, chl_2, cmap='viridis', norm=LogNorm(vmin=0.1, vmax=0.7), zorder=3, alpha=0.9)
        ax_9.scatter(resp_lons['Cast1'], resp_lats['Cast1'], zorder=12, s=280, marker='*', color='orchid', label='Cast 1', edgecolor='black')
        ax_9.scatter(resp_lons['ACM'], resp_lats['ACM'], zorder=11, s=115, marker='d', color='orange', label='Intrusion', edgecolor='black')
        ax_9.scatter(resp_lons['Background'], resp_lats['Background'], zorder=10, s=95, marker='o', color='slateblue', label='Background', edgecolor='white')
        ax_9.legend(bbox_to_anchor=(0.83, 0.16), loc='center', fontsize=14, framealpha=1.0)
        plt.scatter(2.1686, 41.3874, zorder=5, color='white', marker='^', s=200, edgecolor='black')
        ax_9.text(2.12186, 41.475, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
        _left, _bottom, _width, _height = [0.19, 0.68, 0.65, 0.25]
        ax2_4 = _fig.add_axes([_left, _bottom, _width, _height], projection=ccrs.PlateCarree())
        ax2_4.coastlines()
        _resol = '50m'
        _bodr = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=_resol, facecolor='none', alpha=0.7)
        _rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=_resol, edgecolor='b', facecolor='none')
        ax2_4.add_feature(_bodr, linestyle='-', edgecolor='k', alpha=1)
        ax2_4.add_feature(_rivers, linewidth=1)
        ax2_4.add_feature(cartopy.feature.LAND, edgecolor='black', color='tan')
        ax2_4.add_feature(cartopy.feature.OCEAN, color='lightblue')
        ax2_4.set_extent([-6, 16, 34, 44.5])
        _gl = ax2_4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.75, linestyle='--')
        _gl.xlocator = mticker.FixedLocator([-5, 0, 5, 10, 15])
        _gl.ylocator = mticker.FixedLocator([35, 40])
        _gl.top_labels = False
        _gl.right_labels = False
        ax2_4.plot([2.25, 4.5, 4.5, 2.25, 2.25], [40.25, 40.25, 41.75, 41.75, 40.25], color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
        _ = ax2_4.pcolormesh(monthly_chl_ds.lon, monthly_chl_ds.lat, 
                             monthly_chl_ds.isel(time=0).CHL, cmap='viridis', norm=LogNorm(vmin=0.1, vmax=0.7), zorder=3, alpha=0.9)

        ax_9.plot([p1_lons[0], p1_lons[1], p1_lons[1], p1_lons[0], p1_lons[0]], [p1_lats[0], p1_lats[0], p1_lats[1], p1_lats[1], p1_lats[0]], color='red', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
        ax_9.plot([p2_lons[0], p2_lons[1], p2_lons[1], p2_lons[0], p2_lons[0]], [p2_lats[0], p2_lats[0], p2_lats[1], p2_lats[1], p2_lats[0]], color='red', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
        ax_9.plot([p3_lons[0], p3_lons[1], p3_lons[1], p3_lons[0], p3_lons[0]], [p3_lats[0], p3_lats[0], p3_lats[1], p3_lats[1], p3_lats[0]], color='red', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
        ax_9.text(2.4, 40.6, 'R2', color='red', fontsize=17, weight='bold')
        ax_9.text(2.44, 40.965, 'R3', color='red', fontsize=17, weight='bold')
        ax_9.text(3.71, 41.59, 'R1', color='red', fontsize=17, weight='bold')
        cax = ax_9.inset_axes([1.02, 0.0, 0.03, 1])
        cb = _fig.colorbar(cp_1, cax=cax, orientation='vertical', extend='both', ticks=[0.1, 0.2, 0.3, 0.4, 0.6], label='Chl ($\\mu$g L$^{-1}$)')
        cb.ax.tick_params(labelsize=18)
        cb.formatter = ticker.StrMethodFormatter('{x:,.1f}')
        return plt.show()
                #plt.savefig('../../../Manuscript/Figures/regional_map_ecoCTD_CTD_with_Chl_respiration_annotations.png', bbox_inches='tight', dpi=300)


    _()
    return


@app.cell
def _():
    return


@app.cell
def _(glob, np, wkdir, xr):
    i_7 = 2
    chl_files_2 = sorted(glob.glob(wkdir + 'Data/Satellite/Chl/For_calibration/Multi/*'))
    _chl_ds = xr.open_dataset(chl_files_2[i_7])
    _log_chl = np.log10(_chl_ds.isel(time=0).CHL)
    _log_chl = _log_chl.sel(lon=slice(-4, 10), lat=slice(34, 44))
    log_chl_data_2 = _log_chl.values
    chl_3 = _chl_ds.isel(time=0).CHL
    chl_3 = chl_3.sel(lon=slice(-4, 10), lat=slice(34, 44))
    chl_3 = chl_3.values
    print(chl_files_2[i_7])
    return (log_chl_data_2,)


@app.cell
def _(
    cartopy,
    ccrs,
    color_dict_1,
    ecoctd,
    lat_grid_2,
    log_chl_data_2,
    lon_grid_2,
    mticker,
    p1_lats,
    p1_lons,
    p2_lats,
    p2_lons,
    p3_lats,
    p3_lons,
    pe,
    plt,
):
    _fig = plt.figure(figsize=(8, 12), dpi=300)
    ax_10 = plt.axes(projection=ccrs.PlateCarree())
    ax_10.coastlines(lw=2)
    ax_10.add_feature(cartopy.feature.LAND, edgecolor='black', color='tan')
    ax_10.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_10.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.75, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([2, 3, 4, 5])
    _gl.ylocator = mticker.FixedLocator([40.5, 41, 41.5, 42])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_10.set_extent([2.1, 4.5, 40.25, 41.85])
    ax_10.pcolormesh(lon_grid_2, lat_grid_2, log_chl_data_2, cmap='viridis', zorder=3, vmin=-0.95, vmax=-0.15, alpha=0.9)
    plt.scatter(5.3698, 43.2965, zorder=5, color='red', s=50)
    plt.scatter(2.1686, 41.3874, zorder=5, color='white', marker='^', s=200, edgecolor='black')
    ax_10.text(2.12186, 41.475, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
    ax_10.scatter(ecoctd.CTD_lon, ecoctd.CTD_lat, color='black', marker='.', zorder=9, s=20)
    ax_10.plot([p3_lons[0], p3_lons[1], p3_lons[1], p3_lons[0], p3_lons[0]], [p3_lats[0], p3_lats[0], p3_lats[1], p3_lats[1], p3_lats[0]], color=color_dict_1['Phase 3'], linewidth=4, transform=ccrs.PlateCarree(), zorder=5)
    ax_10.plot([p1_lons[0], p1_lons[1], p1_lons[1], p1_lons[0], p1_lons[0]], [p1_lats[0], p1_lats[0], p1_lats[1], p1_lats[1], p1_lats[0]], color=color_dict_1['Phase 1'], linewidth=4, transform=ccrs.PlateCarree(), zorder=5)
    ax_10.plot([p2_lons[0], p2_lons[1], p2_lons[1], p2_lons[0], p2_lons[0]], [p2_lats[0], p2_lats[0], p2_lats[1], p2_lats[1], p2_lats[0]], color=color_dict_1['Phase 2'], linewidth=3, transform=ccrs.PlateCarree(), zorder=5)
    return


@app.cell
def _(color_dict_1):
    color_dict_1
    return


@app.cell
def _(glob, np, wkdir, xr):
    i_8 = 7
    chl_files_3 = sorted(glob.glob(wkdir + 'Data/Satellite/Chl/For_calibration/Multi/*'))
    _chl_ds = xr.open_dataset(chl_files_3[i_8])
    _log_chl = np.log10(_chl_ds.isel(time=0).CHL)
    _log_chl = _log_chl.sel(lon=slice(-4, 10), lat=slice(34, 44))
    log_chl_data_3 = _log_chl.values
    chl_4 = _chl_ds.isel(time=0).CHL
    chl_4 = chl_4.sel(lon=slice(-4, 10), lat=slice(34, 44))
    chl_4 = chl_4.values
    print(chl_files_3[i_8])
    return (log_chl_data_3,)


@app.cell
def _(cartopy, ccrs, lat_grid_2, log_chl_data_3, lon_grid_2, mticker, pe, plt):
    _fig = plt.figure(figsize=(8, 12))
    ax_11 = plt.axes(projection=ccrs.PlateCarree())
    ax_11.coastlines(lw=2)
    ax_11.add_feature(cartopy.feature.LAND, edgecolor='black', color='tan')
    ax_11.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_11.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.75, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([2, 3, 4, 5])
    _gl.ylocator = mticker.FixedLocator([40.5, 41, 41.5, 42])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_11.set_extent([2.1, 4.5, 40.25, 41.85])
    ax_11.pcolormesh(lon_grid_2, lat_grid_2, log_chl_data_3, cmap='viridis', zorder=3, vmin=-0.95, vmax=-0.15, alpha=0.9)
    plt.scatter(5.3698, 43.2965, zorder=5, color='red', s=50)
    plt.scatter(2.1686, 41.3874, zorder=5, color='white', marker='^', s=200, edgecolor='black')
    ax_11.text(2.12186, 41.475, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
    return


@app.cell
def _(glob, np, wkdir, xr):
    i_9 = 15
    chl_files_4 = sorted(glob.glob(wkdir + 'Data/Satellite/Chl/For_calibration/Multi/*'))
    _chl_ds = xr.open_dataset(chl_files_4[i_9])
    _log_chl = np.log10(_chl_ds.isel(time=0).CHL)
    _log_chl = _log_chl.sel(lon=slice(-4, 10), lat=slice(34, 44))
    log_chl_data_4 = _log_chl.values
    chl_5 = _chl_ds.isel(time=0).CHL
    chl_5 = chl_5.sel(lon=slice(-4, 10), lat=slice(34, 44))
    chl_5 = chl_5.values
    print(chl_files_4[i_9])
    return chl_5, log_chl_data_4


@app.cell
def _(cartopy, ccrs, lat_grid_2, log_chl_data_4, lon_grid_2, mticker, pe, plt):
    _fig = plt.figure(figsize=(8, 12))
    ax_12 = plt.axes(projection=ccrs.PlateCarree())
    ax_12.coastlines(lw=2)
    ax_12.add_feature(cartopy.feature.LAND, edgecolor='black', color='tan')
    ax_12.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_12.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.75, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([2, 3, 4, 5])
    _gl.ylocator = mticker.FixedLocator([40.5, 41, 41.5, 42])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_12.set_extent([2.1, 4.5, 40.25, 41.85])
    ax_12.pcolormesh(lon_grid_2, lat_grid_2, log_chl_data_4, cmap='viridis', zorder=3, vmin=-0.95, vmax=-0.3, alpha=0.9)
    plt.scatter(5.3698, 43.2965, zorder=5, color='red', s=50)
    plt.scatter(2.1686, 41.3874, zorder=5, color='white', marker='^', s=200, edgecolor='black')
    ax_12.text(2.12186, 41.475, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
    return


@app.cell
def _(
    cartopy,
    ccrs,
    glob,
    lat_grid_2,
    lon_grid_2,
    mticker,
    np,
    pd,
    pe,
    plt,
    wkdir,
    xr,
):
    _fig, _axes = plt.subplots(4, figsize=(20, 24), subplot_kw={'projection': ccrs.PlateCarree()})
    for ax_13 in _axes:
        ax_13.coastlines(lw=2)
        ax_13.add_feature(cartopy.feature.LAND, edgecolor='black', color='tan')
        ax_13.add_feature(cartopy.feature.OCEAN, color='white')
        _gl = ax_13.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='black', alpha=1, linestyle='--')
        _gl.xlocator = mticker.FixedLocator([2, 3, 4, 5])
        _gl.ylocator = mticker.FixedLocator([40.5, 41, 41.5, 42])
        if ax_13 in [_axes[0], _axes[1], _axes[2]]:
            _gl.bottom_labels = False
        _gl.top_labels = False
        _gl.right_labels = False
        _gl.xlabel_style = {'color': 'black'}
        _gl.xlabel_style = {'color': 'black'}
        ax_13.set_extent([2.1, 4.5, 40.25, 41.85])
        if ax_13 == _axes[0]:
            ax_13.scatter(2.1686, 41.3874, zorder=5, color='white', marker='^', s=200, edgecolor='black')
            ax_13.text(2.12186, 41.475, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
    i_10 = 7
    chl_files_5 = sorted(glob.glob(wkdir + 'Data/Satellite/Chl/For_calibration/Multi/*'))
    _chl_ds = xr.open_dataset(chl_files_5[i_10])
    _log_chl = np.log10(_chl_ds.isel(time=0).CHL)
    _log_chl = _log_chl.sel(lon=slice(-4, 10), lat=slice(34, 44))
    log_chl_data_5 = _log_chl.values
    print(chl_files_5[i_10])
    _axes[0].pcolormesh(lon_grid_2, lat_grid_2, log_chl_data_5, cmap='viridis', zorder=3, vmin=-0.9, vmax=-0.25, alpha=0.9)
    _axes[0].set_title(pd.to_datetime(chl_files_5[i_10].split('/')[-1].split('_')[0]).strftime('%b %d, %Y'), fontsize=20)
    i_10 = 10
    _chl_ds = xr.open_dataset(chl_files_5[i_10])
    _log_chl = np.log10(_chl_ds.isel(time=0).CHL)
    _log_chl = _log_chl.sel(lon=slice(-4, 10), lat=slice(34, 44))
    log_chl_data_5 = _log_chl.values
    print(chl_files_5[i_10])
    _axes[1].pcolormesh(lon_grid_2, lat_grid_2, log_chl_data_5, cmap='viridis', zorder=3, vmin=-0.9, vmax=-0.25, alpha=0.9)
    _axes[1].set_title(pd.to_datetime(chl_files_5[i_10].split('/')[-1].split('_')[0]).strftime('%b %d, %Y'), fontsize=20)
    i_10 = 15
    _chl_ds = xr.open_dataset(chl_files_5[i_10])
    _log_chl = np.log10(_chl_ds.isel(time=0).CHL)
    _log_chl = _log_chl.sel(lon=slice(-4, 10), lat=slice(34, 44))
    log_chl_data_5 = _log_chl.values
    print(chl_files_5[i_10])
    _axes[2].pcolormesh(lon_grid_2, lat_grid_2, log_chl_data_5, cmap='viridis', zorder=3, vmin=-0.9, vmax=-0.25, alpha=0.9)
    _axes[2].set_title(pd.to_datetime(chl_files_5[i_10].split('/')[-1].split('_')[0]).strftime('%b %d, %Y'), fontsize=20)
    i_10 = 21
    _chl_ds = xr.open_dataset(chl_files_5[i_10])
    _log_chl = np.log10(_chl_ds.isel(time=0).CHL)
    _log_chl = _log_chl.sel(lon=slice(-4, 10), lat=slice(34, 44))
    log_chl_data_5 = _log_chl.values
    print(chl_files_5[i_10])
    _axes[3].pcolormesh(lon_grid_2, lat_grid_2, log_chl_data_5, cmap='viridis', zorder=3, vmin=-0.9, vmax=-0.25, alpha=0.9)
    _axes[3].set_title(pd.to_datetime(chl_files_5[i_10].split('/')[-1].split('_')[0]).strftime('%b %d, %Y'), fontsize=20)
    plt.subplots_adjust(hspace=0.1)
    # plt.savefig('figures/context_figures/hydrographic_context_Feb23_Feb26_March3_March9_2022.png', bbox_inches='tight', dpi=300)
    return (log_chl_data_5,)


@app.cell
def _():
    (2200 - 2491)/2491
    return


@app.cell
def _(
    cartopy,
    ccrs,
    ecoctd,
    lat_grid_2,
    log_chl_data_5,
    lon_grid_2,
    mticker,
    pe,
    plt,
    resp_lats,
    resp_lons,
):
    plt.figure(figsize=(8, 12))
    ax_14 = plt.axes(projection=ccrs.PlateCarree())
    ax_14.coastlines()
    ax_14.add_feature(cartopy.feature.LAND, edgecolor='black', color='darkgray')
    ax_14.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_14.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([-6, -4, -2, 0, 2, 4, 6, 8, 10])
    _gl.ylocator = mticker.FixedLocator([34, 36, 38, 40, 42, 44])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_14.set_extent([1, 6, 38, 44])
    ax_14.pcolormesh(lon_grid_2, lat_grid_2, log_chl_data_5, cmap='viridis', zorder=3, vmin=-1.25, vmax=0.15, alpha=0.85)
    ax_14.scatter(ecoctd.CTD_lon, ecoctd.CTD_lat, s=1, c='black', zorder=5, alpha=0.65)
    ax_14.scatter(resp_lons['Cast1'], resp_lats['Cast1'], zorder=10, s=250, marker='*', color='magenta', label='Cast 1', edgecolor='black')
    ax_14.scatter(resp_lons['ACM'], resp_lats['ACM'], zorder=11, s=250, marker='*', color='greenyellow', label='ACM', edgecolor='black')
    ax_14.scatter(resp_lons['Background'], resp_lats['Background'], zorder=10, s=250, marker='*', color='dodgerblue', label='Background', edgecolor='black')
    ax_14.scatter([], [], color='black', s=100, label='EcoCTD')
    ax_14.legend(loc=2, fontsize=18)
    plt.scatter(5.3698, 43.2965, zorder=5, color='red', s=50)
    plt.scatter(2.1686, 41.3874, zorder=5, color='red', s=50)
    ax_14.text(4.4698, 43.4465, 'Marseilles', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], fontsize=18)
    ax_14.text(1.2186, 41.5874, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
    # plt.savefig('figures/context_figures/regional_map_ecoCTD_CTD_with_Chl_respiration.png', bbox_inches='tight', dpi=300)
    return


@app.cell
def _(
    cartopy,
    ccrs,
    ctd,
    ecoctd,
    lat_grid_2,
    log_chl_data_5,
    lon_grid_2,
    mticker,
    pe,
    plt,
):
    plt.figure(figsize=(8, 12))
    ax_15 = plt.axes(projection=ccrs.PlateCarree())
    ax_15.coastlines()
    ax_15.add_feature(cartopy.feature.LAND, edgecolor='black', color='lightgray')
    ax_15.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_15.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([-6, -4, -2, 0, 2, 4, 6, 8, 10])
    _gl.ylocator = mticker.FixedLocator([34, 36, 38, 40, 42, 44])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_15.set_extent([0, 8, 36, 44])
    ax_15.pcolormesh(lon_grid_2, lat_grid_2, log_chl_data_5, cmap='viridis', zorder=3, vmin=-1.25, vmax=0.15)
    ax_15.scatter(ecoctd.CTD_lon, ecoctd.CTD_lat, s=2, color='#FAAF07', zorder=5)
    ax_15.scatter(ctd.LON, ctd.LAT, color='black', s=50, marker='*', zorder=5)
    ax_15.plot([2, 4.5, 4.5, 2, 2], [40, 40, 42, 42, 40], color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    plt.scatter(5.3698, 43.2965, zorder=5, color='red', s=50)
    plt.scatter(2.1686, 41.3874, zorder=5, color='red', s=50)
    ax_15.text(4.4698, 43.4465, 'Marseilles', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], fontsize=18)
    ax_15.text(1.0186, 41.5874, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
    plt.tight_layout()

    plt.show()
    # plt.savefig('figures/context_figures/regional_map_ecoCTD_CTD_with_Chl.png', bbox_inches='tight', dpi=300)
    return


@app.cell
def _(cartopy, ccrs, chl_5, ctd, lat_grid_2, lon_grid_2, mticker, plt):
    plt.figure(figsize=(10, 6))
    ax_16 = plt.axes(projection=ccrs.PlateCarree())
    ax_16.coastlines()
    ax_16.add_feature(cartopy.feature.LAND, edgecolor='black', color='lightgray')
    ax_16.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_16.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([2, 3, 4, 5])
    _gl.ylocator = mticker.FixedLocator([39, 40, 41, 42, 43])
    _gl.top_labels = False
    _gl.right_labels = False
    ax_16.set_extent([2, 4, 40.0, 42])
    cids = [13, 18, 27, 28, 39, 41, 51, 52, 56, 60]
    for i_11 in cids:
        plt.scatter(ctd.isel(n_casts=i_11 - 1).CTD_lon, ctd.isel(n_casts=i_11 - 1).CTD_lat, zorder=10, color='red')
    plt.pcolormesh(lon_grid_2, lat_grid_2, chl_5, cmap='viridis', zorder=6, vmin=0, vmax=3)
    plt.colorbar(label='log10(Chlorophyll)', extend='both')
    ax_16.text(2.08, 41.79, 'Feb. 23, 2022', zorder=10, fontsize=16)
    return


@app.cell
def _(ctd, pd, wkdir):
    mapping_df = pd.read_excel(wkdir + 'Data/Metadata/Resp_Bio_Niskin_mapping.xls', sheet_name=0)
    resp_ctd = ctd.isel(n_casts = ctd.cast.isin([mapping_df['Cast'].values]))
    return


@app.cell
def _(cartopy, ccrs, ctd, ecoctd, mticker, plt):
    plt.figure(figsize=(8, 12))
    ax_17 = plt.axes(projection=ccrs.PlateCarree())
    ax_17.coastlines()
    ax_17.add_feature(cartopy.feature.LAND, edgecolor='black', color='lightgray')
    ax_17.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_17.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([2, 3, 4, 4.5])
    _gl.ylocator = mticker.FixedLocator([40, 40.5, 41.5, 42])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_17.set_extent([2, 4.5, 40.2, 42])
    ax_17.scatter(ecoctd.CTD_lon, ecoctd.CTD_lat, s=5, color='#D66853', label='ecoCTD cast')
    ax_17.scatter(ctd.CTD_lon, ctd.CTD_lat, color='#212D40', s=100, marker='*', label='CTD cast')
    _lgnd = plt.legend()
    _lgnd.legendHandles[0]._sizes = [100]
    _lgnd.legendHandles[1]._sizes = [100]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot for cruise report
    """)
    return


@app.cell
def _(mpl, np):
    cmap = mpl.cm.plasma_r(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[2:, :-1])
    return (cmap,)


@app.cell
def _(cartopy, ccrs, cmap, ctd, make_axes_locatable, mticker, pe, plt):
    _fig = plt.figure(figsize=(8, 12))
    ax_18 = plt.axes(projection=ccrs.PlateCarree())
    ax_18.coastlines()
    ax_18.add_feature(cartopy.feature.LAND, edgecolor='black', color='lightgray')
    ax_18.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_18.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([-6, -4, -2, 0, 2, 3, 4, 6, 8, 10])
    _gl.ylocator = mticker.FixedLocator([34, 36, 38, 40, 41, 42, 44])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_18.set_extent([2, 4.75, 39.75, 42])
    plt.scatter(2.1686, 41.3874, zorder=5, color='red', s=150)
    ax_18.text(2.086, 41.4874, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
    cp_2 = ax_18.scatter(ctd.CTD_lon, ctd.CTD_lat, c=ctd.cast.values, cmap=cmap, s=250, marker='*', zorder=5)
    divider = make_axes_locatable(ax_18)
    ax_cb = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    _fig.add_axes(ax_cb)
    plt.colorbar(cp_2, cax=ax_cb, label='CTD cast number')
    plt.tight_layout()
    plt.savefig('figures/context_figures/CTD_cast_positions_for_cruise_report.png', bbox_inches='tight', dpi=300)
    return


@app.cell
def _(ecoctd, pd):
    phase1_1 = ecoctd.UTC_time < pd.to_datetime('2022-02-21T08:30:00')
    phase2_1 = (ecoctd.UTC_time > pd.to_datetime('2022-02-22T07:25:00')) & (ecoctd.UTC_time < pd.to_datetime('2022-03-01T08:30:00'))
    phase3_1 = (ecoctd.UTC_time > pd.to_datetime('2022-03-01T08:30:00')) & (ecoctd.UTC_time < pd.to_datetime('2022-03-10T10:40:00'))
    return phase1_1, phase2_1, phase3_1


@app.cell
def _(ctd, pd):
    phase2_ctd = ((ctd.UTC_time > pd.to_datetime('2022-02-22T07:25:00')) & 
                (ctd.UTC_time < pd.to_datetime('2022-03-01T08:30:00')))
    phase3_ctd = ((ctd.UTC_time > pd.to_datetime('2022-03-01T08:30:00')) & 
                (ctd.UTC_time < pd.to_datetime('2022-03-10T10:40:00')))
    return phase2_ctd, phase3_ctd


@app.cell
def _(cartopy, ccrs, ctd, mticker, phase2_ctd, phase3_ctd, plt):
    plt.figure(figsize=(8, 12))
    ax_19 = plt.axes(projection=ccrs.PlateCarree())
    ax_19.coastlines()
    ax_19.add_feature(cartopy.feature.LAND, edgecolor='black', color='lightgray')
    ax_19.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_19.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([2, 3, 4, 4.5])
    _gl.ylocator = mticker.FixedLocator([40, 41])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_19.set_extent([1, 4.5, 40, 42.5])
    color_dict_2 = {'Phase 1': '#5E79FD', 'Phase 2': '#FFA400', 'Phase 3': '#FF3399'}
    plt.scatter(ctd.CTD_lon[phase2_ctd], ctd.CTD_lat[phase2_ctd], color=color_dict_2['Phase 2'])
    plt.scatter(ctd.CTD_lon[phase3_ctd], ctd.CTD_lat[phase3_ctd], color=color_dict_2['Phase 3'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Want to plot some sections that we find interesting
    """)
    return


@app.cell
def _(cartopy, ccrs, ecoctd, mticker, plt):
    plt.figure(figsize=(8, 12))
    ax_20 = plt.axes(projection=ccrs.PlateCarree())
    ax_20.coastlines()
    ax_20.add_feature(cartopy.feature.LAND, edgecolor='black', color='lightgray')
    ax_20.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_20.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([2, 3, 4, 4.5])
    _gl.ylocator = mticker.FixedLocator([40, 40.5, 41.5, 42])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_20.set_extent([2, 3.5, 40.5, 42])
    ax_20.scatter(ecoctd.CTD_lon[2042:2076], ecoctd.CTD_lat[2042:2076], color='#CC6677', s=50)
    return


@app.cell
def _(ecoctd):
    #transect = ecoctd.isel(n_casts = slice(1662, 1672))
    #transect = ecoctd.isel(n_casts = slice(1653, 1662))
    transect_2 = ecoctd.isel(n_casts=slice(1458, 1473))
    #transect = ecoctd.isel(n_casts = slice(1843, 1852))
    #transect = ecoctd.isel(n_casts = slice(2042, 2076))
    #transect = ecoctd.isel(n_casts = slice(1110, 1129))
    #transect = ecoctd.isel(n_casts = slice(1858,1871))
    transect_2 = ecoctd.isel(n_casts=slice(1968, 1978))
    return (transect_2,)


@app.cell
def _(haversine_np, np, transect_2):
    dates_1 = transect_2.UTC_time.values
    pres_grid_1 = np.repeat(transect_2.pressure.values, transect_2.dims['n_casts']).reshape((transect_2.dims['n_levels'], transect_2.dims['n_casts']))
    _start_lon, _start_lat = (transect_2.isel(n_casts=0)['CTD_lon'].values, transect_2.isel(n_casts=0)['CTD_lat'].values)
    dists_1 = haversine_np(_start_lon, _start_lat, transect_2['CTD_lon'].values[:], transect_2['CTD_lat'].values[:])
    dist_grid_1 = np.tile(dists_1, pres_grid_1.shape[0]).reshape(pres_grid_1.shape)
    return dist_grid_1, dists_1, pres_grid_1


@app.cell
def _(transect_2):
    i_12 = 0
    _pres_vals = transect_2.isel(n_casts=i_12).pressure.values
    return


@app.cell
def _(sns):
    sns.set(style="white", font_scale=2.75, font="Verdana")
    return


@app.cell
def _(cmocean, dist_grid_1, dists_1, np, plt, pres_grid_1, transect_2):
    _fig, ((_ax0, _ax1), (ax2_5, _ax3)) = plt.subplots(2, 2, figsize=(48, 22))
    _pres_ind = np.inf
    for i_13 in range(transect_2.dims['n_casts']):
        _pres_vals = transect_2.isel(n_casts=i_13).pressure.values
        _pres_ind_temp = np.nanargmax(np.isnan(_pres_vals))
        _pres_ind = int(np.nanmin([_pres_ind, _pres_ind_temp]))
    cp_3 = _ax0.contourf(dist_grid_1, pres_grid_1, transect_2.CTD_CT.values.T, levels=10, cmap=cmocean.cm.thermal)
    _ax0.contour(dist_grid_1, pres_grid_1, transect_2.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    _ax0.set_ylim([250, 0])
    _ax0.set_xlim([0, np.nanmax(dists_1)])
    plt.colorbar(cp_3, label='Conservative temperature ($\\degree$C)', ax=_ax0)
    _ax0.set_xlabel('Distance along transect (km)')
    _ax0.set_ylabel('Pressure (dbar)')
    cp_3 = _ax1.contourf(dist_grid_1, pres_grid_1, transect_2.CTD_SA.values.T, levels=10, cmap=cmocean.cm.haline)
    _ax1.contour(dist_grid_1, pres_grid_1, transect_2.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    _ax1.set_ylim([250, 0])
    _ax1.set_xlim([0, np.nanmax(dists_1)])
    plt.colorbar(cp_3, label='Absolute salinity', ax=_ax1)
    _ax1.set_xlabel('Distance along transect (km)')
    _ax1.set_ylabel('Pressure (dbar)')
    cp_3 = ax2_5.contourf(dist_grid_1, pres_grid_1, transect_2.OXY_O2sat.values.T, levels=10, cmap='rainbow')
    ax2_5.contour(dist_grid_1, pres_grid_1, transect_2.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    ax2_5.set_ylim([250, 0])
    ax2_5.set_xlim([0, np.nanmax(dists_1)])
    plt.colorbar(cp_3, label='Dissolved $O_2$ (% saturation)', ax=ax2_5)
    ax2_5.set_xlabel('Distance along transect (km)')
    ax2_5.set_ylabel('Pressure (dbar)')
    _fl = transect_2.FLS_chl.values.T
    _fl = _fl + np.nanmin(_fl)
    _fl = np.log10(_fl)
    _fl_min = np.nanmin(_fl[~np.isinf(_fl)])
    _fl[np.isinf(_fl)] = _fl_min
    cp_3 = _ax3.contourf(dist_grid_1, pres_grid_1, _fl, levels=10, cmap=cmocean.cm.algae, vmax=1)
    _ax3.contour(dist_grid_1, pres_grid_1, transect_2.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    _ax3.set_ylim([250, 0])
    _ax3.set_xlim([0, np.nanmax(dists_1)])
    plt.colorbar(cp_3, label='log10(Chl) (mg/$m^3$)', ax=_ax3)
    _ax3.set_xlabel('Distance along transect (km)')
    _ax3.set_ylabel('Pressure (dbar)')
    for ax_21 in [_ax0, _ax1, ax2_5, _ax3]:
        ax_21.scatter(dists_1[9] + 0.1, [2], marker='v', s=1000, zorder=100, color='red')
        ax_21.tick_params(bottom=True, left=True)
    return


@app.cell
def _(hydrg_dir, pickle, wkdir):
    filename = wkdir + hydrg_dir + 'RandomForest_Nitrate/nitrate_model_random_forest_O2_SA_CT_061023.sav'
    nitrate_regr = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)
    return (nitrate_regr,)


@app.cell
def _():
    varlist = ['O2_sat', 'SA', 'CT']#, 'pres', 'lat', 'lon']
    return (varlist,)


@app.cell
def _(dist_grid_1, nitrate_regr, np, pd, transect_2, varlist):
    predicted_nitrate = np.zeros(dist_grid_1.shape)
    predicted_nitrate[:] = np.nan
    for i_14 in range(transect_2.dims['n_casts']):
        _i_ds = transect_2.isel(n_casts=i_14)
        _nan_ind = np.isnan(_i_ds.OXY_O2sat.values) | np.isnan(_i_ds.CTD_SA.values) | np.isnan(_i_ds.CTD_CT.values)
        df_2 = pd.DataFrame({'O2_sat': _i_ds.OXY_O2sat.values[~_nan_ind], 'SA': _i_ds.CTD_SA.values[~_nan_ind], 'CT': _i_ds.CTD_CT.values[~_nan_ind], 'pres': _i_ds.pressure.values[~_nan_ind], 'lat': np.tile(_i_ds.CTD_lat, np.sum(~_nan_ind)), 'lon': np.tile(_i_ds.CTD_lon, np.sum(~_nan_ind))})
        predicted_nitrate[~_nan_ind, i_14] = nitrate_regr.predict(df_2[varlist]).flatten()
    return (predicted_nitrate,)


@app.cell
def _(
    cmocean,
    dist_grid_1,
    dists_1,
    np,
    plt,
    predicted_nitrate,
    pres_grid_1,
    sns,
    transect_2,
):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, ax_22 = plt.subplots(figsize=(12, 6))
    ax_22.tick_params(bottom=True, left=True)
    plt.contourf(dist_grid_1, pres_grid_1, predicted_nitrate, cmap=cmocean.cm.thermal, levels=10)
    plt.ylim([250, 0])
    plt.xlim([0, np.nanmax(dists_1)])
    plt.colorbar(label='Predicted nitrate ($\\mu$M)')
    plt.contour(dist_grid_1, pres_grid_1, transect_2.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    ax_22.set_xlabel('Distance along transect (km)')
    ax_22.set_ylabel('Pressure (dbar)')
    return


@app.cell
def _(predicted_nitrate):
    predicted_nitrate1 = predicted_nitrate.copy()
    return (predicted_nitrate1,)


@app.cell
def _(plt, predicted_nitrate1, transect_2):
    plt.scatter(predicted_nitrate1.ravel(), transect_2.OXY_O2sat.values.T.ravel(), s=5)
    return


@app.cell
def _(plt, predicted_nitrate, transect_2):
    plt.scatter(predicted_nitrate.ravel(), transect_2.CTD_Sigma0.values.T.ravel(), s=5)
    return


@app.cell
def _(np, plt, predicted_nitrate, transect_2):
    plt.scatter(predicted_nitrate.ravel(), np.repeat(transect_2.pressure, transect_2.dims['n_casts']), s=5)
    return


@app.cell
def _(dist_grid_1, plt, predicted_nitrate, pres_grid_1, transect_2):
    _fig, ax_23 = plt.subplots(figsize=(12, 6))
    ax_23.tick_params(bottom=True, left=True)
    plt.contourf(dist_grid_1, pres_grid_1, transect_2.FLS_chl.values.T / predicted_nitrate, levels=25, vmax=0.75)
    plt.colorbar()
    plt.ylim([250, 0])
    return


@app.cell
def _(dist_grid_1, plt, pres_grid_1, transect_2):
    _fig, ax_24 = plt.subplots(figsize=(12, 6))
    ax_24.tick_params(bottom=True, left=True)
    plt.contourf(dist_grid_1, pres_grid_1, transect_2.FLS_chl.values.T / transect_2.OXY_O2sat.values.T, levels=10)
    plt.colorbar()
    plt.ylim([250, 0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scenario where nitrate is low/depleted but nitrate-to-Chl ratio is also lower than surrounding waters --
    suggesting water was upwelled less recently? Or never upwelled at all

    whereas high nitrate, low nitrate-to-Chl ratio: water has been recently upwelled but hasn't fueled significant primary productivity yet, for whatever reason
    """)
    return


@app.cell
def _(transects_df):
    transects_df.head()
    return


@app.cell
def _(ecoctd, pd, transects_df):
    i_15 = 5
    time_ind_1 = (pd.to_datetime(ecoctd.UTC_time.values) >= transects_df.iloc[i_15]['Initial_Time']) & (pd.to_datetime(ecoctd.UTC_time.values) <= transects_df.iloc[i_15]['Final_Time'])
    return (time_ind_1,)


@app.cell
def _(np, time_ind_1):
    sl_1 = (np.where(time_ind_1)[0][0], np.where(time_ind_1)[0][-1] + 1)
    return (sl_1,)


@app.cell
def _(ecoctd, plt, sl_1):
    plt.scatter(4.0311, 41.4474)
    plt.scatter(4.2687, 41.2454)
    plt.plot(ecoctd.CTD_lon[sl_1[0]:sl_1[1]], ecoctd.CTD_lat[sl_1[0]:sl_1[1]])
    plt.scatter(ecoctd.CTD_lon[sl_1[0]:sl_1[1]], ecoctd.CTD_lat[sl_1[0]:sl_1[1]])
    return


@app.cell
def _(ecoctd, np, pd, transects_df):
    n_obs = []
    for ind, _row in transects_df.iterrows():
        time_ind_2 = (pd.to_datetime(ecoctd.UTC_time.values) >= _row['Initial_Time']) & (pd.to_datetime(ecoctd.UTC_time.values) <= _row['Final_Time'])
        if np.sum(time_ind_2) != 0:
            sl_2 = (np.where(time_ind_2)[0][0], np.where(time_ind_2)[0][-1] + 1)
            n_obs.append(sl_2[1] - sl_2[0])
        else:
            n_obs.append(0)
    return (n_obs,)


@app.cell
def _(n_obs, transects_df):
    transects_df['NPROF'] = n_obs
    return


@app.cell
def _(transects_df):
    #transects_df.sort_values(by='NPROF').tail(20)
    transects_df.head(200).tail(30)
    return


@app.cell
def _():
    1683
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting for Helena
    """)
    return


@app.cell
def _():
    # subset_df['Initial_Time'].min()
    return


@app.cell
def _():
    # subset_df
    return


@app.cell
def _(transects_df):
    transect1_df = transects_df[transects_df['Transect_Number'].isin(range(118, 121))]
    transect2_df = transects_df[transects_df['Transect_Number'].isin(range(229, 236))]
    transect3_df = transects_df[transects_df['Transect_Number'].isin(range(251, 253))]
    return transect1_df, transect2_df, transect3_df


@app.cell
def _(ecoctd, pd, plt, transect2_df):
    time_ind_3 = (pd.to_datetime(ecoctd.UTC_time.values) >= transect2_df['Initial_Time'].min()) & (pd.to_datetime(ecoctd.UTC_time.values) <= transect2_df['Final_Time'].max())
    for i_16, _row in transect2_df.iterrows():
        plt.plot([_row['Initial_Lon'], _row['Final_Lon']], [_row['Initial_Lat'], _row['Final_Lat']])
        ecoctd.isel(n_casts=time_ind_3).plot.scatter(x='CTD_lon', y='CTD_lat')
        plt.plot(ecoctd.isel(n_casts=time_ind_3).CTD_lon, ecoctd.isel(n_casts=time_ind_3).CTD_lat)
    return


@app.cell
def _(cartopy, ccrs, ecoctd, mticker, pe, plt):
    _fig, ax_25 = plt.subplots(1, figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax_25.coastlines()
    ax_25.add_feature(cartopy.feature.LAND, edgecolor='black', color='lightgray')
    ax_25.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_25.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([-6, -4, -2, 0, 2, 4, 6, 8, 10])
    _gl.xlocator = mticker.FixedLocator([1, 1.5, 2, 2.5, 3, 3.5, 4])
    _gl.ylocator = mticker.FixedLocator([34, 36, 38, 40, 42, 44])
    _gl.ylocator = mticker.FixedLocator([40, 40.5, 41, 41.5])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_25.set_extent([1.5, 4.5, 40, 42])
    ecoctd.plot.scatter(x='CTD_lon', y='CTD_lat', c='lightskyblue', alpha=1, zorder=0, s=7, marker='o', ax=ax_25)
    ax_25.scatter(5.3698, 43.2965, zorder=5, color='red', s=50)
    ax_25.scatter(2.1686, 41.3874, zorder=5, color='red', s=50)
    ax_25.text(1.6186, 41.3874, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
    return


@app.cell
def _(
    cartopy,
    ccrs,
    ecoctd,
    mticker,
    pd,
    pe,
    plt,
    transect1_df,
    transect2_df,
    transect3_df,
):
    transect_dict = {'118-120': transect1_df, '229-235': transect2_df, '251-252': transect3_df}
    transect_dict_colors = {'118-120': 'forestgreen', '229-235': 'red', '251-252': 'darkblue'}
    for tr in transect_dict.keys():
        _fig, ax_26 = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax_26.coastlines()
        ax_26.add_feature(cartopy.feature.LAND, edgecolor='black', color='lightgray')
        ax_26.add_feature(cartopy.feature.OCEAN, color='white')
        _gl = ax_26.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
        _gl.xlocator = mticker.FixedLocator([-6, -4, -2, 0, 2, 4, 6, 8, 10])
        _gl.xlocator = mticker.FixedLocator([1, 1.5, 2, 2.5, 3, 3.5, 4])
        _gl.ylocator = mticker.FixedLocator([34, 36, 38, 40, 42, 44])
        _gl.ylocator = mticker.FixedLocator([40, 40.5, 41, 41.5])
        _gl.top_labels = False
        _gl.right_labels = False
        _gl.xlabel_style = {'color': 'black'}
        _gl.xlabel_style = {'color': 'black'}
        ax_26.set_extent([1.5, 4.5, 40, 42])
        ecoctd.plot.scatter(x='CTD_lon', y='CTD_lat', c='gray', alpha=0.5, zorder=0, s=6, marker='o', ax=ax_26)
        ax_26.scatter(5.3698, 43.2965, zorder=5, color='red', s=50)
        ax_26.scatter(2.1686, 41.3874, zorder=5, color='red', s=50)
        ax_26.text(1.6186, 41.3874, 'Barcelona', color='black', path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=6, fontsize=18)
        time_ind_4 = (pd.to_datetime(ecoctd.UTC_time.values) >= transect_dict[tr]['Initial_Time'].min()) & (pd.to_datetime(ecoctd.UTC_time.values) <= transect_dict[tr]['Final_Time'].max())
        ax_26.plot(ecoctd.isel(n_casts=time_ind_4).CTD_lon, ecoctd.isel(n_casts=time_ind_4).CTD_lat, color=transect_dict_colors[tr], linewidth=5.5)
    return


@app.cell
def _():
    transect_i_1 = 181  # some interesting possible signs of upwelling limb?
    #transect_i = 170
    transect_i_1 = 58
    return (transect_i_1,)


@app.cell
def _(ecoctd, haversine_np, np, pd, transect_i_1, transects_df):
    time_ind_5 = (pd.to_datetime(ecoctd.UTC_time.values) >= transects_df.iloc[transect_i_1]['Initial_Time']) & (pd.to_datetime(ecoctd.UTC_time.values) <= transects_df.iloc[transect_i_1]['Final_Time'])
    sl_3 = (np.where(time_ind_5)[0][0], np.where(time_ind_5)[0][-1] + 1)
    transect_3 = ecoctd.isel(n_casts=slice(sl_3[0], sl_3[1]))
    dates_2 = transect_3.UTC_time.values
    pres_grid_2 = np.repeat(transect_3.pressure.values, transect_3.dims['n_casts']).reshape((transect_3.dims['n_levels'], transect_3.dims['n_casts']))
    _start_lon, _start_lat = (transect_3.isel(n_casts=0)['CTD_lon'].values, transect_3.isel(n_casts=0)['CTD_lat'].values)
    dists_2 = haversine_np(_start_lon, _start_lat, transect_3['CTD_lon'].values[:], transect_3['CTD_lat'].values[:])
    dist_grid_2 = np.tile(dists_2, pres_grid_2.shape[0]).reshape(pres_grid_2.shape)
    return dist_grid_2, dists_2, pres_grid_2, sl_3, transect_3


@app.cell
def _(dist_grid_2, nitrate_regr, np, pd, transect_3, varlist):
    predicted_nitrate_1 = np.zeros(dist_grid_2.shape)
    predicted_nitrate_1[:] = np.nan
    for i_17 in range(transect_3.dims['n_casts']):
        _i_ds = transect_3.isel(n_casts=i_17)
        _nan_ind = np.isnan(_i_ds.OXY_O2sat.values) | np.isnan(_i_ds.CTD_SA.values) | np.isnan(_i_ds.CTD_CT.values)
        df_3 = pd.DataFrame({'O2_sat': _i_ds.OXY_O2sat.values[~_nan_ind], 'SA': _i_ds.CTD_SA.values[~_nan_ind], 'CT': _i_ds.CTD_CT.values[~_nan_ind], 'pres': _i_ds.pressure.values[~_nan_ind], 'lat': np.tile(_i_ds.CTD_lat, np.sum(~_nan_ind)), 'lon': np.tile(_i_ds.CTD_lon, np.sum(~_nan_ind))})
        predicted_nitrate_1[~_nan_ind, i_17] = nitrate_regr.predict(df_3[varlist]).flatten()
    return (predicted_nitrate_1,)


@app.cell
def _(
    cmocean,
    dist_grid_2,
    dists_2,
    np,
    plt,
    predicted_nitrate_1,
    pres_grid_2,
    sns,
    transect_3,
):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2_6) = plt.subplots(1, 2, figsize=(18, 10))
    _ax1.tick_params(bottom=True, left=True)
    cp_4 = _ax1.contourf(dist_grid_2, pres_grid_2, predicted_nitrate_1, cmap=cmocean.cm.thermal, levels=15, vmin=0)
    _ax1.set_ylim([250, 0])
    _ax1.set_xlim([0, np.nanmax(dists_2)])
    plt.colorbar(cp_4, label='Predicted nitrate ($\\mu$M)', orientation='horizontal', ax=_ax1)
    _ax1.contour(dist_grid_2, pres_grid_2, transect_3.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    _ax1.set_xlabel('Distance along transect (km)')
    _ax1.set_ylabel('Pressure (dbar)')
    ax2_6.tick_params(bottom=True, left=True)
    cp_4 = ax2_6.contourf(dist_grid_2, pres_grid_2, transect_3.OXY_O2sat.values.T, cmap=cmocean.cm.thermal_r, levels=15)
    ax2_6.set_ylim([250, 0])
    ax2_6.set_xlim([0, np.nanmax(dists_2)])
    plt.colorbar(cp_4, label='Dissolved oxygen (% saturation)', orientation='horizontal', ax=ax2_6)
    ax2_6.contour(dist_grid_2, pres_grid_2, transect_3.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    ax2_6.set_xlabel('Distance along transect (km)')
    return


@app.cell
def _(np, transect_3, xr):
    transect_3['correctedbbp2'] = xr.where(transect_3.FLS_bbp2 < 0.2, transect_3.FLS_bbp2, np.nan)
    return


@app.cell
def _(
    LogNorm,
    cmocean,
    dist_grid_2,
    dists_2,
    np,
    plt,
    pres_grid_2,
    sns,
    transect_3,
    transect_i_1,
):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2_7) = plt.subplots(2, figsize=(11, 10))
    _ax1.tick_params(bottom=True, left=True)
    chl_6 = transect_3.FLS_chl.values.T + np.abs(np.nanmin(transect_3.FLS_chl.values.T))
    _cf = _ax1.contourf(dist_grid_2, pres_grid_2, chl_6, cmap=cmocean.cm.speed, levels=np.logspace(0.4, 0.75, 10), norm=LogNorm())
    _ax1.set_ylim([250, 5])
    _ax1.set_xlim([0, np.nanmax(dists_2)])
    from matplotlib.ticker import LogFormatter
    _l_f = LogFormatter(10, labelOnlyBase=False)
    cbar_5 = plt.colorbar(_cf, label='Chlorophyll (mg/m$^3$)', ticks=[0.4, 0.5, 0.75], format=_l_f, ax=_ax1)
    _ax1.contour(dist_grid_2, pres_grid_2, transect_3.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    _ax1.set_ylabel('Pressure (dbar)')
    ax2_7.tick_params(bottom=True, left=True)
    _cf = ax2_7.contourf(dist_grid_2, pres_grid_2, transect_3.correctedbbp2.values.T, cmap=cmocean.cm.thermal_r, levels=25)
    ax2_7.set_ylim([250, 5])
    ax2_7.set_xlim([0, np.nanmax(dists_2)])
    plt.colorbar(_cf, label='Dissolved oxygen (% saturation)', ax=ax2_7)
    ax2_7.contour(dist_grid_2, pres_grid_2, transect_3.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    ax2_7.set_xlabel('Distance along transect (km)')
    plt.savefig(f'figures/transects/oxygen_chlorophyll_transects_{transect_i_1}.png', dpi=300)
    return (LogFormatter,)


@app.cell
def _(
    LogFormatter,
    LogNorm,
    cmocean,
    dist_grid_2,
    dists_2,
    np,
    plt,
    pres_grid_2,
    sns,
    transect_3,
):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2_8) = plt.subplots(2, figsize=(11, 10))
    _ax1.tick_params(bottom=True, left=True)
    chl_7 = transect_3.FLS_chl.values.T + np.abs(np.nanmin(transect_3.FLS_chl.values.T))
    _cf = _ax1.contourf(dist_grid_2, pres_grid_2, chl_7, cmap=cmocean.cm.speed, levels=np.logspace(-2, 0.75, 10), norm=LogNorm())
    _ax1.set_ylim([250, 5])
    _ax1.set_xlim([0, np.nanmax(dists_2)])
    _l_f = LogFormatter(10, labelOnlyBase=False)
    cbar_6 = plt.colorbar(_cf, label='Chlorophyll (mg/m$^3$)', ticks=[0.01, 0.5, 0.75], format=_l_f, ax=_ax1)
    _ax1.contour(dist_grid_2, pres_grid_2, transect_3.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    _ax1.set_ylabel('Pressure (dbar)')
    ax2_8.tick_params(bottom=True, left=True)
    _cf = ax2_8.contourf(dist_grid_2, pres_grid_2, transect_3.OXY_O2sat.values.T, cmap=cmocean.cm.thermal_r, levels=15)
    ax2_8.set_ylim([250, 5])
    ax2_8.set_xlim([0, np.nanmax(dists_2)])
    plt.colorbar(_cf, label='Dissolved oxygen (% saturation)', ax=ax2_8)
    ax2_8.contour(dist_grid_2, pres_grid_2, transect_3.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    ax2_8.set_xlabel('Distance along transect (km)')
    return


@app.cell
def _(
    LogNorm,
    cmocean,
    dist_grid_2,
    dists_2,
    np,
    plt,
    predicted_nitrate_1,
    pres_grid_2,
    sns,
    transect_3,
):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2_9) = plt.subplots(2, figsize=(11, 10))
    _ax1.tick_params(bottom=True, left=True)
    chl_8 = transect_3.FLS_chl.values.T + np.abs(np.nanmin(transect_3.FLS_chl.values.T))
    _cf = _ax1.contourf(dist_grid_2, pres_grid_2, chl_8, cmap=cmocean.cm.speed, levels=np.logspace(0.35, 0.75, 15), norm=LogNorm())
    _ax1.set_ylim([200, 5])
    _ax1.set_xlim([0, np.nanmax(dists_2)])
    cbar_7 = plt.colorbar(_cf, label='Chlorophyll (mg/m$^3$)', ticks=[], orientation='vertical', ax=_ax1)
    _ax1.contour(dist_grid_2, pres_grid_2, transect_3.CTD_Sigma0.values.T, levels=np.arange(28.4, 29.1, 0.05), colors='black', linewidths=1)
    _ax1.set_ylabel('Pressure (dbar)')
    ax2_9.tick_params(bottom=True, left=True)
    _cf = ax2_9.contourf(dist_grid_2, pres_grid_2, predicted_nitrate_1, cmap=cmocean.cm.thermal_r, levels=15)
    ax2_9.set_ylim([200, 5])
    ax2_9.set_xlim([0, np.nanmax(dists_2)])
    plt.colorbar(_cf, label='Predicted nitrate ($\\mu$M)', orientation='vertical', ax=ax2_9)
    _CS = ax2_9.contour(dist_grid_2, pres_grid_2, transect_3.CTD_Sigma0.values.T, levels=np.arange(28.4, 28.75, 0.05), colors='black', linewidths=1)
    _CS = ax2_9.contour(dist_grid_2, pres_grid_2, transect_3.CTD_Sigma0.values.T, levels=np.arange(28.75, 29.1, 0.05), colors='black', linewidths=1)
    ax2_9.clabel(_CS, _CS.levels, inline=True, fontsize=10)
    ax2_9.set_xlabel('Distance along transect (km)')
    ax2_9.set_ylabel('Pressure (dbar)')
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculate spice anomaly:
    - First, we want to construct a reference T and S profile at interpolated density levels (say 0.1 kg/m3), by taking the median T and S at a given density level. Some smoothing can be applied if jagged
    - Then, we want to calculate the spice anomaly for a value by taking $\alpha \Delta T + \beta\Delta S$,where $\Delta T = T - T_{ref}$ and $\Delta S = S - S_{ref}$, with $T_{ref}$ and $S_{ref}$ defined as a function of $\sigma$
    """)
    return


@app.cell
def _(ecoctd, np):
    _spacing = 0.01
    density_levels_2 = np.arange(np.round(ecoctd.CTD_Sigma0.min(), 2), np.round(ecoctd.CTD_Sigma0.max(), 2), _spacing)
    return (density_levels_2,)


@app.cell
def _(density_levels_2, ecoctd, np, scipy):
    for i_18 in range(ecoctd.dims['n_casts']):
        _cast = ecoctd.isel(n_casts=i_18)
        _sigma = _cast.CTD_Sigma0.values
        _ct = _cast.CTD_CT.values
        _sa = _cast.CTD_SA.values
        _sa_nan_ind = ~np.isnan(_sigma) & ~np.isnan(_sa)
        _ct_nan_ind = ~np.isnan(_sigma) & ~np.isnan(_sa)
        if np.sum(_ct_nan_ind) == 0:
            _interp_vals = np.zeros(len(density_levels_2))
            _interp_vals[:] = np.nan
        else:
            _f_temp = scipy.interpolate.interp1d(_sigma[_ct_nan_ind], _ct[_ct_nan_ind], bounds_error=False, fill_value=np.nan)
            _interp_vals = _f_temp(density_levels_2)
        if i_18 == 0:
            ct_interp_1 = _interp_vals
        else:
            ct_interp_1 = np.vstack([ct_interp_1, _interp_vals])
        if np.sum(_ct_nan_ind) == 0:
            _interp_vals = np.zeros(len(density_levels_2))
            _interp_vals[:] = np.nan
        else:
            _f_sa = scipy.interpolate.interp1d(_sigma[_sa_nan_ind], _sa[_sa_nan_ind], bounds_error=False, fill_value=np.nan)
            _interp_vals = _f_sa(density_levels_2)
        if i_18 == 0:
            sa_interp_1 = _interp_vals
        else:
            sa_interp_1 = np.vstack([sa_interp_1, _interp_vals])
    return ct_interp_1, sa_interp_1


@app.cell
def _(ct_interp_1, density_levels_2, np, plt):
    plt.plot(density_levels_2, np.nanmedian(ct_interp_1, axis=0))
    return


@app.cell
def _(ct_interp_1, ecoctd, np, plt, sa_interp_1):
    plt.figure(figsize=(8, 8))
    plt.plot(np.nanmedian(sa_interp_1, axis=0), np.nanmedian(ct_interp_1, axis=0), linewidth=3, linestyle='dashed', color='black')
    plt.scatter(ecoctd.CTD_SA, ecoctd.CTD_CT, s=1)
    return


@app.cell
def _(ct_interp_1, np, sa_interp_1):
    median_S_2 = np.nanmedian(sa_interp_1, axis=0)
    median_T_2 = np.nanmedian(ct_interp_1, axis=0)
    return median_S_2, median_T_2


@app.cell
def _(density_levels_2, median_S_2, median_T_2, np):
    ### Remove NaN for where we don't have enough data to calculate
    density_levels_3 = density_levels_2[~np.isnan(median_S_2)]
    median_T_3 = median_T_2[~np.isnan(median_S_2)]
    median_S_3 = median_S_2[~np.isnan(median_S_2)]
    return density_levels_3, median_S_3, median_T_3


@app.cell
def _(density_levels_3, ecoctd, gsw, median_S_3, median_T_3, np, scipy):
    _f_sa = scipy.interpolate.interp1d(density_levels_3, median_S_3, bounds_error=False, fill_value=np.nan)
    _f_ct = scipy.interpolate.interp1d(density_levels_3, median_T_3, bounds_error=False, fill_value=np.nan)
    spice_2 = ecoctd.CTD_SA.values.copy()
    spice_2[:] = np.nan
    for i_19 in range(ecoctd.dims['n_casts']):
        _cast = ecoctd.isel(n_casts=i_19)
        _sigma = _cast.CTD_Sigma0.values
        _ct = _cast.CTD_CT.values
        _sa = _cast.CTD_SA.values
        _pressure = _cast.pressure.values
        _sa_nan_ind = ~np.isnan(_sigma) & ~np.isnan(_sa)
        _ct_nan_ind = ~np.isnan(_sigma) & ~np.isnan(_sa)
        _alpha = gsw.alpha(_ct, _sa, _pressure)
        _beta = gsw.beta(_ct, _sa, _pressure)
        _spice_i = _alpha * (_ct - _f_ct(_sigma)) + _beta * (_sa - _f_sa(_sigma))
        spice_2[i_19] = _spice_i * (_sigma + 1000)
    return (spice_2,)


@app.cell
def _(ecoctd, spice_2):
    ecoctd['Spice_anomaly'] = (('n_casts', 'n_levels'), spice_2)
    ecoctd['Spice_anomaly'].attrs = {'Full_name': 'Spice anomaly on an isopycnal', 'method': '\\rho_0 $\\alpha\\Delta T + \\beta \\Delta S$ calculated from a reference T-S profile'}
    return


@app.cell
def _(ecoctd, sl_3):
    transect_4 = ecoctd.isel(n_casts=slice(sl_3[0], sl_3[1]))
    return (transect_4,)


@app.cell
def _(dist_grid_2, dists_2, np, plt, pres_grid_2, sns, transect_4):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2_10) = plt.subplots(2, figsize=(11, 10))
    _ax1.tick_params(bottom=True, left=True)
    _spice_transect = transect_4.Spice_anomaly.T
    _vnorm_spice = np.nanmax(np.abs(_spice_transect))
    _cf = _ax1.contourf(dist_grid_2, pres_grid_2, _spice_transect, cmap='RdBu_r', vmin=-_vnorm_spice, vmax=_vnorm_spice, levels=20)
    _ax1.set_ylim([200, 5])
    _ax1.set_xlim([0, np.nanmax(dists_2)])
    cbar_8 = plt.colorbar(_cf, label='Spice (kg/m$^3$)', orientation='vertical', ax=_ax1)
    _ax1.contour(dist_grid_2, pres_grid_2, transect_4.CTD_Sigma0.values.T, levels=np.arange(28.4, 29.1, 0.05), colors='black', linewidths=1)
    _ax1.set_ylabel('Pressure (dbar)')
    ax2_10.tick_params(bottom=True, left=True)
    _cf = ax2_10.contourf(dist_grid_2, pres_grid_2, transect_4.AOU.T, cmap='inferno_r', levels=15, vmax=75)
    ax2_10.set_ylim([200, 5])
    ax2_10.set_xlim([0, np.nanmax(dists_2)])
    plt.colorbar(_cf, label='AOU ($\\mu$mol/L)', orientation='vertical', ax=ax2_10)
    ax2_10.contour(dist_grid_2, pres_grid_2, transect_4.CTD_Sigma0.values.T, levels=np.arange(28.4, 29.1, 0.05), colors='black', linewidths=1)
    ax2_10.set_xlabel('Distance along transect (km)')
    ax2_10.set_ylabel('Pressure (dbar)')
    plt.tight_layout()
    return


@app.cell
def _(np, predicted_nitrate_1, transect_4):
    nit_anom = (predicted_nitrate_1.T - np.nanmean(predicted_nitrate_1, axis=1).T).T
    oxy_anom = (transect_4.OXY_O2sat.values - np.nanmean(transect_4.OXY_O2sat.values.T, axis=1)).T
    chl_anom = (transect_4.FLS_chl.values - np.nanmean(transect_4.FLS_chl.values.T, axis=1)).T
    nit_abs_max = np.ceil(np.max([np.abs(np.nanmax(nit_anom)), np.abs(np.nanmin(nit_anom))]))
    oxy_abs_max = np.ceil(np.max([np.abs(np.nanmax(oxy_anom)), np.abs(np.nanmin(oxy_anom))]))
    chl_abs_max = np.ceil(np.max([np.abs(np.nanmax(chl_anom)), np.abs(np.nanmin(chl_anom))])) / 2
    return chl_abs_max, chl_anom, nit_abs_max, nit_anom, oxy_abs_max, oxy_anom


@app.cell
def _(
    cmocean,
    dist_grid_2,
    dists_2,
    nit_abs_max,
    nit_anom,
    np,
    oxy_abs_max,
    oxy_anom,
    plt,
    pres_grid_2,
    sns,
    transect_4,
):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2_11) = plt.subplots(1, 2, figsize=(22, 10))
    _ax1.tick_params(bottom=True, left=True)
    cp_5 = _ax1.contourf(dist_grid_2, pres_grid_2, nit_anom, cmap=cmocean.cm.curl, levels=np.linspace(-nit_abs_max, nit_abs_max, 20))
    _ax1.set_ylim([250, 0])
    _ax1.set_xlim([0, np.nanmax(dists_2)])
    _cbarticks = np.arange(-nit_abs_max, nit_abs_max + 0.5, 1)
    plt.colorbar(cp_5, label='Nitrate anomaly ($\\mu$M)', orientation='horizontal', ax=_ax1, ticks=_cbarticks)
    _ax1.contour(dist_grid_2, pres_grid_2, transect_4.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    _ax1.set_xlabel('Distance along transect (km)')
    _ax1.set_ylabel('Pressure (dbar)')
    ax2_11.tick_params(bottom=True, left=True)
    cp_5 = ax2_11.contourf(dist_grid_2, pres_grid_2, oxy_anom, cmap=cmocean.cm.curl_r, levels=np.linspace(-oxy_abs_max, oxy_abs_max, 20))
    ax2_11.set_ylim([250, 0])
    ax2_11.set_xlim([0, np.nanmax(dists_2)])
    _cbarticks = np.arange(-oxy_abs_max, oxy_abs_max + 1, 6)
    plt.colorbar(cp_5, label='Dissolved oxygen anomaly ($\\mu$mol/L)', orientation='horizontal', ax=ax2_11, ticks=_cbarticks)
    ax2_11.contour(dist_grid_2, pres_grid_2, transect_4.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    ax2_11.set_xlabel('Distance along transect (km)')
    return


@app.cell
def _(
    chl_abs_max,
    chl_anom,
    cmocean,
    dist_grid_2,
    dists_2,
    mpl,
    nit_abs_max,
    nit_anom,
    np,
    plt,
    pres_grid_2,
    sns,
    transect_4,
):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2_12) = plt.subplots(1, 2, figsize=(22, 10))
    _ax1.tick_params(bottom=True, left=True)
    cp_6 = _ax1.contourf(dist_grid_2, pres_grid_2, nit_anom, cmap=cmocean.cm.curl, levels=np.linspace(-nit_abs_max, nit_abs_max, 15))
    _ax1.set_ylim([250, 0])
    _ax1.set_xlim([0, np.nanmax(dists_2)])
    _cbarticks = np.arange(-nit_abs_max, nit_abs_max + 1, 1)
    plt.colorbar(cp_6, label='Nitrate anomaly ($\\mu$M)', orientation='horizontal', ax=_ax1, ticks=_cbarticks)
    _ax1.contour(dist_grid_2, pres_grid_2, transect_4.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    _ax1.set_xlabel('Distance along transect (km)')
    _ax1.set_ylabel('Pressure (dbar)')
    ax2_12.tick_params(bottom=True, left=True)
    norm = mpl.colors.Normalize(vmin=-18, vmax=18)
    cp_6 = ax2_12.contourf(dist_grid_2, pres_grid_2, chl_anom, cmap=cmocean.cm.tarn, levels=np.linspace(-chl_abs_max, chl_abs_max, 20))
    ax2_12.set_ylim([250, 0])
    ax2_12.set_xlim([0, np.nanmax(dists_2)])
    _cbarticks = np.arange(-chl_abs_max, chl_abs_max + 2, 0.5)
    plt.colorbar(cp_6, label='Chl anomaly', orientation='horizontal', ax=ax2_12, ticks=_cbarticks)
    ax2_12.contour(dist_grid_2, pres_grid_2, transect_4.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    ax2_12.set_xlabel('Distance along transect (km) (mg/m$^3$)')
    return (ax2_12,)


@app.cell
def _(
    ax2_12,
    chl_abs_max,
    cmocean,
    dist_grid_2,
    dists_2,
    np,
    plt,
    pres_grid_2,
    transect_4,
):
    cp_7 = plt.contourf(dist_grid_2, pres_grid_2, transect_4.FLS_chl.values.T, cmap=cmocean.cm.curl_r, levels=20)
    plt.ylim([250, 0])
    plt.xlim([0, np.nanmax(dists_2)])
    _cbarticks = np.arange(-chl_abs_max, chl_abs_max + 2, 0.25)
    plt.colorbar(cp_7, label='Dissolved oxygen ($\\mu$mol/L)', orientation='horizontal', ax=ax2_12, ticks=_cbarticks)
    ax2_12.contour(dist_grid_2, pres_grid_2, transect_4.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    ax2_12.set_xlabel('Distance along transect (km)')
    return


@app.cell
def _(
    cartopy,
    ccrs,
    ecoctd,
    mticker,
    phase1_1,
    phase2_1,
    phase3_1,
    plt,
    transect_4,
):
    plt.figure(figsize=(8, 12))
    ax_27 = plt.axes(projection=ccrs.PlateCarree())
    ax_27.coastlines()
    ax_27.add_feature(cartopy.feature.LAND, edgecolor='black', color='lightgray')
    ax_27.add_feature(cartopy.feature.OCEAN, color='white')
    _gl = ax_27.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    _gl.xlocator = mticker.FixedLocator([2, 3, 4, 4.5])
    _gl.ylocator = mticker.FixedLocator([40, 40.5, 41.5, 42])
    _gl.top_labels = False
    _gl.right_labels = False
    _gl.xlabel_style = {'color': 'black'}
    _gl.xlabel_style = {'color': 'black'}
    ax_27.set_extent([2, 4.5, 40.2, 42])
    ax_27.scatter(ecoctd.CTD_lon[phase1_1], ecoctd.CTD_lat[phase1_1], s=10, color='#88CCEE', label='Phase 1')
    ax_27.scatter(ecoctd.CTD_lon[phase2_1], ecoctd.CTD_lat[phase2_1], s=5, color='#CC6677', label='Phase 2')
    ax_27.scatter(ecoctd.CTD_lon[phase3_1], ecoctd.CTD_lat[phase3_1], s=5, color='#DDCC77', label='Phase 3')
    _lgnd = plt.legend()
    _lgnd.legendHandles[0]._sizes = [100]
    _lgnd.legendHandles[1]._sizes = [100]
    _lgnd.legendHandles[2]._sizes = [100]
    plt.plot(transect_4.CTD_lon, transect_4.CTD_lat, color='black', linewidth=3)
    return


@app.cell
def _():
    ### Plotting a section with locations of bottle samples for respiration measurements
    return


@app.cell
def _():
    i_20 = 123
    return (i_20,)


@app.cell
def _(ctd):
    transect_ctd = ctd.isel(n_casts=slice(32, 33))
    return (transect_ctd,)


@app.cell
def _(ecoctd, haversine_np, i_20, np, pd, transect_ctd, transects_df):
    time_ind_6 = (pd.to_datetime(ecoctd.UTC_time.values) >= transects_df.iloc[i_20]['Initial_Time']) & (pd.to_datetime(ecoctd.UTC_time.values) <= transects_df.iloc[i_20]['Final_Time'])
    sl_4 = (np.where(time_ind_6)[0][0], np.where(time_ind_6)[0][-1] + 1)
    transect_5 = ecoctd.isel(n_casts=slice(sl_4[0], sl_4[1]))
    dates_3 = transect_5.UTC_time.values
    pres_grid_3 = np.repeat(transect_5.pressure.values, transect_5.dims['n_casts']).reshape((transect_5.dims['n_levels'], transect_5.dims['n_casts']))
    _start_lon, _start_lat = (transect_5.isel(n_casts=0)['CTD_lon'].values, transect_5.isel(n_casts=0)['CTD_lat'].values)
    dists_3 = haversine_np(_start_lon, _start_lat, transect_5['CTD_lon'].values[:], transect_5['CTD_lat'].values[:])
    dists_ctd = haversine_np(_start_lon, _start_lat, transect_ctd['CTD_lon'].values[:], transect_ctd['CTD_lat'].values[:])
    dist_grid_3 = np.tile(dists_3, pres_grid_3.shape[0]).reshape(pres_grid_3.shape)
    return dist_grid_3, dists_3, dists_ctd, pres_grid_3, transect_5


@app.cell
def _(transect_5):
    transect_5
    return


@app.cell
def _(
    cmocean,
    dist_grid_3,
    dists_3,
    dists_ctd,
    np,
    plt,
    pres_grid_3,
    sns,
    transect_5,
):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2_13) = plt.subplots(1, 2, figsize=(20, 10))
    _ax1.tick_params(bottom=True, left=True)
    chl_9 = transect_5.FLS_chl.values.T + np.abs(np.nanmin(transect_5.FLS_chl.values.T))
    cp_8 = _ax1.contourf(dist_grid_3, pres_grid_3, np.log10(chl_9), cmap=cmocean.cm.speed, levels=10)
    _ax1.set_ylim([250, 0])
    _ax1.set_xlim([0, np.nanmax(dists_3)])
    plt.colorbar(cp_8, label='Log-normalized chlorophyll (mg/m$^3$)', orientation='horizontal', ax=_ax1)
    _ax1.contour(dist_grid_3, pres_grid_3, transect_5.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    _ax1.set_xlabel('Distance along transect (km)')
    _ax1.set_ylabel('Pressure (dbar)')
    ax2_13.tick_params(bottom=True, left=True)
    cp_8 = ax2_13.contourf(dist_grid_3, pres_grid_3, transect_5.OXY_O2sat.values.T, cmap=cmocean.cm.thermal_r, levels=10)
    ax2_13.set_ylim([250, 0])
    ax2_13.set_xlim([0, np.nanmax(dists_3)])
    plt.colorbar(cp_8, label='Dissolved oxygen (% saturation)', orientation='horizontal', ax=ax2_13)
    ax2_13.contour(dist_grid_3, pres_grid_3, transect_5.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    ax2_13.set_xlabel('Distance along transect (km)')
    for ax_28 in [_ax1, ax2_13]:
        ax_28.scatter(dists_ctd, np.repeat(2, len(dists_ctd)), marker='v', s=1000, zorder=100, color='black')
        ax_28.tick_params(bottom=True, left=True)
        ax_28.scatter([dists_ctd[0] + 0.15, dists_ctd[0] + 0.15], [70, 200], zorder=10, color='yellow', s=150, marker='X')
    return


@app.cell
def _(
    cmocean,
    dist_grid_3,
    dists_3,
    dists_ctd,
    np,
    plt,
    pres_grid_3,
    sns,
    transect_5,
):
    sns.set(style='white', font_scale=2, font='Verdana')
    _fig, (_ax1, ax2_14) = plt.subplots(1, 2, figsize=(18, 10))
    _ax1.tick_params(bottom=True, left=True)
    cp_9 = _ax1.contourf(dist_grid_3, pres_grid_3, transect_5.CTD_CT.values.T, cmap=cmocean.cm.thermal, levels=10)
    _ax1.set_ylim([250, 0])
    _ax1.set_xlim([0, np.nanmax(dists_3)])
    plt.colorbar(cp_9, label='Log-normalized chlorophyll (mg/m$^3$)', orientation='horizontal', ax=_ax1)
    _ax1.contour(dist_grid_3, pres_grid_3, transect_5.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    _ax1.set_xlabel('Distance along transect (km)')
    _ax1.set_ylabel('Pressure (dbar)')
    ax2_14.tick_params(bottom=True, left=True)
    cp_9 = ax2_14.contourf(dist_grid_3, pres_grid_3, transect_5.CTD_SA.values.T, cmap=cmocean.cm.haline, levels=10)
    ax2_14.set_ylim([250, 0])
    ax2_14.set_xlim([0, np.nanmax(dists_3)])
    plt.colorbar(cp_9, label='Dissolved oxygen ($\\mu$mol/L)', orientation='horizontal', ax=ax2_14)
    ax2_14.contour(dist_grid_3, pres_grid_3, transect_5.CTD_Sigma0.values.T, levels=15, colors='black', linewidths=1)
    ax2_14.set_xlabel('Distance along transect (km)')
    for ax_29 in [_ax1, ax2_14]:
        ax_29.scatter(dists_ctd, np.repeat(2, len(dists_ctd)), marker='v', s=1000, zorder=100, color='black')
        ax_29.tick_params(bottom=True, left=True)
        ax_29.scatter([dists_ctd[0] + 0.15, dists_ctd[0] + 0.15], [70, 200], zorder=10, color='yellow', s=150, marker='X')
    return


@app.cell
def _(plt, transect_5, transect_ctd):
    plt.scatter(transect_5.CTD_lon, transect_5.CTD_lat)
    plt.scatter(transect_ctd.CTD_lon, transect_ctd.CTD_lat)
    return


@app.cell
def _():
    ### Plotting CTD transect
    return


@app.cell
def _(ctd):
    transect_6 = ctd.isel(n_casts=slice(11, 17))
    return (transect_6,)


@app.cell
def _(haversine_np, np, transect_6):
    dates_4 = transect_6.UTC_time.values
    pres_grid_4 = np.repeat(transect_6.pressure.values, transect_6.dims['n_casts']).reshape((transect_6.dims['n_levels'], transect_6.dims['n_casts']))
    _start_lon, _start_lat = (transect_6.isel(n_casts=0)['CTD_lon'].values, transect_6.isel(n_casts=0)['CTD_lat'].values)
    dists_4 = haversine_np(_start_lon, _start_lat, transect_6['CTD_lon'].values[:], transect_6['CTD_lat'].values[:])
    dist_grid_4 = np.tile(dists_4, pres_grid_4.shape[0]).reshape(pres_grid_4.shape)
    return dist_grid_4, dists_4, pres_grid_4


@app.cell
def _(sns):
    sns.set(style="white", font_scale=3.5, font="Verdana")
    return


@app.cell
def _(cmocean, dist_grid_4, dists_4, np, plt, pres_grid_4, transect_6):
    _fig, ((_ax0, _ax1), (ax2_15, _ax3)) = plt.subplots(2, 2, figsize=(48, 22))
    _pres_ind = np.inf
    for i_21 in range(transect_6.dims['n_casts']):
        _pres_vals = transect_6.isel(n_casts=i_21).pressure.values
        _pres_ind_temp = np.nanargmax(np.isnan(_pres_vals))
        _pres_ind = int(np.nanmin([_pres_ind, _pres_ind_temp]))
    cp_10 = _ax0.contourf(dist_grid_4, pres_grid_4, transect_6.CTD_CT1.values.T, levels=10, cmap=cmocean.cm.thermal)
    _ax0.contour(dist_grid_4, pres_grid_4, transect_6.CTD_Sigma01.values.T, levels=15, colors='black', linewidths=1)
    _ax0.set_ylim([250, 0])
    _ax0.set_xlim([0, np.nanmax(dists_4)])
    plt.colorbar(cp_10, label='Conservative temperature ($\\degree$C)', ax=_ax0)
    _ax0.set_xlabel('Distance along transect (km)')
    _ax0.set_ylabel('Pressure (dbar)')
    cp_10 = _ax1.contourf(dist_grid_4, pres_grid_4, transect_6.CTD_SA1.values.T, levels=10, cmap=cmocean.cm.haline)
    _ax1.contour(dist_grid_4, pres_grid_4, transect_6.CTD_Sigma01.values.T, levels=15, colors='black', linewidths=1)
    _ax1.set_ylim([250, 0])
    _ax1.set_xlim([0, np.nanmax(dists_4)])
    plt.colorbar(cp_10, label='Absolute salinity', ax=_ax1)
    _ax1.set_xlabel('Distance along transect (km)')
    _ax1.set_ylabel('Pressure (dbar)')
    cp_10 = ax2_15.contourf(dist_grid_4, pres_grid_4, transect_6.OXY_O2sat.values.T, levels=10, cmap='rainbow')
    ax2_15.contour(dist_grid_4, pres_grid_4, transect_6.CTD_Sigma01.values.T, levels=15, colors='black', linewidths=1)
    ax2_15.set_ylim([250, 0])
    ax2_15.set_xlim([0, np.nanmax(dists_4)])
    plt.colorbar(cp_10, label='Dissolved $O_2$ (% saturation)', ax=ax2_15)
    ax2_15.set_xlabel('Distance along transect (km)')
    ax2_15.set_ylabel('Pressure (dbar)')
    _fl = transect_6.FLS_chl.values.T
    _fl[_fl < 0] = 0
    _fl = np.log10(_fl)
    _fl_min = np.nanmin(_fl[~np.isinf(_fl)])
    _fl[np.isinf(_fl)] = _fl_min
    cp_10 = _ax3.contourf(dist_grid_4, pres_grid_4, _fl, levels=10, cmap=cmocean.cm.algae, vmax=2)
    _ax3.contour(dist_grid_4, pres_grid_4, transect_6.CTD_Sigma01.values.T, levels=15, colors='black', linewidths=1)
    _ax3.set_ylim([250, 0])
    _ax3.set_xlim([0, np.nanmax(dists_4)])
    plt.colorbar(cp_10, label='Log10(Chl) (mg/$m^3$)', ax=_ax3)
    _ax3.set_xlabel('Distance along transect (km)')
    _ax3.set_ylabel('Pressure (dbar)')
    for ax_30 in [_ax0, _ax1, ax2_15, _ax3]:
        ax_30.scatter(dists_4, np.repeat(2, len(dists_4)), marker='v', s=1000, zorder=100, color='black')
        ax_30.tick_params(bottom=True, left=True)
    return


@app.cell
def _(np, transect_6):
    _fl = transect_6.FLS_chl.values.T
    _fl[_fl < 0] = 0
    _fl = np.log10(_fl)
    _fl_min = np.nanmin(_fl[~np.isinf(_fl)])
    _fl[np.isinf(_fl)] = _fl_min
    return


@app.cell
def _(np):
    np.isinf(np.log10(0))
    return


@app.cell
def _(loadmat):
    loadmat('/Volumes/science/CALYPSO/ecoCTD_Data/*')
    return


@app.cell
def _(glob):
    glob.glob('/Volumes/science/CALYPSO/ecoCTD_Data/BH_processed/*.mat')
    return


@app.cell
def _(loadmat):
    ecoCTD = loadmat('/Volumes/science/CALYPSO/ecoCTD_Data/BH_processed/All_ECO_gridded.mat')
    return (ecoCTD,)


@app.cell
def _(ecoCTD, np):
    np.argmin(np.abs(ecoCTD['LAT'] - 40.46))
    return


@app.cell
def _(ecoCTD, np):
    np.nanargmin(np.abs(ecoCTD['LAT'][0] - 40.46))
    return


@app.cell
def _(ecoCTD):
    ecoCTD['LAT'][0][620:639]
    return


@app.cell
def _(ecoCTD, plt):
    plt.scatter(ecoCTD['LON'][0][537:554], ecoCTD['LAT'][0][537:554], s=5)
    ind_1 = [537, 554]
    return (ind_1,)


@app.cell
def _(ecoCTD):
    ecoCTD.keys()
    return


@app.cell
def _(ecoCTD):
    pres = ecoCTD['p'].flatten()
    dates_5 = ecoCTD['D'].flatten()
    return dates_5, pres


@app.cell
def _(np):
    def haversine_np_1(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)

        All args must be of equal length.    

        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km

    return (haversine_np_1,)


@app.cell
def _(ecoCTD, haversine_np_1, ind_1):
    lon0 = ecoCTD['LON'][0][ind_1[1] - 1]
    lat0 = ecoCTD['LAT'][0][ind_1[1] - 1]
    lat0, lon0 = (40 + 39.101 / 60, 2 + 41.655 / 60)  #alat, alon
    dists_5 = haversine_np_1(lon0, lat0, ecoCTD['LON'][0][ind_1[0]:ind_1[1]], ecoCTD['LAT'][0][ind_1[0]:ind_1[1]])
    return dists_5, lat0, lon0


@app.cell
def _(dists_5):
    dists_5
    return


@app.cell
def _(ecoCTD, ind_1):
    ecoCTD['LON'][0][ind_1[0]:ind_1[1]]
    return


@app.cell
def _(ecoCTD, ind_1):
    ecoCTD['LAT'][0][ind_1[0]:ind_1[1]]
    return


@app.cell
def _(np):
    110*np.sqrt((40.46270626 - 40.58712048)**2 + (2.78345709 - 2.73433276)**2)
    return


@app.cell
def _(dates_5, dists_5, ind_1, np, pres):
    date_grid, pres_grid_5 = np.meshgrid(dates_5[ind_1[0]:ind_1[1]], pres)
    dist_grid_5, pres_gird = np.meshgrid(dists_5, pres)
    return dist_grid_5, pres_grid_5


@app.cell
def _(dist_grid_5, ecoCTD, ind_1, np, plt, pres_grid_5):
    plt.figure(figsize=(12, 6))
    plt.contourf(dist_grid_5, pres_grid_5, np.log(ecoCTD['chl'][:, ind_1[0]:ind_1[1]]), cmap='jet', levels=50)
    plt.ylim([250, 0])
    plt.xlabel('Distance (km)')
    plt.ylabel('Pressure (dbar)')
    return


@app.cell
def _():
    lats, lons = [], []

    alat, alon = 40 + 39.101/60,   2 + 41.655/60
    blat, blon = 40 + 37.617/60,   2 + 42.650/60 
    clat, clon = 40 + 35.433/60,   2 + 43.800/60
    dlat, dlon = 40 + 33.694/60, 2 + 44.583/60
    elat, elon = 40 + 31.208/60, 2 + 45.661/60
    flat, flon = 40 + 28.716/60, 2 + 46.593/60

    lats = [alat, blat, clat, dlat, elat, flat]
    lons = [alon, blon, clon, dlon, elon, flon]
    return lats, lons


@app.cell
def _(haversine_np_1, lat0, lats, lon0, lons):
    ctd_dists = haversine_np_1(lon0, lat0, lons, lats)
    return (ctd_dists,)


@app.cell
def _(ecoCTD, lat0, lats, lon0, lons, plt):
    plt.figure(figsize=(4, 9))
    plt.scatter(lons, lats)
    plt.scatter(ecoCTD['LON'][0][537:555], ecoCTD['LAT'][0][537:555], s=5, zorder=10)
    ind_2 = [537, 555]
    plt.scatter(lon0, lat0)
    return (ind_2,)


@app.cell
def _(lons):
    lons
    return


@app.cell
def _(lats):
    lats
    return


@app.cell
def _(ctd_dists):
    sorted(ctd_dists)
    return


@app.cell
def _(dists_5):
    dists_5
    return


@app.cell
def _(ctd_dists, dist_grid_5, dists_5, ecoCTD, ind_2, np, plt, pres_grid_5):
    plt.figure(figsize=(12, 6))
    plt.contourf(dist_grid_5, pres_grid_5, np.log(ecoCTD['chl'][:, ind_2[0]:ind_2[1]]), cmap='jet', levels=50)
    plt.colorbar(label='Log-normalized Chl')
    plt.ylim([250, 0])
    plt.xlabel('Distance (km)')
    plt.ylabel('Pressure (dbar)')
    for _d in ctd_dists:
        plt.scatter(_d, -2, marker='v', s=500, color='black', zorder=10)
    for i_22 in [-4, -5]:
        plt.scatter(dists_5[i_22], -2, marker='v', s=500, color='green', zorder=10)
    return


@app.cell
def _(ctd_dists, dist_grid_5, dists_5, ecoCTD, ind_2, plt, pres_grid_5):
    plt.figure(figsize=(12, 6))
    plt.contourf(dist_grid_5, pres_grid_5, ecoCTD['o2'][:, ind_2[0]:ind_2[1]], cmap='jet', levels=50)
    plt.colorbar(label='O2 percent saturation')
    plt.ylim([250, 0])
    plt.xlabel('Distance (km)')
    plt.ylabel('Pressure (dbar)')
    for _d in ctd_dists:
        plt.scatter(_d, -2, marker='v', s=500, color='black', zorder=10)
    for i_23 in range(len(dists_5)):
        plt.scatter(dists_5[i_23], -2, marker='v', s=500, color='green', zorder=10)
    return


@app.cell
def _(ctd_dists, dist_grid_5, ecoCTD, ind_2, plt, pres_grid_5):
    plt.figure(figsize=(12, 6))
    plt.contourf(dist_grid_5, pres_grid_5, ecoCTD['t'][:, ind_2[0]:ind_2[1]], cmap='jet', levels=50)
    plt.colorbar(label='Temperature (deg C)')
    plt.ylim([250, 0])
    plt.xlabel('Distance (km)')
    plt.ylabel('Pressure (dbar)')
    for _d in ctd_dists:
        plt.scatter(_d, -2, marker='v', s=500, color='black', zorder=10)
    return


@app.cell
def _(ctd_dists, dist_grid_5, ecoCTD, ind_2, plt, pres_grid_5):
    plt.figure(figsize=(12, 6))
    plt.contourf(dist_grid_5, pres_grid_5, ecoCTD['s'][:, ind_2[0]:ind_2[1]], cmap='jet', levels=50)
    plt.colorbar(label='Salinity (psu)')
    plt.ylim([250, 0])
    plt.xlabel('Distance (km)')
    plt.ylabel('Pressure (dbar)')
    for _d in ctd_dists:
        plt.scatter(_d, -2, marker='v', s=500, color='black', zorder=10)
    return


@app.cell
def _(ecoCTD):
    ecoCTD.keys()
    return


@app.cell
def _(ecoCTD):
    len(ecoCTD['LON'][0][:])
    return


@app.cell
def _():
    lon, lat = 4.30228333333333335, 41.62558333333
    return lat, lon


@app.cell
def _(directory, loadmat):
    gridded_ecoCTD = loadmat(directory + 'All_ECO_gridded.mat')
    return (gridded_ecoCTD,)


@app.cell
def _(gridded_ecoCTD):
    sp = gridded_ecoCTD['s']
    t = gridded_ecoCTD['t']
    p_2 = gridded_ecoCTD['p']
    return p_2, sp, t


@app.cell
def _(gridded_ecoCTD, np, plt):
    plt.pcolormesh(np.flipud(gridded_ecoCTD['t']))
    return


@app.cell
def _(gsw, p_2, sp, t):
    SA = gsw.SA_from_SP(sp, p_2, 4.3022833333333335, 41.62558333333)
    CT = gsw.CT_from_t(SA, t, p_2)
    return CT, SA


@app.cell
def _(gsw):
    dir(gsw)
    return


@app.cell
def _(gridded_ecoCTD, np, plt):
    plt.pcolormesh(np.flipud(gridded_ecoCTD['s']))
    return


@app.cell
def _(gridded_ecoCTD):
    o2 = gridded_ecoCTD['o2']
    return (o2,)


@app.cell
def _(CT, SA, gsw, lat, lon, p_2):
    o2_sat = gsw.O2sol(SA, CT, p_2, lon, lat)
    return (o2_sat,)


@app.cell
def _(o2, o2_sat):
    o2_conc = o2*o2_sat/100
    return (o2_conc,)


@app.cell
def _(np, o2_conc, plt):
    plt.pcolormesh(np.flipud(o2_conc))
    plt.colorbar()
    return


@app.cell
def _(directory, xr):
    Cast4 = xr.open_dataset(directory + 'dcast004_190222cfacldb.nc')
    return (Cast4,)


@app.cell
def _(Cast1):
    Cast1
    return


@app.cell
def _(Cast4):
    list(Cast4.data_vars)
    return


@app.cell
def _(Cast1):
    Cast1['sigma-�00']
    return


@app.cell
def _(Cast1, plt):
    plt.figure(figsize=(6,12))
    plt.plot(Cast1['flECO-AFL'], Cast1.PRES, color='red')
    # plt.xlim([13, 14])
    plt.ylim(630, 0)
    return


@app.cell
def _(Cast1, plt):
    _fig, ax_31 = plt.subplots(figsize=(6, 12))
    plt.plot(Cast1.TEMP, Cast1.PRES, color='red')
    ax_31.set_xlim([13, 14])
    plt.ylim(630, 0)
    _ax1 = ax_31.twiny()
    _ax1.plot(Cast1['sigma-�00'], Cast1.PRES, color='black')
    _ax1.set_xlim([29.02, 29.1])
    ax2_16 = ax_31.twiny()
    ax2_16.plot(Cast1['flECO-AFL'], Cast1.PRES, color='green')
    return


@app.cell
def _(Cast1, plt):
    _fig, ax_32 = plt.subplots(figsize=(6, 12))
    plt.plot(Cast1.TEMP, Cast1.PRES, color='red')
    ax_32.set_xlim([13, 14])
    plt.ylim(630, 0)
    _ax1 = ax_32.twiny()
    _ax1.plot(Cast1['sigma-�00'], Cast1.PRES, color='black')
    _ax1.set_xlim([29.02, 29.1])
    return


@app.cell
def _(Cast1, np):
    cp_11 = Cast1.CStarTr0.copy()
    cp_11 = -1 / 0.25 * np.log(cp_11 / 100) - 0.1552
    POCcp = cp_11 / 1.78  # g C/m^3
    POCcp = POCcp / 12.011  # mol C/m^3
    POCcp = POCcp * 1000000.0 / 1027  # umol C/kg
    return (POCcp,)


@app.cell
def _(Cast1, POCcp, plt):
    plt.plot(POCcp, Cast1.PRES)
    plt.ylim([600, 0])
    return


@app.cell
def _():
    1/1.78
    return


@app.cell
def _(Cast1):
    Cast1.CStarTr0.copy()
    return


@app.cell
def _(Cast1, np):
    cp_12 = Cast1.CStarTr0.copy()
    -1 / 0.25 * np.log(cp_12 / 100)
    return


@app.cell
def _(Cast1, np):
    cp_13 = Cast1.CStarTr0.copy()
    cp_13 = -1 / 0.25 * np.log(cp_13 / 100)
    yy = 0.00206 * cp_13 + 0.0147
    return (cp_13,)


@app.cell
def _(cp_13):
    # yy *= .001
    # yy = yy/12.011; # mol C/m^3
    # yy = yy*1e6/1027; # umol C/kg
    yy_1 = (cp_13 - 0.0147) / 0.002
    return (yy_1,)


@app.cell
def _(Cast1, cp_13, plt):
    plt.plot(Cast1.CStarAt0)
    plt.plot(cp_13)
    return


@app.cell
def _(Cast1, plt, yy_1):
    plt.plot(yy_1, Cast1.PRES)
    # plt.plot(yy, Cast1.PRES)
    # plt.ylim([600, 0])
    plt.ylim([600, 0])
    return


if __name__ == "__main__":
    app.run()
