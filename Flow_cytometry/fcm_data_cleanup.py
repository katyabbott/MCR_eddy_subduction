import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


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

    return np, pd


@app.cell
def _():
    wkdir = '/Users/katyabbott/Documents/MIT_WHOI_Joint_Program/Research_Projects/Microbial_respiration_CALYPSO/'
    fcm_dir = 'Data/Flow cytometry/'
    return fcm_dir, wkdir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Starting with picophytoplankton data
    """)
    return


@app.cell
def _(wkdir):
    # Location of log files from AutoBOD

    output_dir = wkdir + 'Analyses/Respiration/Output/'

    # resp_df_pooled = pd.read_csv(output_dir + 'CALYPSO_all_respiration_rates_QC_with_environmental_data_pooled.csv')
    # resp_df_pooled = resp_df_pooled.drop('Unnamed: 0', axis=1)
    # resp_df_pooled = resp_df_pooled.rename({'Pooled_mean_rate': 'Mean_rate', 'Pooled_sd': 'Std_rate'}, axis=1)
    return


@app.cell
def _(fcm_dir, pd, wkdir):
    pt_fcm_df = pd.read_excel(wkdir + fcm_dir + 'Processed_data/fcm_casts_from_Rachele_051723_KA_edited_120624.xlsx', sheet_name=0)
    pt_fcm_df = pt_fcm_df[pt_fcm_df['Weighted vol'] != 0] #Not sure why some of these samples have no volume listed

    pt_fcm_df.loc[pt_fcm_df['Weighted vol'] == '201,6', 'Weighted vol'] = 201.6

    cast_df = pd.read_excel(wkdir + 'Data/Metadata/CTD_cast_table.xlsx',sheet_name=1)
    cast_df = cast_df.dropna(how='all')
    dilution_factor = 1010/1000*500/490
    pt_fcm_df['Cells-mL'] = pt_fcm_df['Region']*dilution_factor*1000/pt_fcm_df['Weighted vol']
    phototroph_names = {'Syn': 'Synechococcus', 'Euks': 'Picoeukaryotes', 'Pro': 'Prochlorococcus'}
    # phase1_df = pt_fcm_df[pt_fcm_df.Cast.isin(range(1,10))]
    # phase2_df = pt_fcm_df[pt_fcm_df.Cast.isin(range(10,36))]
    # phase3_df = pt_fcm_df[pt_fcm_df.Cast.isin(range(36,62))]
    # phase2_3_df = pt_fcm_df[pt_fcm_df.Cast.isin(range(10,62))]

    pt_fcm_df = pt_fcm_df[['Cast', 'Sample', 'Depth', 'Alias', 'Cells-mL']].copy()
    return (pt_fcm_df,)


@app.cell
def _(pt_fcm_df):
    pt_fcm_df.columns
    return


@app.cell
def _():
    #pt_fcm_df.to_csv(wkdir + fcm_dir + 'Processed_data/cleaned_data_for_NGS/Calypso2022_FCM_phototrophs.csv')
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(fcm_dir, np, pd, wkdir):
    ### Bacteria data

    fcm_df = pd.read_excel(wkdir + fcm_dir + 'Processed_data/FlowJo_results/Bacteria/Bacteria_sample_stats_formatted_for_Python_070224.xlsx')

    ### Drop rows where the alignment was not good
    fcm_df = fcm_df[fcm_df['Alignment notes'].isna()]
    fcm_df = fcm_df.reset_index()

    fcm_df = fcm_df.dropna(subset=['Syn_cells-mL']) # drop the one sample where we have no weight data --> no cells

    ### Drop columns that are not needed for this analysis
    fcm_df = fcm_df.drop(['FlowJo workspace', 'Stained Sample Vol (µl)',
           'YG Beads Vol (µl)', 'Total Vol (µl) ', 'Glut correction',
           'SYBR Green correction', 'Sample Dilution', 'Vol weighted (µl)',
           'All YG beads', 'YG.FSC.692', 
           'All_YG_beads_cells-mL', 'YG.FSC.692_cells-mL',  'YG.FSC.692_FSC_geomean', 'Bacteria_FSC_geomean'], axis=1)           

    fcm_df['Timepoint (h)'] = np.round(fcm_df['Timepoint (h)']) 

    #cast_fcm_df = fcm_df[fcm_df['Timepoint (h)'] == 0]

    fcm_df['Cast_Niskin_Timepoint'] = fcm_df['Cast_Niskin'].astype('str') + '_T' + fcm_df['Timepoint (h)'].astype('str')

    mapping_df = pd.read_excel(wkdir + 'Data/Metadata/Resp_Bio_Niskin_mapping.xls', sheet_name='mapping')
    mapping_df['Depth'] = mapping_df['Depth'].astype('float64')
    fcm_df_mapped = fcm_df.rename({'Depth_mapped_from_Worden': 'Depth'}, axis=1)
    fcm_df_mapped = fcm_df_mapped.dropna(subset=['Cast'])
    fcm_df_mapped['Cast'] = fcm_df_mapped['Cast'].astype('int')

    fcm_df_mapped = pd.merge(fcm_df_mapped, mapping_df, how='left', on=['Cast', 'Depth'])
    fcm_df_mapped['Respiration_msmt'] = ~fcm_df_mapped['Resp_Niskin'].isna()
    #fcm_df_mapped = fcm_df_mapped.dropna(subset=['Resp_Niskin'])
    # fcm_df_mapped['Resp_Niskin'] = fcm_df_mapped['Resp_Niskin'].astype('int')
    fcm_df_mapped['Cast_Niskin'] = 'C' + fcm_df_mapped['Cast'].astype('str') + 'N' + fcm_df_mapped['Resp_Niskin'].astype('str')
    # #fcm_df_mapped['Timepoint (h)'] = np.round(fcm_df_mapped['Timepoint (h)'])  
    return (fcm_df_mapped,)


@app.cell
def _(fcm_df_mapped, fcm_dir, wkdir):
    ### columns we don't need: 'Syn', 'Bacteria', '
    bacteria_fcm_csv = fcm_df_mapped[['Sample_name', 'Cast', 'Niskin', 'Cast_Niskin', 'Depth', 'Timepoint (h)', 'Respiration_msmt', 'Bacteria_cells-mL', 'Bacteria_FSC_normalized_geomean']].copy()

    bacteria_fcm_csv.to_csv(wkdir + fcm_dir + 'Processed_data/cleaned_data_for_NGS/Calypso2022_FCM_bacteria.csv')
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
