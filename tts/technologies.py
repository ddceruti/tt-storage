
import os
import warnings

import pandas as pd
import xarray as xr
import numpy as np

from tts.matrix_collection import MatrixCollection


def annuity(c_i, n):
    """Calculate the annuity factor.

    Args:
        c_i (float): Interest rate
        n (float): Number of years

    Returns:
        float: annuity factor
    """
    a = ((1 + c_i) ** n * c_i) / ((1 + c_i) ** n - 1)
    return a


def pipes(path: os.path.abspath,
          indices=slice(None),
          **kwargs):
    """Read the regression coefficients for the thermal capacity and heat
    losses from csv file.
    """
    # read df and force floats
    df = pd.read_csv(path, **kwargs).loc[indices, :]
    d = {}
    d['invest'] = {"a": df.loc["capacity_a"], "b": df.loc["capacity_b"]}
    d['losses'] = {"a": df.loc["heat_loss_a"], "b": df.loc["heat_loss_b"]}
    d["power_flow_max_kW"] = df.loc["power_flow_max_kW"]
    d["power_flow_max_partload"] = df.loc["power_flow_max_partload"]
    return d


def storages(path: os.path.abspath, indices=slice(None), **kwargs):
    s = {}
    try:
        df = pd.read_csv(path, **kwargs).loc[indices, :]
    except KeyError as exc:
        raise ValueError(f'Storage {indices} must be provided in csv file') from exc

    s['ratios'] = df.loc[:, ('s_ratio_charge', 's_ratio_discharge')].rename(
        columns={'s_ratio_charge': 'charge', 's_ratio_discharge': 'discharge'}).stack()
    s['ratios'].index.names = ['storages', 'c']
    s['ratios'] = xr.DataArray.from_series(s['ratios'])

    s['s_eff'] = df.loc[:, ('charge_eff', 'discharge_eff')].rename(
        columns={'charge_eff': 'charge', 'discharge_eff': 'discharge'}).unstack()
    s['s_eff'].index.names = ['c', 'storages']
    s['s_eff'] = xr.DataArray.from_series(s['s_eff'])

    s['s_self_disch'] = xr.DataArray.from_series(df['self_discharge'])

    s['s_startend'] = xr.DataArray.from_series(df['starting_ending_state_of_charge'])

    df['discounted_inv'] = calc_invest(n_lifetime=df['lifetime'],
                                       discount_rate=df['discount_rate'],
                                       investment_costs=df['investment_costs'])
    s['inv_s'] = xr.DataArray.from_series(df['discounted_inv'])
    return s


def calc_invest(n_lifetime, discount_rate, investment_costs):
    crf = annuity(n=n_lifetime, c_i=discount_rate)
    return investment_costs * crf


def efficiencies(path: os.path.abspath,
                 indices=slice(None),
                 timeseries=None,
                 **kwargs):
    try:
        effs = pd.read_csv(path, **kwargs).loc[indices['producers'], :]
    except KeyError as exc:
        raise ValueError(f'Producer {indices} must be provided in csv file') from exc

    if effs.loc[:, ['efficiency', 'efficiency_ts']].isna().all().all():
        raise ValueError("Efficiency must be provided as constant or time series")

    eta_p = xr.DataArray(np.nan,
                         dims=['producers', 'periods', 'steps'],
                         coords={'producers': effs.index,
                                 'periods': indices['periods'],
                                 'steps': indices['steps']},
                         name='efficiency')

    for i, row in effs.iterrows():
        if pd.isna(row.efficiency_ts) and ~np.isnan(row.efficiency):
            eta_p.loc[{'producers': i}] = row.efficiency
        elif ~pd.isna(row.efficiency_ts) and np.isnan(row.efficiency):
            eta_p.loc[{'producers': i}] = timeseries[row.efficiency_ts]
        else:
            raise ValueError(
                f'Efficiency of producer {i} must be provided as constant or time series')
    return eta_p


def availability(path: os.path.abspath,
                 indices=slice(None),
                 timeseries=None,
                 **kwargs):
    """Read the availability of the producers from csv file."""
    try:
        df_avail = pd.read_csv(path, **kwargs).loc[indices['producers'], :]
    except KeyError:
        raise ValueError(f'Producer {indices} must be provided in csv file')

    da = xr.DataArray(1.,
                         dims=['producers', 'periods', 'steps'],
                         coords={'producers': df_avail.index,
                                 'periods': indices['periods'],
                                 'steps': indices['steps']},
                         name='availability')

    for i, row in df_avail.iterrows():
        if not pd.isna(row.availability_ts):
            da.loc[{'producers': i}] = timeseries[row.availability_ts]
    return da


def producers(path: os.path.abspath, indices=slice(None), **kwargs):
    df = pd.read_csv(path, **kwargs).loc[indices, :]
    df['discounted_inv'] = calc_invest(n_lifetime=df['lifetime'],
                                       discount_rate=df['discount_rate'],
                                       investment_costs=df['investment_costs'])
    inv_costs_gen = xr.DataArray.from_series(df['discounted_inv'])
    return inv_costs_gen, df['name']


def operating_costs(path: os.path.abspath,
                    indices=slice(None),
                    timeseries=None,
                    **kwargs):
    df_costs = pd.read_csv(path, **kwargs).loc[indices['producers'], :]
    if df_costs.loc[:, ['operating_costs_ts', 'operating_costs']].isna().all().all():
        raise ValueError("Operating costs must be provided as constant or time series")
    op = xr.DataArray(np.nan,
                         dims=['producers', 'periods', 'steps'],
                         coords={'producers': df_costs.index,
                                 'periods': indices['periods'],
                                 'steps': indices['steps']},
                         name='operating_costs')

    for i, row in df_costs.iterrows():
        if pd.isna(row.operating_costs_ts) and ~np.isnan(row.operating_costs):
            op.loc[{'producers': i}] = row.operating_costs
        elif ~pd.isna(row.operating_costs_ts) and np.isnan(row.operating_costs):
            op.loc[{'producers': i}] = timeseries[row.operating_costs_ts]
        elif ~pd.isna(row.operating_costs_ts) and ~pd.isna(row.operating_costs):
            op.loc[{'producers': i}] = row.operating_costs
            warnings.warn(f'Operating costs of producer {i} provided as constant and time series. Using constant value.')
        else:
            raise ValueError(
                f'Operating costs of producer {i} must be provided either as constant or time series')
    return op

def read_parameters(path: os.path.abspath,
                    collection: MatrixCollection,
                    timeseries: xr.Dataset = None):
    params = {}
    # check that timeseries periods and steps match the indices
    if timeseries is not None:
        if not timeseries.coords['periods'].isin(collection.indices['periods']).all().item():
            raise ValueError("Time series periods must match the indices")
        if not timeseries.coords['steps'].isin(collection.indices['steps']).all().item():
            raise ValueError("Time series steps must match the indices")

    params['storages'] = storages(
        path=os.path.join(path, 'storages.csv'),
        indices=collection.indices['storages'],
        index_col=0, header=0)
    # @TODO if constant, define in producers.csv, if variable, provide in trimeseries
    params['eta_p'] = efficiencies(
        path=os.path.join(path, 'producers.csv'),
        indices=collection.indices,
        timeseries=timeseries,
        index_col=0, header=0, sep=',')
    params['availability'] = availability(
        path=os.path.join(path, 'producers.csv'),
        indices=collection.indices,
        timeseries=timeseries,
        index_col=0, header=0, sep=',')
    params['inv_p'], params['producer_names'] = producers(
        path=os.path.join(path, 'producers.csv'),
        indices=collection.indices['producers'],
        index_col=0, header=0)
    params['pipes'] = pipes(
        path=os.path.join(path, 'pipes.csv'),
        indices=0,
        sep=',', index_col=0, header=0, dtype=float)
    params['G_max_producers'] = collection.xarrays['Q_c'].max(axis=0).sum()*2
    params['op_costs'] = operating_costs(
        path=os.path.join(path, 'producers.csv'),
        indices=collection.indices,
        timeseries=timeseries,
        index_col=0, header=0)
    params['revenue'] = collection.xarrays['Q_c'].copy(deep=True)
    return params
