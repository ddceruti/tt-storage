"""Time series aggregation with synthetic demandlib profiles and
hourly data."""

import pandas as pd
import tsam.timeseriesaggregation as tsam
import xarray as xr
import numpy as np


def aggregate(timeseries: pd.Series,
              periods: int=3,
              h_per_period: int=24,
              segmentation: bool =True,
              n_segments: int = 6,
              peak_min: int=None,
              peak_max: int=None) -> tsam.TimeSeriesAggregation:
    """Aggregate time series to a given number of periods and 
    segments with tsam.

    Args:
        timeseries (pd.Series): timeseries data to be aggregated.
        periods (int, optional): number of periods to aggregate to. Defaults to 3.
        h_per_period (int, optional): number of hours per period. Defaults to 24.
        segmentation (bool, optional): If time steps should be segmented. Defaults to True.
        n_segments (int, optional): number of time segments per period. Defaults to 6.
        peak_min (int, optional): add extreme period where this column is minimum. Defaults to None.
        peak_max (int, optional): add extreme period where this column is maximum. Defaults to None.

    Returns:
        tsam.TimeSeriesAggregation: aggregated time series object.
    """
    agg = tsam.TimeSeriesAggregation(timeseries, 
                            noTypicalPeriods=periods,
                            hoursPerPeriod=h_per_period,
                            segmentation=segmentation,
                            noSegments=n_segments,
                            rescaleClusterPeriods=False,
                            representationMethod="distributionRepresentation",
                            distributionPeriodWise = False,
                            clusterMethod='hierarchical',
                            extremePeriodMethod='new_cluster_center',
                            addPeakMin=peak_min,
                            addPeakMax=peak_max)

    agg.createTypicalPeriods()
    return agg


def add_design_load(Q_c: xr.DataArray,
                    df_agg: pd.DataFrame,
                    df_mapping: pd.DataFrame,
                    safety_parameter: float = 1.4) -> pd.DataFrame:
    """Add a synthetic peak load design to the aggregated time series.
    Has a synthetic profile which is added to the aggregated time series
    as a new period n+1 with 4 steps and 6 h per step (this is hardcoded,
    can be changed since the code afterwards is flexible).

    @TODO: Add the peak column of the mapping_consumers.csv

    Args:
        Q_c (xr.DataArray): Consumer demand profiles.
        df_agg (pd.DataFrame): Aggregated time series data.
        df_mapping (pd.DataFrame): Consumer mapping data.
        safety_parameter (float, optional): Safety parameter added to the peak.
            Defaults to 1.4.
        defining_demand (str, optional): Column in df_agg which defines the
            peak that duplicates the maximum demand period. Defaults to 'ghd'.

    Returns:
        pd.DataFrame: aggregated dataframe with new period with peak design
            day profile.
    """
    # add synthetic peak load design to the aggregated time series
    # period where any of the columns in Q_c achieves max value
    max_dems = Q_c.max(["periods", "steps"])
    # periods where the most times the max value is achieved
    max_period = Q_c.where(Q_c == Q_c.max(), drop=True)["periods"].item()
    design_period = Q_c.coords["periods"].max().item() + 1

    # the same for the Q_c xr.DataArray
    new_Q_c = Q_c.loc[{"periods": max_period}].copy(deep=True).assign_coords(periods=design_period)

    # Find the step (index along "steps") where max occurs for each consumer
    idx_max_steps = new_Q_c.idxmax(dim="steps")
    # set peaks in dataframe
    consumer_peaks = df_mapping.set_index("consumers")["peak"]
    peak_values = consumer_peaks * safety_parameter
    # substitute max value with safe value for each consumer with df_mapping.peak
    new_Q_c.loc[{"steps": idx_max_steps}] = peak_values
    Q_c_extended = xr.concat([Q_c, new_Q_c], dim="periods")

    if any(peak_values < max_dems):
        raise ValueError("Some peak values are smaller than the maximum demand.")

    idx = [(design_period, i, j) for (_,(i,j)) in enumerate(df_agg.loc[(max_period, slice(None))].index)]
    new_row = pd.DataFrame(
        np.nan,
        index=pd.MultiIndex.from_tuples(idx, names=df_agg.index.names),
        columns=df_agg.columns)
    df_agg = pd.concat([df_agg, new_row])

    return Q_c_extended, df_agg


def add_design_day(df_design: pd.DataFrame,
                   df_idxs: pd.DataFrame) -> pd.DataFrame:
    """Add a design day to the index of the time series (24 h period, 1st
    january). This is necessary for the calculation of the weights of each
    period.

    Args:
        df_design (pd.DataFrame): Design day data.
        df_idxs (pd.DataFrame): Index matching data from tsam.

    Returns:
        pd.DataFrame: Index matching data with the design day added as a
            new day.
    """
    time = df_idxs.index[-1]
    # add one day to the index
    rows = pd.date_range(time, periods=df_idxs.hours.max(), freq='h')
    # concat rows to the df_idxs
    df_idxs = pd.concat([df_idxs, pd.DataFrame(index=rows)])

    # get the design period
    design_period = df_design.index.get_level_values('periods').max()
    df_idxs.loc[rows, 'periods'] = int(design_period)  # due to previous func
    df_idxs.loc[rows, 'hours'] = np.sort(df_idxs.hours.dropna().unique()).astype(int)

    # get the design day indexes
    design_data = df_design.loc[(design_period, slice(None)), :].index.values

    total_h = 0  # initialize counter
    for _, (p, s, delta) in enumerate(design_data):
        # assign step to every hour
        selected = rows.values[total_h:total_h+int(delta)]
        df_idxs.loc[selected, 'steps'] = int(s)
        total_h += delta
    return df_idxs.astype(int)


def sets(aggregated_ts):
    steps = aggregated_ts.index.get_level_values('steps').unique()
    periods = aggregated_ts.index.get_level_values('periods').unique()
    return steps, periods


def weights(aggregated_ts, index_matching, h_per_period):
    delta_t = aggregated_ts.reset_index('step_duration')['step_duration'].to_xarray()

    # select every row where h_per_period is reached
    idxs = index_matching.loc[index_matching.hours == h_per_period-1]
    # calculate occurence of every period
    idxs = idxs.groupby('periods').size().reset_index(name='occurence')
    idxs.index = idxs['periods']
    idxs.drop(columns='periods', inplace=True)
    idxs.occurence = idxs.occurence.astype(float)
    delta_d = xr.DataArray.from_series(idxs['occurence'])
    return delta_d, delta_t


def map_type_to_consumer(A_c: xr.DataArray,
                         df_mapping: pd.DataFrame,
                         df_agg_ts: pd.DataFrame) -> xr.DataArray:
    """This is a function to set the consumer profile for the optimization.
    It maps the typical profiles to a given type in mapping_consumers.csv.
    """
    consumers = A_c.coords['consumers'].values

    # check if all consumers are in the consumer column of df_mapping
    if not set(consumers).issubset(set(df_mapping['consumers'])):
        missing = set(consumers) - set(df_mapping['consumers'])
        raise ValueError(f"Not all consumers {missing} are in the mapping.")

    if not set(df_mapping.type.unique()).issubset(set(df_agg_ts.columns)):
        missing = set(df_mapping.type.unique()) - set(df_agg_ts.columns)
        raise ValueError(f"Not all types {missing} are in the aggregated time series.")

    df_Q_c = pd.DataFrame(index=df_agg_ts.index,
                          columns=consumers,
                          data=np.nan)

    # map df_agg_ts to df_Q_c via the type and consumer column value
    for i in consumers:
        type_ = df_mapping.loc[df_mapping['consumers'] == i, 'type'].values.item()
        df_Q_c.loc[:, i] = df_agg_ts.loc[:, type_]

    df_Q_c.index = df_Q_c.index.droplevel('step_duration')
    df_Q_c = df_Q_c.stack()
    df_Q_c.index = df_Q_c.index.set_names(['periods', 'steps', 'consumers'])
    Q_c = df_Q_c.to_xarray()
    Q_c = Q_c.fillna(0.)
    return Q_c


def scale_demands(collection, df_mapping):
    """Scale the yearly demands of every type as stated in df_mapping"""
    agg_demands = (collection.xarrays['Q_c']
                   * collection.indices['w_d']
                   * collection.indices['w_t']).sum(['periods', 'steps'])
    map_demands = df_mapping.set_index('consumers').loc[:, 'annual_heat'].to_xarray()
    scale = map_demands / agg_demands
    # scale * collection for every consumer
    return collection.xarrays['Q_c'] * scale
