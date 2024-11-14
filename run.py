"""Run the tt-storage optimization for any case study respecting the default folder structure.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tts.model import DailyStorageModel
from tts.matrix_collection import MatrixCollection
from tts import timeseries as ts
from tts import technologies as techs
from tts import plots

plt.style.use('seaborn-v0_8-paper')

PLOTS = True  # if True, plot the results
B_TYPE = ['sfh', 'mfh', 'ghd', 'pub']  # building types to track full load hours accuracy of


def main(case_folder: str,
         results_folder: str,
         forced: bool=True,
         revenue: float=140E-3,
         design_load: bool=False,
         n_periods: int=3,
         n_segments: int=24,
         period_length: int=24) -> None:
    """Main function to run the optimization from the defined case study folder
    and store the results in a subfolder.

    Args:
        results_folder (str): folder name to store the results
        forced (bool, optional): forced expansion. Defaults to True. If false
            the expansion is not forced to all consumers and is economic.
        revenue (float, optional): Revenue of sold heat. Defaults to 140E-3.
        design_load (bool, optional): If a synthetic peak design day is to
            be added. Defaults to False.
        n_periods (int, optional): number of periods to aggregate to. Defaults
            to 3.
        n_segments (int, optional): number of time segments to aggregate to.
            Defaults to 24.
        period_length (int, optional): length of every period in hours. Defaults
            to 24.

    Raises:
        ValueError: 
    """
    print(f'Running case {case_folder} with results in {results_folder}')
    data_path = os.path.join(os.path.dirname(__file__), case_folder, 'data')
    results_path = os.path.join(os.path.dirname(__file__), case_folder,
                          'results-def-100', results_folder)
    # create dir
    os.makedirs(results_path, exist_ok=True)
    # read in time series
    df_ts = pd.read_csv(os.path.join(data_path, 'timeseries.csv'),
                        index_col=0, parse_dates=True, header=0)
    mapping = pd.read_csv(
        os.path.join(data_path, 'mapping_consumers.csv'),
        sep=',', index_col=None, header=0)

    collection = MatrixCollection(data_path).read_csvs().to_xarray()

    # Aggregation of the time series with tsam
    aggregated = ts.aggregate(df_ts,
                            periods=n_periods,
                            h_per_period=period_length,
                            segmentation=True,
                            n_segments=n_segments,
                            peak_min=['temperature'],
                            peak_max=None
                            )

    # drop step_duration dimension from index and then put into series
    # with the remaining two indices
    agg = aggregated.typicalPeriods.copy(deep=True)
    agg.index.names = ['periods', 'steps', 'step_duration']
    agg.to_csv(os.path.join(results_path, 'aggregated.csv'))

    # create weights
    idx_matching = aggregated.indexMatching()
    idx_matching = idx_matching.astype(int)  # force to int
    idx_matching.rename(
        columns={
            'PeriodNum': 'periods',
            'TimeStep': 'hours',
            'SegmentIndex':'steps'},
        inplace=True)
    idx_matching.to_csv(os.path.join(results_path, 'index_matching.csv'))
    
    # set all indices depending on the topology
    collection.set_indices()
    collection.indices['w_d'], collection.indices['w_t'] = ts.weights(
        aggregated_ts=agg,
        index_matching=idx_matching,
        h_per_period=24)

    collection.indices['steps'], collection.indices['periods'] = ts.sets(agg)

    collection.xarrays['Q_c'] = ts.map_type_to_consumer(
        A_c=collection.xarrays.get('A_c'),
        df_mapping=mapping,
        df_agg_ts=agg
        )
    collection.xarrays['Q_c'].name = 'demand'
    collection.xarrays['Q_c'].to_netcdf(os.path.join(results_path, 'Q_c_unscaled.nc'))

    collection.xarrays['Q_c'] = ts.scale_demands(
            collection=collection,
            df_mapping=mapping
        )

    # add synthetic peak load design to the aggregated time series
    if design_load:
        collection.xarrays['Q_c'], agg = ts.add_design_load(
            Q_c=collection.xarrays['Q_c'],
            df_agg=agg,
            df_mapping=mapping,
            safety_parameter=1.0)
        agg.to_csv(os.path.join(results_path, 'aggregated-design.csv'))

        # add one day to the index1
        idx_matching = ts.add_design_day(agg, idx_matching[:8760])  # only one year, duplicated due to indexmatching

        collection.indices['w_d'], collection.indices['w_t'] = ts.weights(
                aggregated_ts=agg,
                index_matching=idx_matching,
                h_per_period=period_length)
        
        collection.indices['w_d'][-1] = 1.
        # collection.indices['w_t'][-1] = 0.
        collection.indices['steps'] = agg.index.get_level_values('steps').unique().values
        collection.indices['periods'] = agg.index.get_level_values('periods').unique().values

    collection.xarrays['Q_c'].to_netcdf(os.path.join(results_path,
                                                     'Q_c_scaled.nc'))

    # add one time step for the storage level
    collection.indices['steps_store'] = np.append(
        collection.indices['steps'], collection.indices['steps'][-1]+1)

    flh = (collection.xarrays['Q_c']
           * collection.indices['w_d']
           * collection.indices['w_t']).sum() \
            / collection.xarrays['Q_c'].sum('consumers').max()
    print(f'aggregated FLH: {flh.values}')
    # flh to a text file
    with open(os.path.join(results_path, 'flh.txt'), 'w', encoding='utf-8') as f:
        f.write(f'aggregated FLH: {flh.values}\n')

    # read in and store all parameters
    params = techs.read_parameters(
        path=os.path.join(data_path, 'technologies'),
        collection=collection,
        timeseries=agg.reset_index(level='step_duration').to_xarray())
    # set revenues manually
    params['revenue'][:] = revenue

    mdl = DailyStorageModel(matrices=collection, parameters=params,
        forced=forced).create()

    # solve the model
    solver_options = {
        'TimeLimit': 3600,
        'MIPGap': 0.0005,
        # random seed
        'Seed': 42,
        # 'ConcurrentMIP': 4,
        # 'Threads': 8,
    }

    # start time
    mdl.solve(solver_name='gurobi', io_api='direct',
              log_fn=os.path.join(results_path, 'log.txt'),
              solution_fn=os.path.join(results_path, 'soln.txt'),
              **solver_options)

    # save the model
    if mdl.status == 'ok': 
        mdl.output_results(results_path)
    elif mdl.status == 'warning':
        print(f'Model status: {mdl.status}')
        mdl.print_infeasibilities()
        raise ValueError(f'Model not solved {mdl.status}')
    else:
        print(f'Model status: {mdl.status}')
        raise ValueError(f'Model not solved {mdl.status}')

    if PLOTS:
        plots.demand_profiles(df_ts.loc[:, B_TYPE],
                              path=os.path.join(results_path, 'day-demand.svg'))

        fig_init, fig = plots.solution(mdl, collection)
        # plt.show()
        fig_init.savefig(os.path.join(results_path, 'topology_init.svg'),
                        bbox_inches='tight')
        fig.savefig(os.path.join(results_path, 'topology.svg'),
                    bbox_inches='tight')

        fig_op, fig_op_stck, fig_op_stck_sum = plots.operation(mdl, params['producer_names'],
                                              collection.xarrays['Q_c'])
        fig_op.savefig(os.path.join(results_path, 'operation.svg'),
                        bbox_inches='tight')
        fig_op_stck.savefig(os.path.join(results_path, 'operation_stacked.svg'),
                        bbox_inches='tight')
        fig_op_stck_sum.savefig(os.path.join(results_path, 'operation_stacked_sum.svg'),
                        bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    main(
        case_folder='bm1',
        # results_folder='bc-design-economic-3p-24s-dayahead-12safety',
        results_folder='results',
        forced=True,
        revenue=100E-3,
        design_load=True,
        n_periods=1,
        n_segments=3,
        period_length=24
    )
