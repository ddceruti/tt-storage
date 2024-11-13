
import os
import time

import xarray as xr
import numpy as np
from linopy import Model
from src.technologies import annuity


def daily_store(matrices, parameters, forced=False):
    """Create the daily storage model."""

    m = Model(force_dim_names=True)  # instantiate the model
    # power flow in the pipes. ex: P_[in, forward, 1, 0, 0]
    idxs = matrices.indices
    mats = matrices.xarrays
    params = parameters

    m.add_variables(
        name='P',
        lower=0, upper=params['pipes']['power_flow_max_kW'], 
        coords=[idxs['in_out'], idxs['directions'], idxs['edges'], idxs['periods'], idxs['steps']],
        dims=['flows', 'directions', 'edges', 'periods', 'steps'])
    
    m.add_variables(
        name='P_build',
        lower=0, upper=params['pipes']['power_flow_max_kW'],
        coords=[idxs['edges']],
        dims=['edges'])
    
    m.add_variables(
        binary=True, name='lambda_build',
        coords=[idxs['edges']],
        dims=['edges'])
    m.add_variables(
        binary=True, name='lambda_operate',
        coords=[idxs['directions'], idxs['edges'], idxs['periods'], idxs['steps']],
        dims=['directions', 'edges', 'periods', 'steps'])
    m.add_variables(
        name='G',
        lower=0, upper=params['G_max_producers'],
        coords=[idxs['producers'], idxs['periods'], idxs['steps']],
        dims=['producers', 'periods', 'steps'])
    m.add_variables(
        name='G_build',
        lower=0, upper=params['G_max_producers'],
        coords=[idxs['producers']],
        dims=['producers'])
    m.add_variables(
        name='S_build',
        lower=0,
        coords=[idxs['storages']],
        dims=['storages'])
    m.add_variables(
        name='S',
        lower=0,
        coords=[idxs['c'], idxs['storages'], idxs['periods'], idxs['steps']],
        dims=['c', 'storages', 'periods', 'steps'])
    m.add_variables(
        name='SoC',
        lower=0,
        coords=[idxs['storages'], idxs['periods'], idxs['steps_store']],
        dims=['storages', 'periods', 'steps'])

    m.add_constraints(
        m.variables.P_build - params['pipes']['power_flow_max_kW'] * m.variables.lambda_build <= 0,
        name='Big-M powerflow build pipe')
    m.add_constraints(m.variables.P - m.variables.P_build <= 0,  # OK
                      name='max powerflow build pipe')
    m.add_constraints(
        m.variables.P - params['pipes']['power_flow_max_kW'] * m.variables.lambda_operate <= 0,
        name='Big-M pipe operation')
    m.add_constraints(
        m.variables.lambda_operate.sum("directions") <= 1,
        name='One direction pipe')
    m.add_constraints(m.variables.lambda_operate - m.variables.lambda_build <= 0,
                      name='Helper pipe operation when built')
    # where consumers are located, the pipe must always be operated in that direction
    _cons_edges = ((mats['A_i'] != 0) & (mats['A_c'] == 1)).any("nodes").any("consumers")
    m.add_constraints(
        (m.variables.lambda_operate.sum("directions") - m.variables.lambda_build).where(_cons_edges) >= 0,
        name='Always operate pipe consumers when built')

    m.add_constraints(m.variables.G - m.variables.G_build <= 0,
                      name='max generation source')  # OK
    
    m.add_constraints(m.variables.SoC - m.variables.S_build <= 0,
                      name='state of charge limit by build')  # OK
    m.add_constraints(m.variables.S * idxs['w_t'] \
                      - params['storages']['ratios'] * m.variables.S_build <= 0,
                      name='max charge of storage')  # OK
    m.add_constraints(m.variables.SoC.sel(steps=0) \
                      - params['storages']['s_startend'] * m.variables.S_build == 0,
                      name='state of charge start of period')  # OK
    m.add_constraints(m.variables.SoC.sel(steps=idxs['steps_store'][-1]) - params['storages']['s_startend'] * m.variables.S_build == 0,
                      name='state of charge end of period')  # OK

    # # # time loop 
    # start = time.time()
    # # group by steps and steps_store so that the sum is over the steps
    # for j in idxs['periods']:
    #     for i in idxs['steps']:
    #         m.add_constraints((m.variables.SoC.sel(periods=j, steps=i+1) - m.variables.SoC.sel(periods=j, steps=i) \
    #                            * (1 - params['storages']['s_self_disch'] * idxs['w_t'].sel(periods=j, steps=i)) \
    #                     - idxs['w_t'].sel(periods=j, steps=i) * (params['storages']['s_eff'].loc["charge", :] * m.variables.S.sel(periods=j, c="charge", steps=i) \
    #                                     - m.variables.S.sel(periods=j, c="discharge", steps=i) / params['storages']['s_eff'].loc["discharge", :])) == 0,
    #                         name=f'state of charge d={j}, t={i}')  # OK
    # end = time.time()
    # print(f"Time elapsed loop SoC: {end - start}")

    start = time.time()
    # Define the constraint as a function
    def SoC_constraint(j, i):
        self_disch = ((1 - params['storages']['s_self_disch']) ** idxs['w_t']).sel(periods=j, steps=i)
        # self_disch = (1 - params['storages']['s_self_disch'] * idxs['w_t']).sel(periods=j, steps=i)
        SoCs = m.variables.SoC.sel(periods=j, steps=i+1) \
            - m.variables.SoC.sel(periods=j, steps=i) * self_disch
        charge = (params['storages']['s_eff'] * m.variables.S).sel(periods=j, c="charge", steps=i)
        disch = (m.variables.S / params['storages']['s_eff']).sel(periods=j, c="discharge", steps=i)
        return SoCs - idxs['w_t'].sel(periods=j, steps=i) * (charge - disch) == 0

    # Apply the function along the 'periods' and 'steps' dimensions
    constraints = xr.apply_ufunc(SoC_constraint, idxs['periods'], idxs['steps'])

    # Add the constraints to the model
    m.add_constraints(constraints, name="state of charge")
    end = time.time()
    print(f"Time elapsed vectorization SoC: {end - start}")

    # Node balance: Out - In = 0
    # direction = 1 ("backward")
    _from_forward = (m.variables.P.sel(flows='in', directions='forward') \
                     - m.variables.P.sel(flows='out', directions='backward')
                     ).where(mats['A_i'] == 1).sum(dims=["edges"])
    # direction = 1 ("forward")
    _from_backward = (- m.variables.P.sel(flows='out', directions='forward') \
                      + m.variables.P.sel(flows='in', directions='backward')
                      ).where(mats['A_i'] == -1).sum(dims=["edges"])
    pipeflows = _from_forward + _from_backward

    # producers input to node
    production = - m.variables.G.where(mats['A_p'] == -1).sum(dims=["producers"])

    # consumers lambda operation included as node output
    _consumers_fwd = (mats['A_i'] == -1) & (mats['A_c'] == 1)
    term4_fwd_lambda = m.variables.lambda_operate.sel(directions='forward').where(
        _consumers_fwd.any("consumers"))
    term4_fwd = term4_fwd_lambda.sum("edges") \
        * (mats['Q_c'].where(_consumers_fwd.any("edges")).sum(["consumers"]))
    _consumers_bwd = (mats['A_i'] == 1) & (mats['A_c'] == 1)
    term4_bwd = m.variables.lambda_operate.sel(directions='backward').where(
        _consumers_bwd.any("consumers")).sum("edges") \
        * (mats['Q_c'].where(_consumers_bwd.any("edges")).sum(["consumers"]))
    _stores = (m.variables.S.sel(c='charge')- m.variables.S.sel(c='discharge')
               ).where(mats['A_s'] == -1).sum(dims=["storages"])
    m.add_constraints(pipeflows + production + term4_fwd + term4_bwd + _stores == 0,
                      name='Node balance')  # OK

    term1 = m.variables.P.sel(flows='in') - m.variables.P.sel(flows='out')
    term2_a = params['pipes']['losses']['a'] * m.variables.P.sel(flows='in')
    term2_b = m.variables.lambda_operate * params['pipes']['losses']['b']
    m.add_constraints(term1 - mats['l_i'] * (term2_a + term2_b) == 0,
                      name='Heat balance pipes')  # OK

    # m.add_constraints(lambda_operate.where(set_consumers == 1) - set_consumers.where(set_consumers == 1) <= 0,  # OK
    #                   name='Conserve direction of consumer')

    # where are consumers located in the network
    _consumers = mats['A_c'].sum(axis=1) == 1
    _c_forward = (mats['A_i'].where(_consumers).sum(axis=0) == -1)
    _c_backward = (mats['A_i'].where(_consumers).sum(axis=0) == 1)
    set_consumers = xr.DataArray(data=[_c_forward.values.squeeze(), _c_backward.values.squeeze()],
                        dims=['directions', 'edges'], coords=[idxs['directions'], idxs['edges']])

    # only operate forward if the consumer is located in the forward direction
    # therefore set lambda_operate to backward == 0 if the consumer is forwards
    # this is only possible because consumers are always at the end of the pipes.
    m.add_constraints(m.variables.lambda_operate.where(_c_forward).sel(directions="backward") - 0 <= 0,
                        name='Only operate forward to consumer')
    m.add_constraints(m.variables.lambda_operate.where(_c_backward).sel(directions="forward") - 0 <= 0,
                        name='Only operate backward to consumer')

    if forced:
        # Tightening of operation wherever we can
        m.add_constraints(m.variables.lambda_operate.where(set_consumers) - 1 >= 0,
                        name='Force direction of consumer')

    # tighted producer direction
    set_prods_no_store_fwd = ((mats['A_i'] == 1) & (mats['A_p'] == -1) & (mats['A_s'] == 0)).any("producers").any("nodes").all("storages")
    set_prods_no_store_bwd = ((mats['A_i'] == -1) & (mats['A_p'] == -1) & (mats['A_s'] == 0)).any("producers").any("nodes").all("storages")
    m.add_constraints(m.variables.lambda_operate.sel(directions='forward').where(set_prods_no_store_fwd) - 1 >= 0,
                        name='Tighten direction of producer without storage fwd')
    m.add_constraints(m.variables.lambda_operate.sel(directions='backward').where(set_prods_no_store_bwd) - 1 >= 0,
                        name='Tighten direction of producer without storage bwd')
    # if any edge is both forward and backward, raise error
    if (set_prods_no_store_fwd & set_prods_no_store_bwd).any("edges"):
        raise ValueError('Lambda operate for edge is both forward and backward')

    invest_prod = (params['inv_p'] * m.variables.G_build).sum("producers")
    operation_prod = (m.variables.G * params['op_costs'] / params['eta_p']).sum("producers") \
        * idxs['w_t'] * idxs['w_d']

    invest_storage = (params['storages']['inv_s'].sel(storages=idxs['storages']) \
                       * m.variables.S_build).where((mats['A_s'] == -1).any("nodes")).sum()
    
    cost_pipe_cap = (m.variables.P_build * params['pipes']['invest']['a'] \
                     + m.variables.lambda_build * params['pipes']['invest']['b'])
    invest_pipes = (cost_pipe_cap* (mats['l_i']*annuity(c_i=0.09, n=40))).sum("edges")

    sold_heat_fwd = (m.variables.lambda_operate.sel(directions='forward') * mats['Q_c']
                     ).where(_consumers_fwd.any("nodes"))
    sold_heat_bwd = (m.variables.lambda_operate.sel(directions='backward') * mats['Q_c']
                     ).where(_consumers_bwd.any("nodes"))
    revenues = ((sold_heat_fwd + sold_heat_bwd) * params['revenue']).sum(["consumers", "edges"]) \
        * idxs['w_t'] * idxs['w_d']

    m.add_objective((invest_prod + invest_storage + invest_pipes + (operation_prod - revenues).sum()),
                    sense='min')
    return m

def output_results(model, path):
    print('\n --------------------Objective Function-----------------------------')
    print(model.objective.value)

    print('\n --------------------Installed Capacities-----------------------------')
    print('P_build')
    # drop 0 rows from dataframe
    df_P_build_sol = model.variables.P_build.solution.to_dataframe()
    print(df_P_build_sol.loc[~(df_P_build_sol==0).all(axis=1)])
    print('G_build')
    print(model.variables.G_build.solution.to_dataframe())
    print('S_build')
    print(model.variables.S_build.solution.to_dataframe())
    print(' ------------------------------------------------------------------------------')

    df_P_build_sol.to_csv(os.path.join(path, 'P_build.csv'))
    model.variables.G_build.solution.to_dataframe().to_csv(os.path.join(path, 'G_build.csv'))
    model.variables.S_build.solution.to_dataframe().to_csv(os.path.join(path, 'S_build.csv'))
    model.variables.P.solution.to_dataframe().to_csv(os.path.join(path, 'P.csv'))
    model.variables.G.solution.to_dataframe().to_csv(os.path.join(path, 'G.csv'))
    model.variables.S.solution.to_dataframe().to_csv(os.path.join(path, 'S.csv'))
    model.variables.SoC.solution.to_dataframe().to_csv(os.path.join(path,'SoC.csv'))
    model.variables.lambda_build.solution.to_dataframe().to_csv(
        os.path.join(path,'lambda_build.csv'))
    model.variables.lambda_operate.solution.to_dataframe().to_csv(
        os.path.join(path,'lambda_operate.csv'))
    np.array(model.objective.value).tofile(os.path.join(path,'objective_value.csv'),
                                       sep=',', format='%10.5f')


# make a child class of the Model class from linopy, which should contain
# a sets property for nodes, edges, consumers, producers, storages, and
# the parameters property for the heat loss and thermal capacity