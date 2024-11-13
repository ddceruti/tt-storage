""" This module contains the model class for the daily storage model.
"""

import os

import xarray as xr
import numpy as np
from linopy import Model

from tts.technologies import annuity
from tts.matrix_collection import MatrixCollection


class DailyStorageModel(Model):
    """Model for the daily storage optimization.

    Args:
        Model (linopy.Model): Base class for the model.

    Attributes:
        idxs (dict): Dictionary with the indices for the model.
        matrices (xarray.Dataset): Dataset with the matrices for the model.
        params (dict): Dictionary with the parameters for the model.
        forced (bool): If True, the model is forced to expand the network to
            all consumers
    """
    def __init__(self,
                 matrices: MatrixCollection,
                 parameters: dict,
                 forced: bool = False):
        """Initialize the model object.

        Args:
            matrices (MatrixCollection): Collection of matrices for the model.
            parameters (dict): dictionary with the parameters for the model.
            forced (bool, optional): True if forced expansion to all
            consumers is desired. Defaults to False.
        """
        super().__init__(force_dim_names=True)  # instantiate the model
        # power flow in the pipes. ex: P_[in, forward, 1, 0, 0]
        self.idxs = matrices.indices
        self._matrices = matrices.xarrays
        self.params = parameters
        self.forced = forced
        self.locate_consumers()

    def create(self):
        """Create the model and writes variables, constraints and objective.
        """
        self.add_vars()
        self.add_constr()
        self.add_obj()
        return self

    def add_vars(self):
        """Add variables to the model.
        """
        self.add_variables(
            name='P',
            lower=0,
            upper=self.params['pipes']['power_flow_max_kW'],
            coords=[self.idxs['in_out'],
                    self.idxs['directions'],
                    self.idxs['edges'],
                    self.idxs['periods'],
                    self.idxs['steps']],
            dims=['flows', 'directions', 'edges', 'periods', 'steps'])

        self.add_variables(
            name='P_build',
            lower=0, upper=self.params['pipes']['power_flow_max_kW'],
            coords=[self.idxs['edges']],
            dims=['edges'])

        self.add_variables(
            binary=True,
            name='lambda_build',
            coords=[self.idxs['edges']],
            dims=['edges'])
        self.add_variables(
            binary=True,
            name='lambda_operate',
            coords=[self.idxs['directions'],
                    self.idxs['edges'],
                    self.idxs['periods'],
                    self.idxs['steps']],
            dims=['directions', 'edges', 'periods', 'steps'])
        self.add_variables(
            name='G',
            lower=0,
            upper=self.params['G_max_producers'],
            coords=[self.idxs['producers'],
                    self.idxs['periods'],
                    self.idxs['steps']],
            dims=['producers', 'periods', 'steps'])
        self.add_variables(
            name='G_build',
            lower=0,
            upper=self.params['G_max_producers'],
            coords=[self.idxs['producers']],
            dims=['producers'])
        self.add_variables(
            name='S_build',
            lower=0,
            coords=[self.idxs['storages']],
            dims=['storages'])
        self.add_variables(
            name='S',
            lower=0,
            coords=[self.idxs['c'],
                    self.idxs['storages'],
                    self.idxs['periods'],
                    self.idxs['steps']],
            dims=['c', 'storages', 'periods', 'steps'])
        self.add_variables(
            name='SoC',
            lower=0,
            coords=[self.idxs['storages'],
                    self.idxs['periods'],
                    self.idxs['steps_store']],
            dims=['storages', 'periods', 'steps'])

        # variables to track the objective function
        self.add_variables(
            name='inv_prod',
            lower=0)
        self.add_variables(
            name='inv_pipes',
            lower=0)
        self.add_variables(
            name='inv_stores',
            lower=0)
        self.add_variables(
            name='op_prod',
            lower=0)
        self.add_variables(
            name='revenues',
            lower=0)
        self.add_variables(
            name='costs',
            lower=0)


    def locate_consumers(self):
        """Writes arrays to locate consumers in the network, their edges and
        the direction of the edge.
        """
        # where are consumers located in the network
        self.node_c = self._matrices['A_c'].sum(axis=1) == 1
        # edge i goes into consumer (A_i[:, i] == -1) 'forward'
        self.edge_c_fwd = (self._matrices['A_i'].where(self.node_c).sum(axis=0) == -1)
        # edge i goes from consumer (A_i[:, i] == 1) 'backward'
        self.edge_c_bwd = (self._matrices['A_i'].where(self.node_c).sum(axis=0) == 1)
        # coords for consumers lambda operation included as node output
        self.c_fwd = (self._matrices['A_i'] == -1) & (self._matrices['A_c'] == 1)
        self.c_bwd = (self._matrices['A_i'] == 1) & (self._matrices['A_c'] == 1)
        self.edge_c = (
            (self._matrices['A_i'] != 0) & (self._matrices['A_c'] == 1)
            ).any('nodes').any('consumers')


    # Define the constraint as a function
    def SoC_constraint(self, j, i):
        """State of charge constraint for the storage.

        Args:
            j (int): period j
            i (int): step i

        Returns:
            linopy.constraints.Constraint: SoC constraint
        """
        self_disch = (
            (1 - self.params['storages']['s_self_disch']) ** self.idxs['w_t']
            ).sel(periods=j, steps=i)
        SoCs = (
            self.variables.SoC.sel(periods=j, steps=i+1)
            - self.variables.SoC.sel(periods=j, steps=i) * self_disch)
        charge = (
            self.params['storages']['s_eff'] * self.variables.S
            ).sel(periods=j, c='charge', steps=i)
        disch = (
            self.variables.S
            / self.params['storages']['s_eff']
            ).sel(periods=j, c='discharge', steps=i)
        return (SoCs
                - self.idxs['w_t'].sel(periods=j, steps=i) * (charge - disch)
                ) == 0


    def add_constr(self):
        """Add constraints to the model.

        Raises:
            ValueError: _description_
        """
        self.add_constraints(
            self.variables.P_build
            - self.params['pipes']['power_flow_max_kW'] * self.variables.lambda_build <= 0,
            name='Big-M powerflow build pipe')  # OK

        self.add_constraints(self.variables.P - self.variables.P_build <= 0,  # OK
                        name='max powerflow build pipe')

        self.add_constraints(
            self.variables.P
            - self.params['pipes']['power_flow_max_kW'] * self.variables.lambda_operate <= 0,
            name='Big-M pipe operation')

        self.add_constraints(
            self.variables.lambda_operate.sum('directions') <= 1,
            name='One direction pipe')

        self.add_constraints(
            self.variables.lambda_operate - self.variables.lambda_build <= 0,
            name='Helper pipe operation when built')

        # where consumers are located, the pipe must always be operated in that direction
        self.add_constraints(
            (self.variables.lambda_operate.sum('directions')
             - self.variables.lambda_build).where(self.edge_c) >= 0,
            name='Always operate pipe consumers when built')

        if self.forced:
            # where consumers are located, the pipe must always be operated in that direction
            self.add_constraints(
                self.variables.lambda_build.where(self.edge_c) >= 1,
                name='Always operate pipe consumers')

            # Tightening of operation wherever we can
            set_consumers = xr.DataArray(
                data=[self.edge_c_fwd.values.squeeze(),
                    self.edge_c_bwd.values.squeeze()],
                dims=['directions', 'edges'],
                coords=[self.idxs['directions'], self.idxs['edges']]
            )
            self.add_constraints(
                self.variables.lambda_operate.where(set_consumers) - 1 >= 0,
                name='Tighten direction of consumer')

        self.add_constraints(
            self.variables.G - self.variables.G_build * self.params['availability'] <= 0,
            name='max generation source')  # OK

        self.add_constraints(
            self.variables.SoC - self.variables.S_build <= 0,
            name='state of charge limit by build')  # OK
        self.add_constraints(
            self.variables.S * self.idxs['w_t'] \
            - self.params['storages']['ratios'] * self.variables.S_build <= 0,
            name='max charge of storage')  # OK
        self.add_constraints(
            self.variables.SoC.sel(steps=0)
            - self.params['storages']['s_startend'] * self.variables.S_build == 0,
            name='state of charge start of period')  # OK
        self.add_constraints(
            self.variables.SoC.sel(steps=self.idxs['steps_store'][-1])
            - self.params['storages']['s_startend'] * self.variables.S_build == 0,
            name='state of charge end of period')  # OK

        # Apply the function along the 'periods' and 'steps' dimensions
        constraints = xr.apply_ufunc(
            self.SoC_constraint,
            kwargs={'j': self.idxs['periods'], 'i': self.idxs['steps']}
        )

        # Add the constraints to the model
        self.add_constraints(constraints, name='state of charge')

        # Node balance: Out - In = 0
        # direction = 1 ('backward')
        _from_forward = (
            self.variables.P.sel(flows='in', directions='forward')
            - self.variables.P.sel(flows='out', directions='backward')
            ).where(self._matrices['A_i'] == 1).sum(dims=['edges'])
        # direction = 1 ('forward')
        _from_backward = (
            - self.variables.P.sel(flows='out', directions='forward')
            + self.variables.P.sel(flows='in', directions='backward')
            ).where(self._matrices['A_i'] == -1).sum(dims=['edges'])
        pipe_flows = _from_forward + _from_backward

        # producers input to node
        production = - self.variables.G.where(self._matrices['A_p'] == -1).sum(dims=['producers'])

        # lambda always operating in the direction of the consumer to deliver
        # heat
        term4_fwd_lambda = self.variables.lambda_operate.sel(directions='forward').where(
            self.c_fwd.any('consumers'))
        term4_fwd = term4_fwd_lambda.sum('edges') \
            * (self._matrices['Q_c'].where(self.c_fwd.any('edges')).sum(['consumers']))

        term4_bwd_lambda = self.variables.lambda_operate.sel(directions='backward').where(
            self.c_bwd.any('consumers'))
        term4_bwd = term4_bwd_lambda.sum('edges') \
            * (self._matrices['Q_c'].where(self.c_bwd.any('edges')).sum(['consumers']))

        # storage balance only where storage is present
        _stores = (self.variables.S.sel(c='charge')
                   - self.variables.S.sel(c='discharge')
                ).where(self._matrices['A_s'] == -1).sum(dims=['storages'])
        self.add_constraints(
            pipe_flows + production + term4_fwd + term4_bwd + _stores == 0,
            name='Node balance')  # OK

        # Heat balance pipes: In - Out = 0
        term1 = (self.variables.P.sel(flows='in')
                 - self.variables.P.sel(flows='out'))
        term2_a = (self.params['pipes']['losses']['a']
                   * self.variables.P.sel(flows='in'))
        term2_b = (self.variables.lambda_operate
                   * self.params['pipes']['losses']['b'])
        self.add_constraints(
            term1 - self._matrices['l_i'] * (term2_a + term2_b) == 0,
                        name='Heat balance pipes')  # OK

        # only operate forward if the consumer is located in the forward direction
        # therefore set lambda_operate to backward == 0 if the consumer is forwards
        # this is only possible because consumers are always at the end of the pipes.
        self.add_constraints(
            self.variables.lambda_operate.where(self.edge_c_fwd).sel(directions='backward') <= 0,
            name='Only operate forward to consumer')
        self.add_constraints(
            self.variables.lambda_operate.where(self.edge_c_bwd).sel(directions='forward') <= 0,
            name='Only operate backward to consumer')

    def add_obj(self):
        """Objective function for the model: minimize CAPEX + OPEX - REVENUES
        """
        invest_prod = (
            self.params['inv_p'] * self.variables.G_build
            ).sum('producers')
        self.add_constraints(
            self.variables.inv_prod - invest_prod == 0,
            name='Investment costs'
        )
        operation_prod = (
            self.variables.G * self.params['op_costs']
            / self.params['eta_p']
            ).sum('producers') * self.idxs['w_t'] * self.idxs['w_d']
        self.add_constraints(
            self.variables.op_prod - operation_prod.sum() == 0,
            name='Operation costs'
        )

        invest_storage = (
            self.params['storages']['inv_s'].sel(storages=self.idxs['storages'])
            * self.variables.S_build
            ).where((self._matrices['A_s'] == -1).any('nodes')).sum()
        self.add_constraints(
            self.variables.inv_stores - invest_storage == 0,
            name='Investment costs storage'
        )

        cost_pipe_cap = (
            self.variables.P_build * self.params['pipes']['invest']['a']
            + self.variables.lambda_build * self.params['pipes']['invest']['b']
            )
        invest_pipes = (
            cost_pipe_cap * (self._matrices['l_i']*annuity(c_i=0.09, n=40))
            ).sum('edges')
        self.add_constraints(
            self.variables.inv_pipes - invest_pipes == 0,
            name='Investment costs pipes'
        )

        sold_heat_fwd = (
            self.variables.lambda_operate.sel(directions='forward')
            * self._matrices['Q_c']).where(self.c_fwd.any('nodes'))
        sold_heat_bwd = (
            self.variables.lambda_operate.sel(directions='backward')
            * self._matrices['Q_c']).where(self.c_bwd.any('nodes'))
        revenues = (
            (sold_heat_fwd + sold_heat_bwd)
            * self.params['revenue']
            ).sum(['consumers', 'edges']) \
                * self.idxs['w_t'] * self.idxs['w_d']
        self.add_constraints(
            self.variables.revenues - revenues.sum() == 0,
            name='Revenues'
        )

        self.add_objective(
            (invest_prod
             + invest_storage
             + invest_pipes
             + (operation_prod - revenues).sum()),
            sense='min')


    def output_results(self, path: os.PathLike):
        """Output the results of the model to a directory.

        Args:
            path (os.PathLike): Path to the directory where the results should
                be stored. 
        """
        print('\n --------------------Objective Function-----------------------------')
        print(self.objective.value)

        print('CAPEX Production')
        print(self.variables.inv_prod.solution.item())
        print('CAPEX Storage')
        print(self.variables.inv_stores.solution.item())
        print('CAPEX Pipes')
        print(self.variables.inv_pipes.solution.item())
        print('OPEX Production')
        print(self.variables.op_prod.solution.item())
        print('Revenues')
        print(self.variables.revenues.solution.item())
    
        print('\n --------------------Installed Capacities-----------------------------')
        print('P_build')
        # drop 0 rows from dataframe
        df_P_build_sol = self.variables.P_build.solution.to_dataframe()
        print(df_P_build_sol.loc[~(df_P_build_sol==0).all(axis=1)])
        print('G_build')
        print(self.variables.G_build.solution.to_dataframe())
        print('S_build')
        print(self.variables.S_build.solution.to_dataframe())
        print(' ------------------------------------------------------------------------------')

        df_P_build_sol.to_csv(os.path.join(path, 'P_build.csv'))
        self.variables.G_build.solution.to_dataframe().to_csv(os.path.join(path, 'G_build.csv'))
        self.variables.S_build.solution.to_dataframe().to_csv(os.path.join(path, 'S_build.csv'))
        self.variables.P.solution.to_dataframe().to_csv(os.path.join(path, 'P.csv'))
        self.variables.G.solution.to_dataframe().to_csv(os.path.join(path, 'G.csv'))
        self.variables.S.solution.to_dataframe().to_csv(os.path.join(path, 'S.csv'))
        self.variables.SoC.solution.to_dataframe().to_csv(os.path.join(path,'SoC.csv'))
        self.variables.lambda_build.solution.to_dataframe().to_csv(
            os.path.join(path,'lambda_build.csv'))
        self.variables.lambda_operate.solution.to_dataframe().to_csv(
            os.path.join(path,'lambda_operate.csv'))
        np.array([self.objective.value,
                 self.variables.inv_prod.solution.item(),
                 self.variables.inv_stores.solution.item(),
                 self.variables.inv_pipes.solution.item(),
                 self.variables.op_prod.solution.item(),
                 self.variables.revenues.solution.item(),
                 ]).tofile(
            os.path.join(path,'objective_value.csv'),
            sep=',', format='%10.5f')
