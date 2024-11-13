"""Matrices for the optimization."""
import os

import pandas as pd
import numpy as np


class MatrixCollection:
    """Class to store and manage matrices for the optimization.
    
    Parameters
    ----------
    path : str
        Path to the directory where the matrices are stored.
    
    Attributes
    ----------
    path : str
        Path to the directory where the matrices are stored.
    matrices : dict
        Dictionary to store the matrices.
    xarrays: dict
        Dictionary to store the xarrays.
    indices: dict
        Dictionary to store the xarray indices.
    """
    def __init__(self, path):
        self.path = path
        self.matrices = {}
        self.xarrays = {}
        self.indices = {}

    def read_matrix(self,
                   name,
                   filename,
                   dimensions,
                   columns_names=None,
                   **kwargs):
        full_path = os.path.join(self.path, filename)
        if filename.endswith('.csv'):
            matrix = pd.read_csv(full_path, **kwargs)
        elif filename.endswith('.npy'):
            matrix = np.load(full_path, **kwargs)
        elif filename.endswith('.parquet'):
            matrix = pd.read_parquet(full_path, **kwargs)
        else:
            raise ValueError("Unsupported file format")

        if columns_names is not None:
            matrix.columns = columns_names
        else:
            # @TODO: flexible handling of index and column names
            matrix.columns = matrix.columns.astype(int)

        if dimensions is not None:
            if len(dimensions) == 2:
                matrix = matrix.stack()
            matrix.index = matrix.index.set_names(dimensions)
        else:
            raise ValueError("Dimensions must be specified")

        # if 'index_col' in kwargs:
        #     matrix.index = matrix.index.set_levels([i.astype(int) for i in matrix.index.levels])
        
        self.matrices[name] = matrix

    def set_matrix(self, name, matrix):
        self.matrices[name] = matrix

    def get_matrix(self, name):
        return self.matrices.get(name)

    def to_xarray(self):
        xarrays = {}
        for name, matrix in self.matrices.items():
            if isinstance(matrix, pd.DataFrame):
                for col in matrix.columns:
                    if len(matrix.columns) == 1:
                        xarrays[name] = matrix[col].to_xarray()
                    else:
                        xarrays[name] = matrix.unstack().to_xarray()
            elif isinstance(matrix, pd.Series):
                xarrays[name] = matrix.to_xarray()
        self.xarrays = xarrays
        return self

    def set_indices(self):
        # Add standard subscripts
        if self.xarrays == {}:
            self.to_xarray()
        
        self.indices['directions'] = pd.Index(['forward', 'backward'],
                                              name='directions')
        self.indices['in_out'] = pd.Index(['in', 'out'], name='flows')
        self.indices['c'] = pd.Index(['charge', 'discharge'], name='c')
        self.indices['producers'] = pd.Index(self.xarrays['A_p'].producers.values,
                                             name='producers')
        self.indices['storages'] = pd.Index(self.xarrays['A_s'].storages.values,
                                            name='storages')
        self.indices['edges'] = pd.Index(self.xarrays['A_i'].edges.values,
                                         name='edges')
        self.indices['nodes'] = pd.Index(self.xarrays['A_i'].nodes.values,
                                         name='nodes')#
        self.indices['consumers'] = pd.Index(self.xarrays['A_c'].consumers.values,
                                             name='consumers')
        return self.indices

    def validate(self):
        # Perform consistency tests here
        pass

    def read_csvs(self):
        """Read the matrices from the data directory and add them to the
        collection."""
        self.read_matrix('A_i', 'A_i.csv', dimensions=['nodes', 'edges'],
                            dtype=int, index_col=0)
        self.read_matrix('A_p', 'A_p.csv', dimensions=['nodes', 'producers'],
                            dtype=int, index_col=0)
        self.read_matrix('A_s', 'A_s.csv', dimensions=['nodes', 'storages'],
                            dtype=int, index_col=0)
        self.read_matrix('A_c', 'A_c.csv', dimensions=['nodes', 'consumers'],
                            dtype=int, index_col=0)
        self.read_matrix('l_i', 'l_i.csv', dimensions=['edges'],
                            index_col=0, columns_names=['length'])
        self.read_matrix('coordinates', 'rel_positions.csv',
                            dimensions=['nodes'],
                            columns_names=['x', 'y'], index_col=0)
        
        return self
