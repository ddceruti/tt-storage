import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import networkx as nx
from matplotlib.colors import ListedColormap


# Define TUM colors
colors = [
    '#003359',
    '#005293',
    '#0065BD',
    '#64A0C8',
    '#98C6EA',
    '#DAD7CB'
]

CMAP = ListedColormap(colors, name='TUM-colors')

ACCENT = '#E37222'
HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
STYLES = [':', '--', '-.', '-',]

plt.rcParams.update({
    # 'text.usetex': True,  # if you want to render with LaTeX (requires LaTeX)
    'font.family': 'Arial'
})

plt.style.use('seaborn-v0_8-paper')


def operation(model, names, q_c):
    df_G = model.variables.G.solution.to_series().unstack('producers')
    df_S = model.variables.S.solution.to_series().unstack('storages')
    df_Q = q_c.to_series().unstack('consumers')

    total_plots = len(df_Q.index.get_level_values('periods').unique())

    # df_G.set_index('producers', append=True, inplace=True)
    fig, axs = plt.subplots(1, total_plots, figsize=(15, 1.2 * total_plots))

    for i in range(total_plots):
        prod = df_G.loc[(i, slice(None)), :]
        prod.index = prod.index.droplevel(0)
        chg = df_S.loc[('charge', i, slice(None)), :]
        chg.index = chg.index.droplevel(('c', 'periods'))
        dchg = df_S.loc[('discharge', i, slice(None)), :]
        dchg.index = dchg.index.droplevel(('c', 'periods'))

        axs[i].plot(df_Q.sum(axis=1).loc[(i, slice(None))],
                    label='Total Demand', linestyle='-.')
        for k in prod.columns.unique().values:
                axs[i].plot(prod.loc[:, k],
                            label=names.loc[k], linestyle='--')
        for s in chg.columns.unique().values:
            axs[i].plot(-chg.loc[:, s],
                        label=f'Charge Storage {s}', linestyle=':')
            axs[i].plot(dchg.loc[:, s],
                        label=f'Discharge Storage {s}')

        if i != 0:
            axs[i].set_ylabel('')
        else:
            axs[i].set_ylabel('Heat Power in kW')
            # prod.solution.plot(ax=axs[i], labels=prod.producers, legend=True)
            # store.soltuion.plot(ax=axs[i], labels=store.storages, legend=True)

        axs[i].set_title(f'Typical Period {i}')
        axs[i].set_xlabel('Time Segment')


    # middle of the figure, of all plots
    middle = total_plots // 2
    axs[middle].legend(bbox_to_anchor=(0.5, 1.1), ncol=3,
                          loc='lower center', title='Heat Source',
                          title_fontsize='large')

    fig2, ax2 = plt.subplots(1, total_plots, figsize=(15, 1.2 * total_plots))

    for i in range(total_plots):
        prod = df_G.loc[(i, slice(None)), :]
        prod.index = prod.index.droplevel(0)
        chg = df_S.loc[('charge', i, slice(None)), :]
        chg.index = chg.index.droplevel(('c', 'periods'))
        dchg = df_S.loc[('discharge', i, slice(None)), :]
        dchg.index = dchg.index.droplevel(('c', 'periods'))

        # Total demand plot (this will not be stacked, just overlaid)
        ax2[i].plot(df_Q.sum(axis=1).loc[(i, slice(None))],
                    label='Total Demand', linestyle='-.',
                    color=ACCENT)

        # Stack the producer data
        prod_values = [prod.loc[:, k].values for k in prod.columns.unique()]
        ax2[i].stackplot(prod.index, *prod_values,
                         labels=[names.loc[k] for k in prod.columns.unique()],
                         colors=CMAP.colors
                         )

        # Stack the charge and discharge data (stack negative charges)
        chg_values = [-chg.loc[:, s].values for s in chg.columns.unique()]
        dchg_values = [dchg.loc[:, s].values for s in dchg.columns.unique()]
        ax2[i].stackplot(chg.index, *chg_values,
                         labels=[f'Charge Storage {s}' for s in chg.columns.unique()],
                         hatch='//',
                         colors=CMAP.colors[3:])
        ax2[i].stackplot(dchg.index, *dchg_values,
                         labels=[f'Discharge Storage {s}' for s in dchg.columns.unique()],
                         hatch='\\\\',
                         colors=CMAP.colors[3:])

        if i != 0:
            ax2[i].set_ylabel('')
        else:
            ax2[i].set_ylabel('Heat Power in kW')

        ax2[i].set_title(f'Typical Period {i}')
        ax2[i].set_xlabel('Time Segment')

    ax2[middle].legend(bbox_to_anchor=(0.5, 1.1), ncol=3,
                loc='lower center', title='Heat Source',
                title_fontsize='large')

    # make fig three with stacked plot but with sum of all producers, storages
    # and demand as line

    fig3, ax3 = plt.subplots(1, total_plots, figsize=(15, 1.2 * total_plots))

    y_max = (df_G.sum(axis=1)
             + df_S.loc[('discharge', slice(None), slice(None)), :].sum(axis=1)
             ).max()
    y_min = df_S.loc[('charge', slice(None), slice(None)), :].sum(axis=1).max()

    for i in range(total_plots):
        # Get the data for the current period
        prod = df_G.loc[(i, slice(None)), :]
        prod.index = prod.index.droplevel(0)  # Dropping the 'periods' index level

        chg = df_S.loc[('charge', i, slice(None)), :]
        chg.index = chg.index.droplevel(('c', 'periods'))  # Dropping unnecessary index levels for charge

        dchg = df_S.loc[('discharge', i, slice(None)), :]
        dchg.index = dchg.index.droplevel(('c', 'periods'))  # Dropping unnecessary index levels for discharge

        # Total demand plot (this will not be stacked, just overlaid)
        ax3[i].plot(df_Q.sum(axis=1).loc[(i, slice(None))],
                    label='Total Demand', linestyle='-.', color=ACCENT)

        columns = names.values.tolist() \
            + [f'Charge Storage {s}' for s in chg.columns.unique()] \
                + [f'Discharge Storage {s}' for s in dchg.columns.unique()]
        df_plot = pd.DataFrame(index=prod.index,
                               columns=columns)
        # map producer index to names and add to df_plot
        for k in prod.columns.unique():
            df_plot.loc[:, names.loc[k]] = prod.loc[:, k].values
        # map charge index to names and add to df_plot
        for s in chg.columns.unique():
            df_plot.loc[:, f'Charge Storage {s}'] = -chg.loc[:, s].values
        # map discharge index to names and add to df_plot
        for s in dchg.columns.unique():
            df_plot.loc[:, f'Discharge Storage {s}'] = dchg.loc[:, s].values

        # Stack the data
        df_plot.plot.area(ax=ax3[i], stacked=True, color=CMAP.colors,
                          legend=False,
                          )

        # Axis labels
        if i != 0:
            ax3[i].set_ylabel('')
        else:
            ax3[i].set_ylabel('Heat Power in kW')

        ax3[i].set_title(f'Typical Period {i}')
        ax3[i].set_xlabel('Time Segment')

        ax3[i].set_ylim(-y_min*1.1, y_max*1.1)

    # Place the legend in the middle subplot
    middle = total_plots // 2
    ax3[middle].legend(bbox_to_anchor=(0.5, 1.1), ncol=3,
                    loc='lower center', title='Heat Source',
                    title_fontsize='large')

    return fig, fig2, fig3


def demand_profiles(df: pd.DataFrame, path: str):
    """Plot the profiles."""
    fig, ax = plt.subplots()

    df_plot = df.rename(columns={'sfh': 'Single Family House',
                       'mfh': 'Multi Family House',
                       'ghd': 'Commercial',
                       'pub': 'Public Building'})
    # Plot each line with different styles
    for i, c in enumerate(df_plot.columns):
        df_plot.iloc[:, i].resample('D').sum().plot(
                ax=ax, color=CMAP.colors[i], label=c,
                linewidth=1.2, linestyle=STYLES[i], alpha=0.8)

    ax.set_ylabel('Heat Demand in kWh/day')
    ax.set_xlabel('Time')
    ax.legend(title='Building Type', loc='upper center',
              frameon=True, ncol=2)
    fig.savefig(path, bbox_inches='tight')
    return


def solution(model, collection):
    df_A_i = collection.get_matrix('A_i').unstack(1)
    df_A_p = collection.get_matrix('A_p').unstack(1)
    df_A_c = collection.get_matrix('A_c').unstack(1)
    df_A_s = collection.get_matrix('A_s').unstack(1)
    df_Q = collection.xarrays['Q_c'].to_series().unstack('consumers')
    coords = collection.get_matrix('coordinates')

    df_nodes = pd.DataFrame(index=df_A_i.index.get_level_values('nodes').unique(),
                            columns=['x', 'y', 'consumer', 'producer',
                                     'junction', 'storage_capacity',
                                     'production_capacity', 'demand'])
    df_nodes.loc[:, 'x'] = coords.x
    df_nodes.loc[:, 'y'] = coords.y
    df_nodes.index.name = 'node'
    # if A_p[node] == -1 then it is a producer
    df_nodes.loc[:, 'producer'] = (df_A_p.values == -1).any(axis=1)
    df_nodes.loc[:, 'consumer'] = (df_A_c.values == 1).any(axis=1)
    df_nodes.loc[:, 'junction'] = (df_A_i.values != 0).any(axis=1)
    df_nodes.loc[:, 'storage'] = (df_A_s.values != 0).any(axis=1)

    df_nodes.junction = df_nodes.junction | ~df_nodes.producer & ~df_nodes.consumer
    df_edges = pd.DataFrame(index=df_A_i.columns.unique(),
                            columns=['node_in', 'node_out', 'P_build'])
    df_edges.loc[:, 'P_build'] = model.variables.P_build.solution.to_series()
    for i in df_edges.index:
        df_edges.loc[i, 'node_in'] = np.where(df_A_i.loc[:, i] == 1)[0][0] #
        df_edges.loc[i, 'node_out'] = np.where(df_A_i.loc[:, i] == -1)[0][0]
        # mix df_coords with df_edges.node_in and node_out to assign coordinates to nodes
    df_edges = df_edges.join(df_nodes.loc[:, ['x', 'y']],
                             on='node_in', rsuffix='_in')
    df_edges = df_edges.join(df_nodes.loc[:, ['x', 'y']],
                             on='node_out', rsuffix='_out')

    df_nodes.loc[np.where(df_A_c.values == 1)[0], 'demand'] = df_Q.max(axis=0).values
    df_nodes.loc[:, 'demand'] = df_nodes.demand.fillna(0.).astype(float)
    df_nodes = df_nodes.infer_objects(copy=False)  # Ensure correct data types

    # put into an undirected graph with weights depending on P_build
    G = nx.Graph()
    for i in collection.indices['edges']:
        G.add_edge(df_edges.loc[i, 'node_in'], df_edges.loc[i, 'node_out'],
                   weight=df_edges.loc[i, 'P_build'] / 20)

    # color nodes depending on 'consumer', 'producer', 'junction'
    node_color = []
    marker = []
    for i in df_nodes.index:
        if df_nodes.loc[i, 'producer']:
            node_color.append('C0')
            marker.append('o')
        elif df_nodes.loc[i, 'consumer']:
            node_color.append('C2')
            marker.append('s')
        # elif df_nodes.loc[i, 'storage']:
        #     node_color.append('C3')
        else:
            node_color.append('C1')
            marker.append('^')

    # node size depending on demand
    node_size = df_nodes.demand**1.5 + 100
    # dhn topology plot
    fig1, ax = plt.subplots(figsize=(5, 4))
    # draw and width of edges depending on weight of edge
    nx.draw(G, coords.values, with_labels=False, ax=ax, edge_color='grey',
            width=4,
            node_color=node_color, node_size=node_size,
            font_size=10, font_weight='bold')
    plt.tight_layout()
    # node legend
    ax.scatter([],[], c='C0', label='Producer and TES', s=100)
    ax.scatter([],[], c='C2', label='Consumer', s=100)
    ax.scatter([],[], c='C1', label='Junction', s=100)
    fig1.legend(scatterpoints=1,
               frameon=True, title='Node Type',
               labelspacing=1, loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, 0.96))
    fig1.show()

    # dhn topology plot
    fig, ax = plt.subplots(figsize=(5, 4))
    # draw and width of edges depending on weight of edge
    nx.draw(G, coords.values, with_labels=False, ax=ax, edge_color='grey',
            width=[G[u][v]['weight'] for u,v in G.edges()],
            node_color=node_color, node_size=node_size, font_size=10)
    plt.tight_layout()
    # node legend
    plt.scatter([],[], c='C0', label='Producer and TES', s=100)
    plt.scatter([],[], c='C2', label='Consumer', s=100)
    plt.scatter([],[], c='C1', label='Junction', s=100)
    plt.legend(scatterpoints=1, frameon=True, title='Node Type',
               labelspacing=1, loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, 0.96))
    return fig1, fig
