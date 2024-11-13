# Topotherm Storage

This is the open source software accompaining the publication submitted to Elsevier.

## Description

A district heating network design model powered by linopy to design renewable heating supply with
network topology and pipe sizing.

Features:

* MILP formulation for pipe routing and topology
* Flexible representative periods and time segmentation
* Renewable energy supply technologies, including weather-dependent (for example solar thermal) and variable operational costs (for example heat pumps)
* Intra-day thermal energy storage

## Installing

Use git to clone this repository into your computer. Then, install topotherm
with a package manager such as Anaconda, or directly with Python.

```git
git clone URL
```
### Anaconda or mamba

We recommend to install the dependencies with anaconda or mamba:

```mamba
cd tt-storage
mamba env create -n tt-storage python=3.12
mamba activate tt-storage
pip install .
```

### Solver

#### Gurobi

The results in the paper were obtained with the commercial solver gurobi.
A free academic license is available and can be installed by following
the documentation [here](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-).

#### Open-source Alternatives

You can try the code on smaller benchmarks with several open source solvers,
such as SCIP. Other popular open-source options are COIN-OR's cbc or HiGHS.

```mamba
mamba activate topotherm
mamba install -c conda-forge pyscipopt
```

## Usage

Generate the input incidence matrices for the district with .parquet format (see example).
Then, modify and run the either one of the three scripts in that folder.

```bash
cd example
python run_sts.py
```

## License

MIT, see LICENSE file.

## Cite

Please cite the full publication. A Zenodo DOI is also available for the code itself.

## Authors

* Ceruti, Amedeo

