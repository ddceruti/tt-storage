# Topotherm Storage

This is the open source software accompaining the publication submitted to Elsevier. The preprint is available here: <https://papers.ssrn.com/abstract=5024324>.

## Description

A district heating network design model powered by linopy to design renewable heating supply with
network topology and pipe sizing.

Features:

* MILP formulation for pipe routing and topology based on topotherm
* Flexible representative periods and time segmentation
* Renewable energy supply technologies, including weather-dependent (for example solar thermal) and variable operational costs (for example heat pumps)
* Intra-day thermal energy storage

## Installing

Use git to clone this repository into your computer. Then, install topotherm
with a package manager such as Anaconda, or directly with Python.

### Cloning the repository

```git
git clone https://github.com/ddceruti/tt-storage.git
```

### Anaconda or mamba

We recommend to install the dependencies with anaconda or mamba:

```mamba
cd tt-storage
mamba env create -n tt-storage python=3.12
mamba activate tt-storage
pip install -r requirements.txt
```

### Solver

#### Gurobi

The results in the paper were obtained with the commercial solver gurobi version 11.
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

Generate the input incidence matrices for the district with .csv format (see examples).
Then, run the main script with the appropiate arguments.

```bash
mamba activate topotherm
python run.py
```

## License

MIT, see LICENSE file.

## Cite

Please cite the full publication. A Zenodo DOI is also available for the code itself.

## Authors

* Ceruti, Amedeo
