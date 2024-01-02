# LocallyUpdatedPlanes
A program of LUP method for optimization of reaction path. (However, implementation of this program is based on nudged elastic bond (NEB) method.)

## Required Modules
 - scipy
 - matplotlib
 - psi4
 - tblite
 - numpy

## Usage
For example, 


`python LUP.py aldol_rxn -ns 20` (use psi4 module for DFT calculation)


`python LUP.py aldol_rxn -ns 50 -xtb GFN2-xTB` (use tblite module for semiempirical QM calculation)
