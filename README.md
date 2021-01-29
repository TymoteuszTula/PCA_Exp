# PCA_Exp
Software that automatize performing principal component analysis on a set of measurements from experiment, stored in textfiles. Software is developed to be used as a python package.

Main goal of this package is to allow for quick implementation of principal component analysis on experimental data. Given few folders with text files that contain different measurements (for example at different temperatures) in the form:
```
x      y      ey
0.1    0.4    0.02
0.2    0.35   0.02
0.3    0.28   0.015
...    ...    ...
```
the code can preprocess it, join different sets of measurements, perform PCA and return the results. 

### Requirements
- Python v3.0 or higher
- numPy 
- matplotlib

### Installation
Put the pca_exp folder in your working directory.

### Usage
Check the jupyter notebook pca_exp_tutorial.ipynb for a quick tutorial that explains most functionalities of the code.

### License
Standard GNU General Public License v3.0. Check COPYING file.
