# pyiron publication template
This is a template repository how you can publish your calculation with pyiron. It consists of the repository [itself](https://github.com/pyiron/pyiron-publication-template), a small [website](http://pyiron.org/pyiron-publication-template/) created with Jupyterbook and a [mybinder environment](https://mybinder.org/v2/gh/pyiron/pyiron-publication-template/HEAD?filepath=notebooks%2Fexample.ipynb) for testing the calculation. 

You can fork this repository and populate it with your own data.

## Step by step
* Move your notebooks to the repository folder and remove the example notebook `example.ipynb`.
* Update the conda `environment.yml` file with the conda dependencies required for your notebook. 
* Include the export of your pyiron database in the `pyiron/calculation` folder or in case no calculation are required you can remove the `pyiron/calculation/save.tar.gz` archive and the `pyiron/calculation/export.csv` database backup file. 
* Include additional pyiron resources in the `pyiron/resources` folder if required, otherwise the `pyiron/resources` folder can be deleted.

