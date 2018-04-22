## Installing ##
1. Clone the repository
2. Import the module into python via `pip install --user -e /PATH/TO/REPO/`
3. Start importing and using the functions


## Usage ##

These functions are mainly for caluclating weighted quality parameters from separations done on weighted events.
You do your crossvalidated separation and write the relevant attributes to a .csv file, one file for each loop of the validation.
The functions need a `label`, `prediction`, and a `confidence`. `weights` are optional.

A function then iterates over all possible values for the confidence between 0 and 1 (granularity is provided by the user) and calculates the number of tp, tn, fp, fn for each cut and step of the crossvalidation.
These numbers are then used to calculate different types of quality parameters.

## Example ##
```from analysis_functions import quality_parameters as qp
corsikaNuNuFeatures = gen_quality_from_csv(path+'*corsika_features*')
oneselection_plotter(corsikaNuNuFeatures, title='largeRmrmr')```
