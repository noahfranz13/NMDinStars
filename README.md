# StellarEvolutionNeutrinoTests

Code to use a modified MESA stellar evolution code to test
the expected luminosity of the TRGB after introducing Neutrino
 Magnetic Dipole Moments to the star. The repo is organized as follows:

* `mesa_mod/mesa-r12778/star/test_suite/NeutrinoMagneticDipoleMesa/` is the directory with the MESA modifications, scripts to run the grid of models, and first post processing pipeline script.
* `ML_models` has all of the machine learning code to train and use the models
* `analysis` holds plotting code to create figures for papers and presentations. Additionally, the `output_analysis` subdirectory has code to merge post processing outputs.
* `makeGrids` has scripts to create the different input grid files to run MESA on.
* `mcmc` has the scripts and plots for the MCMC analysis.
