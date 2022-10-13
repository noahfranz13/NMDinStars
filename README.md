# StellarEvolutionNeutrinoTests

Code to use a modified MESA stellar evolution code to test
the expected luminosity of the TRGB after introducing Neutrino
 Magnetic Dipole Moments to the star. The repo is organized as follows:

* `mesa_mod/mesa-r12778/star/test_suite/NeutrinoMagneticDipoleMesa/` is the directory with the MESA modifications, scripts to run the grid of models, and first post processing pipeline script.
* `ML_models` has all of the machine learning code to train and use the models
* `analysis` holds plotting code to create figures for papers and presentations. Additionally, the `output_analysis` subdirectory has code to merge post processing outputs.
* `makeGrids` has scripts to create the different input grid files to run MESA on.
* `mcmc` has the scripts and plots for the MCMC analysis.

## Installing MESA
These steps follow from: https://docs.mesastar.org/en/release-r22.05.1/installation.html#download-mesa. Here are some additional notes to ensure you are installing the same version of MESA used for this project:

1. Install the MESA SDK from this link: http://user.astro.wisc.edu/~townsend/static.php?ref=mesasdk. You want to download the file with version `mesasdk-x86_64-macos-gcc9-21.9.1.pkg`.
2. You can then install MESA version 12778 from the older releases link here: https://zenodo.org/record/6547951#.Y0dp30zMJD8

