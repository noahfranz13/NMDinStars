# NeutrinoMagneticDiopoleMoments Mod
Directory housing my MESA mod to include Neutrino
Magnetic Dipole Moments.

To run a job array use the following steps:
1. Navigate to `~/NMDinStars/makeGrids/`
2. Run `python3 genGrids.py --n N` where you should
replace `N` with whatever side length you want on
the grid
3. Navigate to `~/NMDinStars/mesa_mod/mesa-r12276/star/test_suite/NeutrinoMagneticDipoleMoments/`
4. *First* run `./genGrid1`
5. Once you are happy with that, run `./runGrid2` to finish the grid

## Sample Inlists
Any file starting with "inlist" is a sample inslist used during this project. The inlist template used for grid runs is `inlist_template`.

## Checking to see if the grid is complete
1. Run the python3 script `findMissingRuns.py`. This will write to a file called `grid_to_rerun.csv`.
2. Change the output path in `nmdm_array_rerun.slurm` and run it to restart models that didn't finish.

## Post Processing
To post process use the slurm files called `postProcess{n}.slurm` where n is a number 1-3. If post processing grid 1, use `postProcess1.slurm`, etc. 

## Flag Key
| Flag No. | Meaning |
| -------- | ------- |
| 0 	   | No issues |
| 1	   | Model did not converge, should be excluded in analysis |
| 2	   | Model He Flashed unexpectedly |
| 3	   | Model is older than the universe (13.77 byo) |
