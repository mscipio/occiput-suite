# Changelog

![#1589F0](https://placehold.it/15/1589F0/000000?text=+)
[**v1.1.1 --> v0.5.0**] - **06 Feb 2018**
![#1589F0](https://placehold.it/15/1589F0/000000?text=+)
### Occiput 1.1.1 is merged into a new module called Occiput-suite 0.5.0

- ***occiput-suites*** now containts *all occiput's dependencies* so to keep everything updated and consistent
- occiput is now compatible with **Python up to 3.6.4**: further testing is required, but it seems to be working fine, being still backward compatible with python 2.7.11 systems
- *minor changes*: now the beatiful ***ProgressBar()*** class is back to work!
- Updated version of example notebooks (still a work in progress)

---

![#1589F0](https://placehold.it/15/1589F0/000000?text=+)
[**v1.1.0**] - **22-Sep-2017**
![#1589F0](https://placehold.it/15/1589F0/000000?text=+)
### Many changes to a stable 1.1.0 version by ***Michele Scipioni***

- General *code polishing* of file *occiput.Reconstruction.PET*.**PET.py**: now it's easier to navigate, with functions with similar behavior grouped togheter inside each PET-object class.
- Solved a **critical error** in ***subset generation*** when using OS-EM version of reconstruction algorithm for all type of PET object (static, dynamic, cyclic, or MultiSlice2D) due to recomputation of the subsetting scheme at each iteration, resulting in non uniform use of the entire raw data space (sinogram).
- New version (minor changes) of the **Quick_inspect** method in *occiput.Reconstruction.PET*.**PET.py**: this is used to show a summary of imported raw data, to see if everything is consistend in terms of scaling. Now it is possible to:
    - set the shape of the generated figure;
    - visualize a legend stating what we are looking at;
    - force the xlim of the plot to match the shape of prompts data (this helps if we want to plot both alongside, to study consistency in scanner sensitivity matrix).
- Fixed a **critical issue** regarding **import_randoms** function in *occiput.Reconstruction.PET*.**PET.py**: old version caused an implicit conversion from *numpy.float32* to *numpy.float64*, causing error during the reconstruction for compatibility issue with the GPU. Now everything should be properly imported.
- Similar to previous point, the method **apply_poisson_noise** in *occiput.Reconstruction.PET*.**PET_projection.py**, being based on *numpy.random.Poisson*, caused an implicit casting of prompts.data from *numpy.float32* to *numpy.int64*, making it impossible for the reconstruction code to properly manage input data. Now the ouput of numpy function is correctly recasted to *numpy.float32*.
- Added templates for function implementing [*** we will need quite some more work to have an actual functioning version ***]:
    - direct reconstruction of 4D dynamic PET
    - regularized reconstruction based on a range of different priors (mainly for dynamic time series, but easily adaptable to static scans as well)
- Minor changes to *occiput.Reconstruction.PET.PET_Static_Scan*.**osem_reconstruction()** in order to:
    + improve printed output of iteration number (visual feedback)
    + add a secondary output with stored the intermediate images after each OS-EM iteration(*mainly testing purposes*)
    + add a boolean flag that enable saving on disk of the output of each iteration (*mainly testing purposes*)

---
![#1589F0](https://placehold.it/15/1589F0/000000?text=+)
[v**1.0.0**] - 10-Jan-2017 
![#1589F0](https://placehold.it/15/1589F0/000000?text=+)
<a href=""><img src="https://cdn4.iconfinder.com/data/icons/pictype-free-vector-icons/16/forward-128.png" width="25" height="10"></a> Launch of version 1.0.0 by ***Stefano Pedemonte***

---
![#1589F0](https://placehold.it/15/1589F0/000000?text=+)
[v0.2.0] - 03-Feb-2015 
![#1589F0](https://placehold.it/15/1589F0/000000?text=+)
<a href=""><img src="https://cdn4.iconfinder.com/data/icons/pictype-free-vector-icons/16/forward-128.png" width="25" height="10"></a> First stable release, utilized in occiput.io

---
![#1589F0](https://placehold.it/15/1589F0/000000?text=+)
[v0.1.0] - 14-Oct-2013 
![#1589F0](https://placehold.it/15/1589F0/000000?text=+)
<a href=""><img src="https://cdn4.iconfinder.com/data/icons/pictype-free-vector-icons/16/forward-128.png" width="25" height="10"></a> Initial release
