# HWPSS-prediction

This code takes an optical chain and telescope and calculates the A2 and A4 signal. 

## Setup 

Python 2.7 is required to run the program. The code is stored in `/src`. 
To set up the pythonpath, run  
```
source env.sh
``` 
from the root directory.

## Structure

To start the code, a Telescope object must be created. 
In the Telescope, data from config files are loaded and a list containing each element in the optical chain is loaded. Light is propagated through the telescope, and A2, A4, a2, and a4 are all calculated. Output files which give HWPSS signals and the contribution from each optical element can be generated to a specified directory.

##Input

As input, the code requires:

- `ExperimentDirectory`: An experiment directory, containing `camera.txt`, `channels.txt`, and `opticalChain.txt`. These are the same input files required by Charlie Hill's sensitivity calculations, and detailed descriptions of them can be found [on his github page.](https://github.com/smsimon/sensitivity_calculators/tree/master/SO_SensitivityCalculator/CHillCalc)
- `AtmosphereFile`: A file containing the atmosphere spectrum.
- `HWPFile`: A file containing the Half wave plate Mueller matrix
- `theta`: the incident angle of the incoming light
- `bandID`: The channel of the detector file being used. (1 or 2).

These are passed as a dictionary to to Telescope when initialized. A sample config file can be found in `/run/config.json`.  

## Running the code

A sample file that creates a telescope and prints A2 and A4 can be seen in `/run/runHWPSS.py`. 

The jupyter nb file `/run/HWPSS_demonstration.ipynb` shows off some other functionality of the code.