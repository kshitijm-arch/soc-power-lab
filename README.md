# SoC Power Lab

An illustrative SoC power modelling project that evaluates processor power and energy across frequency and voltage states. The model uses example assumptions and does not represent vendor measurements.

## Setup

Create a virtual environment and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe .\power_model.py
```

The script runs sanity checks, prints results for the configured P-states, and generates `plot1_energy_vs_freq.png` and `plot2_race_to_idle.png` in the current directory.
