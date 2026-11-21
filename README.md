# G041 — MP093: Ration Shop Distribution (Nagpur)

**Group ID:** G041
**Problem ID:** MP093
**Course:** SCSE3040 MLOps — Mini Project

## Members

| Enrolment No. | Name |
|---|---|
| S24CSEU2415 | Prasantkumar |
| S24CSEU2390 | Rishabh Jain |
| S24CSEU2413 | Sushant Kumar |
| S24CSEU0497 | Shaurya Nigam |

## Description

Fair price (ration) shops in Nagpur hand out grain to ration card holders, and some shops run out of stock by noon. For the Nagpur Food and Civil Supplies Office, this project predicts `grain_kg`, the amount of grain each shop will distribute in a day. It is a regression problem measured by MAE, so the office can send stock in time and no family goes home empty-handed. The repository holds the full MLOps pipeline: versioned data with DVC, finding and cleaning the data problems in `train.csv`, experiment tracking and a model registry with MLflow, a FastAPI prediction service with the client's input rules, a Docker container, CI with GitHub Actions, Kubernetes manifests, structured JSON logging, drift monitoring with PSI, and retraining on newer data.

## Setup (Windows PowerShell)

```powershell
git clone <this repo> ; cd g041-mp093
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .   # makes `import mp093` work
```

`train.csv` and `later.csv` are tracked with DVC, not git (there is no DVC
remote). Copy them **unchanged** from the problem zip into `data/raw/`, then
check that they match the `.dvc` files:

```powershell
.\.venv\Scripts\dvc.exe status    # must say "Data and pipelines are up to date."
```

Never open `train.csv` in Excel and save it: its hash would stop matching.

## Run

| Command | What it does |
|---|---|
| `python -m mp093.check_data` | Looks for the seven data problems in `train.csv` (results go in `reports/findings.yaml`) |
| `python -m mp093.data` | Cleans `train.csv` in memory and checks the result; never writes the file |

Every setting (paths, columns, detection thresholds, cleaning, MLflow names)
is in `params.yaml`.
