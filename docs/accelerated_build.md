# Accelerated PRISM-X Build

The project now contains an end-to-end core runner in addition to the modular scripts.

## Run the full core pipeline

From the `PRISM-X` folder with `(.venv)` active:

```powershell
python scripts/run_prismx.py
```

If the saved price file does not exist, or if it should be refreshed:

```powershell
python scripts/run_prismx.py --download
```

## Open the dashboard

```powershell
streamlit run dashboard/app.py
```

## Optional model layers

Install XGBoost, SHAP and optimisation/network extras:

```powershell
python -m pip install -e ".[advanced]"
```

Install FinBERT:

```powershell
python -m pip install -e ".[sentiment]"
```

Install TensorFlow for the LSTM module:

```powershell
python -m pip install -e ".[deep-learning]"
```

The optional models are not silently substituted with invented outputs. If their dependencies or data are unavailable, the system reports that limitation.
