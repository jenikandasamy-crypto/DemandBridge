# DemandBridge – Grocery & Supermarket Edition

Prototype for customer, channel and demand intelligence using Exasol, Python and Streamlit.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_synthetic_data.py
streamlit run app.py
```

## Public dataset
Download Corporacion Favorita Grocery Sales Forecasting files from Kaggle:
https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data
Accept and follow competition rules. Put `train.csv`, `items.csv`, and `stores.csv` in `data/favorita/`, then run:
```bash
python prepare_favorita.py --input data/favorita --max-rows 1000000
```
Upload `data/favorita_sample.csv` through the app.

**Data honesty:** Favorita provides store/item/date sales and promotion data. It does not provide individual customer IDs, an online/offline split, or stock-on-hand. Those capabilities are demonstrated only with clearly labelled synthetic records unless real business data is supplied.

## Exasol
Install Exasol Personal using https://docs.exasol.com/db/latest/get_started/exasol_personal.htm and run `exasol_setup.sql`. PyExasol is the official Python driver. Automatic bulk upload from the app is a later milestone and is not yet implemented in this starter.

Never upload Kaggle source files or credentials to public GitHub unless the relevant terms allow it. Synthetic data is generated locally by the script.
