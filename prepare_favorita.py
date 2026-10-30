"""Prepare a manageable sample from Kaggle Corporacion Favorita files.
Public data has store/item/date sales and promotions, but no customer IDs,
channel split or stock-on-hand. Do not claim those are real public fields."""
from pathlib import Path
import argparse, pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--input',default='data/favorita'); p.add_argument('--output',default='data/favorita_sample.csv'); p.add_argument('--chunksize',type=int,default=300000); p.add_argument('--max-rows',type=int,default=1000000); a=p.parse_args()
r=Path(a.input); train=r/'train.csv'; items=r/'items.csv'; stores=r/'stores.csv'
if not all(x.exists() for x in [train,items,stores]): raise FileNotFoundError('Put train.csv, items.csv and stores.csv in data/favorita/')
i=pd.read_csv(items); s=pd.read_csv(stores); parts=[]; total=0
for c in pd.read_csv(train,chunksize=a.chunksize):
 c=c[c.unit_sales>=0].copy().merge(i[['item_nbr','family']],on='item_nbr',how='left').merge(s[['store_nbr','city']],on='store_nbr',how='left')
 c['date']=pd.to_datetime(c.date); c['store_id']=c.store_nbr.astype(str); c['product_id']=c.item_nbr.astype(str); c['product_name']=c.family.fillna('Unknown'); c['category']=c.family.fillna('Unknown'); c['channel']='In-store'; c['customer_id']='NOT_PROVIDED'; c['unit_price']=0.0; c['promotion']=c.onpromotion.fillna(False).astype(int); c['stock_available']=pd.NA; c['demand_units']=c.unit_sales.astype(float); c['units_sold']=c.unit_sales.astype(float); c['lost_sales_units']=pd.NA
 parts.append(c[['date','store_id','product_id','product_name','category','channel','customer_id','unit_price','promotion','stock_available','demand_units','units_sold','lost_sales_units']]); total+=len(c)
 if total>=a.max_rows: break
out=pd.concat(parts,ignore_index=True).head(a.max_rows); Path(a.output).parent.mkdir(exist_ok=True); out.to_csv(a.output,index=False); print(f'Saved {len(out):,} rows to {a.output}')
