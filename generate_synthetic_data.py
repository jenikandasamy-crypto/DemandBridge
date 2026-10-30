from pathlib import Path
import numpy as np, pandas as pd

def generate_data(days=180, seed=42):
    rng=np.random.default_rng(seed)
    dates=pd.date_range(end=pd.Timestamp.today().normalize()-pd.Timedelta(days=1), periods=days)
    stores={'CHN-01':1.15,'CHN-02':.95,'CHN-03':1.05,'CHN-04':.85}
    products=[('P001','Rice','Staples',58,22),('P002','Wheat Flour','Staples',42,17),('P003','Milk','Dairy',32,30),('P004','Curd','Dairy',28,18),('P005','Eggs','Dairy',7,24),('P006','Tomato','Produce',36,20),('P007','Onion','Produce',30,19),('P008','Cooking Oil','Staples',145,9),('P009','Biscuits','Snacks',20,26),('P010','Detergent','Household',95,8)]
    customers=[f'C{i:04d}' for i in range(1,801)]; rows=[]
    for d in dates:
      weekend=1.25 if d.dayofweek>=5 else 1.0
      season=1+.12*np.sin(2*np.pi*d.dayofyear/365.25)
      for store,sf in stores.items():
       for pid,name,cat,price,base in products:
        for channel in ['In-store','Online']:
         promo=int(rng.random()<.16); cf=1 if channel=='In-store' else .72; pf=1.35 if promo else 1
         demand=max(0,int(rng.poisson(base*sf*weekend*season*cf*pf)))
         stock=max(0,int(rng.normal(base*sf*weekend*(.82 if rng.random()<.13 else 1.35),5)))
         rows.append([d,store,pid,name,cat,channel,rng.choice(customers) if channel=='Online' else 'WALK_IN',round(price*(.95 if promo else 1),2),promo,stock,demand,min(demand,stock),max(0,demand-stock)])
    cols=['date','store_id','product_id','product_name','category','channel','customer_id','unit_price','promotion','stock_available','demand_units','units_sold','lost_sales_units']
    return pd.DataFrame(rows,columns=cols)
if __name__=='__main__':
    p=Path('data/synthetic_grocery_sales.csv'); p.parent.mkdir(exist_ok=True)
    df=generate_data(); df.to_csv(p,index=False); print(f'Created {p}: {len(df):,} rows')
