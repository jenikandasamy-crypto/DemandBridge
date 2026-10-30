from pathlib import Path
import numpy as np, pandas as pd, streamlit as st, plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
st.set_page_config(page_title='DemandBridge',page_icon='🛒',layout='wide')
st.title('🛒 DemandBridge'); st.caption('AI-powered grocery intelligence | Customers • Channels • Demand')
@st.cache_data
def default_data():
 p=Path('data/synthetic_grocery_sales.csv')
 if not p.exists():
  from generate_synthetic_data import generate_data
  p.parent.mkdir(exist_ok=True); generate_data().to_csv(p,index=False)
 return pd.read_csv(p,parse_dates=['date'])
upload=st.sidebar.file_uploader('Upload DemandBridge-format CSV',type=['csv'])
horizon=st.sidebar.slider('Forecast horizon (days)',7,30,7)
try:
 df=pd.read_csv(upload,parse_dates=['date']) if upload else default_data()
 required=['date','store_id','product_id','product_name','category','channel','customer_id','unit_price','promotion','stock_available','demand_units','units_sold','lost_sales_units']
 missing=[x for x in required if x not in df.columns]
 if missing: st.error('Missing columns: '+', '.join(missing)); st.stop()
 df['date']=pd.to_datetime(df.date); df=df.sort_values('date')
 for c in ['unit_price','promotion','stock_available','demand_units','units_sold','lost_sales_units']: df[c]=pd.to_numeric(df[c],errors='coerce')
except Exception as e: st.error(f'Data loading error: {e}'); st.stop()
public=(df.customer_id=='NOT_PROVIDED').all(); target='units_sold' if public else 'demand_units'
a,b,c,d=st.columns(4); a.metric('Sales records',f'{len(df):,}'); b.metric('Stores',df.store_id.nunique()); c.metric('Products',df.product_id.nunique()); d.metric('Units sold',f'{df.units_sold.sum():,.0f}')
t1,t2,t3,t4=st.tabs(['Overview','Customers & Channels','Demand Forecast','Stock Opportunities'])
with t1:
 st.subheader('Grocery sales overview'); daily=df.groupby('date',as_index=False).units_sold.sum(); st.plotly_chart(px.line(daily,x='date',y='units_sold',title='Daily units sold'),use_container_width=True)
 cat=df.groupby('category',as_index=False).units_sold.sum().sort_values('units_sold',ascending=False); st.plotly_chart(px.bar(cat,x='category',y='units_sold',title='Sales by category'),use_container_width=True)
with t2:
 ch=df.groupby('channel',as_index=False).agg(units_sold=('units_sold','sum'),customers=('customer_id','nunique'))
 x,y=st.columns(2); x.plotly_chart(px.pie(ch,names='channel',values='units_sold',title='Sales share by channel'),use_container_width=True); y.plotly_chart(px.bar(ch,x='channel',y='customers',title='Unique customer IDs by channel'),use_container_width=True)
 if public: st.info('Favorita public data has no individual customer IDs or online/offline split. It contains store sales only. Customer and multi-channel insights are demonstrated using synthetic data.')
with t3:
 st.subheader(f'{horizon}-day demand forecast')
 if st.button('Train model & forecast',type='primary'):
  data=df.dropna(subset=[target]).copy(); data['dayofweek']=data.date.dt.dayofweek; data['month']=data.date.dt.month; cut=data.date.quantile(.8); tr=data[data.date<=cut]; te=data[data.date>cut]
  cats=['store_id','product_id','category','channel']; nums=['unit_price','promotion','dayofweek','month']; prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cats),('num','passthrough',nums)])
  model=Pipeline([('prep',prep),('rf',RandomForestRegressor(n_estimators=100,max_depth=14,min_samples_leaf=2,random_state=42,n_jobs=-1))]); model.fit(tr[cats+nums],tr[target]); pred=np.maximum(0,model.predict(te[cats+nums])); mae=mean_absolute_error(te[target],pred); rmse=mean_squared_error(te[target],pred)**.5
  prod=df.sort_values('date').groupby(['store_id','product_id','product_name','category','channel'],as_index=False).tail(1)[['store_id','product_id','product_name','category','channel','unit_price']]; prod=prod.loc[prod.index.repeat(horizon)].reset_index(drop=True); prod['date']=np.tile(pd.date_range(df.date.max()+pd.Timedelta(days=1),periods=horizon),len(prod)//horizon); prod['promotion']=0; prod['dayofweek']=prod.date.dt.dayofweek; prod['month']=prod.date.dt.month; prod['predicted_demand']=np.maximum(0,model.predict(prod[cats+['unit_price','promotion','dayofweek','month']]))
  st.session_state['fc']=prod; st.session_state['metrics']=(mae,rmse,len(te))
 if 'fc' in st.session_state:
  mae,rmse,n=st.session_state['metrics']; q,w,e=st.columns(3); q.metric('Hold-out MAE',f'{mae:.2f}'); w.metric('Hold-out RMSE',f'{rmse:.2f}'); e.metric('Test records',f'{n:,}'); st.caption(f'Model target: {target}. Chronological hold-out evaluation.')
  fc=st.session_state['fc']; sm=fc.groupby(['date','channel'],as_index=False).predicted_demand.sum(); st.plotly_chart(px.line(sm,x='date',y='predicted_demand',color='channel',title='Forecast demand by channel'),use_container_width=True); st.dataframe(fc.sort_values('predicted_demand',ascending=False).head(30),use_container_width=True); st.download_button('Download forecast CSV',fc.to_csv(index=False),'demandbridge_forecast.csv','text/csv')
 else: st.write('Click the button to train the model and view the forecast.')
with t4:
 if public or df.lost_sales_units.notna().sum()==0: st.info('Public dataset has no stock-on-hand or lost-sales fields. Use synthetic or inventory-enriched data for stockout insights.')
 else:
  opp=df.groupby(['store_id','product_id','product_name'],as_index=False).agg(units_sold=('units_sold','sum'),estimated_lost_units=('lost_sales_units','sum'),latest_stock=('stock_available','last')).sort_values('estimated_lost_units',ascending=False)
  st.plotly_chart(px.bar(opp.head(15),x='product_name',y='estimated_lost_units',color='store_id',title='Estimated lost units by product/store'),use_container_width=True); st.dataframe(opp.head(30),use_container_width=True)
st.divider(); st.caption('Synthetic records are simulated. Forecasts are estimates, not guaranteed future sales.')
