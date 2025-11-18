import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv("DATA\Flight_delay.csv")
engine = create_engine("sqlite:///data.db")
df.to_sql("table_name", engine, if_exists="replace", index=False)
