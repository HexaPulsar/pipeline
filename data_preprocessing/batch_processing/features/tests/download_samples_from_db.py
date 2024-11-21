import sqlalchemy as sa
import requests
import pandas as pd
import os


data_folder = "data"
script_path = os.path.dirname(os.path.abspath(__file__))
data_folder_full_path = os.path.join(script_path, data_folder)

if not os.path.exists(data_folder_full_path):
    os.makedirs(data_folder_full_path)

url = "https://raw.githubusercontent.com/alercebroker/usecases/master/alercereaduser_v4.json"
params = requests.get(url).json()["params"]

engine = sa.create_engine(
    f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['dbname']}"
)
engine.begin()

oids = ["ZTF19aaxqrku", "ZTF18aavoomb", "ZTF20aatqgeo", "ZTF24aahxmmu", "ZTF18abfydvt"]
oids = [f"'{oid}'" for oid in oids]

query_detections = f"""
SELECT * FROM detection
WHERE oid in ({','.join(oids)});
"""
detections = pd.read_sql_query(query_detections, con=engine)
print(detections)
print(detections.columns)

detections.to_parquet(os.path.join(data_folder_full_path, "detections.parquet"))


query_forced_photometry = f"""
SELECT * FROM forced_photometry
WHERE oid in ({','.join(oids)});
"""
forced_photometry = pd.read_sql_query(query_forced_photometry, con=engine)
print(forced_photometry)
print(forced_photometry.columns)

forced_photometry.to_parquet(
    os.path.join(data_folder_full_path, "forced_photometry.parquet")
)

query_xmatch = f"""
SELECT oid, oid_catalog, dist FROM xmatch
WHERE oid in ({','.join(oids)}) and catid='allwise';
"""
xmatch = pd.read_sql_query(query_xmatch, con=engine)
xmatch = xmatch.sort_values("dist").drop_duplicates("oid")
oid_catalog = [f"'{oid}'" for oid in xmatch["oid_catalog"].values]

query_wise = f"""
SELECT oid_catalog, w1mpro, w2mpro, w3mpro, w4mpro FROM allwise
WHERE oid_catalog in ({','.join(oid_catalog)});
"""
wise = pd.read_sql_query(query_wise, con=engine).set_index("oid_catalog")
wise = pd.merge(xmatch, wise, on="oid_catalog", how="outer")
wise = wise[["oid", "w1mpro", "w2mpro", "w3mpro", "w4mpro"]].set_index("oid")

query_ps = f"""
SELECT oid, sgscore1, sgmag1, srmag1, distpsnr1 FROM ps1_ztf
WHERE oid in ({','.join(oids)});
"""
ps = pd.read_sql_query(query_ps, con=engine)
ps = ps.drop_duplicates("oid").set_index("oid")

xmatch = pd.concat([wise, ps], axis=1).reset_index()
print(xmatch)

xmatch.to_parquet(os.path.join(data_folder_full_path, "xmatch.parquet"))
