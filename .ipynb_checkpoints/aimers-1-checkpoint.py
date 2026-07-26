import pandas as pd
Data_URL= "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
COLUMNS=["fLength","fWidth","fSize","fConc",
           "fConc1","fAsym","fM3Long","fM3Trans",
           "fAlpha","fDist","class"]
df=pd.read_csv(Data_URL,header=None,names=COLUMNS)
print(df.shape)
df.head()