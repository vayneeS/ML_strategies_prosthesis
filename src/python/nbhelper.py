import pandas as pd
import json 

def load_data(filename):   
    # df = pd.read_json(filename, lines=True)
    try:
        df = pd.read_json(filename, lines=True)
        # df['createdAt'] = df['createdAt'].apply(lambda x: [i for i in x.values()][0])
        # df['updatedAt'] = df['updatedAt'].apply(lambda x: [i for i in x.values()][0])
    except Exception as e:
        df = pd.DataFrame()
        print(repr(e))
    return df