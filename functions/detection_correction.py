import math
import pandas as pd

def correct_detection(lst_data, step):
    #convertir la liste en dataframe
    colnames = ["x", "y", "width", "height", "conf", "frame"]
    df = pd.DataFrame(lst_data, columns=colnames)

    #filrer les détections inférieures à .7
    threshold_conf = 0.7
    df = df[df["conf"] > threshold_conf].reset_index(drop=True)

    #supprimer les détections statiques
    to_remove = set()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if abs(df.loc[i, "frame"] - df.loc[j, "frame"]) <= step:
                dx = df.loc[i, "x"] - df.loc[j, "x"]
                dy = df.loc[i, "y"] - df.loc[j, "y"]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 5:
                    to_remove.add(i)
                    to_remove.add(j)

    df = df.drop(list(to_remove)).reset_index(drop=True)
    return df