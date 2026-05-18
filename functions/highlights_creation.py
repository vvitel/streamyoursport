import pandas as pd

def create_highlights(df, nb_frame_to_separate):
    df = df.sort_values("frame")
    df["block"] = (df["frame"].diff().fillna(0) >= nb_frame_to_separate).cumsum()

    dico = {}
    for i in df.block.unique():
        df_block = df[df.block == i]
        first_frame = min(df_block.frame)
        last_frame = max(df_block.frame)
        length = len(df_block)
        std_y = df_block.y.std()
        dico[i] = [first_frame, last_frame, length, std_y]

    df = pd.DataFrame.from_dict(dico, orient="index")
    df = df.sort_values(by=df["length"]**2 + df["std_y"]**2, ascending=False)
    first_row = df.iloc[0]
    first_frame = first_row["first_frame"]
    last_frame = first_row["last_frame"]

    print("\U0001F60D : ", first_frame, last_frame)