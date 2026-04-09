import matplotlib.pyplot as plt
import pandas as pd

df_norm = pd.read_csv("C:/Users/Utilisateur/Downloads/gloubigloubus1.csv")
df_quant = pd.read_csv("C:/Users/Utilisateur/Downloads/gloubigloubus2.csv")
df_quant_x2 = pd.read_csv("C:/Users/Utilisateur/Downloads/gloubigloubus3.csv")

#frame sur lesquelles il y a une détection
plt.eventplot(df_norm.frame, lineoffsets=0, colors="blue", label="normal")
plt.eventplot(df_quant.frame, lineoffsets=1, colors="red", label="quantization")
plt.eventplot(df_quant_x2.frame, lineoffsets=2, colors="orange", label="quantizationX2")
plt.legend()
plt.show()