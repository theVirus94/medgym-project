import pandas as pd
from pandas import ExcelWriter
import os
def Merge_CSV(data_root):
    f1 = pd.read_excel(os.path.join(data_root, "Lung_Opacity.metadata.xlsx"))
    f2 = pd.read_excel(os.path.join(data_root, "Normal.metadata.xlsx"))

    f1["LABEL"] = 1
    f2["LABEL"] = 0

    # merging the files
    f3 = pd.concat([f1, f2], ignore_index=True)

    # creating a new file
    out_path = r"C:\Users\caner\Desktop\project\data"
    writer = pd.ExcelWriter((os.path.join(out_path, "Opacity_Normal.metadata.xlsx")), engine="openpyxl", mode="a",
                            if_sheet_exists="overlay")
    f3.to_excel(writer, sheet_name="Opacity_Normal.metadata")
    writer._save()
    writer.close()

def label_smoothing(labels, epsilon=0.1):
    """
    Smooths labels for better training
    """
    smoothed_labels = (1 - epsilon) * labels + epsilon / labels.size(0)
    return smoothed_labels