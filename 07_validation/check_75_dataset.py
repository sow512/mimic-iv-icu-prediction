
from pathlib import Path
import pandas as pd

file = Path(__file__).resolve().parent / "75_Feature_Dataset.xlsx"

print("File exists:", file.exists())

excel = pd.ExcelFile(file)
print("\nSheet names:", excel.sheet_names)

for sheet in excel.sheet_names:
    df = pd.read_excel(file, sheet_name=sheet)

    print("\n" + "=" * 50)
    print("SHEET:", sheet)
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())

    for col in [
        "subject_id",
        "hadm_id",
        "icu_transfer",
        "icu_transfer_within_24h",
        "dataset_split"
    ]:
        if col in df.columns:
            print(f"{col}: PRESENT")

            if col in [
                "icu_transfer",
                "icu_transfer_within_24h",
                "dataset_split"
            ]:
                print(df[col].value_counts(dropna=False))
