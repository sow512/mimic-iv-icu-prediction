# =====================================================
# DATASET 75
# =====================================================

DATASET_75_PATH = Path(
    "."
    "Final Group Project/Post mid-term/"
    "YOUR_75_FEATURE_FILE.xlsx"
)

df_75 = pd.read_excel(
    DATASET_75_PATH,
    sheet_name="Final_75_Features"
)

train_75 = df_75[
    df_75["dataset_split"] == "train"
].copy()

test_75 = df_75[
    df_75["dataset_split"] == "test"
].copy()

assert len(train_75) + len(test_75) == len(df_75)
assert len(train_75) == 66118
assert len(test_75) == 22223

# Create one fixed patient-disjoint validation split.
# Reuse these same patients for the 200-feature experiment.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        train_75,
        train_75[TARGET],
        groups=train_75["subject_id"]
    )
)

validation_subject_ids = set(
    train_75.iloc[val_idx]["subject_id"]
)

# Train and evaluate
results_75 = run_models(
    dataset_name="75_Features",
    train_data=train_75,
    test_data=test_75,
    validation_subject_ids=validation_subject_ids
)

display(results_75.round(4))