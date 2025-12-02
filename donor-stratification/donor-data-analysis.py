import os
import h5py
import pandas as pd

DATA_DIR = "./data"
A9_FILEPATH = os.path.join(DATA_DIR, "SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad")
MTG_FILEPATH = os.path.join(DATA_DIR, "SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad")

DONOR_KEY = "Donor ID"
ADNC_KEY = "ADNC"


def load_obs_column(f, key):
    obj = f["obs"][key]

    codes = obj["codes"][:]  # integer codes
    categories = obj["categories"][:].astype("U")
    return categories[codes]
    
    raise ValueError(f"Unknown format for obs key: {key}")


def load_donors_from_file(filepath, region_label):
    """Return cell-level dataframe + donor-level dataframe."""
    print(f"\n=== Loading {region_label} ===")
    with h5py.File(filepath, "r") as f:
        donors = load_obs_column(f, DONOR_KEY)
        adnc   = load_obs_column(f, ADNC_KEY)

    df = pd.DataFrame({"donor": donors, "ADNC": adnc})
    print("Total cells:", len(df))
    print("Unique donors:", df["donor"].nunique())
    print("Cell-level ADNC counts:")
    print(df["ADNC"].value_counts())

    donor_table = (
        df.groupby("donor")["ADNC"]
          .agg(["nunique", "first", "count"])
          .reset_index()
          .rename(columns={"first": "ADNC_donor", "count": "n_cells"})
    )
    donor_table["region"] = region_label

    print("\nDonors per ADNC level:")
    print(
        donor_table["ADNC_donor"].value_counts().sort_index()
    )

    return df, donor_table


if __name__ == "__main__":
    df_a9, donors_a9 = load_donors_from_file(A9_FILEPATH, "A9")
    df_mtg, donors_mtg = load_donors_from_file(MTG_FILEPATH, "MTG")

    combined_donors = pd.concat([donors_a9, donors_mtg], ignore_index=True)

    print("\n======================")
    print("Combined Donors Summary")
    print("======================")

    print("Total donors (A9 + MTG):", len(combined_donors))

    print("\nCombined donors per ADNC level:")
    print(
        combined_donors["ADNC_donor"].value_counts().sort_index()
    )

    print("\nDonors per ADNC per region:")
    print(
        combined_donors.groupby(["region", "ADNC_donor"]).size()
    )
