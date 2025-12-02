import os
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict

DATA_DIR = "./data"

A9_FILEPATH  = os.path.join(DATA_DIR, "SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad")
MTG_FILEPATH = os.path.join(DATA_DIR, "SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad")

DONOR_KEY = "Donor ID"
ADNC_KEY  = "ADNC"
CLASS_KEY = "Class"
SCVI_KEY  = "X_scVI"

CHUNK_SIZE = 200_000  # stream 200k cells at a time;


def load_obs_column(f, key):
    obj = f["obs"][key]
    codes = obj["codes"][:]
    categories = obj["categories"][:].astype("U")
    return categories[codes]


def extract_region_latents(h5ad_path, region_label):
    print(f"\n=== Extracting donor latents for region: {region_label} ===")
    print(f"File: {h5ad_path}")

    with h5py.File(h5ad_path, "r") as f:
        donors = load_obs_column(f, DONOR_KEY)
        adnc   = load_obs_column(f, ADNC_KEY)
        classes = load_obs_column(f, CLASS_KEY)

        latent = f["obsm"][SCVI_KEY]
        n_cells, latent_dim = latent.shape
        print(f"Latent shape: {n_cells} cells × {latent_dim} dims")

    class_list = sorted(set(classes))
    print(f"Classes present ({len(class_list)}): {class_list}")

    sum_latent = {
        (donor, cls): np.zeros(latent_dim, dtype=np.float64)
        for donor in set(donors)
        for cls in class_list
    }

    count_latent = defaultdict(int)

    # Streamed loading
    with h5py.File(h5ad_path, "r") as f:
        latent = f["obsm"][SCVI_KEY]

        for start in range(0, n_cells, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_cells)

            # load chunk rows: donor ids, classes, latents
            d_slice = donors[start:end]
            c_slice = classes[start:end]
            l_slice = latent[start:end, :]

            for donor_val, cls_val, vec in zip(d_slice, c_slice, l_slice):
                key = (donor_val, cls_val)
                sum_latent[key] += vec
                count_latent[key] += 1

            print(f"Processed {end}/{n_cells} cells", flush=True)

    donor_ids = sorted(set(donors))
    rows = []
    for donor_id in donor_ids:
        for cls in class_list:
            key = (donor_id, cls)
            cnt = count_latent[key]
            if cnt > 0:
                mean_vec = sum_latent[key] / cnt
            else:
                # No cells of this class for this donor --> fill with zeros
                mean_vec = np.zeros(latent_dim)

            rows.append({
                "donor": donor_id,
                "region": region_label,
                "Class": cls,
                "ADNC": None,  # fill later
                **{f"z{d}": mean_vec[d] for d in range(latent_dim)}
            })

    donor_latents = pd.DataFrame(rows)

    adnc_per_donor = (
        pd.DataFrame({"donor": donors, "ADNC": adnc})
        .groupby("donor")["ADNC"]
        .first()
        .to_dict()
    )
    donor_latents["ADNC"] = donor_latents["donor"].map(adnc_per_donor)

    print("Done extracting donor-level latent embeddings.")
    return donor_latents


if __name__ == "__main__":
    latents_a9  = extract_region_latents(A9_FILEPATH,  "A9")
    latents_mtg = extract_region_latents(MTG_FILEPATH, "MTG")

    full_latents = pd.concat([latents_a9, latents_mtg], ignore_index=True)

    print("\nSaving donor-wise latent embeddings to CSV...")
    full_latents.to_csv(os.path.join(DATA_DIR, "donor_latents_per_class_scvi.csv"), index=False)
    print("Saved: donor_latents_per_class_scvi.csv")
