# Monkey patch get_batch_cell_embeddings to force single processor
import types
from scgpt.tasks.cell_emb import get_batch_cell_embeddings as original_get_batch_cell_embeddings
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from scgpt.data_collator import DataCollator
import numpy as np
from tqdm import tqdm

# Define Dataset class at module level
class CellEmbeddingDataset(Dataset):
    """
    Custom dataset class for cell embedding generation.
    Processes single-cell data into a format suitable for the scGPT model.
    """
    def __init__(self, count_matrix, gene_ids, batch_ids=None, vocab=None, model_configs=None):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.batch_ids = batch_ids
        self.vocab = vocab
        self.model_configs = model_configs

    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        # Process a single cell's data
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]
        # append <cls> token at the beginning
        genes = np.insert(genes, 0, self.vocab["<cls>"])
        values = np.insert(values, 0, self.model_configs["pad_value"])
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).float()
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        if self.batch_ids is not None:
            output["batch_labels"] = self.batch_ids[idx]
        return output

def patched_get_batch_cell_embeddings(
    adata,
    cell_embedding_mode: str = "cls",
    model=None,
    vocab=None,
    max_length=1200,
    batch_size=64,
    model_configs=None,
    gene_ids=None,
    use_batch_labels=False,
) -> np.ndarray:
    """
    Patched version of get_batch_cell_embeddings that uses the module-level Dataset class
    and forces num_workers=0 for better compatibility.
    """
    # Convert data to appropriate format
    count_matrix = adata.X
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.toarray()
    )

    # Get gene vocabulary ids
    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    if use_batch_labels:
        batch_ids = np.array(adata.obs["batch_id"].tolist())

    if cell_embedding_mode == "cls":
        # Set up dataset and data loader
        dataset = CellEmbeddingDataset(
            count_matrix, 
            gene_ids, 
            batch_ids if use_batch_labels else None,
            vocab=vocab,
            model_configs=model_configs
        )
        collator = DataCollator(
            do_padding=True,
            pad_token_id=vocab[model_configs["pad_token"]],
            pad_value=model_configs["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=0,  # Force single worker for compatibility
            pin_memory=True,
        )

        # Generate embeddings
        cell_embeddings = np.zeros(
            (len(dataset), model_configs["embsize"]), dtype=np.float32
        )
        with torch.no_grad():
            count = 0
            for data_dict in tqdm(data_loader, desc="Embedding cells"):
                # Process each batch of cells
                input_gene_ids = data_dict["gene"].to(device)
                src_key_padding_mask = input_gene_ids.eq(
                    vocab[model_configs["pad_token"]]
                )
                embeddings = model._encode(
                    input_gene_ids,
                    data_dict["expr"].to(device),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=data_dict["batch_labels"].to(device)
                    if use_batch_labels
                    else None,
                )

                # Extract CLS token embeddings and normalize
                embeddings = embeddings[:, 0, :]  # get the <cls> position embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count : count + len(embeddings)] = embeddings
                count += len(embeddings)
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
    else:
        raise ValueError(f"Unknown cell embedding mode: {cell_embedding_mode}")
    return cell_embeddings

# Replace the original function with our patched version
import scgpt.tasks.cell_emb
scgpt.tasks.cell_emb.get_batch_cell_embeddings = patched_get_batch_cell_embeddings

os.environ['PYTHONWARNINGS'] = 'ignore'

model_dir = MODEL_DIR
gene_col = "feature_name"
cell_type_key = "cell_type"

embedding_file = DATA_DIR / "ref_embed_adata.h5ad"

if embedding_file.exists():
    print(f"Loading existing embeddings from {embedding_file}")
    ref_embed_adata = sc.read_h5ad(str(embedding_file))
else:
    print("Computing new embeddings...")
    ref_embed_adata = scg.tasks.embed_data(
        adata_hvg,
        model_dir,
        gene_col=gene_col,
        obs_to_save=cell_type_key,
        batch_size=64,
        return_new_adata=True,
        device=device,
    )
    print(f"Saving embeddings to {embedding_file}")
    ref_embed_adata.write(str(embedding_file))

