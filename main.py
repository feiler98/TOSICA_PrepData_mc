########################################################################################################################

#             generate filtered AnnData structures and save intermediate and final output as h5

########################################################################################################################

# imports
# ----------------------------------------------------------------------------------------------------------------------
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import scvi
import json


# run code
# ----------------------------------------------------------------------------------------------------------------------

# prepare pathing
path_cwd = Path("/scratch/tmp/feiler").resolve()
path_mc = path_cwd / "benchmark_metacells"
path_out = path_cwd / "outTOSICA_data_prep"
path_out.mkdir(exist_ok=True, parents=True)

def get_json_dict(json_file_path: (str | Path)) -> dict:
    """
    Parameters
    ----------
    json_file_path : str | Path
        Absolute-path of JSON file.

    Returns
    -------
    dict
        JSON file as dictionary.
    """

    if not Path(json_file_path).exists():
        raise FileExistsError("JSON File could not be found!")
    with open(json_file_path, 'r') as in_file:
        return json.load(in_file)

def var_idx_ensg_to_symbol(adata: sc.AnnData) -> sc.AnnData:
    """
    Parameters
    ----------
    adata: sc.AnnData

    Returns
    -------
    sc.AnnData
    """

    path_ensg_to_json = Path(__file__).parent / "ensg_to_symbol.json"
    dict_ensg_to_symbol = get_json_dict(path_ensg_to_json)
    list_accept = [x for x in adata.var.index if x in dict_ensg_to_symbol.keys()]
    adata = adata[:, list_accept]
    adata.var["symbol"] = [dict_ensg_to_symbol[x] for x in adata.var.index]
    adata.var.set_index("symbol", inplace=True)
    return adata

def metacell_out_to_adata(path_metacells_parent_dir: (str | Path),
                          recursive: bool = True,
                          target_file_tag: str = "*soft_assignment",
                          min_metacell_count: (int | None) = None) -> sc.AnnData | None:
    """
    Specialized method for fetching and concat metacells from a previously established SEACells pipeline.
    Fetches soft-assigned metacells which were saved as .csv RCM files.

    Parameters
    ----------
    path_metacells_parent_dir: str | Path
    recursive: bool
        Enables recursive folder search.
    target_file_tag: str
    min_metacell_count: int | None
        If None or x < 1, sets variable to 1.

    Returns
    -------
    sc.AnnData
    """

    path = Path(path_metacells_parent_dir).resolve()
    if not path.exists() and not path.is_dir():
        raise ValueError("Given Path is invalid! A directory was expected.")

    if recursive:
        list_mc_paths = list(path.rglob(f"{target_file_tag}.csv"))
    else:
        list_mc_paths = list(path.glob(f"{target_file_tag}.csv"))

    if min_metacell_count is None or min_metacell_count < 1:
        min_metacell_count = 1

    dict_adata_mc = {}
    for p in list_mc_paths:
        df_import = pd.read_csv(p, index_col="Gene").T
        if len(df_import) < min_metacell_count:
            print(f"""skip >> {p.name}
    --> set min_metacell_count ({min_metacell_count}) > file cell count ({len(df_import)}) """)
        else:
            print(f"import >> {p.name}")
            adata = sc.AnnData(pd.read_csv(p, index_col="Gene").T)
            adata.obs["cell_tags"] = [t.split("__")[-1] for t in adata.obs.index]
            adata.obs["batch_key"] = [p.stem.split("__")[1]]*len(adata)
            dict_adata_mc[p.stem] = adata
    if len(dict_adata_mc) > 0:
        adata_concat = sc.concat(dict_adata_mc, join="outer")
        adata_concat.obs_names_make_unique()
        return adata_concat
    else:
        return None

# data prep
adata_concat = metacell_out_to_adata(path_mc, recursive=True, min_metacell_count=10)
adata_concat.X = np.nan_to_num(adata_concat.X, nan=0)
adata_concat = var_idx_ensg_to_symbol(adata_concat)
adata_concat.obs_names_make_unique()

# relabel cell classes by curated df
df_rename = pd.read_excel("metacell_benchmark_cell_class_curation.xlsx")
dict_rename = {k: v for k,v in zip(list(df_rename["cell_tag"]), list(df_rename["curated_cell_tag"]))}
adata_concat.obs["curated_cell_tag"] = adata_concat.obs["cell_tags"].map(lambda x : dict_rename[x])
filter_remove_list = [idx for idx, tag in zip(adata_concat.obs.index, list(adata_concat.obs["curated_cell_tags"])) if tag != "remove"]
# remove all cells marked for removal in curated_cell_tags
adata_concat = adata_concat[filter_remove_list, :]

# for integration, further cleaning is required
min_instances_per_class = 20
all_cell_tags = list(adata_concat.obs["curated_cell_tag"])
sorted_cell_tags = sorted(list(adata_concat.obs["curated_cell_tag"].unique()), key=str.casefold)
dict_cell_classes = {cell_class: all_cell_tags.count(cell_class) for cell_class in sorted_cell_tags}
list_cells_keep = [cell for cell, class_tag in
                   zip(list(adata_concat.obs.index), list(adata_concat.obs["curated_cell_tag"]))
                   if dict_cell_classes[class_tag] >= min_instances_per_class]
adata_concat = adata_concat[list_cells_keep, :]
adata_concat.write(path_out / "metacell_benchmark_cell_curated_no_integration.h5")

# integration scVI & scANVI
# store original counts
adata_concat.layers["counts"] = adata_concat.X
scvi.model.SCVI.setup_anndata(adata_concat, layer="counts", batch_key="batch_key")
model = scvi.model.SCVI(adata_concat, n_layers=2, n_latent=30, gene_likelihood="nb")
model.train()
scanvi_model = scvi.model.SCANVI.from_scvi_model(
    model,
    adata=adata_concat,
    labels_key='curated_cell_tag',
    unlabeled_category="Unknown",
)
scanvi_model.train(max_epochs=40, n_samples_per_label=100) # 40 epochs sufficient for fine tuning
SCANVI_LATENT_KEY = "X_scANVI"
adata_concat.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata_concat)
sc.pp.neighbors(adata_concat, use_rep=SCANVI_LATENT_KEY)
sc.tl.umap(adata_concat, min_dist=0.3)

sc.pp.highly_variable_genes(adata_concat, inplace=True, n_top_genes=2000, flavor='seurat_v3_paper', subset=True)
list_integration_genes_2K = list(adata_concat.var.index)
with open(str(path_out/"integration_genes_2K")) as f:
    f.write("\n".join(list_integration_genes_2K))
adata_concat.write(path_out / "metacell_benchmark_cell_curated_integration_gene_2K.h5")


# reload adata and visualize the data before integration while using only the 2K genes from the integration
adata_concat = sc.read(path_out / "metacell_benchmark_cell_curated_no_integration.h5")
adata_concat = adata_concat[:, list_integration_genes_2K]
sc.pp.neighbors(adata_concat, use_rep="X")
sc.tl.umap(adata_concat, min_dist=0.3)
adata_concat.write(path_out / "metacell_benchmark_cell_curated_no_integration_gene_2K.h5")