########################################################################################################################

#             generate filtered AnnData structures and save intermediate and final output as h5

########################################################################################################################

# imports
# ----------------------------------------------------------------------------------------------------------------------
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import utilTOSICA.preprocessing as uTpre
import utilTOSICA.utils.vis_utils as uTv
import scvi

# run code
# ----------------------------------------------------------------------------------------------------------------------

# prepare pathing
path_cwd = Path("/scratch/tmp/feiler").resolve()
path_mc = path_cwd / "benchmark_metacells"
path_out = path_cwd / "outTOSICA_data_prep"
path_out.mkdir(exist_ok=True, parents=True)


# data prep
adata_concat = uTpre.metacell_out_to_adata(path_mc, recursive=True, min_metacell_count=10)
adata_concat.X = np.nan_to_num(adata_concat.X, nan=0)
adata_concat = uTpre.var_idx_ensg_to_symbol(adata_concat)
adata_concat.obs_names_make_unique()

# relabel cell classes by curated df
df_rename = pd.read_excel("metacell_benchmark_cell_class_curation.xlsx")
dict_rename = {k: v for k,v in zip(list(df_rename["cell_tag"]), list(df_rename["curated_cell_tag"]))}
adata_concat.obs["curated_cell_tags"] = adata_concat.obs["cell_tags"].map(lambda x : dict_rename[x])
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
    labels_key='curated_cell_tags',
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

uTv.plotly_adata_umap(adata_concat, header="TOSICA train & test data - scVI integrated", save_dir=path_out)

# reload adata and visualize the data before integration while using only the 2K genes from the integration
adata_concat = sc.read(path_out / "metacell_benchmark_cell_curated_no_integration.h5")
adata_concat = adata_concat[:, list_integration_genes_2K]
adata_concat.write(path_out / "metacell_benchmark_cell_curated_no_integration_gene_2K.h5")
sc.pp.neighbors(adata_concat, use_rep="X")
sc.tl.umap(adata_concat, min_dist=0.3)
uTv.plotly_adata_umap(adata_concat, header="TOSICA train & test data - not integrated", save_dir=path_out)