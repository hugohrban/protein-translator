# Protein Translator 📃➡📜

**Ever wanted to translate a protein into an _ancient_ alphabet of 10 amino acids, while keeping the _same structure_? No? Well, now you can!**

_Presented as a poster at ISMB/ECCB 2025. See the poster PDF and a visualization of the translation process [here](https://www.ms.mff.cuni.cz/~hrbanh/eccb2025/)._

## Usage

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download ESMFold Model

First, download ESMFold locally.  
(I prefer this, but you can run it through `transformers` as well. In that case, change the code accordingly.)

```bash
mkdir esmfold_v1 && cd esmfold_v1
base_url="https://huggingface.co/facebook/esmfold_v1/resolve/main"
for file in pytorch_model.bin special_tokens_map.json vocab.txt README.md config.json tokenizer_config.json; do
  curl -OL "$base_url/$file"
done
```

### Download Data

_This step is optional, you can run the translation directly from any FASTA and MMCIF files._

Download file(s) from the PDB (FASTA and MMCIF):

```bash
python3 download_pdb.py 1fe4 6c2u # ... or any other PDB IDs
```

### Run the Translator

Example: running the translator using Clustering + Beam Search algo.

```bash
python3 -u translate.py \
    --input_fasta pdb_downloads/1fe4.fasta \
    --cluster_beam  \
    --beam_size 10 \
    --wrt_pdb \
    --optim_plddt \
    --plddt_scaling_factor 10 \
    --device "cuda:0" \
    --random_seed 0 \
    --translations 5 \
    --distance_threshold 7 \
    --clustering_proportion 0.6 \
    --dmap_reference cb
```

Run `python3 translate.py --help` to see all options and explanations.

### Designs Database

Generated designs are stored in `database/designs` as `.pdb` files. Their filenames are random uuids. Input parameters and final metrics about the designs are stored in `database/designs.jsonl` as a JSON lines file. Each line corresponds to one design. It has the following fields:
- `design_id`: Unique identifier for each design.
- `plddt`: Average predicted Local Distance Difference Test score of the designed structure.
- `rmsd`: Root Mean Square Deviation between the designed structure and the reference structure.
- `tm_score`: TM-score between the designed structure and the reference structure.
- `reference_structure`: Path to the reference structure file used for design.
- `optimization_reference`: Reference structure used during optimization. One of 'pdb' or 'esm' (optimizing rmsd w.r.t. esmfold prediction or original PDB).
- `beam_size`: Beam search parameter (w).
- `optimize_plddt`: Boolean indicating whether to optimize pLDDT.
- `plddt_scaling_factor`: Scaling factor for pLDDT optimization. - scoring functoin is: score = rmsd + (plddt_scaling_factor * (1 - plddt)). we use pLDDT scaled between 0 and 1, since this is what the model outputs.
- `distance_threshold`: Distance threshold for creating the contact map before clustering.
- `clustering_reference`: Part of the residue used for creating the distance map for clustering. One of 'cb' (C-beta) or 'com' (center of mass).
- `clustering_proportion`: Float between 0 and 1, parameter of bottom up hierarchical clustering. If at least this ratio of nodes in two clusters are in contact, the clusters are merged.
- `precision`: Floating point precistion of ESMFold, 'fp32' or 'bf16' or 'fp16'.
- `mutate_early`: Boolean indicating whether to also perform mutation of early residues.
- `random_seed`: Random seed for reproducibility.

`database/DB_LOCK` is a lock file to ensure only one process is writing to the database at a time. It either contains `LOCKED` or is empty.
