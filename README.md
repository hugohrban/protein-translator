# Protein Translator 🧬🧩 📃➡📜

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

Download file(s) from the PDB (FASTA and MMCIF):

```bash
python3 download_pdb.py 1fe4 6c2u # ... or any other PDB IDs
```

### Run the Translator

Example: running the translator using Clustering + Beam Search.

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
    --dmap_reference cb \
```

Run `translate.py --help` to see all options and explanations.
