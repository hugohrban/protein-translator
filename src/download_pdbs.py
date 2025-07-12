import os
import sys
import requests

# List of PDB IDs to download
pdb_ids = sys.argv[1:]

# Directory where downloaded files will be saved
output_dir = "pdb_downloads"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Base URLs for mmCIF and FASTA downloads
mmcif_base_url = "https://files.rcsb.org/download/{}.cif"
fasta_base_url = "https://www.rcsb.org/fasta/entry/{}"

for pdb_id in pdb_ids:
    pdb_id_lower = pdb_id.lower()

    #=== Download mmCIF ===#
    mmcif_url = mmcif_base_url.format(pdb_id_lower)
    mmcif_path = os.path.join(output_dir, f"{pdb_id_lower}.cif")

    try:
        r = requests.get(mmcif_url, timeout=10)
        r.raise_for_status()
        with open(mmcif_path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded mmCIF for {pdb_id} → {mmcif_path}")
    except requests.RequestException as e:
        print(f"Failed to download mmCIF for {pdb_id}: {e}")

    #=== Download FASTA ===#
    fasta_url = fasta_base_url.format(pdb_id_lower)
    fasta_path = os.path.join(output_dir, f"{pdb_id_lower}.fasta")

    try:
        r = requests.get(fasta_url, timeout=10)
        r.raise_for_status()
        # The RCSB FASTA endpoint returns plain text, so we save it as text
        with open(fasta_path, "w") as f:
            f.write(r.text)
        print(f"Downloaded FASTA for {pdb_id} → {fasta_path}")
    except requests.RequestException as e:
        print(f"Failed to download FASTA for {pdb_id}: {e}")

