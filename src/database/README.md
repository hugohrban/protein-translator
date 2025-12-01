# Designs database

<!-- design_id,reference_structure,optimization_reference,beam_size,optimize_plddt,plddt_scaling_factor,distance_threshold,clustering_reference,clustering_threshold,precision,mutate_early,random_seed -->
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

`DB_LOCK` is a lock file to ensure only one process is writing to the database at a time. It either contains `LOCKED` or is empty.