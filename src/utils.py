import os
import sys
import torch
import numpy as np
import argparse
from Bio.PDB import MMCIFParser, PDBParser
from constants import ALL_AA, EARLY_AA, LATE_AA, THREE_TO_ONE_AA
from transformers.models.esm.openfold_utils import atom14_to_atom37, OFProtein, to_pdb


def get_num_early_aa(seq: str) -> int:
    """
    Count the number of early amino acids in a sequence.

    Args:
        seq (str): The protein sequence.

    Returns:
        int: The count of early amino acids.
    """
    return sum(1 for aa in seq if aa in EARLY_AA)


def get_num_late_aa(seq: str) -> int:
    """
    Count the number of late amino acids in a sequence.

    Args:
        seq (str): The protein sequence.

    Returns:
        int: The count of late amino acids.
    """
    return sum(1 for aa in seq if aa in LATE_AA)


def get_ratio_early_aa(seq: str) -> float:
    """
    Calculate the ratio of early amino acids to the total number of amino acids in a sequence.

    Args:
        seq (str): The protein sequence.

    Returns:
        float: The ratio of early amino acids, or 0 if the sequence is empty.
    """
    if len(seq) == 0:
        return 0.0
    return get_num_early_aa(seq) / len(seq)


def get_ratio_late_aa(seq: str) -> float:
    """
    Calculate the ratio of late amino acids to the total number of amino acids in a sequence.

    Args:
        seq (str): The protein sequence.

    Returns:
        float: The ratio of late amino acids, or 0 if the sequence is empty.
    """
    if len(seq) == 0:
        return 0.0
    return get_num_late_aa(seq) / len(seq)


def get_seq_from_structure_file(filename: str) -> str:
    """
    Extract the protein sequence from a PDB / CIF file.

    Args:
        filename (str): The path to the PDB or CIF file.

    Returns:
        str: The protein sequence.
    """
    parser = (
        MMCIFParser(QUIET=True) if filename.endswith(".cif") else PDBParser(QUIET=True)
    )
    structure = parser.get_structure("protein", filename)
    seq = []
    # chain = next(next(structure))
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    try:
                        seq.append(THREE_TO_ONE_AA[residue.resname])
                    except KeyError:
                        seq = []
                        break
            if seq != []:
                break
        if seq != []:
            break
    seq_early_only = "".join((aa if aa in EARLY_AA else ".") for aa in seq)
    seq = "".join(seq)
    return seq_early_only, seq


def round_list_of_lists(lists: list, decimals: int = 2) -> list:
    """
    Round each element in a list of lists to a specified number of decimal places.

    Args:
        lists (list): A list of lists containing numerical values.
        decimals (int): The number of decimal places to round to.

    Returns:
        list: A new list of lists with rounded values.
    """
    return [
        ["{:.02f}".format(round(value, decimals)) for value in sublist]
        for sublist in lists
    ]


def output_to_pdb(output: dict, index: int) -> str:
    """Returns the pbd (file) string from the model given the model output.
    Contrary to the original implementation, this function only creates the pdb string for one structure, not all.
    """
    output = {
        k: (v.to(torch.float32) if v.dtype == torch.bfloat16 else v).to("cpu").numpy()
        for k, v in output.items()
    }
    final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
    final_atom_mask = output["atom37_atom_exists"]
    pdbs = []
    for i in range(output["aatype"].shape[0]):
        if i != index:
            continue
        aa = output["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = output["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=output["plddt"][i],
        )
        pdbs.append(to_pdb(pred))
    return pdbs[0]


def boolean_clustering(
    adj: np.ndarray | torch.Tensor,
    seq: str | None = None,
    linkage: str = "single",
    proportion: float = 0.5,
) -> list[list[int]]:
    """
    Cluster indices given a symmetric Boolean neighbour matrix.

    Parameters
    ----------
    adj : (N, N) bool ndarray
        `adj[i, j] == True`  ⇔  points *i* and *j* are neighbours.
        Must be symmetric; the diagonal is ignored.
    seq : str or None, optional
        Amino acid sequence of length N. If provided, positions corresponding
        to amino acids in EARLY_AA will be excluded from clustering (i.e.,
        their rows and columns in the adjacency matrix are set to False).
    linkage : {'single', 'partial', 'complete'}
        * 'single'   –  merge if **any** edge exists between two clusters
                       (ordinary connected components).
        * 'complete' –  merge only if **every** possible edge between the
                       clusters exists (clique / complete-linkage).
        * 'partial'  –  merge if the fraction of present edges between two
                       clusters is ≥ `proportion` (0 < proportion ≤ 1).
    proportion : float, default 1.0
        Only used with `linkage='partial'`.
        Example: `proportion=0.3` means “at least 30 % of the pairs are
        already neighbours” (average/centroid-style compromise).

    Returns
    -------
    clusters : list[list[int]]
        Each inner list contains sorted point indices belonging to one
        cluster.  Clusters themselves are returned in ascending size order.
    """
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()

    early_aa_ixs = []
    if seq is not None:
        early_aa_ixs = np.array([i for i, aa in enumerate(seq) if aa in EARLY_AA])
        adj[early_aa_ixs, :] = False
        adj[:, early_aa_ixs] = False

    n = adj.shape[0]
    if adj.shape != (n, n):
        raise ValueError("adjacency matrix must be square")
    if not np.all(adj == adj.T):
        raise ValueError("adjacency matrix must be symmetric")

    # ---------- SINGLE-LINKAGE: depth-first search / union–find ----------
    if linkage == "single":
        visited = np.zeros(n, dtype=bool)
        visited[early_aa_ixs] = True  # skip late amino acids
        clusters = []
        for start in range(n):
            if visited[start]:
                continue
            stack = [start]
            comp = []
            visited[start] = True
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in np.where(adj[v])[0]:
                    if not visited[u]:
                        visited[u] = True
                        stack.append(u)
            clusters.append(sorted(comp))
        return clusters

    # ---------- COMPLETE or PARTIAL: iterative agglomeration ----------
    if linkage not in {"partial", "complete"}:
        raise ValueError("linkage must be 'single', 'partial' or 'complete'")

    # start with singletons and iteratively merge
    clusters = [[i] for i in range(n) if i not in set(early_aa_ixs)]
    np.random.shuffle(clusters)  # randomize order to avoid bias
    need_change = True
    if len(clusters) <= 1:
        need_change = False
    while need_change:
        need_change = False
        i = 0
        while i < len(clusters) and not need_change:
            j = i + 1
            while j < len(clusters) and not need_change:
                A, B = clusters[i], clusters[j]
                edges = adj[np.ix_(A, B)]
                if linkage == "complete":
                    cond = edges.all()
                else:  # 'partial'
                    cond = edges.sum() / edges.size >= proportion
                if cond:
                    clusters[i] = A + B  # merge
                    clusters.pop(j)
                    need_change = True
                else:
                    j += 1
            if not need_change:
                i += 1

    return sorted((sorted(c) for c in clusters), key=lambda c: (len(c), c))


def get_tmp_dir_name(args: argparse.Namespace) -> str:
    """
    Get the temporary directory name based on the provided arguments.

    Args:
        args (argparse.Namespace): The command line arguments.

    Returns:
        str: The name of the temporary directory.
    """
    pdb_id = os.path.basename(args.input_fasta).split(".")[0][:4].lower()

    method = ""
    if args.cluster_beam:
        method = f"clust_beam_w{args.beam_size}_d{args.distance_threshold}_c{args.clustering_proportion}"
    elif args.greedy:
        method = "greedy"
    elif args.distance_based:
        method = "distance_based"

    wrt_to = f"wrt{'pdb' if args.wrt_pdb else 'esm'}"
    plddt_optim = f"optim_plddt{f'T_sf{args.plddt_scaling_factor}' if args.optimize_plddt else 'F'}"
    mutate_early = f"mut_early{'T' if args.mutate_early else 'F'}"
    precision = "fp32" if args.fp32 else "bf16"
    reference_coords_dist_map = "ref_" + args.dmap_reference
    tmp_dir = f"auto_temps/{pdb_id}_{method}_{wrt_to}_{plddt_optim}_{mutate_early}_{precision}_{reference_coords_dist_map}"
    return tmp_dir


if __name__ == "__main__":
    task = sys.argv[1]
    assert task in ["stats", "seq_from_file", "get_plddt"]
    if task == "stats":
        seq = sys.argv[2]
        c = get_num_late_aa(seq)
        print(f"count_late = {c}")
        ratio = get_ratio_late_aa(seq)
        print(f"{len(seq)=}\nratio_late = {ratio}")
    elif task == "seq_from_file":
        filename = sys.argv[2]
        seq_early_only, seq = get_seq_from_structure_file(filename)
        print(seq_early_only)
        print(seq)
    elif task == "get_plddt":
        from Bio.PDB import PDBParser, MMCIFParser

        filename = sys.argv[2]
        parser = (
            PDBParser(QUIET=True)
            if filename.endswith(".pdb")
            else MMCIFParser(QUIET=True)
        )
        structure = parser.get_structure("protein", filename)
        plddt = []
        for atom in structure.get_atoms():
            if atom.get_id() == "CA":
                # pLDDT is stored in the b-factor field of the CA atom
                plddt_value = atom.bfactor
                plddt.append(plddt_value)
        plddt = np.array(plddt)
        plddt *= 100
        print(np.mean(plddt), ",")
        # print(f"Mean pLDDT: {np.mean(plddt):.2f}")
        # print(f"Std pLDDT: {np.std(plddt):.2f}")
        # print(f"Min pLDDT: {np.min(plddt):.2f}")
        # print(f"Max pLDDT: {np.max(plddt):.2f}")
