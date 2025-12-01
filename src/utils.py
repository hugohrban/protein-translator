import os
import sys
import re
import torch
import numpy as np
import argparse
import pandas as pd
from dataclasses import dataclass
from Bio.PDB import MMCIFParser, PDBParser
from constants import ALL_AA, EARLY_AA, LATE_AA, THREE_TO_ONE_AA, DB_PATH
from transformers.models.esm.openfold_utils import atom14_to_atom37, OFProtein, to_pdb
from time import sleep
import subprocess
from datetime import datetime


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


def compute_tm_score(reference: str, design_path: str) -> float:
    """
    Compute TM-score between two protein structures using TMscore.

    Args:
        reference (str): String content of the reference structure file or a structure file path
        design_path (str): Path to the design structure file

    Returns:
        float: TM-score between the two structures
    """

    # if the reference is a pdb file content string, write it to a temporary file
    is_temporary_reference = False
    if not os.path.exists(reference):
        is_temporary_reference = True
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("temp_refs", exist_ok=True)
        reference_path = f"temp_refs/ref_{current_timestamp}.pdb"
        with open(reference_path, "w") as f:
            f.write(reference)
    else:
        reference_path = reference

    try:
        result = subprocess.run(
            ["TMalign", design_path, reference_path],
            capture_output=True,
            text=True,
            check=True,
        )
        if is_temporary_reference and os.path.exists(reference_path):
            os.remove(reference_path)

        # Parse the output to extract TM-score
        output_lines = result.stdout.split("\n")
        for line in output_lines:
            if line.startswith("TM-score="):
                # Extract TM-score from line like "TM-score= 0.12345 (if normalized by length of Chain_1)"
                tm_score_str = line.split()[1].strip()
                return float(tm_score_str)

        raise ValueError(
            "TM-score not found in TMalign output: " + "\n".join(output_lines)
        )

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"TMalign command failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "TMalign executable not found. Please ensure TMalign is installed and in PATH."
        )
    except (ValueError, IndexError) as e:
        raise RuntimeError(f"Failed to parse TM-score from output: {e}")


# Database


@dataclass
class Design:
    # Params
    design_id: str | None = (
        None  # uuid4 - not checked for uniqueness, but should be fine
    )
    reference_structure: str | None = None  # path to reference
    optimization_reference: str | None = None  # 'pdb' or 'esm'
    beam_size: int | None = None
    optimize_plddt: bool | None = None
    plddt_scaling_factor: float | None = None
    distance_threshold: float | None = None
    clustering_reference: str | None = None  # 'cb' or 'com'
    clustering_proportion: float | None = None
    precision: str | None = None  # 'bf16' or 'fp32'
    mutate_early: bool | None = None
    random_seed: int | None = None

    # Results
    sequence: str | None = None  # translated sequence, only early AAs
    pdb: str | None = None  # pdb content string, ESMFold prediction of the sequence
    plddt: float | None = None  # plddt average across residues
    rmsd: float | None = None
    tm_score: float | None = None

    def validate(self) -> bool:
        """
        Validate that all fields are properly set and string fields have allowed values.

        Returns:
            bool: True if all validations pass, False otherwise.
        """
        condition_descriptions = [
            (
                "All fields are not None",
                all(getattr(self, field) is not None for field in self.__annotations__),
            ),
            (
                "optimization_reference in {'pdb', 'esm'}",
                self.optimization_reference in {"pdb", "esm"},
            ),
            (
                "clustering_reference in {'cb', 'com'}",
                self.clustering_reference in {"cb", "com"},
            ),
            ("precision in {'bf16', 'fp32'}", self.precision in {"bf16", "fp32"}),
            (
                "clustering_proportion between 0.0 and 1.0",
                0.0 <= self.clustering_proportion <= 1.0,
            ),
            ("distance_threshold >= 0", self.distance_threshold >= 0),
            ("plddt_scaling_factor >= 0", self.plddt_scaling_factor >= 0),
            (
                "sequence contains only EARLY_AA",
                all(aa in EARLY_AA for aa in self.sequence),
            ),
            ("plddt between 0.0 and 1.0", 0.0 <= self.plddt <= 1.0),
            ("rmsd >= 0", self.rmsd >= 0),
            ("tm_score between 0 and 1.0", 1.0 >= self.tm_score >= 0),
        ]

        failed_conditions = [
            desc for desc, condition in condition_descriptions if not condition
        ]

        if failed_conditions:
            print(f"Validation failed for design {self.design_id}:")
            for failed in failed_conditions:
                print(f"  - {failed}")
            return False

        return True

    def save(self) -> None:
        write_design_to_db(self)


def write_design_to_db(design: Design) -> None:
    """
    Design has keys: design_id, sequence, plddt, pdb
    """
    if not design.validate():
        raise ValueError("Design validation failed. Cannot write to database.")

    os.makedirs(DB_PATH, exist_ok=True)
    acquire_db_lock()
    try:
        jsonl_path = os.path.join(DB_PATH, "designs.jsonl")
        df = pd.DataFrame([design])
        df.pop(
            "pdb"
        )  # dont save the pdb contnt here, we save that into a separate file
        df.to_json(jsonl_path, orient="records", lines=True, mode="a")

        os.makedirs(os.path.join(DB_PATH, "designs"), exist_ok=True)
        pdb_design_path = os.path.join(DB_PATH, "designs", design.design_id + ".pdb")
        with open(pdb_design_path, "w") as pdb_file:
            pdb_file.write(design.pdb)

        # fasta_design_path = os.path.join(DB_PATH, "designs", design.design_id + ".fasta")
        # with open(fasta_design_path, 'w') as fasta_file:
        #     fasta_file.write(f">{design.design_id}\n{design.sequence}\n")

    finally:
        release_db_lock()


def acquire_db_lock() -> None:
    MAX_TRIES = 100
    SLEEP_TIME_SECONDS = 0.5
    lock_file = os.path.join(DB_PATH, "DB_LOCK")
    tries = 0
    while os.path.exists(lock_file):
        if tries >= MAX_TRIES:
            raise TimeoutError(
                "Could not acquire database lock after multiple attempts."
            )
        sleep(SLEEP_TIME_SECONDS)
        tries += 1
    with open(lock_file, "w") as f:
        f.write("LOCKED")


def release_db_lock() -> None:
    lock_file = os.path.join(DB_PATH, "DB_LOCK")
    if os.path.exists(lock_file):
        os.remove(lock_file)
