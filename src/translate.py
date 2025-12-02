import os
import sys
import argparse
import torch
import Bio
import shutil
import numpy as np
from io import StringIO

from time import time
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from transformers import EsmForProteinFolding
from transformers.models.esm.modeling_esmfold import (
    collate_dense_tensors,
    EsmForProteinFoldingOutput,
)
from transformers.models.esm.openfold_utils import residue_constants, atom14_to_atom37
from old_translation_algorithms import *


from constants import (
    ALL_AA,
    EARLY_AA,
    IX_TO_LAA,
    LAA_TO_IX,
    LATE_AA,
    THREE_TO_ONE_AA,
    DB_PATH,
)
from utils import *
from kabsch import kabsch_torch
from itertools import product
import random
from uuid import uuid4


def run_clustered_translation(
    model: EsmForProteinFolding,
    seq: str,
    args: argparse.Namespace,
) -> tuple[str, float]:
    """
    Run a clustered translation of the protein sequence.
    First, get clusters of late amino acids based on the distance threshold.
    Then for each cluster run beam search with beam of size `beam_size`.
    Args:
        model (EsmForProteinFolding): The ESMFold model.
        seq (str): The input protein sequence.
        distance_threshold (float): Distance threshold for selecting nearby residues (unit: angstrom).
    Returns:
        tuple: A tuple containing the final sequence and the final RMSD.
    """
    design = Design(
        design_id=str(uuid4()),
        reference_structure=(
            os.path.abspath(args.pdb_path) if args.pdb_path else "ESMFold_prediction"
        ),
        optimization_reference="pdb" if args.pdb_path else "esm",
        beam_size=args.beam_size,
        optimize_plddt=args.optimize_plddt,
        plddt_scaling_factor=args.plddt_scaling_factor,
        distance_threshold=args.distance_threshold,
        clustering_reference=args.dmap_reference,
        clustering_proportion=args.clustering_proportion,
        precision="fp32" if args.fp32 else "bf16",
        mutate_early=False,
        random_seed=args.current_seed,
    )
    orig_seq = seq
    if args.pdb_path is None:
        orig_struct = get_initinal_structure(orig_seq, model)
    else:
        parser = MMCIFParser(QUIET=True)
        orig_struct = parser.get_structure("original", args.pdb_path)
        print("Loaded original structure from PDB file.")
        _seq = "".join(
            [
                THREE_TO_ONE_AA.get(residue.get_resname(), "")
                for residue in orig_struct.get_residues()
                if residue.get_full_id()[1] == 0 and residue.get_full_id()[2] == "A"
            ]
        )
        if _seq != orig_seq:
            # raise ValueError(
            #     f"Sequence from PDB file ({_seq}) does not match the input sequence ({orig_seq})."
            # )
            print(
                f"Warning: Sequence from PDB file ({_seq}) does not match the input sequence ({orig_seq})."
            )
        else:
            print("Sequences match.")
        orig_seq = _seq
        seq = _seq

    orig_coords_ca = torch.tensor(
        np.array(
            [
                atom.coord
                for atom in orig_struct.get_atoms()
                if atom.get_id() == "CA"
                and atom.get_full_id()[1] == 0
                and atom.get_full_id()[2] == "A"
            ]
        )
    )  # (L, 3) - C-alpha for all residues

    orig_coords_cb = torch.tensor(
        np.array(
            [
                atom.coord
                for atom in orig_struct.get_atoms()
                if (atom.get_full_id()[1] == 0 and atom.get_full_id()[2] == "A")
                and (
                    (atom.get_name() == "CB")
                    or (atom.get_name() == "CA" and atom.parent.get_resname() == "GLY")
                )
            ]
        )
    )  # (L, 3) - C-beta for all residues except Glycine, which uses C-alpha instead
    assert (
        orig_coords_ca.shape[0] == orig_coords_cb.shape[0]
    ), "C-alpha and C-beta coordinates must have the same number of residues."

    orig_coords_com = torch.tensor(
        np.array(
            [
                res.center_of_mass()
                for res in orig_struct.get_residues()
                if (res.get_resname() in THREE_TO_ONE_AA)
                and res.parent.get_id() == "A"
                and res.get_full_id()[1] == 0
            ]
        )
    )  # (L, 3) # Center of mass for all residues
    assert (
        orig_coords_ca.shape[0] == orig_coords_com.shape[0]
    ), "C-alpha and center of mass coordinates must have the same number of residues."

    ref_coords = orig_coords_cb if args.dmap_reference == "cb" else orig_coords_com
    dist_map = (ref_coords.unsqueeze(0) - ref_coords.unsqueeze(1)).norm(
        p=2, dim=-1
    )  # (L, L, 3) -> (L, L)

    clusters = boolean_clustering(
        (dist_map <= args.distance_threshold),
        seq=seq,
        linkage="partial",
        proportion=args.clustering_proportion,
    )

    print(len(clusters))
    np.random.shuffle(clusters)
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1} / {len(clusters)}")
        print(cluster)
        start_cluster = time()
        beam = [(seq, None)]  # (sequence, RMSD)

        for _ in range(len(cluster)):
            candidates = []
            for candidate_seq, _ in beam:
                possible_mutation_ixs = [
                    ix for ix in cluster if candidate_seq[ix] not in EARLY_AA
                ]
                for eaa in EARLY_AA:
                    for mutation_ix in possible_mutation_ixs:
                        candidates.append(
                            (
                                candidate_seq[:mutation_ix]
                                + eaa
                                + candidate_seq[mutation_ix + 1 :],
                                None,
                            )
                        )

            print(f"Processing {len(candidates)} candidates...")
            outputs_list = []
            for j in range(0, len(candidates) - 1, args.batch_size):
                print(
                    f"Processing batch {j // args.batch_size + 1} / {len(candidates) // args.batch_size + 1}",
                    end="\r",
                )
                batch = [c[0] for c in candidates[j : j + args.batch_size]]
                outputs_batch = infer(model, batch, num_recycles=0)
                outputs_list.append(outputs_batch)
            print(" " * 100, end="\r")

            pred_coords = torch.cat(
                [
                    atom14_to_atom37(out["positions"][-1], out).detach().cpu()
                    for out in outputs_list
                ]
            )[
                ..., 1, :
            ]  # B x L x 3
            plddts = torch.cat(
                [out["mean_plddt"] for out in outputs_list], dim=0
            )  # B x L
            for j, (_seq, _) in enumerate(candidates):
                _, _, rmsd_val = kabsch_torch(orig_coords_ca, pred_coords[j])
                if args.optimize_plddt:
                    score = rmsd_val.item() - (plddts[j].item() * args.plddt_scaling_factor)
                else:
                    score = rmsd_val.item()
                candidates[j] = (_seq, score)

            beam = sorted(candidates, key=lambda x: x[1])[: args.beam_size]

        best_seq, _ = min(beam, key=lambda x: x[1])
        out_best = infer(model, best_seq)
        _, _, best_rmsd = kabsch_torch(
            orig_coords_ca,
            atom14_to_atom37(out_best["positions"][-1], out_best)
            .detach()
            .cpu()[0, :, 1],
        )
        out_best_str = output_to_pdb(out_best, 0)
        # with open(os.path.join(temp_dir, f"{i}.pdb"), "w") as f:
        #     f.write(out_best_str)

        # if i == len(clusters) - 1:
        #     shutil.copyfile(
        #         os.path.join(temp_dir, f"{i}.pdb"), os.path.join(temp_dir, "final.pdb")
        #     )

        print(f"Elapsed time: {time() - start_cluster:.02f}")
        print(f"pLDDT after step {i+1}: {out_best['mean_plddt'][0].item()}")
        print(f"RMSD after step {i+1}: {best_rmsd}")
        seq = best_seq

    print(orig_seq)
    print("".join(("|" if a == b else " ") for a, b in zip(orig_seq, seq)))
    print(seq)
    print(f"pLDDT final: {out_best['mean_plddt'][0].item()}")
    print(f"Final RMSD: {best_rmsd}")

    design.sequence = seq
    design.pdb = out_best_str
    design.rmsd = best_rmsd.item()
    design.plddt = out_best["mean_plddt"][0].item()
    try:
        if args.pdb_path:
            design.tm_score = compute_tm_score(out_best_str, args.pdb_path)
        else:
            design.tm_score = 0.0
    except Exception:
        design.tm_score = 0.0

    try:
        design.save()
    except Exception as e:
        print(f"Failed to save design: {e}")

    return seq, best_rmsd


def run_clustered_translation_change_early(
    model: EsmForProteinFolding,
    seq: str,
    distance_threshold: float = 5,
    beam_size: int = 5,
    batch_size: int = 100,
    cluster_proportion: float = 0.5,
    pdb_path: str | None = None,
) -> tuple[str, float]:
    """
    Run a clustered translation of the protein sequence.
    First, get clusters of late amino acids based on the distance threshold.
    Then for each cluster run beam search with beam of size `beam_size`.
    Args:
        model (EsmForProteinFolding): The ESMFold model.
        seq (str): The input protein sequence.
        distance_threshold (float): Distance threshold for selecting nearby residues (unit: angstrom).
    Returns:
        tuple: A tuple containing the final sequence and the final RMSD.
    """
    design = Design(
        design_id=str(uuid4()),
        reference_structure=(
            os.path.abspath(args.pdb_path) if args.pdb_path else "ESMFold_prediction"
        ),
        optimization_reference="pdb" if args.pdb_path else "esm",
        beam_size=args.beam_size,
        optimize_plddt=args.optimize_plddt,
        plddt_scaling_factor=args.plddt_scaling_factor,
        distance_threshold=args.distance_threshold,
        clustering_reference=args.dmap_reference,
        clustering_proportion=args.clustering_proportion,
        precision="fp32" if args.fp32 else "bf16",
        mutate_early=True,
        random_seed=args.current_seed,
    )
    orig_seq = seq
    if pdb_path is None:
        orig_struct = get_initinal_structure(orig_seq, model)
    else:
        parser = MMCIFParser(QUIET=True)
        orig_struct = parser.get_structure("original", pdb_path)
        print("Loaded original structure from PDB file.")
        _seq = "".join(
            [
                THREE_TO_ONE_AA.get(residue.get_resname(), "")
                for residue in orig_struct.get_residues()
                if residue.get_full_id()[1] == 0 and residue.get_full_id()[2] == "A"
            ]
        )
        if _seq != orig_seq:
            raise ValueError(
                f"Sequence from PDB file ({_seq}) does not match the input sequence ({orig_seq})."
            )
    orig_coords_ca = torch.tensor(
        np.array(
            [
                atom.coord
                for atom in orig_struct.get_atoms()
                if atom.get_id() == "CA"
                and atom.get_full_id()[1] == 0
                and atom.get_full_id()[2] == "A"
            ]
        )
    )  # (L, 3) - C-alpha for all residues

    orig_coords_cb = torch.tensor(
        np.array(
            [
                atom.coord
                for atom in orig_struct.get_atoms()
                if (atom.get_full_id()[1] == 0 and atom.get_full_id()[2] == "A")
                and (
                    (atom.get_name() == "CB")
                    or (atom.get_name() == "CA" and atom.parent.get_resname() == "GLY")
                )
            ]
        )
    )  # (L, 3) - C-beta for all residues except Glycine, which uses C-alpha instead
    assert (
        orig_coords_ca.shape[0] == orig_coords_cb.shape[0]
    ), "C-alpha and C-beta coordinates must have the same number of residues."

    orig_coords_com = torch.tensor(
        np.array(
            [
                res.center_of_mass()
                for res in orig_struct.get_residues()
                if (res.get_resname() in THREE_TO_ONE_AA)
                and res.parent.get_id() == "A"
                and res.get_full_id()[1] == 0
            ]
        )
    )  # (L, 3) # Center of mass for all residues
    assert (
        orig_coords_ca.shape[0] == orig_coords_com.shape[0]
    ), "C-alpha and center of mass coordinates must have the same number of residues."

    ref_coords = orig_coords_cb if args.dmap_reference == "cb" else orig_coords_com
    dist_map = (ref_coords.unsqueeze(0) - ref_coords.unsqueeze(1)).norm(
        p=2, dim=-1
    )  # (L, L, 3) -> (L, L)
    clusters = boolean_clustering(
        (dist_map <= distance_threshold),
        seq=None,
        linkage="partial",
        proportion=cluster_proportion,
    )
    print(len(clusters))
    np.random.shuffle(clusters)
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1} / {len(clusters)}")
        print(cluster)
        start_cluster = time()
        beam = [(seq, None, cluster.copy())]  # (sequence, RMSD, possible_mutation_ixs)

        for _ in range(len(cluster)):
            candidates = []
            for candidate_seq, _, possible_mutation_ixs in beam:
                for eaa in EARLY_AA:
                    for mutation_ix in possible_mutation_ixs:
                        next_mutation_ixs = possible_mutation_ixs.copy()
                        next_mutation_ixs.remove(mutation_ix)
                        candidates.append(
                            (
                                candidate_seq[:mutation_ix]
                                + eaa
                                + candidate_seq[mutation_ix + 1 :],
                                None,
                                next_mutation_ixs,
                            )
                        )

            print(f"Processing {len(candidates)} candidates...")
            outputs_list = []
            for j in range(0, len(candidates) - 1, batch_size):
                print(
                    f"Processing batch {j // batch_size + 1} / {len(candidates) // batch_size + 1}",
                    end="\r",
                )
                batch = [c[0] for c in candidates[j : j + batch_size]]
                outputs_batch = infer(model, batch, num_recycles=0)
                outputs_list.append(outputs_batch)
            print(" " * 100, end="\r")

            pred_coords = torch.cat(
                [
                    atom14_to_atom37(out["positions"][-1], out).detach().cpu()
                    for out in outputs_list
                ]
            )[
                ..., 1, :
            ]  # B x L x 3
            plddts = torch.cat(
                [out["mean_plddt"] for out in outputs_list], dim=0
            )  # B x L
            for j, (_seq, _, possible_mutation_ixs) in enumerate(candidates):
                _, _, rmsd_val = kabsch_torch(orig_coords_ca, pred_coords[j])
                if args.optimize_plddt:
                    score = rmsd_val.item() - (plddts[j].item() * args.plddt_scaling_factor)
                else:
                    score = rmsd_val.item()
                candidates[j] = (_seq, score, possible_mutation_ixs)

            beam = sorted(candidates, key=lambda x: x[1])[:beam_size]

        best_seq, _, _ = min(beam, key=lambda x: x[1])
        out_best = infer(model, best_seq)
        _, _, best_rmsd = kabsch_torch(
            orig_coords_ca,
            atom14_to_atom37(out_best["positions"][-1], out_best)
            .detach()
            .cpu()[0, :, 1],
        )
        out_best_str = output_to_pdb(out_best, 0)
        # with open(os.path.join(temp_dir, f"{i}.pdb"), "w") as f:
        #     f.write(out_best_str)

        # if i == len(clusters) - 1:
        #     shutil.copyfile(
        #         os.path.join(temp_dir, f"{i}.pdb"), os.path.join(temp_dir, "final.pdb")
        #     )

        print(f"Elapsed time: {time() - start_cluster:.02f}")
        print(f"pLDDT after step {i+1}: {out_best['mean_plddt'][0].item()}")
        print(f"RMSD after step {i+1}: {best_rmsd}")
        seq = best_seq

    print(orig_seq)
    print("".join(("|" if a == b else " ") for a, b in zip(orig_seq, seq)))
    print(seq)
    print(f"pLDDT final: {out_best['mean_plddt'][0].item()}")
    print(f"Final RMSD: {best_rmsd}")

    design.sequence = seq
    design.pdb = out_best_str
    design.rmsd = best_rmsd.item()
    design.plddt = out_best["mean_plddt"][0].item()
    try:
        if args.pdb_path:
            design.tm_score = compute_tm_score(out_best_str, args.pdb_path)
        else:
            design.tm_score = 0.0
    except Exception:
        design.tm_score = 0.0

    try:
        design.save()
    except Exception as e:
        print(f"Failed to save design: {e}")

    return seq, best_rmsd


def main(args):
    orig_seq = str(SeqIO.read(args.input_fasta, "fasta").seq)
    model = (
        EsmForProteinFolding.from_pretrained(args.esmfold_model_path)
        .eval()
        .to(args.device)
        .to(torch.float32 if args.fp32 else torch.bfloat16)
    )
    print(model.dtype)

    assert (
        sum([args.greedy, args.cluster_beam, args.distance_based]) == 1
    ), "Only one translation method can be used at a time."

    print(f"Using device: {args.device}")
    print("Model loaded successfully.")

    for i in range(args.translations):
        print(f"Running translation {i+ 1} / {args.translations}")
        start_translation = time()
        np.random.seed(args.random_seed + i)
        args.current_seed = args.random_seed + i
        if args.greedy:
            if args.wrt_pdb:
                pdb_path = args.input_fasta.replace("_trimmed", "").replace(
                    ".fasta", ".cif"
                )
            else:
                pdb_path = None
            final_seq, final_rmsd = run_greedy_translation(
                model, orig_seq, pdb_path
            )
        elif args.cluster_beam:
            pdb_path = (
                args.input_fasta.replace("_trimmed", "").replace(".fasta", ".cif")
                if args.wrt_pdb
                else None
            )
            args.pdb_path = pdb_path
            if not args.mutate_early:
                final_seq, final_rmsd = run_clustered_translation(
                    model,
                    orig_seq,
                    args,
                )
            else:
                final_seq, final_rmsd = run_clustered_translation_change_early(
                    model,
                    orig_seq,
                    tmp_dir,
                    distance_threshold=args.distance_threshold,
                    beam_size=args.beam_size,
                    batch_size=args.batch_size,
                    cluster_proportion=args.clustering_proportion,
                    pdb_path=pdb_path,
                )
        elif args.distance_based:
            final_seq, final_rmsd = run_distance_based_translation(
                model,
                orig_seq,
                tmp_dir,
                batch_size=args.batch_size,
                distance_threshold=7,
            )
        else:
            raise ValueError(
                "Select a translation method, one of: [--greedy, --cluster_beam, --distance_based]"
            )
        print(f"Translation time: {time() - start_translation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate protein sequences using ESMFold"
    )
    parser.add_argument("--input_fasta", "-i", type=str, required=True)
    parser.add_argument("--random_seed", "-r", type=int, default=42)
    parser.add_argument("--batch_size", "-b", type=int, default=100)
    parser.add_argument("--beam_size", "-w", type=int, default=5)
    parser.add_argument("--clustering_proportion", "-c", type=float, default=0.5)
    parser.add_argument("--distance_threshold", type=int, default=7)
    parser.add_argument("--plddt_scaling_factor", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy translation.",
    )
    parser.add_argument(
        "--cluster_beam",
        action="store_true",
        help="Use clustered beam search translation.",
    )
    parser.add_argument(
        "--distance_based",
        action="store_true",
        help="Use distance-based translation.",
    )
    parser.add_argument(
        "--translations",
        "-n",
        type=int,
        default=1,
        help="How many times to run the translation",
    )
    parser.add_argument(
        "--wrt_esm",
        action="store_true",
        help="Optimize RMSD wrt ESMfold prediction, rather than initial PDB entry.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 precision for the model.",
    )
    parser.add_argument(
        "--mutate_early",
        action="store_true",
        help="Mutate all residues in clustered beam search, not only late ones, but also early.",
    )
    parser.add_argument(
        "--optimize_plddt",
        action="store_true",
        help="Optimize pLDDT as well as RMSD during translation.",
    )
    parser.add_argument(
        "--dmap_reference",
        choices=["cb", "com"],
        type=str,
        default="com",
        help="Use C-beta coordinates for contact map or center of mass.",
    )
    parser.add_argument(
        "--esmfold_model_path",
        type=str,
        default="facebook/esmfold_v1",
        help="Path to the pretrained ESMFold model. Default downloads from HF hub (or loads from local cache), but can be replaced by local path.",
    )
    args = parser.parse_args()
    args.wrt_pdb = not args.wrt_esm
    main(args)
