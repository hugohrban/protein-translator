import os
import sys
import argparse
import torch
import Bio
import shutil
import numpy as np

from time import time
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from transformers import EsmForProteinFolding
from transformers.models.esm.modeling_esmfold import (
    collate_dense_tensors,
    EsmForProteinFoldingOutput,
)
from transformers.models.esm.openfold_utils import residue_constants, atom14_to_atom37


from constants import ALL_AA, EARLY_AA, IX_TO_LAA, LAA_TO_IX, LATE_AA, THREE_TO_ONE_AA
from utils import *
from kabsch import kabsch_torch
from itertools import product
import random


@torch.no_grad()
def infer(
    model: EsmForProteinFolding,
    seqs: str | list[str],
    position_ids=None,
    num_recycles: int | None = None,
) -> EsmForProteinFoldingOutput:
    if isinstance(seqs, str):
        lst = [seqs]
    else:
        lst = seqs
    # Returns the raw outputs of the model given an input sequence.
    model = model.eval()
    device = next(model.parameters()).device
    aatype = collate_dense_tensors(
        [
            torch.from_numpy(
                residue_constants.sequence_to_onehot(
                    sequence=seq,
                    mapping=residue_constants.restype_order_with_x,
                    map_unknown_to_x=True,
                )
            )
            .to(device)
            .argmax(dim=1)
            for seq in lst
        ]
    )  # B=1 x L
    mask = collate_dense_tensors([aatype.new_ones(len(seq)) for seq in lst])
    position_ids = (
        torch.arange(aatype.shape[1], device=device).expand(len(lst), -1)
        if position_ids is None
        else position_ids.to(device)
    )
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)
    output = model(
        aatype,
        mask,
        position_ids=position_ids,
        num_recycles=num_recycles,
    )
    output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(
        dim=(1, 2)
    ) / output["atom37_atom_exists"].sum(dim=(1, 2))
    return output


def fold_l_n(l, n=10, steps=None):
    """
    measure time to fold n seqs of length l
    """
    start = time()
    seqs = ["G" * l] * n
    outputs = infer(model, seqs, num_recycles=0)
    end = time()
    print("Per seq", (end - start) / n)
    print("Per step", (end - start))
    print("Total translation", (end - start) * steps if steps else "N/A")


# model = EsmForProteinFolding.from_pretrained("../esmfold_v1").eval().to("cuda:0")


def get_initinal_structure(
    input_seq: str, model: EsmForProteinFolding, temp_dir: str, from_pdb: bool = False
):
    """
    Get the original structure from the input FASTA file.
    Args:
        input_fasta (str): Path to the input FASTA file.
        temp_dir (str): Temporary directory for storing intermediate files.
    Returns:
        Bio.PDB.Structure.Structure: The original protein structure.
    """
    # with torch.amp.autocast("cuda"):
    output = infer(model, input_seq, num_recycles=0)
    output_str = output_to_pdb(output, 0)
    # output = model.output_to_pdb(output)[0]
    print("Original pLDDT:", output["mean_plddt"][0].item())

    with open(os.path.join(temp_dir, "original.pdb"), "w") as f:
        print(output_str, file=f)

    parser = PDBParser(QUIET=True)
    orig_struct = parser.get_structure(
        "original", os.path.join(temp_dir, "original.pdb")
    )
    return orig_struct


def run_greedy_translation(
    model: EsmForProteinFolding, seq: str, temp_dir: str, pdb_path: str | None = None
) -> tuple[str, float]:
    orig_seq = seq
    if pdb_path is None:
        orig_struct = get_initinal_structure(orig_seq, model, temp_dir)
    else:
        parser = MMCIFParser(QUIET=True)
        orig_struct = parser.get_structure("original", pdb_path)
        print("Loaded original structure from MMCIF file.")
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
            seq = _seq
            orig_seq = _seq
    orig_coords = torch.tensor(
        np.array(
            [
                atom.coord
                for atom in orig_struct.get_atoms()
                if atom.get_id() == "CA"
                and atom.get_full_id()[1] == 0
                and atom.get_full_id()[2] == "A"
            ]
        )
    )
    mutations = {laa: {eaa: 0 for eaa in EARLY_AA} for laa in LATE_AA}
    rmsd_increases = torch.zeros((len(LATE_AA), len(EARLY_AA)), dtype=torch.float32)
    prev_struct = orig_coords
    count_late = torch.zeros(len(LATE_AA), dtype=torch.float32)

    total_steps = get_num_late_aa(orig_seq)
    i = 0
    while get_num_late_aa(seq) > 0:
        print(f"=============\nIteration {i} / {total_steps}\n=============\n")
        scores = {}  # Eearly AA -> (RMSD_v_orig, RMSD_v_prev)

        possible_mutations = [i for i, aa in enumerate(seq) if aa not in set(EARLY_AA)]
        if not possible_mutations:
            return scores, -1

        mutation_ix = np.random.choice(possible_mutations)
        mutated_seqs = []
        for eaa in EARLY_AA:
            mutated_seq = seq[:mutation_ix] + eaa + seq[mutation_ix + 1 :]
            mutated_seqs.append(mutated_seq)

        count_late[LAA_TO_IX[seq[mutation_ix]]] += 1
        start_inf = time()
        outputs = infer(model, mutated_seqs, num_recycles=0)

        rmsds_orig = torch.zeros(len(EARLY_AA), dtype=torch.float32)
        rmsds_prev = torch.zeros(len(EARLY_AA), dtype=torch.float32)
        pred_pos_atom_37 = (
            atom14_to_atom37(outputs["positions"][-1], outputs).detach().cpu()
        )  # B x L x 37 x 3
        for j, eaa in enumerate(EARLY_AA):
            _, _, rmsd_orig = kabsch_torch(
                orig_coords,
                pred_pos_atom_37[j, :, 1],  # atom 1 is C-alpha
            )
            rmsds_orig[j] = rmsd_orig

            _, _, rmsd_prev = kabsch_torch(
                prev_struct,
                pred_pos_atom_37[j, :, 1],
            )
            rmsds_prev[j] = rmsd_prev
            scores[eaa] = rmsd_orig.item(), rmsd_prev.item()

        ix = torch.argmin(rmsds_orig, dim=0).item()
        out_str = output_to_pdb(outputs, ix)
        prev_struct = pred_pos_atom_37[ix, :, 1]
        with open(os.path.join(temp_dir, f"{i}_mutated.pdb"), "w") as f:
            print(out_str, file=f)

        best_sub = EARLY_AA[ix]
        mutations[seq[mutation_ix]][best_sub] += 1
        rmsd_increases[LAA_TO_IX[seq[mutation_ix]], :] += rmsds_prev

        print(f"At position {mutation_ix} mutating {seq[mutation_ix]} to: {best_sub}")
        seq = seq[:mutation_ix] + best_sub + seq[mutation_ix + 1 :]
        print(*scores.items(), sep="\n")
        print(f"Elapsed time: {time()-start_inf:.02f}")

        i += 1

    print(orig_seq)
    print("".join(("|" if a == b else " ") for a, b in zip(orig_seq, seq)))
    print(seq)
    output_final = infer(model, seq)
    out_final_str = output_to_pdb(output_final, 0)
    with open(os.path.join(temp_dir, f"final.pdb"), "w") as f:
        print(out_final_str, file=f)
    pred_pos_atom_37 = (
        atom14_to_atom37(output_final["positions"][-1], output_final).detach().cpu()
    )  # B x L x 37 x 3
    _, _, final_rmsd = kabsch_torch(
        orig_coords,
        pred_pos_atom_37[0, :, 1],  # atom 1 is C-alpha
    )
    # final_rmsd = rmsds_orig[ix].item()
    print("Final RMSD:", final_rmsd.item())
    print("pLDDT after final step:", output_final["mean_plddt"][0].item())
    print("RMSD increases per late amino acid:")
    print(
        *round_list_of_lists(
            (rmsd_increases / (count_late + 1e-6).unsqueeze(1)).tolist()
        ),
        sep="\n",
    )

    # mutations = {laa: {eaa: 0 for eaa in EARLY_AA} for laa in LATE_AA}
    print("Mutations summary:")
    for laa, sub_dict in mutations.items():
        print(f"{laa}: {sub_dict}")

    return seq, final_rmsd


def find_nearby_late_residues(
    mutation_ix: int,
    orig_coords: torch.Tensor,
    seq: str,
    distance_threshold: float,
    num_closest: int = 3,
) -> list[int]:
    """
    Find late residues that are within a certain distance threshold from the selected late amino acid. If there are more than `num_closest`, return the top `num_closest`.
    Args:
        mutation_ix (int): The index of the selected late amino acid.
        orig_coords (torch.Tensor): The coordinates of the original structure., shape: (Length x 3)
        seq (str): The input protein sequence.
        distance_threshold (float): Distance threshold for selecting nearby residues (unit: angstrom).
        num_closest (int): The number of closest residues to return. Default: 3.
    Returns:
        list: A list of indices of nearby residues.
    """
    mutation_coord = orig_coords[mutation_ix]
    dists = torch.norm(orig_coords - mutation_coord, dim=1)
    sorted_ixs = torch.argsort(dists)
    i = 0
    nearby_late_residues = []
    while len(nearby_late_residues) < num_closest and i < orig_coords.shape[0]:
        if dists[sorted_ixs[i]] <= distance_threshold:
            if seq[sorted_ixs[i]] in LATE_AA:
                nearby_late_residues.append(sorted_ixs[i].item())
        else:
            break
        i += 1
    return nearby_late_residues


def get_mutated_seqs(late_residues: list[int], seq: str):
    # do all combinations ( num_late_res ** 10 )
    if len(late_residues) <= 3:
        mutated_seqs = []
        for eaas in product(EARLY_AA, repeat=len(late_residues)):
            mutated_seq = list(seq)
            for i, eaa in zip(late_residues, eaas):
                mutated_seq[i] = eaa
            mutated_seqs.append("".join(mutated_seq))
        return mutated_seqs
    else:
        raise NotImplementedError()


def run_distance_based_translation(
    model: EsmForProteinFolding,
    seq: str,
    temp_dir: str,
    batch_size: int = 100,
    distance_threshold: float = 5,
) -> tuple[str, float]:
    """
    In each step, randomly select a late amino acid, find all residues that are within a certain distance threshold from it,
    try out all possible early amino acid substitutions for the selected late amino acids,
    and select the one that minimizes the RMSD to the original structure.
    Args:
        model (EsmForProteinFolding): The ESMFold model.
        seq (str): The input protein sequence.
        temp_dir (str): Temporary directory for storing intermediate files.
        distance_threshold (float): Distance threshold for selecting nearby residues (unit: angstrom).
    Returns:
        tuple: A tuple containing the final sequence and the final RMSD.
    """
    orig_seq = seq
    orig_struct = get_initinal_structure(orig_seq, model, temp_dir)
    orig_coords = torch.tensor(
        np.array(
            [atom.coord for atom in orig_struct.get_atoms() if atom.get_id() == "CA"]
        )
    )
    mutations = {laa: {eaa: 0 for eaa in EARLY_AA} for laa in LATE_AA}
    rmsd_increases = torch.zeros((len(LATE_AA), len(EARLY_AA)), dtype=torch.float32)
    prev_struct = orig_coords
    # count_late = torch.zeros(len(LATE_AA), dtype=torch.float32)

    total_steps = get_num_late_aa(orig_seq)
    i = 0
    while get_num_late_aa(seq) > 0:
        print(
            f"\n=============\nRemaining late residues: {get_num_late_aa(seq)}\n============="
        )
        # scores = {}  # Early AA -> (RMSD_v_orig, RMSD_v_prev)

        possible_mutations = [i for i, aa in enumerate(seq) if aa not in set(EARLY_AA)]
        # if not possible_mutations:
        #     return scores, -1

        mutation_ix = np.random.choice(possible_mutations)

        # Find up to 3 closest late residues within the distance threshold from the selected late residue.
        nearby_late_residues = find_nearby_late_residues(
            mutation_ix, orig_coords, seq, distance_threshold
        )

        print("Num late neighbors", len(nearby_late_residues))
        mutated_seqs = get_mutated_seqs(nearby_late_residues, seq)

        # count_late[LAA_TO_IX[seq[mutation_ix]]] += 1
        start_inference = time()

        outputs_list = []
        for j in range(0, len(mutated_seqs) - 1, batch_size):
            print(
                f"Processing batch {j // batch_size + 1} / {len(mutated_seqs) // batch_size}",
                end="\r",
            )
            batch = mutated_seqs[j : j + batch_size]
            outputs_batch = infer(model, batch, num_recycles=0)
            outputs_list.append(outputs_batch)
        print(" " * 100, end="\r")  # Clear the line after processing batches
        # outputs = {}
        # if len(outputs_list) == 1:
        #     outputs = outputs_list[0]
        # else:
        #     for key in outputs_list[0]:
        #         print(key)
        #         if outputs_list[0][key].ndim >= 1:
        #             outputs[key] = torch.cat([out[key] for out in outputs_list], dim=0)
        #         else:
        #             if key not in outputs:
        #                 outputs[key] = []
        #             outputs[key].append(out[key].item() for out in outputs_list)

        rmsds_orig = torch.zeros(len(mutated_seqs), dtype=torch.float32)
        rmsds_prev = torch.zeros(len(mutated_seqs), dtype=torch.float32)
        pred_pos_atom_37 = torch.cat(
            [
                atom14_to_atom37(out["positions"][-1], out).detach().cpu()
                for out in outputs_list
            ]
        )  # B x L x 37 x 3
        start_kabsch = time()
        for j, mutated_seq in enumerate(mutated_seqs):
            _, _, rmsd_orig = kabsch_torch(
                orig_coords,
                pred_pos_atom_37[j, :, 1],  # atom 1 is C-alpha
            )
            rmsds_orig[j] = rmsd_orig

            _, _, rmsd_prev = kabsch_torch(
                prev_struct,
                pred_pos_atom_37[j, :, 1],
            )
            rmsds_prev[j] = rmsd_prev
            # scores[eaa] = rmsd_orig.item(), rmsd_prev.item()
        print(f"Kabsch time: {time() - start_kabsch:.02f}")

        ix = torch.argmin(rmsds_orig, dim=0).item()
        print(
            "RMSD_orig, RMSD_prev:",
            round(rmsds_orig[ix].item(), 3),
            round(rmsds_prev[ix].item(), 3),
        )
        out_str = output_to_pdb(outputs_list[0], ix % len(outputs_list[0]))
        prev_struct = pred_pos_atom_37[ix, :, 1]
        # with open(os.path.join(temp_dir, f"{i}_mutated.pdb"), "w") as f:
        #     print(out_str, file=f)

        # best_sub = EARLY_AA[ix]
        # mutations[seq[mutation_ix]][best_sub] += 1
        # rmsd_increases[LAA_TO_IX[seq[mutation_ix]], :] += rmsds_prev

        # print(
        #     f"At position {mutation_ix} mutating {seq[mutation_ix]} to: {best_sub}"
        # )
        # seq = seq[:mutation_ix] + best_sub + seq[mutation_ix + 1 :]
        seq = mutated_seqs[ix]
        # print(*scores.items(), sep="\n")
        print(f"Elapsed time: {time()-start_inference:.02f}")

        i += 1

    print(f"Steps: {i}")
    print(orig_seq)
    print("".join(("|" if a == b else " ") for a, b in zip(orig_seq, seq)))
    print(seq)
    print("Final RMSD:", rmsds_orig[ix].item())
    # print("RMSD increases per late amino acid:")
    # print(*round_list_of_lists((rmsd_increases / (count_late + 1e-6).unsqueeze(1)).tolist()), sep="\n")

    # # mutations = {laa: {eaa: 0 for eaa in EARLY_AA} for laa in LATE_AA}
    # print("Mutations summary:")
    # for laa, sub_dict in mutations.items():
    #     print(f"{laa}: {sub_dict}")

    return seq, 0.0


def run_clustered_translation(
    model: EsmForProteinFolding,
    seq: str,
    temp_dir: str,
    args: argparse.Namespace,
) -> tuple[str, float]:
    """
    Run a clustered translation of the protein sequence.
    First, get clusters of late amino acids based on the distance threshold.
    Then for each cluster run beam search with beam of size `beam_size`.
    Args:
        model (EsmForProteinFolding): The ESMFold model.
        seq (str): The input protein sequence.
        temp_dir (str): Temporary directory for storing intermediate files.
        distance_threshold (float): Distance threshold for selecting nearby residues (unit: angstrom).
    Returns:
        tuple: A tuple containing the final sequence and the final RMSD.
    """
    orig_seq = seq
    if args.pdb_path is None:
        orig_struct = get_initinal_structure(orig_seq, model, temp_dir)
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

    # orig_coords_cb = torch.tensor(
    #     np.array(
    #         [atom.coord for atom in orig_struct.get_atoms() if (atom.get_full_id()[1] == 0 and atom.get_full_id()[2] == "A") and ((atom.get_name() == "CB") or (atom.get_name() == "CA" and atom.parent.get_resname() == "GLY"))]
    #     )
    # )   # (L, 3) - C-beta for all residues except Glycine, which uses C-alpha instead
    # assert orig_coords_ca.shape[0] == orig_coords_cb.shape[0], "C-alpha and C-beta coordinates must have the same number of residues."

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
            # score: rmsd - (max(plddt - 50, 0) / 10)
            for j, (_seq, _) in enumerate(candidates):
                _, _, rmsd_val = kabsch_torch(orig_coords_ca, pred_coords[j])
                if args.optimize_plddt:
                    score = rmsd_val.item() - (
                        (plddts[j].item() - 0.5) * args.plddt_scaling_factor
                    )
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
        with open(os.path.join(temp_dir, f"{i}.pdb"), "w") as f:
            f.write(out_best_str)

        if i == len(clusters) - 1:
            shutil.copyfile(
                os.path.join(temp_dir, f"{i}.pdb"), os.path.join(temp_dir, "final.pdb")
            )

        print(f"Elapsed time: {time() - start_cluster:.02f}")
        print(f"pLDDT after step {i+1}: {out_best['mean_plddt'][0].item()}")
        print(f"RMSD after step {i+1}: {best_rmsd}")
        seq = best_seq

    print(orig_seq)
    print("".join(("|" if a == b else " ") for a, b in zip(orig_seq, seq)))
    print(seq)
    print(f"pLDDT final: {out_best['mean_plddt'][0].item()}")
    print(f"Final RMSD: {best_rmsd}")
    return seq, best_rmsd


def run_clustered_translation_change_early(
    model: EsmForProteinFolding,
    seq: str,
    temp_dir: str,
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
        temp_dir (str): Temporary directory for storing intermediate files.
        distance_threshold (float): Distance threshold for selecting nearby residues (unit: angstrom).
    Returns:
        tuple: A tuple containing the final sequence and the final RMSD.
    """
    orig_seq = seq
    if pdb_path is None:
        orig_struct = get_initinal_structure(orig_seq, model, temp_dir)
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
                if (atom.get_name() == "CB" and atom.parent.get_resname() != "GLY")
                or (atom.get_name() == "CA" and atom.parent.get_resname() == "GLY")
            ]
        )
    )  # (L, 3) - C-beta for all residues except Glycine, which uses C-alpha instead
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
    dist_map = (orig_coords_cb.unsqueeze(0) - orig_coords_cb.unsqueeze(1)).norm(
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
            for j, (_seq, _, possible_mutation_ixs) in enumerate(candidates):
                _, _, rmsd_val = kabsch_torch(orig_coords_ca, pred_coords[j])
                candidates[j] = (_seq, rmsd_val.item(), possible_mutation_ixs)

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
        with open(os.path.join(temp_dir, f"{i}.pdb"), "w") as f:
            f.write(out_best_str)

        if i == len(clusters) - 1:
            shutil.copyfile(
                os.path.join(temp_dir, f"{i}.pdb"), os.path.join(temp_dir, "final.pdb")
            )

        print(f"Elapsed time: {time() - start_cluster:.02f}")
        print(f"pLDDT after step {i+1}: {out_best['mean_plddt'][0].item()}")
        print(f"RMSD after step {i+1}: {best_rmsd}")
        seq = best_seq

    print(orig_seq)
    print("".join(("|" if a == b else " ") for a, b in zip(orig_seq, seq)))
    print(seq)
    print(f"pLDDT final: {out_best['mean_plddt'][0].item()}")
    print(f"Final RMSD: {best_rmsd}")
    return seq, best_rmsd


def main(args):
    orig_seq = str(SeqIO.read(args.input_fasta, "fasta").seq)
    model = (
        EsmForProteinFolding.from_pretrained("../esmfold_v1")
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

    if args.temp_dir is None:
        args.temp_dir = get_tmp_dir_name(args)

    for i in range(args.translations):
        tmp_dir = args.temp_dir + f"/{args.random_seed + i}"
        print(f"Running translation {i+ 1} / {args.translations}")
        start_translation = time()
        np.random.seed(args.random_seed + i)
        os.makedirs(tmp_dir, exist_ok=True)
        if args.greedy:
            if args.wrt_pdb:
                pdb_path = args.input_fasta.replace("_trimmed", "").replace(
                    ".fasta", ".cif"
                )
            else:
                pdb_path = None
            final_seq, final_rmsd = run_greedy_translation(
                model, orig_seq, tmp_dir, pdb_path
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
                    tmp_dir,
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
    # os.remove(args.temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate protein sequences using ESMFold"
    )
    parser.add_argument(
        "--input_fasta",
        "-i",
        type=str,
        required=True,
        help="Path to the input FASTA file with protein sequences.",
    )
    parser.add_argument(
        "--temp_dir",
        "-t",
        type=str,
        default=None,
        help="Temporary directory for storing intermediate files. Default: None (will create a temporary directory with hyperparameters stored in the path name).",
    )
    parser.add_argument("--random_seed", "-r", type=int, default=42)
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=100,
        help="Batch size for inference. Lower values reduce GPU memory usage. Default: 100.",
    )
    parser.add_argument(
        "--beam_size",
        "-w",
        type=int,
        default=5,
        help="Beam size for clustered beam search translation. Default: 5.",
    )
    parser.add_argument(
        "--clustering_proportion",
        "-c",
        type=float,
        default=0.5,
        help="During hierarchical clustering, merge two clusters when there is at least this ratio of links between them. Default: 0.5.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=int,
        default=7,
        help="Distance threshold for clustering late amino acids (in Angstroms). Default: 7.",
    )
    parser.add_argument(
        "--plddt_scaling_factor",
        type=int,
        default=5,
        help="Weight of pLDDT in the final score. Only valid when `--optimize_plddt` is set. Default: 5.",
    )
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
        "--wrt_pdb",
        action="store_true",
        help="Optimize RMSD wrt PDB entry, rather than initial ESMfold prediction.",
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
    args = parser.parse_args()
    main(args)
