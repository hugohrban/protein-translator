import os
import sys
import argparse
import torch
import shutil
import numpy as np
from io import StringIO

from time import time
from transformers import EsmForProteinFolding
from transformers.models.esm.openfold_utils import residue_constants, atom14_to_atom37


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


def run_greedy_translation(
    model: EsmForProteinFolding, seq: str, temp_dir: str, pdb_path: str | None = None
) -> tuple[str, float]:
    raise NotImplementedError(
        "Greedy translation is deprecated. Please use other methods. You can uncomment this error - It works but there is no logging."
    )
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
    raise NotImplementedError(
        "Distance-based translation is deprecated. Please use other methods. You can uncomment this error - It works but there is no logging."
    )
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
