import sys
import torch
import numpy as np


def kabsch_torch(
    P: torch.Tensor, Q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"
    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)
    # Optimal translation
    t = centroid_Q - centroid_P

    # Compute the covariance matrix
    H = torch.matmul((P - centroid_P).T, (Q - centroid_Q)) / P.shape[0]
    # SVD
    U, S, Vt = torch.linalg.svd(H)
    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.T, U.T))
    S = torch.eye(P.shape[1], device=P.device)
    S[-1, -1] = d.item()  # Ensure the last diagonal element is d
    # Optimal rotation
    R = U @ S @ Vt
    t = centroid_P - R @ centroid_Q
    # RMSD
    rmsd = (((P - t) @ R) - Q).square().sum(1).mean(0).sqrt()
    return R, t, rmsd


def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape
    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])
    R = U @ S @ VT
    t = EA - R @ EB
    return R, t


if __name__ == "__main__":
    from Bio.PDB import MMCIFParser, PDBParser

    if sys.argv[1].lower().endswith((".cif", ".mmcif")):
        parser1 = MMCIFParser(QUIET=True)
    else:
        parser1 = PDBParser(QUIET=True)

    if sys.argv[2].lower().endswith((".cif", ".mmcif")):
        parser2 = MMCIFParser(QUIET=True)
    else:
        parser2 = PDBParser(QUIET=True)
    struct1 = parser1.get_structure("1", sys.argv[1])
    struct2 = parser2.get_structure("2", sys.argv[2])
    coords1 = []
    coords2 = []
    for atom in struct1.get_atoms():
        if (
            atom.get_id() == "CA"
            and atom.get_full_id()[1] == 0
            and atom.get_full_id()[2] == "A"
        ):
            coords1.append(atom.coord)
    for atom in struct2.get_atoms():
        if (
            atom.get_id() == "CA"
            and atom.get_full_id()[1] == 0
            and atom.get_full_id()[2] == "A"
        ):
            coords2.append(atom.coord)
    coords1 = torch.tensor(np.array(coords1), dtype=torch.float32)
    coords2 = torch.tensor(np.array(coords2), dtype=torch.float32)
    R, t, rmsd = kabsch_torch(coords1, coords2)
    print(f"{rmsd.item()},")
