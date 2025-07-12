ALL_AA = [
        "A",  # Alanine
        "C",  # Cysteine
        "D",  # Aspartic acid
        "E",  # Glutamic acid
        "F",  # Phenylalanine
        "G",  # Glycine
        "H",  # Histidine
        "I",  # Isoleucine
        "K",  # Lysine
        "L",  # Leucine
        "M",  # Methionine
        "N",  # Asparagine
        "P",  # Proline
        "Q",  # Glutamine
        "R",  # Arginine
        "S",  # Serine
        "T",  # Threonine
        "V",  # Valine
        "W",  # Tryptophan
        "Y",  # Tyrosine
    ]

EARLY_AA = [
        "A",  # Alanine
        "D",  # Aspartic acid
        "E",  # Glutamic acid
        "G",  # Glycine
        "I",  # Isoleucine
        "L",  # Leucine
        "P",  # Proline
        "S",  # Serine
        "T",  # Threonine
        "V",  # Valine
    ]

# ALA+ASP+GLU+GLY+ILE+LEU+PRO+SER+THR+VAL

IX_TO_EAA = { i: aa for i, aa in enumerate(EARLY_AA) }
EAA_TO_IX = { aa: i for i, aa in enumerate(EARLY_AA) }

LATE_AA =[
        "C",  # Cysteine
        "F",  # Phenylalanine
        "H",  # Histidine
        "K",  # Lysine
        "M",  # Methionine
        "N",  # Asparagine
        "Q",  # Glutamine
        "R",  # Arginine
        "W",  # Tryptophan
        "Y",  # Tyrosine
    ]

# CYS+PHE+HIS+LYS+MET+ASN+GLN+ARG+TRP+TYR

IX_TO_LAA = { i: aa for i, aa in enumerate(LATE_AA) }
LAA_TO_IX = { aa: i for i, aa in enumerate(LATE_AA) }

ONE_TO_THREE_AA = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

THREE_TO_ONE_AA = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
