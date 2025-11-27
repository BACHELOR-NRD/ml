import hashlib
from typing import List

from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter


def compute_ensemble_prestep_hash(
    adapters: List[BaseDetectionAdapter],
    train_folds: List[List[tuple]],
    val_folds: List[List[tuple]],
    digest_size: int = 8,
) -> str:
    """
    Compute a combined Blake2b hash of adapters and dataset folds.

    Parameters:
    - adapters: list of adapter objects that have a .hash() method returning a hashable value
    - train_folds: list of folds; each fold is a list of tuples with dataset paths
    - val_folds: list like train_folds for validation datasets
    - digest_size: size of the Blake2b hash output in bytes (default 8 bytes)

    Returns:
    - The hexadecimal digest string of the combined hash
    """

    # Collect adapter hashes
    adapter_hashes = [adapter.hash() for adapter in adapters]

    # Collect string representations of second element in each dataset tuple in folds
    train_fold_strs = [str(ds[1]) for fold in train_folds for ds in fold]
    val_fold_strs = [str(ds[1]) for fold in val_folds for ds in fold]

    fold_hash = int(
        hashlib.sha256(
            str(tuple(train_fold_strs + val_fold_strs)).encode()
        ).hexdigest(),
        16,
    )

    # Combine all parts into one tuple and convert to string
    combined_data = tuple(adapter_hashes + [fold_hash])
    combined_str = str(combined_data).encode()

    # Compute and return blake2b hash hex digest
    return hashlib.blake2b(combined_str, digest_size=digest_size).hexdigest()
