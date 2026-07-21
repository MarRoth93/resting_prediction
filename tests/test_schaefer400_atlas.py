import numpy as np

from src.data.prepare_schaefer400_atlas import rasterize_surface_labels_wta
from src.data.schaefer400 import validate_schaefer400_atlas


def test_surface_wta_rasterizes_all_canonical_parcels():
    labels = np.arange(1, 401, dtype=np.int16)
    coordinates = np.column_stack(
        [
            np.arange(1, 401, dtype=np.float32),
            np.ones(400, dtype=np.float32),
            np.ones(400, dtype=np.float32),
        ]
    )
    atlas = rasterize_surface_labels_wta([(labels, coordinates)], (400, 1, 1))
    counts = validate_schaefer400_atlas(atlas)
    np.testing.assert_array_equal(atlas[:, 0, 0], labels)
    np.testing.assert_array_equal(counts, 1)
