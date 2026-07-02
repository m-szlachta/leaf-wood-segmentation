import numpy as np
from sklearn.neighbors import NearestNeighbors


def smooth_labels(points, wood_mask, k=10, iters=2):
    """
    KNN majority-vote smoothing of a binary wood/leaf label mask.

    Removes isolated speckle: wood points embedded in leaf (e.g. false wood in the
    crown) and leaf points embedded in wood (e.g. stray leaf on the trunk) get
    flipped to match their local neighbourhood. The k nearest neighbours are computed
    once on the (static) point coordinates and the vote is iterated `iters` times.

    Parameters
    ----------
    points : array
        (m x 3) point coordinates.
    wood_mask : array
        Boolean mask over `points`, True where wood.
    k : int
        Number of nearest neighbours used for the vote.
    iters : int
        Number of smoothing passes.

    Returns
    -------
    mask : array
        Smoothed boolean wood mask.
    """
    mask = np.asarray(wood_mask, dtype=bool).copy()

    # Search k+1 so the point itself (distance 0) is included and gets a vote,
    # which stabilises the result and avoids flip-flopping.
    n_neighbors = min(k + 1, points.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean',
                            leaf_size=15, n_jobs=-1).fit(points)
    _, idx = nbrs.kneighbors(points)

    for _ in range(iters):
        wood_votes = mask[idx].sum(axis=1)
        mask = wood_votes > (n_neighbors / 2.0)

    return mask


def fill_trunk(points, wood_mask, axis_xy, trunk_radius, radius_factor=2.0,
               spread_factor=3.0, slice_height=0.5, min_slice_pts=5):
    """
    Force wood for points near the vertical trunk axis below the crown base.

    The lower trunk is bare, so any point close to the trunk axis and below the
    height where the crown opens out cannot physically be a leaf. This flips such
    points to wood, removing the leaf-on-trunk artefact.

    Parameters
    ----------
    points : array
        (m x 3) point coordinates. Assumes the trunk grows parallel to +Z.
    wood_mask : array
        Boolean mask over `points`, True where wood.
    axis_xy : array
        (2,) X,Y location of the trunk axis (the fitted root centre).
    trunk_radius : float
        Radius of the circle fitted to the trunk base.
    radius_factor : float
        Points within `radius_factor * trunk_radius` of the axis are in the trunk
        zone (a margin over the base radius to allow taper and noise).
    spread_factor : float
        The crown base is the lowest height slice whose 95th-percentile radial
        spread exceeds `spread_factor * trunk_radius`.
    slice_height : float
        Height (Z) of each slice used to locate the crown base, in the same units
        as `points`.
    min_slice_pts : int
        Slices with fewer points than this are skipped when locating the crown base.

    Returns
    -------
    mask : array
        Updated boolean wood mask.
    crown_base_z : float
        Detected crown-base height (absolute Z), for logging/inspection.
    """
    mask = np.asarray(wood_mask, dtype=bool).copy()
    axis_xy = np.asarray(axis_xy, dtype=float)

    z = points[:, 2]
    z_min = z.min()
    z_max = z.max()
    horiz = np.linalg.norm(points[:, :2] - axis_xy, axis=1)

    # Locate the crown base: scan slices bottom-up, stop where the radial spread
    # opens out well beyond the trunk. Fallback = z_max (fill nothing) if it never
    # opens, so the step can only ever help, never over-fill a whole crown.
    crown_base_z = z_max
    n_slices = int(np.ceil((z_max - z_min) / slice_height))
    for s in range(n_slices):
        lo = z_min + s * slice_height
        in_slice = (z >= lo) & (z < lo + slice_height)
        if in_slice.sum() < min_slice_pts:
            continue
        spread = np.percentile(horiz[in_slice], 95)
        if spread > spread_factor * trunk_radius:
            crown_base_z = lo
            break

    trunk_zone = (z < crown_base_z) & (horiz < radius_factor * trunk_radius)
    mask[trunk_zone] = True

    return mask, crown_base_z
