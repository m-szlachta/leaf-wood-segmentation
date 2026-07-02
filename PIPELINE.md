# Graph-Based Wood / Leaf Separation — Pipeline Documentation

This document explains, end to end, how the single-tree **wood/leaf separation**
pipeline in this repository works: every stage, the functions involved, how they
work internally, and which parameters you can tune to improve segmentation
quality.

The method implements *"Graph-based Leaf–Wood Separation Method for Individual
Trees Using Terrestrial Lidar Point Clouds"* (Tian Zhilin, 2022), built on top of
the TLSeparation graph machinery (Matheus Boni Vicari).

> **Goal of this project:** take a single-tree LiDAR point cloud (LAS/LAZ),
> segment it into **wood** and **leaf** points, and write the labelled result
> back out as a coloured `.las`.

---

## 1. High-level overview

The whole pipeline lives in a **single script**:

| Script | Role |
|--------|------|
| `GBSeparation/GBSeparation.py` | **Main pipeline.** Reads the `.laz`/`.las` directly with `laspy`, shifts it to a local origin, runs the full separation, and writes a labelled/coloured `.las`. |

The core idea of the algorithm:

1. **Build a graph** over all points, rooted at the tree base.
2. **Compute shortest paths** from every point back to the root (Dijkstra). Wood
   forms the "skeleton" that all paths flow along, so path geometry is highly
   informative.
3. **Cut the graph** into small clusters using distance + direction rules, at
   **multiple scales**.
4. **Classify each cluster** by geometric shape — cylindrical or linear clusters
   are wood; blobby clusters are leaves.
5. **Grow the wood** back out from those confident wood seeds along the paths.
6. **Post-process** the labels: KNN majority-vote smoothing to remove isolated
   speckle, then a trunk-fill step to force wood on the bare lower trunk.
7. **Label, colour, and export.**

```
LAS/LAZ ──read + offset shift (laspy, inline)──▶ points array
                              │
                              ▼
                          root fitting
                              │
                              ▼
                   graph construction (kNN)
                              │
                              ▼
              shortest-path info (Dijkstra)
                              │
                              ▼
        initial wood extraction (multi-scale shape classify)
                              │
                              ▼
             final wood extraction (region growing)
                              │
                              ▼
      post-processing (KNN label smoothing → trunk fill)
                              │
                              ▼
               label → colour → write .las
```

---

## 2. Stage 0 — Read + local-origin shift (inline in `GBSeparation.py`)

```python
INPUT_PATH  = 'data/tree_v2.laz'
OUTPUT_PATH = 'data/segmented_tree_v2.las'

las = laspy.read(INPUT_PATH)
points = np.vstack((las.x, las.y, las.z)).T
offset = points.min(axis=0)          # local origin
points_local = points - offset       # shift so values are small

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_local)
pcd = np.asarray(pcd.points)         # (N, 3) numpy array used everywhere below
```

The LAS/LAZ file is read **directly** — there is no separate conversion script
and no `.pcd` intermediate. The cloud is round-tripped through Open3D only to
obtain a plain `(N, 3)` numpy array.

**Why the offset shift matters.** LAS files are usually in a national grid
(large coordinates, e.g. ~6,000,000). Working in 32-bit float space loses
sub-metre precision at those magnitudes, so subtracting a local origin
(`points.min(axis=0)`) keeps values small and preserves full precision.

> ⚠️ **The offset is *not* saved.** `offset` is computed but never written to
> disk, and the output `.las` is in the local (shifted) frame. If you need true
> georeferenced coordinates, persist `offset` yourself and add it back to X/Y/Z
> (or set the LAS header offset), see Stage 6.

> ⚠️ **Units.** The whole downstream algorithm assumes coordinates are in
> **metres**. Every distance parameter below is metric. Do not rescale the cloud.

**Parameters here:**

| Parameter | Location | Meaning | Tuning |
|-----------|----------|---------|--------|
| `INPUT_PATH` | top of `GBSeparation.py` | input `.laz`/`.las` file | — |
| `OUTPUT_PATH` | top of `GBSeparation.py` | labelled output `.las` | — |

---

## 3. Stage 1 — Read + root-point fitting (`GBSeparation.py`, `LS_circle.py`)

```python
treeHeight = np.max(pcd[:, 2]) - np.min(pcd[:, 2])

root, fit_seg, trunk_radius = getRootPt(pcd, lower_h=0.0, upper_h=0.2)
pcd = np.append(pcd, root, axis=0)
root_id = pcd.shape[0] - 1
```

**Preconditions (important):**
- The tree's growth direction must be **parallel to the Z axis** (trunk vertical).
- The cloud should be a **single tree with no large gaps**.

### `getRootPt(arr, lower_h, upper_h)` — `LS_circle.py`

Fits a synthetic **root point** at the trunk base:

1. Take all points in the height slice `[min_z + lower_h, min_z + upper_h]` — a
   ring of trunk near the ground.
2. Fit a 2-D circle to their `(x, y)` via least squares (`circleFit`).
3. Return the circle centre `(x, y)` at height `min_z + lower_h` as the root, the
   indices of the trunk-slice points, **and the fitted `trunk_radius`**.

This root becomes a single node all shortest paths converge to. Fitting a circle
centre (rather than picking a real point) gives a stable, well-centred root even
if the trunk base is noisy or partially occluded. The returned `trunk_radius` is
reused later by the trunk-fill post-processing step (Stage 5.5).

**Supporting functions:**
- `circleFit(arr)` — closed-form least-squares 2-D circle fit → centre + radius.
- `circleFitError(arr)` — fits a circle and returns the **relative RMS error**
  and radius. Used later to test how "cylindrical" a cluster is.

**Parameters to tune:**

| Parameter | Location | Effect | When to change |
|-----------|----------|--------|----------------|
| `lower_h` | `getRootPt` call, line 22 | height of the root point above the lowest point | raise if the very bottom of the cloud is noisy ground clutter |
| `upper_h` | `getRootPt` call, line 22 | top of the trunk slice used for the circle fit | choose a slice that is a clean, near-cylindrical section of trunk. Too tall → the taper/branches distort the fit; too thin → too few points |

---

## 4. Stage 2 — Graph construction (`Graph_Path.py::array_to_graph`)

```python
G = array_to_graph(pcd, root_id, kpairs=3, knn=300,
                   nbrs_threshold=treeHeight/30,
                   nbrs_threshold_step=treeHeight/60)
print(">>>connected components: ", nx.number_connected_components(G))
```

Builds a **weighted, undirected graph** where nodes are points and edge weights
are Euclidean distances between neighbouring points.

### How it works

1. **kNN search.** For every point, find its `knn` nearest neighbours
   (`sklearn.NearestNeighbors`).
2. **Seed from the root.** Connect the root to its neighbours; mark them
   processed.
3. **Region-grow the graph.** Iteratively, for each freshly added point, connect
   it to up to `kpairs+1` of its *unprocessed* neighbours (`add_nodes`). This
   "walks" outward from the root along the cloud, keeping the graph a connected
   skeleton rather than a dense mesh.
4. **Gap bridging.** When a growth front stalls (isolated points remain),
   the code falls back to a distance threshold `nbrs_threshold`. If still
   nothing connects, it **increments the threshold** by `nbrs_threshold_step`
   and retries — guaranteeing every point eventually joins the graph.
5. `add_nodes` only adds an edge if its length ≤ `graph_threshold` (default
   `inf`, i.e. no cap).

**Why it's built this way:** growing outward from the root produces a graph whose
shortest paths naturally follow the tree structure (trunk → branch → twig),
which is exactly what the later stages exploit.

### Parameters to tune — **this stage dominates quality**

| Parameter | Default | Meaning | Tuning guidance |
|-----------|---------|---------|-----------------|
| `knn` | `300` | neighbours searched per point | **Increase** for dense/high-res clouds so paths follow the true stem. Memory-intensive — larger = slower & more RAM. Decrease for sparse clouds. |
| `kpairs` | `3` | edges created per point during growth | Higher → more connected, more robust paths, more edges to process. |
| `nbrs_threshold` | `treeHeight/30` | initial gap-bridging distance | If the graph is **fragmented** (see below), increase. If distant parts wrongly merge, decrease. |
| `nbrs_threshold_step` | `treeHeight/60` | increment when bridging fails | Smaller = finer control but slower convergence; larger = bridges big gaps faster but riskier merges. |
| `graph_threshold` | `inf` | max allowed edge length | Set a finite value to reject unrealistically long edges (e.g. across gaps between separate objects). |

> ✅ **Watch the "connected components" print (line 36).** Ideally it is **1**
> (or very close). A high count means the cloud is fragmented, shortest-path
> distances are unreliable, and everything downstream degrades. Fix this first by
> raising `knn` / `nbrs_threshold`, or by cleaning gaps in the input cloud.

### Supporting functions

- `add_nodes(G, base_node, indices, distance, threshold)` — adds weighted edges
  from `base_node` to each neighbour whose distance ≤ `threshold`.

---

## 5. Stage 3 — Shortest-path information (`Graph_Path.py::extract_path_info`)

```python
path_dis, path_list = extract_path_info(G, root_id, return_path=True)
```

Runs **Dijkstra from the root** over the whole graph:

- `path_dis[node]` → accumulated shortest-path **distance** from that node back to
  the root (how far along the tree it is).
- `path_list[node]` → the **ordered list of nodes** on that shortest path
  (`[root, ..., node]`). `path_list[node][-2]` is the node's **predecessor**
  (its "parent" toward the root).

These two structures power every subsequent decision: cluster scale, local growth
direction, and region-growing containment.

No tunable parameters here — it's a deterministic consequence of the graph.

---

## 6. Stage 4 — Initial wood extraction (`ExtractInitWood.py` + `Components_classify.py`)

```python
init_wood_ids = extract_init_wood(pcd, G, root_id, path_dis, path_list,
                                  split_interval=[0.1, 0.2, 0.3, 0.5, 1],
                                  max_angle=0.25*np.pi,
                                  t_linearity=T_LINEARITY, t_error=T_ERROR,
                                  curve_threshold=CURVE_THRESHOLD)
```

This is the heart of the method. It finds **confident wood seed points** by
cutting the graph into clusters and testing each cluster's *shape*.

> **Note (important fix).** The shape thresholds are now driven by the
> `T_LINEARITY`, `T_ERROR` and `CURVE_THRESHOLD` constants defined at the top of
> `GBSeparation.py` and passed all the way through `extract_init_wood` →
> `components_classify` → `classify_info`. Previously these were **hardcoded**
> inside `ExtractInitWood.py` (`t_linearity=0.90, t_error=0.5`), which silently
> overrode the `components_classify` defaults — so editing the thresholds
> anywhere else had **no effect**. If you tune these values, do it via the
> constants in `GBSeparation.py`.

### 6.1 Precursor edge cutting (direction + distance)

For every edge `(u, v)` (except those touching the root), it computes:

- the **step distance** of `u` and `v` from their predecessors, and
- the **local direction vectors** (`pcd[u] - pcd[predecessor]`).

An edge is **cut** if either:
- `weight > 2 * min(step_u, step_v)` — the edge is much longer than the local
  spacing (a jump between structures), **or**
- `getAngle3D(dir_u, dir_v) > max_angle` — the two points' path directions
  diverge by more than `max_angle` (a bend / branching junction).

This pre-fragments the graph at natural discontinuities so that later clusters are
locally straight, coherent pieces.

### 6.2 Multi-scale segmentation

For each scale in `split_interval` (metres of path distance), points are binned
into rings by `floor(path_dis / interval)`. Within each ring, `networkx`
**connected components** become candidate clusters. Small intervals capture fine
twigs; large intervals capture the trunk and thick limbs — hence **multi-scale**.

### 6.3 Shape classification (`components_classify` → `classify_info`)

Each cluster is classified as wood or "other":

`classify_info` returns:
- `> 0` (the fitted **radius**) → **cylinder wood**,
- `< 0` (the negative **linearity**) → **linear wood**,
- `0` → **other** (leaf / non-wood).

Steps inside `classify_info`:

1. **Size filter** — clusters smaller than `max(10, N/20000)` points are rejected
   (`0`).
2. **Axial estimate** — average the local path directions of the cluster's points
   to get its main axis.
3. **Axial-aligned PCA** (`svd_eigen`, `eigenUpdate`, `pca_transform`) — build an
   eigen-frame aligned with that axis and transform the cluster into it.
4. **Dimension filter** — cluster extent along the axis must be within
   `±25%` of `split_interval` (rejects clusters that don't match the current
   scale).
5. **Cylinder test** — fit a 2-D circle to the cross-section
   (`circleFitError`); if `FitError < t_error` and curvature `> curve_threshold`
   → wood (return radius).
6. **Linearity test** — if `evals[0]/sum(evals) > t_linearity` → wood (return
   `-linearity`).
7. Otherwise → `0` (other).

### 6.4 Path-based correction

`components_classify` then runs up to **3 correction passes**: walking each wood
cluster's path toward the root, if it meets an *upstream* wood cluster whose value
is smaller (thinner), the current cluster is demoted to non-wood. This removes
implausible wood clusters that sit "above" thinner wood (branches can't get
thicker as they get farther from the trunk).

### Parameters to tune — **primary precision/recall knobs**

| Parameter | Location | Default | Effect | Tuning |
|-----------|----------|---------|--------|--------|
| `T_LINEARITY` | `GBSeparation.py` constant | `0.95` | variance fraction along main axis to call a cluster "linear wood" | **Lower** (0.85) → more branches captured, more leaf contamination. **Higher** (0.95) → cleaner wood, misses fine branches. |
| `T_ERROR` | `GBSeparation.py` constant | `0.10` | relative cylinder-fit error tolerance | **Higher** → accepts messier cylinders (more recall, more leaf FP). **Lower** → only clean cylinders. *The single biggest precision dial* — the old effective value of `0.5` let almost any leaf blob pass as wood. |
| `CURVE_THRESHOLD` | `GBSeparation.py` constant | `0.03` | min cross-sectional curvature (thickness) before a cluster can be a cylinder | **Raise** to reject near-flat leaf clusters; **lower** toward `0.01` to allow thinner cylinders. |
| `split_interval` | `extract_init_wood` call | `[0.1,0.2,0.3,0.5,1]` | list of path-distance scales (metres) | **Add smaller** (e.g. `0.05`) to catch twigs; **add larger** for thick trunks. More scales = better coverage, slower. Must be metric. |
| `max_angle` | `extract_init_wood` call | `0.25π` | max path-direction angle before an edge is cut | **Smaller** → aggressive cutting, straighter/finer clusters (may over-fragment curves). **Larger** → keeps curved branches whole (may merge in leaves). |
| size filter `max(10, N/20000)` | `classify_info` | — | min cluster size | Raise to suppress tiny noisy clusters; lower to keep small twigs. |
| dimension tolerance `0.25` | `classify_info` | `0.25` | how tightly cluster length must match the scale | Widen for more permissive acceptance. |
| correction passes `itera_num < 3` | `components_classify` | `3` | number of path-based cleanup passes | More = more aggressive demotion of implausible wood. |

**Rule of thumb:**
- Leaves labelled as wood (false positives)? → **raise `T_LINEARITY`, lower
  `T_ERROR`, raise `CURVE_THRESHOLD`**, possibly raise the size filter.
- Branches lost (false negatives)? → **lower `T_LINEARITY`, raise `T_ERROR`**, add
  a smaller `split_interval`.

### Supporting functions (`Eigen_transform.py`, `Components_classify.py`)

- `svd_eigen(arr)` — centroid + eigenvalues/eigenvectors via SVD of the scatter
  matrix (PCA).
- `pca_transform(points, centroid, evecs)` — rotate points into the eigen-frame.
- `getAngle3D(v1, v2)` — unsigned angle between two vectors, folded to `[0, π/2]`.
- `eigenUpdate(axial, evals, evecs)` — reorders eigen components so the one most
  aligned with the path axis comes first.

---

## 7. Stage 5 — Final wood extraction (`ExtractFinalWood.py::extract_final_wood`)

```python
final_wood_mask = extract_final_wood(pcd, root_id, path_dis, path_list,
                                     init_wood_ids, G, max_iter=100)
```

Turns the confident **wood seeds** into the full wood set in three phases:

1. **Path backfill.** Every seed's entire shortest path to the root is marked
   wood (the skeleton connecting seeds to the trunk must be wood).
2. **Downward region growing.** Iteratively spread wood to graph neighbours whose
   path distance is **≤** the current node's (i.e. grow *toward* the root /
   downstream), up to `max_iter` iterations. This recovers wood in awkward places
   — bifurcations, curved/broken branches, leaf-surrounded branches.
3. **Neighbourhood smoothing.** For each wood point, mark neighbours as wood if
   the edge is shorter than `2 * predecessor_step` — fills thin gaps along
   branches.
4. **Stump extraction.** All direct neighbours of the root, plus any point below
   the root's height, are forced to wood (captures the stump/base).

### Parameters to tune

| Parameter | Location | Effect | Tuning |
|-----------|----------|--------|--------|
| `max_iter` | `extract_final_wood` call / default, line 3 | max region-growing iterations | Increase if wood regions look truncated / growth stops short on long branches. |
| smoothing factor `2 *` | line 74 | how far growth spills into neighbours during smoothing | Raise → more aggressive fill (risk leaf pickup); lower → conservative. |
| growth condition `path_dis[key] <= path_dis[i]` | line 55 | direction/containment of growth | Structural; change only if you understand the consequences. |

---

## 7.5 Stage 5.5 — Post-processing (`PostProcess.py`)

Two optional cleanup passes run on `final_wood_mask` **after** region growing and
**before** the wood/leaf split. They target the two failure modes the shape
classifier leaves behind: scattered mislabelled points, and leaf points sitting on
the bare lower trunk (physically impossible).

```python
# 1) KNN majority-vote smoothing (removes isolated speckle in both classes)
if SMOOTH_K > 0:
    final_wood_mask = smooth_labels(pcd, final_wood_mask, k=SMOOTH_K, iters=SMOOTH_ITERS)

# 2) Trunk fill (force wood on the bare lower trunk)
if FILL_TRUNK:
    final_wood_mask, crown_base_z = fill_trunk(pcd, final_wood_mask, root[0, :2], trunk_radius,
                                               radius_factor=TRUNK_RADIUS_FACTOR,
                                               spread_factor=TRUNK_SPREAD_FACTOR)
```

### 7.5.1 KNN label smoothing (`smooth_labels`)

Computes each point's `k` nearest neighbours once (including itself, so a point
gets a stabilising vote for its own label), then iterates a **majority vote**
`iters` times. A wood point surrounded by leaf flips to leaf, and vice-versa. This
erases isolated wood specks in the crown and isolated leaf specks on the trunk,
while leaving coherent regions untouched. Because neighbours are static, the kNN
search runs once and only the vote iterates.

### 7.5.2 Trunk fill (`fill_trunk`)

Uses the fact that the **lower trunk is bare** — anything close to the trunk axis
below the crown cannot be a leaf:

1. **Detect the crown base.** Scan height slices bottom-up; the crown base is the
   lowest slice whose 95th-percentile radial spread (distance from the trunk axis)
   exceeds `spread_factor * trunk_radius`.
2. **Fill.** Every point **below the crown base** and **within
   `radius_factor * trunk_radius`** of the axis (the fitted root centre) is forced
   to wood.
3. **Safe fallback.** If the crown never "opens" (spread never exceeds the
   threshold), the crown base defaults to the top of the tree and *nothing* is
   filled — so this step can only ever help, never eat a whole crown.

The trunk axis is `root[0, :2]` (the fitted root centre from Stage 1) and
`trunk_radius` is the circle radius returned by `getRootPt`. This assumes a
roughly **straight, vertical** trunk (the same precondition as Stage 1).

### Parameters to tune

| Parameter | Location | Default | Effect | Tuning |
|-----------|----------|---------|--------|--------|
| `SMOOTH_K` | `GBSeparation.py` constant | `10` (0 disables) | neighbours in the majority vote | Higher → smoother, may erode thin real branches; lower → gentler. `0` turns smoothing off. |
| `SMOOTH_ITERS` | `GBSeparation.py` constant | `2` | number of smoothing passes | More passes = stronger smoothing. |
| `FILL_TRUNK` | `GBSeparation.py` constant | `True` | enable/disable trunk fill | — |
| `TRUNK_RADIUS_FACTOR` | `GBSeparation.py` constant | `2.0` | trunk zone width = this × base radius | **Raise** if leaf still clings to the trunk; **lower** if the fill grabs low branches/crown. |
| `TRUNK_SPREAD_FACTOR` | `GBSeparation.py` constant | `3.0` | crown base = where radial spread exceeds this × base radius | **Raise** → crown base detected higher (fills more trunk). **Lower** (→2) → crown base lower (fills less). |
| `slice_height`, `min_slice_pts` | `fill_trunk` signature | `0.5`, `5` | crown-base scan resolution | Rarely need changing; shrink `slice_height` for very short trees. |

> The script prints the detected `trunk_radius` and `crown_base height` each run —
> use them to sanity-check that crown-base detection is sensible for your tree.

---

## 8. Stage 6 — Labelling, colouring, export (`GBSeparation.py`)

```python
final_wood_mask[-1] = False          # drop the synthetic root point
wood = pcd[final_wood_mask]
final_wood_mask[-1] = True
leaf = pcd[~final_wood_mask]

# labels: wood=0, leaf=1
# colours: wood = saddle brown (160,82,45), leaf = green (60,170,70)
new_las = laspy.LasData(header)      # point_format=2, LAS 1.2
new_las.x/y/z = ...
new_las.classification = labels
new_las.red/green/blue = rgb16       # 8-bit colours scaled ×257 to 16-bit
new_las.write('data/segmented_tree.las')
```

- The synthetic root point (appended in Stage 1) is excluded from the wood output
  but the mask is restored so the leaf set is computed against the full cloud.
- LAS colour channels are 16-bit, so 8-bit RGB is scaled by `257` (= `65535/255`).
- **Classification codes:** `0 = wood`, `1 = leaf` (see `WOOD_LABEL` / `LEAF_LABEL`).

> 💡 The output `.las` is written in the **local (offset-shifted)** coordinate
> frame from Stage 0. The `offset` is not persisted by the script, so to restore
> true georeferenced coordinates you must save `offset` yourself and add it back
> to X/Y/Z (or set the LAS header offset/scale accordingly).

**Parameters to tune:** colours (`rgb8` values), class codes, LAS
`point_format`/`version` — cosmetic/format only, no effect on segmentation.

---

## 9. Helper / visualization functions (`Visualization.py`)

These are **debugging aids** — call them (many are commented out in
`GBSeparation.py`) to *see* what each stage produced. Highly recommended while
tuning: uncomment one at a time.

| Function | What it shows | Suggested use |
|----------|---------------|---------------|
| `show_pcd(pcd)` | raw point cloud in red | sanity-check input & root point (line 26) |
| `show_graph(pcd, G)` | the full graph as a line set | inspect connectivity after Stage 2 (line 37); spot fragmentation |
| `sp_graph(path_list, root_id)` | builds a graph of just the shortest-path (skeleton) edges | pass to `show_graph` to view the tree skeleton (line 42) |
| `graph_cluster(pcd, G)` | labels points by connected component of `G` | with `show_clusters`, see clusters after edge cutting (`ExtractInitWood.py:43`) |
| `graph_cluster2(pcd, components)` | labels points by an explicit component list | with `show_clusters`, see per-scale clusters (`ExtractInitWood.py:68`) |
| `show_clusters(clusters)` | colours an `(x,y,z,label)` array with a `tab20` colormap | visualise any clustering result |
| `show_save_pcd_fmt(wood, leaf, path)` | shows wood (gold) + leaf (green) and saves a `.pcd` | inspect / export the final split |

**Tuning workflow with visualization:**
1. After Stage 2, `show_graph(pcd, G)` → is the graph connected and following the
   tree? If not, fix `knn` / `nbrs_threshold`.
2. After edge cutting, `show_clusters(graph_cluster(pcd, G))` → are junctions cut
   sensibly? Adjust `max_angle`.
3. Per scale, `show_clusters(graph_cluster2(pcd, components))` → do cluster sizes
   match `split_interval`?
4. Final, `show_save_pcd_fmt(wood, leaf, ...)` → inspect precision/recall visually,
   then adjust `t_linearity` / `t_error`.

---

## 10. Accuracy evaluation (`Accuracy_evaluation.py`)

Optional — only if you have **manually labelled reference clouds**.

- `clouds_matching(classify_cloud, reference_cloud, tolerance)` — nearest-neighbour
  match to count points common to both clouds (within `tolerance`, default
  `1e-5`).
- `evaluate_indicators(classify_wood, classify_leaf, reference_wood, reference_leaf, components)`
  — prints **Precision, Recall, F1** for wood and leaf, plus overall **Accuracy**
  and **Kappa**. With `components=True`, returns a labelled array tagging each
  point as correct/false wood/leaf (for error visualisation).

Use this to **quantify** the effect of each parameter change instead of eyeballing.
`tolerance` must be small enough that only truly identical points match — keep it
near `1e-5` unless your clouds were resampled.

> Note: `evaluate_indicators` is imported in `GBSeparation.py` but not currently
> called. To use it, keep a reference wood/leaf split and call it before/after
> export.

---

## 11. Quick tuning cheat-sheet (priority order)

1. **Units & graph connectivity first.** Confirm metres; confirm
   `number_connected_components ≈ 1`. Fix with `knn`, `nbrs_threshold`, or by
   cleaning input gaps. *Nothing else matters until this is right.*
2. **Root fit.** Make `lower_h`/`upper_h` cover a clean cylindrical trunk slice.
3. **Precision vs recall.** `t_linearity` (↑ = stricter) and `t_error`
   (↓ = stricter) are the main dials.
4. **Scale coverage.** Tune `split_interval` to your tree's branch thicknesses;
   add a small scale for twigs.
5. **Junction handling.** `max_angle` controls how aggressively branches are cut
   apart.
6. **Completeness of wood.** `max_iter` and the smoothing factor in
   `extract_final_wood` recover missed wood.
7. **Post-processing.** `SMOOTH_K`/`SMOOTH_ITERS` clean isolated speckle;
   `TRUNK_RADIUS_FACTOR`/`TRUNK_SPREAD_FACTOR` clear leaf off the bare trunk.
8. **Measure** with `evaluate_indicators` (if you have references) and **look**
   with the visualization helpers at every step.

---

## 12. Parameter reference table (all in one place)

| Parameter | File / location | Default | Stage |
|-----------|-----------------|---------|-------|
| `offset` (local origin) | `GBSeparation.py` (inline) | `points.min(axis=0)` | 0 |
| `INPUT_PATH`, `OUTPUT_PATH` | `GBSeparation.py` constants | — | 0 |
| `lower_h`, `upper_h` | `getRootPt` call | `0.0`, `0.2` | 1 |
| `knn` | `array_to_graph` call | `300` | 2 |
| `kpairs` | `array_to_graph` call | `3` | 2 |
| `nbrs_threshold` | `array_to_graph` call | `treeHeight/30` | 2 |
| `nbrs_threshold_step` | `array_to_graph` call | `treeHeight/60` | 2 |
| `graph_threshold` | `Graph_Path.py` | `inf` | 2 |
| `split_interval` | `extract_init_wood` call | `[0.1,0.2,0.3,0.5,1]` | 4 |
| `max_angle` | `extract_init_wood` call | `0.25π` | 4 |
| `T_LINEARITY` | `GBSeparation.py` constant | `0.95` | 4 |
| `T_ERROR` | `GBSeparation.py` constant | `0.10` | 4 |
| `CURVE_THRESHOLD` | `GBSeparation.py` constant | `0.03` | 4 |
| size filter | `Components_classify.py` | `max(10, N/20000)` | 4 |
| dimension tolerance | `Components_classify.py` | `0.25` | 4 |
| correction passes | `Components_classify.py` | `3` | 4 |
| `max_iter` | `ExtractFinalWood.py` | `100` | 5 |
| smoothing factor | `ExtractFinalWood.py` | `2×` | 5 |
| `SMOOTH_K` | `GBSeparation.py` constant | `10` (0 disables) | 5.5 |
| `SMOOTH_ITERS` | `GBSeparation.py` constant | `2` | 5.5 |
| `FILL_TRUNK` | `GBSeparation.py` constant | `True` | 5.5 |
| `TRUNK_RADIUS_FACTOR` | `GBSeparation.py` constant | `2.0` | 5.5 |
| `TRUNK_SPREAD_FACTOR` | `GBSeparation.py` constant | `3.0` | 5.5 |
| colours / class codes | `GBSeparation.py` | brown/green, 0/1 | 6 |
| matching `tolerance` | `Accuracy_evaluation.py` | `1e-5` | eval |
