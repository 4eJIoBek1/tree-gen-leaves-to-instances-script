"""
tree-gen foliage mesh -> instances converter (lossless to float32 precision).

Converts a baked tree-gen "Leaves"/"Blossom" mesh (many identical rigid leaf copies)
into a point cloud + per-point quaternion + a single base-leaf object, instanced via
Geometry Nodes. File size drops ~8-10x for the foliage part.

Guarantee: the evaluated (realized) geometry is compared against the original vertices
BEFORE the result is activated. It is accepted only if the max deviation fits within a
few float32 ULPs (adaptive, scale-independent). Otherwise the object is rolled back.
The printed ULP report uses the verified realized snapshot captured during selection
(the final modifier intentionally has no Realize Instances, so it renders as instances).

Naming: modifier = "GeometryNodes" (default), node group = "<object name>GN".

MODES (two safety toggles):
  USE_SELECTED=True , AUTO_DETECT=False : convert exactly the selected MESH objects.
  USE_SELECTED=True , AUTO_DETECT=True  : convert only selected objects that look like
                                          foliage (detector runs within the selection).
  USE_SELECTED=False, AUTO_DETECT=True  : scan the whole scene with the detector.
  USE_SELECTED=False, AUTO_DETECT=False : do nothing (safety).

BASE-LEAF PLACEMENT (BASE_PLACEMENT):
  'SAME_COLLECTION' : link the base leaf next to the source object (default).
  'PARENT'          : also parent it to the source object's parent (tree empty).
  'SEPARATE'        : link into a dedicated hidden collection (ultra-safe fallback).
  The base leaf is named "<source object name>SinglePiece".

NOTES / LIMITATIONS:
  - Works on clean tree-gen foliage (one leaf shape per object, rigid copies).
  - Objects with modifiers, shared meshes (users>1), color attributes or shape keys
    are skipped.
  - Per-face material slots, UVs and materials are preserved on the base leaf.
  - Base leaves are hidden (disabled in viewport & render) but still read by Object Info.
  - If BASE_PLACEMENT puts the base leaf inside a collection that an UPPER-LEVEL GN feeds
    into a Collection Info node and you ever see a stray leaf at tree-copy origins,
    switch to BASE_PLACEMENT='SEPARATE'.
  - After conversion do not change the object's or its parent's transform
    (the base leaf bakes a compensation for the current matrix_world), and do not move
    the base leaf itself.
  - random/noise patterns shader nodes if REALIZE_OUTPUT  = False: materials driving effects with per-island / per-face /
    per-index Random Value or White Noise (keyed to element INDICES) will produce a
    DIFFERENT pattern after instancing, because element/island numbering changes.
    Patterns keyed to position / UV / normal are unaffected. A plain Principled BSDF is
    unaffected. If you need index-keyed noise to stay identical, bake it to an attribute or just use realize instances node.
  - Custom split normals are not transferred (flat low-poly leaves are fine).
"""
import bpy
import numpy as np
from mathutils import Matrix
from collections import deque

# ============================ SETTINGS ============================
USE_SELECTED    = True    # toggle 1: operate on the current MESH selection
AUTO_DETECT     = False   # toggle 2: geometry-based foliage detector
DRY_RUN         = False   # analyze & report only, modify nothing
KEEP_BACKUP     = False   # keep original mesh alive (fake user) so it can be restored
BASE_PLACEMENT  = 'PARENT'   # 'SAME_COLLECTION' | 'PARENT' | 'SEPARATE'
REALIZE_OUTPUT  = True   # True = non-laggy viewport (realized mesh); False = reduces render RAM sometimes (pure instances)
ULP_MULT        = 4       # acceptance tolerance in float32 bits (scale-independent)
POS_TOL_FLOOR   = 1e-6    # lower bound for the tolerance (float-noise guard)
RESIDUAL_TOL    = 1e-3    # Kabsch residual tolerance (rigidity of each leaf copy)
MAX_LEAF_V      = 64      # max verts per leaf (anti false-positive on branches)
BASE_COLL       = "ZZ_leaf_bases"     # used only when BASE_PLACEMENT='SEPARATE'
# =================================================================

assert bpy.app.version >= (4, 5, 0), "Requires Blender 4.5+"

def ulp_at(x):
    """One float32 ULP for magnitude |x|."""
    return float(np.exp2(np.floor(np.log2(max(abs(float(x)), 1e-30))) - 23))

def tol_for(expected):
    """Adaptive tolerance = max(floor, ULP_MULT * ulp(max|expected|))."""
    return max(POS_TOL_FLOOR, ULP_MULT * ulp_at(np.max(np.abs(expected))))

def connected_components(me):
    nV = len(me.vertices)
    adj = [[] for _ in range(nV)]
    for e in me.edges:
        a, b = e.vertices; adj[a].append(b); adj[b].append(a)
    seen = bytearray(nV); comps = []
    for s in range(nV):
        if seen[s]: continue
        stack = [s]; seen[s] = 1; comp = []
        while stack:
            v = stack.pop(); comp.append(v)
            for nb in adj[v]:
                if not seen[nb]: seen[nb] = 1; stack.append(nb)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: c[0])
    return comps

def is_foliage(me):
    """Foliage/blossom = >=2 connected components of identical size (verts & polys)."""
    comps = connected_components(me)
    if len(comps) < 2: return False, comps
    n = len(comps[0])
    if n > MAX_LEAF_V or not all(len(c) == n for c in comps): return False, comps
    set0 = set(comps[0]); setL = set(comps[-1])
    np0 = sum(1 for p in me.polygons if all(v in set0 for v in p.vertices))
    npl = sum(1 for p in me.polygons if all(v in setL for v in p.vertices))
    if np0 == 0 or npl != np0: return False, comps
    return True, comps

def get_targets():
    sel = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if USE_SELECTED and AUTO_DETECT:
        return [o for o in sel if is_foliage(o.data)[0]]
    if USE_SELECTED:
        return sel
    if AUTO_DETECT:
        return [o for o in bpy.context.scene.objects
                if o.type == 'MESH' and is_foliage(o.data)[0]]
    return []

def kabsch(P, Q):
    """Q = R@P + t with known vertex correspondence. Returns R(3x3), t(3), max residual."""
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    res = float(np.max(np.linalg.norm(Qc - (R @ Pc.T).T, axis=1)))
    return R, t, res

def read_coords(me):
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)

def estimate_bytes(me, n_leaves, n_leaf_v, loops0):
    """Rough in-file cost estimate (bytes): original vs instanced."""
    orig = len(me.vertices) * 24 + len(me.loops) * 16
    new  = n_leaves * 28 + n_leaf_v * 12 + loops0 * 16
    return orig, new

def build_base_object(name, V0, c0, comp0, me, obj, placement):
    base_co = (V0 - c0).tolist()
    c0set = set(comp0)
    comp0_polys = [p for p in me.polygons if all(v in c0set for v in p.vertices)]
    faces_local = [tuple(comp0.index(v) for v in p.vertices) for p in comp0_polys]
    base_me = bpy.data.meshes.new(name)
    base_me.from_pydata(base_co, [], faces_local); base_me.update()
    for uv in me.uv_layers:                      # UV (per-loop), order matches
        buv = base_me.uv_layers.new(name=uv.name)
        for i, pb in enumerate(base_me.polygons):
            po = comp0_polys[i]
            for lj in range(len(pb.loop_indices)):
                buv.data[pb.loop_indices[lj]].uv = uv.data[po.loop_indices[lj]].uv
    for i, pb in enumerate(base_me.polygons):    # keep per-face material slots
        pb.material_index = comp0_polys[i].material_index
    for m in me.materials:                       # materials (keep order)
        base_me.materials.append(m)

    base_obj = bpy.data.objects.new(name, base_me)
    if placement in ('SAME_COLLECTION', 'PARENT') and obj.users_collection:
        coll = obj.users_collection[0]           # place it next to the tree parts
    else:
        coll = bpy.data.collections.get(BASE_COLL) or bpy.data.collections.new(BASE_COLL)
        if coll.name not in [c.name for c in bpy.context.scene.collection.children]:
            bpy.context.scene.collection.children.link(coll)
    coll.objects.link(base_obj)
    if placement == 'PARENT' and obj.parent:     # nest under the tree empty (organizational)
        base_obj.parent = obj.parent
        base_obj.matrix_parent_inverse = Matrix.Identity(4)
    base_obj.hide_viewport = True                # hidden, but still readable by Object Info
    base_obj.hide_render = True
    return base_obj

def set_base_world(base_obj, obj, mode):
    """Apply the transform compensation AFTER placement/parenting.
       mode='match' -> base.matrix_world == obj.matrix_world (RELATIVE compensation);
       mode='identity' -> base.matrix_world = identity (for ORIGINAL)."""
    base_obj.matrix_world = obj.matrix_world.copy() if mode == 'match' else Matrix.Identity(4)
    bpy.context.view_layer.update()

def write_quat_attr(pts_me, quats, order):
    if "rot" in pts_me.attributes:
        pts_me.attributes.remove(pts_me.attributes["rot"])
    attr = pts_me.attributes.new("rot", type='QUATERNION', domain='POINT')
    for i, q in enumerate(quats):
        attr.data[i].value = (q.x, q.y, q.z, q.w) if order == 'xyzw' else (q.w, q.x, q.y, q.z)

def build_gn(obj, base_obj, tspace, realize):
    """Node group '<obj>GN', modifier 'GeometryNodes', compact layout."""
    ng = bpy.data.node_groups.new(obj.name + "GN", 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='INPUT',  socket_type='NodeSocketGeometry')
    n, l = ng.nodes, ng.links
    gin  = n.new('NodeGroupInput');  gin.location = (-340, 200)
    ip   = n.new('GeometryNodeInstanceOnPoints'); ip.location = (-80, 200)
    gout = n.new('NodeGroupOutput'); gout.location = (180, 200)
    oinf = n.new('GeometryNodeObjectInfo');       oinf.location = (-340, -40)
    oinf.inputs['Object'].default_value = base_obj; oinf.transform_space = tspace
    na   = n.new('GeometryNodeInputNamedAttribute'); na.location = (-340, -280)
    na.data_type = 'QUATERNION'; na.inputs['Name'].default_value = "rot"
    l.new(gin.outputs['Geometry'],  ip.inputs['Points'])
    l.new(oinf.outputs['Geometry'], ip.inputs['Instance'])
    l.new(na.outputs['Attribute'],  ip.inputs['Rotation'])
    if realize:
        ri = n.new('GeometryNodeRealizeInstances'); ri.location = (180, 200)
        gout.location = (450, 200)
        l.new(ip.outputs['Instances'], ri.inputs['Geometry'])
        l.new(ri.outputs['Geometry'],  gout.inputs['Geometry'])
    else:
        l.new(ip.outputs['Instances'], gout.inputs['Geometry'])
    mod = obj.modifiers.new("GeometryNodes", 'NODES'); mod.node_group = ng
    return mod

def evaluated_coords(obj):
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps); m = ev.to_mesh()
    co = np.empty(len(m.vertices) * 3, dtype=np.float64)
    m.vertices.foreach_get("co", co); ev.to_mesh_clear()
    return co.reshape(-1, 3)

def ulp_report(ev, expected):
    if len(ev) != len(expected):
        print("    ULP report skipped: length mismatch (instances not realized).")
        return
    delta = np.abs(ev - expected)
    g_ulp = ulp_at(np.max(np.abs(expected)))     # single global ulp for the whole mesh
    n_ulps = delta / g_ulp
    exact = int(np.sum(np.all(delta == 0, axis=1)))   # verts with ALL components equal
    print(f"    ULP report: verts={len(ev)} exact_verts={exact} ({100*exact/len(ev):.2f}%) "
          f"max={n_ulps.max():.1f} ulp  <=1ulp: {100*np.mean(n_ulps <= 1.0+1e-9):.2f}%  "
          f"maxD={delta.max():.3e}  1ulp@max={g_ulp:.3e}")

def process(obj):
    bpy.context.view_layer.update()
    if obj.modifiers:
        return ("SKIP", "has modifiers - apply/remove them first", 0, 0, 0.0)
    me = obj.data
    if me.users > 1:
        return ("SKIP", f"mesh shared by {me.users} objects - make single-user first", 0, 0, 0.0)
    if me.shape_keys:
        return ("SKIP", "has shape keys - not supported", 0, 0, 0.0)
    if me.color_attributes:
        return ("SKIP", "has color attributes (not transferred) - skipped for safety", 0, 0, 0.0)
    ok, comps = is_foliage(me)
    if not ok:
        return ("FAIL", "not foliage/blossom (components unequal or <2)", 0, 0, 0.0)
    N = len(comps[0])
    M = np.array(obj.matrix_world, dtype=np.float64)
    verts_co = read_coords(me)
    V0 = verts_co[comps[0]]; c0 = V0.mean(0)

    quats, trans, bad = [], [], []
    for i, c in enumerate(comps):
        R, t, res = kabsch(V0, verts_co[c])
        if res > RESIDUAL_TOL: bad.append((i, res))
        quats.append(Matrix(R).to_quaternion()); trans.append(R @ c0 + t)
    if bad:
        return ("FAIL", f"{len(bad)} leaves are not rigid copies (res={bad[0][1]:.2e})", N, len(comps), 0.0)

    c0set = set(comps[0])
    loops0 = sum(len(p.loop_indices) for p in me.polygons if all(v in c0set for v in p.vertices))
    ob, nb = estimate_bytes(me, len(comps), N, loops0)
    saved_mb = (ob - nb) / 1e6

    if DRY_RUN:
        return ("DRY", f"verts/leaf={N} leaves={len(comps)} ~saves={saved_mb:.1f}MB (no changes)", N, len(comps), saved_mb)

    expected_local = np.vstack([verts_co[c] for c in comps])
    expected_world = (M[:3, :3] @ expected_local.T + M[:3, 3:4]).T
    tol_l = tol_for(expected_local); tol_w = tol_for(expected_world)

    old_mesh = obj.data
    base_obj = build_base_object(obj.name + "SinglePiece", V0, c0, comps[0], me, obj, BASE_PLACEMENT)
    pts_me = bpy.data.meshes.new(obj.name + "_pts")
    pts_me.from_pydata([tuple(t) for t in trans], [], []); pts_me.update()
    obj.data = pts_me                            # in-place: upper-GN references stay valid

    STRATS = [('RELATIVE', 'match'), ('ORIGINAL', 'identity')]
    best = None
    for tspace, mode in STRATS:
        set_base_world(base_obj, obj, mode)
        mod = build_gn(obj, base_obj, tspace, realize=True)
        for order in ('xyzw', 'wxyz'):
            write_quat_attr(pts_me, quats, order); pts_me.update()
            ev = evaluated_coords(obj)
            if len(ev) != len(expected_local): continue
            dl = float(np.abs(ev - expected_local).max())
            dw = float(np.abs(ev - expected_world).max())
            pl, pw = (dl <= tol_l), (dw <= tol_w)
            if pl or pw:
                nl = (dl / tol_l) if pl else float('inf')
                nw = (dw / tol_w) if pw else float('inf')
                et = 'local' if nl <= nw else 'world'
                metric = min(nl, nw)
                if best is None or metric < best[0]:
                    best = (metric, min(dl, dw), tspace, mode, order, et, ev.copy())
        ng = mod.node_group; obj.modifiers.remove(mod); bpy.data.node_groups.remove(ng)

    if best is None:                             # ---- ROLLBACK ----
        obj.data = old_mesh
        bpy.data.objects.remove(base_obj, do_unlink=True)
        bpy.data.meshes.remove(pts_me)
        return ("FAIL", f"no strategy reached 1-to-1 (tol_l={tol_l:.2e} tol_w={tol_w:.2e})", N, len(comps), 0.0)

    metric, absd, tspace, mode, order, et, best_ev = best   # ---- COMMIT ----
    set_base_world(base_obj, obj, mode)          # re-apply compensation after placement
    write_quat_attr(pts_me, quats, order); pts_me.update()
    build_gn(obj, base_obj, tspace, realize=REALIZE_OUTPUT)   # realize toggles viewport comfort vs render RAM

    # name the point-cloud mesh data exactly after the object:
    # free the name held by the old mesh first (it is still alive at this point)
    if old_mesh.name == obj.name:
        old_mesh.name = obj.name + "_orig"
    pts_me.name = obj.name

    if KEEP_BACKUP:
        old_mesh.use_fake_user = True            # backup stays as "<name>_orig"

    ulp_report(best_ev, expected_local if et == 'local' else expected_world)
    return ("OK", f"verts/leaf={N} leaves={len(comps)} maxD={absd:.2e} "
                  f"strat={tspace}/{mode}/{order}/{et} ~saves={saved_mb:.1f}MB", N, len(comps), saved_mb)

# ============================== RUN ==============================
targets = get_targets()
if not targets:
    print("SAFETY: nothing to do. Select objects (USE_SELECTED) and/or enable AUTO_DETECT.")
rep = []; total_saved = 0.0
for o in targets:
    try:
        s, msg, n, nl, saved = process(o)
    except Exception as ex:                      # isolate per-object errors
        s, msg, saved = "ERROR", str(ex), 0.0
    total_saved += saved
    rep.append((o.name, s, msg))
    print(f"[{s}] {o.name}: {msg}")
print("\n===== SUMMARY =====  OK:", sum(1 for _, s, _ in rep if s == "OK"), "/", len(rep),
      f" est. saved: {total_saved:.1f} MB")
for name, s, msg in rep:
    if s not in ("OK", "DRY"): print(f"  {s}: {name} -> {msg}")