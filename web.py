"""
Sinter — L-PBF (EOS M 290) Support & Cost Studio
================================================
Pure-Python (Streamlit) app, dedicated to Laser Powder Bed Fusion, that:
  1. Accepts an STL file upload (STEP is detected but flagged — see note below).
  2. Lets you choose the MATERIAL and EOS parameter set, and the part's
     ORIENTATION on the build plate.
  3. Generates supports from the orientation.
  4. Predicts the cost in INR (₹) with a transparent parametric model of the
     EOS M 290 workflow (powder, machine, utilities, post-processing).
  5. Appends every processed part's features + predicted cost to a CSV file
     (manufacturing_data.csv) that doubles as a growing training set, and
     retrains a scikit-learn model on it so the tool can give a data-driven
     ("learned") price estimate too.

Run:
    pip install -r requirements.txt
    streamlit run app.py

NOTE — L-PBF is modelled specifically for the EOS M 290 (400 W, 250x250x325 mm):
build envelope, power draw and focus come from the EOS M 290 data sheet, and each
powder uses EOS's published parameter set (layer thickness + volume rate) from the
EOS process/material data sheets. Everything else — all prices, rates, labour,
gas and overhead defaults — is ILLUSTRATIVE
(rounded values plausible for budgetary comparison). Every value can be edited in
the sidebar; replace them with your own shop numbers and supplier quotes for a
meaningful quote.

NOTE — L-PBF post-processing and utilities are modelled explicitly: EOS unit and
chiller electricity, shield gas (argon / nitrogen), powder waste and reusable powder,
compressed air (powder handling + blasting) on the Chicago Pneumatic CPA15 TMD 300
(15 HP, 62 CFM), build-plate cost spread over re-use cycles, muffle-furnace stress
relief, wire-cut EDM cut-off on the EZEE WIN (400x400 mm, Mo wire 0.18 mm) and sand
blasting. Chiller, furnace, EDM and blasting figures are illustrative defaults.

NOTE — STEP files: tessellating STEP/IGES properly requires a real CAD kernel
(OpenCASCADE), so this app works with STL only. A .step/.stp upload is flagged
instead of silently producing wrong geometry.
"""

import os
import io
import math
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import trimesh
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manufacturing_data.csv")
CSV_COLUMNS = [
    "timestamp", "filename", "process", "material", "parameter_set", "cost_model",
    "rot_x_deg", "rot_y_deg", "rot_z_deg", "orientation_label",
    "volume_cm3", "support_volume_cm3", "surface_area_cm2", "overhang_area_cm2",
    "bbox_x_mm", "bbox_y_mm", "bbox_z_mm",
    "triangle_count", "num_supports", "n_parts", "layer_um", "volume_rate_mm3_s",
    "mass_kg", "build_hours",
    "shield_gas", "gas_m3_part", "gas_cost_inr",
    "powder_waste_kg", "powder_waste_inr", "reusable_powder_kg", "reusable_powder_inr",
    "energy_kwh_part", "edm_minutes_part", "blast_minutes_part",
    "formula_cost_inr",
]
NUMERIC_FEATURES = [
    "rot_x_deg", "rot_y_deg", "rot_z_deg",
    "volume_cm3", "support_volume_cm3", "surface_area_cm2", "overhang_area_cm2",
    "bbox_x_mm", "bbox_y_mm", "bbox_z_mm",
    "triangle_count", "num_supports", "n_parts", "layer_um", "volume_rate_mm3_s",
    "mass_kg", "build_hours",
    "gas_m3_part", "powder_waste_kg", "reusable_powder_kg",
    "energy_kwh_part", "edm_minutes_part", "blast_minutes_part",
]

# Shield (inert process) gases for the L-PBF chamber. Prices are ILLUSTRATIVE
# (₹/m³ at delivery pressure) — argon from cylinders/bundles/liquid, nitrogen from
# an on-site generator or bulk liquid, which is why it is several times cheaper.
SHIELD_GASES = {
    "Argon":    {"default_inr_m3": 250, "note": "Fully inert; suitable for every alloy."},
    "Nitrogen": {"default_inr_m3": 60,  "note": "Cheaper, but reacts with Ti (nitrides) and "
                                                "is generally not used for nickel superalloys."},
}
# Alloys for which nitrogen is normally NOT used as the process gas (flagged, not blocked).
N2_INCOMPATIBLE = {"Ti-6Al-4V", "Inconel 718"}

# Version tag of each process's cost formula, stored with every logged row. When
# a formula changes (as L-PBF did when it became EOS M 290-specific) bump its tag:
# rows logged under an older formula are then excluded from ML training so two
# different pricing models never get mixed into one "learned" model.
COST_MODEL_VERSION = {
    "L-PBF": "eos_m290_v3",     # v3: + EOS/chiller/furnace/EDM/blasting/compressor energy, plate cycles
                                # (v2: shield-gas choice + explicit powder waste / reusable powder)
}

# EOS M 290 hardware facts (EOS M 290 system data sheet).
EOS_M290 = dict(
    plate_x_mm=250, plate_y_mm=250,
    height_incl_plate_mm=325,        # Z build volume INCLUDES the build plate
    laser_w=400,                     # single Yb-fibre laser
    max_scan_mm_s=7000, focus_um=100,
    avg_power_kw=2.4,                # average electrical draw
    avg_power_heated_kw=3.2,         # ... with build-platform heating
)

# Post-processing equipment of the user's lab (read from the machine nameplates).
# Used as defaults / captions in the sidebar; power draw and rates are still editable.
EDM_EZEE_WIN = dict(                 # Wire-cut EDM used to cut parts off the build plate
    model="EZEE WIN", year=2023,
    work_x_mm=400, work_y_mm=400,
    wire="Molybdenum", wire_mm=0.18,
)
COMPRESSOR_CPA15 = dict(             # Air compressor for powder handling and sand blasting
    model="Chicago Pneumatic CPA15 TMD 300", year=2021,
    hp=15, cfm=62, pressure_kgcm2=7.2,
)

# ----------------------------------------------------------------------------
# PROCESS DEFINITIONS
# Each process lists the materials it can run and the material-specific
# numbers the cost model needs. Process-wide parameters (layer thickness,
# machine rate, shrinkage...) live in the sidebar, see sidebar_params_*().
# ----------------------------------------------------------------------------
PROCESSES = {
    "L-PBF": {
        "full_name": "Laser Powder Bed Fusion (L-PBF · EOS M 290 DMLS)",
        "description": (
            "EOS M 290: 400 W Yb-fibre laser melting thin layers of fine metal powder "
            "in an argon- or nitrogen-filled 250×250×325 mm chamber. Best accuracy and surface finish "
            "and strong, dense parts; needs supports for overhangs "
            "and heat removal, and build time is driven by melted volume AND layer count "
            "(so part height / orientation matters twice)."
        ),
        "materials": {
            # param_sets: EOS parameter set -> (layer thickness µm, skin volume rate mm³/s)
            #   from the EOS process / material data sheets for the M 290 (400 W).
            # density_g_cm3: density of EOS-built parts (EOS data sheets).
            # Sets marked "(est.)" are NOT EOS-published: every alloy is offered at 30/40/60/80 µm
            # so the dataset and quotes can span them, and their volume rate is interpolated /
            # scaled from the nearest published set. Replace with EOS values for your licence.
            # feed_cost_inr_kg: illustrative powder price — replace with your quote.
            "Ti-6Al-4V": {
                "density_g_cm3": 4.43, "feed_cost_inr_kg": 28000,
                "default_set": "30 µm · Performance",
                "param_sets": {"30 µm · Performance": (30, 5.0), "60 µm · Speed": (60, 9.0),
                               "40 µm (est.)": (40, 6.3), "80 µm (est.)": (80, 11.7)},
            },
            "AlSi10Mg": {
                "density_g_cm3": 2.67, "feed_cost_inr_kg": 5500,
                "default_set": "30 µm",
                "param_sets": {"30 µm": (30, 5.1), "60 µm": (60, 10.5),
                               "40 µm (est.)": (40, 6.9), "80 µm (est.)": (80, 14.1)},
            },
            "SS 316L": {
                "density_g_cm3": 7.97, "feed_cost_inr_kg": 7200,
                "default_set": "40 µm",
                "param_sets": {"20 µm": (20, 2.0), "40 µm": (40, 3.7), "80 µm": (80, 8.4),
                               "30 µm (est.)": (30, 2.9), "60 µm (est.)": (60, 6.1)},
            },
            "Inconel 718": {
                "density_g_cm3": 8.15, "feed_cost_inr_kg": 14500,
                "default_set": "40 µm",
                "param_sets": {"40 µm": (40, 4.2), "40 µm HiPro": (40, 5.2), "80 µm HiPro": (80, 8.2),
                               "30 µm (est.)": (30, 3.2), "60 µm (est.)": (60, 6.2)},
            },
        },
    },
}

# Quick-pick build orientations. Values are (rot_x_deg, rot_y_deg, rot_z_deg)
# applied to the part before it's dropped onto the plate. Tilting about X/Y
# changes which faces count as overhangs (support volume / setter area) AND the
# Z-height (layer count / build time) — so supports and cost respond to these.
# Z-rotation only spins the part about the vertical axis, so it doesn't change
# overhangs or height, only footprint/plate alignment.
ORIENTATION_PRESETS = {
    "As uploaded (0°, 0°, 0°)": (0, 0, 0),
    "Tilt 30° about X": (30, 0, 0),
    "Tilt 45° about X": (45, 0, 0),
    "Tilt 90° about X (on its side)": (90, 0, 0),
    "Tilt 45° about Y": (0, 45, 0),
    "Tilt 90° about Y (on its side)": (0, 90, 0),
    "Upside down (180° about X)": (180, 0, 0),
    "Custom": None,
}


# ----------------------------------------------------------------------------
# GEOMETRY HELPERS
# ----------------------------------------------------------------------------
def load_mesh(uploaded_file) -> trimesh.Trimesh:
    """Load an STL upload into a trimesh.Trimesh, oriented Z-up, resting on the plate."""
    name = uploaded_file.name.lower()
    if name.endswith((".step", ".stp")):
        raise ValueError(
            "STEP files need a full CAD kernel (OpenCASCADE) to tessellate correctly, "
            "which isn't available in this environment. Please export the model as "
            "STL from your CAD tool and upload that instead."
        )
    if not name.endswith(".stl"):
        raise ValueError("Unsupported file type — please upload a .stl file.")

    data = uploaded_file.read()
    mesh = trimesh.load(io.BytesIO(data), file_type="stl")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not parse a valid triangle mesh from this file.")

    # Drop the part so its lowest point sits on the build plate (z = 0)
    mesh.apply_translation([-mesh.bounds[0][0] - mesh.extents[0] / 2,
                             -mesh.bounds[0][1] - mesh.extents[1] / 2,
                             -mesh.bounds[0][2]])
    return mesh


def orient_mesh(mesh: trimesh.Trimesh, rot_x_deg: float, rot_y_deg: float, rot_z_deg: float) -> trimesh.Trimesh:
    """
    Return a rotated copy of `mesh` representing a chosen build-plate
    orientation, re-seated so it rests on the plate (z = 0) and is centered
    in X/Y.

    Rotation is applied about the part's own centroid (X, then Y, then Z),
    like re-orienting a part in slicer / build-prep software. Because every
    downstream step derives from this mesh's actual geometry — supports and
    overhang area read face normals, cost reads the Z height — changing
    orientation here changes supports and cost with no other code needed.
    """
    oriented = mesh.copy()

    # Rotate about the part's own center, not the world origin, so tilting
    # doesn't fling the part sideways off the plate.
    center = oriented.bounds.mean(axis=0)
    oriented.apply_translation(-center)

    if rot_x_deg:
        oriented.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rot_x_deg), [1, 0, 0]))
    if rot_y_deg:
        oriented.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rot_y_deg), [0, 1, 0]))
    if rot_z_deg:
        oriented.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rot_z_deg), [0, 0, 1]))

    # Re-seat on the build plate: lowest point at z = 0, centered in X/Y.
    oriented.apply_translation([-oriented.bounds[0][0] - oriented.extents[0] / 2,
                                 -oriented.bounds[0][1] - oriented.extents[1] / 2,
                                 -oriented.bounds[0][2]])
    return oriented


def overhang_area_cm2(mesh: trimesh.Trimesh, angle_deg: float) -> float:
    """
    Area (cm²) of downward-facing surface steeper than the self-supporting
    angle, excluding faces already lying on the build plate. Used by the
    binder-jetting model (sinter setters) and logged for the ML features.
    """
    sin_thresh = math.sin(math.radians(angle_deg))
    low_z = mesh.triangles[:, :, 2].min(axis=1)
    mask = (mesh.face_normals[:, 2] < -sin_thresh) & (low_z > 0.25)
    return float(mesh.area_faces[mask].sum()) / 100.0


def generate_supports(mesh: trimesh.Trimesh, angle_deg: float, spacing_mm: float, radius_mm: float):
    """
    Grid ray-cast support generator.

    A regular XY grid (spacing_mm) of rays is fired UP from below the build
    plate through the part. Walking up each ray, every time it enters the part
    through a downward-facing surface steeper than the self-supporting angle
    (`angle_deg` from vertical) that is not resting on the plate, a cylindrical
    pillar is placed from whatever is below (the plate, or the top of another
    part feature) up to that surface.

    Because the grid — not the mesh triangulation — decides where pillars go,
    a large flat overhang made of just two triangles gets the same pillar
    density as one made of thousands.
    """
    sin_thresh = math.sin(math.radians(angle_deg))
    (xmin, ymin, _), (xmax, ymax, _) = mesh.bounds

    # Cap the ray count for huge footprints / tiny spacings; each pillar then
    # stands in for a larger cell, so its volume is weighted up to compensate.
    MAX_RAYS = 40000
    footprint = max((xmax - xmin) * (ymax - ymin), 1e-9)
    eff_spacing = max(spacing_mm, math.sqrt(footprint / MAX_RAYS))
    weight = (eff_spacing / spacing_mm) ** 2

    xs = np.arange(xmin + eff_spacing / 2, xmax, eff_spacing)
    ys = np.arange(ymin + eff_spacing / 2, ymax, eff_spacing)
    if len(xs) == 0 or len(ys) == 0:
        return [], 0.0
    gx, gy = np.meshgrid(xs, ys)
    origins = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -1.0)])
    directions = np.tile([0.0, 0.0, 1.0], (len(origins), 1))

    try:
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=directions, multiple_hits=True
        )
    except Exception as e:
        raise ValueError(
            "Support generation needs ray casting, which requires the 'rtree' package "
            f"(pip install rtree). Original error: {e}"
        )
    if len(locations) == 0:
        return [], 0.0

    hits_by_ray = {}
    for loc, ridx, tri in zip(locations, index_ray, index_tri):
        hits_by_ray.setdefault(int(ridx), []).append((float(loc[2]), float(mesh.face_normals[tri][2])))

    supports = []  # (x, y, base_z, height)
    total_volume = 0.0
    for ridx, hits in hits_by_ray.items():
        # Sort bottom-to-top; at equal heights process "leaving" (upward-facing)
        # hits before "entering" ones, so touching bodies read as continuous
        # solid instead of spawning a pillar through the lower body.
        hits.sort(key=lambda h: (round(h[0], 4), -h[1]))
        base_z = 0.0
        inside = False                        # is the ray currently within solid material?
        prev = None
        for z, nz in hits:
            entering = nz < 0
            if prev is not None and abs(z - prev[0]) < 1e-4 and entering == prev[1]:
                continue                      # same point hit twice (ray grazed a shared edge)
            prev = (z, entering)
            if entering:                      # through a downward-facing face
                # Only an overhang if we are coming from empty space. Where two bodies
                # touch, trimesh can return just ONE of the two coincident faces, so a
                # lone "entry" while already inside solid is an internal face — ignore it.
                if (not inside and nz < -sin_thresh and z > 0.25 and (z - base_z) > 0.35):
                    height = z - base_z
                    supports.append((float(origins[ridx][0]), float(origins[ridx][1]), base_z, height))
                    total_volume += math.pi * radius_mm ** 2 * height * weight
                inside = True
            else:                             # leaving the part through an upward-facing face
                inside = False
                base_z = z

    return supports, total_volume


def build_support_mesh(supports, radius_mm):
    """Merge all support cylinders into one trimesh for plotting."""
    if not supports:
        return None
    cyls = []
    for x, y, base_z, height in supports:
        cyl = trimesh.creation.cylinder(radius=radius_mm, height=height, sections=8)
        cyl.apply_translation([x, y, base_z + height / 2])
        cyls.append(cyl)
    return trimesh.util.concatenate(cyls)


# ----------------------------------------------------------------------------
# COST MODELS (INR) — one per process, all returning the same result shape
#
#   geom = dict(volume_cm3, support_volume_cm3, surface_area_cm2,
#               overhang_area_cm2, height_mm, num_supports)
#   result = dict(lines=[(label, ₹)...], info=[(label, text)...],
#                 build_hours, mass_kg, subtotal, margin, total)
#
# Orientation reaches every model through geom: height_mm (layer count /
# print time), support_volume_cm3 and overhang_area_cm2 (support / setter cost).
# ----------------------------------------------------------------------------
def lpbf_machine_rate(p):
    """EOS M 290 hourly machine rate (₹/hr): a manual override if given, otherwise
    built up from capex depreciation + service contract + overheads, spread over
    the machine's productive hours."""
    if p["machine_rate_override"] > 0:
        return float(p["machine_rate_override"])
    productive_hrs_yr = 8760 * p["utilization_pct"] / 100.0
    depreciation = p["capex_inr"] / (p["life_years"] * productive_hrs_yr)
    maintenance = p["capex_inr"] * (p["maintenance_pct"] / 100.0) / productive_hrs_yr
    return depreciation + maintenance + p["overhead_inr_hr"]


def lpbf_plate_capacity(bx, by, gap_mm):
    """Rough grid estimate of how many bx × by footprints fit on the EOS M 290
    plate, giving each part a (bx+gap) × (by+gap) cell."""
    nx = int(EOS_M290["plate_x_mm"] // (bx + gap_mm))
    ny = int(EOS_M290["plate_y_mm"] // (by + gap_mm))
    return max(0, nx) * max(0, ny)


def compressor_electric_kw(p):
    """Electrical input (kW) of the air compressor at full load:
    rated shaft HP -> kW, divided by the motor efficiency."""
    return p["compressor_hp"] * 0.7457 / (p["compressor_eff_pct"] / 100.0)


def compressed_air_kwh(p, cfm_used, hours):
    """Electricity to draw `cfm_used` of free air for `hours`. The compressor loads in
    proportion to the air drawn (its rated delivery is `compressor_cfm`)."""
    load = min(1.0, max(0.0, cfm_used) / max(1.0, p["compressor_cfm"]))
    return compressor_electric_kw(p) * load * hours


def cost_lpbf(material, g, p):
    """
    EOS M 290 L-PBF cost model.

    Build time is built from EOS's published numbers rather than a generic rate:
      exposure   = melted volume / EOS skin volume rate  × overhead factor
                   (the EOS "volume rate" covers skin only, so contours, up/down-
                   skin and jumps are added via the overhead factor)
      recoating  = layers × recoat time        (layers = height / layer thickness)
      fixed time = inerting/preheat + cool-down  (machine is occupied, no exposure)

    Orientation therefore acts on cost three ways: support volume (powder, exposure
    time, removal), part height (layer count -> recoat time, gas, machine hours)
    and whether the part fits the 250×250×325 mm envelope at all.

    Job-level costs (machine, EOS unit + chiller electricity, gas, plate, consumables,
    labour, powder-handling air, furnace) are shared by every part nested in the same
    build. Part-level costs scale with the part: powder, supports, wire-EDM cut-off,
    sand blasting, inspection.

    Post-processing chain modelled after the build:
      muffle-furnace stress relief (whole plate) -> wire-EDM cut-off from the plate
      -> support removal -> sand blasting -> inspection (-> HIP, optional)
    """
    M = EOS_M290
    cfg = PROCESSES["L-PBF"]["materials"][material]
    set_name = p["param_set_by_material"].get(material, cfg["default_set"])
    if set_name not in cfg["param_sets"]:
        set_name = cfg["default_set"]
    layer_um, vol_rate = cfg["param_sets"][set_name]
    n = max(1, int(p["parts_per_build"]))
    tariff = p["tariff_inr_kwh"]

    # ---- build time ----
    melt_mm3 = g["volume_cm3"] * 1000 + g["support_volume_cm3"] * 1000 / max(0.1, p["support_speed_factor"])
    exposure_h_part = melt_mm3 / vol_rate * p["exposure_overhead"] / 3600
    num_layers = (g["height_mm"] + p["plate_offset_mm"]) / (layer_um / 1000)
    recoat_h = num_layers * p["recoat_sec"] / 3600
    exposure_h_job = exposure_h_part * n
    job_hours = exposure_h_job + recoat_h + p["inert_setup_h"] + p["cooldown_h"]
    part_hours = job_hours / n

    # ---- EOS unit + chiller electricity (job level) ----
    kw_print = p["eos_kw_heated"] if p["platform_heating"] else p["eos_kw_print"]
    print_h_job = exposure_h_job + recoat_h + p["inert_setup_h"]
    eos_kwh_job = kw_print * print_h_job + p["eos_kw_idle"] * p["cooldown_h"]
    chiller_kwh_job = p["chiller_kw"] * p["chiller_duty_pct"] / 100.0 * job_hours

    # ---- shield gas (job level) ----
    gas_minutes = (exposure_h_job + recoat_h) * 60
    gas = p["shield_gas"]
    gas_m3_job = (p["gas_flood_L"] + p["gas_flow_L_min"] * gas_minutes) / 1000
    gas_price = p["gas_inr_m3"][gas]
    gas_cost_part = gas_m3_job * gas_price / n
    gas_ok = not (gas == "Nitrogen" and material in N2_INCOMPATIBLE)
    machine_rate = lpbf_machine_rate(p)

    # ---- compressed air for powder handling (sieving, conveying, depowdering) ----
    powder_air_kwh_job = compressed_air_kwh(p, p["powder_air_cfm"], p["powder_air_h"])

    # ---- masses ----
    mass_kg = (g["volume_cm3"] + g["support_volume_cm3"]) * cfg["density_g_cm3"] / 1000
    share = f" (÷{n})" if n > 1 else ""

    # ---- powder accounting ----
    # Three streams of powder:
    #  1. fused      = part + supports (mass_kg): bought, and leaves in the product.
    #  2. waste      = spatter / condensate / filter fines (a % of the fused mass)
    #                  + sieve rejects, handling and ageing discards of the recovered
    #                  powder. Bought and lost -> charged to the part.
    #  3. reusable   = unfused powder in the bed + overflow that is recovered, sieved
    #                  and returned to stock. NOT a cost (it is reused on the next
    #                  job), only its reconditioning is charged; its stock value is
    #                  reported for information.
    powder_price = p["feed_cost"][material]
    bed_area_mm2 = (M["plate_x_mm"] * M["plate_y_mm"] if p["bed_full_plate"]
                    else min(M["plate_x_mm"] * M["plate_y_mm"],
                             n * (g["bbox_x_mm"] + 2 * p["bed_margin_mm"]) * (g["bbox_y_mm"] + 2 * p["bed_margin_mm"])))
    bed_cm3 = bed_area_mm2 / 100.0 * (g["height_mm"] + p["plate_offset_mm"]) / 10.0
    solid_in_bed_cm3 = n * (g["volume_cm3"] + g["support_volume_cm3"])
    loose_cm3 = max(0.0, bed_cm3 - solid_in_bed_cm3)
    loose_kg_job = (loose_cm3 * cfg["density_g_cm3"] * p["packing_pct"] / 100.0 / 1000.0
                    * (1 + p["overflow_pct"] / 100.0))
    recovery_loss_kg_job = loose_kg_job * p["recovery_loss_pct"] / 100.0
    reusable_kg_part = (loose_kg_job - recovery_loss_kg_job) / n
    process_waste_kg = mass_kg * p["process_waste_pct"] / 100.0
    waste_kg_part = process_waste_kg + recovery_loss_kg_job / n
    waste_inr = waste_kg_part * powder_price
    reusable_inr = reusable_kg_part * powder_price           # stock value, not a cost
    recond_inr = reusable_kg_part * p["recondition_inr_kg"]
    powder_bought_kg = mass_kg + waste_kg_part

    # ---- build plate (bed): price spread over its re-use cycles + resurfacing ----
    plate_cycles = max(1, int((p["plate_thickness_mm"] - p["plate_min_thickness_mm"])
                              // max(0.01, p["plate_loss_mm"])))
    plate_price = p["plate_price_inr"][material]
    plate_job_inr = plate_price / plate_cycles + p["plate_resurface_inr"]

    # ---- muffle furnace stress relief (whole plate, job level) ----
    furnace_hot_h = p["furnace_ramp_h"] + p["furnace_soak_h"]
    furnace_kwh_job = p["furnace_kw"] * p["furnace_duty_pct"] / 100.0 * furnace_hot_h
    furnace_busy_h = furnace_hot_h + p["furnace_cool_h"]
    furnace_job_inr = (furnace_kwh_job * tariff + furnace_busy_h * p["furnace_rate_inr_hr"]
                       + p["furnace_labour_h"] * p["operator_inr_hr"] + p["furnace_atmosphere_inr"])

    # ---- wire-cut EDM: cut each part off the plate ----
    # The wire passes under the part, so the cut surface is its footprint (reduced by a
    # fill factor because the base is rarely a full rectangle). Fixturing, zeroing and
    # wire threading happen once per plate and are shared by the nested parts.
    cut_area_mm2 = g["bbox_x_mm"] * g["bbox_y_mm"] * p["edm_fill_pct"] / 100.0
    edm_cut_h = cut_area_mm2 / max(1.0, p["edm_speed_mm2_min"]) / 60.0
    edm_setup_h = p["edm_setup_h"] / n
    edm_machine_inr = (edm_cut_h + edm_setup_h) * p["edm_rate_inr_hr"] + edm_setup_h * p["operator_inr_hr"]
    edm_kwh_part = edm_cut_h * p["edm_kw"]
    edm_power_inr = edm_kwh_part * tariff
    edm_consum_inr = edm_cut_h * p["edm_consumables_inr_hr"]

    # ---- sand blasting ----
    blast_h = g["surface_area_cm2"] / max(1.0, p["blast_cm2_min"]) / 60.0
    blast_media_inr = blast_h * p["blast_media_kg_hr"] * p["blast_media_inr_kg"]
    blast_air_kwh = compressed_air_kwh(p, p["blast_nozzle_cfm"], blast_h)
    blast_air_inr = blast_air_kwh * tariff
    blast_labour_inr = blast_h * (p["blast_rate_inr_hr"] + p["operator_inr_hr"])

    # ---- electricity per part (all sources) ----
    kwh_eos = eos_kwh_job / n
    kwh_chiller = chiller_kwh_job / n
    kwh_powder_air = powder_air_kwh_job / n
    kwh_furnace = furnace_kwh_job / n
    energy_kwh_part = kwh_eos + kwh_chiller + kwh_powder_air + kwh_furnace + edm_kwh_part + blast_air_kwh

    # ---- cost lines (per part) ----
    lines = [
        ("Powder in part + supports", mass_kg * powder_price),
        ("Powder waste (spatter, filter, sieve rejects)", waste_inr),
        ("Reusable-powder reconditioning (sieve / dry)", recond_inr),
        (f"Machine time{share}", job_hours * machine_rate / n),
        (f"EOS unit electricity{share}", eos_kwh_job * tariff / n),
        (f"Chiller electricity{share}", chiller_kwh_job * tariff / n),
        (f"Compressed air — powder handling{share}", powder_air_kwh_job * tariff / n),
        (f"{gas}{share}", gas_cost_part),
        (f"Build plate (bed) + resurfacing{share}", plate_job_inr / n),
        (f"Recoater blade, filters etc.{share}", p["consumables_inr"] / n),
        (f"Operator labour{share}", (p["labour_setup_h"] + p["labour_unload_h"]) * p["operator_inr_hr"] / n),
        (f"Muffle furnace stress relief{share}", furnace_job_inr / n),
        ("Support removal", g["support_volume_cm3"] * p["support_removal_inr_cm3"]),
        ("Wire-EDM cut-off — machine + setup", edm_machine_inr),
        ("Wire-EDM cut-off — power", edm_power_inr),
        ("Wire-EDM cut-off — wire / dielectric", edm_consum_inr),
        ("Sand blasting — media", blast_media_inr),
        ("Sand blasting — compressed air", blast_air_inr),
        ("Sand blasting — booth + operator", blast_labour_inr),
        ("Inspection", p["inspection_inr"]),
    ]
    if p["hip_on"]:
        lines.append(("HIP", p["hip_fee_inr"]))

    # ---- build-envelope / nesting check ----
    usable_h = M["height_incl_plate_mm"] - p["plate_thickness_mm"] - p["plate_offset_mm"]
    bx, by, bz = g["bbox_x_mm"], g["bbox_y_mm"], g["height_mm"]
    capacity = lpbf_plate_capacity(bx, by, p["nest_gap_mm"])
    problems = []
    if bx > M["plate_x_mm"] or by > M["plate_y_mm"]:
        problems.append(f"footprint {bx:.0f}×{by:.0f} mm exceeds the {M['plate_x_mm']}×{M['plate_y_mm']} mm plate")
    if bz > usable_h:
        problems.append(f"height {bz:.0f} mm exceeds ≈{usable_h:.0f} mm usable Z")
    if not problems and n > capacity:
        problems.append(f"{n} parts requested but only ≈{capacity} fit on the plate")
    if problems:
        envelope = "⚠️ " + "; ".join(problems) + " — reorient, split the part, or reduce parts per build"
    else:
        envelope = f"✅ fits ({bx:.0f}×{by:.0f}×{bz:.0f} mm; ≈{capacity} parts per plate)"

    info = [
        ("EOS parameter set", f"{set_name} — {layer_um} µm, {vol_rate} mm³/s"),
        ("Build envelope", envelope),
        ("Layers", f"{num_layers:,.0f}"),
        ("Exposure time (this part)", f"{exposure_h_part:.2f} hr"),
        ("Recoat time (whole job)", f"{recoat_h:.2f} hr"),
        ("Job time", f"{job_hours:.2f} hr for {n} part(s) → {part_hours:.2f} hr charged per part"),
        ("Machine rate", f"{inr(machine_rate)}/hr"),
        ("Mass (part + supports)", f"{mass_kg:.3f} kg"),
        ("Electricity (this part)", f"{energy_kwh_part:.1f} kWh — EOS {kwh_eos:.1f} · chiller {kwh_chiller:.1f} · "
                                    f"powder air {kwh_powder_air:.1f} · furnace {kwh_furnace:.1f} · "
                                    f"EDM {edm_kwh_part:.1f} · blasting air {blast_air_kwh:.2f}"),
        ("Shield gas", f"{gas} — {gas_m3_job:.1f} m³ per job @ {inr(gas_price)}/m³ → {inr(gas_cost_part)} per part"
                       + ("" if gas_ok else f"  ⚠️ not recommended for {material}")),
        ("Build plate", f"{inr(plate_price)} ÷ {plate_cycles} re-use cycles + {inr(p['plate_resurface_inr'])} "
                        f"resurfacing = {inr(plate_job_inr)} per job"),
        ("Wire-EDM cut-off", f"{cut_area_mm2 / 100:.1f} cm² cut at {p['edm_speed_mm2_min']:.0f} mm²/min → "
                             f"{edm_cut_h * 60:.0f} min (+{edm_setup_h * 60:.0f} min setup share)"),
        ("Sand blasting", f"{blast_h * 60:.1f} min for {g['surface_area_cm2']:.0f} cm²"),
        ("Powder waste (this part)", f"{waste_kg_part:.3f} kg → {inr(waste_inr)}  "
                                     f"(process {process_waste_kg:.3f} kg + recovery loss {recovery_loss_kg_job / n:.3f} kg)"),
        ("Reusable powder (this part's share)", f"{reusable_kg_part:.3f} kg → {inr(reusable_inr)} stock value "
                                                f"(returned to stock, not charged)"),
        ("Powder bought for this part", f"{powder_bought_kg:.3f} kg (fused {mass_kg:.3f} + waste {waste_kg_part:.3f})"),
        ("Loose powder in bed (whole job)", f"{loose_kg_job:.2f} kg over {bed_area_mm2 / 100:.0f} cm² × "
                                            f"{(g['height_mm'] + p['plate_offset_mm']):.0f} mm"),
    ]
    features = dict(parameter_set=set_name, layer_um=layer_um, volume_rate_mm3_s=vol_rate, n_parts=n,
                    shield_gas=gas, gas_m3_part=gas_m3_job / n, gas_cost_inr=gas_cost_part,
                    powder_waste_kg=waste_kg_part, powder_waste_inr=waste_inr,
                    reusable_powder_kg=reusable_kg_part, reusable_powder_inr=reusable_inr,
                    energy_kwh_part=energy_kwh_part, edm_minutes_part=edm_cut_h * 60,
                    blast_minutes_part=blast_h * 60)
    return dict(lines=lines, info=info, build_hours=part_hours, mass_kg=mass_kg, features=features,
                gas_ok=gas_ok, waste_kg=waste_kg_part, waste_inr=waste_inr,
                reusable_kg=reusable_kg_part, reusable_inr=reusable_inr, energy_kwh=energy_kwh_part)


COST_FUNCTIONS = {
    "L-PBF": cost_lpbf,
}


def compute_cost(process, material, geom, params):
    """Dispatch to the selected process's cost model and add margin + total."""
    res = COST_FUNCTIONS[process](material, geom, params)
    subtotal = sum(v for _, v in res["lines"])
    margin = subtotal * (params["margin_pct"] / 100)
    res.update(subtotal=subtotal, margin=margin, total=subtotal + margin)
    return res


def inr(value):
    """Format a number as an Indian-grouped rupee string, e.g. ₹1,23,456.78"""
    s = f"{value:,.2f}"
    whole, frac = s.split(".")
    whole = whole.replace(",", "")
    neg = whole.startswith("-")
    if neg:
        whole = whole[1:]
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts) + "," + tail
    return ("-" if neg else "") + "₹" + whole + "." + frac


# ----------------------------------------------------------------------------
# CSV LOG + ML MODEL
# ----------------------------------------------------------------------------
def _normalize_log(df: pd.DataFrame) -> pd.DataFrame:
    """Fill defaults for rows written by older versions of the app: rows with no
    process were all L-PBF, and L-PBF rows with no cost_model tag came from the
    old generic L-PBF formula ("legacy"), so they're tagged as such."""
    df = df.reindex(columns=CSV_COLUMNS)
    df["process"] = df["process"].fillna("L-PBF")
    df["shield_gas"] = df["shield_gas"].fillna("N/A")      # only L-PBF logs a shield gas
    default_tag = df["process"].map(lambda pr: "legacy" if pr == "L-PBF" else COST_MODEL_VERSION.get(pr, "v1"))
    df["cost_model"] = df["cost_model"].fillna(default_tag)
    return df


def append_to_csv(row: dict):
    """Append a row, re-writing the file so column sets always match — this
    keeps older logs (from before newer columns existed) working instead of
    erroring on a header mismatch."""
    df_row = pd.DataFrame([row])
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        combined = pd.concat([existing, df_row], ignore_index=True)
    else:
        combined = df_row
    _normalize_log(combined).to_csv(CSV_PATH, mode="w", header=True, index=False)


def load_csv() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        return _normalize_log(pd.read_csv(CSV_PATH))
    return pd.DataFrame(columns=CSV_COLUMNS)


def usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows logged under each process's CURRENT cost-model version."""
    if df.empty:
        return df
    return df[df["cost_model"] == df["process"].map(COST_MODEL_VERSION)]


def train_model(df: pd.DataFrame):
    """Train a RandomForest on the accumulated CSV log to predict cost from
    process + material + geometry + orientation."""
    MIN_ROWS = 8
    df = usable_rows(df)          # never train across different cost-model versions
    if len(df) < MIN_ROWS:
        return None, None

    data = pd.get_dummies(df.copy(), columns=["process", "material", "shield_gas"],
                          prefix=["proc", "mat", "gas"])
    gas_dummies = {f"gas_{g}" for g in list(SHIELD_GASES) + ["N/A"]}
    cat_cols = [c for c in data.columns if c.startswith(("proc_", "mat_")) or c in gas_dummies]
    feature_cols = cat_cols + [c for c in NUMERIC_FEATURES if c in data.columns]
    X = data[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    y = data["formula_cost_inr"].astype(float)

    if len(data) >= 15:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, preds) if len(set(y_test)) > 1 else float("nan"),
        "mae": mean_absolute_error(y_test, preds),
        "n_rows": len(df),
        "feature_cols": feature_cols,
    }
    return model, metrics


def predict_with_model(model, feature_cols, row: dict):
    """Predict price for a row dict holding the same fields that get logged."""
    x = {c: 0.0 for c in feature_cols}
    for c in feature_cols:
        if c in row and not c.startswith(("proc_", "mat_")):
            x[c] = float(row[c])
    for key, prefix, default in (("process", "proc_", None), ("material", "mat_", None),
                                 ("shield_gas", "gas_", "N/A")):
        col = f"{prefix}{row.get(key, default)}"
        if col in x:
            x[col] = 1.0
    return float(model.predict(pd.DataFrame([x], columns=feature_cols))[0])


# ----------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------
def make_figure(mesh: trimesh.Trimesh, support_mesh):
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        color="#b9bec4", opacity=1.0, flatshading=True, name="part",
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3, roughness=0.5),
    ))
    if support_mesh is not None:
        fig.add_trace(go.Mesh3d(
            x=support_mesh.vertices[:, 0], y=support_mesh.vertices[:, 1], z=support_mesh.vertices[:, 2],
            i=support_mesh.faces[:, 0], j=support_mesh.faces[:, 1], k=support_mesh.faces[:, 2],
            color="#e0a64b", opacity=1.0, flatshading=True, name="supports",
        ))
    ext = mesh.extents
    plate = max(ext[0], ext[1]) * 1.4
    fig.add_trace(go.Mesh3d(
        x=[-plate/2, plate/2, plate/2, -plate/2], y=[-plate/2, -plate/2, plate/2, plate/2],
        z=[0, 0, 0, 0], i=[0], j=[1], k=[2], color="#20232a", opacity=0.35, name="plate", showscale=False,
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", bgcolor="#14161a",
        ),
        paper_bgcolor="#14161a", margin=dict(l=0, r=0, t=0, b=0), height=520,
        showlegend=False,
    )
    return fig


# ----------------------------------------------------------------------------
# SIDEBAR BUILDERS — each returns the parameter dict its cost model needs.
# Defaults that depend on a choice (e.g. the EOS parameter set) include that
# choice in the widget key so they refresh when it changes.
# ----------------------------------------------------------------------------
def sidebar_supports(process="L-PBF"):
    """Support-generation controls. Returns dict(enabled, angle, spacing, radius, fill)."""
    angle = st.slider("Self-supporting angle (° from vertical)", 20, 70, 45, key="pbf_angle")
    spacing = st.number_input("Grid spacing (mm)", 0.5, 20.0, 3.0, 0.5, key="pbf_spacing")
    radius = st.number_input("Pillar radius (mm)", 0.1, 3.0, 0.4, 0.1, key="pbf_radius")
    return dict(enabled=True, angle=angle, spacing=spacing, radius=radius, fill=1.0)


def sidebar_params_lpbf(material):
    cfgs = PROCESSES["L-PBF"]["materials"]
    cfg = cfgs[material]
    M = EOS_M290
    E = EDM_EZEE_WIN
    C = COMPRESSOR_CPA15
    p = {}

    st.caption(f"**EOS M 290** · {M['laser_w']} W Yb-fibre · {M['plate_x_mm']}×{M['plate_y_mm']}×"
               f"{M['height_incl_plate_mm']} mm (Z incl. plate) · scan ≤ {M['max_scan_mm_s'] / 1000:.0f} m/s · "
               f"{M['focus_um']} µm focus")
    # list the sets from thinnest to thickest layer (est. sets sit among the published ones)
    names = sorted(cfg["param_sets"], key=lambda k: (cfg["param_sets"][k][0], "est." in k, k))
    chosen = st.selectbox(f"EOS parameter set — {material}", names,
                          index=names.index(cfg["default_set"]), key=f"pbf_set_{material}")
    layer_um, vol_rate = cfg["param_sets"][chosen]
    st.caption(f"Layer **{layer_um} µm** · skin volume rate **{vol_rate} mm³/s** "
               f"({vol_rate * 3.6:.1f} cm³/hr) · built density {cfg['density_g_cm3']} g/cm³ "
               "(EOS data sheets). Other alloys in the comparison table use their default set.")
    if "est." in chosen:
        st.warning("This set is an estimate (volume rate interpolated from the nearest EOS-published set), "
                   "not an EOS data-sheet value — replace it with your EOS parameter file before quoting.")
    p["param_set_by_material"] = {material: chosen}

    p["parts_per_build"] = st.number_input("Parts per build (nesting)", 1, 100, 1, 1, key="pbf_nest")
    p["nest_gap_mm"] = st.number_input("Spacing between nested parts (mm)", 2.0, 40.0, 10.0, 1.0, key="pbf_gap")
    p["tariff_inr_kwh"] = st.number_input("Electricity tariff (₹/kWh) — used for every machine below",
                                          1.0, 30.0, 9.0, 0.5, key="pbf_tariff")

    with st.expander("Build-time assumptions"):
        p["recoat_sec"] = st.number_input("Recoat time per layer (s)", 1, 30, 8, 1, key="pbf_recoat")
        p["exposure_overhead"] = st.number_input("Exposure overhead × (contours, up/down-skin, jumps)",
                                                 1.0, 3.0, 1.3, 0.05, key="pbf_expovh")
        p["support_speed_factor"] = st.number_input("Support exposure speed vs skin ×", 0.5, 5.0, 1.0, 0.1,
                                                    key="pbf_supspeed")
        p["plate_offset_mm"] = st.number_input("Part raised above plate for EDM cut-off (mm)", 0.0, 30.0, 5.0, 0.5,
                                               key="pbf_offset")
        p["inert_setup_h"] = st.number_input("Inerting + levelling + preheat (hr)", 0.0, 6.0, 1.0, 0.25,
                                             key="pbf_inert")
        p["cooldown_h"] = st.number_input("Cool-down before unloading (hr)", 0.0, 12.0, 1.0, 0.25, key="pbf_cool")

    with st.expander("EOS M 290 unit — power & machine economics"):
        st.caption(f"EOS data sheet: {M['avg_power_kw']} kW average draw, {M['avg_power_heated_kw']} kW with "
                   "platform heating. Edit to match your own energy-meter readings.")
        p["eos_kw_print"] = st.number_input("EOS unit power while building (kW)", 0.1, 15.0, float(M["avg_power_kw"]),
                                            0.1, key="pbf_kw")
        p["eos_kw_heated"] = st.number_input("EOS unit power with platform heating (kW)", 0.1, 15.0,
                                             float(M["avg_power_heated_kw"]), 0.1, key="pbf_kwheat")
        p["platform_heating"] = st.checkbox("Build-platform heating on", value=True, key="pbf_heat")
        p["eos_kw_idle"] = st.number_input("EOS unit power during cool-down / idle (kW)", 0.0, 10.0, 1.0, 0.1,
                                           key="pbf_kwidle")
        p["capex_inr"] = st.number_input("EOS M 290 + peripherals (₹)", 5_000_000, 200_000_000, 30_000_000,
                                         1_000_000, key="pbf_capex")
        p["life_years"] = st.number_input("Depreciation life (years)", 3, 15, 7, 1, key="pbf_life")
        p["utilization_pct"] = st.number_input("Machine utilisation (%)", 10, 95, 70, 5, key="pbf_utilpct")
        p["maintenance_pct"] = st.number_input("Service contract + spares (% of capex / yr)", 0.0, 20.0, 8.0, 0.5,
                                               key="pbf_maint")
        p["overhead_inr_hr"] = st.number_input("Facility, software & overheads (₹/hr)", 0, 5000, 600, 50,
                                               key="pbf_ovhinr")
        p["machine_rate_override"] = st.number_input("Manual machine rate (₹/hr, 0 = use derived)", 0, 30000, 0, 100,
                                                     key="pbf_rate")
        st.caption(f"Machine rate used: **{inr(lpbf_machine_rate(p))}/hr** (depreciation + service + overheads; "
                   "electricity is charged separately).")

    with st.expander("Chiller (machine cooling)"):
        st.caption("Chiller runs for the whole job. Replace the defaults with your chiller's nameplate kW.")
        p["chiller_kw"] = st.number_input("Chiller electrical power (kW)", 0.0, 30.0, 3.0, 0.5, key="pbf_chkw")
        p["chiller_duty_pct"] = st.number_input("Chiller average duty over the job (%)", 10, 100, 70, 5,
                                                key="pbf_chduty")

    with st.expander("Shield gas & consumables"):
        p["shield_gas"] = st.radio("Shield gas", list(SHIELD_GASES.keys()), horizontal=True, key="pbf_gas")
        st.caption(SHIELD_GASES[p["shield_gas"]]["note"])
        if p["shield_gas"] == "Nitrogen" and material in N2_INCOMPATIBLE:
            st.warning(f"Nitrogen is not normally used for {material} — use argon for a realistic quote.")
        p["gas_flood_L"] = st.number_input("Gas to flood chamber (L per build)", 0, 5000, 800, 50, key="pbf_flood")
        p["gas_flow_L_min"] = st.number_input("Gas flow while printing (L/min)", 0, 200, 25, 1, key="pbf_flow")
        p["gas_inr_m3"] = {gname: st.number_input(f"{gname} price (₹/m³)", 5, 2000, gcfg["default_inr_m3"], 5,
                                                  key=f"pbf_gasprice_{gname}")
                           for gname, gcfg in SHIELD_GASES.items()}
        p["consumables_inr"] = st.number_input("Recoater blade, filters etc. (₹ per job)", 0, 50000, 1500, 100,
                                               key="pbf_cons")

    with st.expander("Powder waste & reuse"):
        st.caption("**Waste** = powder bought and lost (charged to the part). **Reusable** = loose powder "
                   "recovered from the bed/overflow, sieved and returned to stock (not charged — only its "
                   "reconditioning is).")
        p["process_waste_pct"] = st.number_input("Process waste: spatter, condensate, filter (% of fused mass)",
                                                 0.0, 50.0, 4.0, 0.5, key="pbf_wproc")
        p["recovery_loss_pct"] = st.number_input("Recovery loss: sieve rejects, handling, ageing (% of loose powder)",
                                                 0.0, 50.0, 3.0, 0.5, key="pbf_wrec")
        p["bed_full_plate"] = st.checkbox("Count powder over the full plate (not just nested footprint)",
                                          value=False, key="pbf_fullplate")
        p["bed_margin_mm"] = st.number_input("Powder margin around nested parts (mm)", 0.0, 100.0, 15.0, 1.0,
                                             key="pbf_margin")
        p["packing_pct"] = st.number_input("Loose powder bed density (% of solid)", 30, 80, 55, 1, key="pbf_pack")
        p["overflow_pct"] = st.number_input("Overflow / dosing excess (% of bed powder)", 0.0, 100.0, 10.0, 5.0,
                                            key="pbf_overflow")
        p["recondition_inr_kg"] = st.number_input("Reconditioning cost: sieve, dry, QA (₹/kg reusable)",
                                                  0, 5000, 150, 10, key="pbf_recond")

    with st.expander("Air compressor (powder handling & blasting)"):
        st.caption(f"Lab compressor: **{C['model']}** · {C['hp']} HP · {C['cfm']} CFM · "
                   f"max {C['pressure_kgcm2']} kg/cm² · {C['year']}. The compressor loads in proportion to the "
                   "air drawn, so a 25 CFM job on this 62 CFM unit costs ≈40% of its full-load power.")
        p["compressor_hp"] = st.number_input("Compressor motor (HP)", 1.0, 200.0, float(C["hp"]), 1.0, key="pbf_comphp")
        p["compressor_cfm"] = st.number_input("Compressor free-air delivery (CFM)", 5, 1000, int(C["cfm"]), 1,
                                              key="pbf_compcfm")
        p["compressor_eff_pct"] = st.number_input("Motor + drive efficiency (%)", 50, 100, 90, 1, key="pbf_compeff")
        st.caption(f"Full-load electrical input ≈ **{compressor_electric_kw(p):.1f} kW**")
        p["powder_air_cfm"] = st.number_input("Air drawn for powder sieving / conveying / depowdering (CFM)",
                                              0, 500, 25, 1, key="pbf_pairc")
        p["powder_air_h"] = st.number_input("Powder-handling air time per build (hr)", 0.0, 24.0, 1.5, 0.25,
                                            key="pbf_pairh")

    with st.expander("Build plate (bed)"):
        st.caption("Plate price is spread over the number of times it can be re-used: each cycle the wire-EDM "
                   "cut-off and resurfacing remove material until the plate reaches its minimum thickness.")
        p["plate_thickness_mm"] = st.number_input("New plate thickness (mm)", 10.0, 60.0, 25.0, 1.0, key="pbf_platet")
        p["plate_min_thickness_mm"] = st.number_input("Minimum usable thickness (mm)", 5.0, 50.0, 15.0, 1.0,
                                                      key="pbf_platemin")
        p["plate_loss_mm"] = st.number_input("Thickness lost per cycle: EDM + resurfacing (mm)", 0.1, 10.0, 1.0, 0.1,
                                             key="pbf_plateloss")
        p["plate_resurface_inr"] = st.number_input("Resurfacing per cycle: grinding / milling (₹)", 0, 50000, 1500,
                                                   100, key="pbf_plateresurf")
        st.markdown("**Plate price (₹ each)** — the plate material is usually matched to the powder")
        plate_defaults = {"Ti-6Al-4V": 60000, "AlSi10Mg": 12000, "SS 316L": 18000, "Inconel 718": 22000}
        p["plate_price_inr"] = {m: st.number_input(f"Plate for {m}", 500, 500000, plate_defaults.get(m, 20000), 500,
                                                   key=f"pbf_plateprice_{m}")
                                for m in cfgs}

    with st.expander("Wire-cut EDM (bed cut-off)"):
        st.caption(f"Lab machine: **{E['model']}** ({E['year']}) · work area {E['work_x_mm']}×{E['work_y_mm']} mm · "
                   f"{E['wire']} wire Ø{E['wire_mm']} mm. The {M['plate_x_mm']}×{M['plate_y_mm']} mm EOS plate "
                   "fits inside its work area.")
        p["edm_speed_mm2_min"] = st.number_input("Cutting speed (mm² of cut surface per min)", 5, 400, 50, 5,
                                                 key="pbf_edmspeed")
        p["edm_fill_pct"] = st.number_input("Cut area as % of part footprint", 10, 100, 70, 5, key="pbf_edmfill")
        p["edm_setup_h"] = st.number_input("Fixturing + zeroing + threading (hr per plate)", 0.0, 5.0, 0.5, 0.25,
                                           key="pbf_edmsetup")
        p["edm_rate_inr_hr"] = st.number_input("EDM machine rate (₹/hr)", 0, 10000, 600, 50, key="pbf_edmrate")
        p["edm_kw"] = st.number_input("EDM power draw (kW)", 0.1, 20.0, 2.5, 0.1, key="pbf_edmkw")
        p["edm_consumables_inr_hr"] = st.number_input("Wire, filters & dielectric (₹ per cutting hr)", 0, 5000, 80, 10,
                                                      key="pbf_edmcons")

    with st.expander("Muffle furnace (stress relief)"):
        st.caption("The whole plate is stress-relieved before cut-off, so the cycle is shared by all nested parts.")
        p["furnace_kw"] = st.number_input("Furnace rated power (kW)", 0.5, 60.0, 6.0, 0.5, key="pbf_furkw")
        p["furnace_duty_pct"] = st.number_input("Average power draw while heating / holding (% of rated)", 10, 100,
                                                60, 5, key="pbf_furduty")
        p["furnace_ramp_h"] = st.number_input("Ramp-up time (hr)", 0.0, 12.0, 2.0, 0.25, key="pbf_furramp")
        p["furnace_soak_h"] = st.number_input("Soak / hold time (hr)", 0.0, 24.0, 2.0, 0.25, key="pbf_fursoak")
        p["furnace_cool_h"] = st.number_input("Cool-down in furnace, power off (hr)", 0.0, 24.0, 4.0, 0.5,
                                              key="pbf_furcool")
        p["furnace_rate_inr_hr"] = st.number_input("Furnace rate while occupied (₹/hr)", 0, 5000, 150, 10,
                                                   key="pbf_furrate")
        p["furnace_labour_h"] = st.number_input("Loading / unloading labour (hr per cycle)", 0.0, 8.0, 0.5, 0.25,
                                                key="pbf_furlab")
        p["furnace_atmosphere_inr"] = st.number_input("Protective atmosphere / argon purge (₹ per cycle)", 0, 20000,
                                                      0, 100, key="pbf_furatm")

    with st.expander("Sand blasting"):
        p["blast_cm2_min"] = st.number_input("Blasting speed (cm² of surface per min)", 5, 500, 40, 5,
                                             key="pbf_blastspeed")
        p["blast_nozzle_cfm"] = st.number_input("Nozzle air consumption (CFM)", 1, 200, 30, 1, key="pbf_blastcfm")
        p["blast_media_kg_hr"] = st.number_input("Media consumption (kg per blasting hr)", 0.0, 100.0, 10.0, 1.0,
                                                 key="pbf_blastkg")
        p["blast_media_inr_kg"] = st.number_input("Media price (₹/kg)", 5, 2000, 90, 5, key="pbf_blastprice")
        p["blast_rate_inr_hr"] = st.number_input("Blasting cabinet rate (₹/hr)", 0, 5000, 100, 10, key="pbf_blastrate")

    with st.expander("Labour & other post-processing (₹)"):
        p["operator_inr_hr"] = st.number_input("Operator rate (₹/hr)", 100, 5000, 600, 50, key="pbf_oprate")
        p["labour_setup_h"] = st.number_input("Job prep + loading (hr per build)", 0.0, 12.0, 2.0, 0.25, key="pbf_labsetup")
        p["labour_unload_h"] = st.number_input("Unload, depowder, sieve (hr per build)", 0.0, 12.0, 1.5, 0.25,
                                               key="pbf_labunload")
        p["support_removal_inr_cm3"] = st.number_input("Support removal (₹/cm³)", 0, 5000, 700, 50, key="pbf_suprem")
        p["inspection_inr"] = st.number_input("Inspection (₹/part)", 0, 50000, 1500, 100, key="pbf_insp")
        p["hip_on"] = st.checkbox("Include HIP treatment", value=False, key="pbf_hip")
        p["hip_fee_inr"] = st.number_input("HIP fee (₹)", 0, 100000, 14000, 500, key="pbf_hipfee")

    st.markdown("**Powder cost (₹/kg)**")
    p["feed_cost"] = {m: st.number_input(m, 100, 100000, cfgs[m]["feed_cost_inr_kg"], 100, key=f"pbf_pc_{m}")
                      for m in cfgs}
    return p


SIDEBAR_PARAM_BUILDERS = {
    "L-PBF": sidebar_params_lpbf,
}


# ----------------------------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Sinter — L-PBF Cost Studio", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #14161a; color: #e9eaea; }
[data-testid="stSidebar"] { background-color: #1b1e23; }
div[data-testid="stMetricValue"] { font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("Sinter — L-PBF Support & Cost Studio (EOS M 290)")
st.caption("Pick a material, upload an STL, set its build orientation → supports and ₹ cost are generated for "
           "EOS M 290 laser powder bed fusion, including post-processing (muffle furnace, wire-EDM cut-off, "
           "sand blasting). Every run is logged to a CSV that also trains a price-prediction model.")

# ---- process + orientation (sidebar, top) ----
with st.sidebar:
    st.header("1 · Process")
    process = "L-PBF"          # this app models L-PBF only
    st.caption(PROCESSES[process]["description"])

    st.header("2 · Part orientation on plate")
    orientation_choice = st.selectbox("Orientation preset", list(ORIENTATION_PRESETS.keys()))
    if ORIENTATION_PRESETS[orientation_choice] is None:
        rot_x = st.slider("Rotate about X (°)", 0, 360, 0, 5)
        rot_y = st.slider("Rotate about Y (°)", 0, 360, 0, 5)
        rot_z = st.slider("Rotate about Z (°)", 0, 360, 0, 5)
    else:
        rot_x, rot_y, rot_z = ORIENTATION_PRESETS[orientation_choice]
        st.caption(f"X={rot_x}°  ·  Y={rot_y}°  ·  Z={rot_z}°")
    st.caption("Tilting about X/Y changes which surfaces need support and the part's "
               "build height — so supports and cost update with orientation. "
               "Z-rotation only spins the footprint.")

# ---- material (main area; options depend on the process) ----
material = st.selectbox("Primary material", list(PROCESSES[process]["materials"].keys()),
                        key=f"material_{process}")

# ---- supports + process parameters (sidebar, rest) ----
with st.sidebar:
    st.header("3 · Supports")
    sup = sidebar_supports(process)

    st.header(f"4 · {process} parameters (₹)")
    params = SIDEBAR_PARAM_BUILDERS[process](material)
    params["margin_pct"] = st.number_input("Margin (%)", 0, 200, 20, 5, key=f"{process}_margin")

uploaded = st.file_uploader("Upload part (.stl)", type=["stl", "step", "stp"])

if uploaded is not None:
    try:
        with st.spinner("Reading geometry…"):
            mesh = load_mesh(uploaded)

        with st.spinner("Applying build orientation…"):
            mesh = orient_mesh(mesh, rot_x, rot_y, rot_z)

        with st.spinner("Detecting overhangs and generating supports…"):
            supports, support_volume_mm3 = [], 0.0
            if sup["enabled"]:
                supports, support_volume_mm3 = generate_supports(mesh, sup["angle"], sup["spacing"], sup["radius"])
            support_mesh = build_support_mesh(supports, sup["radius"]) if supports else None
            overhang_cm2 = overhang_area_cm2(mesh, sup["angle"])

        volume_cm3 = mesh.volume / 1000.0
        support_volume_cm3 = (support_volume_mm3 / 1000.0) * sup["fill"]   # effective printed support volume
        surface_area_cm2 = mesh.area / 100.0
        bbox = mesh.extents  # mm
        triangle_count = len(mesh.faces)

        geom = dict(
            volume_cm3=volume_cm3, support_volume_cm3=support_volume_cm3,
            surface_area_cm2=surface_area_cm2, overhang_area_cm2=overhang_cm2,
            height_mm=float(bbox[2]), num_supports=len(supports),
            bbox_x_mm=float(bbox[0]), bbox_y_mm=float(bbox[1]),
        )

        col_viz, col_cost = st.columns([3, 2])

        with col_viz:
            st.plotly_chart(make_figure(mesh, support_mesh), use_container_width=True)
            st.caption(f"Process: {PROCESSES[process]['full_name']}  ·  "
                       f"Orientation: X={rot_x}°, Y={rot_y}°, Z={rot_z}°  ({orientation_choice})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Volume", f"{volume_cm3:.2f} cm³")
            m2.metric("Supports", f"{support_volume_cm3:.2f} cm³" if sup["enabled"] else "none")
            m3.metric("Triangles", f"{triangle_count:,}")
            m4.metric("Bounding box", f"{bbox[0]:.0f}×{bbox[1]:.0f}×{bbox[2]:.0f} mm")
            n1, n2, n3 = st.columns(3)
            n1.metric("Surface area", f"{surface_area_cm2:.1f} cm²")
            n2.metric("Overhang area", f"{overhang_cm2:.1f} cm²")
            n3.metric("Build height", f"{bbox[2]:.1f} mm")

        with col_cost:
            st.subheader(f"Cost estimate — {material}")
            st.caption(PROCESSES[process]["full_name"])
            result = compute_cost(process, material, geom, params)

            for label, val in result["info"]:
                st.write(f"{label}: **{val}**")
            st.divider()
            for label, val in result["lines"]:
                st.write(f"{label}: {inr(val)}")
            st.write(f"Margin ({params['margin_pct']}%): {inr(result['margin'])}")
            st.markdown(f"### {inr(result['total'])}")
            st.caption("Parametric estimate for budgetary comparison — not a formal quote.")

        # ---- powder waste / reusable powder / shield gas summary (L-PBF) ----
        if "waste_kg" in result:
            st.subheader("Powder & shield gas")
            w1, w2 = st.columns(2)
            w1.metric("Powder waste (lost)", f"{result['waste_kg']:.3f} kg", inr(result["waste_inr"]),
                      delta_color="inverse")
            w2.metric("Reusable powder (recovered)", f"{result['reusable_kg']:.3f} kg", inr(result["reusable_inr"]),
                      delta_color="off")
            st.caption("Per part. Waste is charged in the cost above; reusable powder is returned to stock "
                       "(value shown for reference, only its reconditioning is charged).")
            if not result["gas_ok"]:
                st.warning(f"Nitrogen is not normally used for {material} — consider argon.")

        # ---- log this run to CSV ----
        row = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "filename": uploaded.name,
            "process": process,
            "material": material,
            "rot_x_deg": rot_x, "rot_y_deg": rot_y, "rot_z_deg": rot_z,
            "orientation_label": orientation_choice,
            "volume_cm3": round(volume_cm3, 4),
            "support_volume_cm3": round(support_volume_cm3, 4),
            "surface_area_cm2": round(surface_area_cm2, 3),
            "overhang_area_cm2": round(overhang_cm2, 3),
            "bbox_x_mm": round(bbox[0], 2), "bbox_y_mm": round(bbox[1], 2), "bbox_z_mm": round(bbox[2], 2),
            "triangle_count": triangle_count,
            "num_supports": len(supports),
            "mass_kg": round(result["mass_kg"], 4),
            "build_hours": round(result["build_hours"], 3),
            "formula_cost_inr": round(result["total"], 2),
        }
        row["cost_model"] = COST_MODEL_VERSION[process]
        row.update(result.get("features", {}))      # process-specific logged fields (L-PBF: parameter set, layer, rate, parts)
        append_to_csv(row)
        st.success(f"Logged this part to `{os.path.basename(CSV_PATH)}` ({row['timestamp']}).")

        # ---- material comparison within the selected process ----
        st.subheader(f"Comparison across {process} materials")
        st.caption(f"At the current orientation (X={rot_x}°, Y={rot_y}°, Z={rot_z}°) — "
                   "change the orientation or process parameters in the sidebar to see cost shift.")
        comp_rows = []
        for mat in PROCESSES[process]["materials"]:
            r = compute_cost(process, mat, geom, params)
            entry = {"Material": mat, "Mass (kg)": round(r["mass_kg"], 3),
                     "Build time (hr)": round(r["build_hours"], 2)}
            if "waste_kg" in r:
                entry["Waste (kg)"] = round(r["waste_kg"], 3)
                entry["Reusable (kg)"] = round(r["reusable_kg"], 3)
                entry["Energy (kWh)"] = round(r["energy_kwh"], 1)
                entry["Gas check"] = "OK" if r["gas_ok"] else "⚠️ not recommended"
            entry["Price"] = inr(r["total"])
            comp_rows.append(entry)
        st.table(pd.DataFrame(comp_rows).set_index("Material"))

        # ---- parameter-set comparison for the selected material ----
        st.subheader(f"Comparison across EOS parameter sets — {material}")
        st.caption("Same part, orientation and settings; only the parameter set (layer thickness and skin "
                   "volume rate) changes. Sets marked (est.) are interpolated, not EOS-published.")
        set_rows = []
        mat_sets = PROCESSES[process]["materials"][material]["param_sets"]
        for sname in sorted(mat_sets, key=lambda k: (mat_sets[k][0], "est." in k, k)):
            p_set = dict(params, param_set_by_material={**params["param_set_by_material"], material: sname})
            r = compute_cost(process, material, geom, p_set)
            set_rows.append({"Parameter set": sname + ("  ◀ selected" if sname == params["param_set_by_material"][material] else ""),
                             "Layer (µm)": mat_sets[sname][0],
                             "Volume rate (mm³/s)": mat_sets[sname][1],
                             "Build time (hr)": round(r["build_hours"], 2),
                             "Price": inr(r["total"])})
        st.table(pd.DataFrame(set_rows).set_index("Parameter set"))

        # ---- ML model trained on the CSV log ----
        st.subheader("Learned price model (trained on the CSV log)")
        df_log = load_csv()
        df_usable = usable_rows(df_log)
        n_old = len(df_log) - len(df_usable)
        model, metrics = train_model(df_log)
        if n_old:
            st.caption(f"{n_old} older logged row(s) came from a previous cost formula or another process (earlier versions of this app "
                       "also logged DED / Binder Jetting / MEX parts) and are excluded from training so they don't mix with the current estimates.")
        if model is None:
            st.info(f"{len(df_usable)} usable logged part(s) so far — need at least 8 to train a data-driven "
                     "price model. The parametric estimate above is used until then.")
        else:
            ml_price = predict_with_model(model, metrics["feature_cols"], row)
            c1, c2, c3 = st.columns(3)
            c1.metric("Model-predicted price", inr(ml_price))
            c2.metric("Formula price", inr(result["total"]))
            c3.metric("Training rows", metrics["n_rows"])
            st.caption(
                f"RandomForestRegressor trained on {metrics['n_rows']} logged parts "
                f"(hold-out MAE ≈ {inr(metrics['mae'])}"
                + (f", R² ≈ {metrics['r2']:.2f}" if not math.isnan(metrics["r2"]) else "")
                + "). The model learns the relationship between material, shield gas, geometry, orientation and the "
                  "logged formula price — replace `formula_cost_inr` values in the CSV with real invoiced "
                  "prices over time to have it learn from actual costs instead. Predictions are least "
                  "reliable for materials with few logged rows."
            )

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong while processing this file: {e}")

else:
    st.info("Upload an STL file to generate supports and get a cost estimate.")

# ---- CSV log viewer ----
st.divider()
st.subheader("Manufacturing data log")
df_log = load_csv()
if df_log.empty:
    st.caption("No parts logged yet. Upload a file above to start building the dataset.")
else:
    st.dataframe(df_log.sort_values("timestamp", ascending=False), use_container_width=True, height=260)
    st.download_button("Download CSV", df_log.to_csv(index=False).encode("utf-8"),
                        file_name="manufacturing_data.csv", mime="text/csv")
