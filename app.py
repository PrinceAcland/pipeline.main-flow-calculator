# AI Documentation
# AI tool used: Claude (Anthropic)
#
# Key prompts used:
# 1. "Build a Streamlit fluid flow calculator for a pipeline - sidebar inputs
#    for fluid type, pipe diameter, length and flow rate, a chart of pressure
#    drop vs flow rate, and a results table with Reynolds number and friction
#    factor."
# 2. "Do the unit conversions properly in SI units instead of using memorized
#    oilfield constants, so the pressure drop numbers are actually correct."
# 3. "Add error handling so invalid inputs (zero or negative values) show a
#    warning message instead of crashing the app."
#
# Most important thing manually fixed/verified:
# I checked the pressure drop calculation by hand for one case (2 in pipe,
# 500 bbl/day, water) using the Darcy-Weisbach equation and compared it to
# the app output to make sure the SI unit conversions (bbl -> m3, in -> m,
# cP -> Pa.s, Pa -> psi) were not introducing errors. I also had to correct
# the Reynolds number threshold used to switch between the laminar (64/Re)
# and turbulent (Swamee-Jain) friction factor equations.

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pipeline Flow Calculator", layout="centered")

st.title("Pipeline Fluid Flow Calculator")
st.subheader("Pressure drop and flow regime for liquid flow in a pipeline")
st.write(
    "Enter the pipe and fluid details in the sidebar. The app calculates "
    "flow velocity, Reynolds number, friction factor and pressure drop "
    "along the pipe, using the Darcy-Weisbach equation."
)

# --- unit conversion constants (SI base) ---
BBL_TO_M3 = 0.158987
FT_TO_M = 0.3048
IN_TO_M = 0.0254
LBFT3_TO_KGM3 = 16.0185
CP_TO_PAS = 0.001
PA_TO_PSI = 1 / 6894.757

# --- fluid presets (density in lb/ft3, viscosity in cP) ---
fluid_presets = {
    "Light Crude Oil": (53.0, 5.0),
    "Heavy Crude Oil": (58.0, 100.0),
    "Water": (62.4, 1.0),
    "Custom": (None, None),
}

roughness_options = {
    "Smooth (drawn tubing)": 0.0000015,
    "Commercial steel pipe": 0.000045,
    "Cast iron / rough pipe": 0.00026,
}

# --- sidebar inputs ---
st.sidebar.header("Pipe and Fluid Inputs")

fluid_type = st.sidebar.selectbox("Fluid type", list(fluid_presets.keys()))

if fluid_type == "Custom":
    density = st.sidebar.number_input("Fluid density (lb/ft3)", value=55.0, min_value=0.0)
    viscosity = st.sidebar.number_input("Fluid viscosity (cP)", value=5.0, min_value=0.0)
else:
    density, viscosity = fluid_presets[fluid_type]
    st.sidebar.write(f"Density: {density} lb/ft3, Viscosity: {viscosity} cP")

diameter_in = st.sidebar.number_input("Pipe internal diameter (inches)", value=4.0, min_value=0.0)
length_ft = st.sidebar.number_input("Pipe length (ft)", value=5000.0, min_value=0.0)
flow_rate = st.sidebar.slider("Flow rate (bbl/day)", min_value=0, max_value=20000, value=3000, step=100)
roughness_choice = st.sidebar.selectbox("Pipe roughness", list(roughness_options.keys()))
roughness = roughness_options[roughness_choice]


def friction_factor(re, rel_roughness):
    """Return Darcy friction factor. Laminar below Re = 2300, else Swamee-Jain."""
    if re < 2300:
        return 64 / re
    return 0.25 / (np.log10(rel_roughness / 3.7 + 5.74 / re ** 0.9)) ** 2


def compute_pressure_drop(q_bbl_day, d_in, l_ft, rho_lbft3, mu_cp, eps_m):
    """Compute velocity (ft/s), Reynolds number, friction factor and
    pressure drop (psi) for given flow conditions. Returns None values
    if inputs are invalid."""
    if q_bbl_day <= 0 or d_in <= 0 or l_ft <= 0 or rho_lbft3 <= 0 or mu_cp <= 0:
        return None

    d_m = d_in * IN_TO_M
    l_m = l_ft * FT_TO_M
    rho = rho_lbft3 * LBFT3_TO_KGM3
    mu = mu_cp * CP_TO_PAS
    q_m3s = q_bbl_day * BBL_TO_M3 / 86400

    area = np.pi / 4 * d_m ** 2
    v_ms = q_m3s / area

    re = rho * v_ms * d_m / mu
    rel_rough = eps_m / d_m
    f = friction_factor(re, rel_rough)

    dp_pa = f * (l_m / d_m) * (rho * v_ms ** 2 / 2)
    dp_psi = dp_pa * PA_TO_PSI

    v_fts = v_ms / FT_TO_M

    return {
        "velocity_fts": v_fts,
        "reynolds": re,
        "friction_factor": f,
        "pressure_drop_psi": dp_psi,
    }


result = compute_pressure_drop(flow_rate, diameter_in, length_ft, density, viscosity, roughness)

if result is None:
    st.warning("Please enter positive, non-zero values for all inputs to run the calculation.")
else:
    regime = "Laminar" if result["reynolds"] < 2300 else "Turbulent"

    st.subheader("Results")
    results_df = pd.DataFrame({
        "Parameter": [
            "Flow velocity (ft/s)",
            "Reynolds number",
            "Flow regime",
            "Friction factor",
            "Pressure drop (psi)",
            "Pressure drop (psi per 1000 ft)",
        ],
        "Value": [
            f"{result['velocity_fts']:.2f}",
            f"{result['reynolds']:.0f}",
            regime,
            f"{result['friction_factor']:.4f}",
            f"{result['pressure_drop_psi']:.1f}",
            f"{result['pressure_drop_psi'] / (length_ft / 1000):.2f}",
        ],
    })
    st.table(results_df)

    st.subheader("Pressure Drop vs Flow Rate")
    q_range = np.linspace(100, 20000, 60)
    dp_values = []
    for q in q_range:
        r = compute_pressure_drop(q, diameter_in, length_ft, density, viscosity, roughness)
        dp_values.append(r["pressure_drop_psi"] if r else np.nan)

    fig, ax = plt.subplots()
    ax.plot(q_range, dp_values, color="steelblue", label="Pressure drop")
    ax.scatter([flow_rate], [result["pressure_drop_psi"]], color="firebrick", zorder=5, label="Current input")
    ax.set_xlabel("Flow rate (bbl/day)")
    ax.set_ylabel("Pressure drop (psi)")
    ax.set_title("Pressure Drop vs Flow Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.caption(
    "Notes: pressure drop is calculated using the Darcy-Weisbach equation. "
    "Friction factor uses 64/Re for laminar flow (Re < 2300) and the "
    "Swamee-Jain approximation for turbulent flow."
)
