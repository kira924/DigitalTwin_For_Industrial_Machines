"""
cmapss_descaler.py
------------------
Converts normalized CMAPSS sensor readings back to their real physical values
and prints them in a human-readable format.

Background
----------
The C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset
used in the PHM'08 competition applies min-max normalization to scale all
sensor outputs to the range [-1, 1] before release.

    x_scaled  = 2 * (x_raw - x_min) / (x_max - x_min) - 1
    x_raw     = (x_scaled + 1) / 2 * (x_max - x_min) + x_min

Sensor definitions are taken from Table 2 of:
    Saxena et al., "Damage Propagation Modeling for Aircraft Engine
    Run-to-Failure Simulation", PHM 2008.
"""

import json

# ── Sensor catalogue ──────────────────────────────────────────────────────────
# Each entry: (symbol, full name, unit, x_min, x_max)
# Ranges are derived from the full CMAPSS FD001–FD004 dataset.
SENSOR_CATALOGUE = {
    # Operational settings
    "setting_1": ("alt",         "Altitude",                              "ft",          0.0,     42000.0),
    "setting_2": ("Mach",        "Mach Number",                           "—",           0.0,     0.84),

    # Sensor channels
    "s_2":  ("T24",      "Total Temperature at LPC Outlet",     "°R",      488.16,  644.44),
    "s_3":  ("T30",      "Total Temperature at HPC Outlet",     "°R",     1021.65, 1616.88),
    "s_4":  ("T50",      "Total Temperature at LPT Outlet",     "°R",     1064.84, 1516.30),
    "s_7":  ("P30",      "Total Pressure at HPC Outlet",        "psia",    199.20,  556.10),
    "s_8":  ("Nf",       "Physical Fan Speed",                  "rpm",    2112.0,  2388.0),
    "s_9":  ("Nc",       "Physical Core Speed",                 "rpm",    7852.0,  9250.0),
    "s_11": ("Ps30",     "Static Pressure at HPC Outlet",       "psia",     46.20,   51.48),
    "s_12": ("phi",      "Fuel Flow / Ps30 Ratio",              "pps/psi", 185.04,  322.74),
    "s_13": ("NRf",      "Corrected Fan Speed",                 "rpm",    2028.0,  2388.0),
    "s_14": ("NRc",      "Corrected Core Speed",                "%",        84.93,  100.0),
    "s_15": ("BPR",      "Bypass Ratio",                        "—",         4.74,    8.85),
    "s_17": ("htBleed",  "Bleed Enthalpy",                      "—",       100.0,   400.0),
    "s_20": ("W31",      "HPT Coolant Bleed Flow",              "lbm/s",    13.20,   23.46),
    "s_21": ("W32",      "LPT Coolant Bleed Flow",              "lbm/s",     8.44,   23.30),
}

# ── Inverse scaler ────────────────────────────────────────────────────────────

def inverse_minmax(scaled_value: float, x_min: float, x_max: float) -> float:
    """Reverse the [-1, 1] min-max normalization."""
    return (scaled_value + 1.0) / 2.0 * (x_max - x_min) + x_min


def descale_readings(scaled: dict) -> dict:
    """
    Takes a dict of scaled sensor readings and returns a dict of
    {sensor_key: (symbol, full_name, physical_value, unit)}.
    """
    results = {}
    for key, scaled_value in scaled.items():
        if key not in SENSOR_CATALOGUE:
            results[key] = (key, "Unknown sensor", scaled_value, "?")
            continue
        symbol, full_name, unit, x_min, x_max = SENSOR_CATALOGUE[key]
        physical = inverse_minmax(scaled_value, x_min, x_max)
        results[key] = (symbol, full_name, physical, unit)
    return results


def print_readable(scaled: dict) -> None:
    """Pretty-print a complete sensor snapshot in human-readable form."""

    descaled = descale_readings(scaled)

    # Separate settings from sensors for cleaner output
    settings_keys = [k for k in descaled if k.startswith("setting_")]
    sensor_keys   = [k for k in descaled if not k.startswith("setting_")]

    # ── Header ──
    print("=" * 65)
    print("  C-MAPSS ENGINE SNAPSHOT  (physical / real-world values)")
    print("=" * 65)

    # ── Flight conditions ──
    print("\n  FLIGHT CONDITIONS")
    print("  " + "-" * 55)
    for key in settings_keys:
        symbol, full_name, value, unit = descaled[key]
        unit_str = f" {unit}" if unit not in ("—", "") else ""
        print(f"  {symbol:<12} {full_name:<35} {value:>10.2f}{unit_str}")

    # ── Engine sensors ──
    print("\n  ENGINE SENSOR READINGS")
    print("  " + "-" * 55)
    for key in sorted(sensor_keys, key=lambda k: int(k.split("_")[1])):
        symbol, full_name, value, unit = descaled[key]
        unit_str = f" {unit}" if unit not in ("—", "") else ""

        # Choose decimal places per sensor type
        if unit in ("rpm", "°R"):
            fmt = f"{value:>10.1f}"
        elif unit in ("psia",):
            fmt = f"{value:>10.2f}"
        elif unit in ("lbm/s", "pps/psi"):
            fmt = f"{value:>10.3f}"
        else:
            fmt = f"{value:>10.4f}"

        print(f"  {symbol:<12} {full_name:<35} {fmt}{unit_str}")

    print("\n" + "=" * 65)


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    current_sensor_readings = {
        "setting_1":  0.2777777910232544,
        "setting_2": -0.27272728085517883,
        "s_2":  -0.5276073813438416,
        "s_3":  -0.37871885299682617,
        "s_4":  -0.658422589302063,
        "s_7":   0.6550307869911194,
        "s_8":  -0.699999988079071,
        "s_9":  -0.602117657661438,
        "s_11": -0.8206896781921387,
        "s_12":  0.6349206566810608,
        "s_13": -0.4883720874786377,
        "s_14": -0.4329584538936615,
        "s_15": -0.32743361592292786,
        "s_17": -0.6000000238418579,
        "s_20":  0.698113203048706,
        "s_21":  0.1472916305065155,
    }

    print_readable(current_sensor_readings)

    # ── Also export to JSON ──
    descaled = descale_readings(current_sensor_readings)
    export = {
        key: {
            "symbol":        v[0],
            "full_name":     v[1],
            "physical_value": round(v[2], 4),
            "unit":          v[3],
        }
        for key, v in descaled.items()
    }
    output_path = "descaled_readings.json"
    with open(output_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n  → Physical values also saved to '{output_path}'\n")
