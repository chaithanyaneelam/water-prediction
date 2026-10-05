"""Irrigation water-quality RULE ENGINE (deterministic domain formulas).

These are standard, citable formulas - not machine learning:
- SAR  = Na / sqrt((Ca + Mg) / 2)          (all in meq/L; inputs here are mg/L)
- RSC  = (CO3 + HCO3) - (Ca + Mg)          (meq/L)
- USSL class: salinity C1-C4 from EC (uS/cm), sodium S1-S4 from SAR (Richards 1954)
- RSC class:  P.S. (< 1.25), MR (1.25-2.5), U.S. (> 2.5)

An ML model (train_irrigation.py) predicts the USSL class from chemistry as a
cross-check; the rule engine is the primary, exact path and is verified against
the dataset's own SAR column in tests (99.5% agreement within 5%).
"""
import math

from backend.ml.config import RSC_CLASSES, USSL_EC_CLASSES, USSL_SAR_CLASSES

# meq/L conversion factors (mg/L per meq)
F_CA, F_MG, F_NA, F_K = 20.04, 12.15, 22.99, 39.10
F_HCO3, F_CO3 = 61.02, 30.0


def compute_sar(na_mgl: float, ca_mgl: float, mg_mgl: float) -> float:
    """Sodium Adsorption Ratio. Na, Ca, Mg in mg/L; result in (meq/L)^0.5."""
    na = na_mgl / F_NA
    ca = ca_mgl / F_CA
    mg = mg_mgl / F_MG
    return na / math.sqrt((ca + mg) / 2.0)


def compute_rsc(co3_mgl: float, hco3_mgl: float, ca_mgl: float, mg_mgl: float) -> float:
    """Residual Sodium Carbonate in meq/L. CO3/HCO3/Ca/Mg in mg/L."""
    return (co3_mgl / F_CO3 + hco3_mgl / F_HCO3) - (ca_mgl / F_CA + mg_mgl / F_MG)


def _lookup(value: float, table) -> dict:
    for code, lo, hi, meaning in table:
        if lo <= value < hi:
            return {"code": code, "meaning": meaning}
    code, _lo, _hi, meaning = table[-1]
    return {"code": code, "meaning": meaning}


def ussl_class(ec_us: float, sar: float) -> dict:
    """USSL combined class, e.g. 'C3S1' with per-hazard meanings."""
    c = _lookup(ec_us, USSL_EC_CLASSES)
    s = _lookup(sar, USSL_SAR_CLASSES)
    return {"class": f"{c['code']}{s['code']}", "salinity": c, "sodium": s}


def classify_rsc(rsc: float) -> dict:
    return _lookup(rsc, RSC_CLASSES)


def verdict(ec_us: float, sar: float, rsc: float) -> dict:
    """Human-readable irrigation suitability verdict from the three indices."""
    u = ussl_class(ec_us, sar)
    r = classify_rsc(rsc)
    sal, sod = u["salinity"]["code"], u["sodium"]["code"]
    notes = [f"Salinity hazard {sal}: {u['salinity']['meaning']}.",
             f"Sodium hazard {sod}: {u['sodium']['meaning']}."]

    suitable = True
    if sal in ("C4",):
        suitable = False
        notes.append("Very high salinity water can dehydrate plant roots; use only "
                     "with heavy leaching and salt-tolerant crops, if at all.")
    elif sal == "C3":
        notes.append("Use salt-tolerant crops (e.g. barley, cotton, sugarbeet) and "
                     "ensure drainage/leaching to prevent salt build-up.")
    if sod in ("S3", "S4"):
        suitable = False
        notes.append("High sodium damages soil structure (dispersion, reduced "
                     "infiltration); apply gypsum and organic matter, ensure drainage.")
    elif sod == "S2":
        notes.append("Acceptable for well-drained soils; monitor soil sodium.")
    if r["code"] == "U.S.":
        suitable = False
        notes.append("High residual sodium carbonate will precipitate calcium and "
                     "raise soil pH; avoid or treat (gypsum/acidulation).")
    elif r["code"] == "MR":
        notes.append("Marginal RSC: monitor soil carbonate levels over time.")

    return {
        "sar": round(sar, 3),
        "rsc": round(rsc, 3),
        "ussl_class": u["class"],
        "salinity": u["salinity"],
        "sodium": u["sodium"],
        "rsc_class": r["code"],
        "rsc_meaning": r["meaning"],
        "suitable": suitable,
        "verdict": ("SUITABLE for irrigation (with the noted precautions)"
                    if suitable else
                    "NOT SUITABLE for most irrigation uses (see notes)"),
        "notes": notes,
    }
