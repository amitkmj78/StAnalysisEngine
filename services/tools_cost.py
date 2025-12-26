def formulation_cost_estimator(drug_name: str, route: str, dosage_form: str, batch_scale: str) -> str:
    # Heuristic multipliers (you can refine)
    base = {
        "oral": (0.5, 3.0),
        "topical": (0.8, 4.0),
        "injectable": (3.0, 15.0),
        "inhaled": (5.0, 25.0),
    }
    r = route.lower()
    low, high = base.get(r, (1.0, 10.0))

    # Complexity multiplier examples
    mult = 1.0
    df = dosage_form.lower()
    if "extended" in df or "controlled" in df:
        mult *= 1.8
    if "lyoph" in df:
        mult *= 2.0
    if r == "injectable" and ("sterile" in df or "vial" in df or "prefilled" in df):
        mult *= 1.7

    # Batch scale factor
    if batch_scale.lower() == "pilot":
        mult *= 1.4

    est_low = round(low * mult, 2)
    est_high = round(high * mult, 2)

    return (
        f"Estimated formulation + CMC development cost range for {drug_name} "
        f"({route}, {dosage_form}, {batch_scale}): ${est_low}M–${est_high}M.\n"
        f"Main drivers: API complexity, process development, analytical/QC, stability, "
        f"manufacturing steps (sterile/lyo/CR if applicable), packaging."
    )
