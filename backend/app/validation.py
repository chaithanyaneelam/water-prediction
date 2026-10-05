"""Input validation with clear, physical-range error messages."""
from backend.ml.config import PHYSICAL_RANGES

FEATURE_FIELDS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
REQUIRED_FIELDS = [f for f in FEATURE_FIELDS if f != "ph"]  # ph optional


class ValidationError(Exception):
    def __init__(self, errors: list):
        self.errors = errors
        super().__init__("; ".join(errors))


def _to_float(value):
    """Accept numbers or numeric strings; reject bools, blanks become None."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("must be a number")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{value}' is not a number")


def validate_reading(data: dict, *, require_all: bool = True) -> dict:
    """Validate one reading dict. Returns typed floats. Raises ValidationError.

    Physical range checks come from config.PHYSICAL_RANGES (ph 0-14, others >= 0).
    Missing values are allowed only for ph (the saved imputer handles them).
    """
    errors = []
    parsed = {}

    for field in REQUIRED_FIELDS:
        try:
            v = _to_float(data.get(field))
        except ValueError as exc:
            errors.append(f"{field}: {exc}")
            continue
        if v is None:
            errors.append(f"{field}: this field is required")
            continue
        parsed[field] = v

    # ph is optional (missing -> imputed later), but if given must be numeric/in-range.
    if "ph" in data:
        try:
            v = _to_float(data.get("ph"))
        except ValueError as exc:
            errors.append(f"ph: {exc}")
            v = "error"
        if v != "error" and v is not None:
            parsed["ph"] = v

    # Physical range checks on everything present.
    for field, v in parsed.items():
        lo, hi = PHYSICAL_RANGES.get(field, (0.0, None))
        if lo is not None and v < lo:
            errors.append(f"{field}: {v} is below the physical minimum ({lo})")
        if hi is not None and v > hi:
            errors.append(f"{field}: {v} is above the physical maximum ({hi})")

    if errors:
        raise ValidationError(errors)
    return parsed
