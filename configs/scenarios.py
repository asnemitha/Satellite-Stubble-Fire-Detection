"""Demo rainfall scenario presets.

These are illustrative intensity presets for exercising the model end-to-end
without a live radar feed -- NOT fitted/calibrated against gauge data for any
specific real event. Peak intensities are chosen to span the range documented
in Indian Meteorological Department rainfall-intensity classifications (light
through "extremely heavy" / cloudburst-range), so the demo can show a
believable spread from "nothing to worry about" to "significant street
flooding" rather than a single fixed storm. See docs/DESIGN.md sec 7 for what
real calibration against a historical event would require.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str
    description: str
    peak_mm_per_hr: float
    duration_note: str


SCENARIOS: dict[str, Scenario] = {
    "light": Scenario(
        id="light",
        label="Light shower",
        description=(
            "Brief light rain. IMD 'light rain' range. "
            "Drainage should keep up comfortably."
        ),
        peak_mm_per_hr=12.0,
        duration_note="typical passing shower",
    ),
    "moderate": Scenario(
        id="moderate",
        label="Moderate monsoon rain",
        description=(
            "Steady monsoon rain, IMD 'moderate' range. "
            "Minor ponding possible at weak points."
        ),
        peak_mm_per_hr=45.0,
        duration_note="typical monsoon spell",
    ),
    "heavy": Scenario(
        id="heavy",
        label="Heavy monsoon storm",
        description=(
            "IMD 'heavy' to 'very heavy' range. "
            "Expect surcharging at undersized/silted pipes."
        ),
        peak_mm_per_hr=85.0,
        duration_note="active monsoon depression",
    ),
    "extreme": Scenario(
        id="extreme",
        label="Extreme cloudburst",
        description=(
            "IMD 'extremely heavy rain' territory -- "
            "peak intensity in the range reported during "
            "India's most severe urban cloudburst events. "
            "Widespread street flooding expected."
        ),
        peak_mm_per_hr=180.0,
        duration_note="short, intense convective cell",
    ),
}


def get_scenario(scenario_id: str) -> Scenario:
    if scenario_id not in SCENARIOS:
        valid = ", ".join(SCENARIOS)

        raise KeyError(
            f"Unknown scenario '{scenario_id}'. "
            f"Valid options: {valid}"
        )

    return SCENARIOS[scenario_id]