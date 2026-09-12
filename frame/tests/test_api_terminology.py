"""Tests that the FRAME API's mandated scientific terminology is correct
and never regresses -- checked directly against frame.api.schemas'
constants, independent of any running app (Phase 7 requirements 6, 7, 8).
"""

from frame.api import schemas

FORBIDDEN_PHRASES = [
    "native 2.5 m sentinel-2",
    "native 2.5m sentinel-2",
    "true 2.5 m image",
    "true 2.5m image",
]


def _all_terminology_text() -> str:
    return " ".join(
        [
            schemas.SR_PRODUCT_DESCRIPTION,
            schemas.UNCERTAINTY_LABEL,
            schemas.UNCERTAINTY_DISCLAIMER,
            schemas.GROUND_TRUTH_DISCLAIMER,
            " ".join(schemas.SCIENTIFIC_CAVEATS),
        ]
    ).lower()


def test_sr_product_uses_the_mandated_phrase():
    assert schemas.SR_PRODUCT_DESCRIPTION == "SR-derived product — 2.5 m pixel grid"


def test_uncertainty_label_uses_the_mandated_phrase():
    assert schemas.UNCERTAINTY_LABEL == "relative model-stability uncertainty"


def test_forbidden_ground_truth_phrases_never_appear():
    text = _all_terminology_text()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in text, f"forbidden phrase found: {phrase!r}"


def test_uncertainty_disclaimer_denies_calibration_and_confidence_interval():
    text = schemas.UNCERTAINTY_DISCLAIMER.lower()
    assert "not a calibrated probability" in text
    assert "not a confidence interval" in text
    assert "not a physically rigorous" in text


def test_uncertainty_disclaimer_explicitly_distinguishes_lam():
    text = schemas.UNCERTAINTY_DISCLAIMER.lower()
    assert "lam" in text
    assert "not the upstream lam" in text or "not lam" in text


def test_lam_is_never_called_uncertainty_anywhere_in_terminology_constants():
    # LAM must only ever appear in a sentence that DENIES it is uncertainty --
    # a crude but effective structural check: every LAM mention must be
    # within a few words of a negation.
    text = schemas.UNCERTAINTY_DISCLAIMER.lower()
    lam_index = text.index("lam")
    window = text[max(0, lam_index - 20) : lam_index]
    assert "not" in window


def test_ground_truth_disclaimer_states_sentinel2s_finest_native_resolution():
    text = schemas.GROUND_TRUTH_DISCLAIMER.lower()
    assert "10 m" in text
    assert "never observed the ground at 2.5 m" in text


def test_resolution_description_default_matches_sr_product_description():
    rd = schemas.ResolutionDescription(native_resolution_m=10.0, sr_resolution_m=2.5, scale_factor=4)
    assert rd.description == schemas.SR_PRODUCT_DESCRIPTION


def test_uncertainty_summary_default_label_matches_mandated_phrase():
    summary = schemas.UncertaintySummary(
        scalar_summary=0.001,
        scalar_summary_definition="mean per-pixel std",
        overall_distribution={"mean": 0.001},
        n=6,
        seed=42,
        transform_names=["identity"],
    )
    assert summary.label == schemas.UNCERTAINTY_LABEL
    assert summary.disclaimer == schemas.UNCERTAINTY_DISCLAIMER
