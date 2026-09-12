/**
 * The exact phrases frame/api/schemas.py fixes for this project (see that
 * module's docstring and frame/api/README.md's "Scientific terminology"
 * section). Kept as one constants module so every component that needs to
 * label something before an API response exists (e.g. the landing-page
 * pipeline diagram) uses the identical wording the backend itself returns
 * once a real result comes back -- never an approximation, never a
 * marketing paraphrase.
 *
 * NEVER say "native 2.5 m Sentinel-2" or "true 2.5 m image" anywhere in
 * this app. NEVER call the uncertainty a "calibrated confidence" or
 * "probability of error." NEVER call LAM "uncertainty."
 */

export const SR_PRODUCT_DESCRIPTION = 'SR-derived product — 2.5 m pixel grid'
export const UNCERTAINTY_LABEL = 'relative model-stability uncertainty'

export const GROUND_TRUTH_DISCLAIMER = "Sentinel-2's finest native band resolution is 10 m — it has never observed the ground at 2.5 m. The SR-derived product is a learned statistical inference resampled onto a 2.5 m pixel grid, not a directly observed 2.5 m measurement."

export const UNCERTAINTY_EXPLAINER = 'Higher values indicate lower stability under the selected test-time perturbations. This is a relative model-stability signal, not a calibrated probability of error.'

export const LAM_DISTINCTION = 'This is not the same as LAM (upstream sensitivity/explainability). LAM answers which input pixels influence the output; this uncertainty answers how much the model disagrees with itself across equivalent views of the same input. Neither is exposed as the other.'

export const VALIDATION_DISCLAIMER = 'External-reference comparisons (e.g. downstream NDVI agreement) are not native Sentinel-2 ground truth — they measure internal consistency and downstream utility, not proof of physical accuracy.'

export const STATIC_SCIENTIFIC_CAVEATS: string[] = [GROUND_TRUTH_DISCLAIMER, UNCERTAINTY_EXPLAINER, VALIDATION_DISCLAIMER]
