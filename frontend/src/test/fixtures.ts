import type { NDVIAnalysisResponse, SRResultResponse, UploadResponse } from '../api/types'

export const uploadFixture: UploadResponse = {
  upload_id: 'upload-abc123',
  filename: 'scene.tif',
  valid: true,
  band_names: ['B04', 'B03', 'B02', 'B08'],
  width: 128,
  height: 128,
  crs: 'EPSG:32630',
  resolution_m: 10.0,
  input_scale: 'raw_digital_number',
  validation_messages: ['Upload accepted: required bands and shape present, CRS valid.'],
}

export const jobResultFixture: SRResultResponse = {
  job_id: 'job-abc123',
  status: 'completed',
  upload_id: 'upload-abc123',
  model_name: 'SEN2SRLite/NonReference_RGBN_x4',
  input_shape: [4, 128, 128],
  output_shape: [4, 512, 512],
  resolution: {
    native_resolution_m: 10.0,
    sr_resolution_m: 2.5,
    scale_factor: 4,
    description: 'SR-derived product — 2.5 m pixel grid',
  },
  bands: ['B04', 'B03', 'B02', 'B08'],
  crs: 'EPSG:32630',
  uncertainty: {
    label: 'relative model-stability uncertainty',
    scalar_summary: 2.1e-5,
    scalar_summary_definition: 'Mean per-pixel standard deviation across the TTA ensemble, averaged over bands. A relative model-stability proxy, not a calibrated confidence value.',
    overall_distribution: { mean: 1.9e-5, median: 1.5e-11, std: 1.1e-4, p90: 4.2e-8, p95: 7.6e-5, min: 4.2e-12, max: 4.8e-3, n_pixels: 262144 },
    n: 6,
    seed: 42,
    transform_names: ['identity', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270'],
    disclaimer:
      'This is a relative, architecture-conditioned model-stability proxy from test-time perturbation ensembling. It is NOT a calibrated probability of error, NOT a confidence interval, and NOT a physically rigorous uncertainty bound. It is also NOT the upstream LAM explainability tool (sen2sr/xai/lam.py) -- LAM answers a different question (which input pixels influence the output) via a different mechanism (gradients on blurred input copies) and is not exposed by this API.',
  },
  self_consistency: {
    downsample_rmse: 1.5e-5,
    ndvi_discrepancy_mean_abs: 0.111,
    b08_b04_ratio_discrepancy_mean_abs: 0.327,
    note: 'Compares the SR output against its own LR input only -- not a ground-truth accuracy check.',
  },
  metadata: {
    inference_seconds: 1.582,
    device: 'cuda',
    output_geospatial: {
      crs: 'EPSG:32630',
      transform: [2.5, 0.0, 720285.0, 0.0, -2.5, 4375125.0],
      bounds: [720285.0, 4373845.0, 721565.0, 4375125.0],
      width: 512,
      height: 512,
    },
    preprocessing_mask_coverage: 1.0,
  },
  scientific_caveats: [
    "Sentinel-2's finest native band resolution is 10 m -- it has never observed the ground at 2.5 m. The SR-derived product is a learned statistical inference resampled onto a 2.5 m pixel grid, not a directly observed 2.5 m measurement.",
    'This is a relative, architecture-conditioned model-stability proxy from test-time perturbation ensembling. It is NOT a calibrated probability of error, NOT a confidence interval, and NOT a physically rigorous uncertainty bound.',
    "Self-consistency diagnostics measure agreement with the model's own LR input, not ground-truth accuracy.",
  ],
  artifacts: {
    sr_geotiff: '/workspace/jobs/job-abc123/sr_mean.tif',
    uncertainty_geotiff: '/workspace/jobs/job-abc123/uncertainty.tif',
  },
  created_at: '2026-09-11T00:00:00Z',
}

export const analysisResultFixture: NDVIAnalysisResponse = {
  analysis_id: 'analysis-abc123',
  job_id: 'job-abc123',
  status: 'completed',
  ndvi_formula: 'NDVI = (NIR - Red) / (NIR + Red), where NIR = B08, Red = B04',
  native_resolution_m: 10.0,
  sr_resolution_m: 2.5,
  comparison: {
    status: 'COMPUTABLE',
    valid_pixel_count: 16331,
    mean_abs_difference: 0.159,
    rmse: 0.658,
    max_abs_difference: 17.77,
    resampling_method: 'area_average_pool',
  },
  uncertainty_weighted_summary: {
    status: 'COMPUTABLE',
    correlation_uncertainty_vs_abs_diff: 0.4696,
    uncertainty_weighted_mean_abs_diff: 1.5188,
    unweighted_mean_abs_diff: 0.159,
  },
  scientific_caveats: [
    'SR-derived 2.5 m NDVI is a learned inference on a 2.5 m pixel grid, not a directly observed native 2.5 m vegetation measurement.',
    'The native 10 m NDVI and SR-derived NDVI are not independent ground truths -- the SR output was itself derived from the same underlying 10 m observation.',
    'Agreement or disagreement between them measures internal consistency and downstream utility, not proof of physical accuracy.',
    'Uncertainty here is the Phase 5 relative model-stability proxy, not a calibrated probability of NDVI error.',
  ],
  metadata: {
    ndvi_formula: 'NDVI = (NIR - Red) / (NIR + Red), where NIR = B08, Red = B04',
    band_names: ['B04', 'B03', 'B02', 'B08'],
    scale_factor: 4,
    native_resolution_m: 10.0,
    sr_resolution_m: 2.5,
    resampling_method: 'area_average_pool',
    native_valid_pixel_count: 16384,
    sr_valid_pixel_count: 262030,
    comparison_valid_pixel_count: 16331,
  },
  artifacts: {
    native_ndvi_geotiff: '/workspace/jobs/job-abc123/ndvi_native_analysis-abc123.tif',
    sr_ndvi_geotiff: '/workspace/jobs/job-abc123/ndvi_sr_analysis-abc123.tif',
    ndvi_diff_geotiff: '/workspace/jobs/job-abc123/ndvi_diff_analysis-abc123.tif',
  },
  created_at: '2026-09-11T00:00:01Z',
}
