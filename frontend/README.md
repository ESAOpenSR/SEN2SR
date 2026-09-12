# FRAME frontend (Phase 8)

A single-page console for demonstrating the FRAME pipeline end to end:
upload a Sentinel-2 L2A GeoTIFF, run FRAME, and inspect the SR-derived
output, its model-stability uncertainty, and the downstream NDVI analysis
— entirely through the real FastAPI backend (`frame/api/`), with no
Python imported into the browser.

## Running it

Requires the FRAME API running first (see `frame/api/README.md`):

```bash
# terminal 1 -- backend
sen2sr_venv/bin/uvicorn frame.api.app:app --port 8000

# terminal 2 -- frontend
cd frontend
npm install
npm run dev
```

Open the printed `http://localhost:5173` URL.

## Backend URL configuration

The frontend never hard-codes a backend host. It reads `VITE_FRAME_API_URL`
at build/dev time (see `src/api/client.ts`), defaulting to
`http://127.0.0.1:8000` if unset — the backend's own default port. To point
at a different backend:

```bash
cp .env.example .env.local
# edit .env.local: VITE_FRAME_API_URL=http://your-host:8000
```

The backend's own CORS configuration (`FRAME_API_CORS_ORIGINS`, see
`frame/api/README.md`) must list the frontend's origin — the backend's
default already includes `http://localhost:5173`.

## Supported input

Sentinel-2 L2A GeoTIFFs with exactly the four bands this prototype's
backend supports: **B04, B03, B02, B08** (order-independent), on a 128×128
pixel grid, with an embedded CRS. This is the only path the API validates
end to end — see `frame/api/README.md`'s "Scope" note. Uploading anything
else (wrong bands, wrong shape, no CRS, a non-GeoTIFF file) is rejected by
the backend with a clear message, shown inline rather than as a crash.

## User workflow

1. **Upload** a GeoTIFF — it is validated immediately (`POST /upload`, no
   SR run yet) and its metadata (dimensions, resolution, bands, CRS)
   appears as the input preview.
2. **Run FRAME** — a single explicit action that fires `POST /sr/run`.
   The pipeline-stage indicator reflects the two real HTTP requests this
   prototype actually makes (upload, then the single synchronous SR run)
   — it does not fabricate a step-by-step progress percentage, since the
   backend itself doesn't report one (preprocessing, the frozen SR model,
   and the uncertainty ensemble all execute inside that one request).
3. **Overview** tab — a comparison slider between the native 10 m input
   and the **SR-derived product — 2.5 m pixel grid**.
4. **Uncertainty** tab — the **relative model-stability uncertainty** map,
   toggleable between a standalone heatmap and an SR-image overlay whose
   opacity is modulated by the uncertainty value itself (so high-
   uncertainty regions are visually obvious, not a binary mask), plus the
   real distribution statistics from the API response.
5. **NDVI** tab — triggers `POST /analysis/ndvi` on demand, then shows the
   native NDVI, SR-derived NDVI, their absolute difference, and the
   measured comparison/uncertainty-relationship statistics — all numbers
   and rasters come from the backend; nothing is recomputed in the
   browser.
6. **Metadata** tab — the full reproducibility record (model, seed, TTA
   transforms, device, inference time, CRS, self-consistency numbers).
7. **Downloads** (sidebar, always visible once a result exists) — the SR
   GeoTIFF, uncertainty GeoTIFF, NDVI GeoTIFFs, and the raw JSON result/
   analysis, each a real file from the backend.

## Terminology / scientific caveats

This app repeats, rather than paraphrases, the backend's own fixed
language (`src/constants/terminology.ts` mirrors `frame/api/schemas.py`):

- The SR output is always **"SR-derived product — 2.5 m pixel grid"** —
  never "native 2.5 m Sentinel-2" or a "true 2.5 m image."
- The Phase 5 signal is always **"relative model-stability uncertainty"**
  — never a calibrated confidence or probability of error.
- LAM (upstream sensitivity/explainability) is never called
  "uncertainty," and is not exposed by any view in this app.
- NDVI agreement between the native and SR grids demonstrates internal
  consistency, not proof of physical accuracy — stated on the NDVI tab
  using the backend's own `scientific_caveats`.

`frame/tests/test_api_terminology.py` (backend) and
`src/__tests__/terminology.test.tsx` (frontend) both enforce these
structurally, independent of each other.

## Visualization notes

The backend returns GeoTIFFs, not preview images. This app decodes them
client-side with the `geotiff` npm package (the browser equivalent of the
backend's own `rasterio` reads) and renders them to `<canvas>` with a
percentile stretch — a display transform only, the same category as the
backend's own `normalize_for_visualization` (`frame/uncertainty/README.md`).
No NDVI, uncertainty statistic, or resampling is computed in the browser —
see `src/lib/geotiff.ts`, `src/lib/colormap.ts`, and `src/lib/raster.ts`
for exactly where that line is drawn.

## Reduced-motion behavior

The interface includes a restrained ambient satellite/starfield animation
(`src/components/atmosphere/SatelliteField.tsx`) — a couple of small
satellites drifting across the hero area, and a faint static-feeling
starfield elsewhere. It respects `prefers-reduced-motion: reduce`
(`src/hooks/useReducedMotion.ts`): with that OS-level preference set, the
canvas renders one static frame instead of running a
`requestAnimationFrame` loop, and every CSS transition/animation
elsewhere in the app collapses to near-zero duration
(`src/styles/global.css`'s `@media (prefers-reduced-motion: reduce)`
block). None of this motion ever blocks interaction — it is `aria-hidden`
and `pointer-events: none` throughout.

## Development / build commands

```bash
npm run dev         # start the Vite dev server
npm run build       # tsc -b && vite build -- production build
npm run preview     # preview the production build locally
npm run test        # run the Vitest suite once
npm run test:watch  # run Vitest in watch mode
npm run lint        # oxlint
```

## What this app deliberately does not do

- No basemap, map tiles, or AOI-driven imagery fetch — `POST /aoi/preview`
  only validates a request shape, it doesn't fetch real Sentinel-2 scenes
  (see `frame/api/README.md`); this app doesn't imply otherwise.
- No new scientific computation anywhere in the browser — every number
  and raster shown comes from a `frame/api/` response.
- No authentication, database, or job queue — this mirrors the backend's
  own prototype scope (`frame/api/README.md`'s "Prototype behavior").
