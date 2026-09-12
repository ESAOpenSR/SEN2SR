import '@testing-library/jest-dom/vitest'

// jsdom does not implement matchMedia -- provide a default (no-preference)
// stub so components calling `useReducedMotion` don't throw during tests.
// Individual tests override `window.matchMedia` to exercise the
// reduced-motion branch explicitly.
if (!window.matchMedia) {
  window.matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })
}

// jsdom's canvas 2D context is not implemented -- raster/atmosphere
// components call getContext('2d') to draw; stub just enough of the API
// surface so mounting them in tests doesn't throw. No test asserts on
// actual pixel output (that's a browser-only concern, verified manually).
if (!HTMLCanvasElement.prototype.getContext) {
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext
}
