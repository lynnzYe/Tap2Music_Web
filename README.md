# Readme

1. Install dependencies:
   `npm install`
2. Run the app:
   `npm run dev`

## How to add new models?

1. To add a model, modify `InferenceEngine.ts` for API update, `lstm.js` and `tap2music.js` for model implementation/test/wrapper.
2. Then, update `App.tsx` to add new modes for selection. Remember to run test only once for each mode and dispose unused models. Prepare input accordingly at `engineRefcurrent.run()`
