// core/Hyperparams.ts
export const Hyperparams = {
  learningRate: 0.001,
  batchSize: 4,
  epochs: 10000,
  hiddenSize: 4,
  inputSize: 2,
  outputSize: 1,
  adamBeta1: 0.9,
  adamBeta2: 0.999,
  adamEpsilon: 1e-8,
  gradientClipValue: 1.0,
  leakyReluAlpha: 0.01,
  initializationScale: 0.01,
  epsilon: 1e-8
} as const;
