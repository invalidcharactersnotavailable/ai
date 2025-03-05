export const Hyperparams = {
    learningRate: 0.001,
    batchSize: 2,
    epochs: 2000,
    hiddenSize: 4,
    inputSize: 2,
    outputSize: 1,
    adamBeta1: 0.9,
    adamBeta2: 0.999,
    adamEpsilon: 1e-8,
    gradientClipRange: 1.0,
    initializationScale: 0.1
} as const;
  