// core/Layer.ts
import type { ActivationFunction } from './Activation';
import { MathUtils } from '../utils/MathUtils';
import type { Optimizer } from './Optimizer';
import { LeakyReLU, Sigmoid } from './Activation';
import { Hyperparams } from './Hyperparams';

export class Layer {
  weights: number[][];
  biases: number[];
  activation: ActivationFunction;
  lastOutput?: number[];

  constructor(
    public inputSize: number,
    public outputSize: number,
    isOutputLayer: boolean = false
  ) {
    this.activation = isOutputLayer ? new Sigmoid() : new LeakyReLU();
    const scale = isOutputLayer 
      ? Math.sqrt(2 / (inputSize + outputSize)) 
      : Math.sqrt(2 / inputSize);
    
    this.weights = MathUtils.randomMatrix(outputSize, inputSize, scale);
    this.biases = new Array(outputSize).fill(0).map(() => (Math.random() - 0.5) * scale);
  }

  forward(inputs: number[]): number[] {
    const weightedSum = MathUtils.matrixVectorMult(this.weights, inputs)
      .map((sum, i) => sum + this.biases[i]);
    this.lastOutput = this.activation.forward(weightedSum);
    return this.lastOutput;
  }

  backward(gradient: number[]): number[] {
    if (!this.lastOutput) throw new Error('Must call forward before backward');
    
    const activationGrad = this.activation.backward(gradient, this.lastOutput);
    const clippedGrad = activationGrad.map(g => 
      Math.max(-Hyperparams.gradientClipValue, 
              Math.min(Hyperparams.gradientClipValue, g))
    );
    
    return MathUtils.matrixTranspose(this.weights)
      .map(row => MathUtils.dot(row, clippedGrad));
  }

  updateWeights(optimizer: Optimizer, weightGrads: number[][], biasGrads: number[]) {
    optimizer.update(this.weights, this.biases, weightGrads, biasGrads);
  }
}
