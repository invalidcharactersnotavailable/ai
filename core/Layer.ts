import { ActivationFunction } from './Activation';
import { MathUtils } from '../utils/MathUtils';
import { Optimizer } from './Optimizer';
import { ReLU, Sigmoid } from './Activation';

export class Layer {
  weights: number[][];
  biases: number[];
  activation: ActivationFunction;

  constructor(
    public inputSize: number,
    public outputSize: number,
    activation: ActivationFunction = new Sigmoid()
  ) {
    // He initialization for ReLU
    const scale = activation instanceof ReLU ? Math.sqrt(2 / inputSize) : 0.1;
    this.weights = MathUtils.randomMatrix(outputSize, inputSize, scale);
    this.biases = new Array(outputSize).fill(0);
    this.activation = activation;
  }

  forward(inputs: number[]): number[] {
    const weightedSum = MathUtils.matrixVectorMult(this.weights, inputs)
      .map((sum, i) => sum + this.biases[i]);
    return this.activation.forward(weightedSum);
  }

  backward(gradient: number[]): number[] {
    const activationGrad = this.activation.backward(gradient);
    return MathUtils.matrixTranspose(this.weights)
      .map((row: number[]) => MathUtils.dot(row, activationGrad));
  }

  updateWeights(optimizer: Optimizer) {
    optimizer.update(this.weights, this.biases);
  }
}
