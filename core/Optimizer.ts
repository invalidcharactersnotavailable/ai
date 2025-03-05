// core/Optimizer.ts
import { Hyperparams } from "./Hyperparams";

export abstract class Optimizer {
  abstract update(
    weights: number[][],
    biases: number[],
    weightGrads: number[][],
    biasGrads: number[]
  ): void;
  learningRate: number = Hyperparams.learningRate;
}

export class Adam extends Optimizer {
  private mWeights: number[][] = [];
  private vWeights: number[][] = [];
  private mBiases: number[] = [];
  private vBiases: number[] = [];
  private t = 0;

  constructor(
    public learningRate: number = Hyperparams.learningRate,
    private beta1: number = Hyperparams.adamBeta1,
    private beta2: number = Hyperparams.adamBeta2,
    private epsilon: number = Hyperparams.adamEpsilon
  ) {
    super();
  }

  update(
    weights: number[][],
    biases: number[],
    weightGrads: number[][],
    biasGrads: number[]
  ): void {
    this.t++;

    if (this.mWeights.length === 0) {
      this.mWeights = weights.map(row => row.map(() => 0));
      this.vWeights = weights.map(row => row.map(() => 0));
      this.mBiases = biases.map(() => 0);
      this.vBiases = biases.map(() => 0);
    }

    // Update weights
    weights.forEach((row, i) => {
      row.forEach((_, j) => {
        const g = Math.max(-5, Math.min(5, weightGrads[i][j]));
        this.mWeights[i][j] = this.beta1 * this.mWeights[i][j] + (1 - this.beta1) * g;
        this.vWeights[i][j] = this.beta2 * this.vWeights[i][j] + (1 - this.beta2) * g ** 2;
        
        const mHat = this.mWeights[i][j] / (1 - Math.pow(this.beta1, this.t));
        const vHat = this.vWeights[i][j] / (1 - Math.pow(this.beta2, this.t));
        
        weights[i][j] -= this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
      });
    });

    // Update biases
    biases.forEach((_, i) => {
      const g = Math.max(-5, Math.min(5, biasGrads[i]));
      this.mBiases[i] = this.beta1 * this.mBiases[i] + (1 - this.beta1) * g;
      this.vBiases[i] = this.beta2 * this.vBiases[i] + (1 - this.beta2) * g ** 2;
      
      const mHat = this.mBiases[i] / (1 - Math.pow(this.beta1, this.t));
      const vHat = this.vBiases[i] / (1 - Math.pow(this.beta2, this.t));
      
      biases[i] -= this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
    });
  }
}
