export abstract class LossFunction {
  abstract calculate(predicted: number[], actual: number[]): number;
  abstract gradient(predicted: number[], actual: number[]): number[];
}

export class MSELoss extends LossFunction {
  calculate(predicted: number[], actual: number[]): number {
    return predicted.reduce((sum, p, i) => 
      sum + Math.pow(p - actual[i], 2), 0) / predicted.length;
  }

  gradient(predicted: number[], actual: number[]): number[] {
    return predicted.map((p, i) => 
      2 * (p - actual[i]) / predicted.length
    );
  }
}

export class CrossEntropyLoss extends LossFunction {
  calculate(predicted: number[], actual: number[]): number {
    const eps = 1e-15;
    return -actual.reduce((sum, a, i) => 
      sum + a * Math.log(Math.min(Math.max(predicted[i], eps), 1 - eps))
    , 0);
  }

  gradient(predicted: number[], actual: number[]): number[] {
    return predicted.map((p, i) => p - actual[i]);
  }
}
