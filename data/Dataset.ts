export type DataPoint = {
  input: number[];
  target: number[];
};

export class Dataset {
  constructor(public data: DataPoint[]) {}

  shuffle(): void {
    for (let i = this.data.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.data[i], this.data[j]] = [this.data[j], this.data[i]];
    }
  }

  normalize(): void {
    const numFeatures = this.data[0].input.length;
    const means = Array(numFeatures).fill(0);
    const stds = Array(numFeatures).fill(0);

    this.data.forEach(({ input }) => {
      input.forEach((val, i) => means[i] += val);
    });
    means.forEach((_, i) => means[i] /= this.data.length);

    this.data.forEach(({ input }) => {
      input.forEach((val, i) => stds[i] += Math.pow(val - means[i], 2));
    });
    stds.forEach((_, i) => stds[i] = Math.sqrt(stds[i] / this.data.length));

    this.data = this.data.map(({ input, target }) => ({
      input: input.map((val, i) => 
        stds[i] !== 0 ? (val - means[i]) / stds[i] : 0
      ),
      target
    }));
  }

  createBatches(batchSize: number): DataPoint[][] {
    const batches = [];
    for (let i = 0; i < this.data.length; i += batchSize) {
      batches.push(this.data.slice(i, i + batchSize));
    }
    return batches;
  }
}
