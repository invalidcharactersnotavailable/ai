// data/Split.ts
export class Split {
    static trainTestSplit(data: DataPoint[], testSize: number = 0.2): [DataPoint[], DataPoint[]] {
      const shuffled = [...data].sort(() => Math.random() - 0.5);
      const splitIdx = Math.floor(data.length * (1 - testSize));
      return [
        shuffled.slice(0, splitIdx),
        shuffled.slice(splitIdx)
      ];
    }
  }
  