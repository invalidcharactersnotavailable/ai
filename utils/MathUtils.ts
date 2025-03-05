export class MathUtils {
  static randomMatrix(rows: number, cols: number, scale: number = 1): number[][] {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
    );
  }

  static matrixVectorMult(matrix: number[][], vector: number[]): number[] {
    return matrix.map(row => this.dot(row, vector));
  }

  static dot(a: number[], b: number[]): number {
    return a.reduce((sum: number, val: number, i: number) => sum + val * b[i], 0);
  }

  static matrixTranspose(matrix: number[][]): number[][] {
    return matrix[0].map((_, i) => matrix.map(row => row[i]));
  }
}
