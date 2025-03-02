export class Matrix {
    rows: number;
    cols: number;
    data: number[][];
    
    constructor(rows: number, cols: number) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(rows).fill(0).map(() => Array(cols).fill(0));
    }
    
    static fromArray(arr: number[]): Matrix {
        return new Matrix(arr.length, 1).map((_, i) => arr[i]);
    }
    
    static subtract(a: Matrix, b: Matrix): Matrix {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error('Matrices must have the same dimensions for subtraction');
        }
        return new Matrix(a.rows, a.cols).map((_, i, j) => a.data[i][j] - b.data[i][j]);
    }
    
    map(func: (val: number, i: number, j: number) => number): Matrix {
        return new Matrix(this.rows, this.cols).map((_, i, j) => func(this.data[i][j], i, j));
    }
    
    multiply(n: number | Matrix): Matrix {
        if (n instanceof Matrix) {
            if (this.cols !== n.rows) {
            throw new Error('Columns of A must match rows of B');
            }
            return new Matrix(this.rows, n.cols).map((_, i, j) => {
            let sum = 0;
            for (let k = 0; k < this.cols; k++) {
                sum += this.data[i][k] * n.data[k][j];
            }
            return sum;
            });
        } else {
            return this.map(val => val * n);
        }
    }
    
    add(n: number | Matrix): Matrix {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
            throw new Error('Matrices must have the same dimensions for addition');
            }
            return this.map((val, i, j) => val + n.data[i][j]);
        } else {
            return this.map(val => val + n);
        }
    }
    
    transpose(): Matrix {
        return new Matrix(this.cols, this.rows).map((_, i, j) => this.data[j][i]);
    }
    
    toArray(): number[] {
        return this.data.flat();
    }
}
