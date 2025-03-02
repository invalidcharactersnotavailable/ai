export const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
export const dsigmoid = (y: number): number => y * (1 - y);
