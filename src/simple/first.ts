import { TraningData, XYTrainingData } from '../interface';
import { NeuralNetwork } from './neural-network';

// y = 3x
export const trainingData: XYTrainingData = [
  [1, 3],
  [2, 6],
  [3, 9],
  [4, 12],
  [5, 15],
];

export function init(trainingCount: number) {
  const commonTrainingData: TraningData = trainingData.map((each) => [
    [each[0]],
    [each[1]],
  ]);

  return () =>
    new NeuralNetwork({
      trainingCount,
      trainingData: commonTrainingData,
    });
}
