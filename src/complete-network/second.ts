import { TraningData, XYTrainingData } from '../interface';
import { NeuralNetwork } from './neural-network';

export const trainingData: XYTrainingData = [
  [1, 1],
  [2, 3],
  [3, 7],
  [4, 9],
  [5, 10],
  [6, 11],
  [7, 13],
  [8, 16],
  [9, 20],
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
      layers: [
        { count: 3, activation: 'leakyRelu', inputCount: 1 },
        { count: 3, activation: 'leakyRelu' },
        { count: 1, activation: 'leakyRelu' },
      ],
    });
}
