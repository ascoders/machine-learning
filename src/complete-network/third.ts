import { TraningData, XYTrainingData } from '../interface';
import { NeuralNetwork } from './neural-network';

export const trainingData: XYTrainingData = [
  [1, 3],
  [2, 4],
  [3, 5],
  [4, 7.5],
  [5, 9],
  [6, 10],
  [7, 9],
  [8, 7],
  [9, 5],
  [10, 4],
  [11, 3],
  [12, 2],
  [13, 1],
  [14, 0],
];

export function init(trainingCount: number) {
  const commonTrainingData: TraningData = trainingData.map((each) => [
    [each[0]],
    [each[1]],
  ]);

  return () =>
    new NeuralNetwork({
      learningRate: 0.0001,
      maxNorm: 10,
      trainingCount,
      trainingData: commonTrainingData,
      layers: [
        { count: 1, activation: 'sigmoid', inputCount: 1 },
        { count: 4, activation: 'sigmoid' },
        { count: 1, activation: 'sigmoid' },
      ],
    });
}

// 并发神经网络数
export const concurrentNetworkCount = 1;

// 训练次数
export const traingCount = 1000;
