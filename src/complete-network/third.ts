import { TraningData, XYTrainingData } from '../interface';
import { NeuralNetwork } from './neural-network';

export const trainingData: XYTrainingData = [
  [1, 3],
  [2, 4],
  [3, 5],
  [4, 6.5],
  [5, 8],
  [6, 9.5],
  [7, 9],
  [8, 8],
  [9, 7],
  [10, 5.5],
  [11, 4],
  [12, 3],
  [13, 2],
  [14, 1],
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
        { count: 4, activation: 'sigmoid', inputCount: 1 },
        { count: 4, activation: 'sigmoid' },
        { count: 1, activation: 'leakyRelu' },
      ],
    });
}

// 并发神经网络数
export const concurrentNetworkCount = 1000;

// 训练次数
export const traingCount = 100000;
