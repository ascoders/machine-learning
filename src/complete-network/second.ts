import { TraningData, XYTrainingData } from '../interface';
import { NeuralNetwork } from './neural-network';

export const trainingData: XYTrainingData = [
  [1, 3.2],
  [2, 7],
  [3, 8],
  [4, 11.2],
  [5, 15.3],
];

export function init(trainingCount: number) {
  const commonTrainingData: TraningData = trainingData.map((each) => [
    [each[0]],
    [each[1]],
  ]);

  return () =>
    new NeuralNetwork({
      learningRate: 0.0001,
      maxNorm: 1,
      trainingCount,
      trainingData: commonTrainingData,
      layers: [
        { count: 5, activation: 'leakyRelu', inputCount: 1 },
        { count: 1, activation: 'leakyRelu' },
      ],
    });
}

// 并发神经网络数
export const concurrentNetworkCount = 1000;

// 训练次数
export const traingCount = 100000;
