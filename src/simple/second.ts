import { TraningData, XYTrainingData } from '../interface';
import { NeuralNetwork } from './neural-network';

// y = 近似 3x
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
      trainingCount,
      trainingData: commonTrainingData,
    });
}

// 并发神经网络数
export const concurrentNetworkCount = 1;

// 训练次数
export const traingCount = 1000;
