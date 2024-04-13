import { TraningData, TraningItem } from '../interface';
import { getRandomNumber } from '../utils';

export class NeuralNetwork {
  // 训练数据
  private trainingData: TraningData;
  private trainingCount: number;

  // 参数 b
  private b = 0;
  // 参数 w
  private w = 0;

  // 学习速率系数
  private learningRate = 0.003;

  constructor({
    trainingData,
    trainingCount,
  }: {
    trainingData: TraningData;
    trainingCount: number;
  }) {
    this.trainingData = trainingData;
    this.trainingCount = trainingCount;

    this.b = getRandomNumber();
    this.w = getRandomNumber();
  }

  private modelFunction(trainingItem: TraningItem) {
    const x = trainingItem[0][0];
    return [this.b + this.w * x];
  }

  private lossFunction(trainingItem: TraningItem) {
    const y = trainingItem[1][0];
    const cy = this.modelFunction(trainingItem);
    return Math.pow(y - cy[0], 2);
  }

  private optimization(trainingData: TraningData) {
    let currentLoss = 0;
    let gradB = 0;
    let gradW = 0;

    trainingData.forEach((trainingItem) => {
      const x = trainingItem[0][0];
      const y = trainingItem[1][0];
      const cy = this.modelFunction(trainingItem);
      const loss = this.lossFunction(trainingItem);

      currentLoss += loss;

      gradB += -2 * (y - cy[0]);
      gradW += -2 * x * (y - cy[0]);
    });

    this.b -= gradB * this.learningRate;
    this.w -= gradW * this.learningRate;

    return currentLoss;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  public fit(index: number) {
    const loss = this.optimization(this.trainingData);

    return {
      loss,
      params: {},
      type: 'function',
      f: (inputs: number[]) => this.modelFunction([inputs, []]),
    };
  }
}
