import { ActivationType } from './interface';

/* eslint-disable no-console */
export function getRandomNumber() {
  const min = 1;
  const max = -1;
  return Math.random() * (max - min) + min;
}

/** 进度提示工具 */
export class ProcessSpeaker {
  /** 共有多少个任务 */
  private count = 0;

  /** 记录 0~100 是否都有播报过 */
  private numberSet = new Set();

  constructor(count: number) {
    this.count = count;
  }

  /** 完成了第几个任务，下标从 1 开始 */
  public done(index: number) {
    // 当前百分比进度
    const currentPercentNumber = Math.floor((index / this.count) * 100);
    if (!this.numberSet.has(currentPercentNumber)) {
      this.numberSet.add(currentPercentNumber);
      console.log(`${currentPercentNumber}%`);
    }
  }
}

export function wait(time = 0) {
  return new Promise((resolve) => setTimeout(resolve, time));
}

/** 高斯函数，作为 learningRate 系数 */
export function gaussianLearnRate(x: number, max: number, learningRate: number) {
  // 定义两个阶段的分界点
  const transitionPoint = max * 0.2;

  if (x < transitionPoint) {
    // 第一阶段：快速上升到接近 max
    // 使用一个适当的函数，例如一个快速逼近的指数函数
    return (1 - Math.exp((-10 * x) / transitionPoint)) * learningRate;
  } else {
    // 第二阶段：逐渐下降至接近0
    // 使用一个逐渐减慢下降速度的指数衰减函数
    const decayRate = 5; // 衰减速率
    return (
      Math.exp((-decayRate * (x - transitionPoint)) / (max - transitionPoint)) *
      learningRate
    );
  }
}

/** sigmoid 启动函数 */
export const sigmoid = (z: number) => {
  return 1 / (1 + Math.pow(Math.E, -z));
};

const leakyReluA = 0.01;

/** LeakyReLU 启动函数 */
export const leakyRelu = (z: number) => {
  return Math.max(leakyReluA * z, z);
};

/** LeakyReLU 微分函数 */
export const leakyReluDifferential = (z: number) => {
  if (z > 0) {
    return 1;
  }
  return leakyReluA;
};

/** 根据 functionType 自动路由不同的启动函数 */
export const functionByType = (functionType: ActivationType) => (x: number) => {
  switch (functionType) {
    case 'none':
      return x;
    case 'sigmoid':
      return sigmoid(x);
    case 'leakyRelu':
      return leakyRelu(x);
  }
};

/** 根据 functionType 自动路由不同的导数函数 */
export const dFunctionByType =
  (functionType: ActivationType, activateResult: number) => (hResult: number) => {
    switch (functionType) {
      case 'none':
        return hResult;
      case 'sigmoid':
        return activateResult * (1 - activateResult);
      case 'leakyRelu':
        return leakyRelu(hResult);
    }
  };
