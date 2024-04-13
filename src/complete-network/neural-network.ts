import { Layer, NetworkStructor, TraningData, TraningItem } from '../interface';
import { dFunctionByType, functionByType as activate, getRandomNumber } from '../utils';

export class NeuralNetwork {
  // 输入长度
  private inputCount = 0;
  // 网络结构
  private networkStructor: NetworkStructor;
  // 训练数据
  private trainingData: TraningData;
  private trainingCount: number;
  // 学习速率系数
  private learningRate = 0;
  // 上一次 loss
  private lastLoss = 0;
  // 梯度裁剪阈值
  private maxNorm = Infinity;

  constructor({
    trainingData,
    layers,
    trainingCount,
    learningRate,
    maxNorm,
  }: {
    trainingData: TraningData;
    layers: Layer[];
    trainingCount: number;
    learningRate: number;
    maxNorm: number;
  }) {
    this.trainingData = trainingData;
    this.inputCount = layers[0].inputCount!;
    this.trainingCount = trainingCount;
    this.learningRate = learningRate;
    this.maxNorm = maxNorm;
    this.networkStructor = layers.map(({ activation, count }, index) => {
      const previousNeuralCount = index === 0 ? this.inputCount : layers[index - 1].count;
      return {
        activation,
        neurals: Array.from({ length: count }).map(() => ({
          value: 0,
          w: Array.from({
            length: previousNeuralCount,
          }).map(() => getRandomNumber()),
          b: getRandomNumber(),
          weightCount: 0,
          dlossByDx: 0,
          dlossByDb: 0,
          dlossByDw: Array.from({
            length: previousNeuralCount,
          }).map(() => 0),
        })),
      };
    });
  }

  /** 获取上一层神经网络各节点的值 */
  private getPreviousLayerValues(layerIndex: number, trainingItem: TraningItem) {
    if (layerIndex >= 0) {
      return this.networkStructor[layerIndex].neurals.map((neural) => neural.value);
    }
    return trainingItem[0];
  }

  private modelFunction(trainingItem: TraningItem) {
    this.networkStructor.forEach((layer, layerIndex) => {
      layer.neurals.forEach((neural) => {
        // 前置节点的值 * w 的总和
        let previousValueCountWithWeight = 0;
        this.getPreviousLayerValues(layerIndex - 1, trainingItem).forEach(
          (value, index) => {
            previousValueCountWithWeight += value * neural.w[index];
          },
        );
        neural.weightCount = previousValueCountWithWeight;
        const activateResult = activate(layer.activation)(
          previousValueCountWithWeight + neural.b,
        );
        neural.value = activateResult;
      });
    });

    // 输出最后一层网络的值
    return this.networkStructor[this.networkStructor.length - 1].neurals.map(
      (neural) => neural.value,
    );
  }

  private lossFunction(trainingItem: TraningItem) {
    // 预测值
    const xList = this.modelFunction(trainingItem);
    // 实际值
    const tList = trainingItem[1];

    const lastLayer = this.networkStructor[this.networkStructor.length - 1];
    const lastLayerNeuralCount = lastLayer.neurals.length;
    // 最后一层每一个神经元在此样本的 loss
    const lossList: number[] = Array.from({ length: lastLayerNeuralCount }).map(() => 0);
    // 最后一层每一个神经元在此样本 loss 的导数
    const dlossByDxList: number[] = Array.from({ length: lastLayerNeuralCount }).map(
      () => 0,
    );

    for (let i = 0; i < xList.length; i++) {
      // loss(x) = (x-t)²
      lossList[i] = Math.pow(tList[i] - xList[i]!, 2);
      // ∂loss/∂x = 2 * (x-t)
      dlossByDxList[i] += 2 * (xList[i]! - tList[i]);
    }

    return { lossList, dlossByDxList };
  }

  // 优化
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private optimization(trainingData: TraningData, trainingIndex: number) {
    // 当前 loss
    let currentLoss = 0;

    // 每个训练数据单独计算
    trainingData.forEach((trainingItem) => {
      const { dlossByDxList, lossList } = this.lossFunction(trainingItem);
      currentLoss += lossList.reduce((result, next) => result + next, 0);

      // 反向传播求每个参数的导数
      for (let i = this.networkStructor.length - 1; i >= 0; i--) {
        const layer = this.networkStructor[i];

        layer.neurals.forEach((neural, neuralIndex) => {
          // f(x) = sigmoid(g(x))
          // g(x) = wᵢ * xᵢ + b

          // 求出该节点的 dloss/dx 并保存
          if (i === this.networkStructor.length - 1) {
            // 输出层，就是 dlossByDx
            neural.dlossByDx = dlossByDxList[neuralIndex];
          } else {
            // 非输出层，是下一层的神经元 (q规则) 加和
            // q规则 是该层 dloss/dx * dx/dx' * w
            neural.dlossByDx = 0;
            const nextLayer = this.networkStructor[i + 1];
            nextLayer.neurals.forEach((nextNeural) => {
              neural.dlossByDx +=
                nextNeural.dlossByDx *
                dFunctionByType(
                  nextLayer.activation,
                  nextNeural.value,
                )(nextNeural.value) *
                nextNeural.w[neuralIndex];
            });
          }

          // 求 dloss/db = dloss/dx * dx/db
          // 其中 dx/db = 1
          neural.dlossByDb += neural.dlossByDx;

          // 求每个 dloss/dwi = dloss/dx * dx/dwi
          // 其中 dx/dwi = 前一个对应神经元的输出 x
          neural.w.forEach((w, wi) => {
            neural.dlossByDw[wi] +=
              neural.dlossByDx * this.getPreviousLayerValues(i - 1, trainingItem)[wi];
          });
        });
      }
    });

    // const learningRate =
    //   this.learningRate *
    //   gaussianLearnRate(trainingIndex, this.trainingCount, this.learningRate);

    // 根据计算结果，更新每层节点的参数
    for (let i = this.networkStructor.length - 1; i >= 0; i--) {
      const layer = this.networkStructor[i];

      layer.neurals.forEach((neural) => {
        // 更新参数 b
        const dbMean = neural.dlossByDb / trainingData.length;
        neural.b += this.applyMaxNorm(-dbMean * this.learningRate);
        neural.dlossByDb = 0;

        // 更新参数 w
        neural.w.forEach((w, wi) => {
          const dwMean = neural.dlossByDw[wi] / trainingData.length;
          neural.w[wi] += this.applyMaxNorm(-dwMean * this.learningRate);
          neural.dlossByDw[wi] = 0;
        });
      });
    }

    return currentLoss;
  }

  // 应用梯度裁剪
  private applyMaxNorm(dValue: number) {
    if (dValue >= 0) {
      if (dValue <= this.maxNorm) {
        return dValue;
      } else {
        return this.maxNorm;
      }
    } else {
      if (dValue >= -this.maxNorm) {
        return dValue;
      } else {
        return -this.maxNorm;
      }
    }
  }

  public fit(index: number) {
    const loss = this.optimization(this.trainingData, index);

    if (index === 0) {
      this.lastLoss = loss;
    } else {
      // 500 次之后速率就不要变了
      if (index < 500) {
        // 每训练 n 次，调整一次学习速率
        if (index % 10 === 0) {
          // loss 不变也要增加学习速率
          if (loss <= this.lastLoss) {
            this.learningRate *= 1.1;
          } else {
            // loss 变大后要显著降低学习速率
            this.learningRate *= 0.8;
          }

          this.lastLoss = loss;
        }
      }
    }

    return {
      loss,
      params: {},
      type: 'function',
      f: (inputs: number[]) => this.modelFunction([inputs, []]),
    };
  }
}
