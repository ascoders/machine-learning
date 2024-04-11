/** 最简单的 x y 函数训练数据结构 */
export type XYTrainingData = Array<[number, number]>;

/** 任意输入输出数量的训练结构 */
export type TraningData = Array<[number[], number[]]>;

/** 任意输入输出数量的训练结构，单项 */
export type TraningItem = [number[], number[]];

/** 训练结果 */
export type TrainResult =
  | {
      type: 'expression';
      /** 函数表达式 */
      expression: string;
      /** 当前参数 */
      params: {
        [key: string]: any;
      };
      /** loss 值 */
      loss: number;
    }
  | {
      type: 'function';
      /** 函数体 */
      f: (...args: any[]) => any;
      /** 当前参数 */
      params: {
        [key: string]: any;
      };
      /** loss 值 */
      loss: number;
    }
  | undefined;

/** 生成训练函数 */
export type TrainFunction = (
  /** 训练次数，必须从 0 开始 */
  index: number,
) => TrainResult;

export interface Layer {
  count: number;
  activation: ActivationType;
  inputCount?: number; // 输入长度，仅第一层需要
}

/** 启动函数类型 */
export type ActivationType = 'sigmoid' | 'leakyRelu' | 'none';

/** 神经网络结构数据 */
export type NetworkStructor = Array<{
  // 启动函数类型
  activation: ActivationType;
  // 节点
  neurals: Neural[];
}>;

export interface Neural {
  /** 当前该节点的值 */
  value: number;
  /** 上一层每个节点连接到该节点乘以的系数 w */
  w: Array<number>;
  /** 该节点的常数系数 b */
  b: number;
  /** 该节点前置节点 w * x 加和的值 */
  weightCount: number;
  dlossByDx: number;
  dlossByDb: number;
  dlossByDw: Array<number>;
}
