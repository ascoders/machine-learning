import '@arco-design/web-react/dist/css/arco.css';

import { Tag } from '@arco-design/web-react';
import functionPlot from 'function-plot';
import React from 'react';
import ReactDOM from 'react-dom';

import { TrainResult } from './interface';
import { init, trainingData } from './simple/first';
import { NeuralNetwork } from './simple/neural-network';
import { wait } from './utils';

const Main = () => {
  const updateTargetFn = React.useCallback((data: TrainResult) => {
    if (!domRef.current) {
      return;
    }

    functionPlot({
      target: domRef.current as any,
      width: domRef.current.clientWidth,
      height: domRef.current.clientHeight,
      yAxis: {
        domain: [
          Math.min(...trainingData.map((each) => each[1])) - 1,
          Math.max(...trainingData.map((each) => each[1])) + 1,
        ],
      },
      xAxis: {
        domain: [
          Math.min(...trainingData.map((each) => each[0])) - 1,
          Math.max(...trainingData.map((each) => each[0])) + 1,
        ],
      },
      data: [
        {
          points: trainingData,
          fnType: 'points',
          graphType: 'scatter',
          color: 'red',
        },
        ...(data === undefined
          ? []
          : data.type === 'expression'
          ? [
              {
                fn: data.expression,
                color: 'rgb(52,145,250)',
              },
            ]
          : ([
              {
                graphType: 'polyline',
                color: 'rgb(52,145,250)',
                fn: function (scope: any) {
                  return data.f([scope.x])[0];
                },
              },
            ] as any)),
      ],
    });
  }, []);

  const domRef = React.useRef<HTMLDivElement>();

  // 当前迭代到几次
  const [index, setIndex] = React.useState(0);

  // 当前最低 loss
  const [lowestLoss, setLowestLoss] = React.useState(Infinity);

  const handleRun = React.useCallback(async () => {
    // 并发神经网络数
    const concurrentNetworkCount = 1;

    // 训练次数
    const traingCount = 1000;

    // 最低的 loss
    let lowestLoss = Infinity;

    const neuralNetworks: NeuralNetwork[] = [];

    for (let i = 0; i < concurrentNetworkCount; i++) {
      neuralNetworks.push(init(traingCount)());
    }

    for (let j = 0; j < traingCount; j++) {
      for (let i = 0; i < concurrentNetworkCount; i++) {
        const trainResult = neuralNetworks[i].fit(j);

        // 如果本次 loss 是全局最低，更新 ui
        // if (trainResult.loss < lowestLoss) {
        lowestLoss = trainResult.loss;
        updateTargetFn(trainResult as any);
        setParams(trainResult.params);
        setLowestLoss(lowestLoss);

        await wait();
        // }
      }

      setIndex(j);
      await wait();
    }
  }, [updateTargetFn]);

  // 当前参数
  const [params, setParams] = React.useState<{
    [key: string]: any;
  }>({});

  React.useEffect(() => {
    updateTargetFn(undefined);
  }, [updateTargetFn]);

  return (
    <div>
      <button onClick={handleRun} style={{ marginBottom: 20 }}>
        run
      </button>

      <div style={{ marginLeft: 100, display: 'flex' }}>
        <div
          style={{ width: 300, height: 300, marginBottom: 10 }}
          ref={domRef as any}></div>
        <div style={{ marginLeft: 30 }}>
          <div style={{ marginBottom: 5 }}>
            <Tag color="blue">Training count: {index === 0 ? 0 : index + 1}</Tag>
          </div>
          <div style={{ marginBottom: 5 }}>
            <Tag color="blue">Loss: {lowestLoss}</Tag>
          </div>
          {Object.keys(params).map((key, index) => (
            <div key={index} style={{ marginBottom: 5 }}>
              <Tag color="green">
                {key}: {params[key]}
              </Tag>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

ReactDOM.render(<Main />, document.getElementById('root'));
