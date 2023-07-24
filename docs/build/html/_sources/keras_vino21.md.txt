
# openvino21.2.185を用いてdocker上でkerasモデルを推定
先ほど作った環境上で自身が作ったプログラムをCPU・VPUで実行する。


## 環境
    workspace
    ├── inference
    │   ├── data
    │   │   ├── x_test.npy
    │   │   └── y_test.npy
    │   ├── hardware.py
    │   └── run.py
    ├── install.sh
    └── model
        ├── IR
        │   ├── model.bin
        │   ├── model.mapping
        │   └── model.xml
        └── pb
            └── model.pb

新たにinferenceディレクトリを追加する。
- `data` 推論を行うデータを格納するディレクトリ
- `run.py`　実行のためのpythonファイル　

    ```python
    from sklearn.metrics import classification_report,accuracy_score
    import numpy as np
    from openvino.inference_engine import IECore
    import time


    #device = "CPU"
    device = "MYRIAD"

    def main():
        ir_dir = "/workspace/model/IR/"
        model_name = "model"    
        ir_path = f"{ir_dir}{model_name}.xml"
        weight_path = f"{ir_dir}{model_name}.bin"

        ie = IECore()
        net = ie.read_network(model=ir_path, weights=weight_path)
        exec_net = ie.load_network(network=net, device_name=device, num_requests=0)

        x_test = np.load('.//data/x_test.npy')
        y_test = np.load('./data/y_test.npy')
        y_test = np.argmax(y_test,axis=1)

        cal(x_test,y_test,net,exec_net)


    def cal(x_test,y_test,net,exec_net):

        result = []
        start = time.perf_counter()
        for i in range(len(y_test)):
            img = x_test[i,:,:,:]
            img = img.transpose(2,0,1)
            img = np.expand_dims(img, axis=0)

            out = exec_net.requests[0].async_infer({next(iter(net.input_info)): img})
            for i in range(99999):
                infer_status = exec_net.requests[0].wait(0)
                if infer_status == 0:
                    break
                else:
                    time.sleep(0.001)

            out = exec_net.requests[0].output_blobs[next(iter(net.outputs))].buffer
            out = np.squeeze(out)
            out = np.argmax(out,axis=0)
            result.append(out)
        print(f"time is {time.perf_counter()-start} with {device};")
        ac_score = accuracy_score(result, y_test)
        print(f"score is :{ac_score}")


    if __name__=='__main__':
        main()
    ```

必要に応じてライブラリを追加
```python
 python3 -m pip install scikit-learn
```


## VPUの確認
以下のファイルを作成


hardware.py
```python
from openvino.inference_engine import IECore
ie = IECore()
devices = ie.available_devices
print(devices)
```

作成したファイルを実行

``` bash
$ python3 hardware.py 
[E:] [BSL] found 0 ioexpander device
['CPU', 'GNA', 'MYRIAD']
```

`MYRIAD`があればVPUが使用できる状態になっている。


## VPUを使って自身で作ったモデルの性能を評価

```bash
$ cd /workspace/inference
$ python3 run.py 
time is 3.9357731910422444 with MYRIAD;
score is :0.99125
```
となればMYRIADを使って実行が出来ている。

`device = CPU` とすればCPUでの推論となる。

```bash
$ python3 run.py 
time is 1.077825287124142 with CPU;
score is :0.99125
```

docker-compose.ymlで指定した有効CPUの数が1の時、以下のようなエラーが出る場合があるが今のところ問題ない。


```bash
E: [global] [    124159] [python3] XLink_sem_wait:94     XLink_sem_inc(sem) method call failed with an error: -1
E: [global] [    124159] [python3] XLinkResetRemote:257 can't wait dispatcherClosedSem
```


