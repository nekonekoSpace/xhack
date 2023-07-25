
# openvino21.2.185を用いてdocker上でkerasモデルを推定
先ほど作った環境上で自身が作ったプログラムをCPU・VPUで実行する。


##  docker環境の構築

ご自身が作られた`pb`形式モデルをOpenvinoで利用可能なIR形式に変換するための環境を構築します。


### 環境
 - Ubuntu 18.04
 - Docker Compose version v2.6.1
 - NCS2 (VPU)


### ディレクトリ構成

    openvino_2021
    ├── docker-compose.yml
    ├── Dockerfile
    └── workspace
        ├── data
        └── model
            ├── pb
            └── IR





### コンテナ立ち上げまでの手順

1.  ディレクトリ構成を基本にして必要なファイルを作成する
　
    - Dockerfile
        
        ```Dockerfile
        FROM spacecloud.unibap.com/unibap/framework-baseimage:latest
        SHELL ["/bin/bash", "-c"]
        RUN  apt-get update  
        RUN  apt-get install sudo  
        RUN python3.6 -m pip install setuptools

        WORKDIR  /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites
        RUN bash ./install_prerequisites_tf2.sh  
        ```


    - docker-compose.yml

        ```Dockerfile
        version: "3.2"
            services:
            ubuntu:
                cpuset : "0"        #dokcer から見えるCPUの番号 ex) "0,1,3"
                build:
                context: .
                dockerfile: Dockerfile
                container_name: "your_container_name"
                device_cgroup_rules:
                - 'c 189:* rmw'
                volumes:
                - /dev/bus/usb:/dev/bus/usb   
                - ./workspace:/workspace
                tty: true
        ```
    - workspace
        
        ドッカー環境とマウントして操作を行う
        
        


1. Docker imageを作成

    `./openvino_2021`で
    ```bash
    $ docker compose build 
     [+] Building 0.1s (5/5) FINISHED                                                                                          
    => [internal] load .dockerignore                                                                                                                                                                                          0.0s
    => => transferring context: 2B                                                                                                                                                                                            0.0s
    => [internal] load build definition from Dockerfile                                                                                                                                                                       0.0s
    => => transferring dockerfile: 126B                                                                                                                                                                                       0.0s
    => [internal] load metadata for spacecloud.unibap.com/unibap/framework-baseimage:latest                                                                                                                                   0.0s
    => CACHED [1/1] FROM spacecloud.unibap.com/unibap/framework-baseimage:latest                                                                                                                                              0.0s
    => exporting to image                                                                                                                                                                                                     0.0s
    => => exporting layers                                                                                                                                                                                                    0.0s
    => => writing image sha256:a842a0629c502410d5193ed9ca1f9722a4e2349382b6e3cf3d6c98343dbbe159                                                                                                                               0.0s
    => => naming to docker.io/library/your_docker_image:latest                                                                                                                                                                0.0s
    ```



1. コンテナを立ち上げる

    ```bash
    $ docker compose up 
    [+] Running 1/0
    ⠿ Container your_container_name  Created                                                                                                                                                                                  0.0s
    Attaching to your_container_name
    your_container_name  | [setupvars.sh] OpenVINO environment initialized
    ```

1. 新しいターミナルで立ち上げたdocker環境に入る
    ```bash
    $ docker compose exec openvino_2021 bash
    [setupvars.sh] OpenVINO environment initialized
    root@32cbe7171e82:/# 
    ```
    もしくは
    ```bash
    $ docker exec -it your_container_name bash
    [setupvars.sh] OpenVINO environment initialized
    root@32cbe7171e82:/# 
    ```


### 作られるdockerの環境
    
- openvino: 2021.2.185
- python: 3.6.9
- tensorflow: 2.6.2



### pbモデルをIRモデルに変換する


`./workspace/model/pb`に作った`model.pb`を入れる
その後以下のコマンドを実行してpbのモデルをIR形式に変換する

```bash
$ cd  /opt/intel/openvino_2021/deployment_tools/model_optimizer/
$ python3 mo.py --input_model "/workspace/model/pb/model.pb" --output_dir "/workspace/model/IR"  --input_shape [1,80,80,3]

Model Optimizer arguments:
Common parameters:
        - Path to the Input Model:      /workspace/model/pb/model.pb
        - Path for generated IR:        /workspace/model/IR
        - IR output name:       model
        - Log level:    ERROR
        - Batch:        Not specified, inherited from the model
        - Input layers:         Not specified, inherited from the model
        - Output layers:        Not specified, inherited from the model
        - Input shapes:         [1,80,80,3]
        - Mean values:  Not specified
        - Scale values:         Not specified
        - Scale factor:         Not specified
        - Precision of IR:      FP32
        - Enable fusing:        True
        - Enable grouped convolutions fusing:   True
        - Move mean values to preprocess section:       None
        - Reverse input channels:       False
TensorFlow specific parameters:
        - Input model in text protobuf format:  False
        - Path to model dump for TensorBoard:   None
        - List of shared libraries with TensorFlow custom layers implementation:        None
        - Update the configuration file with input/output node names:   None
        - Use configuration file used to generate the model with Object Detection API:  None
        - Use the config file:  None
Model Optimizer version:        2021.2.0-1877-176bdf51370-releases/2021/2
2023-07-14 02:49:21.590357: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/intel/openvino/opencv/lib:/opt/intel/openvino/deployment_tools/ngraph/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/hddl/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/gna/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/mkltiny_lnx/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib:/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64
2023-07-14 02:49:21.590402: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

[ SUCCESS ] Generated IR version 10 model.
[ SUCCESS ] XML file: /workspace/model/IR/model.xml
[ SUCCESS ] BIN file: /workspace/model/IR/model.bin
[ SUCCESS ] Total execution time: 3.88 seconds. 
[ SUCCESS ] Memory consumed: 307 MB. 
It's been a while, check for a new version of Intel(R) Distribution of OpenVINO(TM) toolkit here https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html?cid=other&source=Prod&campid=ww_2021_bu_IOTG&content=upg_pro&medium=organic_uid_agjj or on the GitHub*
```

`workspace/model/IR/`に
- `model.bin`
- `model.xml`
- `model.mapping`

の3つのフォルダが生成される


この時オプションとして　
```--data_type 'FP16', 'FP32', 'half', 'float'``` を選ぶことができる
（デフォルトはFP32）



## 推論


### ディレクトリ構造
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


### VPUの確認
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


### VPUを使って自身で作ったモデルの性能を評価

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


```
E: [global] [    124159] [python3] XLink_sem_wait:94     XLink_sem_inc(sem) method call failed with an error: -1
E: [global] [    124159] [python3] XLinkResetRemote:257 can't wait dispatcherClosedSem
```


