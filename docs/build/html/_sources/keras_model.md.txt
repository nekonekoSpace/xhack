# Docker環境におけるkerasを用いたモデル構築
tensorflow.kerasが使えるdocker環境を作り、データセットを使って船舶識別を行うモデル作成まで行います。

ご自身でモデル構築ができる人は飛ばしてください。
 

## 現状の環境

- ubuntu: 18.04
-  Driver Version: 515.105.01
-  CUDA Version: 11.7  

```
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA TITAN V      On   | 00000000:06:00.0 Off |                  N/A |
| 32%   47C    P8    28W / 250W |     25MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA TITAN V      On   | 00000000:0A:00.0 Off |                  N/A |
| 28%   39C    P8    25W / 250W |      5MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2177      G   /usr/lib/xorg/Xorg                  9MiB |
|    0   N/A  N/A      2500      G   /usr/bin/gnome-shell               14MiB |
|    1   N/A  N/A      2177      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
```



## 出来上がる環境    
- tensorflow 2.6.0
- python3.6.9


## nvidia-container-toolkitの導入
すでに導入されている場合は飛ばして良い。
```
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$(. /etc/os-release;echo $ID$VERSION_ID)/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt update
$ sudo apt -y install nvidia-container-toolkit
```
以下のコマンドが実行出来れば完了
```
$ nvidia-container-cli info
NVRM version:   515.105.01
CUDA version:   11.7

Device Index:   0
Device Minor:   1
Model:          NVIDIA TITAN V
Brand:          TITAN
GPU UUID:       GPU-8a3843c1-8daf-fa8c-a6c1-12a24c75a6b9
Bus Location:   00000000:06:00.0
Architecture:   7.0

Device Index:   1
Device Minor:   0
Model:          NVIDIA TITAN V
Brand:          TITAN
GPU UUID:       GPU-8480bc57-d009-1175-2d29-b8ad2b4523f6
Bus Location:   00000000:0a:00.0
Architecture:   7.0

```
一度dockerを再起動しておく
```
$ sudo systemctl restart docker
```
コンテナを立ち上げてエラーが出なければ問題なし
```
$ docker run --rm --gpus all nvidia/cuda:11.0.3-base nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA TITAN V      On   | 00000000:06:00.0 Off |                  N/A |
| 32%   47C    P8    28W / 250W |     25MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA TITAN V      On   | 00000000:0A:00.0 Off |                  N/A |
| 28%   39C    P8    25W / 250W |      5MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

エラーが出るようであれば、自身のcudaのバージョンを確認してみる



## docker 環境の構築

    ./
    ├── workspace
    │   ├── 
    │   │   ├── x_test.npy
    │   │   └── y_test.npy
    │   ├── 
    │   └── 
    ├── Docker
    └── docker-compose.yml

- docker-compose.yml
    ```Docker
    version: "3.2"
    services:
    tensorflow_keras:
        build:
        context: .
        dockerfile: Dockerfile
        image: "tensorflwo-keras26"
        container_name: "your_container_name"
        volumes:
        - ./workspace:/workspace
        ports:
        - 8888:8888
        deploy:
        resources:
            reservations:
            devices:
                - driver: nvidia
                count: 1
                capabilities: [gpu]
        environment:
        - NVIDIA_VISIBLE_DEVICES=all
        tty: true
    ```
- Docker
    ```Docker
    FROM tensorflow/tensorflow:2.6.0-gpu-jupyter
    SHELL ["/bin/bash", "-c"]
    ```


## データセットの取得

今回はkaggleで利用されているデータセットを用いて学習を行う。

https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imager

このデータセットは80 $\times$ 80ピクセルの4000枚のRGB画像が含まれており、それらは船か船以外かのラベルを持っている。
詳しくはコンテストページを参照していただきたい。


データセットをダウンロードをダウンロードすると`/shipsnet/shipsnet`に画像データが存在しており、`./shipsnet.json`に数値データが入っているので、使いやすいようにnpyデータとラベルのセットを作成する。


- make_dataset.py

    ```python 
    import pandas  as pd 
    import json 
    import numpy as np 
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split


    with open('shipsnet.json') as data_file:
        dataset = json.load(data_file)
    shipsnet= pd.DataFrame(dataset)
    shipsnet.head()

    shipsnet = shipsnet[["data", "labels"]]
    x = np.array(dataset['data']).astype('uint8')
    y = np.array(dataset['labels']).astype('uint8')
    x_reshaped = x.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
    y_reshaped = to_categorical(y, num_classes=2)
    x_reshaped = x_reshaped / 255
    x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y_reshaped,
                                                            test_size = 0.20, random_state = 42)
    #学習データとテストデータを8：2で分割

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                    test_size = 0.25, random_state = 42)
    #学習データのうちの25％を検証データとして利用
    np.save("out/x_train",x_train)
    np.save("out/y_train",y_train)
    np.save("out/x_val",x_val)
    np.save("out/y_val",y_val)
    np.save("out/y_test",y_test)
    np.save("out/x_test",x_test)
    ```
    実行すると./outに学習データ、検証データ、テストデータが生成されるのでworkspaceに展開する。
    今後はこれらのデータを使って学習と推論を進める。


## モデルの構築

今回のモデルでは単純な3層CNNを作成する。外観は以下のようになる。
![cnn-fig](./fig/cnn-fig.png)

以下コード

- model.py

    ```python

    import os 
    import numpy as np 


    from sklearn.metrics import classification_report,accuracy_score
    from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense
    from tensorflow.keras import Model
    from tensorflow.keras import callbacks
    from tensorflow.keras.models import load_model

    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    import time



    model_save = True
    model_name = "cnn-model.h5"

    def convert_h5_to_pb():
        tf.keras.backend.clear_session()
        save_pb_dir = './pb'
        model_fname = f'./keras/{model_name}'
        model = load_model(model_fname, compile=False)     
        file,ext = os.path.splitext(model_name)
        # model.save(f"{save_pb_dir}/{file}")

        #もしtensorflowがv1だとfalseになる
        print(tf.executing_eagerly())  

        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(tf.TensorSpec(model.input_shape, tf.float32, name="input_L"))
        # Get frozen ConcreteFunction
        concrete_function = convert_variables_to_constants_v2(full_model)
        concrete_function.graph.as_graph_def()
        layers = [op.name for op in concrete_function.graph.get_operations()]

        #モデルを固定する(frozen modelに変換する)
        frozen_model = convert_variables_to_constants_v2(concrete_function)
        tf.io.write_graph(frozen_model.graph, save_pb_dir, "cnn-model.pb", as_text=False)



    def main():
        x_test = np.load('./out/x_test.npy')
        x_train = np.load('./out/x_train.npy')
        x_val = np.load('./out/x_val.npy')
        y_test = np.load('./out/y_test.npy')
        y_train = np.load('./out/y_train.npy')
        y_val= np.load('./out/y_val.npy')


        if os.path.exists(os.path.join("./keras",model_name)):
            print("tuning .....")
            model = load_model(os.path.join("./keras",model_name))
            model.summary()
        else:
            print(f'load model:{model_name}')
            model =tuning(x_train,x_val,y_train,y_val)  

        s = time.perf_counter()
        pred = model.predict(x_test)
        print(f'time:{time.perf_counter()-s}')
        pred = np.argmax(pred,axis=1)
        y_test = np.argmax(y_test,axis=1)
        ac_score = accuracy_score(pred, y_test)
        print(ac_score)
        convert_h5_to_pb()

        return 


    def tuning(x_train,x_val,y_train,y_val):
        #自作モデルはここを書き換え

        inputs = Input(shape=(80,80,3), name='input_L')
        x=  Conv2D(filters=64,kernel_size=(4,4),padding='same',activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(5,5))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu')(x)
        x = MaxPooling2D(pool_size=(3,3),strides=(1,1))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(filters=16,kernel_size=(2,2),padding='same',activation='relu')(x)
        x = MaxPooling2D(pool_size=(3,3),strides=(1,1))(x)
        x = Dropout(0.25)(x)
        x =  Flatten()(x)
        x = Dense(200,activation='relu',name ='hidden200')(x)
        x = Dropout(0.5)(x)
        x = Dense(100,activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(100,activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(50,activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(2, activation='softmax', name='output_L')(x)
        model = Model(inputs=inputs,outputs=predictions)
        model.compile(optimizer='adam',
                    loss = 'categorical_crossentropy',
                    metrics=["accuracy"])
        model.summary()
        earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                                mode ="min", patience = 10, 
                                                restore_best_weights = True)
        hist    = model.fit(x_train,y_train,validation_data=(x_val, y_val),epochs=100,callbacks=[earlystopping])
        if model_save:
            model.save(f"./keras/{model_name}")

        return model 
        

    if __name__=="__main__":
        main()

    ```

プログラムを実行すると`./pb`と`./keras`にモデルが出力される。

`h5`形式から`pb`形式に変換する際にエラーが出る場合はtensorflowのバージョンなどを確認してみてください。







次には`./pb`に出力された`cnn-model.pb`を