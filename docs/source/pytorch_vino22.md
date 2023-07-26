# openvino22.3.1を用いてローカルでyoloV7モデルを推定
先ほど作った環境上で自身が作ったプログラムをCPU・VPUで実行する。


## Construct Environment (Workstation)

### Environment

 - Ubuntu 18.04
 - OpenVino 2022.3.1
 - NCS2 (VPU)

### Install OpenVino

#### Runtime
Runtimeのダウンロード。OpenVinoを動かす際のCoreなシステムである。また、NSC2を認識させるためのファイル等も含まれている。

[OpenVino](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)

LTSのため、2022.3.1を使用。

<b>※因みに非常に重要だが、本家のTutorialではなくて、ここからダウンロードしないとNCS2のデバイスファイルがない。</b>

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/177db560-5c37-21fb-f9bf-60034591d51c.png)

`Download Archives`を押すと次の画面が開く。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/381152f2-7ff9-0cdb-affd-deebac690399.png)

この内、Ubuntuで利用するため、`l_openvino_toolkit_ubuntu18_2022.3.1.9227.cf2c7da5689_x86_64.tgz`をダウンロード

``` bash
$ cd ~
$ wget l_openvino_toolkit_ubuntu18_2022.3.1.9227.cf2c7da5689_x86_64.tgz
$ tar xf l_openvino_toolkit_ubuntu18_2022.3.1.9227.cf2c7da5689_x86_64.tgz
```

解凍が終わったら設置

``` bash
$ sudo mkdir -p /opt/intel
$ sudo mv l_openvino_toolkit_ubuntu18_2022.3.1.9227.cf2c7da5689_x86_64 /opt/intel/openvino_2022.3.0
```

次に依存関係のダウンロード

```bash
$ cd /opt/intel/openvino_2022.3.0
$ sudo -E ./install_dependencies/install_openvino_dependencies.sh
```

最後にアクセスしやすいようにリンク

``` bash
cd /opt/intel
sudo ln -s openvino_2022.3.0 openvino_2022
```

起動時に設定ファイルが読み込まれるように追記して、再読み込み

``` bash
$ echo `source /opt/intel/openvino_2022/setupvars.sh` > ~/.bashrc
$ source ~/.bashrc
```

次のように表示されれば成功

```
[setupvars.sh] OpenVINO environment initialized
```

#### Development tool
次のような記述でTrochやTensorflowをインストール可能

```bash
$ pip install openvino-dev[caffe,kaldi,mxnet,pytorch,onnx,tensorflow2]==2022.3.0
```

### Install NCS2

設定ファイルをコピー

``` bash
$ sudo usermod -a -G users "$(whoami)"
$ cd /opt/intel/openvino_2022/install_dependencies/
$ sudo cp 97-myriad-usbboot.rules /etc/udev/rules.d/
```

最後に設定ファイルを`udevadm`に読み込ませてOpenVinoから認識できるようにする。

``` bash
$ sudo udevadm control --reload-rules
$ sudo udevadm trigger
$ sudo ldconfig
```

### Install GPU

``` bash
$ cd /opt/intel/openvino_2022/install_dependencies/
$ sudo -E ./install_NEO_OCL_driver.sh
```


## Run of Sample Code

### ハードウェアテスト
以下のPythonコードを実行する

``` python
from openvino.runtime import Core
ie = Core()
devices = ie.available_devices

for device in devices:
    device_name = ie.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")
```

次のように認識されていればOK

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/74e1fd5d-d0a9-b733-4204-787a98a6262e.png)


### サンプルコードを動かす

``` bash
$ cd /opt/intel/openvino_2022/samples/python/classification_sample_async
```

まずはRequirementsのインストール

```bash
$ pip install -r requirements.txt
```

設定ファイルのダウンロード

``` bash
$ omz_downloader --name alexnet
$ omz_converter --name alexnet
```

`-d MYRIAD`または`-d CPU`でNCS2とCPUを切り替えることができる。

``` bash
$ python classification_sample_async.py -m public/alexnet/FP16/alexnet.xml -i test.jpg -d MYRIAD
```

次のように動作したらOK
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/bae34b1c-abda-74fe-934c-7cdae1761e2c.png)
