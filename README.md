# Linux env - Tensorflow Object Detection
## 參考資料
* [實驗室的資料]
*  [windows_TensorFlow-GPU 1.5 Install](https://www.youtube.com/watch?v=uIm3DMprk7M&t=12s)
* [winodws Object Detection](https://www.youtube.com/watch?v=Rgpfk6eYxJA&t=382s)

## 前置作業
``` shell=
sudo apt-get update
sudo apt-get install python3-pip        #安裝pip3
sudo apt-get install python-pip         #安裝pip
```
* 為什麼要安裝 兩個pip 因為後面的套件我已經忘記是版本要用哪一個所以下面的套件都是利用pip 若有部分錯誤麻煩自行尋找那個套件的pip3方法



## 安裝CUDA
* [cuda載點](https://developer.nvidia.com/cuda-toolkit-archive) or [cuda載點](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb)

```shell=
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
```
解壓縮並安裝
```shell=
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-9-0
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 2505,875 
```
nvidia-smi -ac 2505,875 # performance optimziation from google suggestion
## add to environment variable
```shell=
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc
nvidia-smi
```
之後可以用 nvidia-smi 確認裝好沒

![](https://i.imgur.com/QahZUSu.png)



# 安裝CUDNN
* [cuDNN載點](https://developer.nvidia.com/rdp/cudnn-archive)下載  cuDNN v7.0.5 (Dec 5, 2017) cuDNN 
* For CUDA 9.0內的cuDNN v7.0.5 Library for Linux
* 利用本機下載Google Cloud SDK Shell ， 將下載下來的CUDNN傳到VM上
* 若出現 ....... short 檔案可能要重抓
```shell=
gcloud compute scp "CUDNN在本機的位置" "instance@XXX:~/"
```
```shell=
sudo tar -xvf "檔案名稱" or sudo unzip "檔案名稱"

sudo cp cuda/include/cudnn.h /usr/local/cuda/include

sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64

sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```
複製到cuda資料夾。
chmod指令a代表所有人，r代表讀取權。
# Install Anaconda
下載一個.sh檔並安裝。
```shell=
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
sha256sum Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh
source ~/.bashrc
conda list
```
# Create Virtual Environment

``` shell=
conda create --name py35 python=3.5.2 anaconda
```
進入虛擬環境內
``` shell=
source activate py35
```
執行以下指令
``` shell=
pip3 install --upgrade tensorflow-gpu
```
裝好後便可以確認
``` shell=
python
```
輸入下面指令
``` shell=
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
成功會出現類似的圖

![](https://i.imgur.com/SOMkVQO.png)

# 安裝套件 ---尚未嘗試
```shell=
sudo apt-get install protobuf-compiler python-lxml
sudo pip install pandas
sudo pip install Cython
sudo pip install opencv-python
```
# 前置 --確定
```shell=
pip install jupyter
pip3 install matplotlib
pip3 install pillow

```
# jupyter notebook
在~/下執行
```shell=
jupyter notebook --generate-config
```
cd 到 產生出來的jupyter下
```shell=
cd .jupyter/
```
執行
```shell=
ipython
```
```shell=
from notebook.auth import passwd
passwd()
```
不要輸入密碼，便可以取得一串"sh....."的字串複製起來等等用的到
```shell=
sudo nano jupyter_notebook_config.py
```
複製下面到文件裡 or 一個一個找並改成下面這樣
GCP的防火牆規則要改
```shell=
c.NotebookApp.allow_remote_access = True
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.password = u'輸入剛才複製的字串'
c.NotebookApp.port = 8888 
```
執行
```shell=
jupyter notebook
```
或是 利用背景執行
```shell=
jupyter notebook &
```
結束背景執行
```shell=
ps -aux | grep notebook
```
找到執行代碼 利用
```shell=
kill -9 '執行代碼'
```
便可以改結束 notebook 
* 利用gcp VM的外部IP :8888便可以連線到 jupyter
```shell=
外部IP:8888
```
# 安裝Tensorflow Object Detection
```shell=
sudo git clone https://github.com/tensorflow/models.git
```
* 請在models/research下
```shell=
sudo protoc object_detection/protos/*.proto --python_out=.
```
若無法執行 protoc 請做以下動作
```shell=
sudo wget https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
sudo apt-get install unzip
sudo unzip protoc-3.6.1-linux-x86_64.zip -d protoc3
sudo mv protoc3/bin/* /usr/local/bin/
sudo mv protoc3/include/* /usr/local/include/
```
加入環境變數

```shell=
sudo nano ~/.bashrc
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
### 準備好你的圖片，可以利用ffmpeg影片轉圖片。
* 還要將圖片經LabelImg後變成XML檔
# 先進行狗狗的測試
## 遇到這個error
```shell=
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
```
我會到下面這裡將上面這段#掉
```shell=
/home/g1105137233/models/research/object_detection/utils/visualization_utils.py
```
## 遇到這個error
```shell=
PermissionError: [Errno 13] Permission denied: 'ssd_mobilenet_v1_coco_2017_11_17.tar.gz'
```
我會去改object_detection的權限
```shell=
這裡還不熟權限，不曉得怎麼改是最好的。
sudo chmod -R 757 object_detection/
```
# 如果遇到
```shell=
The kernel appears to have died. It will restart automatically.
```
### GCP log error
```shell=
Loaded runtime CuDNN library: 7.0.5 but source was compiled with: 7.2.1 ..........
```
請將Tensorflow-gpu 版本改成1.10.1
```shell=
pip3 install --upgrade --force-reinstall tensorflow-gpu==1.10.1 --user
```
我改完虛擬環境py35內的GPU版本後依然出現錯誤
所以我離開虛擬環境並將外部的GPU也改成1.10.1
```shell=
source deactivate 
pip install tensorflow-gpu==1.10.1
```
# 2018/10/22 ----
* 在models/research
```shell=
sudo python setup.py build
sudo python setup.py install
```
# ----------------------------------------


* 我使用以下版本 
* python = 3.5.2
```shell=
python --version
```
* CUDA = 9.0
```shell=
cat /usr/local/cuda/version.txt
```
* CUDNN = 7.0.5
```shell=
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
* tensorflow-gpu = 1.10.1
```shell=
python 
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
# master.zip
```shell=
git clone https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10.git
```
* 將下載下來的檔案移到object_detection
```shell=
sudo mv TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/* models/research/object_detection/
```
* 移動到 object_detection下
```shell=
cd ~/models/research/object_detection
```
* 刪除images 
```shell=
sudo rm -R images
```
* 回到家目錄 下載圖片
```shell=
cd     #回家目錄
git clone https://github.com/winteras/GCPTraining1018.git
```
* 資料夾名稱GCPTraining1018
```shell=
cd GCPTraining1018
```
* 先幫image改名成images
```shell=
sudo mv image images
```
* 把images移到object_detection下
```shell=
sudo mv images/ ~/models/research/object_detection/
```
* 回到object_detection下
```shell=
cd ~/models/research/object_detection/    
```
* 其餘步驟
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
Install the other necessary packages by issuing the following commands:
...
...
...
往下做即可
# 成品
![](https://i.imgur.com/scthYnh.png)
![](https://i.imgur.com/Vd28t1e.png)
![](https://i.imgur.com/6tiBX1v.png)
