# ĐỒ ÁN MÔN MÁY HỌC (CS114.N21.KHCL)

- Tên đề tài: Nhận dạng giọng nói bằng tiếng Anh
- Tên lớp: CS114.N21.KHCL
- Tên nhóm: Nhóm 8 (N08)

## Danh sách thành viên
- Nguyễn Đắc Cường (21521902)
- Đỗ Bá Huy (21522137)
- Nguyễn Mai Chí Tấn (21521414)
- Huỳnh Nhân Thập (21521457)

## Danh sách các file trong repo
- model_1_with_ljspeech.ipynb: Jupyter notebook huấn luyện và đánh giá model 1 trên bộ dữ liệu LJ Speech
- model_1_with_team_dataset.ipynb: Jupyter notebook huấn luyện và đánh giá model 1 trên bộ dữ liệu do nhóm tự thu thập
- model_2_with_ljspeech.ipynb: Jupyter notebook huấn luyện và đánh giá model 1 trên bộ dữ liệu LJ Speech
- model_2_with_team_dataset.ipynb: Jupyter notebook huấn luyện và đánh giá model 2 trên bộ dữ liệu do nhóm tự thu thập
- model_1_with_ljspeech_epoch025.keras: File lưu lại model 1 train trên bộ dữ liệu LJ Speech sau 25 epoch.
- model_1_with_team_dataset_epoch050.keras: File lưu tại model 1 train trên bộ dữ liệu LJ Speech sau 50 epoch.

Để có thể sử dụng hai file model (.keras), hãy chạy đoạn code sau:

```
import tensorflow as tf
from tensorflow import keras

def ctc_loss(target, prediction):
    batch_size = tf.cast(tf.shape(target)[0], dtype='int64')
    input_size = tf.cast(tf.shape(prediction)[1], dtype='int64')
    label_size = tf.cast(tf.shape(target)[1], dtype='int64')
    temp = tf.ones(shape=(batch_size, 1), dtype='int64')
    input_size = input_size * temp
    label_size = label_size * temp
    loss = keras.backend.ctc_batch_cost(target, prediction, input_size, label_size)
    return loss

# To load saved model
model = keras.models.load_model('/content/drive/My Drive/model-doan-datatuthuthap-epoch50.keras', custom_objects = {"ctc_loss": ctc_loss}) 

# Uses model.predict(...) to predict
```

## Danh sách đường dẫn
- datato2.zip: https://drive.google.com/file/d/1-1JqAcxUPUNyTF4tKmvQZ-RVR3GGrJlV Các file audio trong bộ dữ liệu nhóm tự thu thập
- metadata02.csv: https://drive.google.com/file/d/1YV86EDntW_CUrkZzcuiXntf60BkymWsD Metadata của bộ dữ liệu nhóm tự thu thập
