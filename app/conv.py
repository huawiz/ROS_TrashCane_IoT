import tensorflow as tf

# 加載 Keras 模型
model = tf.keras.models.load_model('C:\\projects\\ROS_TrashCane_IoT\\app\\keras_Model.h5')

# 轉換為 TFLite 模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存 TFLite 模型
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)