#include "esp_camera.h"

#define CAMERA_MODEL_AI_THINKER // 定義相機模型
#include "camera_pins.h"

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;  // 直接设置为 QVGA
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
  config.fb_count = 1;

  // 相機初始化
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }

  sensor_t * s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_QVGA);  // 确保设置为 QVGA

  Serial.println("Camera Ready!");
  return true;
}

void setup() {
  Serial.begin(921600);
  Serial.setDebugOutput(true);
  Serial.println();

  if (!initCamera()) {
    ESP.restart();  // 如果相机初始化失败，重启 ESP32
  }
}

void loop() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(1000);
    return;
  }

  // 发送 JPEG 开始标记
  Serial.write((uint8_t*)"\xFF\xD8\xFF", 3);

  // 發送圖像大小（4字節）
  uint32_t imageSize = fb->len;
  Serial.write((uint8_t *)&imageSize, 4);

  // 發送圖像數據
  Serial.write(fb->buf, fb->len);

  esp_camera_fb_return(fb);

  // 检查是否有重置命令
  if (Serial.available() && Serial.read() == 'R') {
    Serial.println("Resetting camera...");
    esp_camera_deinit();
    delay(100);
    if (!initCamera()) {
      ESP.restart();
    }
  }

  // 等待一段時間再捕獲下一幀
  delay(100);
}