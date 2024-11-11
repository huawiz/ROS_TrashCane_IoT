#include "esp_camera.h"

#define CAMERA_MODEL_AI_THINKER
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
  
  // 维持较高时钟频率以确保图像质量
  config.xclk_freq_hz = 20000000;
  
  // 使用SVGA分辨率，在细节和速度之间平衡
  config.frame_size = FRAMESIZE_SVGA;
  
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  
  // 保持较高的JPEG质量以保留标志细节
  config.jpeg_quality = 10;
  
  // 使用单帧缓冲减少延迟
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }

  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    // 优化图像设置以提高标志识别效果
    s->set_quality(s, 10);           // 保持较好的图像质量
    s->set_framesize(s, FRAMESIZE_SVGA);
    
    // 增强对比度和锐度以突出标志特征
    s->set_contrast(s, 2);           // 增加对比度
    s->set_brightness(s, 1);         // 略微提高亮度
    s->set_saturation(s, 1);         // 略微提高饱和度
    s->set_sharpness(s, 2);          // 增加锐度
    
    // 保持自动白平衡和曝光控制以适应不同光线条件
    s->set_whitebal(s, 1);           // 启用自动白平衡
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 0);            // 自动白平衡模式
    s->set_exposure_ctrl(s, 1);      // 启用自动曝光
    s->set_aec2(s, 1);               // 启用高级自动曝光控制
    s->set_ae_level(s, 0);           // 默认曝光等级
    s->set_aec_value(s, 300);        // 默认曝光值
    
    // 启用图像增强功能
    s->set_dcw(s, 1);                // 启用下采样
    s->set_bpc(s, 1);                // 启用坏点校正
    s->set_wpc(s, 1);                // 启用白点校正
    s->set_lenc(s, 1);               // 启用镜头畸变校正
    
    // 关闭不必要的效果
    s->set_special_effect(s, 0);     // 无特殊效果
    s->set_hmirror(s, 0);            // 根据需要调整镜像
    s->set_vflip(s, 0);              // 根据需要调整翻转
    
    // 降噪设置
    s->set_denoise(s, 1);            // 启用降噪
  }

  Serial.println("Camera Ready! Logo Detection Optimized Mode");
  return true;
}

void setup() {
  Serial.begin(921600);
  Serial.setDebugOutput(false);

  if (!initCamera()) {
    ESP.restart();
  }
}

void loop() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(100);
    return;
  }

  // 发送 JPEG 头
  Serial.write((uint8_t*)"\xFF\xD8\xFF", 3);
  
  // 发送大小
  uint32_t imageSize = fb->len;
  Serial.write((uint8_t *)&imageSize, 4);

  // 发送图像数据
  Serial.write(fb->buf, fb->len);

  esp_camera_fb_return(fb);

  // 适当的采集间隔
  delay(50);
}