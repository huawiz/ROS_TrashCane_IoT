#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "wifi";      // Wi-Fi SSID
const char* password = "0921848743";  // Wi-Fi 密碼

const int ledPin1 = 16; // LED 1 引腳 (模擬反轉)
const int ledPin2 = 17; // LED 2 引腳 (模擬正轉)

WebServer server(80); // 創建 Web 伺服器對象，監聽端口 80

void setup() {
    Serial.begin(115200);
    
    // 連接 Wi-Fi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println(" connected!");

    // 顯示獲取到的 IP 地址
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());

    // 設定 LED 引腳為輸出
    pinMode(ledPin1, OUTPUT);
    pinMode(ledPin2, OUTPUT);

    // 初始化 LED 狀態為關閉
    digitalWrite(ledPin1, LOW);
    digitalWrite(ledPin2, LOW);

    // 設定路由器
    server.on("/R", handleReverse); // 當訪問 /R 時調用 handleReverse()
    server.on("/L", handleForward); // 當訪問 /L 時調用 handleForward()
    server.on("/OFF", handleOff);   // 當訪問 /OFF 時調用 handleOff()
    server.begin(); // 開始伺服器
    Serial.println("HTTP server started");
}

void loop() {
    server.handleClient(); // 處理客戶端請求
}

void handleReverse() {
    digitalWrite(ledPin1, HIGH); // 打開 LED 1 (反轉)
    digitalWrite(ledPin2, LOW);  // 關閉 LED 2
    Serial.println("Reverse: LED 1 ON, LED 2 OFF");
    server.send(200, "text/plain", "Reverse: LED 1 ON, LED 2 OFF");
}

void handleForward() {
    digitalWrite(ledPin1, LOW);  // 關閉 LED 1
    digitalWrite(ledPin2, HIGH); // 打開 LED 2 (正轉)
    Serial.println("Forward: LED 1 OFF, LED 2 ON");
    server.send(200, "text/plain", "Forward: LED 1 OFF, LED 2 ON");
}

void handleOff() {
    digitalWrite(ledPin1, LOW);
    digitalWrite(ledPin2, LOW);
    Serial.println("All LEDs OFF");
    server.send(200, "text/plain", "All LEDs OFF");
}