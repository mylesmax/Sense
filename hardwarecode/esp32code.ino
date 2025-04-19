#include <Arduino.h>
#include <NimBLEDevice.h>
#include <ArduinoJson.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include "bme68xLibrary.h"

#define SAMPLES_PER_READ 64
#define ADC_RESOLUTION 12

Adafruit_INA219 ina219(0x43);
float batteryVoltage = 0.0;
float batteryCurrent = -1.0;
float batteryPercentage = -1.0;
unsigned long lastBatterySample = 0;
const unsigned long BATTERY_SAMPLE_INTERVAL = 5000;
bool ina219Available = false;

const int s1 = 36;  // SVP channel
const int s2 = 39;  // SVN channel
const int s3 = 34;  // P34 channel
const int s4 = 32;  // P32 channel
const int s5 = 33;  // P33 channel
const int s6 = 27;  // P27 channel
const int s7 = 14;  // P14 channel
const int s8 = 12;  // P12 channel
const int sensorPins[] = {s1, s2, s3, s4, s5, s6, s7, s8};
const int numSensors = 8;

int readADC(int pin) {
    return analogRead(pin);
}

void updateBatteryData() {
    float busVoltage_V = 0;
    float current_mA = 0;
    bool readSuccess = false;

    if (!ina219Available) {
        Wire.beginTransmission(0x43);
        if (Wire.endTransmission() == 0) {
            if (ina219.begin(&Wire)) {
                Serial.println("INA219 is back");
            }
        }
    }

    Wire.beginTransmission(0x43);
    if (Wire.endTransmission() == 0) {
        busVoltage_V = ina219.getBusVoltage_V();
        current_mA = ina219.getCurrent_mA();
        
        if (isnan(current_mA)) {
            current_mA = 0;
        }
        
        if (busVoltage_V > 0 && busVoltage_V < 15) {
            readSuccess = true;
            ina219Available = true;
            
            batteryCurrent = current_mA;
            float rawPercent = (busVoltage_V - 3.0f) / 1.2f * 100.0f;
            batteryPercentage = constrain(rawPercent, 0.0f, 100.0f);
            
            Serial.print("Bus Voltage: "); Serial.print(busVoltage_V, 2); Serial.println(" V");
            Serial.print("Current: "); Serial.print(batteryCurrent, 2); Serial.println(" mA");
            Serial.print("Battery: "); Serial.print(batteryPercentage, 2); Serial.println("%");
        }
    }

    if (!readSuccess) {
        batteryCurrent = -1.0;
        batteryPercentage = -1.0;
        
        if (ina219Available) {
            Serial.println("battery read failed");
            ina219Available = false;
        }
    }
}

#ifndef PIN_CS
#define PIN_CS SS
#endif

Bme68x bme;

//BLE
#define SERVICE_UUID        "2cc12ee8-c5b6-4d7f-a3de-9c793653f271"
#define CHARACTERISTIC_UUID "15216e4f-bf54-4482-8a91-74a92ccfeb37"

NimBLEServer* pServer = nullptr;
NimBLECharacteristic* pCharacteristic = nullptr;

//timer, thanks noah bryson
hw_timer_t * timer = NULL;
volatile SemaphoreHandle_t timerSemaphore;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;
volatile uint32_t isrCounter = 0;

void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timerMux);
  isrCounter++;
  portEXIT_CRITICAL_ISR(&timerMux);
  xSemaphoreGiveFromISR(timerSemaphore, NULL);
}

const int BLE_LED = 13;
const int FAN_PIN = 17;
//fan pulsing. experimental.
const unsigned long FAN_ON_DURATION = 4000;
const unsigned long FAN_OFF_DURATION = 0;//nah
const bool FAN_CONTINUOUS = (FAN_OFF_DURATION == 0);
unsigned long lastPulseTime = 0;
bool fanState = false;

void resetI2CBus() {
  pinMode(21, OUTPUT);  // SDA
  pinMode(22, OUTPUT);  // SCL
  digitalWrite(21, HIGH);
  digitalWrite(22, HIGH);
  delay(20);
  
  for(int i = 0; i < 10; i++) {
    digitalWrite(22, LOW);
    delay(1);
    digitalWrite(22, HIGH);
    delay(1);
  }
  
  digitalWrite(21, LOW);
  delay(1);
  digitalWrite(22, HIGH);
  delay(1);
  digitalWrite(21, HIGH);
  delay(20);
  
  pinMode(21, INPUT_PULLUP);
  pinMode(22, INPUT_PULLUP);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Starting Combined Sensor and BME688 with BLE...");

  resetI2CBus();
  delay(100);

  Wire.begin(21, 22);
  Wire.setClock(100000);
  
  Serial.print("SDA (GPIO21) state: "); Serial.println(digitalRead(21));
  Serial.print("SCL (GPIO22) state: "); Serial.println(digitalRead(22));
  
  if (digitalRead(22) == LOW) {
    Serial.println("SCL still LOW after reset");
    for(int i = 0; i < 5; i++) {
      pinMode(22, OUTPUT);
      digitalWrite(22, HIGH);
      delay(10);
      pinMode(22, INPUT_PULLUP);
      delay(10);
      Serial.print("SCL state after toggle "); 
      Serial.print(i);
      Serial.print(": ");
      Serial.println(digitalRead(22));
    }
  }

  Serial.println("\nscanning I2C bus...");
  byte error, address;
  int nDevices = 0;
  for(address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    
    if (error == 0) {
      if (address < 16) Serial.print("0");
      Serial.println(address, HEX);
      nDevices++;
    }
    else if (error == 4) {
      if (address < 16) Serial.print("0");
      Serial.println(address, HEX);
    }
  }
  
  if (!ina219.begin(&Wire)) {
    batteryCurrent = -1.0;
    batteryPercentage = -1.0;
    ina219Available = false;
  } else {
    Serial.println("found INA219");
    ina219Available = true;
    
    float testVoltage = ina219.getBusVoltage_V();
    float testCurrent = ina219.getCurrent_mA();
    Serial.print("test voltage: "); Serial.print(testVoltage);
    Serial.print("test current: "); Serial.print(testCurrent);
    if (testVoltage == 0 && testCurrent == 0) {
      batteryCurrent = -1.0;
      batteryPercentage = -1.0;
      ina219Available = false;
    }
  }

  pinMode(BLE_LED, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);
  digitalWrite(BLE_LED, LOW);
  digitalWrite(FAN_PIN, LOW);

  analogReadResolution(ADC_RESOLUTION);
  
  analogSetAttenuation(ADC_11db);

  for (int i = 0; i < numSensors; i++) {
    pinMode(sensorPins[i], INPUT);
  }

  //bme688
  SPI.begin();
  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);

  bme.begin(PIN_CS, SPI);

  //if bme688 is not working, KILL IT!
  if(bme.checkStatus()) {
    if (bme.checkStatus() == BME68X_ERROR) {
      Serial.println("BME688 error: " + String(bme.statusString()));
      while (1);
    }
    else if (bme.checkStatus() == BME68X_WARNING) {
      Serial.println("BME688 Warning: " + String(bme.statusString()));
    }
  } else {
    Serial.println("BME688 sensor good");
  }

  bme.setTPH();
  
  //bme688 heater
  bme.setHeaterProf(200, 100);
  bme.setOpMode(BME68X_FORCED_MODE);

  //timer
  timerSemaphore = xSemaphoreCreateBinary();
  timer = timerBegin(0, 80, true); //80MHz clock divided by 80 gives 1MHz timebase
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, 330000, true);//clock is 80MHz, 330ms is 330000us
  timerAlarmEnable(timer);

  NimBLEDevice::init("03senseV3");
  pServer = NimBLEDevice::createServer();

  NimBLEService* pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      NIMBLE_PROPERTY::NOTIFY
                    );

  pService->start();
  NimBLEAdvertising* pAdvertising = NimBLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  NimBLEDevice::startAdvertising();
  digitalWrite(BLE_LED, HIGH);
  Serial.println("started BLE advertising...");
}

const int MAX_PACKET_SIZE = 20;

void sendJSONViaBLE(const String& jsonString) {
  int length = jsonString.length();
  int packets = (length + MAX_PACKET_SIZE - 1) / MAX_PACKET_SIZE;

  pCharacteristic->setValue("START");
  pCharacteristic->notify();
  delay(15);

  for (int i = 0; i < packets; i++) {
    String chunk = jsonString.substring(i * MAX_PACKET_SIZE, min((i + 1) * MAX_PACKET_SIZE, length));
    std::string utf8Chunk = chunk.c_str();
    pCharacteristic->setValue(utf8Chunk);
    pCharacteristic->notify();
    delay(15);
  }

  pCharacteristic->setValue("END");
  pCharacteristic->notify();
}

void loop() {
  unsigned long currentTime = millis();
  
  //experimental.
  if (FAN_CONTINUOUS) {
    if (!fanState) {
      fanState = true;
      digitalWrite(FAN_PIN, HIGH);
    }
  } else {
    if (fanState) {
      if (currentTime - lastPulseTime >= FAN_ON_DURATION) {
        fanState = false;
        digitalWrite(FAN_PIN, fanState);
        lastPulseTime = currentTime;
      }
    } else {
      if (currentTime - lastPulseTime >= FAN_OFF_DURATION) {
        fanState = true;
        digitalWrite(FAN_PIN, fanState);
        lastPulseTime = currentTime;
      }
    }
  }

  if (currentTime - lastBatterySample >= BATTERY_SAMPLE_INTERVAL) {
    lastBatterySample = currentTime;
    updateBatteryData();
  }

  if (xSemaphoreTake(timerSemaphore, 0) == pdTRUE) {
    unsigned long cycleStartTime = millis();
    
    float sensorValues[8];
    for (int i = 0; i < numSensors; i++) {
      int sensorValue = readADC(sensorPins[i]);
      sensorValues[i] = sensorValue * (3.3 / 4095.0);
    }
    
    if (isnan(batteryCurrent)) {
      batteryCurrent = -1.0;
    }
    if (isnan(batteryPercentage)) {
      batteryPercentage = -1.0;
    }
    
    bme68xData data[2];
    uint8_t nFields = 0;
    uint16_t tempSequence[2] = {250, 350};
    
    for (int i = 0; i < 2; i++) {
        bme.setHeaterProf(tempSequence[i], 100);
        bme.setOpMode(BME68X_FORCED_MODE);
        delay(75);
        
        if (bme.fetchData()) {
            bme.getData(data[i]);
            nFields++;
        }
    }
    
    if (nFields == 2) {
      StaticJsonDocument<512> doc;
      doc["x"] = millis();
      
      for (int i = 0; i < numSensors; i++) {
        char voltageStr[10];
        sprintf(voltageStr, "%.4f", sensorValues[i]);
        doc["n" + String(i+1)] = atof(voltageStr);
      }
      
      char tempStr[10], pressStr[10], humStr[10];
      sprintf(tempStr, "%.4f", data[1].temperature);
      sprintf(pressStr, "%.4f", data[1].pressure);
      sprintf(humStr, "%.4f", data[1].humidity);
      doc["t"] = atof(tempStr);
      doc["p"] = atof(pressStr);
      doc["h"] = atof(humStr);
      
      char gasStr[2][10];
      for (int i = 0; i < 2; i++) {
        sprintf(gasStr[i], "%.4f", data[i].gas_resistance);
        doc["g" + String(i+1)] = atof(gasStr[i]);
      }
      
      char currentStr[10], percentStr[10];
      sprintf(currentStr, "%.2f", batteryCurrent);
      sprintf(percentStr, "%.2f", batteryPercentage);
      doc["b_i"] = atof(currentStr);
      doc["b_p"] = atof(percentStr);
      
      String jsonString;
      serializeJson(doc, jsonString);
      Serial.println(jsonString);
      sendJSONViaBLE(jsonString);
      
      unsigned long cycleTime = millis() - cycleStartTime;
      static unsigned long maxCycleTime = 0;
      if (cycleTime > maxCycleTime) {
        maxCycleTime = cycleTime;
        Serial.print("max cycle time: ");
        Serial.print(maxCycleTime);
        Serial.println(" ms");
      }
    } else {
      Serial.println("Failed to get both BME688 readings");
    }
  }
}