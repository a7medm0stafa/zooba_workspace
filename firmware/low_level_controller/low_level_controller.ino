/*
 * Low-Level Controller — Arduino Firmware (PI Speed Control + IMU)
 * ================================================================
 * Receives serial commands from Raspberry Pi 4B and controls:
 *   - DC motor via L298N motor driver (open-loop PWM or closed-loop PI)
 *   - Servo motor (steering / heading angle)
 *   - Reads encoder ticks for speed feedback
 *   - Reads HW-123 IMU (MPU6050) via I2C for orientation
 *
 * Encoder:
 *   - Channel A → Pin 2 (INT0)
 *   - Channel B → Pin 3 (INT1)
 *   - Quadrature decoding for direction-aware tick counting
 *
 * IMU (HW-123 / MPU6050):
 *   - SDA → A4
 *   - SCL → A5
 *   - I2C address: 0x68
 *   - Complementary filter for yaw estimation
 *
 * Serial Protocol (115200 baud):
 *   Receive:  <mode>,<value>,<servo_angle>\n
 *             mode 0:  open-loop   → value = PWM (0–255), direction from sign
 *             mode 1:  PI control  → value = target RPM × 10 (signed integer)
 *             servo_angle: 45–135 degrees (90 = center/straight)
 *   Examples:
 *     "0,150,90\n"   → open-loop forward, PWM 150, steering centered
 *     "0,-150,90\n"  → open-loop reverse, PWM 150, steering centered
 *     "1,500,90\n"   → PI control, target 50.0 RPM forward, centered
 *     "1,-300,80\n"  → PI control, target 30.0 RPM reverse, servo 80°
 *
 *   Feedback: "FB:<rpm>,<ticks>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<yaw>\n"
 *             rpm:   actual output shaft RPM (float, 1 decimal)
 *             ticks: cumulative encoder ticks (long)
 *             ax/ay/az: accelerometer [m/s² × 100, integer]
 *             gx/gy/gz: gyroscope [rad/s × 100, integer]
 *             yaw:   complementary-filter yaw [degrees × 10, integer]
 *   Ack:      "OK\n" after each valid command
 *
 * Commands latch: motor/servo hold last command until a new one is received.
 * Send velocity=0 to stop.
 *
 * Wiring:
 *   DC Motor (L298N):
 *     EN  → Pin 5  (PWM speed control)
 *     IN1 → Pin 7  (direction)
 *     IN2 → Pin 8  (direction)
 *   Encoder:
 *     Channel A → Pin 2  (INT0)
 *     Channel B → Pin 3  (INT1)
 *   Servo:
 *     Signal → Pin 9
 *   IMU (HW-123 / MPU6050):
 *     SDA → A4
 *     SCL → A5
 *     VCC → 5V
 *     GND → GND
 */

#include <Servo.h>
#include <Wire.h>

// ==================== Pin Definitions ====================
// DC Motor (L298N)
const int PIN_EN  = 5;   // PWM speed (pin 5 to keep pins 2,3 for encoder)
const int PIN_IN1 = 7;   // Direction pin 1
const int PIN_IN2 = 8;   // Direction pin 2

// Encoder
const int PIN_ENC_A = 2; // Encoder channel A (INT0)
const int PIN_ENC_B = 3; // Encoder channel B (INT1)

// Servo
const int PIN_SERVO = 9;

// IMU (HW-123 / MPU6050) uses I2C: SDA=A4, SCL=A5 (handled by Wire library)
const int MPU6050_ADDR = 0x68;

// ==================== Encoder Configuration ====================
// JGA25-370 Motor with Hall Encoder (12V)
// Encoder: 11 PPR (pulses per revolution) on motor shaft
// Quadrature decoding (CHANGE on both channels): 11 × 4 = 44 counts/rev (motor shaft)
// Gear ratio: 1:44.727
// Output shaft CPR = 44 × 44.727 ≈ 1968
const int MOTOR_CPR    = 44;      // Counts per motor shaft revolution
const float GEAR_RATIO = 44.727;  // Gear reduction ratio
const int ENCODER_CPR  = 1968;    // Effective CPR at output shaft (44 × 44.727)

// ==================== PI Controller Configuration ====================
// Tunable gains for closed-loop speed control
float PI_KP = 1.5;      // Proportional gain
float PI_KI = 0.8;      // Integral gain
float PI_INTEGRAL_MAX = 200.0;  // Anti-windup limit
int   PI_PWM_MIN = 30;         // Minimum PWM to overcome static friction

// PI state
float piIntegral    = 0.0;
float piTargetRPM   = 0.0;
bool  piEnabled     = false;

// ==================== IMU Configuration ====================
// MPU6050 sensitivity scales (default ±2g accel, ±250°/s gyro)
const float ACCEL_SCALE = 16384.0;  // LSB/g
const float GYRO_SCALE  = 131.0;    // LSB/(°/s)
const float G_TO_MS2    = 9.80665;  // g → m/s²

// Complementary filter
const float COMP_ALPHA = 0.98;  // Weight for gyro in complementary filter

// IMU state
float imuAccelX = 0.0, imuAccelY = 0.0, imuAccelZ = 0.0;  // m/s²
float imuGyroX  = 0.0, imuGyroY  = 0.0, imuGyroZ  = 0.0;  // rad/s
float imuYaw    = 0.0;  // Complementary filter yaw (degrees)
bool  imuReady  = false;

// ==================== Timing ====================
const unsigned long BAUD_RATE       = 115200;

const unsigned long FB_INTERVAL_MS  = 100;    // Feedback at 10Hz
const unsigned long RPM_INTERVAL_MS = 100;    // RPM calculation interval
const unsigned long IMU_INTERVAL_MS = 10;     // IMU read at 100Hz

// ==================== Servo Constants ====================
const int SERVO_CENTER = 90;
const int SERVO_MIN    = 45;
const int SERVO_MAX    = 135;

// ==================== Serial Buffer ====================
const int SERIAL_BUFFER_SIZE = 64;
char serialBuffer[SERIAL_BUFFER_SIZE];
int  bufferIndex = 0;

// ==================== Globals ====================
Servo steeringServo;

// Encoder state (volatile — modified in ISR)
volatile long encoderTicks = 0;

// RPM measurement
float    currentRPM     = 0.0;
long     lastRPMTicks   = 0;
unsigned long lastRPMTime = 0;

// Current motor state
int currentPWM = 0;
int currentDirection = 1;  // 1=forward, 0=reverse

// Timing
unsigned long lastFBTime  = 0;
unsigned long lastIMUTime = 0;

// ==================== Encoder ISR ====================
void encoderISR_A()
{
    if (digitalRead(PIN_ENC_B) == LOW)
        encoderTicks++;
    else
        encoderTicks--;
}

void encoderISR_B()
{
    if (digitalRead(PIN_ENC_A) == LOW)
        encoderTicks--;
    else
        encoderTicks++;
}

// ==================== IMU Functions ====================
float gyroZOffset = 0.0;

void initIMU()
{
    Wire.begin();
    Wire.setClock(400000);  // 400kHz I2C fast mode

    // Wake up MPU6050 (exit sleep mode)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x6B);  // PWR_MGMT_1 register
    Wire.write(0x00);  // Wake up
    Wire.endTransmission(true);

    delay(100);

    // Set accelerometer range to ±2g (default)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x1C);  // ACCEL_CONFIG register
    Wire.write(0x00);  // ±2g
    Wire.endTransmission(true);

    // Set gyroscope range to ±250°/s (default)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x1B);  // GYRO_CONFIG register
    Wire.write(0x00);  // ±250°/s
    Wire.endTransmission(true);

    // Configure DLPF (Digital Low Pass Filter) — bandwidth ~44Hz
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x1A);  // CONFIG register
    Wire.write(0x03);  // DLPF_CFG = 3
    Wire.endTransmission(true);

    // Verify communication
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x75);  // WHO_AM_I register
    Wire.endTransmission(false);
    Wire.requestFrom(MPU6050_ADDR, 1);
    if (Wire.available())
    {
        uint8_t whoami = Wire.read();
        if (whoami == 0x68 || whoami == 0x98)
        {
            // Calculate Gyro Z offset (Calibration)
            long zSum = 0;
            int samples = 200;
            for (int i = 0; i < samples; i++) {
                Wire.beginTransmission(MPU6050_ADDR);
                Wire.write(0x47); // GYRO_ZOUT_H
                Wire.endTransmission(false);
                Wire.requestFrom(MPU6050_ADDR, 2, true);
                if (Wire.available() == 2) {
                    int16_t rawZ = (Wire.read() << 8) | Wire.read();
                    zSum += rawZ;
                }
                delay(2);
            }
            gyroZOffset = (float)zSum / samples;
            
            imuReady = true;
            Serial.println("IMU_OK");
        }
        else
        {
            Serial.print("IMU_ERR:WHO_AM_I=0x");
            Serial.println(whoami, HEX);
        }
    }
    else
    {
        Serial.println("IMU_ERR:NO_RESPONSE");
    }
}

void readIMU()
{
    if (!imuReady) return;

    // Read 14 bytes starting from ACCEL_XOUT_H (0x3B)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU6050_ADDR, 14, true);

    if (Wire.available() < 14) return;

    // Accelerometer (raw → m/s²)
    int16_t rawAccX = (Wire.read() << 8) | Wire.read();
    int16_t rawAccY = (Wire.read() << 8) | Wire.read();
    int16_t rawAccZ = (Wire.read() << 8) | Wire.read();

    // Temperature (skip 2 bytes)
    Wire.read(); Wire.read();

    // Gyroscope (raw → rad/s)
    int16_t rawGyroX = (Wire.read() << 8) | Wire.read();
    int16_t rawGyroY = (Wire.read() << 8) | Wire.read();
    int16_t rawGyroZ = (Wire.read() << 8) | Wire.read();

    // Convert to physical units
    imuAccelX = (rawAccX / ACCEL_SCALE) * G_TO_MS2;
    imuAccelY = (rawAccY / ACCEL_SCALE) * G_TO_MS2;
    imuAccelZ = (rawAccZ / ACCEL_SCALE) * G_TO_MS2;

    imuGyroX = (rawGyroX / GYRO_SCALE) * DEG_TO_RAD;
    imuGyroY = (rawGyroY / GYRO_SCALE) * DEG_TO_RAD;
    
    // Apply offset calibration for Gyro Z to prevent drift
    float correctedGyroZ = rawGyroZ - gyroZOffset;
    // Invert Z-axis so CCW is positive and CW is negative (Right-Hand Rule)
    imuGyroZ = -(correctedGyroZ / GYRO_SCALE) * DEG_TO_RAD;
}

void updateYaw(float dt)
{
    if (!imuReady || dt <= 0.0) return;

    // Gyro integration (yaw rate around Z axis, convert back to degrees for filter)
    float gyroYawRate = imuGyroZ / DEG_TO_RAD;  // rad/s → °/s

    // Complementary filter: blend gyro integration with accumulated yaw
    // (No magnetometer on HW-123, so no absolute reference — gyro-only integration
    //  with drift compensation would require EKF; for now, pure gyro integration)
    imuYaw += gyroYawRate * dt;

    // Wrap yaw to [-180, 180]
    while (imuYaw > 180.0)  imuYaw -= 360.0;
    while (imuYaw < -180.0) imuYaw += 360.0;
}

// ==================== PI Controller ====================
int computePI(float targetRPM, float actualRPM, float dt)
{
    if (dt <= 0.0) return 0;

    float error = targetRPM - actualRPM;

    // Integral with anti-windup
    piIntegral += error * dt;
    piIntegral = constrain(piIntegral, -PI_INTEGRAL_MAX, PI_INTEGRAL_MAX);

    // PI output
    float output = PI_KP * error + PI_KI * piIntegral;

    // Convert to PWM (0-255)
    int pwm = (int)abs(output);
    pwm = constrain(pwm, 0, 255);

    // Apply minimum PWM threshold to overcome static friction
    if (abs(targetRPM) > 0.1 && pwm < PI_PWM_MIN && pwm > 0)
        pwm = PI_PWM_MIN;

    return pwm;
}

void resetPI()
{
    piIntegral = 0.0;
    piTargetRPM = 0.0;
}

// ==================== Setup ====================
void setup()
{
    Serial.begin(BAUD_RATE);

    // Motor pins
    pinMode(PIN_EN, OUTPUT);
    pinMode(PIN_IN1, OUTPUT);
    pinMode(PIN_IN2, OUTPUT);

    // Encoder pins
    pinMode(PIN_ENC_A, INPUT_PULLUP);
    pinMode(PIN_ENC_B, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_ENC_A), encoderISR_A, RISING);
    attachInterrupt(digitalPinToInterrupt(PIN_ENC_B), encoderISR_B, RISING);

    // Servo
    steeringServo.attach(PIN_SERVO);

    // IMU
    initIMU();

    // Safe initial state
    stopMotor();
    steeringServo.write(SERVO_CENTER);

    lastRPMTime = millis();
    lastFBTime  = millis();
    lastIMUTime = millis();

    Serial.println("LOW_LEVEL_READY");
}

// ==================== Main Loop ====================
void loop()
{
    unsigned long now = millis();

    // --- Read serial data ---
    while (Serial.available())
    {
        char c = Serial.read();

        if (c == '\n' || c == '\r')
        {
            if (bufferIndex > 0)
            {
                serialBuffer[bufferIndex] = '\0';
                processCommand(serialBuffer);
                bufferIndex = 0;
            }
        }
        else
        {
            if (bufferIndex < SERIAL_BUFFER_SIZE - 1)
                serialBuffer[bufferIndex++] = c;
            else
                bufferIndex = 0;
        }
    }

    // --- Read IMU (100Hz) ---
    if (now - lastIMUTime >= IMU_INTERVAL_MS)
    {
        float imuDt = (now - lastIMUTime) / 1000.0;
        lastIMUTime = now;
        readIMU();
        updateYaw(imuDt);
    }

    // --- Calculate RPM from encoder ---
    if (now - lastRPMTime >= RPM_INTERVAL_MS)
    {
        noInterrupts();
        long ticks = encoderTicks;
        interrupts();

        long deltaTicks = ticks - lastRPMTicks;
        float dt = (now - lastRPMTime) / 1000.0;
        lastRPMTicks = ticks;
        lastRPMTime = now;

        if (dt > 0)
            currentRPM = ((float)abs(deltaTicks) / (float)ENCODER_CPR) * (60.0 / dt);

        // --- PI control (if enabled) ---
        if (piEnabled)
        {
            float signedRPM = (deltaTicks >= 0) ? currentRPM : -currentRPM;
            int pwm = computePI(piTargetRPM, signedRPM, dt);

            if (abs(piTargetRPM) < 0.1)
            {
                stopMotor();
                resetPI();
            }
            else
            {
                // Direction from target
                if (piTargetRPM >= 0)
                {
                    digitalWrite(PIN_IN1, HIGH);
                    digitalWrite(PIN_IN2, LOW);
                }
                else
                {
                    digitalWrite(PIN_IN1, LOW);
                    digitalWrite(PIN_IN2, HIGH);
                }
                analogWrite(PIN_EN, pwm);
                currentPWM = pwm;
            }
        }
    }

    // --- Send feedback (10Hz) ---
    if (now - lastFBTime >= FB_INTERVAL_MS)
    {
        sendFeedback();
        lastFBTime = now;
    }

}

// ==================== Process Command ====================
void processCommand(const char* cmd)
{
    // Format: "<mode>,<value>,<servo_angle>"
    // Mode 0: open-loop  → value = signed PWM (-255 to 255)
    // Mode 1: PI control → value = target RPM × 10 (signed)
    int mode = 0;
    int value = 0;
    int servoAngle = SERVO_CENTER;

    int parsed = sscanf(cmd, "%d,%d,%d", &mode, &value, &servoAngle);

    if (parsed != 3)
    {
        Serial.print("ERR_PARSE:");
        Serial.println(cmd);
        return;
    }

    // Validate & clamp servo
    servoAngle = constrain(servoAngle, SERVO_MIN, SERVO_MAX);

    if (mode == 0)
    {
        // === Open-loop mode ===
        piEnabled = false;
        resetPI();

        int direction = (value >= 0) ? 1 : 0;
        int pwm = constrain(abs(value), 0, 255);

        if (pwm == 0)
        {
            stopMotor();
        }
        else
        {
            if (direction == 1)
            {
                digitalWrite(PIN_IN1, HIGH);
                digitalWrite(PIN_IN2, LOW);
            }
            else
            {
                digitalWrite(PIN_IN1, LOW);
                digitalWrite(PIN_IN2, HIGH);
            }
            analogWrite(PIN_EN, pwm);
        }

        currentDirection = direction;
        currentPWM = pwm;
    }
    else if (mode == 1)
    {
        // === PI control mode ===
        piEnabled = true;
        piTargetRPM = value / 10.0;  // Decode: value is RPM × 10

        if (abs(piTargetRPM) < 0.1)
        {
            stopMotor();
            resetPI();
            piEnabled = false;
        }

        currentDirection = (piTargetRPM >= 0) ? 1 : 0;
    }
    else
    {
        Serial.print("ERR_MODE:");
        Serial.println(mode);
        return;
    }

    // Apply servo
    steeringServo.write(servoAngle);

    Serial.println("OK");
}

// ==================== Send Feedback ====================
void sendFeedback()
{
    noInterrupts();
    long ticks = encoderTicks;
    interrupts();

    // Format: FB:<rpm>,<ticks>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<yaw>
    // IMU values × 100 as integers for fast parsing
    Serial.print("FB:");
    Serial.print(currentRPM, 1);
    Serial.print(",");
    Serial.print(ticks);

    if (imuReady)
    {
        Serial.print(",");
        Serial.print((int)(imuAccelX * 100));
        Serial.print(",");
        Serial.print((int)(imuAccelY * 100));
        Serial.print(",");
        Serial.print((int)(imuAccelZ * 100));
        Serial.print(",");
        Serial.print((int)(imuGyroX * 100));
        Serial.print(",");
        Serial.print((int)(imuGyroY * 100));
        Serial.print(",");
        Serial.print((int)(imuGyroZ * 100));
        Serial.print(",");
        Serial.print((int)(imuYaw * 10));
    }

    Serial.println();
}

// ==================== Helper ====================
void stopMotor()
{
    digitalWrite(PIN_IN1, LOW);
    digitalWrite(PIN_IN2, LOW);
    analogWrite(PIN_EN, 0);
    currentPWM = 0;
}
