/*
 * Low-Level Controller — Arduino Firmware (Open-Loop + Encoder Feedback)
 * ======================================================================
 * Receives serial commands from Raspberry Pi 4B and controls:
 *   - DC motor via L298N motor driver (open-loop PWM)
 *   - Servo motor (steering / heading angle)
 *   - Reads encoder ticks for feedback (no PID — open-loop for now)
 *
 * Encoder:
 *   - Channel A → Pin 2 (INT0)
 *   - Channel B → Pin 3 (INT1)
 *   - Quadrature decoding for direction-aware tick counting
 *
 * Serial Protocol (115200 baud):
 *   Receive:  <direction>,<pwm>,<servo_angle>\n
 *             direction:    1 = forward, 0 = reverse
 *             pwm:          0–255 motor speed (direct open-loop)
 *             servo_angle:  45–135 degrees (90 = center/straight)
 *   Example:  "1,150,90\n" → forward, PWM 150, steering centered
 *
 *   Feedback: "FB:<actual_rpm>,<encoder_ticks>\n"  (sent at ~10Hz)
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
 */

#include <Servo.h>

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

// ==================== Encoder Configuration ====================
// JGA25-370 Motor with Hall Encoder (12V)
// Encoder: 11 PPR (pulses per revolution) on motor shaft
// Quadrature decoding (CHANGE on both channels): 11 × 4 = 44 counts/rev (motor shaft)
// Output shaft CPR = 44 × gear_ratio
//
// Common JGA25-370 gear ratios and resulting CPR:
//   1:21.3  → 44 × 21.3 =  937 CPR
//   1:30    → 44 × 30   = 1320 CPR
//   1:45    → 44 × 45   = 1980 CPR
//   1:75    → 44 × 75   = 3300 CPR
//   1:103   → 44 × 103  = 4532 CPR
//
// *** CHANGE THIS to match YOUR gear ratio ***
const int ENCODER_CPR = 937;   // Default: 1:21.3 gear ratio

// ==================== Timing ====================
const unsigned long BAUD_RATE       = 115200;

const unsigned long FB_INTERVAL_MS  = 100;    // Feedback at 10Hz
const unsigned long RPM_INTERVAL_MS = 100;    // RPM calculation interval

// ==================== Servo Constants ====================
const int SERVO_CENTER = 90;
const int SERVO_MIN    = 45;
const int SERVO_MAX    = 135;

// ==================== Serial Buffer ====================
const int SERIAL_BUFFER_SIZE = 32;
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

// Timing
unsigned long lastFBTime = 0;

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
    attachInterrupt(digitalPinToInterrupt(PIN_ENC_A), encoderISR_A, CHANGE);
    attachInterrupt(digitalPinToInterrupt(PIN_ENC_B), encoderISR_B, CHANGE);

    // Servo
    steeringServo.attach(PIN_SERVO);

    // Safe initial state
    stopMotor();
    steeringServo.write(SERVO_CENTER);

    lastRPMTime = millis();
    lastFBTime  = millis();

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
    // Format: "direction,pwm,servo_angle"
    int direction = 0;
    int pwm = 0;
    int servoAngle = SERVO_CENTER;

    int parsed = sscanf(cmd, "%d,%d,%d", &direction, &pwm, &servoAngle);

    if (parsed != 3)
    {
        Serial.print("ERR_PARSE:");
        Serial.println(cmd);
        return;
    }

    // Validate & clamp
    direction = (direction >= 1) ? 1 : 0;
    pwm = constrain(pwm, 0, 255);
    servoAngle = constrain(servoAngle, SERVO_MIN, SERVO_MAX);

    // Apply motor command (open-loop — direct PWM)
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

    Serial.print("FB:");
    Serial.print(currentRPM, 1);
    Serial.print(",");
    Serial.println(ticks);
}

// ==================== Helper ====================
void stopMotor()
{
    digitalWrite(PIN_IN1, LOW);
    digitalWrite(PIN_IN2, LOW);
    analogWrite(PIN_EN, 0);
}
