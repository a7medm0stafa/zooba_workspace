/*
 * Low-Level Controller — Arduino Firmware (PI Speed Control + IMU + EKF)
 * ======================================================================
 * AHMED AND HAZIM WERE HERE, AND CLAUDE HELPED!
 * Receives serial commands from Raspberry Pi 4B and controls:
 *   - DC motor via L298N motor driver (open-loop PWM or closed-loop PI)
 *   - Servo motor (steering / heading angle)
 *   - Reads encoder ticks for speed feedback
 *   - Reads HW-123 IMU (MPU6050) via I2C for orientation
 *   - Runs a 4-state EKF for localization [x, y, theta, v]
 *
 * Serial Protocol (115200 baud):
 *   Receive:  <mode>,<value>,<servo_angle>\n
 *   Feedback:
 * "FB:<rpm>,<ticks>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<yaw>,<angle>,<ekf_x>,<ekf_y>,<ekf_theta>,<ekf_v>\n"
 */

#include <Servo.h>
#include <Wire.h>

// ==================== Pin Definitions ====================
const int PIN_EN = 5;
const int PIN_IN1 = 7;
const int PIN_IN2 = 8;
const int PIN_ENC_A = 2;
const int PIN_ENC_B = 3;
const int PIN_SERVO = 9;
const int MPU6050_ADDR = 0x68;

// ==================== Drivetrain Configuration ====================
const int MOTOR_CPR = 44;
const float INTERNAL_GEAR_RATIO = 44.727;
const float EXTERNAL_GEAR_RATIO = 45.45 / 16.35;
const float GEAR_RATIO = INTERNAL_GEAR_RATIO * EXTERNAL_GEAR_RATIO;
const int ENCODER_CPR = 5471;
const float WHEEL_RADIUS_M = 0.033;
const float MAX_OUTPUT_RPM = 71.95;
const float WHEELBASE = 0.265;
const float MAX_STEERING_ANGLE_DEG = 45.0;

// ==================== PI Controller ====================
float PI_KP = 1.5;
float PI_KI = 0.8;
float PI_INTEGRAL_MAX = 400.0;
int PI_PWM_MIN = 30;
float piIntegral = 0.0;
float piTargetRPM = 0.0;
bool piEnabled = false;

// ==================== IMU Configuration ====================
const float ACCEL_SCALE = 16384.0;
const float GYRO_SCALE = 131.0;
const float G_TO_MS2 = 9.80665;
const float COMP_ALPHA = 0.98;

float imuAccelX = 0.0, imuAccelY = 0.0, imuAccelZ = 0.0;
float imuGyroX = 0.0, imuGyroY = 0.0, imuGyroZ = 0.0;
float imuYaw = 0.0;
bool imuReady = false;
float gyroZOffset = 0.0;

// ==================== Timing ====================
const unsigned long BAUD_RATE = 115200;
const unsigned long FB_INTERVAL_MS = 50; // 20Hz feedback
const unsigned long RPM_INTERVAL_MS = 100;
const unsigned long IMU_INTERVAL_MS = 10; // 100Hz IMU + EKF

// ==================== Servo Constants ====================
const int SERVO_CENTER = 82; // Calibrated for rightward drift
const int SERVO_MIN = 37;
const int SERVO_MAX = 127;

// ==================== Serial Buffer ====================
const int SERIAL_BUFFER_SIZE = 64;
char serialBuffer[SERIAL_BUFFER_SIZE];
int bufferIndex = 0;

// ==================== Globals ====================
Servo steeringServo;
volatile long encoderTicks = 0;
float currentRPM = 0.0;
long lastRPMTicks = 0;
unsigned long lastRPMTime = 0;
int currentPWM = 0;
int currentDirection = 1;
unsigned long lastFBTime = 0;
unsigned long lastIMUTime = 0;
int lastServoAngle = SERVO_CENTER;

// ==================== EKF State ====================
// State vector: [x, y, theta, v]
float ekf_x[4] = {0, 0, 0, 0};
// Covariance matrix P (4x4, stored as flat array row-major)
float ekf_P[16];

// Process noise standard deviations
const float EKF_SIGMA_V = 0.1;      // m/s velocity noise
const float EKF_SIGMA_OMEGA = 0.05; // rad/s gyro noise

// Measurement noise variances
const float EKF_R_ENCODER = 0.0025; // (0.05)^2
const float EKF_R_ACKERMANN = 0.04; // (0.2)^2
const float EKF_R_ZUPT = 0.000001;  // (0.001)^2

// ZUPT threshold
const float ZUPT_VEL_THRESH = 0.02; // m/s

// Temporary pre-predict theta for Ackermann
float ekf_theta_pre = 0.0;

// Encoder velocity for EKF
float ekf_encoder_velocity = 0.0;

// ==================== 4x4 Matrix Helpers ====================
// All matrices stored as float[16] in row-major order
// Index: M[r][c] = arr[r*4 + c]

void mat4_zero(float *M) {
  for (int i = 0; i < 16; i++)
    M[i] = 0.0;
}

void mat4_identity(float *M) {
  mat4_zero(M);
  M[0] = M[5] = M[10] = M[15] = 1.0;
}

void mat4_copy(float *dst, const float *src) {
  for (int i = 0; i < 16; i++)
    dst[i] = src[i];
}

void mat4_add(float *C, const float *A, const float *B) {
  for (int i = 0; i < 16; i++)
    C[i] = A[i] + B[i];
}

void mat4_multiply(float *C, const float *A, const float *B) {
  float tmp[16];
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      float s = 0.0;
      for (int k = 0; k < 4; k++) {
        s += A[r * 4 + k] * B[k * 4 + c];
      }
      tmp[r * 4 + c] = s;
    }
  }
  mat4_copy(C, tmp);
}

void mat4_transpose(float *T, const float *M) {
  float tmp[16];
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++)
      tmp[c * 4 + r] = M[r * 4 + c];
  mat4_copy(T, tmp);
}

// Multiply 4x4 matrix by 4x1 vector: out = M * v
void mat4_vec_mul(float *out, const float *M, const float *v) {
  for (int r = 0; r < 4; r++) {
    out[r] = 0;
    for (int c = 0; c < 4; c++)
      out[r] += M[r * 4 + c] * v[c];
  }
}

// ==================== EKF Functions ====================

void ekfInit() {
  ekf_x[0] = 0;
  ekf_x[1] = 0;
  ekf_x[2] = 0;
  ekf_x[3] = 0;
  mat4_zero(ekf_P);
  ekf_P[0] = 0.01;  // x variance
  ekf_P[5] = 0.01;  // y variance
  ekf_P[10] = 0.01; // theta variance
  ekf_P[15] = 0.01; // v variance
}

void ekfPredict(float gyro_z, float dt) {
  if (dt <= 0.0 || dt > 1.0)
    return;

  float theta = ekf_x[2];
  float v = ekf_x[3];
  float ct = cos(theta);
  float st = sin(theta);

  // Save pre-predict theta for Ackermann update
  ekf_theta_pre = theta;

  // State prediction: unicycle model
  ekf_x[0] += v * ct * dt; // x
  ekf_x[1] += v * st * dt; // y
  ekf_x[2] += gyro_z * dt; // theta (gyro already bias-corrected)
  // ekf_x[3] unchanged (constant velocity model)

  // Normalize theta to [-PI, PI]
  while (ekf_x[2] > PI)
    ekf_x[2] -= 2.0 * PI;
  while (ekf_x[2] < -PI)
    ekf_x[2] += 2.0 * PI;

  // Jacobian F = df/dx
  float F[16];
  mat4_identity(F);
  F[0 * 4 + 2] = -v * st * dt; // dx/dtheta
  F[0 * 4 + 3] = ct * dt;      // dx/dv
  F[1 * 4 + 2] = v * ct * dt;  // dy/dtheta
  F[1 * 4 + 3] = st * dt;      // dy/dv

  // Noise input Jacobian G (4x2: maps [noise_v, noise_omega] to state)
  // G = [[ct*dt, 0], [st*dt, 0], [0, dt], [1, 0]]
  // Q_input = diag(sigma_v^2, sigma_omega^2)
  // Q = G * Q_input * G^T (4x4)
  float gv0 = ct * dt, gv1 = st * dt;
  float sv2 = EKF_SIGMA_V * EKF_SIGMA_V;
  float so2 = EKF_SIGMA_OMEGA * EKF_SIGMA_OMEGA;
  float dt2 = dt * dt;

  float Q[16];
  mat4_zero(Q);
  Q[0 * 4 + 0] = sv2 * gv0 * gv0;
  Q[0 * 4 + 1] = sv2 * gv0 * gv1;
  Q[0 * 4 + 3] = sv2 * gv0;
  Q[1 * 4 + 0] = sv2 * gv1 * gv0;
  Q[1 * 4 + 1] = sv2 * gv1 * gv1;
  Q[1 * 4 + 3] = sv2 * gv1;
  Q[2 * 4 + 2] = so2 * dt2;
  Q[3 * 4 + 0] = sv2 * gv0;
  Q[3 * 4 + 1] = sv2 * gv1;
  Q[3 * 4 + 3] = sv2;

  // P = F * P * F^T + Q
  float FP[16], FT[16], FPFt[16];
  mat4_multiply(FP, F, ekf_P);
  mat4_transpose(FT, F);
  mat4_multiply(FPFt, FP, FT);
  mat4_add(ekf_P, FPFt, Q);
}

// Scalar measurement update: z_meas for state index `idx`
// H = [0...1...0] with 1 at position idx
void ekfScalarUpdate(float z_meas, int idx, float R) {
  // Innovation: y = z_meas - x[idx]
  float innov = z_meas - ekf_x[idx];

  // For heading (idx=2), normalize innovation
  if (idx == 2) {
    while (innov > PI)
      innov -= 2.0 * PI;
    while (innov < -PI)
      innov += 2.0 * PI;
  }

  // S = P[idx][idx] + R
  float S = ekf_P[idx * 4 + idx] + R;
  if (S < 1e-12)
    return;

  // Kalman gain K (4x1) = P[:,idx] / S
  float K[4];
  for (int i = 0; i < 4; i++) {
    K[i] = ekf_P[i * 4 + idx] / S;
  }

  // State update: x += K * innov
  for (int i = 0; i < 4; i++) {
    ekf_x[i] += K[i] * innov;
  }

  // Covariance update: P -= K * P[idx,:] (Joseph form simplified)
  float P_row[4];
  for (int c = 0; c < 4; c++)
    P_row[c] = ekf_P[idx * 4 + c];
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      ekf_P[r * 4 + c] -= K[r] * P_row[c];
    }
  }

  // Normalize theta
  while (ekf_x[2] > PI)
    ekf_x[2] -= 2.0 * PI;
  while (ekf_x[2] < -PI)
    ekf_x[2] += 2.0 * PI;
}

void ekfUpdateVelocity(float v_measured) {
  ekfScalarUpdate(v_measured, 3, EKF_R_ENCODER);
}

void ekfZUPT() { ekfScalarUpdate(0.0, 3, EKF_R_ZUPT); }

void ekfAckermannUpdate(int servo_angle, float dt) {
  if (dt <= 0.0 || abs(ekf_x[3]) < 0.01)
    return;

  // Convert actual servo angle to steering angle in radians
  float steering_deg;
  if (servo_angle >= SERVO_CENTER) {
    // Servo above center = turn right (negative steering)
    steering_deg = -(float)(servo_angle - SERVO_CENTER) /
                   (float)(SERVO_MAX - SERVO_CENTER) * MAX_STEERING_ANGLE_DEG;
  } else {
    // Servo below center = turn left (positive steering)
    steering_deg = (float)(SERVO_CENTER - servo_angle) /
                   (float)(SERVO_CENTER - SERVO_MIN) * MAX_STEERING_ANGLE_DEG;
  }
  float steering_rad = steering_deg * DEG_TO_RAD;

  // Clamp steering
  float max_steer = MAX_STEERING_ANGLE_DEG * DEG_TO_RAD;
  if (steering_rad > max_steer)
    steering_rad = max_steer;
  if (steering_rad < -max_steer)
    steering_rad = -max_steer;

  // Ackermann heading rate
  float omega_ack = ekf_x[3] * tan(steering_rad) / WHEELBASE;
  // Expected heading = pre-predict theta + ackermann rate * dt
  float theta_expected = ekf_theta_pre + omega_ack * dt;

  // Update heading state
  ekfScalarUpdate(theta_expected, 2, EKF_R_ACKERMANN);
}

// ==================== Encoder ISR ====================
void encoderISR_A() {
  if (digitalRead(PIN_ENC_A) != digitalRead(PIN_ENC_B))
    encoderTicks++;
  else
    encoderTicks--;
}

void encoderISR_B() {
  if (digitalRead(PIN_ENC_A) == digitalRead(PIN_ENC_B))
    encoderTicks++;
  else
    encoderTicks--;
}

// ==================== IMU Functions ====================
void initIMU() {
  Wire.begin();
  Wire.setClock(400000);

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);
  delay(100);

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x1B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x1A);
  Wire.write(0x03);
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x75);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 1);
  if (Wire.available()) {
    uint8_t whoami = Wire.read();
    if (whoami == 0x68 || whoami == 0x98) {
      long zSum = 0;
      int samples = 500; // More samples = better bias calibration
      for (int i = 0; i < samples; i++) {
        Wire.beginTransmission(MPU6050_ADDR);
        Wire.write(0x47);
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
    } else {
      Serial.print("IMU_ERR:WHO_AM_I=0x");
      Serial.println(whoami, HEX);
    }
  } else {
    Serial.println("IMU_ERR:NO_RESPONSE");
  }
}

void readIMU() {
  if (!imuReady)
    return;

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 14, true);

  if (Wire.available() < 14)
    return;

  int16_t rawAccX = (Wire.read() << 8) | Wire.read();
  int16_t rawAccY = (Wire.read() << 8) | Wire.read();
  int16_t rawAccZ = (Wire.read() << 8) | Wire.read();
  Wire.read();
  Wire.read(); // skip temp

  int16_t rawGyroX = (Wire.read() << 8) | Wire.read();
  int16_t rawGyroY = (Wire.read() << 8) | Wire.read();
  int16_t rawGyroZ = (Wire.read() << 8) | Wire.read();

  imuAccelX = (rawAccX / ACCEL_SCALE) * G_TO_MS2;
  imuAccelY = (rawAccY / ACCEL_SCALE) * G_TO_MS2;
  imuAccelZ = (rawAccZ / ACCEL_SCALE) * G_TO_MS2;

  imuGyroX = (rawGyroX / GYRO_SCALE) * DEG_TO_RAD;
  imuGyroY = (rawGyroY / GYRO_SCALE) * DEG_TO_RAD;

  float correctedGyroZ = rawGyroZ - gyroZOffset;
  imuGyroZ = -(correctedGyroZ / GYRO_SCALE) * DEG_TO_RAD;
}

void updateYaw(float dt) {
  if (!imuReady || dt <= 0.0)
    return;
  float gyroYawRate = imuGyroZ / DEG_TO_RAD;
  imuYaw += gyroYawRate * dt;
  while (imuYaw > 180.0)
    imuYaw -= 360.0;
  while (imuYaw < -180.0)
    imuYaw += 360.0;
}

// ==================== PI Controller ====================
int computePI(float targetRPM, float actualRPM, float dt) {
  if (dt <= 0.0)
    return 0;
  float error = targetRPM - actualRPM;
  piIntegral += error * dt;
  piIntegral = constrain(piIntegral, -PI_INTEGRAL_MAX, PI_INTEGRAL_MAX);
  float output = PI_KP * error + PI_KI * piIntegral;
  int pwm = (int)abs(output);
  pwm = constrain(pwm, 0, 255);
  if (abs(targetRPM) > 0.1 && pwm < PI_PWM_MIN && pwm > 0)
    pwm = PI_PWM_MIN;
  return pwm;
}

void resetPI() {
  piIntegral = 0.0;
  piTargetRPM = 0.0;
}

// ==================== Setup ====================
void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(PIN_EN, OUTPUT);
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_ENC_A, INPUT_PULLUP);
  pinMode(PIN_ENC_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_A), encoderISR_A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_B), encoderISR_B, CHANGE);

  steeringServo.attach(PIN_SERVO);
  initIMU();
  ekfInit();

  stopMotor();
  steeringServo.write(SERVO_CENTER);

  lastRPMTime = millis();
  lastFBTime = millis();
  lastIMUTime = millis();

  Serial.println("LOW_LEVEL_READY");
}

// ==================== Main Loop ====================
void loop() {
  unsigned long now = millis();

  // --- Read serial data ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (bufferIndex > 0) {
        serialBuffer[bufferIndex] = '\0';
        processCommand(serialBuffer);
        bufferIndex = 0;
      }
    } else {
      if (bufferIndex < SERIAL_BUFFER_SIZE - 1)
        serialBuffer[bufferIndex++] = c;
      else
        bufferIndex = 0;
    }
  }

  // --- Read IMU + EKF Predict (100Hz) ---
  if (now - lastIMUTime >= IMU_INTERVAL_MS) {
    float imuDt = (now - lastIMUTime) / 1000.0;
    lastIMUTime = now;
    readIMU();
    updateYaw(imuDt);

    // EKF Predict: use bias-corrected gyro_z
    // NOTE: imuGyroZ is negated once in readIMU(). The Pi LLC negated it
    // AGAIN before feeding the Pi EKF. We must negate here to match
    // the REP-103 convention (CCW = positive yaw rate).
    // When stationary, feed zero gyro to prevent heading drift from
    // residual bias not captured by the 200-sample boot calibration.
    bool stationary = (abs(ekf_encoder_velocity) < ZUPT_VEL_THRESH);
    float gyro_for_ekf = stationary ? 0.0 : (-imuGyroZ);
    ekfPredict(gyro_for_ekf, imuDt);

    // EKF Ackermann heading update: uses ACTUAL servo angle
    // Skip when stationary (no meaningful heading info from steering)
    if (!stationary) {
      ekfAckermannUpdate(lastServoAngle, imuDt);
    }

    // ZUPT when nearly stationary: clamp velocity AND heading
    if (stationary) {
      ekfZUPT();
      // Also lock heading — car can't turn when not moving
      ekfScalarUpdate(ekf_x[2], 2, EKF_R_ZUPT);
    }
  }

  // --- Calculate RPM from encoder ---
  if (now - lastRPMTime >= RPM_INTERVAL_MS) {
    noInterrupts();
    long ticks = encoderTicks;
    interrupts();

    long deltaTicks = ticks - lastRPMTicks;
    float dt = (now - lastRPMTime) / 1000.0;
    lastRPMTicks = ticks;
    lastRPMTime = now;

    if (dt > 0)
      currentRPM = ((float)abs(deltaTicks) / (float)ENCODER_CPR) * (60.0 / dt);

    // Compute encoder velocity for EKF
    if (dt > 0) {
      float wheelCirc = 2.0 * PI * WHEEL_RADIUS_M;
      ekf_encoder_velocity = (currentRPM * wheelCirc) / 60.0;
      // Apply direction from encoder delta
      if (deltaTicks < 0)
        ekf_encoder_velocity = -ekf_encoder_velocity;
    }

    // EKF velocity measurement update
    ekfUpdateVelocity(ekf_encoder_velocity);

    // PI control
    if (piEnabled) {
      float signedRPM = (deltaTicks >= 0) ? currentRPM : -currentRPM;
      int pwm = computePI(piTargetRPM, signedRPM, dt);
      if (abs(piTargetRPM) < 0.1) {
        stopMotor();
        resetPI();
      } else {
        if (piTargetRPM >= 0) {
          digitalWrite(PIN_IN1, HIGH);
          digitalWrite(PIN_IN2, LOW);
        } else {
          digitalWrite(PIN_IN1, LOW);
          digitalWrite(PIN_IN2, HIGH);
        }
        analogWrite(PIN_EN, pwm);
        currentPWM = pwm;
      }
    }
  }

  // --- Send feedback (20Hz) ---
  if (now - lastFBTime >= FB_INTERVAL_MS) {
    sendFeedback();
    lastFBTime = now;
  }
}

// ==================== Process Command ====================
void processCommand(const char *cmd) {
  int mode = 0;
  int value = 0;
  int servoAngle = SERVO_CENTER;

  int parsed = sscanf(cmd, "%d,%d,%d", &mode, &value, &servoAngle);
  if (parsed != 3) {
    Serial.print("ERR_PARSE:");
    Serial.println(cmd);
    return;
  }

  servoAngle = constrain(servoAngle, SERVO_MIN, SERVO_MAX);

  if (mode == 0) {
    piEnabled = false;
    resetPI();
    int direction = (value >= 0) ? 1 : 0;
    int pwm = constrain(abs(value), 0, 255);
    if (pwm == 0) {
      stopMotor();
    } else {
      if (direction == 1) {
        digitalWrite(PIN_IN1, HIGH);
        digitalWrite(PIN_IN2, LOW);
      } else {
        digitalWrite(PIN_IN1, LOW);
        digitalWrite(PIN_IN2, HIGH);
      }
      analogWrite(PIN_EN, pwm);
    }
    currentDirection = direction;
    currentPWM = pwm;
  } else if (mode == 1) {
    piEnabled = true;
    piTargetRPM = value / 10.0;
    if (abs(piTargetRPM) < 0.1) {
      stopMotor();
      resetPI();
      piEnabled = false;
    }
    currentDirection = (piTargetRPM >= 0) ? 1 : 0;
  } else {
    Serial.print("ERR_MODE:");
    Serial.println(mode);
    return;
  }

  // Apply servo and remember the actual angle
  steeringServo.write(servoAngle);
  lastServoAngle = servoAngle;

  Serial.println("OK");
}

// ==================== Send Feedback ====================
void sendFeedback() {
  noInterrupts();
  long ticks = encoderTicks;
  interrupts();

  // Original fields
  Serial.print("FB:");
  Serial.print(currentRPM, 1);
  Serial.print(",");
  Serial.print(ticks);

  if (imuReady) {
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

  // Shaft angle
  float angle_deg = ((float)ticks / ENCODER_CPR) * 360.0;
  Serial.print(",");
  Serial.print(angle_deg, 2);

  // EKF state fields (appended)
  Serial.print(",");
  Serial.print(ekf_x[0], 4); // x
  Serial.print(",");
  Serial.print(ekf_x[1], 4); // y
  Serial.print(",");
  Serial.print(ekf_x[2], 4); // theta (rad)
  Serial.print(",");
  Serial.print(ekf_x[3], 4); // v (m/s)

  Serial.println();

  Serial.print("PWM:");
  Serial.println(currentPWM);
}

// ==================== Helper ====================
void stopMotor() {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);
  analogWrite(PIN_EN, 0);
  currentPWM = 0;
}
