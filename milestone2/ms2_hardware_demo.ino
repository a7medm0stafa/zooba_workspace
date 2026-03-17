char command;

int EN = 3;   // Enable pin (PWM speed)
int IN1 = 7;  // Direction pin
int IN2 = 8;  // Direction pin

void setup()
{
  Serial.begin(9600);

  pinMode(EN, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // Motor initially stopped
  stopMotor();
}

void loop()
{
  if (Serial.available())
  {
    command = Serial.read();

    // ignore newline characters
    if (command == '\n' || command == '\r')
      return;

    // RED → STOP
    if (command == 'R')
    {
      stopMotor();
    }

    // YELLOW → MOVE
    if (command == 'Y')
    {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(EN, 80);
    } 
    }

    // GREEN → MOVE
    if (command == 'G')
    {
      moveMotor();
    }
  }
}

void moveMotor()
{
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(EN, 150);
}

void stopMotor()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(EN, 0);
}