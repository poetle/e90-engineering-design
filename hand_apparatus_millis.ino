#include  <Servo.h>
#include "Adafruit_BNO08x_RVC.h" // UART-RVC mode

Adafruit_BNO08x_RVC rvc = Adafruit_BNO08x_RVC();

// declare class instances
Servo wrist; // roll
Servo thumb; // pitch

// define sweep
// tested input angle range of both servos: 0-180 degrees
const int wrist_angle_i = 25; // 25,
const int wrist_angle_f = 65; // 65
const int thumb_angle_i = 0; // 105,
const int thumb_angle_f = 150; // 150

const unsigned long wrist_rote_event = 120;
const unsigned long thumb_rote_event = 120;
unsigned long prev_time_wrist = 0; // for millis()
unsigned long prev_time_thumb = 0; // for millis()

bool to_fro_wrist = true;
bool to_fro_thumb = true;

void setup() { // runs once
  // inertial measurement unit (IMU) sensor
  // await for serial monitor to open
  Serial.begin(115200); // baud rate specified by datasheet
  while (!Serial)
    delay(10);

  if (!rvc.begin(&Serial)) { // connect to sensor over hardware serial
    while (1)
      delay(10);
  }
  // print label values
  Serial.print(F("Yaw"));
  Serial.print(F("\tPitch"));
  Serial.print(F("\tRoll"));
  Serial.print(F("\tx"));
  Serial.print(F("\ty"));
  Serial.println(F("\tz"));

  // hand apparatus, servos
  wrist.attach(5); // assign servos to pins
  thumb.attach(6);
  wrist.write(wrist_angle_i);
  thumb.write(thumb_angle_i);
}

void loop() {
  // non-blocking approach
  unsigned long current_time = millis();

  // inertial measurement unit
  BNO08x_RVC_Data heading;

  if (!rvc.read(&heading)) {
    // data not available or parsable, keep trying
    return;
  }

  Serial.print(heading.yaw);Serial.print(F(","));
  Serial.print(heading.pitch);Serial.print(F(","));
  Serial.print(heading.roll);Serial.print(F(","));
  Serial.print(heading.x_accel);Serial.print(F(","));
  Serial.print(heading.y_accel);Serial.print(F(","));
  Serial.print(heading.z_accel);
  Serial.println("");

  // hand apparatus, servos
  // wrist event
  if (current_time - prev_time_wrist >= wrist_rote_event) {
    if (to_fro_wrist) {
      wrist.write(wrist_angle_i);
    }
    else {
      wrist.write(wrist_angle_f);
    }
    prev_time_wrist = current_time;
    to_fro_wrist = !to_fro_wrist;
    Serial.print(to_fro_wrist);
  }

  // thumb event
  if (current_time - prev_time_thumb >= thumb_rote_event) {
    if (to_fro_thumb) {
      thumb.write(thumb_angle_i);
    }
    else {
      thumb.write(thumb_angle_f);
    }
    prev_time_thumb = current_time;
    to_fro_thumb = !to_fro_thumb;
  }
}