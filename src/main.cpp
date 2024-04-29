/*
 * Copyright (c) 2023 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS
 * IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Include ----------------------------------------------------------------- */
#include "Particle.h"
#include <EnergyMonitor_inferencing.h>
#include <Wire.h>

SYSTEM_THREAD(ENABLED);

// defines global
typedef enum
{
    IDLE,
    WAIT_FOR_EDGE,
    CAPTURE_DATA,
    COPY_DATA,
    RUN_INFERENCE,
    POST_PROCESS,
    DISPLAY_RESULTS
} state_t;

#define ADC_COUNTS (1 << 12) // ADC bits
#define NUM_SAMPLES 40       // number of samples per event
#define SAMPLE_PERIOD 1500   // Sampling interval for each event in milliseconds

// Pin definitions
int LED_B = D9;        // blinker on custom analog front end PCB
int voltagePIN = D13;//A2;   // voltage is on A2
int currentPIN = A5;   // current is pn A5
int zeroCrossPIN = S3; // zero cross, digital signal on A5

// Forward declerations
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr);
void setup();
void loop();
void sampleData(void);
void zeroCrossing(void);
void setGain(uint8_t gainValue);
void fsm(void);

// objects
//SerialLogHandler logHandler(LOG_LEVEL_ERROR);
Timer sample(1, sampleData);

// global vaiables
//static const float features[] = {2059, 2061, 2049, 2051, 1992, 2005, 2017, 2018, 2059, 2076, 2084, 2084, 2154, 2112, 2089, 2149, 2135, 2125, 2056, 2059, 2061, 2059, 2013, 1958, 2020, 1980, 1992, 2009, 2057, 2072, 2078, 2129, 2165, 2184, 2182, 2152, 2138, 2103, 2061, 2058};

int32_t voltageADC = 0;
int32_t currentADC = 0;
int payloadBuffer[2][NUM_SAMPLES];
int sampleIndex = 0;
double filteredCurrent = 0;
double offsetCurrent = 0;
float sumCurrent = 0;
double squareCurrent = 0;
float rmsCurrent = 0;
bool sendData = false;
bool firstSample = true;
bool samplingArmed = false;
bool edgeDetected = false;
bool keyCPressed = false;
double gridFreq = 0.0;
double usSample;
double usSampleStart;
unsigned long now;
double CURRENT_CAL = 0.027566;

state_t state, state_next = IDLE;

/**
 * @brief      Copy raw feature data in out_ptr
 *             Function called by inference library
 *
 * @param[in]  offset   The offset
 * @param[in]  length   The length
 * @param      out_ptr  The out pointer
 *
 * @return     0
 */
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr)
{
    int out_ptr_ix = 0;

    for (int j = 0; j < NUM_SAMPLES; j++)
    {
        //out_ptr[out_ptr_ix++] = (float)(payloadBuffer[0][j]); //voltage unused
        out_ptr[out_ptr_ix++] = (float)(payloadBuffer[1][j]); 
    }
    return 0;
}

void print_inference_result(ei_impulse_result_t result);

/**
 * @brief      Particle setup function
 */
void setup()
{
    waitFor(Serial.isConnected, 15000);
    delay(2000);
    ei_printf("Edge Impulse inference runner for Particle devices\r\n");
    attachInterrupt(zeroCrossPIN, zeroCrossing, FALLING);
    setGain(245);
}

/**
 * @brief      Particle main function
 */
void loop()
{
    fsm();
}

void print_inference_result(ei_impulse_result_t result)
{

    // Print how long it took to perform inference
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
              result.timing.dsp,
              result.timing.classification,
              result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++)
    {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0)
        {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                  bb.label,
                  bb.value,
                  bb.x,
                  bb.y,
                  bb.width,
                  bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
    {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif
}
void zeroCrossing(void)
{
    static unsigned long lastUs, nowUs;
    unsigned long periodUs;
    lastUs = nowUs;
    nowUs = micros();
    periodUs = nowUs - lastUs;
    gridFreq = (double)1000000.0 / periodUs;
    edgeDetected = true;
    // start sampling here to ensure allignment with zero crossing
    if (samplingArmed == true)
    {
        sample.reset();
        samplingArmed = false;
    }
}

/**
 * @brief collect data an 1Khz called from timer interruot
 *
 */
void sampleData(void)
{
    // usSampleStart = micros();
    voltageADC = analogRead(voltagePIN); // read ADC values
    currentADC = analogRead(currentPIN);
    payloadBuffer[0][sampleIndex] = voltageADC;
    payloadBuffer[1][sampleIndex] = currentADC;
    sampleIndex++;

    filteredCurrent = currentADC - 2048; // 2048 should be replaced by Vref
    squareCurrent = filteredCurrent * filteredCurrent;
    sumCurrent += squareCurrent;

    if (sampleIndex == NUM_SAMPLES) // we are done sampling for this event
    {
        sampleIndex = 0;
        sample.stop();
        rmsCurrent = sqrt(sumCurrent / NUM_SAMPLES);
        rmsCurrent -= 8;
        if (rmsCurrent < 0)
        {
            rmsCurrent = 0;
        }
        rmsCurrent *= CURRENT_CAL;
        sumCurrent = 0;
        sendData = true;
    }
    // else
    //   usSample =  micros() - usSampleStart;
}

void setGain(uint8_t gainValue)
{
  char gainRet = 0;
  Wire.beginTransmission(0x2f);
  Wire.write(0x00);
  Wire.write(gainValue);
  Wire.endTransmission();
  delay(5);
  Wire.requestFrom(0x2f, 1);
  gainRet = Wire.read();
  Wire.endTransmission();
  if (gainRet == gainValue)
  {
    ei_printf("SUCCESS SETTING GAIN: %d", gainRet);
  }
  else
  {
    ei_printf("FAILED TO SET GAIN: %d", gainRet);
  }
}

void fsm(void)
{
    switch (state)
    {
    case IDLE:
        state_next = WAIT_FOR_EDGE;
        break;

    case WAIT_FOR_EDGE:
        // arm sampling trigger, set in ISR to oensure allignemt with zero crossing
        samplingArmed = true;
        if (edgeDetected == true)
        {
            edgeDetected = false;
            state_next = CAPTURE_DATA;
        }
        break;

    case CAPTURE_DATA:
        if (sendData == true)
            state_next = COPY_DATA;
        break;

    case COPY_DATA:
        // copy function setup as callback...
        state_next = RUN_INFERENCE;
        break;

    case RUN_INFERENCE:

        ei_impulse_result_t result = {0};
        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = 40;//sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;
        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);
        if (res != EI_IMPULSE_OK)
        {
            ei_printf("ERR: Failed to run classifier (%d)\n", res);
            break;
        }
        // print inference return code
        ei_printf("run_classifier returned: %d\r\n", res);
        print_inference_result(result);
        state_next = IDLE;
        delay(1000);
        break;
    }

    if (state != state_next)
    {
        ei_printf("State: %d\r\n", state);
    }

    state = state_next;
}