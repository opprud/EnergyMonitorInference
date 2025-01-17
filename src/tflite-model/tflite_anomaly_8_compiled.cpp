/* Generated by Edge Impulse
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
// Generated on: 14.04.2024 21:00:30

#include <stdio.h>
#include <stdlib.h>
#include "edge-impulse-sdk/tensorflow/lite/c/builtin_op_data.h"
#include "edge-impulse-sdk/tensorflow/lite/c/common.h"
#include "edge-impulse-sdk/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "edge-impulse-sdk/porting/ei_classifier_porting.h"

#if EI_CLASSIFIER_PRINT_STATE
#if defined(__cplusplus) && EI_C_LINKAGE == 1
extern "C" {
    extern void ei_printf(const char *format, ...);
}
#else
extern void ei_printf(const char *format, ...);
#endif
#endif

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#elif defined __ICCARM__
#define ALIGN(x) __attribute__((aligned(x)))
#endif

#ifndef EI_MAX_SCRATCH_BUFFER_COUNT
#ifndef CONFIG_IDF_TARGET_ESP32S3
#define EI_MAX_SCRATCH_BUFFER_COUNT 4
#else
#define EI_MAX_SCRATCH_BUFFER_COUNT 8
#endif // CONFIG_IDF_TARGET_ESP32S3
#endif // EI_MAX_SCRATCH_BUFFER_COUNT

#ifndef EI_MAX_OVERFLOW_BUFFER_COUNT
#define EI_MAX_OVERFLOW_BUFFER_COUNT 10
#endif // EI_MAX_OVERFLOW_BUFFER_COUNT

using namespace tflite;
using namespace tflite::ops;
using namespace tflite::ops::micro;

namespace {

#if defined(EI_CLASSIFIER_ALLOCATION_STATIC_HIMAX) || defined(EI_CLASSIFIER_ALLOCATION_STATIC_HIMAX_GNU)
constexpr int kTensorArenaSize = 2080;
#else
constexpr int kTensorArenaSize = 1056;
#endif

#if defined(EI_CLASSIFIER_ALLOCATION_STATIC)
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);
#elif defined(EI_CLASSIFIER_ALLOCATION_STATIC_HIMAX)
#pragma Bss(".tensor_arena")
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);
#pragma Bss()
#elif defined(EI_CLASSIFIER_ALLOCATION_STATIC_HIMAX_GNU)
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16) __attribute__((section(".tensor_arena")));
#else
#define EI_CLASSIFIER_ALLOCATION_HEAP 1
uint8_t* tensor_arena = NULL;
#endif

static uint8_t* tensor_boundary;
static uint8_t* current_location;

template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};

enum used_operators_e {
  OP_FULLY_CONNECTED, OP_SUB, OP_MUL, OP_RESHAPE, OP_SUM, OP_ADD, OP_REDUCE_MAX, OP_EXP, OP_LOG, OP_DIV, OP_ABS,  OP_LAST
};

struct TensorInfo_t { // subset of TfLiteTensor used for initialization from constant memory
  TfLiteAllocationType allocation_type;
  TfLiteType type;
  void* data;
  TfLiteIntArray* dims;
  size_t bytes;
};

typedef struct {
  TfLiteTensor tensor;
  int16_t index;
} TfLiteTensorWithIndex;

typedef struct {
  TfLiteEvalTensor tensor;
  int16_t index;
} TfLiteEvalTensorWithIndex;

TfLiteContext ctx{};
static const int MAX_TFL_TENSOR_COUNT = 3;
static TfLiteTensorWithIndex tflTensors[MAX_TFL_TENSOR_COUNT];
static const int MAX_TFL_EVAL_COUNT = 3;
static TfLiteEvalTensorWithIndex tflEvalTensors[MAX_TFL_EVAL_COUNT];
TfLiteRegistration registrations[OP_LAST];

namespace g0 {
const TfArray<2, int> tensor_dimension0 = { 2, { 1,9 } };
const ALIGN(16) float tensor_data1[36] = { -0.89872878789901733, 23.821197509765625, 3.732879638671875, 0.63457489013671875, 1.0355010032653809, -1.3274173736572266, 0.56094551086425781, -0.20310664176940918, -2.2849559783935547, 0.65175461769104004, 10.059597969055176, 1.0972938537597656, 2.6150650978088379, 0.38352680206298828, -0.93059110641479492, -0.64655351638793945, 1.8192358016967773, -0.70028114318847656, 2.0134093761444092, 2.6880817413330078, 0.39969158172607422, 0.97265625, 0.08287811279296875, 1.0919218063354492, 0.18310546875, 1.0030717849731445, -0.076694488525390625, 1.8502830266952515, 13.300388336181641, 2.0414962768554688, 0.49666309356689453, 0.078160285949707031, -2.4489364624023438, 2.9488906860351562, 1.0815532207489014, 0.33674478530883789, };
const TfArray<1, int> tensor_dimension1 = { 1, { 36 } };
const ALIGN(16) float tensor_data2[1*4] = { 
  11.288955688476562, 9.4367647171020508, 11.498250961303711, 9.0497598648071289, 
};
const TfArray<2, int> tensor_dimension2 = { 2, { 1,4 } };
const ALIGN(8) int32_t tensor_data3[2] = { 1, 1, };
const TfArray<1, int> tensor_dimension3 = { 1, { 2 } };
const ALIGN(4) int32_t tensor_data4[1] = { 1, };
const TfArray<1, int> tensor_dimension4 = { 1, { 1 } };
const ALIGN(4) int32_t tensor_data5[1] = { 2, };
const TfArray<1, int> tensor_dimension5 = { 1, { 1 } };
const ALIGN(8) int32_t tensor_data6[3] = { 1, 4, 9, };
const TfArray<1, int> tensor_dimension6 = { 1, { 3 } };
const ALIGN(16) float tensor_data7[36*9] = { 
  2.2871317863464355, 0, 0, 0, 0, 0, 0, 0, 0, 
  1.6395131349563599, 6.0577116012573242, 0, 0, 0, 0, 0, 0, 0, 
  0.062431361526250839, -63.762931823730469, 64.692779541015625, 0, 0, 0, 0, 0, 0, 
  0.0010211537592113018, 20.606334686279297, -21.620540618896484, 2.0735907554626465, 0, 0, 0, 0, 0, 
  -0.22084502875804901, -22.214757919311523, 21.948442459106445, -1.4143540859222412, 2.3627328872680664, 0, 0, 0, 0, 
  -0.73493891954421997, 4.5266618728637695, -6.7411055564880371, 0.49110716581344604, 0.17967088520526886, 2.5323164463043213, 0, 0, 0, 
  0.024201894178986549, 20.781990051269531, -20.463603973388672, -0.015152338892221451, 0.078725375235080719, -4.9964370727539062, 4.977959156036377, 0, 0, 
  -0.21073594689369202, -1.5850856304168701, 0.76437121629714966, -0.085103832185268402, -0.16394738852977753, 0.17398126423358917, -0.13842751085758209, 1.9214478731155396, 0, 
  -0.20653359591960907, 7.7369780540466309, -8.6380596160888672, 0.037381876260042191, -0.2479289174079895, 0.38309305906295776, -0.1835031658411026, -2.425480842590332, 3.0751953125, 
  0.22721928358078003, 0, 0, 0, 0, 0, 0, 0, 0, 
  0.015791932120919228, 3.0729708671569824, 0, 0, 0, 0, 0, 0, 0, 
  -0.021566510200500488, -18.643856048583984, 19.047542572021484, 0, 0, 0, 0, 0, 0, 
  0.021254712715744972, 0.98272424936294556, -2.1321115493774414, 2.663231372833252, 0, 0, 0, 0, 0, 
  -0.012567002326250076, -3.1560816764831543, 3.075639009475708, -5.2196345329284668, 5.6435484886169434, 0, 0, 0, 0, 
  -0.1185472384095192, -1.8200082778930664, 1.1904753446578979, -0.39058256149291992, -0.26761865615844727, 1.6977274417877197, 0, 0, 0, 
  -0.030656389892101288, 1.4235539436340332, -1.7940559387207031, -0.12254549562931061, -0.050271540880203247, -2.7575323581695557, 3.2756226062774658, 0, 0, 
  -0.079421848058700562, -1.3000184297561646, -0.62211567163467407, -0.19524626433849335, 0.62196856737136841, -0.25319099426269531, -0.29120826721191406, 3.4850506782531738, 0, 
  -0.013013571500778198, 2.4785289764404297, -2.7509353160858154, 0.59763491153717041, -0.85353368520736694, 0.18240842223167419, -0.09677521139383316, -10.123381614685059, 10.419156074523926, 
  0.11063149571418762, 0, 0, 0, 0, 0, 0, 0, 0, 
  -0.0037105563096702099, 1.1637910604476929, 0, 0, 0, 0, 0, 0, 0, 
  -0.0038661821745336056, -6.6957893371582031, 6.8850603103637695, 0, 0, 0, 0, 0, 0, 
  -0.031127724796533585, -2.2148268222808838, -1.3358956575393677, 4.2740354537963867, 0, 0, 0, 0, 0, 
  -0.0089571801945567131, 6.5854511260986328, -6.3777155876159668, -14.916204452514648, 14.981056213378906, 0, 0, 0, 0, 
  -0.026361672207713127, 0.59693872928619385, 0.11608201265335083, -4.7055158615112305, -0.57712686061859131, 5.291285514831543, 0, 0, 0, 
  0.0053900494240224361, 0.35890400409698486, -0.62497597932815552, 7.5665555000305176, -8.1957483291625977, -18.025562286376953, 18.892879486083984, 0, 0, 
  -0.0032039589714258909, -0.89314031600952148, -0.57988953590393066, 0.84812247753143311, -0.087427236139774323, -4.7024884223937988, -0.041401904076337814, 5.8707327842712402, 0, 
  -0.0032459753565490246, -3.7863376140594482, 3.9631516933441162, 4.0872406959533691, -4.322300910949707, 12.307311058044434, -12.254754066467285, -15.891567230224609, 15.877071380615234, 
  0.43073546886444092, 0, 0, 0, 0, 0, 0, 0, 0, 
  0.20563529431819916, 3.1065633296966553, 0, 0, 0, 0, 0, 0, 0, 
  0.020595362409949303, -30.830684661865234, 31.254068374633789, 0, 0, 0, 0, 0, 0, 
  0.056695125997066498, 4.8347702026367188, -5.8596954345703125, 1.7324504852294922, 0, 0, 0, 0, 0, 
  0.01490104291588068, -1.3256012201309204, 0.8378104567527771, -1.4453831911087036, 2.2362945079803467, 0, 0, 0, 0, 
  -0.16407530009746552, 8.6291389465332031, -11.521943092346191, 0.025203976780176163, -0.16597943007946014, 2.8655457496643066, 0, 0, 0, 
  0.00039842264959588647, 7.9160823822021484, -7.7654614448547363, -0.12892451882362366, 0.16639082133769989, -17.209186553955078, 17.873756408691406, 0, 0, 
  -0.034562453627586365, -0.8707771897315979, 0.32218709588050842, 0.071730844676494598, -0.01473679207265377, -0.2781863808631897, 0.056516323238611221, 1.6545107364654541, 0, 
  -0.0057030767202377319, 1.2259980440139771, -1.8393739461898804, 0.10260915756225586, -0.01859445683658123, -1.8121200799942017, 1.8738847970962524, -1.4755396842956543, 2.3984978199005127, 
};
const TfArray<2, int> tensor_dimension7 = { 2, { 36,9 } };
const ALIGN(16) float tensor_data8[1*4] = { 
  1, 1, 1, 1, 
};
const TfArray<2, int> tensor_dimension8 = { 2, { 1,4 } };
const ALIGN(16) float tensor_data9[1*4] = { 
  -0.5, -0.5, -0.5, -0.5, 
};
const TfArray<2, int> tensor_dimension9 = { 2, { 1,4 } };
const ALIGN(16) float tensor_data10[1*4] = { 
  16.5408935546875, 16.5408935546875, 16.5408935546875, 16.5408935546875, 
};
const TfArray<2, int> tensor_dimension10 = { 2, { 1,4 } };
const ALIGN(4) float tensor_data11[1*1] = { 
  4.750669002532959, 
};
const TfArray<2, int> tensor_dimension11 = { 2, { 1,1 } };
const ALIGN(4) float tensor_data12[1*1] = { 
  -2.5330245494842529, 
};
const TfArray<2, int> tensor_dimension12 = { 2, { 1,1 } };
const TfArray<2, int> tensor_dimension13 = { 2, { 1,36 } };
const TfArray<2, int> tensor_dimension14 = { 2, { 1,36 } };
const TfArray<2, int> tensor_dimension15 = { 2, { 1,36 } };
const TfArray<3, int> tensor_dimension16 = { 3, { 1,4,9 } };
const TfArray<2, int> tensor_dimension17 = { 2, { 1,4 } };
const TfArray<2, int> tensor_dimension18 = { 2, { 1,4 } };
const TfArray<2, int> tensor_dimension19 = { 2, { 1,4 } };
const TfArray<2, int> tensor_dimension20 = { 2, { 1,4 } };
const TfArray<1, int> tensor_dimension21 = { 1, { 1 } };
const TfArray<2, int> tensor_dimension22 = { 2, { 1,1 } };
const TfArray<2, int> tensor_dimension23 = { 2, { 1,4 } };
const TfArray<2, int> tensor_dimension24 = { 2, { 1,4 } };
const TfArray<2, int> tensor_dimension25 = { 2, { 1,4 } };
const TfArray<1, int> tensor_dimension26 = { 1, { 1 } };
const TfArray<1, int> tensor_dimension27 = { 1, { 1 } };
const TfArray<1, int> tensor_dimension28 = { 1, { 1 } };
const TfArray<2, int> tensor_dimension29 = { 2, { 1,1 } };
const TfArray<2, int> tensor_dimension30 = { 2, { 1,1 } };
const TfArray<2, int> tensor_dimension31 = { 2, { 1,1 } };
const TfArray<2, int> tensor_dimension32 = { 2, { 1,1 } };
const TfLiteFullyConnectedParams opdata0 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs0 = { 3, { 0,7,-1 } };
const TfArray<1, int> outputs0 = { 1, { 13 } };
const TfLiteSubParams opdata1 = { kTfLiteActNone };
const TfArray<2, int> inputs1 = { 2, { 13,1 } };
const TfArray<1, int> outputs1 = { 1, { 14 } };
const TfLiteMulParams opdata2 = { kTfLiteActNone };
const TfArray<2, int> inputs2 = { 2, { 14,14 } };
const TfArray<1, int> outputs2 = { 1, { 15 } };
const TfLiteReshapeParams opdata3 = { { 0, 0, 0, 0, 0, 0, 0, 0, }, 0 };
const TfArray<2, int> inputs3 = { 2, { 15,6 } };
const TfArray<1, int> outputs3 = { 1, { 16 } };
const ALIGN(1) uint8_t opdata4[1] = { 0,  }; /* op type 74=SUM */
const TfArray<2, int> inputs4 = { 2, { 16,5 } };
const TfArray<1, int> outputs4 = { 1, { 17 } };
const TfLiteAddParams opdata5 = { kTfLiteActNone };
const TfArray<2, int> inputs5 = { 2, { 17,10 } };
const TfArray<1, int> outputs5 = { 1, { 18 } };
const TfLiteMulParams opdata6 = { kTfLiteActNone };
const TfArray<2, int> inputs6 = { 2, { 18,9 } };
const TfArray<1, int> outputs6 = { 1, { 19 } };
const TfLiteAddParams opdata7 = { kTfLiteActNone };
const TfArray<2, int> inputs7 = { 2, { 19,2 } };
const TfArray<1, int> outputs7 = { 1, { 20 } };
const ALIGN(1) uint8_t opdata8[1] = { 0,  }; /* op type 82=REDUCE_MAX */
const TfArray<2, int> inputs8 = { 2, { 20,4 } };
const TfArray<1, int> outputs8 = { 1, { 21 } };
const TfLiteReshapeParams opdata9 = { { 0, 0, 0, 0, 0, 0, 0, 0, }, 0 };
const TfArray<2, int> inputs9 = { 2, { 21,3 } };
const TfArray<1, int> outputs9 = { 1, { 22 } };
const TfLiteMulParams opdata10 = { kTfLiteActNone };
const TfArray<2, int> inputs10 = { 2, { 22,8 } };
const TfArray<1, int> outputs10 = { 1, { 23 } };
const TfLiteSubParams opdata11 = { kTfLiteActNone };
const TfArray<2, int> inputs11 = { 2, { 20,23 } };
const TfArray<1, int> outputs11 = { 1, { 24 } };
const TfArray<1, int> inputs12 = { 1, { 24 } };
const TfArray<1, int> outputs12 = { 1, { 25 } };
const ALIGN(1) uint8_t opdata13[1] = { 0,  }; /* op type 74=SUM */
const TfArray<2, int> inputs13 = { 2, { 25,4 } };
const TfArray<1, int> outputs13 = { 1, { 26 } };
const TfArray<1, int> inputs14 = { 1, { 26 } };
const TfArray<1, int> outputs14 = { 1, { 27 } };
const TfLiteAddParams opdata15 = { kTfLiteActNone };
const TfArray<2, int> inputs15 = { 2, { 27,21 } };
const TfArray<1, int> outputs15 = { 1, { 28 } };
const TfLiteReshapeParams opdata16 = { { 0, 0, 0, 0, 0, 0, 0, 0, }, 0 };
const TfArray<2, int> inputs16 = { 2, { 28,3 } };
const TfArray<1, int> outputs16 = { 1, { 29 } };
const TfLiteSubParams opdata17 = { kTfLiteActNone };
const TfArray<2, int> inputs17 = { 2, { 29,12 } };
const TfArray<1, int> outputs17 = { 1, { 30 } };
const ALIGN(4) uint8_t opdata18[4] = { 0, 0, 0, 0,  }; /* op type 42=DIV */
const TfArray<2, int> inputs18 = { 2, { 30,11 } };
const TfArray<1, int> outputs18 = { 1, { 31 } };
const TfArray<1, int> inputs19 = { 1, { 31 } };
const TfArray<1, int> outputs19 = { 1, { 32 } };
};

TensorInfo_t tensorData[] = {
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 0), (TfLiteIntArray*)&g0::tensor_dimension0, 36, },
{ kTfLiteMmapRo, kTfLiteFloat32, (int32_t*)g0::tensor_data1, (TfLiteIntArray*)&g0::tensor_dimension1, 144, },
{ kTfLiteMmapRo, kTfLiteFloat32, (int32_t*)g0::tensor_data2, (TfLiteIntArray*)&g0::tensor_dimension2, 16, },
{ kTfLiteMmapRo, kTfLiteInt32, (int32_t*)g0::tensor_data3, (TfLiteIntArray*)&g0::tensor_dimension3, 8, },
{ kTfLiteMmapRo, kTfLiteInt32, (int32_t*)g0::tensor_data4, (TfLiteIntArray*)&g0::tensor_dimension4, 4, },
{ kTfLiteMmapRo, kTfLiteInt32, (int32_t*)g0::tensor_data5, (TfLiteIntArray*)&g0::tensor_dimension5, 4, },
{ kTfLiteMmapRo, kTfLiteInt32, (int32_t*)g0::tensor_data6, (TfLiteIntArray*)&g0::tensor_dimension6, 12, },
{ kTfLiteMmapRo, kTfLiteFloat32, (int32_t*)g0::tensor_data7, (TfLiteIntArray*)&g0::tensor_dimension7, 1296, },
{ kTfLiteMmapRo, kTfLiteFloat32, (int32_t*)g0::tensor_data8, (TfLiteIntArray*)&g0::tensor_dimension8, 16, },
{ kTfLiteMmapRo, kTfLiteFloat32, (int32_t*)g0::tensor_data9, (TfLiteIntArray*)&g0::tensor_dimension9, 16, },
{ kTfLiteMmapRo, kTfLiteFloat32, (int32_t*)g0::tensor_data10, (TfLiteIntArray*)&g0::tensor_dimension10, 16, },
{ kTfLiteMmapRo, kTfLiteFloat32, (int32_t*)g0::tensor_data11, (TfLiteIntArray*)&g0::tensor_dimension11, 4, },
{ kTfLiteMmapRo, kTfLiteFloat32, (int32_t*)g0::tensor_data12, (TfLiteIntArray*)&g0::tensor_dimension12, 4, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 144), (TfLiteIntArray*)&g0::tensor_dimension13, 144, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 0), (TfLiteIntArray*)&g0::tensor_dimension14, 144, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 144), (TfLiteIntArray*)&g0::tensor_dimension15, 144, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 0), (TfLiteIntArray*)&g0::tensor_dimension16, 144, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 144), (TfLiteIntArray*)&g0::tensor_dimension17, 16, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 16), (TfLiteIntArray*)&g0::tensor_dimension18, 16, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 0), (TfLiteIntArray*)&g0::tensor_dimension19, 16, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 48), (TfLiteIntArray*)&g0::tensor_dimension20, 16, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 32), (TfLiteIntArray*)&g0::tensor_dimension21, 4, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 0), (TfLiteIntArray*)&g0::tensor_dimension22, 4, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 16), (TfLiteIntArray*)&g0::tensor_dimension23, 16, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 0), (TfLiteIntArray*)&g0::tensor_dimension24, 16, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 16), (TfLiteIntArray*)&g0::tensor_dimension25, 16, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 0), (TfLiteIntArray*)&g0::tensor_dimension26, 4, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 16), (TfLiteIntArray*)&g0::tensor_dimension27, 4, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 0), (TfLiteIntArray*)&g0::tensor_dimension28, 4, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 16), (TfLiteIntArray*)&g0::tensor_dimension29, 4, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 0), (TfLiteIntArray*)&g0::tensor_dimension30, 4, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 16), (TfLiteIntArray*)&g0::tensor_dimension31, 4, },
{ kTfLiteArenaRw, kTfLiteFloat32, (int32_t*)(tensor_arena + 0), (TfLiteIntArray*)&g0::tensor_dimension32, 4, },
};

#ifndef TF_LITE_STATIC_MEMORY
TfLiteNode tflNodes[20] = {
{ (TfLiteIntArray*)&g0::inputs0, (TfLiteIntArray*)&g0::outputs0, (TfLiteIntArray*)&g0::inputs0, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata0)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs1, (TfLiteIntArray*)&g0::outputs1, (TfLiteIntArray*)&g0::inputs1, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata1)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs2, (TfLiteIntArray*)&g0::outputs2, (TfLiteIntArray*)&g0::inputs2, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata2)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs3, (TfLiteIntArray*)&g0::outputs3, (TfLiteIntArray*)&g0::inputs3, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata3)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs4, (TfLiteIntArray*)&g0::outputs4, (TfLiteIntArray*)&g0::inputs4, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata4)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs5, (TfLiteIntArray*)&g0::outputs5, (TfLiteIntArray*)&g0::inputs5, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata5)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs6, (TfLiteIntArray*)&g0::outputs6, (TfLiteIntArray*)&g0::inputs6, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata6)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs7, (TfLiteIntArray*)&g0::outputs7, (TfLiteIntArray*)&g0::inputs7, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata7)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs8, (TfLiteIntArray*)&g0::outputs8, (TfLiteIntArray*)&g0::inputs8, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata8)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs9, (TfLiteIntArray*)&g0::outputs9, (TfLiteIntArray*)&g0::inputs9, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata9)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs10, (TfLiteIntArray*)&g0::outputs10, (TfLiteIntArray*)&g0::inputs10, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata10)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs11, (TfLiteIntArray*)&g0::outputs11, (TfLiteIntArray*)&g0::inputs11, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata11)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs12, (TfLiteIntArray*)&g0::outputs12, (TfLiteIntArray*)&g0::inputs12, nullptr, nullptr, nullptr, nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs13, (TfLiteIntArray*)&g0::outputs13, (TfLiteIntArray*)&g0::inputs13, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata13)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs14, (TfLiteIntArray*)&g0::outputs14, (TfLiteIntArray*)&g0::inputs14, nullptr, nullptr, nullptr, nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs15, (TfLiteIntArray*)&g0::outputs15, (TfLiteIntArray*)&g0::inputs15, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata15)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs16, (TfLiteIntArray*)&g0::outputs16, (TfLiteIntArray*)&g0::inputs16, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata16)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs17, (TfLiteIntArray*)&g0::outputs17, (TfLiteIntArray*)&g0::inputs17, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata17)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs18, (TfLiteIntArray*)&g0::outputs18, (TfLiteIntArray*)&g0::inputs18, nullptr, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata18)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs19, (TfLiteIntArray*)&g0::outputs19, (TfLiteIntArray*)&g0::inputs19, nullptr, nullptr, nullptr, nullptr, 0, },
};
#else
TfLiteNode tflNodes[20] = {
{ (TfLiteIntArray*)&g0::inputs0, (TfLiteIntArray*)&g0::outputs0, (TfLiteIntArray*)&g0::inputs0, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata0)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs1, (TfLiteIntArray*)&g0::outputs1, (TfLiteIntArray*)&g0::inputs1, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata1)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs2, (TfLiteIntArray*)&g0::outputs2, (TfLiteIntArray*)&g0::inputs2, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata2)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs3, (TfLiteIntArray*)&g0::outputs3, (TfLiteIntArray*)&g0::inputs3, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata3)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs4, (TfLiteIntArray*)&g0::outputs4, (TfLiteIntArray*)&g0::inputs4, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata4)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs5, (TfLiteIntArray*)&g0::outputs5, (TfLiteIntArray*)&g0::inputs5, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata5)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs6, (TfLiteIntArray*)&g0::outputs6, (TfLiteIntArray*)&g0::inputs6, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata6)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs7, (TfLiteIntArray*)&g0::outputs7, (TfLiteIntArray*)&g0::inputs7, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata7)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs8, (TfLiteIntArray*)&g0::outputs8, (TfLiteIntArray*)&g0::inputs8, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata8)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs9, (TfLiteIntArray*)&g0::outputs9, (TfLiteIntArray*)&g0::inputs9, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata9)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs10, (TfLiteIntArray*)&g0::outputs10, (TfLiteIntArray*)&g0::inputs10, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata10)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs11, (TfLiteIntArray*)&g0::outputs11, (TfLiteIntArray*)&g0::inputs11, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata11)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs12, (TfLiteIntArray*)&g0::outputs12, (TfLiteIntArray*)&g0::inputs12, nullptr, nullptr, nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs13, (TfLiteIntArray*)&g0::outputs13, (TfLiteIntArray*)&g0::inputs13, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata13)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs14, (TfLiteIntArray*)&g0::outputs14, (TfLiteIntArray*)&g0::inputs14, nullptr, nullptr, nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs15, (TfLiteIntArray*)&g0::outputs15, (TfLiteIntArray*)&g0::inputs15, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata15)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs16, (TfLiteIntArray*)&g0::outputs16, (TfLiteIntArray*)&g0::inputs16, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata16)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs17, (TfLiteIntArray*)&g0::outputs17, (TfLiteIntArray*)&g0::inputs17, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata17)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs18, (TfLiteIntArray*)&g0::outputs18, (TfLiteIntArray*)&g0::inputs18, nullptr, const_cast<void*>(static_cast<const void*>(&g0::opdata18)), nullptr, 0, },
{ (TfLiteIntArray*)&g0::inputs19, (TfLiteIntArray*)&g0::outputs19, (TfLiteIntArray*)&g0::inputs19, nullptr, nullptr, nullptr, 0, },
};
#endif

used_operators_e used_ops[] =
{OP_FULLY_CONNECTED, OP_SUB, OP_MUL, OP_RESHAPE, OP_SUM, OP_ADD, OP_MUL, OP_ADD, OP_REDUCE_MAX, OP_RESHAPE, OP_MUL, OP_SUB, OP_EXP, OP_SUM, OP_LOG, OP_ADD, OP_RESHAPE, OP_SUB, OP_DIV, OP_ABS, };


// Indices into tflTensors and tflNodes for subgraphs
const size_t tflTensors_subgraph_index[] = {0, 33, };
const size_t tflNodes_subgraph_index[] = {0, 20, };

// Input/output tensors
static const int in_tensor_indices[] = {
  0, 
};

static const int out_tensor_indices[] = {
  32, 
};


size_t current_subgraph_index = 0;

static void init_tflite_tensor(size_t i, TfLiteTensor *tensor) {
  tensor->type = tensorData[i].type;
  tensor->is_variable = false;

#if defined(EI_CLASSIFIER_ALLOCATION_HEAP)
  tensor->allocation_type = tensorData[i].allocation_type;
#else
  tensor->allocation_type = (tensor_arena <= tensorData[i].data && tensorData[i].data < tensor_arena + kTensorArenaSize) ? kTfLiteArenaRw : kTfLiteMmapRo;
#endif
  tensor->bytes = tensorData[i].bytes;
  tensor->dims = tensorData[i].dims;

#if defined(EI_CLASSIFIER_ALLOCATION_HEAP)
  if(tensor->allocation_type == kTfLiteArenaRw){
    uint8_t* start = (uint8_t*) ((uintptr_t)tensorData[i].data + (uintptr_t) tensor_arena);

    tensor->data.data =  start;
  }
  else {
      tensor->data.data = tensorData[i].data;
  }
#else
  tensor->data.data = tensorData[i].data;
#endif // EI_CLASSIFIER_ALLOCATION_HEAP
  tensor->quantization.type = kTfLiteNoQuantization;

}

static void init_tflite_eval_tensor(int i, TfLiteEvalTensor *tensor) {

  tensor->type = tensorData[i].type;

  tensor->dims = tensorData[i].dims;

#if defined(EI_CLASSIFIER_ALLOCATION_HEAP)
  auto allocation_type = tensorData[i].allocation_type;
  if(allocation_type == kTfLiteArenaRw) {
    uint8_t* start = (uint8_t*) ((uintptr_t)tensorData[i].data + (uintptr_t) tensor_arena);

    tensor->data.data =  start;
  }
  else {
    tensor->data.data = tensorData[i].data;
  }
#else
  tensor->data.data = tensorData[i].data;
#endif // EI_CLASSIFIER_ALLOCATION_HEAP
}

static void* overflow_buffers[EI_MAX_OVERFLOW_BUFFER_COUNT];
static size_t overflow_buffers_ix = 0;
static void * AllocatePersistentBufferImpl(struct TfLiteContext* ctx,
                                       size_t bytes) {
  void *ptr;
  uint32_t align_bytes = (bytes % 16) ? 16 - (bytes % 16) : 0;

  if (current_location - (bytes + align_bytes) < tensor_boundary) {
    if (overflow_buffers_ix > EI_MAX_OVERFLOW_BUFFER_COUNT - 1) {
      ei_printf("ERR: Failed to allocate persistent buffer of size %d, does not fit in tensor arena and reached EI_MAX_OVERFLOW_BUFFER_COUNT\n",
        (int)bytes);
      return NULL;
    }

    // OK, this will look super weird, but.... we have CMSIS-NN buffers which
    // we cannot calculate beforehand easily.
    ptr = ei_calloc(bytes, 1);
    if (ptr == NULL) {
      ei_printf("ERR: Failed to allocate persistent buffer of size %d\n", (int)bytes);
      return NULL;
    }
    overflow_buffers[overflow_buffers_ix++] = ptr;
    return ptr;
  }

  current_location -= bytes;

  // align to the left aligned boundary of 16 bytes
  current_location -= 15; // for alignment
  current_location += 16 - ((uintptr_t)(current_location) & 15);

  ptr = current_location;
  memset(ptr, 0, bytes);

  return ptr;
}

typedef struct {
  size_t bytes;
  void *ptr;
} scratch_buffer_t;

static scratch_buffer_t scratch_buffers[EI_MAX_SCRATCH_BUFFER_COUNT];
static size_t scratch_buffers_ix = 0;

static TfLiteStatus RequestScratchBufferInArenaImpl(struct TfLiteContext* ctx, size_t bytes,
                                                int* buffer_idx) {
  if (scratch_buffers_ix > EI_MAX_SCRATCH_BUFFER_COUNT - 1) {
    ei_printf("ERR: Failed to allocate scratch buffer of size %d, reached EI_MAX_SCRATCH_BUFFER_COUNT\n",
      (int)bytes);
    return kTfLiteError;
  }

  scratch_buffer_t b;
  b.bytes = bytes;

  b.ptr = AllocatePersistentBufferImpl(ctx, b.bytes);
  if (!b.ptr) {
    ei_printf("ERR: Failed to allocate scratch buffer of size %d\n",
      (int)bytes);
    return kTfLiteError;
  }

  scratch_buffers[scratch_buffers_ix] = b;
  *buffer_idx = scratch_buffers_ix;

  scratch_buffers_ix++;

  return kTfLiteOk;
}

static void* GetScratchBufferImpl(struct TfLiteContext* ctx, int buffer_idx) {
  if (buffer_idx > (int)scratch_buffers_ix) {
    return NULL;
  }
  return scratch_buffers[buffer_idx].ptr;
}

static const uint16_t TENSOR_IX_UNUSED = 0x7FFF;

static void ResetTensors() {
  for (size_t ix = 0; ix < MAX_TFL_TENSOR_COUNT; ix++) {
    tflTensors[ix].index = TENSOR_IX_UNUSED;
  }
  for (size_t ix = 0; ix < MAX_TFL_EVAL_COUNT; ix++) {
    tflEvalTensors[ix].index = TENSOR_IX_UNUSED;
  }
}

static TfLiteTensor* GetTensorImpl(const struct TfLiteContext* context,
                               int tensor_idx) {

  tensor_idx = tflTensors_subgraph_index[current_subgraph_index] + tensor_idx;

  for (size_t ix = 0; ix < MAX_TFL_TENSOR_COUNT; ix++) {
    // already used? OK!
    if (tflTensors[ix].index == tensor_idx) {
      return &tflTensors[ix].tensor;
    }
    // passed all the ones we've used, so end of the list?
    if (tflTensors[ix].index == TENSOR_IX_UNUSED) {
      // init the tensor
      init_tflite_tensor(tensor_idx, &tflTensors[ix].tensor);
      tflTensors[ix].index = tensor_idx;
      return &tflTensors[ix].tensor;
    }
  }

  ei_printf("ERR: GetTensor called beyond MAX_TFL_TENSOR_COUNT (%d)\n", MAX_TFL_TENSOR_COUNT);
  return nullptr;
}

static TfLiteEvalTensor* GetEvalTensorImpl(const struct TfLiteContext* context,
                                       int tensor_idx) {

  tensor_idx = tflTensors_subgraph_index[current_subgraph_index] + tensor_idx;

  for (size_t ix = 0; ix < MAX_TFL_EVAL_COUNT; ix++) {
    // already used? OK!
    if (tflEvalTensors[ix].index == tensor_idx) {
      return &tflEvalTensors[ix].tensor;
    }
    // passed all the ones we've used, so end of the list?
    if (tflEvalTensors[ix].index == TENSOR_IX_UNUSED) {
      // init the tensor
      init_tflite_eval_tensor(tensor_idx, &tflEvalTensors[ix].tensor);
      tflEvalTensors[ix].index = tensor_idx;
      return &tflEvalTensors[ix].tensor;
    }
  }

  ei_printf("ERR: GetTensor called beyond MAX_TFL_EVAL_COUNT (%d)\n", (int)MAX_TFL_EVAL_COUNT);
  return nullptr;
}

class EonMicroContext : public MicroContext {
 public:
 
  EonMicroContext(): MicroContext(nullptr, nullptr, nullptr) { }

  void* AllocatePersistentBuffer(size_t bytes) {
    return AllocatePersistentBufferImpl(nullptr, bytes);
  }

  TfLiteStatus RequestScratchBufferInArena(size_t bytes,
                                           int* buffer_index) {
  return RequestScratchBufferInArenaImpl(nullptr, bytes, buffer_index);
  }

  void* GetScratchBuffer(int buffer_index) {
    return GetScratchBufferImpl(nullptr, buffer_index);
  }
 
  TfLiteTensor* AllocateTempTfLiteTensor(int tensor_index) {
    return GetTensorImpl(nullptr, tensor_index);
  }

  void DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
    return;
  }

  bool IsAllTempTfLiteTensorDeallocated() {
    return true;
  }

  TfLiteEvalTensor* GetEvalTensor(int tensor_index) {
    return GetEvalTensorImpl(nullptr, tensor_index);
  }

};


} // namespace

TfLiteStatus tflite_anomaly_8_init( void*(*alloc_fnc)(size_t,size_t) ) {
#ifdef EI_CLASSIFIER_ALLOCATION_HEAP
  tensor_arena = (uint8_t*) alloc_fnc(16, kTensorArenaSize);
  if (!tensor_arena) {
    ei_printf("ERR: failed to allocate tensor arena\n");
    return kTfLiteError;
  }
#else
  memset(tensor_arena, 0, kTensorArenaSize);
#endif
  tensor_boundary = tensor_arena;
  current_location = tensor_arena + kTensorArenaSize;

  EonMicroContext micro_context_;
  
  // Set microcontext as the context ptr
  ctx.impl_ = static_cast<void*>(&micro_context_);
  // Setup tflitecontext functions
  ctx.AllocatePersistentBuffer = &AllocatePersistentBufferImpl;
  ctx.RequestScratchBufferInArena = &RequestScratchBufferInArenaImpl;
  ctx.GetScratchBuffer = &GetScratchBufferImpl;
  ctx.GetTensor = &GetTensorImpl;
  ctx.GetEvalTensor = &GetEvalTensorImpl;
  ctx.ReportError = &MicroContextReportOpError;

  ctx.tensors_size = 33;
  for (size_t i = 0; i < 33; ++i) {
    TfLiteTensor tensor;
    init_tflite_tensor(i, &tensor);
    if (tensor.allocation_type == kTfLiteArenaRw) {
      auto data_end_ptr = (uint8_t*)tensor.data.data + tensorData[i].bytes;
      if (data_end_ptr > tensor_boundary) {
        tensor_boundary = data_end_ptr;
      }
    }
  }

  if (tensor_boundary > current_location /* end of arena size */) {
    ei_printf("ERR: tensor arena is too small, does not fit model - even without scratch buffers\n");
    return kTfLiteError;
  }

  registrations[OP_FULLY_CONNECTED] = Register_FULLY_CONNECTED();
  registrations[OP_SUB] = Register_SUB();
  registrations[OP_MUL] = Register_MUL();
  registrations[OP_RESHAPE] = Register_RESHAPE();
  registrations[OP_SUM] = Register_SUM();
  registrations[OP_ADD] = Register_ADD();
  registrations[OP_REDUCE_MAX] = Register_REDUCE_MAX();
  registrations[OP_EXP] = Register_EXP();
  registrations[OP_LOG] = Register_LOG();
  registrations[OP_DIV] = Register_DIV();
  registrations[OP_ABS] = Register_ABS();

  for (size_t g = 0; g < 1; ++g) {
    current_subgraph_index = g;
    for(size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {
      if (registrations[used_ops[i]].init) {
        tflNodes[i].user_data = registrations[used_ops[i]].init(&ctx, (const char*)tflNodes[i].builtin_data, 0);
      }
    }
  }
  current_subgraph_index = 0;

  for(size_t g = 0; g < 1; ++g) {
    current_subgraph_index = g;
    for(size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {
      if (registrations[used_ops[i]].prepare) {
        ResetTensors();
        TfLiteStatus status = registrations[used_ops[i]].prepare(&ctx, &tflNodes[i]);
        if (status != kTfLiteOk) {
          return status;
        }
      }
    }
  }
  current_subgraph_index = 0;

  return kTfLiteOk;
}

TfLiteStatus tflite_anomaly_8_input(int index, TfLiteTensor *tensor) {
  init_tflite_tensor(in_tensor_indices[index], tensor);
  return kTfLiteOk;
}

TfLiteStatus tflite_anomaly_8_output(int index, TfLiteTensor *tensor) {
  init_tflite_tensor(out_tensor_indices[index], tensor);
  return kTfLiteOk;
}

TfLiteStatus tflite_anomaly_8_invoke() {
  for (size_t i = 0; i < 20; ++i) {
    ResetTensors();

    TfLiteStatus status = registrations[used_ops[i]].invoke(&ctx, &tflNodes[i]);

#if EI_CLASSIFIER_PRINT_STATE
    ei_printf("layer %lu\n", i);
    ei_printf("    inputs:\n");
    for (size_t ix = 0; ix < tflNodes[i].inputs->size; ix++) {
      auto d = tensorData[tflNodes[i].inputs->data[ix]];

      size_t data_ptr = (size_t)d.data;

      if (d.allocation_type == kTfLiteArenaRw) {
        data_ptr = (size_t)tensor_arena + data_ptr;
      }

      if (d.type == TfLiteType::kTfLiteInt8) {
        int8_t* data = (int8_t*)data_ptr;
        ei_printf("        %lu (%zu bytes, ptr=%p, alloc_type=%d, type=%d): ", ix, d.bytes, data, (int)d.allocation_type, (int)d.type);
        for (size_t jx = 0; jx < d.bytes; jx++) {
          ei_printf("%d ", data[jx]);
        }
      }
      else {
        float* data = (float*)data_ptr;
        ei_printf("        %lu (%zu bytes, ptr=%p, alloc_type=%d, type=%d): ", ix, d.bytes, data, (int)d.allocation_type, (int)d.type);
        for (size_t jx = 0; jx < d.bytes / 4; jx++) {
          ei_printf("%f ", data[jx]);
        }
      }
      ei_printf("\n");
    }
    ei_printf("\n");

    ei_printf("    outputs:\n");
    for (size_t ix = 0; ix < tflNodes[i].outputs->size; ix++) {
      auto d = tensorData[tflNodes[i].outputs->data[ix]];

      size_t data_ptr = (size_t)d.data;

      if (d.allocation_type == kTfLiteArenaRw) {
        data_ptr = (size_t)tensor_arena + data_ptr;
      }

      if (d.type == TfLiteType::kTfLiteInt8) {
        int8_t* data = (int8_t*)data_ptr;
        ei_printf("        %lu (%zu bytes, ptr=%p, alloc_type=%d, type=%d): ", ix, d.bytes, data, (int)d.allocation_type, (int)d.type);
        for (size_t jx = 0; jx < d.bytes; jx++) {
          ei_printf("%d ", data[jx]);
        }
      }
      else {
        float* data = (float*)data_ptr;
        ei_printf("        %lu (%zu bytes, ptr=%p, alloc_type=%d, type=%d): ", ix, d.bytes, data, (int)d.allocation_type, (int)d.type);
        for (size_t jx = 0; jx < d.bytes / 4; jx++) {
          ei_printf("%f ", data[jx]);
        }
      }
      ei_printf("\n");
    }
    ei_printf("\n");
#endif // EI_CLASSIFIER_PRINT_STATE

    if (status != kTfLiteOk) {
      return status;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus tflite_anomaly_8_reset( void (*free_fnc)(void* ptr) ) {
#ifdef EI_CLASSIFIER_ALLOCATION_HEAP
  free_fnc(tensor_arena);
#endif

  // scratch buffers are allocated within the arena, so just reset the counter so memory can be reused
  scratch_buffers_ix = 0;

  // overflow buffers are on the heap, so free them first
  for (size_t ix = 0; ix < overflow_buffers_ix; ix++) {
    ei_free(overflow_buffers[ix]);
  }
  overflow_buffers_ix = 0;
  return kTfLiteOk;
}
