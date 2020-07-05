#ifndef vtk_m_PortableHalf2
#define vtk_m_PortableHal2

// half support on GPU
#include "cuda_fp16.h"

// half support on CPU
#include "Half.h"

#include "PortableHalf.h"

#include <cuda.h>

#ifndef  VTKM_CUDA
#include <cmath>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#endif


//#define cudaF2H __float2half
//#define cudaH2F __half2float 

//#define hostF2H half_float::half_cast<half_float::half, std::round_to_nearest>
//#define hostH2F half_float::half_cast<float,            std::round_to_nearest>

#define cudaF2H2 __floats2half2_rn
//#define cudaH22F2 __half22float2

#define cudaHH22F  __high2float
#define cudaLH22F  __low2float

#define deviceHH22H __high2half
#define deviceLH22H __low2half

#define deviceHH2H2 __halves2half2

namespace ph {

struct PortableHalf2 {
private:
  half2 data;
public:
  VTKM_EXEC_CONT
  PortableHalf2() {
    data = cudaF2H2(0.0f, 0.0f);
  }

  VTKM_EXEC_CONT
  PortableHalf2(half2 h2) {
    data = h2;
  }

  VTKM_EXEC_CONT
  PortableHalf2(half h1, half h2) {
#ifdef  __CUDA_ARCH__
    data = deviceHH2H2(h1, h2);
#else
    data = cudaF2H2(cudaH2F(h1), cudaH2F(h2));
#endif
  }

  VTKM_EXEC_CONT
  PortableHalf2(PortableHalf ph1, PortableHalf ph2) {
    set(ph1.get(), ph2.get());
  }
/*
  VTKM_EXEC_CONT
  PortableHalf2(float2 f2) {
    data = cudaF22H2(f2);
  }
*/
  VTKM_EXEC_CONT
  PortableHalf2(float f1, float f2) {
    data = cudaF2H2(f1, f2);
  }

  /* Copy constructor */
  VTKM_EXEC_CONT
  PortableHalf2(const PortableHalf2 &ph2) {
    data = ph2.get();
  }

  VTKM_EXEC_CONT
  half2 get() const {
    return data;
  }

  VTKM_EXEC_CONT
  half get_high() const {
#ifdef  __CUDA_ARCH__
    return deviceHH22H(data);
#else
    return cudaF2H(cudaHH22F(data));
#endif
  }

  VTKM_EXEC_CONT
  half get_low() const {
#ifdef  __CUDA_ARCH__
    return deviceLH22H(data);
#else
    return cudaF2H(cudaLH22F(data));
#endif
  }


  VTKM_EXEC_CONT
  void set(half2 h2) {
    data = h2;
  }

  VTKM_EXEC_CONT
  void set(half h1, half h2) {
#ifdef  __CUDA_ARCH__
    data = deviceHH2H2(h1, h2);
#else
    data = cudaF2H2(cudaH2F(h1), cudaH2F(h2));
#endif
  }

  VTKM_EXEC_CONT
  void set_high(half h) {
    set(get_low(), h);
  }

  VTKM_EXEC_CONT
  void set_low(half h) {
    set(h, get_high());
  }


  VTKM_EXEC_CONT
  float to_float_high() const {
    return cudaHH22F(data);
  }

  VTKM_EXEC_CONT
  float to_float_low() const {
    return cudaLH22F(data);
  }


  VTKM_EXEC_CONT
  PortableHalf2 operator + (PortableHalf2 h2){
#ifdef  __CUDA_ARCH__
   return PortableHalf2(__hadd2(data, h2.get()));
#else
   PortableHalf ph1 = PortableHalf(get_high()) + PortableHalf(h2.get_high());
   PortableHalf ph2 = PortableHalf(get_low()) + PortableHalf(h2.get_low());
   return PortableHalf2(ph1, ph2);
#endif
 }
  
  VTKM_EXEC_CONT
  PortableHalf2 operator - (PortableHalf2 h2){
#ifdef  __CUDA_ARCH__
   return PortableHalf2(__hsub2(data, h2.get()));
#else
   PortableHalf ph1 = PortableHalf(get_high()) - PortableHalf(h2.get_high());
   PortableHalf ph2 = PortableHalf(get_low()) - PortableHalf(h2.get_low());
   return PortableHalf2(ph1, ph2);
#endif
 }

  VTKM_EXEC_CONT
  PortableHalf2 operator * (PortableHalf2 h2){
#ifdef  __CUDA_ARCH__
   return PortableHalf2(__hmul2(data, h2.get()));
#else
   PortableHalf ph1 = PortableHalf(get_high()) * PortableHalf(h2.get_high());
   PortableHalf ph2 = PortableHalf(get_low()) * PortableHalf(h2.get_low());
   return PortableHalf2(ph1, ph2);
#endif
 }

  VTKM_EXEC_CONT
  PortableHalf2 operator / (PortableHalf2 h2){
//#ifdef  __CUDA_ARCH__
//   return PortableHalf2(__hdiv2(data, h2.get()));
//#else
   PortableHalf ph1 = PortableHalf(get_high()) / PortableHalf(h2.get_high());
   PortableHalf ph2 = PortableHalf(get_low()) / PortableHalf(h2.get_low());
   return PortableHalf2(ph1, ph2);
//#endif
 }

VTKM_EXEC_CONT
 PortableHalf2 operator - (){
#ifdef  __CUDA_ARCH__
   return PortableHalf2(__hneg2(data));
#else
   PortableHalf ph1 = -PortableHalf(get_high());
   PortableHalf ph2 = -PortableHalf(get_low());
   return PortableHalf2(ph1, ph2);
#endif
 }

//VTKM_EXEC_CONT
// PortableHalf& operator=(float rhs) {
//   data = cudaF2H(rhs);
//   return *this;
// }

 //VTKM_EXEC_CONT
 //operator float() const {
 //  return to_float_high();
 //}

 //VTKM_CONT
 //friend std::ostream& operator<<(std::ostream& os, const PortableHalf2& ph);
 //VTKM_CONT
 //std::ostream& operator<<(std::ostream& os)
 //{
 //    os << "(" << to_float_high() << ", " << to_float_low() << ")";
 //    return os;
 //}

};

 VTKM_CONT
 static inline std::ostream& operator<<(std::ostream& os, const PortableHalf2& ph2)
 {
     os << "(" << ph2.to_float_high() << ", " << ph2.to_float_low() << ")";
     return os;
 }

}

#endif //vtk_m_PortableHalf2


