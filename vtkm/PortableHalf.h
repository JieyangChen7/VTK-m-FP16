#ifndef vtk_m_PortableHalf
#define vtk_m_PortableHalf

// half support on GPU
#include "cuda_fp16.h"

// half support on CPU
#include "Half.h"

#ifndef  VTKM_CUDA
#include <cmath>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#endif

#include <iostream>

#define cudaF2H __float2half
#define cudaH2F __half2float 

#define hostF2H half_float::half_cast<half_float::half, std::round_to_nearest>
#define hostH2F half_float::half_cast<float,            std::round_to_nearest>

namespace ph {

struct PortableHalf {
private:
  half data;
public:
  VTKM_EXEC_CONT
  PortableHalf() {
    data = cudaF2H(0.0f);
  }

  VTKM_EXEC_CONT
  PortableHalf(half h) {
    data = h;
  }

  VTKM_EXEC_CONT
  PortableHalf(float f) {
    data = cudaF2H(f);
  }

  /* Copy constructor */
  VTKM_EXEC_CONT
  PortableHalf(const PortableHalf &ph) {
    data = ph.get();
  }

  VTKM_EXEC_CONT
  half get() const {
    return data;
  }

  VTKM_EXEC_CONT
  void set(PortableHalf ph) {
    data = ph.get();
  }

  VTKM_EXEC_CONT
  float to_float() const {
    return cudaH2F(data);
  }


 VTKM_EXEC_CONT
  PortableHalf operator + (PortableHalf h){
#ifdef  __CUDA_ARCH__
   return PortableHalf(__hadd(data, h.get()));
#else
   return PortableHalf(hostH2F( hostF2H(cudaH2F(data)) + hostF2H(cudaH2F(h.get())) ));
#endif
 }

 VTKM_EXEC_CONT
 PortableHalf operator - (PortableHalf h){
#ifdef  __CUDA_ARCH__
   return PortableHalf(__hsub(data, h.get()));
#else
   return PortableHalf(hostH2F( hostF2H(cudaH2F(data)) - hostF2H(cudaH2F(h.get())) ));
#endif
 }

 VTKM_EXEC_CONT
 PortableHalf operator * (PortableHalf h){
#ifdef  __CUDA_ARCH__
   return PortableHalf(__hmul(data, h.get()));
#else
   return PortableHalf(hostH2F( hostF2H(cudaH2F(data)) * hostF2H(cudaH2F(h.get())) ));
#endif
 }

VTKM_EXEC_CONT
 PortableHalf operator / (PortableHalf h){
#ifdef  __CUDA_ARCH__
   return PortableHalf(__hdiv(data, h.get()));
#else
   return PortableHalf(hostH2F( hostF2H(cudaH2F(data)) / hostF2H(cudaH2F(h.get())) ));
#endif
 }

VTKM_EXEC_CONT
 PortableHalf operator - (){
#ifdef  __CUDA_ARCH__
   return PortableHalf(__hneg(data));
#else
   return PortableHalf(hostH2F(-hostF2H(cudaH2F(data))));
#endif
 }

VTKM_EXEC_CONT
 PortableHalf& operator=(float rhs) {
   data = cudaF2H(rhs);
   return *this;
 }

 VTKM_EXEC_CONT
 PortableHalf& operator=(PortableHalf ph) {
   data = ph.get();
   return *this;
 }

// VTKM_EXEC_CONT
// operator float() const {
//   return cudaH2F(data);
// }

//  VTKM_CONT
// std::ostream& operator<<(std::ostream& os)
// {
//     os << to_float();
//     return os;
// }

 // VTKM_CONT
 // friend std::ostream& operator<<(std::ostream& os, const PortableHalf& ph);


};

 VTKM_CONT
 static inline std::ostream& operator<<(std::ostream& os, const PortableHalf& ph)
 {
     os << ph.to_float();
     return os;
 }

 VTKM_CONT
 static inline void print_half(const PortableHalf& ph)
 {
   std::cout << ph.to_float();
 }

}

#endif //vtk_m_PortableHalf


