%module cloud_handler

%include <std_string.i>

%{
  #define SWIG_FILE_WITH_INIT
  #include "cloud.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (float* INPLACE_ARRAY1, int DIM1) {(float* data, int n)};
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* depth_data, int depth_h, int depth_w)};
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* prediction_data, int prediction_h, int prediction_w)};

%include "cloud.h"
