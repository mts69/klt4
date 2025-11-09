/*********************************************************************
 * klt_gpu_tracking.cu - GPU FEATURE TRACKING KERNEL
 * 
 * Parallel tracking of features using texture memory for interpolation
 * Each thread tracks ONE feature through the pyramid
 * 
 * Strategy: Simple and clean - 1 thread per feature (Option A)
 *********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

extern "C" {
#include "base.h"
#include "error.h"
#include "klt.h"
}

#define MAX_PYRAMID_LEVELS 5
#define MAX_WINDOW_SIZE 49  // 7x7

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

/*********************************************************************
 * DEVICE-SIDE STRUCTURES
 *********************************************************************/

// Feature structure on GPU
typedef struct {
  float x, y;
  int val;
  int pad;  // Alignment
} FeatureGPU;

// Pyramid metadata for tracking
typedef struct {
  cudaTextureObject_t tex_img[MAX_PYRAMID_LEVELS];
  cudaTextureObject_t tex_gradx[MAX_PYRAMID_LEVELS];
  cudaTextureObject_t tex_grady[MAX_PYRAMID_LEVELS];
  int ncols[MAX_PYRAMID_LEVELS];
  int nrows[MAX_PYRAMID_LEVELS];
  int nLevels;
  int subsampling;
} PyramidGPUInfo;

/*********************************************************************
 * DEVICE HELPER FUNCTIONS
 *********************************************************************/

// Bilinear interpolation using hardware texture unit
__device__ inline float interpolate(
  cudaTextureObject_t tex,
  float x,
  float y)
{
  // Add 0.5 to get pixel center coordinates
  return tex2D<float>(tex, x + 0.5f, y + 0.5f);
}

// Check if out of bounds
__device__ inline bool isOutOfBounds(
  float x, float y,
  int ncols, int nrows,
  int border)
{
  return (x < border || x > ncols - 1 - border ||
          y < border || y > nrows - 1 - border);
}

// Compute intensity difference window
__device__ void computeIntensityDiff(
  cudaTextureObject_t tex1,
  cudaTextureObject_t tex2,
  float x1, float y1,
  float x2, float y2,
  int hw, int hh,
  float* imgdiff)
{
  int idx = 0;
  for (int j = -hh; j <= hh; j++) {
    for (int i = -hw; i <= hw; i++) {
      float g1 = interpolate(tex1, x1 + i, y1 + j);
      float g2 = interpolate(tex2, x2 + i, y2 + j);
      imgdiff[idx++] = g1 - g2;
    }
  }
}

// Compute gradient sum
__device__ void computeGradientSum(
  cudaTextureObject_t tex_gx1,
  cudaTextureObject_t tex_gy1,
  cudaTextureObject_t tex_gx2,
  cudaTextureObject_t tex_gy2,
  float x1, float y1,
  float x2, float y2,
  int hw, int hh,
  float* gradx,
  float* grady)
{
  int idx = 0;
  for (int j = -hh; j <= hh; j++) {
    for (int i = -hw; i <= hw; i++) {
      float gx1 = interpolate(tex_gx1, x1 + i, y1 + j);
      float gx2 = interpolate(tex_gx2, x2 + i, y2 + j);
      float gy1 = interpolate(tex_gy1, x1 + i, y1 + j);
      float gy2 = interpolate(tex_gy2, x2 + i, y2 + j);
      
      gradx[idx] = gx1 + gx2;
      grady[idx] = gy1 + gy2;
      idx++;
    }
  }
}

// Compute 2x2 gradient matrix
__device__ void compute2by2Matrix(
  const float* gradx,
  const float* grady,
  int window_size,
  float* gxx,
  float* gxy,
  float* gyy)
{
  *gxx = 0.0f;
  *gxy = 0.0f;
  *gyy = 0.0f;
  
  for (int i = 0; i < window_size; i++) {
    float gx = gradx[i];
    float gy = grady[i];
    *gxx += gx * gx;
    *gxy += gx * gy;
    *gyy += gy * gy;
  }
}

// Compute error vector
__device__ void compute2by1Error(
  const float* imgdiff,
  const float* gradx,
  const float* grady,
  int window_size,
  float step_factor,
  float* ex,
  float* ey)
{
  *ex = 0.0f;
  *ey = 0.0f;
  
  for (int i = 0; i < window_size; i++) {
    float diff = imgdiff[i];
    *ex += diff * gradx[i];
    *ey += diff * grady[i];
  }
  
  *ex *= step_factor;
  *ey *= step_factor;
}

// Solve 2x2 linear system
__device__ int solveEquation(
  float gxx, float gxy, float gyy,
  float ex, float ey,
  float small,
  float* dx,
  float* dy)
{
  float det = gxx * gyy - gxy * gxy;
  
  if (det < small) {
    return KLT_SMALL_DET;
  }
  
  *dx = (gyy * ex - gxy * ey) / det;
  *dy = (gxx * ey - gxy * ex) / det;
  
  return KLT_TRACKED;
}

// Sum absolute values in window
__device__ float sumAbsWindow(
  const float* window,
  int size)
{
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += fabsf(window[i]);
  }
  return sum;
}

/*********************************************************************
 * MAIN TRACKING KERNEL
 * 
 * Each thread tracks ONE feature through all pyramid levels
 * Uses coarse-to-fine strategy with Newton-Raphson iterations
 *********************************************************************/
__global__ void trackFeaturesKernel(
  FeatureGPU* features,
  int nFeatures,
  PyramidGPUInfo pyr1,
  PyramidGPUInfo pyr2,
  int window_width,
  int window_height,
  float step_factor,
  int max_iterations,
  float min_determinant,
  float min_displacement,
  float max_residue,
  int borderx,
  int bordery)
{
  const int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (feat_idx >= nFeatures) return;
  
  // Load feature
  FeatureGPU feat = features[feat_idx];
  
  // Skip if already lost
  if (feat.val < 0) return;
  
  const int hw = window_width / 2;
  const int hh = window_height / 2;
  const int window_size = window_width * window_height;
  
  // Local arrays for this feature (in registers/local memory)
  float imgdiff[MAX_WINDOW_SIZE];
  float gradx[MAX_WINDOW_SIZE];
  float grady[MAX_WINDOW_SIZE];
  
  float x1 = feat.x;
  float y1 = feat.y;
  float x2 = feat.x;  // Initial guess
  float y2 = feat.y;
  
  // Scale to coarsest pyramid level
  for (int r = 0; r < pyr1.nLevels; r++) {
    x1 /= (float)pyr1.subsampling;
    y1 /= (float)pyr1.subsampling;
    x2 /= (float)pyr1.subsampling;
    y2 /= (float)pyr1.subsampling;
  }
  
  int status = KLT_TRACKED;
  
  // ============ COARSE-TO-FINE TRACKING ============
  for (int level = pyr1.nLevels - 1; level >= 0; level--) {
    
    // Scale back up from previous level
    if (level < pyr1.nLevels - 1) {
      x1 *= (float)pyr1.subsampling;
      y1 *= (float)pyr1.subsampling;
      x2 *= (float)pyr1.subsampling;
      y2 *= (float)pyr1.subsampling;
    }
    
    const int ncols = pyr1.ncols[level];
    const int nrows = pyr1.nrows[level];
    
    // ============ NEWTON-RAPHSON ITERATIONS ============
    int iteration = 0;
    while (iteration < max_iterations) {
      
      // Check bounds
      if (isOutOfBounds(x1, y1, ncols, nrows, hw + borderx + 1, hh + bordery + 1) ||
          isOutOfBounds(x2, y2, ncols, nrows, hw + borderx + 1, hh + bordery + 1)) {
        status = KLT_OOB;
        break;
      }
      
      // Extract windows
      computeIntensityDiff(
        pyr1.tex_img[level], pyr2.tex_img[level],
        x1, y1, x2, y2, hw, hh, imgdiff);
      
      computeGradientSum(
        pyr1.tex_gradx[level], pyr1.tex_grady[level],
        pyr2.tex_gradx[level], pyr2.tex_grady[level],
        x1, y1, x2, y2, hw, hh, gradx, grady);
      
      // Build matrix and error vector
      float gxx, gxy, gyy, ex, ey;
      compute2by2Matrix(gradx, grady, window_size, &gxx, &gxy, &gyy);
      compute2by1Error(imgdiff, gradx, grady, window_size,
                      step_factor, &ex, &ey);
      
      // Solve for displacement
      float dx, dy;
      status = solveEquation(gxx, gxy, gyy, ex, ey,
                            min_determinant, &dx, &dy);
      
      if (status == KLT_SMALL_DET) break;
      
      // Update position
      x2 += dx;
      y2 += dy;
      iteration++;
      
      // Check convergence
      if (fabsf(dx) < min_displacement && fabsf(dy) < min_displacement) {
        break;
      }
    }
    
    // If lost at this level, stop
    if (status != KLT_TRACKED) break;
    
    // Check if exceeded max iterations
    if (iteration >= max_iterations) {
      status = KLT_MAX_ITERATIONS;
      break;
    }
  }
  
  // ============ FINAL VALIDATION (at finest level) ============
  if (status == KLT_TRACKED) {
    const int level = 0;
    const int ncols = pyr1.ncols[level];
    const int nrows = pyr1.nrows[level];
    
    // Final bounds check
    if (isOutOfBounds(x2, y2, ncols, nrows, borderx, bordery)) {
      status = KLT_OOB;
    } else {
      // Check residue
      computeIntensityDiff(
        pyr1.tex_img[level], pyr2.tex_img[level],
        x1, y1, x2, y2, hw, hh, imgdiff);
      
      float residue = sumAbsWindow(imgdiff, window_size) / (float)window_size;
      
      if (residue > max_residue) {
        status = KLT_LARGE_RESIDUE;
      }
    }
  }
  
  // ============ WRITE RESULTS ============
  if (status == KLT_TRACKED) {
    features[feat_idx].x = x2;
    features[feat_idx].y = y2;
    features[feat_idx].val = KLT_TRACKED;
  } else {
    features[feat_idx].x = -1.0f;
    features[feat_idx].y = -1.0f;
    features[feat_idx].val = status;
  }
}

/*********************************************************************
 * HOST-SIDE API
 *********************************************************************/

// Global feature storage
static struct {
  FeatureGPU* d_features;
  FeatureGPU* h_features;
  int capacity;
  bool initialized;
} g_features = {NULL, NULL, 0, false};

// Initialize feature storage
extern "C" void KLT_InitGPUFeatures(int max_features) {
  if (g_features.initialized) return;
  
  printf("🚀 Initializing GPU feature storage: %d features\n", max_features);
  
  g_features.capacity = max_features;
  
  CUDA_CHECK(cudaMalloc(&g_features.d_features,
                        max_features * sizeof(FeatureGPU)));
  
  g_features.h_features = (FeatureGPU*)malloc(
    max_features * sizeof(FeatureGPU));
  
  g_features.initialized = true;
  
  printf("✅ GPU features initialized\n\n");
}

// Cleanup
extern "C" void KLT_CleanupGPUFeatures() {
  if (!g_features.initialized) return;
  
  cudaFree(g_features.d_features);
  free(g_features.h_features);
  
  g_features.initialized = false;
}

// Main tracking function - replaces KLTTrackFeatures
extern "C" void KLT_TrackFeaturesGPU(
  KLT_TrackingContext tc,
  int pyr1_idx,  // Index in batch for pyramid 1
  int pyr2_idx,  // Index in batch for pyramid 2
  KLT_FeatureList featurelist)
{
  if (!g_features.initialized) {
    fprintf(stderr, "ERROR: GPU features not initialized!\n");
    return;
  }
  
  const int nFeatures = featurelist->nFeatures;
  if (nFeatures == 0) return;
  
  // Get pyramid data from batch storage
  float **d_img1, **d_gradx1, **d_grady1;
  float **d_img2, **d_gradx2, **d_grady2;
  cudaTextureObject_t *tex_img1, *tex_gradx1, *tex_grady1;
  cudaTextureObject_t *tex_img2, *tex_gradx2, *tex_grady2;
  int *ncols1, *nrows1, *ncols2, *nrows2;
  
  // Declare these functions in header
  extern void KLT_GetBatchPyramid(
    int batch_idx,
    float*** d_img_out, float*** d_gradx_out, float*** d_grady_out,
    cudaTextureObject_t** tex_img_out,
    cudaTextureObject_t** tex_gradx_out,
    cudaTextureObject_t** tex_grady_out,
    int** ncols_out, int** nrows_out);
  
  KLT_GetBatchPyramid(pyr1_idx,
    &d_img1, &d_gradx1, &d_grady1,
    &tex_img1, &tex_gradx1, &tex_grady1,
    &ncols1, &nrows1);
  
  KLT_GetBatchPyramid(pyr2_idx,
    &d_img2, &d_gradx2, &d_grady2,
    &tex_img2, &tex_gradx2, &tex_grady2,
    &ncols2, &nrows2);
  
  // Copy features to GPU
  for (int i = 0; i < nFeatures; i++) {
    g_features.h_features[i].x = featurelist->feature[i]->x;
    g_features.h_features[i].y = featurelist->feature[i]->y;
    g_features.h_features[i].val = featurelist->feature[i]->val;
  }
  
  CUDA_CHECK(cudaMemcpy(
    g_features.d_features,
    g_features.h_features,
    nFeatures * sizeof(FeatureGPU),
    cudaMemcpyHostToDevice));
  
  // Build pyramid info structures
  PyramidGPUInfo pyr1_info, pyr2_info;
  pyr1_info.nLevels = tc->nPyramidLevels;
  pyr1_info.subsampling = tc->subsampling;
  pyr2_info.nLevels = tc->nPyramidLevels;
  pyr2_info.subsampling = tc->subsampling;
  
  for (int i = 0; i < tc->nPyramidLevels; i++) {
    pyr1_info.tex_img[i] = tex_img1[i];
    pyr1_info.tex_gradx[i] = tex_gradx1[i];
    pyr1_info.tex_grady[i] = tex_grady1[i];
    pyr1_info.ncols[i] = ncols1[i];
    pyr1_info.nrows[i] = nrows1[i];
    
    pyr2_info.tex_img[i] = tex_img2[i];
    pyr2_info.tex_gradx[i] = tex_gradx2[i];
    pyr2_info.tex_grady[i] = tex_grady2[i];
    pyr2_info.ncols[i] = ncols2[i];
    pyr2_info.nrows[i] = nrows2[i];
  }
  
  // Launch kernel
  const int threads = 256;
  const int blocks = (nFeatures + threads - 1) / threads;
  
  trackFeaturesKernel<<<blocks, threads>>>(
    g_features.d_features,
    nFeatures,
    pyr1_info,
    pyr2_info,
    tc->window_width,
    tc->window_height,
    tc->step_factor,
    tc->max_iterations,
    tc->min_determinant,
    tc->min_displacement,
    tc->max_residue,
    tc->borderx,
    tc->bordery);
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy results back
  CUDA_CHECK(cudaMemcpy(
    g_features.h_features,
    g_features.d_features,
    nFeatures * sizeof(FeatureGPU),
    cudaMemcpyDeviceToHost));
  
  // Update feature list
  int tracked = 0;
  for (int i = 0; i < nFeatures; i++) {
    featurelist->feature[i]->x = g_features.h_features[i].x;
    featurelist->feature[i]->y = g_features.h_features[i].y;
    featurelist->feature[i]->val = g_features.h_features[i].val;
    
    if (g_features.h_features[i].val == KLT_TRACKED) {
      tracked++;
    }
  }
  
  printf("   📍 Tracked: %d/%d features\n", tracked, nFeatures);
}