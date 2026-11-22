/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <functional>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* dc,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* rects,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		const bool* culling,
		bool prefiltered);
	
	// Perform initial steps for each Gaussian prior to rasterization with depth and normals.
	void preprocess_w_depth_and_normals(
		int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* dc,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const float kernel_size,
		int* radii,
		float2* rects,
		float2* points_xy_image,
		float3* view_points,
		float* depths,
		float* camera_planes,
		float2* ray_planes,
		float* ts,
		float3* normals,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		const bool* culling,
		bool prefiltered,
		bool integrate,
		int total_channels = 3,
		const float* extra_features = nullptr,
		int extra_channels = 0,
		float* invraycov3Ds = nullptr,
		bool* condition = nullptr
	);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const uint32_t* per_tile_bucket_offset, uint32_t* bucket_to_tile,
		float* sampled_T, float* sampled_ar,
		float* sampled_alpha, float* sampled_normal,
		float* sampled_coord, float* sampled_mcoord,
		float* sampled_depth, float* sampled_mdepth,
		int W, int H,
		const float* view_points,
		const float2* points_xy_image,
		const float* colors,
		const float* ts,
		const float* camera_planes,
		const float2* ray_planes,
		const float3* normals,
		const bool flag_max_count,
		float* accum_max_count,
		const float4* conic_opacity,
		const float focal_x, float focal_y,
		const float cx, float cy,
		float* out_alpha,
		float* final_T,
		uint32_t* n_contrib,
		uint32_t* max_contrib,
		const float* bg_color,
		float* out_color,
		float* out_coord,
		float* out_mcoord,
		float* out_normal,
		float* out_depth,
		float* out_mdepth,
		float* accum_coord,
		float* accum_depth,
		float* normal_length,
		int* out_gidx,
		bool require_coord,
		bool require_depth,
		int total_channels = 3
		);
		

	void render_simp(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,	
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		
		float* accum_weights_p,
		int* accum_weights_count,
		float* accum_max_count,

		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color
		);		


	void render_depth(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,

		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_pts,

		float* out_depth,
		float* accum_alpha,
		int* gidx,
		float* discriminants,

		const float* means3D,
		const glm::vec3* scales,
		const glm::vec4* rotations,

		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos
		);

	//follow code is adopted from GOF for marching tetrahedra https://github.com/autonomousvision/gaussian-opacity-fields
	// Perform initial steps for each Point prior to integration.
	void preprocess_points(int PN, int D, int M,
		const float* points3D,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		float2* points2D,
		float* depths,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Points rasterization method.
	void integrate(
		const dim3 grid, dim3 block,
		const uint2* gaussian_ranges,
		const uint2* point_ranges,
		const uint32_t* gaussian_list,
		const uint32_t* point_list,
		int W, int H,
		float focal_x, float focal_y,
		const float2* subpixel_offset,
		const float2* points2D,
		const float2* gaussians2D,
		const float* features,
		const float* depths_plane,
		const float2* ray_planes,
		const float* cov3Ds,
		const float* viewmatrix,
		const float3* points3D,
		const float3* gaussians3D,
		const float3* scales,
		const float* invraycov3Ds,
		const float* point_depths,
		const float* gaussian_depths,
		const float4* conic_opacity,
		const bool* condition,
		float* final_T,
		uint32_t* n_contrib,
		// float* center_depth,
		// float4* center_alphas,
		const float* bg_color,
		float* out_color,
		float* out_alpha_integrated,
		float* out_color_integrated,
		float* out_coordinate2d,
		float* out_sdf);
}


#endif