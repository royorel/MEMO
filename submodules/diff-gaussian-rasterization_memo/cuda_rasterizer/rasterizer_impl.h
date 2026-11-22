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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		float* camera_planes;
		float2* ray_planes;
		float* ts;
		float3* normals;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* rects2D;
		float2* means2D;
		float* view_points;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P, int channels = 3);
	};

	struct PointState
	{
		size_t scan_size;
		float* depths;
		float2* points2D;
		char* scanning_space;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static PointState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint32_t *bucket_count;
		uint32_t *bucket_offsets;
		size_t bucket_count_scan_size;
		char * bucket_count_scanning_space;
		float* pixel_colors;
		float* pixel_alphas;
		float* pixel_normals;
		float* pixel_coords;
		float* pixel_mcoords;
		float* pixel_depths;
		float* pixel_mdepths;
		float* accum_coords;
		float* accum_depths;
		uint32_t* max_contrib;
		size_t scan_size;
		uint2* ranges;
		uint2* point_ranges;
		uint32_t* n_contrib;
		float* final_T;
		char* contrib_scan;
		float* normal_lengths;
		int* gidx;
		static ImageState fromChunk(char*& chunk, size_t N, int channels = 3);
	};

	struct BinningState
	{
		size_t scan_size;
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		int* scan_src;
		int* scan_dst;
		char* scan_space;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	struct SampleState
	{
		uint32_t *bucket_to_tile;
		float *T;
		float *ar;
		float *alpha;
		float *normal;
		float *coord;
		float *mcoord;
		float *depth;
		float *mdepth;
		static SampleState fromChunk(char*& chunk, size_t C, int channels = 3);
	};

	template<typename T, typename... Args>
	size_t required(size_t P, Args... args)
	{
		char* size = nullptr;
		T::fromChunk(size, P, args...);
		return ((size_t)size) + 128;
	}
};