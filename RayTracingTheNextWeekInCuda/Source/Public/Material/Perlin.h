#pragma once

#include "Random.h"
#include "Vec3.h"

#include <cuda_runtime.h>

class Perlin
{
public:
	__device__ double Noise( const Point3& p ) const
	{
		int i = static_cast<int>( p.X() * 4 ) & 255;
		int j = static_cast<int>( p.Y() * 4 ) & 255;
		int k = static_cast<int>( p.Z() * 4 ) & 255;

		return m_randFloat[m_permX[i] ^ m_permY[j] ^ m_permZ[k]];
	}

	__device__ static void Generate( Perlin& perlin, curandState_t* randState )
	{
		int offset = blockDim.x * blockIdx.x + threadIdx.x;

		if ( offset < PointCount )
		{
			perlin.m_randFloat[offset] = RandomDouble( randState );

			__shared__ int localPermX[PointCount];
			__shared__ int localPermY[PointCount];
			__shared__ int localPermZ[PointCount];

			localPermX[offset] = offset;
			localPermY[offset] = offset;
			localPermZ[offset] = offset;

			__syncthreads( );

			if ( offset == 0 )
			{
				for ( int i = 0; i < PointCount; ++i )
				{
					int target = RandomInt( randState, 0, i );
					Swap( localPermX[i], localPermX[target] );

					target = RandomInt( randState, 0, i );
					Swap( localPermY[i], localPermY[target] );

					target = RandomInt( randState, 0, i );
					Swap( localPermZ[i], localPermZ[target] );
				}
			}

			__syncthreads( );

			perlin.m_permX[offset] = localPermX[offset];
			perlin.m_permY[offset] = localPermY[offset];
			perlin.m_permZ[offset] = localPermZ[offset];
		}

		__syncthreads( );
	}

	__device__ static void Swap( int& lhs, int& rhs )
	{
		int tmp = lhs;
		lhs = rhs;
		rhs = tmp;
	}

private:
	static const int PointCount = 256;
	double m_randFloat[PointCount];
	int m_permX[PointCount];
	int m_permY[PointCount];
	int m_permZ[PointCount];
};

__global__ void GeneratePerlinTexture( Perlin* perlin )
{
	curandState_t randState;
	curand_init( blockIdx.x, threadIdx.x, 0, &randState );
	Perlin::Generate( *perlin, &randState );
}
