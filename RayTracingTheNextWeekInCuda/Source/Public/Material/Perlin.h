#pragma once

#include "Random.h"
#include "Vec3.h"

#include <cuda_runtime.h>

class Perlin
{
public:
	__device__ double Noise( const Point3& p ) const
	{
		int i = static_cast<int>( floor( p.X( ) ) );
		int j = static_cast<int>( floor( p.Y( ) ) );
		int k = static_cast<int>( floor( p.Z( ) ) );

		Vec3 c[2][2][2];

		for ( int di = 0; di < 2; ++di )
		{
			for ( int dj = 0; dj < 2; ++dj )
			{
				for ( int dk = 0; dk < 2; ++dk )
				{
					c[di][dj][dk] = m_randVec[m_permX[(i + di) & 255]
											^ m_permY[(j + dj) & 255]
											^ m_permZ[(k + dk) & 255]];
				}
			}
		}

		double u = p.X( ) - floor( p.X( ) );
		double v = p.Y( ) - floor( p.Y( ) );
		double w = p.Z( ) - floor( p.Z( ) );

		return TrilinearInterp( c, u, v, w );
	}

	__device__ double Turb( const Point3& p, int depth = 7 ) const
	{
		double accum = 0;
		Vec3 temp_p = p;
		double weight = 1.0;

		for ( int i = 0; i < depth; ++i )
		{
			accum += weight * Noise( temp_p );
			weight *= 0.5;
			temp_p *= 2;
		}

		return fabs( accum );
	}

	__device__ static void Generate( Perlin& perlin, curandState_t* randState )
	{
		int offset = blockDim.x * blockIdx.x + threadIdx.x;

		if ( offset < PointCount )
		{
			perlin.m_randVec[offset] = Random( randState, -1, 1 );

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

	__device__ static double TrilinearInterp( Vec3 c[2][2][2], double u, double v, double w )
	{
		double uu = u * u * ( 3 - 2 * u );
		double vv = v * v * ( 3 - 2 * v );
		double ww = w * w * ( 3 - 2 * w );
		double accum = 0.0;

		for ( int i = 0; i < 2; ++i )
		{
			for ( int j = 0; j < 2; ++j )
			{
				for ( int k = 0; k < 2; ++k )
				{
					accum += ( i * uu + ( 1 - i ) * ( 1.0 - uu ) )
							* ( j * vv + ( 1 - j ) * ( 1.0 - vv ) )
							* ( k * ww + ( 1 - k ) * ( 1.0 - ww ) )
							* Dot( c[i][j][k], Vec3( u - i, v - j, k - w ) );
				}
			}
		}

		return accum;
	}

private:
	static const int PointCount = 256;
	Vec3 m_randVec[PointCount];
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
