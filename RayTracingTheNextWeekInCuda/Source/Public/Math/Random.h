#pragma once

#include "Vec3.h"

#include <curand_kernel.h>
#include <math_constants.h>

__device__ double RandomDouble( curandState_t* randState )
{
	return 1.0 - curand_uniform_double( randState );
}

__device__ double RandomDouble( curandState_t* randState, double min, double max )
{
	return min + ( max - min ) * RandomDouble( randState );
}

__device__ int RandomInt( curandState_t* randState, int min, int max )
{
	return static_cast<int>( RandomDouble( randState ), static_cast<double>( min ), static_cast<double>( max ) );
}

__device__ Vec3 Random( curandState_t* randState )
{
	return Vec3( RandomDouble( randState ), RandomDouble( randState ), RandomDouble( randState ) );
}

__device__ Vec3 Random( curandState_t* randState, double min, double max )
{
	return Vec3( RandomDouble( randState, min, max ), RandomDouble( randState, min, max ), RandomDouble( randState, min, max ) );
}

__device__ Vec3 RandomInUnitSphere( curandState_t* randState )
{
	while ( true )
	{
		Vec3 p = Random( randState, -1, 1 );
		if ( p.LengthSquard( ) <= 1 )
		{
			return p;
		}
	}
}

__device__ Vec3 RandomUnitVector( curandState_t* randState )
{
	double theta = RandomDouble( randState, 0, 2 * CUDART_PI );
	double z = RandomDouble( randState, -1, 1 );
	double r = sqrt( 1 - z * z );
	return Vec3( r * cos( theta ), r * sin( theta ), z );
}

__device__ Vec3 RandomInHemishpere( curandState_t* randState, const Vec3& normal )
{
	Vec3 unitVector = RandomUnitVector( randState );
	if ( Dot( normal, unitVector ) >= 0 )
	{
		return unitVector;
	}

	return -unitVector;
}

__device__ Vec3 RandomInUnitDisk( curandState_t* randState )
{
	while ( true )
	{
		Vec3 p( RandomDouble( randState, -1, 1 ), RandomDouble( randState, -1, 1 ), 0 );
		if ( p.LengthSquard( ) <= 1 )
		{
			return p;
		}
	}
}