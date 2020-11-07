#pragma once

#include "Vec3.h"

#include <cuda_runtime.h>

class Ray
{
public:
	__device__ Ray( ) {}
	__device__ Ray( const Point3& origin, const Vec3& direction ) :
		m_origin( origin ), m_direction( direction ) {}

	__device__ Point3 Origin( ) const
	{
		return m_origin;
	}

	__device__ Vec3 Direction( ) const
	{
		return m_direction;
	}

	__device__ Point3 At( double t ) const
	{
		return m_origin + m_direction * t;
	}

private:
	Point3 m_origin;
	Vec3 m_direction;
};