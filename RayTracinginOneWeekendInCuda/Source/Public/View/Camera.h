#pragma once

#include "Random.h"
#include "Ray.h"
#include "Vec3.h"

#include <cuda_runtime.h>
#include <math_constants.h>

class Camera
{
public:
	__host__ Camera( Point3 lookFrom, Point3 lookAt,  double fov, double aspectRatio, double aperture, double focusDist )
	{
		double theta = fov * ( 1 / 180.0 ) * CUDART_PI;
		double h = tan( theta * 0.5 );

		double viewportHeight = 2.0 * h;
		double viewportWidth = aspectRatio * viewportHeight;

		Vec3 forward = Normalize( lookFrom - lookAt );
		m_right = Normalize( Cross( Vec3( 0, 1, 0 ), forward ) );
		m_up = Normalize( Cross( forward, m_right ) );

		m_origin = lookFrom;
		m_horizontal = focusDist * viewportWidth * m_right;
		m_vertical = focusDist * viewportHeight * m_up;
		m_lowerLeftCorner = m_origin - m_horizontal * 0.5 - m_vertical * 0.5 - focusDist * forward;

		m_lensRadius = aperture * 0.5;
	}

	__device__ Ray GetRay( curandState_t* randState, double u, double v ) const
	{
		Vec3 rd = m_lensRadius * RandomInUnitDisk( randState );
		Vec3 offset = m_right * rd.X( ) + m_up * rd.Y( );

		return Ray( m_origin + offset, m_lowerLeftCorner + u * m_horizontal + v * m_vertical - m_origin - offset );
	}

private:
	Point3 m_origin;
	Vec3 m_horizontal;
	Vec3 m_vertical;
	Vec3 m_right;
	Vec3 m_up;
	Point3 m_lowerLeftCorner;
	double m_lensRadius;
};