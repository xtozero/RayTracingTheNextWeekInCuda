#pragma once

#include <cuda_runtime.h>

class Vec3
{
public:
	__device__ __host__ Vec3( ) 
#ifndef __CUDA_ARCH__
		: m_component{ 0, 0, 0 }
#endif
	{}

	__device__ __host__ Vec3( double x, double y, double z )
		: m_component{ x, y, z }
	{}

	__device__ __host__ double X( ) const
	{
		return m_component[0];
	}

	__device__ __host__ double Y( ) const
	{
		return m_component[1];
	}

	__device__ __host__ double Z( ) const
	{
		return m_component[2];
	}

	__device__ __host__ double Length( ) const
	{
		return sqrt( LengthSquard( ) );
	}

	__device__ __host__ double LengthSquard( ) const
	{
		return m_component[0] * m_component[0]
			+ m_component[1] * m_component[1]
			+ m_component[2] * m_component[2];
	}

	__device__ __host__ bool NearZero( ) const
	{
		const double epsilon = 1e-8;
		return ( fabs( m_component[0] ) < epsilon )
			&& ( fabs( m_component[1] ) < epsilon )
			&& ( fabs( m_component[2] ) < epsilon );
	}

	__device__ __host__ Vec3 operator-( ) const
	{
		return Vec3( -m_component[0], -m_component[1], -m_component[2] );
	}

	__device__ __host__ double operator[]( int i ) const
	{
		return m_component[i];
	}

	__device__ __host__ double& operator[]( int i )
	{
		return m_component[i];
	}

	__device__ __host__ Vec3& operator+=( const Vec3& v )
	{
		for ( int i = 0; i < 3; ++i )
		{
			m_component[i] += v.m_component[i];
		}
		return *this;
	}

	__device__ __host__ Vec3& operator*=( const double s )
	{
		for ( int i = 0; i < 3; ++i )
		{
			m_component[i] *= s;
		}
		return *this;
	}

	__device__ __host__ Vec3& operator/=( const double s )
	{
		return *this *= 1 / s;
	}

	__device__ __host__ friend Vec3 operator+( const Vec3& lhs, const Vec3& rhs )
	{
		return Vec3( lhs.m_component[0] + rhs.m_component[0],
					lhs.m_component[1] + rhs.m_component[1],
					lhs.m_component[2] + rhs.m_component[2]);
	}

	__device__ __host__ friend Vec3 operator-( const Vec3& lhs, const Vec3& rhs )
	{
		return Vec3( lhs.m_component[0] - rhs.m_component[0],
					lhs.m_component[1] - rhs.m_component[1],
					lhs.m_component[2] - rhs.m_component[2] );
	}

	__device__ __host__ friend Vec3 operator*( const Vec3& lhs, const Vec3& rhs )
	{
		return Vec3( lhs.m_component[0] * rhs.m_component[0],
					lhs.m_component[1] * rhs.m_component[1],
					lhs.m_component[2] * rhs.m_component[2] );
	}

	__device__ __host__ friend Vec3 operator*( double lhs, const Vec3& rhs )
	{
		return Vec3( lhs * rhs.m_component[0],
					lhs * rhs.m_component[1],
					lhs * rhs.m_component[2] );
	}

	__device__ __host__ friend Vec3 operator*( const Vec3& lhs, double rhs )
	{
		return rhs * lhs;
	}

	__device__ __host__ friend Vec3 operator/( const Vec3& lhs, double rhs )
	{
		return ( 1.0 / rhs ) * lhs;
	}

	__device__ __host__ friend double Dot( const Vec3& lhs, const Vec3& rhs )
	{
		return lhs.m_component[0] * rhs.m_component[0]
			+ lhs.m_component[1] * rhs.m_component[1]
			+ lhs.m_component[2] * rhs.m_component[2];
	}

	__device__ __host__ friend Vec3 Cross( const Vec3& lhs, const Vec3& rhs )
	{
		return Vec3( lhs.m_component[1] * rhs.m_component[2] - lhs.m_component[2] * rhs.m_component[1],
					lhs.m_component[2] * rhs.m_component[0] - lhs.m_component[0] * rhs.m_component[2],
					lhs.m_component[0] * rhs.m_component[1] - lhs.m_component[1] * rhs.m_component[0] );
	}

	__device__ __host__ friend Vec3 Normalize( const Vec3& v )
	{
		return v / v.Length( );
	}

	__device__ __host__ friend Vec3 Reflect( const Vec3& v, const Vec3& n )
	{
		return v - 2 * Dot( v, n ) * n;
	}

	__device__ __host__ friend Vec3 Refract( const Vec3& v, const Vec3& n, double refractionRatio )
	{
		double cosTheta = fmin( Dot( -v, n ), 1.0 );
		Vec3 outPerpendicular = refractionRatio * ( v + cosTheta * n );
		Vec3 outParallel = -sqrt( fabs( 1 - outPerpendicular.LengthSquard() ) ) * n;
		return outPerpendicular + outParallel;
	}

private:
	double m_component[3];
};

using Point3 = Vec3;
using Color = Vec3;

__device__ void WriteColor( unsigned char* outColor, const Color& inColor, double samplePerPixel )
{
	double r = inColor.X( );
	double g = inColor.Y( );
	double b = inColor.Z( );

	double rSamplePerPixel = 1.0 / samplePerPixel;
	r = sqrt( rSamplePerPixel * r );
	g = sqrt( rSamplePerPixel * g );
	b = sqrt( rSamplePerPixel * b );

	outColor[0] = static_cast<unsigned char>( 255 * __saturatef( r ) );
	outColor[1] = static_cast<unsigned char>( 255 * __saturatef( g ) );
	outColor[2] = static_cast<unsigned char>( 255 * __saturatef( b ) );
}