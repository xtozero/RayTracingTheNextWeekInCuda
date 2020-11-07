#pragma once

#include "Hittable.h"

class MovingSphere : public Hittable
{
public:
	__device__ MovingSphere( ) {}
	__device__ MovingSphere( Point3 center0, Point3 center1, double time0, double time1, double radius, Material* material )
		: Hittable( material ), m_center0( center0 ), m_center1( center1 ), m_time0( time0 ), m_time1( time1 ), m_radius( radius )
	{}

	__device__ virtual bool Hit( const Ray& r, double tMin, double tMax, HitRecord& rec ) const override
	{
		Point3 center = Center( r.Time( ) );
		Vec3 oc = r.Origin( ) - center;
		double a = r.Direction( ).LengthSquard( );
		double halfB = Dot( r.Direction( ), oc );
		double c = oc.LengthSquard( ) - m_radius * m_radius;
		double discriment = halfB * halfB - a * c;
		if ( discriment > 0 )
		{
			double root = sqrt( discriment );

			float t = ( -halfB - root ) / a;
			if ( t > tMin && t < tMax )
			{
				rec.m_t = t;
				rec.m_hitPosition = r.At( t );
				Vec3 outwardNormal = ( rec.m_hitPosition - center ) / m_radius;
				rec.SetFrontFaceNormal( r, outwardNormal );
				rec.m_material = m_material;
				return true;
			}

			t = ( -halfB + root ) / a;
			if ( t > tMin && t < tMax )
			{
				rec.m_t = t;
				rec.m_hitPosition = r.At( t );
				Vec3 outwardNormal = ( rec.m_hitPosition - center ) / m_radius;
				rec.SetFrontFaceNormal( r, outwardNormal );
				rec.m_material = m_material;
				return true;
			}
		}

		return false;
	}

private:
	__device__ Point3 Center( double time ) const
	{
		return m_center0 + ( ( time - m_time0 ) / ( m_time1 - m_time0 ) ) * ( m_center1 - m_center0 );
	}

	Point3 m_center0;
	Point3 m_center1;
	double m_time0;
	double m_time1;
	double m_radius;
};