#pragma once

#include "Hittable.h"

class Sphere : public Hittable
{
public:
	__device__ Sphere( ) {}
	__device__ Sphere( Point3 center, double radius, Material* material )
		: Hittable( material ), m_center( center ), m_radius( radius )
	{}

	__device__ virtual bool Hit( const Ray& r, double tMin, double tMax, HitRecord& rec ) const override
	{
		Vec3 oc = r.Origin( ) - m_center;
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
				Vec3 outwardNormal = ( rec.m_hitPosition - m_center ) / m_radius;
				rec.SetFrontFaceNormal( r, outwardNormal );
				rec.m_material = m_material;
				return true;
			}

			t = ( -halfB + root ) / a;
			if ( t > tMin && t < tMax )
			{
				rec.m_t = t;
				rec.m_hitPosition = r.At( t );
				Vec3 outwardNormal = ( rec.m_hitPosition - m_center ) / m_radius;
				rec.SetFrontFaceNormal( r, outwardNormal );
				rec.m_material = m_material;
				return true;
			}
		}
		
		return false;
	}

private:
	Point3 m_center;
	double m_radius;
};