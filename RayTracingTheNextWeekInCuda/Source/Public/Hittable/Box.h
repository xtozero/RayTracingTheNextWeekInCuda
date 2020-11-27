#pragma once

#include "AARect.h"
#include "Hittable.h"
#include "Vec3.h"

class Box : public Hittable
{
public:
	__device__ Box( ) {}
	__device__ ~Box( )
	{
		for ( Hittable* side : m_sides )
		{
			delete side->GetMaterial( );
			delete side;
		}
	}

	__device__ Box( const Point3& min, const Point3& max, Material* material ) 
	{
		m_sides[0] = new XYRect( min.X( ), max.X( ), min.Y( ), max.Y( ), min.Z( ), material );
		m_sides[1] = new XYRect( min.X( ), max.X( ), min.Y( ), max.Y( ), max.Z( ), material );
		m_sides[2] = new YZRect( min.Y( ), max.Y( ), min.Z( ), max.Z( ), min.X( ), material );
		m_sides[3] = new YZRect( min.Y( ), max.Y( ), min.Z( ), max.Z( ), max.X( ), material );
		m_sides[4] = new XZRect( min.X( ), max.X( ), min.Z( ), max.Z( ), min.Y( ), material );
		m_sides[5] = new XZRect( min.X( ), max.X( ), min.Z( ), max.Z( ), max.Y( ), material );
	}

	__device__ virtual bool Hit( const Ray& r, double tMin, double tMax, HitRecord& rec ) const override
	{
		HitRecord tempRec;
		bool hitAnything = false;
		double closestT = tMax;

		for ( Hittable* side : m_sides )
		{
			if ( side->Hit( r, tMin, closestT, tempRec ) )
			{
				hitAnything = true;
				closestT = tempRec.m_t;
				rec = tempRec;
			}
		}

		return hitAnything;
	}

private:
	Hittable* m_sides[6];
};