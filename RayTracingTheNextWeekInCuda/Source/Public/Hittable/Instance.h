#pragma once

#include "Hittable.h"
#include "Material.h"

#include <math_constants.h>

class Translate : public Hittable
{
public:
	__device__ Translate( ) {}
	__device__ Translate( Hittable* hittable, const Vec3& displacement ) : m_hittable( hittable ), m_offset( displacement ) {}

	__device__ ~Translate( ) 
	{
		delete m_hittable;
	}

	__device__ virtual bool Hit( const Ray& r, double tMin, double tMax, HitRecord& rec ) const override
	{
		Ray movedRay( r.Origin( ) - m_offset, r.Direction( ), r.Time( ) );
		if ( m_hittable->Hit( movedRay, tMin, tMax, rec ) == false )
		{
			return false;
		}

		rec.m_hitPosition += m_offset;
		rec.SetFrontFaceNormal( movedRay, rec.m_normal );

		return true;
	}

	__device__ virtual Material* GetMaterial( ) { return m_hittable->GetMaterial( ); }

private:
	Hittable* m_hittable;
	Vec3 m_offset;
};

class RotateY : public Hittable
{
public:
	__device__ RotateY( ) {}
	__device__ RotateY( Hittable* hittable, double angle ) : m_hittable( hittable )
	{
		auto radians = CUDART_PI * angle / 180.0;
		m_sinTheta = sin( radians );
		m_cosTheta = cos( radians );
	}

	__device__ ~RotateY( )
	{
		delete m_hittable;
	}

	__device__ virtual bool Hit( const Ray& r, double tMin, double tMax, HitRecord& rec ) const override
	{
		auto origin = r.Origin( );
		auto direction = r.Direction( );

		origin[0] = m_cosTheta * r.Origin( )[0] - m_sinTheta * r.Origin( )[2];
		origin[2] = m_sinTheta * r.Origin( )[0] + m_cosTheta * r.Origin( )[2];

		direction[0] = m_cosTheta * r.Direction( )[0] - m_sinTheta * r.Direction( )[2];
		direction[2] = m_sinTheta * r.Direction( )[0] + m_cosTheta * r.Direction( )[2];

		Ray rotatedRay( origin, direction, r.Time( ) );

		if ( m_hittable->Hit( rotatedRay, tMin, tMax, rec ) == false )
		{
			return false;
		}

		auto p = rec.m_hitPosition;
		auto normal = rec.m_normal;

		p[0] = m_cosTheta * rec.m_hitPosition[0] + m_sinTheta * rec.m_hitPosition[2];
		p[2] = -m_sinTheta * rec.m_hitPosition[0] + m_cosTheta * rec.m_hitPosition[2];

		normal[0] = m_cosTheta * rec.m_normal[0] + m_sinTheta * rec.m_normal[2];
		normal[2] = -m_sinTheta * rec.m_normal[0] + m_cosTheta * rec.m_normal[2];

		rec.m_hitPosition = p;
		rec.SetFrontFaceNormal( rotatedRay, normal );

		return true;
	}

	__device__ virtual Material* GetMaterial( ) { return m_hittable->GetMaterial( ); }

private:
	Hittable* m_hittable;
	double m_sinTheta;
	double m_cosTheta;
};