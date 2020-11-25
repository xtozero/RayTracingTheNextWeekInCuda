#pragma once

#include "Hittable.h"

#include <cuda_runtime.h>

class XYRect : public Hittable
{
public:
	__device__ XYRect( ) {}

	__device__ XYRect( double x0, double x1, double y0, double y1, double k, Material* mat ) : m_x0( x0 ), m_x1( x1 ), m_y0( y0 ), m_y1( y1 ), m_k( k ), m_material( mat ) {}

	__device__ virtual bool Hit( const Ray& r, double tMin, double tMax, HitRecord& rec ) const override
	{
		double t = ( m_k - r.Origin( ).Z( ) ) / r.Direction( ).Z( );

		if ( t < tMin || t > tMax )
		{
			return false;
		}

		double x = r.Origin( ).X() + t * r.Direction( ).X();
		double y = r.Origin( ).Y() + t * r.Direction( ).Y();

		if ( x < m_x0 || x > m_x1 || y < m_y0 || y > m_y1 )
		{
			return false;
		}

		rec.m_u = ( x - m_x0 ) / ( m_x1 - m_x0 );
		rec.m_v = ( y - m_y0 ) / ( m_y1 - m_y0 );
		rec.m_t = t;
		rec.SetFrontFaceNormal( r, Vec3( 0, 0, 1 ) );
		rec.m_material = m_material;
		rec.m_hitPosition = r.At( t );
		return true;
	}

private:
	Material* m_material;

	double m_x0;
	double m_x1;
	double m_y0;
	double m_y1;
	double m_k;
};

class XZRect : public Hittable
{
public:
	__device__ XZRect( ) {}

	__device__ XZRect( double x0, double x1, double z0, double z1, double k, Material* mat ) : m_x0( x0 ), m_x1( x1 ), m_z0( z0 ), m_z1( z1 ), m_k( k ), m_material( mat ) {}

	__device__ virtual bool Hit( const Ray& r, double tMin, double tMax, HitRecord& rec ) const override
	{
		double t = ( m_k - r.Origin( ).Y( ) ) / r.Direction( ).Y( );

		if ( t < tMin || t > tMax )
		{
			return false;
		}

		double x = r.Origin( ).X( ) + t * r.Direction( ).X( );
		double z = r.Origin( ).Z( ) + t * r.Direction( ).Z( );

		if ( x < m_x0 || x > m_x1 || z < m_z0 || z > m_z1 )
		{
			return false;
		}

		rec.m_u = ( x - m_x0 ) / ( m_x1 - m_x0 );
		rec.m_v = ( z - m_z0 ) / ( m_z1 - m_z0 );
		rec.m_t = t;
		rec.SetFrontFaceNormal( r, Vec3( 0, 1, 0 ) );
		rec.m_material = m_material;
		rec.m_hitPosition = r.At( t );
		return true;
	}

private:
	Material* m_material;

	double m_x0;
	double m_x1;
	double m_z0;
	double m_z1;
	double m_k;
};

class YZRect : public Hittable
{
public:
	__device__ YZRect( ) {}

	__device__ YZRect( double y0, double y1, double z0, double z1, double k, Material* mat ) : m_y0( y0 ), m_y1( y1 ), m_z0( z0 ), m_z1( z1 ), m_k( k ), m_material( mat ) {}

	__device__ virtual bool Hit( const Ray& r, double tMin, double tMax, HitRecord& rec ) const override
	{
		double t = ( m_k - r.Origin( ).X( ) ) / r.Direction( ).X( );

		if ( t < tMin || t > tMax )
		{
			return false;
		}

		double y = r.Origin( ).Y( ) + t * r.Direction( ).Y( );
		double z = r.Origin( ).Z( ) + t * r.Direction( ).Z( );

		if ( y < m_y0 || y > m_y1 || z < m_z0 || z > m_z1 )
		{
			return false;
		}

		rec.m_u = ( y - m_y0 ) / ( m_y1 - m_y0 );
		rec.m_v = ( z - m_z0 ) / ( m_z1 - m_z0 );
		rec.m_t = t;
		rec.SetFrontFaceNormal( r, Vec3( 1, 0, 0 ) );
		rec.m_material = m_material;
		rec.m_hitPosition = r.At( t );
		return true;
	}

private:
	Material* m_material;

	double m_y0;
	double m_y1;
	double m_z0;
	double m_z1;
	double m_k;
};