#pragma once

#include "Hittable.h"
#include "Random.h"
#include "Ray.h"
#include "Vec3.h"

class Material
{
public:
	__device__ virtual bool Scatter( curandState_t* randState, const Ray& inRay, const HitRecord& rec, Color& attenuation, Ray& scattered ) const = 0;
};

class Lambertian : public Material
{
public:
	__device__ Lambertian( const Color& albedo ) : m_albedo( albedo ) {}

	__device__ virtual bool Scatter( curandState_t* randState, const Ray& inRay, const HitRecord& rec, Color& attenuation, Ray& scattered ) const override
	{
		Vec3 scatteredDir = rec.m_normal + RandomUnitVector( randState ); // RandomInHemishpere( randState, rec.m_normal );

		if ( scatteredDir.NearZero( ) )
		{
			scatteredDir = rec.m_normal;
		}

		scattered = Ray( rec.m_hitPosition, scatteredDir, inRay.Time( ) );
		attenuation = m_albedo;
		return true;
	}

private:
	Color m_albedo;
};

class Metal : public Material
{
public:
	__device__ Metal( const Color& albedo, double fuzz ) : m_albedo( albedo ), m_fuzz( fuzz < 1 ? fuzz : 1 ) {}

	__device__ virtual bool Scatter( curandState_t* randState, const Ray& inRay, const HitRecord& rec, Color& attenuation, Ray& scattered ) const override
	{
		Vec3 reflected = Reflect( Normalize( inRay.Direction() ), rec.m_normal );
		scattered = Ray( rec.m_hitPosition, reflected + m_fuzz * RandomInUnitSphere( randState ), inRay.Time( ) );
		attenuation = m_albedo;
		return ( Dot( reflected, rec.m_normal ) > 0 );
	}

private:
	Color m_albedo;
	double m_fuzz;
};

class Dielectric : public Material
{
public:
	__device__ Dielectric( double indexOfRefraction ) : m_ir( indexOfRefraction ) {}

	__device__ virtual bool Scatter( curandState_t* randState, const Ray& inRay, const HitRecord& rec, Color& attenuation, Ray& scattered ) const override
	{
		attenuation = Color( 1.0, 1.0, 1.0 );
		double refractionRatio = rec.m_frontFace ? ( 1.0 / m_ir ) : m_ir;

		Vec3 unitDirection = Normalize( inRay.Direction( ) );
		double cosTheta = fmin( Dot( -unitDirection, rec.m_normal ), 1.0 );
		double sinTheta = sqrt( 1 - cosTheta * cosTheta );

		bool cannotRefract = refractionRatio * sinTheta > 1.0;
		Vec3 direction;

		if ( cannotRefract || ( Reflectance( cosTheta, refractionRatio ) > RandomDouble( randState ) ) )
		{
			direction = Reflect( unitDirection, rec.m_normal );
		}
		else
		{
			direction = Refract( unitDirection, rec.m_normal, refractionRatio );
		}

		scattered = Ray( rec.m_hitPosition, direction, inRay.Time( ) );
		return true;
	}

private:
	__device__ static double Reflectance( double cosine, double refractiveIdx )
	{
		double r0 = ( 1.0 - refractiveIdx ) / ( 1.0 + refractiveIdx );
		r0 = r0 * r0;
		return r0 + ( 1.0 - r0 ) * pow( ( 1.0 - cosine ), 5 );
	}

	double m_ir;
};