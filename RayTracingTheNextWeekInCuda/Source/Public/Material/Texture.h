#pragma once

#include "Perlin.h"
#include "Vec3.h"

#include <cuda_runtime.h>

class Texture
{
public:
	__device__ virtual ~Texture( ) {}

	__device__ virtual Color Value( double u, double v, const Point3& p ) const = 0;
};

class SolidColor : public Texture
{
public:
	__device__ SolidColor( ) {}
	__device__ SolidColor( Color c ) : m_colorValue( c ) {}

	__device__ virtual Color Value( double u, double v, const Point3& p ) const override
	{
		return m_colorValue;
	}

private:
	Color m_colorValue;
};

class CheckerTexture : public Texture
{
public:
	__device__ CheckerTexture( ) {}
	__device__ CheckerTexture( Texture* odd, Texture* even  ) : m_odd( odd ), m_even( even ) {}
	__device__ CheckerTexture( Color odd, Color even ) : m_odd( new SolidColor( odd ) ), m_even( new SolidColor( even ) ) {}

	__device__ ~CheckerTexture( ) 
	{
		delete m_odd;
		delete m_even;
	}

	__device__ virtual Color Value( double u, double v, const Point3& p ) const override
	{
		auto sines = sin( 10 * p.X( ) ) * sin( 10 * p.Y( ) ) * sin( 10 * p.Z( ) );
		if ( sines < 0 )
		{
			return m_odd->Value( u, v, p );
		}
		else
		{
			return m_even->Value( u, v, p );
		}
	}

private:
	Texture* m_odd = nullptr;
	Texture* m_even = nullptr;
};

class NoiseTexture : public Texture
{
public:
	__device__ NoiseTexture( Perlin* perlin ) : m_perlin( perlin ) {}

	__device__ virtual Color Value( double u, double v, const Point3& p ) const override
	{
		return Color( 1, 1, 1 ) * m_perlin->Noise( p );
	}

private:
	Perlin* m_perlin;
};