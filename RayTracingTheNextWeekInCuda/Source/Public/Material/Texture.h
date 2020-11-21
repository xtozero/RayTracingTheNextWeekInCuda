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
	__device__ NoiseTexture( Perlin* perlin, double scale ) : m_perlin( perlin ), m_scale( scale ) {}

	__device__ virtual Color Value( double u, double v, const Point3& p ) const override
	{
		return Color( 1, 1, 1 ) * 0.5 * ( 1 + sin( m_scale * p.Z( ) + 10 * m_perlin->Turb( p ) ) );
	}

private:
	Perlin* m_perlin;
	double m_scale = 1;
};

class ImageTexture : public Texture
{
public:
	__device__ ImageTexture( texture<uchar4, 2> image, int width, int height ) : m_image( image ), m_width( width ), m_height( height ) {}

	__device__ virtual Color Value( double u, double v, const Point3& p ) const override
	{
		uchar4 c = tex2D( m_image, static_cast<float>( u * m_width ), static_cast<float>( ( 1.0 - v ) * m_height ) );

		constexpr double denominator = ( 1.0 / 255.0 );
		return Color( c.x * denominator, c.y * denominator, c.z * denominator );
	}

private:
	texture<uchar4, 2> m_image;
	int m_width;
	int m_height;
};