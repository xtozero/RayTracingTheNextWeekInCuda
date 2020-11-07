#pragma once

#include <cuda_runtime.h>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

struct Pixel
{
	constexpr static std::size_t ColorChannel = 3;
	unsigned char m_color[ColorChannel];
};

class Canvas
{
public:
	Canvas( std::size_t width, std::size_t height ) : m_width( width ), m_height( height )
	{
		m_pixels.resize( m_width * m_height );
	}

	__host__ Pixel* Pixels( )
	{
		return m_pixels.data( );
	}

	__host__ std::size_t Size( ) const
	{
		return m_width * m_height * sizeof( Pixel );
	}

	__host__ std::size_t Width( ) const
	{
		return m_width;
	}

	__host__ std::size_t Height( ) const
	{
		return m_height;
	}

	void WriteFile( const fs::path& filePath ) const;

private:
	std::size_t m_width;
	std::size_t m_height;

	std::vector<Pixel> m_pixels;
};