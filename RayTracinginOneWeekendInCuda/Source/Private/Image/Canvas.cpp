#include "Canvas.h"

#include <cstdio>
#include <cstring>
#include <fstream>

void Canvas::WriteFile( const fs::path& filePath ) const
{
	std::ofstream ppm( filePath );
	if ( ppm.good( ) )
	{
		ppm << "P3\n" << m_width << ' ' << m_height << "\n255\n";

		char color[13] = ""; // 3 space + 3 color channel( max 3 digits ) + null character = 13
		std::size_t writedChar = 1;
		for ( std::size_t i = 0; i < m_pixels.size( ); ++i )
		{
			const Pixel& pixel = m_pixels[i];
			sprintf( color, "%d %d %d ", pixel.m_color[0], pixel.m_color[1], pixel.m_color[2] );

			std::size_t len = strlen( color );
			if ( writedChar + len > 70 )
			{
				writedChar = 1;
				ppm << "\n ";
			}

			writedChar += len;
			ppm << color;
		}
		ppm << "\n ";
	}
}
