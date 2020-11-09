#include "Camera.h"
#include "Canvas.h"
#include "HittableList.h"
#include "Material.h"
#include "MovingSphere.h"
#include "Random.h"
#include "Ray.h"
#include "Sphere.h"
#include "Vec3.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ Color RayColor( curandState_t* randState, const Ray& r, HittableList** world, int depth )
{
	HitRecord rec;
	Color totalAttenuation( 1.0, 1.0, 1.0 );
	Ray curRay = r;

	for ( int i = 0; i < depth; ++i )
	{
		if ( ( *world )->Hit( curRay, 0.001, DBL_MAX, rec ) )
		{
			Color attenuation( 1.0, 1.0, 1.0 );
			Ray scattered;

			if ( rec.m_material->Scatter( randState, curRay, rec, attenuation, scattered ) )
			{
				totalAttenuation = totalAttenuation * attenuation;
				curRay = scattered;
			}
			else
			{
				return Color( 0, 0, 0 );
			}
		}
		else
		{
			Vec3 unitDirection = Normalize( curRay.Direction( ) );
			double t = 0.5 * ( unitDirection.Y( ) + 1 );
			Color c = ( 1 - t ) * Color( 1, 1, 1 ) + t * Color( 0.5, 0.7, 1 );
			return totalAttenuation * c;
		}
	}

	return Color( 0, 0, 0 );
}

__global__ void FillCanvas( Pixel* devPixels, HittableList** world, std::size_t width, std::size_t height, Camera cam, int numSample, int maxRayDepth )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * width;
	y = height - y - 1; // invert y

	if ( x < width && y < height )
	{
		curandState_t randState;
		curand_init( offset, 0, 0, &randState );
		Color pixelColor( 0, 0, 0 );
		for ( int i = 0; i < numSample; ++i )
		{
			double u = double( x + RandomDouble( &randState ) ) / ( width - 1 );
			double v = double( y + RandomDouble( &randState ) ) / ( height - 1 );
			Ray r = cam.GetRay( &randState, u, v );
			pixelColor += RayColor( &randState, r, world, maxRayDepth );
		}
		
		WriteColor( devPixels[offset].m_color, pixelColor, numSample );
	}
}

__global__ void CreateWorld( HittableList** world )
{
	*world = new HittableList( );
	//( *world )->Add( new Sphere( Point3( 0, -100.5, -1 ), 100, new Lambertian( Color( 0.8, 0.8, 0.0 ) ) ) );
	//( *world )->Add( new Sphere( Point3( 0, 0, -1 ), 0.5, new Lambertian( Color( 0.1, 0.2, 0.5 ) ) ) );
	//( *world )->Add( new Sphere( Point3( -1, 0, -1 ), 0.5, new Dielectric( 1.5 ) ) );
	//( *world )->Add( new Sphere( Point3( -1, 0, -1 ), -0.45, new Dielectric( 1.5 ) ) );
	//( *world )->Add( new Sphere( Point3( 1, 0, -1 ), 0.5, new Metal( Color( 0.8, 0.6, 0.2 ), 0.0 ) ) );

	//double R = cos( CUDART_PI / 4 );
	//( *world )->Add( new Sphere( Point3( -R, 0, -1 ), R, new Lambertian( Color( 0, 0, 1 ) ) ) );
	//( *world )->Add( new Sphere( Point3( R, 0, -1 ), R, new Lambertian( Color( 1, 0, 0 ) ) ) );

	( *world )->Add( new Sphere( Point3( 0, -1000, 0 ), 1000, new Lambertian( new CheckerTexture( Color( 0.2, 0.3, 0.1 ), Color( 0.9, 0.9, 0.9 ) ) ) ) );

	curandState_t randState;
	curand_init( 1024, 768, 0, &randState );
	for ( int i = -11; i < 11; ++i )
	{
		for ( int j = -11; j < 11; ++j )
		{
			double chooseMaterial = RandomDouble( &randState );
			Point3 center( i + 0.9 * RandomDouble( &randState ), 0.2, j + 0.9 * RandomDouble( &randState ) );

			if ( ( center - Point3( 4, 0.2, 0 ) ).Length( ) > 0.9 ) {
				if ( chooseMaterial < 0.8 ) {
					// diffuse
					Color albedo = Random( &randState ) * Random( &randState );
					Point3 center2 = center + Vec3( 0, RandomDouble( &randState, 0, 0.5 ), 0 );
					( *world )->Add( new MovingSphere( center, center2, 0.0, 1.0, 0.2, new Lambertian( albedo ) ) );
				}
				else if ( chooseMaterial < 0.95 ) {
					// metal
					Color albedo = Random( &randState, 0.5, 1 );
					double fuzz = RandomDouble( &randState, 0, 0.5 );
					( *world )->Add( new Sphere( center, 0.2, new Metal( albedo, fuzz ) ) );
				}
				else {
					// glass
					( *world )->Add( new Sphere( center, 0.2, new Dielectric( 1.5 ) ) );
				}
			}
		}
	}

	( *world )->Add( new Sphere( Point3( 0, 1, 0 ), 1.0, new Dielectric( 1.5 ) ) );

	( *world )->Add( new Sphere( Point3( -4, 1, 0 ), 1.0, new Lambertian( Color( 0.4, 0.2, 0.1 ) ) ) );

	( *world )->Add( new Sphere( Point3( 4, 1, 0 ), 1.0, new Metal( Color( 0.7, 0.6, 0.5 ), 0 ) ) );
}

__global__ void CreateTwoSpheresWorld( HittableList** world )
{
	*world = new HittableList( );

	( *world )->Add( new Sphere( Point3( 0, -10, 0 ), 10, new Lambertian( new CheckerTexture( Color( 0.2, 0.3, 0.1 ), Color( 0.9, 0.9, 0.9 ) ) ) ) );
	( *world )->Add( new Sphere( Point3( 0, 10, 0 ), 10, new Lambertian( new CheckerTexture( Color( 0.2, 0.3, 0.1 ), Color( 0.9, 0.9, 0.9 ) ) ) ) );
}

__global__ void DestroyWorld( HittableList** world )
{
	(*world)->Clear( );
	delete *world;
}

int main( )
{
	Point3 lookFrom( 13, 2, 3 );
	Point3 lookAt( 0, 0, 0 );
	double fov = 40.0;
	double aperture = 0.0;

	HittableList** world = nullptr;
	cudaMalloc( (void**)&world, sizeof( HittableList* ) );

	switch ( 0 )
	{
	case 1:
		CreateWorld<<<1, 1>>>( world );
		fov = 20.0;
		aperture = 0.1;
		break;
	default:
	case 2:
		CreateTwoSpheresWorld<<<1, 1>>>( world );
		fov = 20.0;
		break;
	}

	// camera
	constexpr double aspectRatio = 16.0 / 9.0;
	Camera cam( lookFrom, lookAt, fov, aspectRatio, aperture, 10, 0.0, 1.0 );

	// canvas
	constexpr int canvasWidth = 400;
	const int canvasHeight = static_cast<int>( canvasWidth / aspectRatio );
	Canvas canvas( canvasWidth, canvasHeight );

	Pixel* devPixels = nullptr;
	cudaMalloc( (void**)&devPixels, canvas.Size( ) );

	int curDevice = 0;
	cudaGetDevice( &curDevice );

	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, curDevice );

	dim3 grids( static_cast<unsigned int>( ( canvas.Width() + 7 ) / 8 ) , static_cast<unsigned int>( ( canvas.Height( ) + 7 ) / 8 ) );
	dim3 threads( 8, 8 );
	FillCanvas<<<grids, threads>>>( devPixels, world, canvas.Width(), canvas.Height(), cam, 100, 50 );

	DestroyWorld<<<1, 1>>>( world );

	cudaMemcpy( canvas.Pixels(), devPixels, canvas.Size( ), cudaMemcpyDeviceToHost );
	cudaFree( devPixels );
	cudaFree( world );

	canvas.WriteFile( "./image2_2.ppm" );
}