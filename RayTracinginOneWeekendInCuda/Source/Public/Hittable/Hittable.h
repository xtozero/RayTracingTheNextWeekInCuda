#pragma once

#include "Ray.h"
#include "Vec3.h"

class Material;

class HitRecord
{
public:
	Point3 m_hitPosition;
	Vec3 m_normal;
	Material* m_material;
	double m_t;
	bool m_frontFace;

	__device__ void SetFrontFaceNormal( const Ray& r, const Vec3& outwardNormal )
	{
		m_frontFace = Dot( r.Direction( ), outwardNormal ) < 0;
		m_normal = m_frontFace ? outwardNormal : -outwardNormal;
	}
};

class Hittable
{
public:
	__device__ Hittable( ) {}
	__device__ Hittable( Material* material ) : m_material( material ) {}
	__device__ virtual bool Hit( const Ray& r, double tMin, double tMax, HitRecord& rec ) const = 0;
	__device__ Material* GetMaterial() { return m_material; }

protected:
	Material* m_material;
};