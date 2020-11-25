#pragma once

#include "Hittable.h"
#include "Material.h"

class HittableList : public Hittable
{
public:
	__device__ virtual bool Hit( const Ray& r, double tMin, double tMax, HitRecord& rec ) const override
	{
		HitRecord tempRec;
		bool hitAnything = false;
		double closestT = tMax;

		for ( std::size_t i = 0; i < m_size; ++i )
		{
			if ( m_objects[i]->Hit( r, tMin, closestT, tempRec ) )
			{
				hitAnything = true;
				closestT = tempRec.m_t;
				rec = tempRec;
			}
		}

		return hitAnything;
	}

	__device__ void Add( Hittable* object )
	{
		if ( m_capacity == m_size )
		{
			Grow( );
		}

		m_objects[m_size] = object;
		++m_size;
	}

	__device__ void Grow( )
	{
		m_capacity = m_capacity * 2 + 1;
		Hittable** newObjects = new Hittable*[m_capacity];

		for ( std::size_t i = 0; i < m_size; ++i )
		{
			newObjects[i] = m_objects[i];
			m_objects[i] = nullptr;
		}

		delete[] m_objects;
		m_objects = newObjects;
	}

	__device__ void Clear( )
	{
		for ( std::size_t i = 0; i < m_size; ++i )
		{
			delete m_objects[i]->GetMaterial( );
			delete m_objects[i];
		}

		delete[] m_objects;
	}

private:
	Hittable** m_objects = nullptr;
	std::size_t m_size = 0;
	std::size_t m_capacity = 0;
};