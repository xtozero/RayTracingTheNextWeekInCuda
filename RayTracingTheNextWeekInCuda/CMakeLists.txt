set (ENTRY_POINT
	./Source/Private/EntryPoint/Main.cu
)

set (HITTABLE
	./Source/Public/Hittable/Hittable.h
	./Source/Public/Hittable/HittableList.h
	./Source/Public/Hittable/MovingSphere.h
	./Source/Public/Hittable/Sphere.h
)

set (IMAGE
	./Source/Public/Image/Canvas.h
	./Source/Private/Image/Canvas.cpp
)

set (MATERIAL
	./Source/Public/Material/Material.h
	./Source/Public/Material/Texture.h
)

set (MATH
	./Source/Public/Math/Random.h
	./Source/Public/Math/Ray.h
	./Source/Public/Math/Vec3.h
)

set (View
	./Source/Public/View/Camera.h
)

add_executable(RayTracinginOneWeekendInCuda ${ENTRY_POINT}
											${HITTABLE}
											${IMAGE}
											${MATERIAL}
											${MATH}
											${View})

target_include_directories(RayTracinginOneWeekendInCuda
	PRIVATE
		Source/Public/Hittable
		Source/Public/Image
		Source/Public/Material
		Source/Public/Math
		Source/Public/View
)