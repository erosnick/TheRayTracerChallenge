#include "World.h"
#include "Sphere.h"
#include "Ray.h"
#include "Intersection.h"
#include "Memory.h"
#include "Array.h"

template<typename T>
CUDA_HOST_DEVICE bool compareLess(const T& a, const T& b) {
    return a < b;
}

template<typename T>
using compareFunctor = bool (*)(const T& a, const T& b);

template<typename T>
CUDA_HOST_DEVICE void sort(const Array<T>& array, const compareFunctor<T>& compare) {
    for (auto i = 1; i < array.size(); i++) {
        auto j = i;

        while (j > 0 && compare(array[j], array[j - 1])) {
            Memory::swap(array[j], array[j - 1]);
            j--;
        }
    }
}

CUDA_HOST_DEVICE World::~World() {
    for (auto i = 0; i < objects.size(); i++) {
        delete objects[i];
    }

    for (auto i = 0; i < lights.size(); i++) {
        delete lights[i];
    }
}

CUDA_HOST_DEVICE void World::intersect(const Ray& ray, Array<Intersection>& totalIntersections) {
    for (auto i = 0; i < objectCount(); i++) {
        objects[i]->intersect(ray, totalIntersections);
    }

    if (totalIntersections.size() > 0) {
        sort(totalIntersections, compareLess);
    }
}

//CUDA_HOST_DEVICE void World::intersect(const Ray& ray, Array<Intersection>& totalIntersections) {
//    for (auto i = 0; i < objectCount(); i++) {
//        Array<Intersection> intersections;
//        if (objects[i]->intersect(ray, intersections)) {
//            for (auto j = 0; j < intersections.size(); j++) {
//                totalIntersections.add(intersections[j]);
//            }
//        }
//    }
//
//    if (totalIntersections.size() > 0) {
//        sort(totalIntersections, compareLess);
//    }
//}