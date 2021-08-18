#include "World.h"
#include "Sphere.h"
#include "Ray.h"
#include "Intersection.h"

CUDA_HOST_DEVICE void swap(Intersection* intersections, int32_t a, int32_t b) {
    auto temp = intersections[b];
    intersections[b] = intersections[a];
    intersections[a] = temp;
}

CUDA_HOST_DEVICE void sort(Intersection* intersections, int32_t count) {
    for (auto i = 1; i < count; i++) {
        auto j = i;

        while (j > 0 && intersections[j].t < intersections[j - 1].t) {
            swap(intersections, j, j - 1);
            j--;
        }
    }
}

void World::intersect(const Ray& ray, Intersection* totalIntersections, int32_t* count) {
    for (auto i = 0; i < objectCount(); i++) {
        Intersection intersections[2];
        if (objects[i]->intersect(ray, intersections)) {
            totalIntersections[(*count)++] = intersections[0];
            totalIntersections[(*count)++] = intersections[1];
        }
    }

    if ((*count) > 0) {
        sort(totalIntersections, (*count));
    }
}