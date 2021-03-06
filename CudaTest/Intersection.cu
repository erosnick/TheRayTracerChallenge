#pragma once

#include "Intersection.h"
#include "Shape.h"
#include "Material.h"

CUDA_HOST_DEVICE HitInfo prepareComputations(const Intersection& hit, const Ray& ray, const Array<Intersection>& intersections) {
    // Instantiate a data structure for storing some precomputed values
    HitInfo hitInfo;

    // Copy the intersection's properties for convenience
    hitInfo.bHit = hit.bHit;
    hitInfo.t = hit.t;
    hitInfo.object = hit.object;

    // Precompute some useful values
    //hitInfo.position = ray.position(hitInfo.t);
    hitInfo.position = ray.position(hitInfo.t);
    hitInfo.viewDirection = -ray.direction;
    hitInfo.normal = hitInfo.object->normalAt(hitInfo.position);

    if (hitInfo.normal.dot(hitInfo.viewDirection) < 0.0) {
        hitInfo.bInside = true;
        hitInfo.normal = -hitInfo.normal;
    }

    hitInfo.overPosition = hitInfo.position + hitInfo.normal * Math::epsilon;
    hitInfo.underPosition = hitInfo.position - hitInfo.normal * Math::epsilon;

    if (hitInfo.object->material->reflective > 0.0) {
        hitInfo.reflectVector = reflect(ray.direction, hitInfo.normal);
    }

    // List of objects have been encountered but not yet exited
    Array<Shape*> container;

    for (const auto& intersection : intersections) {
        // If the intersection is the hit, set n1 to the refractive index of the last object
        // in the containers list.If that list is empty, then there is no containing object,
        // and n1 should be set to 1.
        if (intersection == hit) {
            if (container.size() == 0) {
                hitInfo.n1 = 1.0;
            }
            else {
                hitInfo.n1 = container[container.size() - 1]->material->refractiveIndex;
            }
        }

        // If the intersection's object is already in the containers list, then this 
        // intersection must be exiting the object.Remove the object from the containers
        // list in this case.Otherwise, the intersection is entering the object, and
        // the object should be added to the end of the list.
        auto bFound = false;
        for (auto i = 0; i < container.size(); i++) {
            if (intersection.object == container[i]) {
                container.remove(intersection.object);
                bFound = true;
                break;
            }
        }

        if (!bFound) {
            container.add(intersection.object);
        }

        // If the intersection is the hit, set n2 to the refractive index of the last object
        // in the containers list.If that list is empty, then again, there is no containing
        // object and n2 should be set to 1.
        if (intersection == hit) {
            if (container.size() == 0) {
                hitInfo.n2 = 1.0;
            }
            else {
                hitInfo.n2 = container[container.size() - 1]->material->refractiveIndex;
            }
            break;
        }
    }

    return hitInfo;
}

template<typename T>
CUDA_HOST_DEVICE void remove(T array[], int32_t& size, T value) {
    int32_t foundIndex = -1;

    for (auto i = 0; i < size; i++) {
        if (array[i] == value) {
            foundIndex = i;
            break;
        }
    }

    if (foundIndex >= 0) {
        for (auto i = foundIndex; i < size - 1; i++) {
            array[i] = array[i + 1];
        }

        size--;
    }
}

CUDA_HOST_DEVICE HitInfo prepareComputations(const Intersection& hit, const Ray& ray, Intersection intersections[], int32_t size) {
    // Instantiate a data structure for storing some precomputed values
    HitInfo hitInfo;

    // Copy the intersection's properties for convenience
    hitInfo.bHit = hit.bHit;
    hitInfo.t = hit.t;
    hitInfo.object = hit.object;

    // Precompute some useful values
    //hitInfo.position = ray.position(hitInfo.t);
    hitInfo.position = ray.position(hitInfo.t);
    hitInfo.viewDirection = -ray.direction;
    hitInfo.normal = hitInfo.object->normalAt(hitInfo.position);

    if (hitInfo.normal.dot(hitInfo.viewDirection) < -Math::epsilon) {
        hitInfo.bInside = true;
        hitInfo.normal = -hitInfo.normal;
    }

    hitInfo.overPosition = hitInfo.position + hitInfo.normal * Math::epsilon;
    hitInfo.underPosition = hitInfo.position - hitInfo.normal * Math::epsilon;

    if (hitInfo.object->material->reflective > Math::epsilon) {
        hitInfo.reflectVector = reflect(ray.direction, hitInfo.normal);
    }

    // List of objects have been encountered but not yet exited
    Shape* container[4];
    auto containerSize = 0;
    for (auto i = 0; i < size; i++) {
        // If the intersection is the hit, set n1 to the refractive index of the last object
        // in the containers list.If that list is empty, then there is no containing object,
        // and n1 should be set to 1.
        auto intersection = intersections[i];
        if (intersection == hit) {
            if (containerSize == 0) {
                hitInfo.n1 = 1.0;
            }
            else {
                hitInfo.n1 = container[containerSize - 1]->material->refractiveIndex;
            }
        }

        // If the intersection's object is already in the containers list, then this 
        // intersection must be exiting the object.Remove the object from the containers
        // list in this case.Otherwise, the intersection is entering the object, and
        // the object should be added to the end of the list.
        auto bFound = false;
        for (auto i = 0; i < containerSize; i++) {
            if (intersection.object == container[i]) {
                remove(container, containerSize, intersection.object);
                bFound = true;
                break;
            }
        }

        if (!bFound) {
            container[containerSize] = intersection.object;
            containerSize++;
        }

        // If the intersection is the hit, set n2 to the refractive index of the last object
        // in the containers list.If that list is empty, then again, there is no containing
        // object and n2 should be set to 1.
        if (intersection == hit) {
            if (containerSize == 0) {
                hitInfo.n2 = 1.0;
            }
            else {
                hitInfo.n2 = container[containerSize - 1]->material->refractiveIndex;
            }
            break;
        }
    }

    return hitInfo;
}