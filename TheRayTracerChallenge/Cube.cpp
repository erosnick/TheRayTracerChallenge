#include "Cube.h"
#include "Intersection.h"
#include <algorithm>
#include "Plane.h"

void Cube::setTransformation(const Matrix4& inTransformation) {
    Shape::setTransformation(inTransformation);

    for (auto& plane : planes) {
        plane->transform(transformation);
    }
}

Tuple Cube::normalAt(const Tuple& position) const {
    //auto rotation = transformation;
    //rotation[0][3] = 0.0;
    //rotation[1][3] = 0.0;
    //rotation[2][3] = 0.0;
    //auto transformedPosition = position;

    //auto maxComponent = std::max(std::abs(transformedPosition.x),
    //                    std::max(std::abs(transformedPosition.y),
    //                             std::abs(transformedPosition.z)));

    //auto normal = vector(0.0);

    //if ((maxComponent - std::abs(transformedPosition.x)) < Math::epsilon) {
    //    normal = vector(transformedPosition.x, 0.0, 0.0);
    //}
    //else if ((maxComponent - std::abs(transformedPosition.y)) < Math::epsilon) {
    //    normal = vector(0.0, transformedPosition.y, 0.0);
    //}

    //normal = vector(0.0, 0.0, -transformedPosition.z);

    //normal = transformation * normal;

    //return normal.normalize();

    for (const auto& plane : planes) {
        if (plane->onPlane(position)) {
            return plane->normal;
        }
    }

    return vector(0.0);
}

InsersectionSet Cube::intersect(const Ray& ray, bool bTransformRay) {
    
    //auto transformedRay = ray;
    auto xtmin = -Math::infinityd;
    if (planes[0]->intersect(ray).size() > 0) {
        xtmin = planes[0]->intersect(ray)[0].t;
    }

    auto xtmax = -Math::infinityd;

    if (planes[1]->intersect(ray).size() > 0) {
        xtmax = planes[1]->intersect(ray)[0].t;
    }

    if (xtmin > xtmax) {
        std::swap(xtmin, xtmax);
    }

    auto ytmin = -Math::infinityd;

    if (planes[2]->intersect(ray).size() > 0) {
        ytmin = planes[2]->intersect(ray)[0].t;
    }

    auto ytmax = -Math::infinityd;

    if (planes[3]->intersect(ray).size() > 0) {
        ytmax = planes[3]->intersect(ray)[0].t;
    }

    if (ytmin > ytmax) {
        std::swap(ytmin, ytmax);
    }

    auto ztmin = -Math::infinityd;

    if (planes[4]->intersect(ray).size() > 0) {
        ztmin = planes[4]->intersect(ray)[0].t;
    }

    auto ztmax = -Math::infinityd;

    if (planes[5]->intersect(ray).size() > 0) {
        ztmax = planes[5]->intersect(ray)[0].t;
    }

    if (ztmin > ztmax) {
        std::swap(ztmin, ztmax);
    }

    auto tmin = std::max(xtmin, std::max(ytmin, ztmin));
    auto tmax = std::min(xtmax, std::min(ytmax, ztmax));

    //transformedRay.origin = transformation.inverse() * transformedRay.origin;
    //transformedRay.direction = transformation.inverse() * transformedRay.direction;

    //auto [xtmin, xtmax] = checkAxis(transformedRay.origin.x, transformedRay.direction.x);
    //auto [ytmin, ytmax] = checkAxis(transformedRay.origin.y, transformedRay.direction.y);
    //auto [ztmin, ztmax] = checkAxis(transformedRay.origin.z, transformedRay.direction.z);

    if ((tmin > tmax) || (tmin == -Math::infinityd || tmax == -Math::infinityd)) {
        return InsersectionSet();
    }

    auto position1 = ray.position(tmin);
    auto position2 = ray.position(tmax);
    auto normal1 = normalAt(position1);
    auto normal2 = normalAt(position2);

    return { { true, !bIsLight, 1, tmin, GetPtr(), position1, normal1, ray }, { true, !bIsLight, 1, tmax, GetPtr(), position2, normal2, ray } };
    return InsersectionSet();
}

std::tuple<double, double> Cube::checkAxis(double origin, double direction) {
    auto tminNumerator = -1.0 - origin;
    auto tmaxNumerator = 1.0 - origin;

    auto tmin = Math::infinityd;
    auto tmax = Math::infinityd;

    if (std::abs(direction) >= Math::epsilon) {
        tmin = tminNumerator / direction;
        tmax = tmaxNumerator / direction;
    }
    else {
        tmin = tminNumerator * Math::infinityd;
        tmax = tmaxNumerator * Math::infinityd;
    }
    if (tmin > tmax) {
        std::swap(tmin, tmax);
    }

    return { tmin, tmax };
}

void Cube::initPlanes() {
    planes.resize(6);

    planes[0] = std::make_shared<Plane>(point(-1.0,  0.0,  0.0), -Math::xAxis);
    planes[1] = std::make_shared<Plane>(point( 1.0,  0.0,  0.0),  Math::xAxis);
    planes[2] = std::make_shared<Plane>(point( 0.0, -1.0,  0.0), -Math::yAxis);
    planes[3] = std::make_shared<Plane>(point( 0.0,  1.0,  0.0),  Math::yAxis);
    planes[4] = std::make_shared<Plane>(point( 0.0,  0.0, -1.0),  Math::zAxis);
    planes[5] = std::make_shared<Plane>(point( 0.0,  0.0,  1.0), -Math::zAxis);
}
