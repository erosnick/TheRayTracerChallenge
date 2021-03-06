#include "Shading.h"
#include "Tuple.h"
#include "Material.h"
#include "Light.h"
#include "World.h"
#include <Intersection.h>
#include "Pattern.h"
#include "Shape.h"

#include <algorithm>

CUDA_HOST_DEVICE Tuple lighting(Material* material, Shape* object, Light* light, const Tuple& position,
               const Tuple& viewDirection, const Tuple& normal, bool bInShadow, 
               bool bHalfLambert, bool bBlinnPhong) {
    auto materialColor = material->color;

    if (material->pattern) {
        materialColor = material->pattern->patternAtShape(object, position);
    }

    auto ambientColor = materialColor * material->ambient;
    
    if (bInShadow) {
        return ambientColor;
    }

    auto surfaceColor = light->intensity * materialColor;
    auto diffuseColor = surfaceColor * material->diffuse;
    auto specularColor = light->intensity * material->specular;
    
    auto lightDirection = (light->transformedPosition - position);
    auto distance = lightDirection.magnitude();

    auto attenuation = 1.0f;

    if (light->bAttenuation) {
        attenuation = 1.0f / (light->constant + light->linear * distance + light->quadratic * (distance * distance));
    }

    lightDirection = lightDirection / distance;

    auto diffuseTerm = normal.dot(lightDirection);
    auto diffuse = max(diffuseTerm, 0.0f) * attenuation;

    if (bHalfLambert) {
        ambientColor = Color::black;
        diffuse = diffuse * 0.5f + 0.5f;
    }

    auto specular = 0.0f;

    if (diffuseTerm > Math::epsilon) {
        auto reflectVector = 2.0f * (diffuseTerm) * normal - lightDirection;
        if (bBlinnPhong) {
            auto halfVector = (lightDirection + viewDirection) / (lightDirection + viewDirection).magnitude();
            specular = pow(max(halfVector.dot(normal), 0.0f), material->shininess * 2) * attenuation;
        }
        else {
            specular = pow(max(reflectVector.dot(viewDirection), 0.0f), material->shininess) * attenuation;
        }
    }

    auto finalColor = ambientColor + diffuseColor * diffuse + specularColor * specular;
    return finalColor;
}

CUDA_HOST_DEVICE Tuple lighting(Material* material, Shape* object, Light* light,
                           const HitInfo& hitInfo, bool bInShadow, 
                           bool bHalfLambert, bool bBlinnPhong) {
    return lighting(material, object, light, hitInfo.overPosition, hitInfo.viewDirection, hitInfo.normal, bInShadow, bHalfLambert, bBlinnPhong);
}

//CUDA_HOST_DEVICE bool isShadow(World* world, Light* light, const Tuple& position) {
//    auto toLight = light->transformedPosition - position;
//    const auto distance = toLight.magnitude();
//
//    auto ray = Ray(position, toLight.normalize());
//
//    Array<Intersection> intersections;
//    world->intersect(ray, intersections);
//
//    if (intersections.size() > 0) {
//        const auto& intersection = nearestHit(intersections);
//
//        if (intersection.bHit 
//        && !intersection.object->bIsLight
//        &&  intersection.object->material->bCastShadow
//        &&  intersection.t < distance) {
//            return true;
//        }
//    }
//
//    return false;
//}

CUDA_HOST_DEVICE bool isShadow(World* world, Light* light, const Tuple& position) {
    auto toLight = light->transformedPosition - position;
    const auto distance = toLight.magnitude();

    auto ray = Ray(position, toLight.normalize());

    Intersection intersections[4];
    auto size = 0;
    world->intersect(ray, intersections, size);

    if (size > 0) {
        const auto& intersection = nearestHit(intersections, size);

        if (intersection.bHit
            && !intersection.object->bIsLight
            && intersection.object->material->bCastShadow
            && intersection.t < distance) {
            return true;
        }
    }

    return false;
}

CUDA_HOST_DEVICE Tuple computeReflectionAndRefraction(const HitInfo& hitInfo, World* world, int32_t depth) {
    auto material = hitInfo.object->material;

    auto reflected = Color::black;

    if (material->reflective > Math::epsilon) {
        reflected = Color::white;
        auto reflectedHitInfo = hitInfo;
        for (auto i = 0; i < depth; i++) {
            if (!reflectedHitInfo.bHit) break;
            reflected *= reflectedColor(world, reflectedHitInfo);
        }
    }

    auto refracted = Color::black;

    if (material->transparency > Math::epsilon) {
        refracted = Color::white;
        auto refractedHitInfo = hitInfo;
        for (auto i = 0; i < depth; i++) {
            if (!refractedHitInfo.bHit) break;
            refracted *= refractedColor(world, refractedHitInfo);
        }
    }

    if (material->reflective > Math::epsilon && material->transparency > Math::epsilon) {
        auto reflectance = schlick(hitInfo);
        return reflected * reflectance + refracted * (1.0f - reflectance);
    }
    else {
        return reflected + refracted;
    }
}

CUDA_HOST_DEVICE Tuple shadeHit(World* world, const HitInfo& hitInfo, bool bHalfLambert, bool bBlinnPhong) {
    auto surface = Color::black;

    for (auto i = 0; i < world->ligthCount(); i++) {
        auto light = world->getLight(i);
        auto inShadow = isShadow(world, light, hitInfo.overPosition);
        surface += lighting(hitInfo.object->material, hitInfo.object, light, hitInfo, inShadow, bHalfLambert, bBlinnPhong);
    }

    return surface;
}

// colorat() 
//  - world.intersect()
//  - prepareComputations()
//  - shadeHit() -> lighting()
CUDA_HOST_DEVICE HitInfo colorAt(World* world, const Ray& ray) {
    auto surface = Color::black;

    HitInfo hitInfo;

    //Array<Intersection> totalIntersections;
    Intersection totalIntersections[4];
    auto count = 0;
    world->intersect(ray, totalIntersections, count);

    if (count == 0) {
        auto t = 0.5 * (ray.direction.y() + 1.0);
        auto missColor = (1.0 - t) * Color::White() + t * Color::LightCornflower();
        missColor = Color::background;
        hitInfo.surface = missColor;
        return hitInfo;
    }

    // Nearest intersection
    const auto& hit = nearestHit(totalIntersections, count);

    if (!hit.bShading) {
        surface = Color::white;
        hitInfo.surface = surface;
        return hitInfo;
    }

    hitInfo = prepareComputations(hit, ray, totalIntersections, count);

    hitInfo.surface = shadeHit(world, hitInfo, false, true);

    return hitInfo;
}

CUDA_HOST_DEVICE Tuple reflectedColor(World* world, HitInfo& inHitInfo) {
    if (inHitInfo.object->material->reflective < Math::epsilon) {
        return Color::black;
    }

    auto reflectedRay = Ray(inHitInfo.overPosition, inHitInfo.reflectVector);
    auto hitInfo = colorAt(world, reflectedRay);
    
    auto color = hitInfo.surface * inHitInfo.object->material->reflective;

    inHitInfo = hitInfo;

    return color;
}

CUDA_HOST_DEVICE Tuple refract(const Tuple& uv, const Tuple& n, Float etaiOverEtat) {
    auto costheta = min(-uv.dot(n), 1.0f);
    Tuple rOutPerp = etaiOverEtat * (uv + costheta * n);
    Tuple rOutParallel = -sqrt(std::fabs(1.0f - rOutPerp.magnitudeSqured())) * n;
    return rOutPerp + rOutParallel;
}

CUDA_HOST_DEVICE Tuple refractedColor(World* world, HitInfo& inHitInfo) {
    if (inHitInfo.object->material->transparency < Math::epsilon) {
        return  Color::black;
    }

    // Find the ratio of first index of refraction to the second.
    // (Yup, this is inverted from the definition of Snell's Law.)
    auto ratio = inHitInfo.n1 / inHitInfo.n2;

    // cos(??i) is the same as the dot product of the two vectors
    auto cos??i = inHitInfo.viewDirection.dot(inHitInfo.normal);

    auto sin??i = sqrt(1.0f - cos??i * cos??i);

    // Find sin(??t)^2 via trigonometric identity
    auto sin??t2 = ratio * ratio * (1 - cos??i * cos??i);

    if (ratio * sin??i > 1.0f) {
        auto angle = Math::degrees(std::asin(sin??i));

        if (angle > 41.5f) {
            //std::cout << angle << std::endl;
        }

        return Color::red;
        ratio = 1.0f;
        sin??t2 = ratio * ratio * (1 - cos??i * cos??i);
    }

    // Find cos(??t) via trigonometric identity
    auto cos??t = sqrt(1.0f - sin??t2);

    // Compute the direction of the refracted ray
    // For the first recursion, viewDirection is the "real" view direction
    // after this viewDirect == -ray.direction(ray is incident ray)
    auto direction = inHitInfo.normal * (ratio * cos??i - cos??t) - inHitInfo.viewDirection * ratio;

    direction = refract(-inHitInfo.viewDirection, inHitInfo.normal, ratio);

    // Create the refracted ray
    auto refractedRay = Ray(inHitInfo.underPosition, direction);

    // Find the color of the refracted ray, making sure to multiply
    // by the transparency value to account for any opacity
    auto hitInfo = colorAt(world, refractedRay);

    auto color = hitInfo.surface * inHitInfo.object->material->transparency;

    inHitInfo = hitInfo;

    return color;
}

CUDA_HOST_DEVICE Tuple shadeHit(World* world, const HitInfo& hitInfo,
    int32_t remaining, bool bHalfLambert, bool bBlinnPhong) {
    auto surface = Color::black;

    for (auto i = 0; i < world->ligthCount(); i++) {
        auto light = world->getLight(i);
        auto inShadow = isShadow(world, light, hitInfo.overPosition);
        surface += lighting(hitInfo.object->material, hitInfo.object, light, hitInfo, inShadow, bHalfLambert, bBlinnPhong);
    }

    auto material = hitInfo.object->material;

    auto reflected = Color::black;

    if (material->reflective > Math::epsilon) {
        reflected = reflectedColor(world, hitInfo, remaining);
    }

    auto refracted = Color::black;

    if (material->transparency > Math::epsilon) {
        refracted = refractedColor(world, hitInfo, remaining);
    }

    if (material->reflective > Math::epsilon && material->transparency > Math::epsilon) {
        auto reflectance = schlick(hitInfo);
        return surface + reflected * reflectance + refracted * (1.0f - reflectance);
    }
    else {
        return surface + reflected + refracted;
    }
}

// colorat() 
//  - world.intersect()
//  - prepareComputations()
//  - shadeHit() -> lighting()
//CUDA_HOST_DEVICE Tuple colorAt(World* world, const Ray& ray, int32_t remaining) {
//    auto surface = Color::black;
//
//    Array<Intersection> intersections;
//    world->intersect(ray, intersections);
//
//    if (intersections.size() == 0) {
//        auto t = 0.5 * (ray.direction.y() + 1.0);
//        auto missColor = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
//        missColor = Color::background;
//        return missColor;
//    }
//
//    // Nearest intersection
//    const auto& hit = nearestHit(intersections);
//
//    if (!hit.bShading) {
//        surface = Color::white;
//        return surface;
//    }
//
//    auto hitInfo = prepareComputations(hit, ray, intersections);
//
//    surface = shadeHit(world, hitInfo, remaining, false, true);
//
//    return surface;
//}

CUDA_HOST_DEVICE Tuple colorAt(World* world, const Ray& ray, int32_t remaining) {
    auto surface = Color::black;

    Intersection totalIntersections[6];
    auto size = 0;
    world->intersect(ray, totalIntersections, size);

    if (size == 0) {
        auto t = 0.5f * (ray.direction.y() + 1.0f);
        auto missColor = (1.0f - t) * Color::White() + t * Color::LightCornflower();
        missColor = Color::background;
        surface = missColor;
        return surface;
    }

    // Nearest intersection
    const auto& hit = nearestHit(totalIntersections, size);

    if (!hit.bShading) {
        surface = Color::white;
        surface = surface;
        return surface;
    }

    auto hitInfo = prepareComputations(hit, ray, totalIntersections, size);

    surface = shadeHit(world, hitInfo, remaining, false, true);

    return surface;
}

CUDA_HOST_DEVICE Tuple reflectedColor(World* world, const HitInfo& hitInfo, int32_t reflectionRemaining) {
    if (hitInfo.object->material->reflective == 0.0f || reflectionRemaining == 0) {
        return Color::black;
    }

    auto reflectedRay = Ray(hitInfo.overPosition, hitInfo.reflectVector);
    auto color = colorAt(world, reflectedRay, reflectionRemaining - 1);

    return color * hitInfo.object->material->reflective;
}

CUDA_HOST_DEVICE Tuple refractedColor(World* world, const HitInfo& hitInfo, int32_t refractionRemaining) {
    if (hitInfo.object->material->transparency == 0.0f || refractionRemaining == 0) {
        return  Color::black;
    }

    // Find the ratio of first index of refraction to the second.
    // (Yup, this is inverted from the definition of Snell's Law.)
    auto ratio = hitInfo.n1 / hitInfo.n2;

    if (ratio > 1.0f) {
        int a = 0;
    }

    // cos(??i) is the same as the dot product of the two vectors
    auto cos??i = hitInfo.viewDirection.dot(hitInfo.normal);

    auto sin??i = sqrt(1.0f - cos??i * cos??i);

    // Find sin(??t)^2 via trigonometric identity
    auto sin??t2 = ratio * ratio * (1 - cos??i * cos??i);

    if (ratio * sin??i > 1.0f) {
        auto angle = Math::degrees(std::asin(sin??i));

        if (angle > 41.5f) {
            //std::cout << angle << std::endl;
        }

        return Color::red;
        ratio = 1.0;
        sin??t2 = ratio * ratio * (1 - cos??i * cos??i);
    }

    //if (sin??t2 > 1.0) {
    //    return Color::black;
    //}

    // Find cos(??t) via trigonometric identity
    auto cos??t = sqrt(1.0f - sin??t2);

    // Compute the direction of the refracted ray
    // For the first recursion, viewDirection is the "real" view direction
    // after this viewDirect == -ray.direction(ray is incident ray)
    auto direction = hitInfo.normal * (ratio * cos??i - cos??t) - hitInfo.viewDirection * ratio;

    direction = refract(-hitInfo.viewDirection, hitInfo.normal, ratio);

    //if (refractionRemaining < 2) {
    //    direction = -hitInfo.viewDirection;
    //}

    // Create the refracted ray
    auto refractedRay = Ray(hitInfo.underPosition, direction);

    // Find the color of the refracted ray, making sure to multiply
    // by the transparency value to account for any opacity
    auto color = colorAt(world, refractedRay, refractionRemaining - 1) * hitInfo.object->material->transparency;

    return color;
}

CUDA_HOST_DEVICE Float schlick(const HitInfo& hitInfo) {
    // Find the cosine of the angle between the eye and normal vectors
    auto cos?? = hitInfo.viewDirection.dot(hitInfo.normal);

    // Total internal reflection can only occur if n1 > n2
    if (hitInfo.n1 > hitInfo.n2) {
        auto n = hitInfo.n1 / hitInfo.n2;
        auto sin??t2 = n * n * (1.0f - cos?? * cos??);
        if (sin??t2 > 1.0f) {
            return 1.0f;
        }

        // Compute cosine of ??t using trigonometric identity
        auto cos??t = sqrt(1.0f - sin??t2);

        // When n1 > n2, use cos(??t) instead
        cos?? = cos??t;
    }

    // Return anything but 1.0 here, so that the test will fail
    // Appropriately if something goes wrong.
    auto r0 = std::pow(((hitInfo.n1 - hitInfo.n2) / (hitInfo.n1 + hitInfo.n2)), 2.0f);
    return r0 + (1.0 - r0) * std::pow((1.0f - cos??), 5.0f);
}
