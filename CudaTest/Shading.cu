#include "Shading.h"
#include "Tuple.h"
#include "Material.h"
#include "Light.h"
#include "World.h"
#include <Intersection.h>
#include "Pattern.h"
#include "Shape.h"

#include <algorithm>

CUDA_DEVICE Tuple lighting(Material* material, Shape* object, Light* light, const Tuple& position,
               const Tuple& viewDirection, const Tuple& normal, bool bInShadow, 
               bool bHalfLambert, bool bBlinnPhong) {
    auto materialColor = material->color;

    if (material->pattern) {
        materialColor = material->pattern->patternAtShape(object, position);
    }

    auto ambientColor = materialColor * material->ambient;
    
    if (bInShadow) {
        return Tuple();
    }

    auto surfaceColor = light->intensity * materialColor;
    auto diffuseColor = surfaceColor * material->diffuse;
    auto specularColor = light->intensity * material->specular;
    
    auto lightDirection = (light->position - position);
    auto distance = lightDirection.magnitude();

    auto attenuation = 1.0;

    if (light->bAttenuation) {
        attenuation = 1.0 / (light->constant + light->linear * distance + light->quadratic * (distance * distance));
    }

    lightDirection = lightDirection / distance;

    auto diffuseTerm = normal.dot(lightDirection);
    auto diffuse = std::max(diffuseTerm, 0.0) * attenuation;

    if (bHalfLambert) {
        ambientColor = color(0.0, 0.0, 0.0);
        diffuse = diffuse * 0.5 + 0.5;
    }

    auto specular = 0.0;

    if (diffuseTerm > 0) {
        auto reflectVector = 2.0 * (diffuseTerm) * normal - lightDirection;
        if (bBlinnPhong) {
            auto halfVector = (lightDirection + viewDirection) / (lightDirection + viewDirection).magnitude();
            specular = std::pow(std::max(halfVector.dot(normal), 0.0), material->shininess * 2) * attenuation;
        }
        else {
            specular = std::pow(std::max(reflectVector.dot(viewDirection), 0.0), material->shininess) * attenuation;
        }
    }

    auto finalColor = ambientColor + diffuseColor * diffuse + specularColor * specular;
    return finalColor;
}

CUDA_DEVICE Tuple lighting(Material* material, Shape* object, Light* light,
                           const HitInfo& hitInfo, bool bInShadow, 
                           bool bHalfLambert, bool bBlinnPhong) {
    return lighting(material, object, light, hitInfo.overPosition, hitInfo.viewDirection, hitInfo.normal, bInShadow, bHalfLambert, bBlinnPhong);
}

CUDA_DEVICE bool isShadow(World* world, Light* light, const Tuple& position) {
    auto toLight = light->position - position;
    const auto distance = toLight.magnitude();

    auto ray = Ray(position, toLight.normalize());

    Intersection intersections[MAXELEMENTS];

    auto count = 0;
    world->intersect(ray, intersections, &count);

    if (count > 0) {
        const auto& intersection = nearestHit(intersections, count);

        if (intersection.bHit 
        && !intersection.object->bIsLight
        &&  intersection.object->material->bCastShadow
        &&  intersection.t < distance) {
            return true;
        }
    }

    return false;
}

CUDA_DEVICE Tuple computeReflectionAndRefraction(const HitInfo& hitInfo, World* world) {
    auto material = hitInfo.object->material;

    auto reflected = color(0.0f);

    if (material->reflective > Math::epsilon) {
        //reflected = reflectedColor(world, hitInfo);
    }

    auto refracted = color(0.0f);

    if (material->transparency > Math::epsilon) {
        //refracted = refractedColor(world, hitInfo);
    }

    if (material->reflective > Math::epsilon && material->transparency > Math::epsilon) {
        auto reflectance = schlick(hitInfo);
        return reflected * reflectance + refracted * (1.0 - reflectance);
    }
    else {
        return reflected + refracted;
    }
}

CUDA_DEVICE Tuple shadeHit(World* world, const HitInfo& hitInfo, bool bHalfLambert, bool bBlinnPhong) {
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
CUDA_DEVICE HitInfo colorAt(World* world, const Ray& ray) {
    auto surface = Color::black;

    HitInfo hitInfo;

    Intersection intersections[MAXELEMENTS];

    auto count = 0;
    world->intersect(ray, intersections, &count);

    if (count == 0) {
        auto t = 0.5 * (ray.direction.y() + 1.0);
        auto missColor = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
        missColor = Color::background;
        hitInfo.surface = missColor;
        return hitInfo;
    }

    // Nearest intersection
    const auto& hit = nearestHit(intersections, count);

    if (!hit.bShading) {
        surface = Color::white;
        hitInfo.surface = surface;
        return hitInfo;
    }

    hitInfo = prepareComputations(hit, hit.ray, intersections);

    hitInfo.surface = shadeHit(world, hitInfo, false, true);

    return hitInfo;
}

CUDA_DEVICE Tuple reflectedColor(World* world, const HitInfo& inHitInfo) {
    if (inHitInfo.object->material->reflective) {
        return Color::black;
    }

    auto reflectedRay = Ray(inHitInfo.overPosition, inHitInfo.reflectVector);
    auto hitInfo = colorAt(world, reflectedRay);

    return hitInfo.surface * inHitInfo.object->material->reflective;
}

CUDA_DEVICE Tuple refract(const Tuple& uv, const Tuple& n, double etaiOverEtat) {
    auto costheta = std::fmin(-uv.dot(n), 1.0);
    Tuple rOutPerp = etaiOverEtat * (uv + costheta * n);
    Tuple rOutParallel = -std::sqrt(std::fabs(1.0 - rOutPerp.magnitudeSqured())) * n;
    return rOutPerp + rOutParallel;
}

CUDA_DEVICE Tuple refractedColor(World* world, const HitInfo& inHitInfo) {
    if (inHitInfo.object->material->transparency == 0.0) {
        return  Color::black;
    }

    // Find the ratio of first index of refraction to the second.
    // (Yup, this is inverted from the definition of Snell's Law.)
    auto ratio = inHitInfo.n1 / inHitInfo.n2;

    // cos(¦Èi) is the same as the dot product of the two vectors
    auto cos¦Èi = inHitInfo.viewDirection.dot(inHitInfo.normal);

    auto sin¦Èi = std::sqrt(1.0 - cos¦Èi * cos¦Èi);

    // Find sin(¦¨t)^2 via trigonometric identity
    auto sin¦Èt2 = ratio * ratio * (1 - cos¦Èi * cos¦Èi);

    if (ratio * sin¦Èi > 1.0) {
        auto angle = Math::degrees(std::asin(sin¦Èi));

        if (angle > 41.5) {
            //std::cout << angle << std::endl;
        }

        return Color::red;
        ratio = 1.0;
        sin¦Èt2 = ratio * ratio * (1 - cos¦Èi * cos¦Èi);
    }

    //if (sin¦Èt2 > 1.0) {
    //    return Color::black;
    //}

    // Find cos(¦Èt) via trigonometric identity
    auto cos¦Èt = std::sqrt(1.0 - sin¦Èt2);

    // Compute the direction of the refracted ray
    // For the first recursion, viewDirection is the "real" view direction
    // after this viewDirect == -ray.direction(ray is incident ray)
    auto direction = inHitInfo.normal * (ratio * cos¦Èi - cos¦Èt) - inHitInfo.viewDirection * ratio;

    direction = refract(-inHitInfo.viewDirection, inHitInfo.normal, ratio);

    //if (refractionRemaining < 2) {
    //    direction = -hitInfo.viewDirection;
    //}

    // Create the refracted ray
    auto refractedRay = Ray(inHitInfo.underPosition, direction);

    // Find the color of the refracted ray, making sure to multiply
    // by the transparency value to account for any opacity
    auto hitInfo = colorAt(world, refractedRay);

    return hitInfo.surface * inHitInfo.object->material->transparency;;
}

CUDA_DEVICE double schlick(const HitInfo& hitInfo) {
    // Find the cosine of the angle between the eye and normal vectors
    auto cos¦È = hitInfo.viewDirection.dot(hitInfo.normal);

    // Total internal reflection can only occur if n1 > n2
    if (hitInfo.n1 > hitInfo.n2) {
        auto n = hitInfo.n1 / hitInfo.n2;
        auto sin¦Èt2 = n * n * (1.0 - cos¦È * cos¦È);
        if (sin¦Èt2 > 1.0) {
            return 1.0;
        }

        // Compute cosine of ¦Èt using trigonometric identity
        auto cos¦Èt = std::sqrt(1.0 - sin¦Èt2);

        // When n1 > n2, use cos(¦Èt) instead
        cos¦È = cos¦Èt;
    }

    // Return anything but 1.0 here, so that the test will fail
    // Appropriately if something goes wrong.
    auto r0 = std::pow(((hitInfo.n1 - hitInfo.n2) / (hitInfo.n1 + hitInfo.n2)), 2.0);
    return r0 + (1.0 - r0) * std::pow((1.0 - cos¦È), 5.0);
}
