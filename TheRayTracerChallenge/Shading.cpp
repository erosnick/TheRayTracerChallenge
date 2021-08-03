#include "Shading.h"
#include "Tuple.h"
#include "Material.h"
#include "Light.h"
#include "World.h"
#include "Intersection.h"
#include "Pattern.h"
#include "Shape.h"

#include <algorithm>

Tuple lighting(const Material& material, const ShapePtr& object, const Light& light, const Tuple& position,
               const Tuple& viewDirection, const Tuple& normal, bool bInShadow, 
               bool bHalfLambert, bool bBlinnPhong) {
    auto materialColor = material.color;

    if (material.pattern.has_value()) {
        materialColor = material.pattern.value()->patternAtShape(object, position);
    }

    auto ambientColor = materialColor * material.ambient;
    
    if (bInShadow) {
        return ambientColor;
    }

    auto surfaceColor = light.intensity * materialColor;
    auto diffuseColor = surfaceColor * material.diffuse;
    auto specularColor = light.intensity * material.specular;
    
    auto lightDirection = (light.position - position);
    auto distance = lightDirection.magnitude();

    auto attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
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
            specular = std::pow(std::max(halfVector.dot(normal), 0.0), material.shininess * 2) * attenuation;
        }
        else {
            specular = std::pow(std::max(reflectVector.dot(viewDirection), 0.0), material.shininess) * attenuation;
        }
    }

    auto finalColor = ambientColor + diffuseColor * diffuse + specularColor * specular;

    return finalColor;
}

Tuple lighting(const Material& material, const ShapePtr& object, const Light& light,
               const HitInfo& hitInfo, bool bInShadow, 
               bool bHalfLambert, bool bBlinnPhong) {
    return lighting(material, object, light, hitInfo.position, hitInfo.viewDirection, hitInfo.normal, bInShadow, bHalfLambert, bBlinnPhong);
}

bool isShadow(const World& world, const Light& light, const Tuple& position) {
    auto toLight = light.position - position;
    const auto distance = toLight.magnitude();

    auto ray = Ray(position, toLight.normalize());
    auto intersections = world.intersect(ray);

    if (intersections.size() > 0) {
        const auto& intersection = intersections[0];

        if (!intersection.object->bIsLight && intersection.t < distance) {
            return true;
        }
    }

    return false;
}

Tuple shadeHit(const World& world, const HitInfo& hitInfo,
    int32_t remaining, bool bHalfLambert, bool bBlinnPhong) {
    auto surface = Color::black;

    for (const auto& light : world.getLights()) {
        //auto transformedLight = light;
        //transformedLight.transform(hitInfo.object.transform.inverse());
        auto inShadow = isShadow(world, light, hitInfo.overPosition);
        surface += lighting(hitInfo.object->material, hitInfo.object, light, hitInfo, inShadow, bHalfLambert, bBlinnPhong);
    }

    auto reflected = Color::black; // reflectedColor(world, hitInfo, remaining);

    auto refracted = refractedColor(world, hitInfo, remaining);

    return surface + reflected + refracted;
}

// colorat() 
//  - world.intersect()
//  - prepareComputations()
//  - shadeHit() -> lighting()
Tuple colorAt(const World& world, Ray& ray, int32_t remaining) {
    auto surface = Color::black;

    auto intersections = world.intersect(ray);

    if (intersections.size() == 0) {
        auto t = 0.5 * (ray.direction.y + 1.0);
        auto missColor = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
        return missColor;
    }

    // Nearest intersection
    const auto& hit = intersections[0];

    if (!hit.bShading) {
        surface = Color::white;
        return surface;
    }

    auto hitInfo = prepareComputations(hit, hit.ray);

    surface = shadeHit(world, hitInfo, remaining, false, true);

    return surface;
}

Tuple reflectedColor(const World& world, const HitInfo& hitInfo, int32_t remaining) {
    if (hitInfo.object->material.reflective == 0.0 || remaining == 0) {
        return Color::black;
    }

    auto reflectedRay = Ray(hitInfo.overPosition, hitInfo.reflectVector);
    auto color = colorAt(world, reflectedRay, remaining - 1);

    return color * hitInfo.object->material.reflective;
}

Tuple refractedColor(const World& world, const HitInfo& hitInfo, int32_t remaining) {
    if (hitInfo.object->material.transparency == 0.0 || remaining == 0) {
        return  Color::black;
    }

    // Find the ratio of first index of refraction to the second.
    // (Yup, this is inverted from the definition of Snell's Law.)
    auto ratio = hitInfo.n1 / hitInfo.n2;

    // cos(��i) is the same as the dot product of the two vectors
    auto cos��i = hitInfo.viewDirection.dot(hitInfo.normal);

    // Find sin(��t)^2 via trigonometric identity
    auto sin��t2 = ratio * ratio * (1 - cos��i * cos��i);

    if (sin��t2 > 1.0) {
        return Color::black;
    }

    // Find cos(��t) via trigonometric identity
    auto cos��t = std::sqrt(1.0 - sin��t2);

    // Compute the direction of the refracted ray
    // For the first recursion, viewDirection is the "real" view direction
    // after this viewDirect == -ray.direction(ray is incident ray)
    auto direction = hitInfo.normal * (ratio * cos��i - cos��t) - hitInfo.viewDirection * ratio;

    // Create the refracted ray
    auto refractedRay = Ray(hitInfo.underPosition, direction);

    // Find the color of the refracted ray, making sure to multiply
    // by the transparency value to account for any opacity
    auto color = colorAt(world, refractedRay, remaining - 1) * hitInfo.object->material.transparency;

    return color;
}
