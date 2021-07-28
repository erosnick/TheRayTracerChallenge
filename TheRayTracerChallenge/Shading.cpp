#include "Shading.h"
#include <algorithm>

Tuple lighting(const Material& material, const Light& light, const Tuple& position, const Tuple& viewDirection, const Tuple& normal, bool bHalfLambert, bool bBlinnPhong) {
    auto surfaceColor = light.intensity * material.color;
    auto ambientColor = material.color * material.ambient;
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

Tuple lighting(const Material& material, const Light& light,
    const HitInfo& hitInfo, bool bHalfLambert, bool bBlinnPhong) {
    return lighting(material, light, hitInfo.position, hitInfo.viewDirection, hitInfo.normal, bHalfLambert, bBlinnPhong);
}