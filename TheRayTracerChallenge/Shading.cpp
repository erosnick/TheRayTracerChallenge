#include "Shading.h"
#include <algorithm>

Tuple Lighting(const Material& material, const Tuple& position, const Light& light, const Tuple& eye, const Tuple& normal) {
    auto surfaceColor = light.intensity * material.color;
    auto ambientColor = surfaceColor * material.ambient;
    auto diffuseColor = surfaceColor * material.diffuse;
    auto specularColor = light.intensity * material.specular;
    
    auto lightDirection = (light.position -position);
    auto distance = lightDirection.magnitude();

    auto attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    lightDirection = lightDirection / distance;

    auto diffuseTerm = normal.dot(lightDirection);
    auto diffuse = std::max(diffuseTerm, 0.0) * attenuation;

    auto specular = 0.0;

    if (diffuseTerm > 0) {
        auto reflectVector = 2.0 * (diffuseTerm) * normal - lightDirection;
        auto viewDirection = (eye - position).normalize();
        specular = std::pow(std::max(lightDirection.dot(reflectVector), 0.0), 128.0) * attenuation;
    }

    auto finalColor = ambientColor + diffuseColor * diffuse + specularColor * specular;

    return finalColor;
}