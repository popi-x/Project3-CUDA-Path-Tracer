#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    
    if (m.hasRefractive > 0.0f) {
		thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec3 wi = pathSegment.ray.direction;


        float etaI = outside ? 1.0f : m.indexOfRefraction;
        float etaT = outside ? m.indexOfRefraction : 1.0f;
        float eta = etaI / etaT;

        float cosI = glm::clamp(glm::dot(-wi, normal), 0.0f, 1.0f);
        float sin2T = eta * eta * (1.0f - cosI * cosI);
        bool totalInternalReflection = sin2T > 1.0f;

        float fresnel = 1.0f;
        if (!totalInternalReflection)
        {
            float cosX = (etaI > etaT) ? sqrtf(1.0f - sin2T) : cosI;
            float r0 = (etaI - etaT) / (etaI + etaT);
            r0 = r0 * r0;
            fresnel = r0 + (1.0f - r0) * powf(1.0f - cosX, 5.0f);
        }

        if (u01(rng) < fresnel)
        {
            // Reflect
            pathSegment.ray.origin = intersect + 0.001f * normal;
            pathSegment.ray.direction = glm::reflect(wi, normal);
        }
        else
        {
            // Refract
            pathSegment.ray.origin = intersect - 0.001f * normal;
            pathSegment.ray.direction = glm::refract(wi, normal, eta);
		}

        pathSegment.color *= m.color;
        return;
    }


	glm::vec3 direction = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.ray.origin = intersect + 0.001f * normal;
    pathSegment.ray.direction = direction;
    pathSegment.color *= m.color;
    
}
