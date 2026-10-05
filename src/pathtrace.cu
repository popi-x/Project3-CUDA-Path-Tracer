#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define ANTIALIASING 0
#define STREAM_COMPACTION 1
#define SORT_MAT 0
#define RUSSIAN_ROULETTE 1
#define RUSSIAN_ROULETTE_DEPTH 3

//better monte-carlo samping
#define BETTER_SAMPLING 1
#define GRID_SIZE 4

#define DIRECT_LIGHTING 1
#define DIRECT_LIGHTING_MAX_WEIGHT 10.0f


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_lights = NULL;
static int num_lights = 0;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    std::vector<int> lightIndices;
    for (int i = 0; i < scene->geoms.size(); i++) {
        if (scene->materials[scene->geoms[i].materialid].emittance > 0.0f) {
            lightIndices.push_back(i);
        }
    }
    num_lights = lightIndices.size();
    cudaMalloc(&dev_lights, glm::max(num_lights, 1) * sizeof(int));
    if (num_lights > 0) {
        cudaMemcpy(dev_lights, lightIndices.data(), num_lights * sizeof(int), cudaMemcpyHostToDevice);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_lights);

    checkCUDAError("pathtraceFree");
}

__host__ __device__ glm::vec2 betterSample2D(
    int iter, int pixelIndex, unsigned int salt,
    thrust::default_random_engine& rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	const unsigned int n = GRID_SIZE * GRID_SIZE;
	unsigned int j = (unsigned int)iter % n;
	unsigned int block = (unsigned int)iter / n;
    
    unsigned int h = utilhash((unsigned int)pixelIndex ^ utilhash(block * 2654435761u + salt));
    unsigned int a = ((h >> 8) % n) | 1u;
    unsigned int b = h % n;
    unsigned int k = (a * j + b) % n;

    unsigned int cx = k % GRID_SIZE;
    unsigned int cy = k / GRID_SIZE;

    return glm::vec2((cx + u01(rng)) / (float)GRID_SIZE,
        (cy + u01(rng)) / (float)GRID_SIZE);

}




/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> uJitter(-0.5f, 0.5f);
		thrust::uniform_real_distribution<float> u01(0, 1);
        
        float jx, jy;
		jx = 0.0f;
		jy = 0.0f;

        if (ANTIALIASING == 1) {
            if (BETTER_SAMPLING == 1) 
            { 
                glm::vec2 s = betterSample2D(iter, index, 0x1234u, rng);
                jx = s.x - 0.5f;
                jy = s.y - 0.5f;
            }
            else {
                jx = uJitter(rng);
                jy = uJitter(rng);
            }
        }
        
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jy - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;

        if (cam.aperture <= 0.0f) return;

        //won't work if the camera doesn't have depth of field setup
		float ft = cam.focalDistance / dot(segment.ray.direction, cam.view);
		glm::vec3 fP = segment.ray.origin + ft * segment.ray.direction; //focal point

        float u1, u2;
        if (BETTER_SAMPLING) {
			glm::vec2 s = betterSample2D(iter, index, 0x5678u, rng);
			u1 = s.x;
			u2 = s.y;
        }
        else {
			u1 = u01(rng);
			u2 = u01(rng);
        }

		float angle = 2 * PI * u1;
		float r = cam.aperture * sqrt(u2);
		float dx = r * cos(angle);
		float dy = r * sin(angle);

        segment.ray.origin = cam.position + cam.right * dx + cam.up * dy;
		segment.ray.direction = glm::normalize(fP - segment.ray.origin);
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;
        bool hit_outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
				hit_outside = outside;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
			intersections[path_index].outside = hit_outside;
        }
    }
}

__host__ __device__ void sampleCubeLight(
    const Geom& light, thrust::default_random_engine& rng,
    glm::vec3& point, glm::vec3& normal, float& area)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 s = glm::abs(light.scale);
    float ax = s.y * s.z;
    float ay = s.x * s.z;
    float az = s.x * s.y;
    area = 2.0f * (ax + ay + az);

    float pick = u01(rng) * (ax + ay + az);
    int axis = (pick < ax) ? 0 : ((pick < ax + ay) ? 1 : 2);
    float side = (u01(rng) < 0.5f) ? -0.5f : 0.5f;

    glm::vec3 p(u01(rng) - 0.5f, u01(rng) - 0.5f, u01(rng) - 0.5f);
    p[axis] = side;
    glm::vec3 n(0.0f);
    n[axis] = (side > 0.0f) ? 1.0f : -1.0f;

    point = multiplyMV(light.transform, glm::vec4(p, 1.0f));
    normal = glm::normalize(multiplyMV(light.invTranspose, glm::vec4(n, 0.0f)));
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int* lights,
    int num_lights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& pathSeg = pathSegments[idx];

        if (pathSegments[idx].remainingBounces <= 0) return;
        
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSeg.remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSeg.remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                glm::vec3 intersectPt = getPointOnRay(pathSegments[idx].ray, intersection.t);

                scatterRay(pathSegments[idx], intersectPt, intersection.surfaceNormal, intersection.outside, material, rng);
                pathSegments[idx].remainingBounces--;
                
                if (pathSeg.remainingBounces == 0) pathSeg.color = glm::vec3(0.0f);
                else if (DIRECT_LIGHTING == 1 && pathSeg.remainingBounces == 1
                    && material.hasRefractive <= 0.0f && num_lights > 0)
                {
                    int li = glm::min((int)(u01(rng) * num_lights), num_lights - 1);
                    const Geom& light = geoms[lights[li]];

                    if (light.type == CUBE)
                    {
                        glm::vec3 lightPoint, lightNormal;
                        float lightArea;
                        sampleCubeLight(light, rng, lightPoint, lightNormal, lightArea);

                        glm::vec3 origin = intersectPt + 0.001f * intersection.surfaceNormal;
                        glm::vec3 toLight = lightPoint - origin;
                        float dist2 = glm::dot(toLight, toLight);
                        glm::vec3 wi = toLight / sqrtf(dist2);

                        float cosSurface = glm::dot(intersection.surfaceNormal, wi);
                        float cosLight = glm::dot(lightNormal, -wi);

                        if (cosSurface > 0.0f && cosLight > 0.0f)
                        {
                            pathSeg.ray.origin = origin;
                            pathSeg.ray.direction = wi;
                            float weight = cosSurface * cosLight * lightArea * (float)num_lights / (PI * dist2);
                            pathSeg.color *= glm::min(weight, DIRECT_LIGHTING_MAX_WEIGHT);
                        }
                        else
                        {
                            pathSeg.color = glm::vec3(0.0f);
                            pathSeg.remainingBounces = 0;
                        }
                    }
                }
				else if (RUSSIAN_ROULETTE == 1 && depth > RUSSIAN_ROULETTE_DEPTH) { //implement Russian Roulette
					float maxColor = glm::min(1.0f, glm::max(pathSeg.color.r, glm::max(pathSeg.color.g, pathSeg.color.b)));
                    
                    if (u01(rng) >= maxColor){
                        pathSeg.remainingBounces = 0;
                        pathSeg.color = glm::vec3(0.0f);
                    }
                    else {
                        pathSeg.color /= maxColor;
                    }
                }
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSeg.remainingBounces = 0;
        }
    }
}


// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

// helper struct

struct isPathAlive {
    __host__ __device__ bool operator()(const PathSegment& p) {
        return p.remainingBounces > 0;
    }
};

struct idSmallerThan {
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
        return a.materialId < b.materialId;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        /*shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );*/
		if (SORT_MAT == 1)
		    thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, idSmallerThan());

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            dev_lights,
            num_lights
        );
        
		if (STREAM_COMPACTION == 1)
            num_paths = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, isPathAlive()) - dev_paths;

        if (iter == 10) printf("depth %d: %d\n", depth, num_paths);
        
        iterationComplete = num_paths == 0 || depth >= traceDepth ? true : false; // TODO: should be based off stream compaction results.

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
