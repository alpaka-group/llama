#include "../common/Stopwatch.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include <algorithm>
#include <array>
#include <boost/container/static_vector.hpp>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <llama/llama.hpp>
#include <numbers>
#include <numeric>
#include <random>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stdexcept>
#include <string>
#include <tiny_obj_loader.h>
#include <unordered_map>
#include <variant>

namespace
{
    template <typename F>
    struct Vector
    {
        inline auto operator[](std::size_t index) -> F&
        {
            return values[index];
        }

        inline auto operator[](std::size_t index) const -> F
        {
            return values[index];
        }

        inline auto operator-() const -> Vector
        {
            return {-values[0], -values[1], -values[2]};
        }

        inline auto operator+=(const Vector& v) -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] += v.values[i];
            return *this;
        }

        inline auto operator-=(const Vector& v) -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] -= v.values[i];
            return *this;
        }

        template <typename Scalar>
        inline auto operator*=(Scalar scalar) -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] *= scalar;
            return *this;
        }

        inline auto lengthSqr() const
        {
            F r = 0;
            for (int i = 0; i < 3; i++)
                r += values[i] * values[i];
            return r;
        }

        inline auto length() const
        {
            return std::sqrt(lengthSqr());
        }

        inline void normalize()
        {
            const auto l = length();
            for (int i = 0; i < 3; i++)
                values[i] /= l;
        }

        inline auto normalized() const -> Vector
        {
            auto r = *this;
            r.normalize();
            return r;
        }

        friend inline auto operator+(const Vector& a, const Vector& b) -> Vector
        {
            auto r = a;
            r += b;
            return r;
        }

        friend inline auto operator-(const Vector& a, const Vector& b) -> Vector
        {
            auto r = a;
            r -= b;
            return r;
        }

        friend inline auto operator*(const Vector& v, F scalar) -> Vector
        {
            auto r = v;
            r *= scalar;
            return r;
        }

        friend inline auto operator*(F scalar, const Vector& v) -> Vector
        {
            return v * scalar;
        }

        friend inline auto operator/(F scalar, const Vector& v) -> Vector
        {
            return {scalar / v[0], scalar / v[1], scalar / v[2]};
        }

        friend inline auto operator>>(std::istream& is, Vector& v) -> std::istream&
        {
            for (int i = 0; i < 3; i++)
                is >> v[i];
            return is;
        }

        friend inline auto operator<<(std::ostream& os, const Vector& v) -> std::ostream&
        {
            for (int i = 0; i < 3; i++)
                os << v[i] << " ";
            return os;
        }

        std::array<F, 3> values = {{0, 0, 0}};
    };

    template <typename F>
    inline auto dot(const Vector<F>& a, const Vector<F>& b) -> F
    {
        F r = 0;
        for (int i = 0; i < 3; i++)
            r += a[i] * b[i];
        return r;
    }

    template <typename F>
    inline auto cross(const Vector<F>& a, const Vector<F>& b) -> Vector<F>
    {
        Vector<F> r;
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
        return r;
    }

    using VectorF = Vector<float>;

    inline auto solveQuadraticEquation(double a, double b, double c) -> std::vector<double>
    {
        const double discriminat = b * b - 4 * a * c;
        if (discriminat < 0)
            return {};

        if (discriminat == 0)
            return {-b / 2 * a};

        const auto x1 = (-b - std::sqrt(discriminat)) / 2 * a;
        const auto x2 = (-b + std::sqrt(discriminat)) / 2 * a;
        return {x1, x2};
    }

    struct Camera
    {
        float fovy; // in degree
        VectorF position;
        VectorF view;
        VectorF up;
    };

    struct Sphere
    {
        VectorF center;
        float radius;
    };

    struct Vertex
    {
        VectorF pos;
        float u;
        float v;
    };

    struct Triangle : std::array<Vertex, 3>
    {
        int texIndex;
    };

    struct AABB
    {
        VectorF lower;
        VectorF upper;

        inline auto center() const -> VectorF
        {
            return (lower + upper) * 0.5f;
        }

        inline auto contains(VectorF v) const -> bool
        {
            return lower[0] <= v[0] && v[0] <= upper[0] && lower[1] <= v[1] && v[1] <= upper[1] && lower[2] <= v[2]
                && v[2] <= upper[2];
        }
    };

    struct PreparedTriangle
    {
        VectorF vertex0;
        VectorF edge1;
        VectorF edge2;
        std::array<float, 3> u;
        std::array<float, 3> v;
        int texIndex;

        inline auto normal() const -> VectorF
        {
            return cross(edge1, edge2).normalized();
        }
    };

    inline auto prepare(Triangle t) -> PreparedTriangle
    {
        return {
            t[0].pos,
            t[1].pos - t[0].pos,
            t[2].pos - t[0].pos,
            {t[0].u, t[1].u, t[2].u},
            {t[0].v, t[1].v, t[2].v},
            t.texIndex};
    }

    class Image
    {
    public:
        using Pixel = Vector<unsigned char>;

        Image(const std::filesystem::path& filename)
        {
            int x = 0;
            int y = 0;
            int comp = 0;
            unsigned char* data = stbi_load(filename.string().c_str(), &x, &y, &comp, 3);
            stbi__vertical_flip(data, x, y, 3);
            if (data == nullptr)
                throw std::runtime_error("Failed to read image " + filename.string());
            // if (comp != 3)
            //    throw std::runtime_error(
            //        "Image " + filename.string() + " does not have 3 channels but " + std::to_string(comp));
            w = x;
            h = y;
            pixels.resize(w * h);
            std::memcpy(pixels.data(), data, w * h * 3);
            stbi_image_free(data);
        }

        Image(unsigned int width, unsigned int height) : w(width), h(height), pixels(width * height)
        {
        }

        inline auto width() const
        {
            return w;
        }

        inline auto height() const
        {
            return h;
        }

        inline auto operator()(unsigned int x, unsigned int y) -> Pixel&
        {
            return pixels[y * w + x];
        }

        inline auto operator()(unsigned int x, unsigned int y) const -> const Pixel&
        {
            return pixels[y * w + x];
        }

        void write(const std::filesystem::path& filename) const
        {
            if (!stbi_write_png(filename.string().c_str(), w, h, 3, pixels.data(), 0))
                throw std::runtime_error("Failed to write image " + filename.string());
        }

    private:
        unsigned int w;
        unsigned int h;
        std::vector<Pixel> pixels;
    };

    constexpr auto noHit = std::numeric_limits<float>::infinity();

    struct Intersection
    {
        float distance = noHit;
        VectorF normal;
        float texU = 0;
        float texV = 0;
        int texIndex = -1;
    };

    struct Ray
    {
        VectorF origin;
        VectorF direction;
    };

    inline auto createRay(const Camera& camera, unsigned int width, unsigned int height, unsigned int x, unsigned int y)
        -> Ray
    {
        // we imagine a plane with the image just 1 before the camera, and then we shoot at those pixels

        const auto center = camera.position + camera.view;
        const auto xVec = cross(camera.view, camera.up);
        const auto yVec = camera.up;

        const auto delta = (std::tan(camera.fovy * std::numbers::pi_v<float> / 180.0f) * 2) / (height - 1);
        const auto xDeltaVec = xVec * delta;
        const auto yDeltaVec = yVec * delta;

        const auto xRel = (x - static_cast<float>(width - 1) / 2);
        const auto yRel = (y - static_cast<float>(height - 1) / 2);

        const auto pixel = center + xDeltaVec * xRel + yDeltaVec * yRel;

        Ray r;
        r.origin = center;
        r.direction = (pixel - camera.position).normalized();

        assert(!std::isnan(r.direction[0]) && !std::isnan(r.direction[1]) && !std::isnan(r.direction[2]));
        // std::cout << r.direction << std::endl;

        return r;
    }

    // from:
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    inline auto intersectBox(const Ray& r, const AABB& box) -> std::pair<float, float>
    {
        const auto invdir = 1.0f / r.direction;
        const VectorF bounds[] = {box.lower, box.upper};
        const int sign[] = {r.direction[0] < 0, r.direction[1] < 0, r.direction[2] < 0};

        float tmin = (bounds[sign[0]][0] - r.origin[0]) * invdir[0];
        float tmax = (bounds[1 - sign[0]][0] - r.origin[0]) * invdir[0];
        const float tymin = (bounds[sign[1]][1] - r.origin[1]) * invdir[1];
        const float tymax = (bounds[1 - sign[1]][1] - r.origin[1]) * invdir[1];
        if ((tmin > tymax) || (tymin > tmax))
            return {noHit, noHit};
        if (tymin > tmin)
            tmin = tymin;
        if (tymax < tmax)
            tmax = tymax;

        const float tzmin = (bounds[sign[2]][2] - r.origin[2]) * invdir[2];
        const float tzmax = (bounds[1 - sign[2]][2] - r.origin[2]) * invdir[2];
        if ((tmin > tzmax) || (tzmin > tmax))
            return {noHit, noHit};
        if (tzmin > tmin)
            tmin = tzmin;
        if (tzmax < tmax)
            tmax = tzmax;

        return {tmin, tmax};
    }

    inline auto intersect(const Ray& ray, const Sphere& sphere) -> Intersection
    {
        // from
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection

        // solve quadratic equation
        const auto a = 1;
        const auto b = 2 * dot(ray.direction, (ray.origin - sphere.center));
        const auto c = (ray.origin - sphere.center).lengthSqr() - sphere.radius * sphere.radius;

        const auto solutions = solveQuadraticEquation(a, b, c);
        if (solutions.empty())
            return {};

        // report the closer intersection
        const auto t = static_cast<float>(*std::min_element(std::begin(solutions), std::end(solutions)));
        const auto point = ray.origin + t * ray.direction;
        return {t, (point - sphere.center).normalized()};
    }

    // modified Möller and Trumbore's version
    inline auto intersect(const Ray& ray, const PreparedTriangle& triangle) -> Intersection
    {
        constexpr auto epsilon = 0.000001f;

        const auto pvec = cross(ray.direction, triangle.edge2);
        const auto det = dot(triangle.edge1, pvec);
        if (det > -epsilon && det < epsilon)
            return {};

        const auto inv_det = 1.0f / det;
        const auto tvec = ray.origin - triangle.vertex0;
        const auto u = dot(tvec, pvec) * inv_det;
        if (u < 0.0f || u > 1.0f)
            return {};

        const auto qvec = cross(tvec, triangle.edge1);
        const auto v = dot(ray.direction, qvec) * inv_det;
        if (v < 0.0f || u + v >= 1.0f)
            return {};
        const auto t = dot(triangle.edge2, qvec) * inv_det;
        if (t < 0)
            return {};

        const auto texU = (1 - u - v) * triangle.u[0] + u * triangle.u[1] + v * triangle.u[2];
        const auto texV = (1 - u - v) * triangle.v[0] + u * triangle.v[1] + v * triangle.v[2];
        return {t, triangle.normal(), texU, texV, triangle.texIndex};
    }

    // from: https://stackoverflow.com/questions/4578967/cube-sphere-intersection-test
    inline auto overlaps(const Sphere& s, const AABB& box) -> bool
    {
        auto sqr = [](auto x) { return x * x; };

        float r2 = s.radius * s.radius;
        float dmin = 0.0f;
        for (auto i : {0, 1, 2})
        {
            if (s.center[i] < box.lower[i])
                dmin += sqr(s.center[i] - box.lower[i]);
            else if (s.center[i] > box.upper[i])
                dmin += sqr(s.center[i] - box.upper[i]);
        }
        return dmin <= r2;
    }

    // from https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt
    auto overlaps(const PreparedTriangle& t, const AABB& box) -> bool
    {
#define FINDMINMAX(x0, x1, x2, min, max)                                                                               \
    min = max = x0;                                                                                                    \
    if (x1 < min)                                                                                                      \
        min = x1;                                                                                                      \
    if (x1 > max)                                                                                                      \
        max = x1;                                                                                                      \
    if (x2 < min)                                                                                                      \
        min = x2;                                                                                                      \
    if (x2 > max)                                                                                                      \
        max = x2;

        auto planeBoxOverlap = [](VectorF normal, VectorF vert, VectorF maxbox) -> bool
        {
            VectorF vmin, vmax;
            for (int q : {0, 1, 2})
            {
                float v = vert[q];
                if (normal[q] > 0.0f)
                {
                    vmin[q] = -maxbox[q] - v;
                    vmax[q] = maxbox[q] - v;
                }
                else
                {
                    vmin[q] = maxbox[q] - v;
                    vmax[q] = -maxbox[q] - v;
                }
            }

            if (dot(normal, vmin) > 0.0f)
                return false;
            if (dot(normal, vmax) >= 0.0f)
                return true;
            return false;
        };

#define AXISTEST_X01(a, b, fa, fb)                                                                                     \
    p0 = a * v0[1] - b * v0[2];                                                                                        \
    p2 = a * v2[1] - b * v2[2];                                                                                        \
    if (p0 < p2)                                                                                                       \
    {                                                                                                                  \
        min = p0;                                                                                                      \
        max = p2;                                                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        min = p2;                                                                                                      \
        max = p0;                                                                                                      \
    }                                                                                                                  \
    rad = fa * boxhalfsize[1] + fb * boxhalfsize[2];                                                                   \
    if (min > rad || max < -rad)                                                                                       \
        return false;

#define AXISTEST_X2(a, b, fa, fb)                                                                                      \
    p0 = a * v0[1] - b * v0[2];                                                                                        \
    p1 = a * v1[1] - b * v1[2];                                                                                        \
    if (p0 < p1)                                                                                                       \
    {                                                                                                                  \
        min = p0;                                                                                                      \
        max = p1;                                                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        min = p1;                                                                                                      \
        max = p0;                                                                                                      \
    }                                                                                                                  \
    rad = fa * boxhalfsize[1] + fb * boxhalfsize[2];                                                                   \
    if (min > rad || max < -rad)                                                                                       \
        return false;

#define AXISTEST_Y02(a, b, fa, fb)                                                                                     \
    p0 = -a * v0[0] + b * v0[2];                                                                                       \
    p2 = -a * v2[0] + b * v2[2];                                                                                       \
    if (p0 < p2)                                                                                                       \
    {                                                                                                                  \
        min = p0;                                                                                                      \
        max = p2;                                                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        min = p2;                                                                                                      \
        max = p0;                                                                                                      \
    }                                                                                                                  \
    rad = fa * boxhalfsize[0] + fb * boxhalfsize[2];                                                                   \
    if (min > rad || max < -rad)                                                                                       \
        return false;

#define AXISTEST_Y1(a, b, fa, fb)                                                                                      \
    p0 = -a * v0[0] + b * v0[2];                                                                                       \
    p1 = -a * v1[0] + b * v1[2];                                                                                       \
    if (p0 < p1)                                                                                                       \
    {                                                                                                                  \
        min = p0;                                                                                                      \
        max = p1;                                                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        min = p1;                                                                                                      \
        max = p0;                                                                                                      \
    }                                                                                                                  \
    rad = fa * boxhalfsize[0] + fb * boxhalfsize[2];                                                                   \
    if (min > rad || max < -rad)                                                                                       \
        return false;

#define AXISTEST_Z12(a, b, fa, fb)                                                                                     \
    p1 = a * v1[0] - b * v1[1];                                                                                        \
    p2 = a * v2[0] - b * v2[1];                                                                                        \
    if (p2 < p1)                                                                                                       \
    {                                                                                                                  \
        min = p2;                                                                                                      \
        max = p1;                                                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        min = p1;                                                                                                      \
        max = p2;                                                                                                      \
    }                                                                                                                  \
    rad = fa * boxhalfsize[0] + fb * boxhalfsize[1];                                                                   \
    if (min > rad || max < -rad)                                                                                       \
        return false;

#define AXISTEST_Z0(a, b, fa, fb)                                                                                      \
    p0 = a * v0[0] - b * v0[1];                                                                                        \
    p1 = a * v1[0] - b * v1[1];                                                                                        \
    if (p0 < p1)                                                                                                       \
    {                                                                                                                  \
        min = p0;                                                                                                      \
        max = p1;                                                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        min = p1;                                                                                                      \
        max = p0;                                                                                                      \
    }                                                                                                                  \
    rad = fa * boxhalfsize[0] + fb * boxhalfsize[1];                                                                   \
    if (min > rad || max < -rad)                                                                                       \
        return false;

        const auto boxcenter = box.center();
        const auto boxhalfsize = (box.upper - box.lower) * 0.5;
        float min, max, p0, p1, p2, rad;

        // const auto v0 = t[0] - boxcenter;
        // const auto v1 = t[1] - boxcenter;
        // const auto v2 = t[2] - boxcenter;
        const auto v0 = t.vertex0 - boxcenter;
        const auto v1 = t.vertex0 + t.edge1 - boxcenter;
        const auto v2 = t.vertex0 + t.edge2 - boxcenter;

        const auto e0 = v1 - v0;
        const auto e1 = v2 - v1;
        const auto e2 = v0 - v2;

        float fex = fabsf(e0[0]);
        float fey = fabsf(e0[1]);
        float fez = fabsf(e0[2]);
        AXISTEST_X01(e0[2], e0[1], fez, fey);
        AXISTEST_Y02(e0[2], e0[0], fez, fex);
        AXISTEST_Z12(e0[1], e0[0], fey, fex);

        fex = fabsf(e1[0]);
        fey = fabsf(e1[1]);
        fez = fabsf(e1[2]);
        AXISTEST_X01(e1[2], e1[1], fez, fey);
        AXISTEST_Y02(e1[2], e1[0], fez, fex);
        AXISTEST_Z0(e1[1], e1[0], fey, fex);

        fex = fabsf(e2[0]);
        fey = fabsf(e2[1]);
        fez = fabsf(e2[2]);
        AXISTEST_X2(e2[2], e2[1], fez, fey);
        AXISTEST_Y1(e2[2], e2[0], fez, fex);
        AXISTEST_Z12(e2[1], e2[0], fey, fex);

        FINDMINMAX(v0[0], v1[0], v2[0], min, max);
        if (min > boxhalfsize[0] || max < -boxhalfsize[0])
            return false;

        FINDMINMAX(v0[1], v1[1], v2[1], min, max);
        if (min > boxhalfsize[1] || max < -boxhalfsize[1])
            return false;

        FINDMINMAX(v0[2], v1[2], v2[2], min, max);
        if (min > boxhalfsize[2] || max < -boxhalfsize[2])
            return false;

        const auto normal = cross(e0, e1);
        if (!planeBoxOverlap(normal, v0, boxhalfsize))
            return false;

        return true;
    }

    struct OctreeNode
    {
        AABB box{};

        struct Objects
        {
            std::vector<PreparedTriangle> triangles;
            std::vector<Sphere> spheres;
        };
        using Children = std::array<OctreeNode*, 8>;
        std::variant<Objects, Children> content;

        inline auto hasChildren() const -> bool
        {
            return std::holds_alternative<Children>(content);
        }

        inline auto objects() -> Objects&
        {
            return std::get<Objects>(content);
        }

        inline auto objects() const -> const Objects&
        {
            return std::get<Objects>(content);
        }

        inline auto children() -> Children&
        {
            return std::get<Children>(content);
        }

        inline auto children() const -> const Children&
        {
            return std::get<Children>(content);
        }

        template <typename T>
        void addObject(std::deque<OctreeNode>& pool, const T& object, int depth = 0)
        {
            if (hasChildren())
            {
                for (auto& c : children())
                    if (overlaps(object, c->box))
                        c->addObject(pool, object, depth + 1);
            }
            else
            {
                if (shouldSplit(depth))
                {
                    split(pool, depth);
                    addObject(pool, object, depth);
                }
                else
                {
                    if constexpr (std::is_same_v<T, PreparedTriangle>)
                        objects().triangles.push_back(object);
                    else
                        objects().spheres.push_back(object);
                }
            }
        }

    private:
        static constexpr auto maxTrianglesPerNode = 32;
        static constexpr auto maxDepth = 16;

        inline auto shouldSplit(int depth) const -> bool
        {
            auto& objects = std::get<Objects>(content);
            return depth < maxDepth && objects.triangles.size() >= maxTrianglesPerNode;
        }

        inline void split(std::deque<OctreeNode>& pool, int depth)
        {
            auto objects = std::move(std::get<Objects>(content));
            auto& children = content.emplace<Children>();
            const VectorF points[] = {box.lower, box.center(), box.upper};
            for (auto x : {0, 1})
                for (auto y : {0, 1})
                    for (auto z : {0, 1})
                    {
                        const auto childBox = AABB{
                            {
                                points[x][0],
                                points[y][1],
                                points[z][2],
                            },
                            {
                                points[x + 1][0],
                                points[y + 1][1],
                                points[z + 1][2],
                            }};
                        children[z * 4 + y * 2 + x] = &pool.emplace_back(OctreeNode{childBox});
                    }

            for (const auto& s : objects.spheres)
                addObject(pool, s, depth);
            for (const auto& t : objects.triangles)
                addObject(pool, t, depth);
        }
    };

    // from: https://github.com/rumpfc/CGG/blob/master/cgg07_Octrees/OctreeNode.cpp
    void intersectNodeRecursive(const Ray& ray, const OctreeNode& node, Intersection& nearestHit)
    {
        if (node.hasChildren())
        {
            const auto& children = node.children();

            // iterate on children nearer than our current intersection in the order they are hit by the ray
            boost::container::static_vector<std::pair<int, float>, 8> childDists;
            for (int i = 0; i < 8; i++)
                if (const auto [tmin, tmax] = intersectBox(ray, children[i]->box);
                    tmin != noHit && tmax > 0 && tmin < nearestHit.distance)
                    childDists.emplace_back(i, tmin);
            std::sort(childDists.begin(), childDists.end(), [](auto a, auto b) { return a.second < b.second; });

            for (const auto [childIndex, childDist] : childDists)
                intersectNodeRecursive(ray, *children[childIndex], nearestHit);
        }
        else
        {
            const auto& objects = node.objects();
            for (const auto& sphere : objects.spheres)
                if (const auto hit = intersect(ray, sphere);
                    hit.distance != noHit && hit.distance < nearestHit.distance)
                    nearestHit = hit;
            for (const auto& triangle : objects.triangles)
                if (const auto hit = intersect(ray, triangle);
                    hit.distance != noHit && hit.distance < nearestHit.distance)
                    nearestHit = hit;
        }
    }

    inline auto intersect(const Ray& ray, const OctreeNode& tree) -> Intersection
    {
        Intersection nearestHit{};
        if (intersectBox(ray, tree.box).first != noHit)
            intersectNodeRecursive(ray, tree, nearestHit);
        return nearestHit;
    }

    inline auto colorByIntersectionNormal(Intersection hit) -> Image::Pixel
    {
        if (hit.distance == noHit)
            return {}; // black
        Image::Pixel r;
        for (int i = 0; i < 3; i++)
            r[i] = static_cast<unsigned char>(std::abs(hit.normal[i]) * 255);
        return r;
    }

    auto tex2D(const Image& tex, float u, float v) -> Image::Pixel
    {
        // texture coordinate behavior is repeat
        auto texCoordToTexelCoord = [](float coord, unsigned int imgSize)
        {
            const auto maxIndex = static_cast<float>(imgSize - 1);
            auto normalizedCoord = coord - static_cast<int>(coord);
            if (normalizedCoord < 0)
                normalizedCoord++;
            return std::clamp(normalizedCoord * maxIndex, 0.0f, maxIndex);
        };

        auto toVec = [](Image::Pixel p) { return VectorF{p[0] / 255.0f, p[1] / 255.0f, p[2] / 255.0f}; };
        const float x = texCoordToTexelCoord(u, tex.width());
        const float y = texCoordToTexelCoord(v, tex.height());

        //// nearest texel
        // return tex(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));

        // bilinear
        const float x0 = std::floor(x);
        const float x1 = std::ceil(x);
        const float xFrac = x - static_cast<int>(x);
        const float y0 = std::floor(y);
        const float y1 = std::ceil(y);
        const float yFrac = y - static_cast<int>(y);
        const auto color = (toVec(tex(x0, y0)) * (1 - xFrac) + toVec(tex(x1, y0)) * xFrac) * (1 - yFrac)
            + (toVec(tex(x0, y1)) * (1 - xFrac) + toVec(tex(x1, y1)) * xFrac) * yFrac;
        Image::Pixel rgb;
        for (int i = 0; i < 3; i++)
            rgb[i] = static_cast<unsigned char>(color[i] * 255);
        return rgb;
    }

    inline auto colorByTexture(const std::vector<Image>& textures, Intersection hit) -> Image::Pixel
    {
        if (hit.distance == noHit)
            return {}; // black
        Image::Pixel r;
        if (hit.texIndex != -1)
        {
            const auto& tex = textures[hit.texIndex];
            return tex2D(tex, hit.texU, hit.texV);
        }
        else
            for (int i = 0; i < 3; i++)
                r[i] = static_cast<unsigned char>(std::abs(hit.normal[i]) * 255);
        return r;
    }

    struct Scene
    {
        Camera camera;
        OctreeNode tree;
        std::deque<OctreeNode> nodePool;
        std::vector<Image> textures;
    };

    auto raycast(const Scene& scene, unsigned int width, unsigned int height) -> Image
    {
        Image img(width, height);

        for (auto y = 0u; y < height; y++)
        {
            for (auto x = 0u; x < width; x++)
            {
                const auto ray = createRay(scene.camera, width, height, x, height - 1 - y); // flip
                const auto nearestHit = intersect(ray, scene.tree);
                img(x, y) = colorByTexture(scene.textures, nearestHit);
            }
        }

        return img;
    }

    inline auto lookAt(float fovy, VectorF pos, VectorF lookAt, VectorF up) -> Camera
    {
        const auto view = (lookAt - pos).normalized();
        const auto right = cross(view, up);
        const auto up2 = cross(right, view).normalized();
        return Camera{fovy, pos, view, up2};
    }

    // auto cubicBallsScene() -> Scene
    //{
    //    const auto camera = lookAt(45, {5, 5.5, 6}, {0, 0, 0}, {0, 1, 0});
    //    auto spheres = std::vector<Sphere>{};
    //    for (auto z = -2; z <= 2; z++)
    //        for (auto y = -2; y <= 2; y++)
    //            for (auto x = -2; x <= 2; x++)
    //                spheres.push_back(Sphere{{(float) x, (float) y, (float) z}, 0.8f});
    //    return Scene{camera, std::move(spheres)};
    //}

    // auto axisBallsScene() -> Scene
    //{
    //    const auto camera = lookAt(45, {5, 5, 10}, {0, 0, 0}, {0, 1, 0});
    //    auto spheres = std::vector<Sphere>{
    //        {{0, 0, 0}, 3.0f},
    //        {{0, 0, 5}, 2.0f},
    //        {{0, 5, 0}, 2.0f},
    //        {{5, 0, 0}, 2.0f},
    //        {{0, 0, -5}, 1.0f},
    //        {{0, -5, 0}, 1.0f},
    //        {{-5, 0, 0}, 1.0f}};
    //    return Scene{camera, std::move(spheres)};
    //}

    // auto randomSphereScene() -> Scene
    //{
    //    constexpr auto count = 1024;

    //    const auto camera = lookAt(45, {5, 5.5, 6}, {0, 0, 0}, {0, 1, 0});
    //    auto spheres = std::vector<Sphere>{};

    //    std::default_random_engine eng;
    //    std::uniform_real_distribution d{-2.0f, 2.0f};
    //    for (auto i = 0; i < count; i++)
    //        spheres.push_back({{d(eng), d(eng), d(eng)}, 0.2f});
    //    return Scene{camera, std::move(spheres)};
    //}

    //// not the original one, but a poor attempt
    // auto cornellBox() -> Scene
    //{
    //    Scene scene;
    //    scene.camera = lookAt(45, {0, 0, 7}, {0, 0, 0}, {0, 1, 0});
    //    scene.spheres.push_back({{-2.5f, -2.5f, -2.5f}, 1.5f});
    //    scene.spheres.push_back({{2.5f, -2.5f, 0.0f}, 1.5f});
    //    // back plane
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, -5}, {5, -5, -5}, {5, 5, -5}}}}));
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, -5}, {5, 5, -5}, {-5, 5, -5}}}}));
    //    // left plane
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, 5}, {-5, -5, -5}, {-5, 5, -5}}}}));
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, 5}, {-5, 5, -5}, {-5, 5, 5}}}}));
    //    // right plane
    //    scene.triangles.push_back(prepare(Triangle{{{{5, -5, 5}, {5, 5, -5}, {5, -5, -5}}}}));
    //    scene.triangles.push_back(prepare(Triangle{{{{5, -5, 5}, {5, 5, 5}, {5, 5, -5}}}}));
    //    // bottom plane
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, 5}, {-5, -5, -5}, {5, -5, -5}}}}));
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, 5}, {5, -5, -5}, {5, -5, 5}}}}));
    //    // top plane
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, 5, 5}, {5, 5, -5}, {-5, 5, -5}}}}));
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, 5, 5}, {5, 5, 5}, {5, 5, -5}}}}));

    //    return scene;
    //}

    auto sponzaScene(const std::filesystem::path objFile) -> Scene
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;

        const bool ret = tinyobj::LoadObj(
            &attrib,
            &shapes,
            &materials,
            &warn,
            &err,
            objFile.string().c_str(),
            objFile.parent_path().string().c_str());
        if (!warn.empty())
            std::cout << warn << std::endl;
        if (!err.empty())
            std::cerr << err << std::endl;
        if (!ret)
            throw std::runtime_error{"Failed to load sponza scene"};

        Scene scene;
        scene.camera = lookAt(45, {200, 100, 0}, {0, 100, 0}, {0, 1, 0});

        std::unordered_map<std::string, int> textureToIndex;

        std::vector<Triangle> triangles;
        constexpr auto l = std::numeric_limits<float>::max();
        constexpr auto u = std::numeric_limits<float>::lowest();
        AABB box{{l, l, l}, {u, u, u}};

        // size_t write = 0;
        for (const auto& shape : shapes)
        {
            const auto& mesh = shape.mesh;

            size_t indexOffset = 0;
            for (auto f = 0; f < mesh.num_face_vertices.size(); f++)
            {
                const auto texIndex = [&]
                {
                    const auto matId = mesh.material_ids[f];
                    if (matId == -1)
                        return -1;
                    const auto texName = materials.at(matId).diffuse_texname;
                    if (texName == "")
                        return -1;
                    if (const auto it = textureToIndex.find(texName); it != end(textureToIndex))
                        return it->second;
                    const auto texIndex = static_cast<int>(scene.textures.size());
                    textureToIndex[texName] = texIndex;
                    scene.textures.push_back(Image{objFile.parent_path() / texName});
                    return texIndex;
                }();

                const auto vertexCount = mesh.num_face_vertices[f];
                if (vertexCount == 3)
                {
                    Triangle t;
                    for (const auto v : {0, 1, 2})
                    {
                        const tinyobj::index_t idx = mesh.indices[indexOffset + v];
                        for (const auto c : {0, 1, 2})
                        {
                            t[v].pos[c] = attrib.vertices[3 * idx.vertex_index + c];
                            box.lower[c] = std::min(box.lower[c], t[v].pos[c]);
                            box.upper[c] = std::max(box.upper[c], t[v].pos[c]);
                        }
                        t[v].u = attrib.texcoords[2 * idx.texcoord_index + 0];
                        t[v].v = attrib.texcoords[2 * idx.texcoord_index + 1];
                    }
                    t.texIndex = texIndex;
                    triangles.push_back(t);
                    // const auto pt = prepare(t);
                    // scene.triangles[write](Vertex0{}, X{}) = pt.vertex0[0];
                    // scene.triangles[write](Vertex0{}, Y{}) = pt.vertex0[1];
                    // scene.triangles[write](Vertex0{}, Z{}) = pt.vertex0[2];
                    // scene.triangles[write](Edge1{}, X{}) = pt.edge1[0];
                    // scene.triangles[write](Edge1{}, Y{}) = pt.edge1[1];
                    // scene.triangles[write](Edge1{}, Z{}) = pt.edge1[2];
                    // scene.triangles[write](Edge2{}, X{}) = pt.edge2[0];
                    // scene.triangles[write](Edge2{}, Y{}) = pt.edge2[1];
                    // scene.triangles[write](Edge2{}, Z{}) = pt.edge2[2];
                    // write++;
                }
                indexOffset += vertexCount;
            }
        }

        const auto sphere1 = Sphere{{-30.0f, 30.0f, -30.0f}, 30.0f};
        const auto sphere2 = Sphere{{30.0f, 30.0f, 30.0f}, 30.0f};
        for (const auto& s : {sphere1, sphere2})
            for (const auto c : {0, 1, 2})
            {
                box.lower[c] = std::min(box.lower[c], s.center[c] - s.radius);
                box.upper[c] = std::max(box.upper[c], s.center[c] + s.radius);
            }

        std::cout << "Loaded " << triangles.size() << " triangles (" << (triangles.size() * sizeof(Triangle)) / 1024
                  << "KiB)\n";

        scene.tree = OctreeNode{box};
        scene.tree.addObject(scene.nodePool, sphere1);
        scene.tree.addObject(scene.nodePool, sphere2);
        for (const auto& t : triangles)
            scene.tree.addObject(scene.nodePool, prepare(t));

        return scene;
    }

    template <typename F>
    void visitNodes(const OctreeNode& node, const F& f)
    {
        f(node);
        if (node.hasChildren())
            for (const auto& child : node.children())
                visitNodes(*child, f);
    }

    void printMemoryFootprint(const Scene& scene)
    {
        std::size_t triangleCount = 0;
        std::size_t nodeCount = 0;
        visitNodes(
            scene.tree,
            [&](const OctreeNode& node)
            {
                nodeCount++;
                if (!node.hasChildren())
                    triangleCount += node.objects().triangles.size();
            });
        std::cout << "Tree stores " << triangleCount << " triangles (" << (triangleCount * sizeof(Triangle)) / 1024
                  << "KiB) in " << nodeCount << " nodes (" << (nodeCount * sizeof(OctreeNode)) / 1024 << "KiB) \n";
    }
} // namespace

int main(int argc, const char* argv[])
try
{
    const auto width = 1024;
    const auto height = 768;

    // const auto scene = loadScene(sceneFile);
    // const auto scene = cubicBallsScene();
    // const auto scene = axisBallsScene();
    // const auto scene = randomSphereScene();
    // const auto scene = cornellBox();
    if (argc != 2)
    {
        std::cerr << "Please pass the location of sponza.obj as argument. The Sponza scene is available as git "
                     "submodule inside <gitrepo>/examples/raycast/Sponza\n";
        return 1;
    }
    Stopwatch watch;
    const auto scene = sponzaScene(argv[1]);
    watch.printAndReset("Loading");
    printMemoryFootprint(scene);
    watch.printAndReset("Visit  ");

    double avg = 0;
    constexpr auto repetitions = 5;
    for (auto i = 0; i < repetitions; i++)
    {
        const auto image = raycast(scene, width, height);
        avg += watch.printAndReset("Raycast");
        if (i == repetitions - 1)
            image.write("out.png");
    }
    std::cout << "Average " << avg / repetitions << " s\n";
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
    return 2;
}
