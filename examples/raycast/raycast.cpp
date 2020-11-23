#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <optional>
#include <sstream>
#include <stb_image_write.h>
#include <stdexcept>
#include <string>

namespace
{
    constexpr auto pi = 3.14159265359f;

    template <typename F>
    struct Vector
    {
        auto operator[](std::size_t index) -> F&
        {
            return values[index];
        }

        auto operator[](std::size_t index) const -> F
        {
            return values[index];
        }

        auto operator+=(const Vector& v) noexcept -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] += v.values[i];
            return *this;
        }

        auto operator-=(const Vector& v) noexcept -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] -= v.values[i];
            return *this;
        }

        template <typename Scalar>
        auto operator*=(Scalar scalar) noexcept -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] *= scalar;
            return *this;
        }

        auto length() const noexcept
        {
            return std::sqrt((*this) * (*this));
        }

        void normalize()
        {
            const auto l = length();
            for (int i = 0; i < 3; i++)
                values[i] /= l;
        }

        auto normalized() const -> Vector
        {
            auto r = *this;
            r.normalize();
            return r;
        }

        std::array<F, 3> values = {{0, 0, 0}};
    };

    template <typename F>
    auto operator+(const Vector<F>& a, const Vector<F>& b) noexcept -> Vector<F>
    {
        auto r = a;
        r += b;
        return r;
    }

    template <typename F>
    auto operator-(const Vector<F>& a, const Vector<F>& b) noexcept -> Vector<F>
    {
        auto r = a;
        r -= b;
        return r;
    }

    template <typename F, typename Scalar>
    auto operator*(const Vector<F>& v, Scalar scalar) noexcept -> Vector<F>
    {
        auto r = v;
        r *= scalar;
        return r;
    }

    template <typename Scalar, typename F>
    auto operator*(Scalar scalar, const Vector<F>& v) noexcept -> Vector<F>
    {
        return v * scalar;
    }

    // dot product
    template <typename F>
    auto operator*(const Vector<F>& a, const Vector<F>& b) noexcept -> F
    {
        F r = 0;
        for (int i = 0; i < 3; i++)
            r += a[i] * b[i];
        return r;
    }

    // cross product
    template <typename F>
    auto operator%(const Vector<F>& a, const Vector<F>& b) -> Vector<F>
    {
        Vector<F> r;
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
        return r;
    }

    template <typename F>
    auto operator>>(std::istream& is, Vector<F>& v) -> std::istream&
    {
        for (int i = 0; i < 3; i++)
            is >> v[i];
        return is;
    }

    using VectorF = Vector<float>;
    using VectorD = Vector<double>;

    auto solveQuadraticEquation(double a, double b, double c) -> std::vector<double>
    {
        const double discriminat = std::pow(b, 2) - 4 * a * c;
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

    struct Scene
    {
        Scene(const std::filesystem::path& filepath)
        {
            std::ifstream f(filepath);
            if (!f)
                throw std::ios::failure("Failed to open file " + filepath.string() + " for reading");

            int lineCounter = 0;
            std::string line;
            while (std::getline(f, line))
            {
                if (line.empty())
                    continue;
                lineCounter++;
                std::stringstream ss(line);
                std::string token;

                auto expect = [&](const std::string& s) {
                    ss >> token;
                    if (token != s)
                        throw std::runtime_error("Excepted token: " + s + " on line " + std::to_string(lineCounter));
                };

                ss >> token;
                if (token == "sphere")
                {
                    Sphere sphere;
                    expect("radius");
                    ss >> sphere.radius;
                    expect("center");
                    ss >> sphere.center;

                    _spheres.push_back(sphere);
                }
                else if (token == "camera")
                {
                    expect("fovy");
                    ss >> _camera.fovy;
                    expect("position");
                    ss >> _camera.position;
                    expect("lookAt");
                    VectorF la;
                    ss >> la;
                    _camera.view = (la - _camera.position).normalized();
                    expect("up");
                    VectorF up;
                    ss >> up;
                    _camera.up = (_camera.view % (_camera.view % up)).normalized();
                }
                else
                    throw std::runtime_error(
                        "Unrecognized line command: " + token + " at start of line " + std::to_string(lineCounter));
            }
        }

        auto camera() const -> const Camera&
        {
            return _camera;
        }

        auto spheres() const -> const std::vector<Sphere>&
        {
            return _spheres;
        }

    private:
        Camera _camera;
        std::vector<Sphere> _spheres;
    };

    class Image
    {
    public:
        using Pixel = Vector<unsigned char>;

        Image(unsigned int width, unsigned int height) : width(width), height(height), pixels(width * height)
        {
        }

        auto operator()(unsigned int x, unsigned int y) -> Pixel&
        {
            return pixels[y * width + x];
        }

        auto operator()(unsigned int x, unsigned int y) const -> const Pixel&
        {
            return pixels[y * width + x];
        }

        void write(const std::filesystem::path& filename) const
        {
            if (!stbi_write_png(filename.string().c_str(), width, height, 3, pixels.data(), 0))
                throw std::runtime_error("Failed to write image " + filename.string());
        }

    private:
        unsigned int width;
        unsigned int height;
        std::vector<Pixel> pixels;
    };

    struct Intersection
    {
        float distance;
        VectorF point;
        VectorF normal;
    };

    struct Ray
    {
        VectorF origin;
        VectorF direction;
    };

    auto createRay(const Camera& camera, unsigned int width, unsigned int height, unsigned int x, unsigned int y) -> Ray
    {
        // we imagine a plane with the image just 1 before the camera, and then we shoot at those pixels

        const auto center = camera.position + camera.view;
        const auto xVec = camera.view % camera.up;
        const auto yVec = camera.up;

        const auto delta = (std::tan(camera.fovy * pi / 180.0f) * 2) / (height - 1);
        const auto xDeltaVec = xVec * delta;
        const auto yDeltaVec = yVec * delta;

        const auto xRel = (x - static_cast<float>(width - 1) / 2);
        const auto yRel = (y - static_cast<float>(height - 1) / 2);

        const auto pixel = center + xDeltaVec * xRel + yDeltaVec * yRel;

        Ray r;
        r.origin = center;
        r.direction = (pixel - camera.position).normalized();
        return r;
    }

    auto intersect(const Ray& ray, const Sphere& sphere) -> std::optional<Intersection>
    {
        // from
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection

        // solve quadratic equation
        const auto a = 1;
        const auto b = 2 * (ray.direction * (ray.origin - sphere.center));
        const auto c = std::pow((ray.origin - sphere.center).length(), 2) - std::pow(sphere.radius, 2);

        const auto solutions = solveQuadraticEquation(a, b, c);
        if (solutions.empty())
            return {};

        // report the closer intersection
        const auto t = static_cast<float>(*std::min_element(std::begin(solutions), std::end(solutions)));

        Intersection inter;
        inter.distance = t;
        inter.point = ray.origin + t * ray.direction;
        inter.normal = (inter.point - sphere.center).normalized();
        return inter;
    }

    auto colorByRay(const Ray& ray) -> Image::Pixel
    {
        Image::Pixel c;
        for (int i = 0; i < 3; i++)
            c[i] = static_cast<unsigned char>(std::abs(ray.direction[i]) * 255);
        return c;
    }

    auto colorByNearestIntersectionNormal(const std::vector<Intersection>& hits) -> Image::Pixel
    {
        const auto it = std::min_element(std::begin(hits), std::end(hits), [](const auto& a, const auto& b) {
            return a.distance < b.distance;
        });

        if (it == std::end(hits))
        {
            // no hit, black color
            return {};
        }
        else
        {
            Image::Pixel r;
            for (int i = 0; i < 3; i++)
                r[i] = static_cast<unsigned char>(std::abs(it->normal[i]) * 255);
            return r;
        }
    }

    auto colorByIntersectionNormal(std::vector<Intersection> hits) -> Image::Pixel
    {
        constexpr auto translucency = 0.5f;

        std::sort(std::begin(hits), std::end(hits), [](const auto& a, const auto& b) {
            return a.distance < b.distance;
        });

        auto t = translucency;
        Image::Pixel r;
        for (const auto& hit : hits)
        {
            // each hit contributes to the color, lesser with each iteration
            for (int i = 0; i < 3; i++)
                r[i] += static_cast<unsigned char>(std::abs(t * hit.normal[i]) * 255);
            t *= translucency;
        }

        for (int i = 0; i < 3; i++)
            r[i] = std::clamp<unsigned char>(r[i], 0, 255);

        return r;
    }

    auto raycast(const Scene& scene, unsigned int width, unsigned int height) -> Image
    {
        Image img(width, height);

        for (auto y = 0u; y < height; y++)
        {
            for (auto x = 0u; x < width; x++)
            {
                const auto ray = createRay(scene.camera(), width, height, x, y);

                std::vector<Intersection> hits;
                for (const auto& sphere : scene.spheres())
                    if (const auto hit = intersect(ray, sphere))
                        hits.push_back(*hit);

                // img(x, y) = colorByRay(ray);
                // img(x, y) = colorByNearestIntersectionNormal(hits);
                img(x, y) = colorByIntersectionNormal(hits);
            }
        }

        return img;
    }
} // namespace

int main(int argc, const char* argv[])
try
{
    if (argc != 4)
    {
        std::cerr << "Invalid number of arguments. Usage:\n\n";
        std::cerr << argv[0] << " sceneFile width height\n";
        return 1;
    }

    const std::filesystem::path sceneFile = argv[1];
    const unsigned int width = std::stoi(argv[2]);
    const unsigned int height = std::stoi(argv[3]);

    if (!sceneFile.has_stem())
        throw std::invalid_argument("Except scene file " + sceneFile.string() + " to have a stem");
    auto imageFile = sceneFile;
    imageFile.replace_extension("png");

    Scene scene(sceneFile);

    const auto start = std::chrono::high_resolution_clock::now();
    const auto image = raycast(scene, width, height);
    const auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Raycast took " << std::chrono::duration<double>(end - start).count() << "s\n";

    image.write(imageFile);
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
    return 2;
}
