#include "common.hpp"

namespace
{
    template<typename T>
    struct FakeProxyRef : llama::ProxyRefOpMixin<FakeProxyRef<T>, T>
    {
        using value_type = T;

        explicit FakeProxyRef(T& r) : r(r)
        {
        }

        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        operator T() const
        {
            return r;
        }

        auto operator=(T t) -> FakeProxyRef&
        {
            r = t;
            return *this;
        }

        T& r;
    };
} // namespace

TEST_CASE("FakeProxyRef.init")
{
    int i = 42;
    auto r = FakeProxyRef<int>{i};
    CHECK(r == 42);
}

TEST_CASE("proxyrefopmixin.operator+=")
{
    int i = 42;
    auto r = FakeProxyRef<int>{i};
    r += 5;
    CHECK(r == 47);

    int j = 100;
    auto r2 = FakeProxyRef<int>{j};
    r2 += r += 5;
    CHECK(r2 == 152);
}

TEST_CASE("proxyrefopmixin.operator-=")
{
    int i = 42;
    auto r = FakeProxyRef<int>{i};
    r -= 5;
    CHECK(r == 37);

    int j = 100;
    auto r2 = FakeProxyRef<int>{j};
    r2 -= r -= 7;
    CHECK(r2 == 70);
}

TEST_CASE("proxyrefopmixin.operator*=")
{
    int i = 42;
    auto r = FakeProxyRef<int>{i};
    r *= 2;
    CHECK(r == 84);

    int j = 100;
    auto r2 = FakeProxyRef<int>{j};
    r2 *= r *= 4;
    CHECK(r2 == 33600);
}

TEST_CASE("proxyrefopmixin.operator/=")
{
    int i = 42;
    auto r = FakeProxyRef<int>{i};
    r /= 2;
    CHECK(r == 21);

    int j = 100;
    auto r2 = FakeProxyRef<int>{j};
    r2 /= r /= 4;
    CHECK(r2 == 20);
}

TEST_CASE("proxyrefopmixin.operator%=")
{
    int i = 42;
    auto r = FakeProxyRef<int>{i};
    r %= 11;
    CHECK(r == 9);

    int j = 100;
    auto r2 = FakeProxyRef<int>{j};
    r2 %= r %= 6;
    CHECK(r2 == 1);
}

TEST_CASE("proxyrefopmixin.operator<<=")
{
    unsigned i = 0b1u;
    auto r = FakeProxyRef<unsigned>{i};
    r <<= 2u;
    CHECK(r == 0b100u);

    unsigned j = 100u;
    auto r2 = FakeProxyRef<unsigned>{j};
    r2 <<= r <<= 2u;
    CHECK(r2 == 100u << 16u);
}

TEST_CASE("proxyrefopmixin.operator>>=")
{
    unsigned i = 0b10000u;
    auto r = FakeProxyRef<unsigned>{i};
    r >>= 2u;
    CHECK(r == 0b100u);

    unsigned j = 100u;
    auto r2 = FakeProxyRef<unsigned>{j};
    r2 >>= r >>= 1u;
    CHECK(r2 == 100u >> 2u);
}

TEST_CASE("proxyrefopmixin.operator&=")
{
    unsigned i = 0b10111u;
    auto r = FakeProxyRef<unsigned>{i};
    r &= 0b01111u;
    CHECK(r == 0b00111u);

    unsigned j = 0b01010u;
    auto r2 = FakeProxyRef<unsigned>{j};
    r2 &= r &= 0b00011u;
    CHECK(r2 == 0b00010u);
}

TEST_CASE("proxyrefopmixin.operator|=")
{
    unsigned i = 0b00100u;
    auto r = FakeProxyRef<unsigned>{i};
    r |= 0b00001u;
    CHECK(r == 0b00101u);

    unsigned j = 0b10000u;
    auto r2 = FakeProxyRef<unsigned>{j};
    r2 |= r |= 0b00011u;
    CHECK(r2 == 0b10111u);
}

TEST_CASE("proxyrefopmixin.operator^=")
{
    unsigned i = 0b10111u;
    auto r = FakeProxyRef<unsigned>{i};
    r ^= 0b00001u;
    CHECK(r == 0b10110u);

    unsigned j = 0b11000u;
    auto r2 = FakeProxyRef<unsigned>{j};
    r2 ^= r ^= 0b00011u;
    CHECK(r2 == 0b01101u);
}

TEST_CASE("proxyrefopmixin.operator++")
{
    int i = 42;
    auto r = FakeProxyRef<int>{i};
    ++r;
    CHECK(r == 43);
    ++++r;
    CHECK(r == 45);
}

TEST_CASE("proxyrefopmixin.operator++(int)")
{
    int i = 42;
    auto r = FakeProxyRef<int>{i};
    auto old = r++;
    CHECK(r == 43);
    STATIC_REQUIRE(std::is_same_v<decltype(old), int>);
    CHECK(old == 42);
}

namespace
{
    template<typename ProxyReference>
    void testProxyRef(ProxyReference&& r)
    {
        r = 42;
        r += 2;
        CHECK(r == 44);
        r -= 3;
        CHECK(r == 41);
        r *= 4;
        CHECK(r == 164);
        r /= 82;
        CHECK(r == 2);

        if constexpr(std::is_integral_v<typename ProxyReference::value_type>)
        {
            r = 20;
            r %= 15;
            CHECK(r == 5);
            r <<= 2;
            CHECK(r == 20);
            r >>= 1;
            CHECK(r == 10);
            r &= 7;
            CHECK(r == 2);
            r |= 9;
            CHECK(r == 11);
            r ^= 8;
            CHECK(r == 3);
        }

        r = 9; // NOLINT(readability-misleading-indentation)
        r++;
        CHECK(r == 10);
        ++r;
        CHECK(r == 11);
        r--;
        CHECK(r == 10);
        --r;
        CHECK(r == 9);
    }
} // namespace

TEST_CASE("proxyrefopmixin.Bytesplit")
{
    auto view = llama::allocView(
        llama::mapping::Bytesplit<llama::ArrayExtents<4>, Vec3I, llama::mapping::PreconfiguredAoS<>::type>{{}});
    testProxyRef(view(2)(tag::X{}));
}

TEST_CASE("proxyrefopmixin.BitPackedIntSoA")
{
    auto view = llama::allocView(llama::mapping::BitPackedIntSoA<llama::ArrayExtents<4>, Vec3I>{12, {}});
    testProxyRef(view(2)(tag::X{}));
}

TEST_CASE("proxyrefopmixin.BitPackedFloatSoA")
{
    auto view = llama::allocView(llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<4>, Vec3D>{6, 20, {}});
    testProxyRef(view(2)(tag::X{}));
}

TEST_CASE("proxyrefopmixin.ChangeType")
{
    auto view = llama::allocView(llama::mapping::ChangeType<
                                 llama::ArrayExtents<4>,
                                 Vec3D,
                                 llama::mapping::PreconfiguredAoS<false>::type,
                                 boost::mp11::mp_list<boost::mp11::mp_list<double, float>>>{{}});
    testProxyRef(view(2)(tag::X{}));
}
