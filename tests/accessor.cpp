#include "common.hpp"

TEST_CASE("view.allocView.Default")
{
    auto mapping = llama::mapping::AoS{llama::ArrayExtents{3, 4}, Particle{}};
    auto view = llama::allocView(mapping, llama::bloballoc::Vector{}, llama::accessor::Default{});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("view.allocView.ByValue")
{
    auto mapping = llama::mapping::AoS{llama::ArrayExtents{3, 4}, Vec3I{}};
    auto view = llama::allocView(mapping, llama::bloballoc::Vector{}, llama::accessor::ByValue{});
    STATIC_REQUIRE(std::is_same_v<decltype(view(1, 2)(tag::X{})), int>);
}

TEST_CASE("view.allocView.Const")
{
    auto mapping = llama::mapping::AoS{llama::ArrayExtents{3, 4}, Vec3I{}};
    auto view = llama::allocView(mapping, llama::bloballoc::Vector{}, llama::accessor::Const{});
    STATIC_REQUIRE(std::is_same_v<decltype(view(1, 2)(tag::X{})), const int&>);
}

#ifdef __cpp_lib_atomic_ref
TEST_CASE("view.allocView.Atomic")
{
    auto mapping = llama::mapping::AoS{llama::ArrayExtents{3, 4}, Vec3I{}};
    auto view = llama::allocView(mapping, llama::bloballoc::Vector{}, llama::accessor::Atomic{});
    STATIC_REQUIRE(std::is_same_v<decltype(view(1, 2)(tag::X{})), std::atomic_ref<int>>);
    iotaFillView(view);
    iotaCheckView(view);
}
#endif

TEST_CASE("view.withAccessor.Default.Vector")
{
    auto view
        = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Vec3I{}}, llama::bloballoc::Vector{});
    auto* addr = &view(1, 2)(tag::X{});
    auto view2 = llama::withAccessor<llama::accessor::Default>(view); // copies
    auto* addr2 = &view2(1, 2)(tag::X{});
    CHECK(addr != addr2);
}

TEST_CASE("view.withAccessor.Default.Vector.move")
{
    auto view
        = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Vec3I{}}, llama::bloballoc::Vector{});
    auto* addr = &view(1, 2)(tag::X{});
    auto view2 = llama::withAccessor<llama::accessor::Default>(std::move(view));
    auto* addr2 = &view2(1, 2)(tag::X{});
    CHECK(addr == addr2);
}

TEST_CASE("view.withAccessor.Default.SharedPtr")
{
    auto view
        = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Vec3I{}}, llama::bloballoc::SharedPtr{});
    auto* addr = &view(1, 2)(tag::X{});
    auto view2 = llama::withAccessor<llama::accessor::Default>(view); // copies shared pointers, but not memory chunks
    auto* addr2 = &view2(1, 2)(tag::X{});
    CHECK(addr == addr2);
}

TEMPLATE_TEST_CASE(
    "view.withAccessor.shallowCopy.Default",
    "",
    llama::bloballoc::Vector,
    llama::bloballoc::SharedPtr,
    llama::bloballoc::UniquePtr)
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Particle{}}, TestType{});
    auto view2 = llama::withAccessor<llama::accessor::Default>(llama::shallowCopy(view));
    iotaFillView(view2);
    iotaCheckView(view);
}

#ifdef __cpp_lib_atomic_ref
TEMPLATE_TEST_CASE(
    "view.withAccessor.shallowCopy.Atomic",
    "",
    llama::bloballoc::Vector,
    llama::bloballoc::SharedPtr,
    llama::bloballoc::UniquePtr)
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Particle{}}, TestType{});
    auto view2 = llama::withAccessor<llama::accessor::Atomic>(llama::shallowCopy(view));
    iotaFillView(view2);
    iotaCheckView(view);
}
#endif

TEMPLATE_TEST_CASE(
    "view.withAccessor.shallowCopy.Restrict",
    "",
    llama::bloballoc::Vector,
    llama::bloballoc::SharedPtr,
    llama::bloballoc::UniquePtr)
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Particle{}}, TestType{});
    auto view2 = llama::withAccessor<llama::accessor::Restrict>(llama::shallowCopy(view));
    iotaFillView(view2);
    iotaCheckView(view);
}

namespace
{
    struct OffsetFloatAccessor
    {
        float offset;

        template<typename T>
        auto operator()(T& ref) -> decltype(auto)
        {
            if constexpr(std::is_same_v<T, float>)
                return ref + offset;
            else
                return ref;
        }
    };
} // namespace

TEST_CASE("view.withAccessor.OffsetFloatAccessor")
{
    auto view
        = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{4}, Particle{}}, llama::bloballoc::SharedPtr{});
    view(0)(tag::Pos{})(tag::X{}) = 2.0;
    view(0)(tag::Mass{}) = 2.0f;

    auto view2 = llama::withAccessor(view, OffsetFloatAccessor{3});

    CHECK(view2(0)(tag::Pos{})(tag::X{}) == 2.0);
    CHECK(view2(0)(tag::Mass{}) == 5.0f);

    view2.accessor().offset = 10;

    CHECK(view2(0)(tag::Pos{})(tag::X{}) == 2.0);
    CHECK(view2(0)(tag::Mass{}) == 12.0f);
}
