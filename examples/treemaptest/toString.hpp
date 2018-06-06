#pragma once

namespace llama
{

namespace mapping
{

namespace tree
{

template< >
struct ToString< st::Pos >
{
    auto
    operator()( const st::Pos )
    -> std::string
    {
        return "Pos";
    }
};

template< >
struct ToString< st::X >
{
    auto
    operator()( const st::X )
    -> std::string
    {
        return "X";
    }
};

template< >
struct ToString< st::Y >
{
    auto
    operator()( const st::Y )
    -> std::string
    {
        return "Y";
    }
};

template< >
struct ToString< st::Z >
{
    auto
    operator()( const st::Z )
    -> std::string
    {
        return "Z";
    }
};

template< >
struct ToString< st::Momentum >
{
    auto
    operator()( const st::Momentum )
    -> std::string
    {
        return "Momentum";
    }
};

template< >
struct ToString< st::Weight >
{
    auto
    operator()( const st::Weight )
    -> std::string
    {
        return "Weight";
    }
};

} // namespace tree

} // namespace mapping

} // namespace llama
