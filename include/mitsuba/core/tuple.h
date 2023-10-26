#pragma once

#include <functional>
#include <vector>
#include <tuple>
#include <iostream>

NAMESPACE_BEGIN(mitsuba)

template<typename Tuple, size_t N>
struct tuple_show
{
    static void show(const Tuple &t, std::ostream& os)
    {
        tuple_show<Tuple, N - 1>::show(t, os);
        os << ", " << std::get<N - 1>(t);
    }
};


// 偏特性，可以理解为递归的终止
template<typename Tuple>
struct tuple_show < Tuple, 1>
{
    static void show(const Tuple &t, std::ostream &os)
    {
        os <<  std::get<0>(t);
    }
};



// 自己写个函数，调用上面的递归展开，
template<typename... Args>
std::ostream& operator << (std::ostream &os, const std::tuple<Args...>& t)
{
    os << "[";
    tuple_show<decltype(t), sizeof...(Args)>::show(t, os);
    os << "]";

    return os;
}

NAMESPACE_END(end)
