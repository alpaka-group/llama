/* cc by-sa 3.0 Ali
 * https://stackoverflow.com/questions/281818/
 * unmangling-the-result-of-stdtype-infoname
 */

#pragma once

#include <string>
#include <typeinfo>
#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>

std::string demangle(const char* name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };

    return (status==0) ? res.get() : name ;
}

#else

// does nothing if not g++
std::string demangle(const char* name) {
    return name;
}

#endif

template <class T>
std::string type(const T& t) {

    return demangle(typeid(t).name());
}
