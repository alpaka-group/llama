/* cc by-sa 3.0 Ali
 * https://stackoverflow.com/questions/281818/
 * unmangling-the-result-of-stdtype-infoname
 */

#pragma once

#include <string>
#include <typeinfo>
#include <sstream>
#include <vector>
#include <iterator>
#include <boost/algorithm/string/replace.hpp>

template <class T>
std::string type(const T& t) {

    return llama::demangleType(typeid(t).name());
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

std::string nSpaces( int n )
{
    std::string result = "";
    for (int i = 0; i < n; ++i )
        result += " ";
    return result;
}

std::string addLineBreaks( std::string raw )
{
    boost::replace_all(raw, "<", "<\n");
    boost::replace_all(raw, ", ", ",\n");
    boost::replace_all(raw, " >", ">");
    boost::replace_all(raw, ">", "\n>");
    auto tokens = split(raw, '\n');
    std::string result = "";
    int indent = 0;
    for ( auto t : tokens )
    {
        if ( t.back() == '>' ||
            ( t.length() > 1 && t[ t.length()-2 ] == '>' ) )
            indent-=4;
        result += nSpaces(indent) + t + "\n";
        if ( t.back() == '<' )
            indent+=4;
    }
    return result;
}
