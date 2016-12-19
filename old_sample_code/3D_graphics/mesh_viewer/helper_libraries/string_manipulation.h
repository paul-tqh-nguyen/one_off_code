// String Manipulation Library
// Meant to be convenient, not fast (string manipulation never is) 

#ifndef _STRING_MANIPULATION_H_
#define _STRING_MANIPULATION_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <assert.h>

using std::string;
using std::vector;
using std::cout;
using std::endl;

#define DEBUG BUILD_DEBUG
#define ASSERT(x, msg) if (DEBUG && !(x)) { fprintf(stderr, "Assertion failed in %s(%d): %s\n", __FILE__, __LINE__, msg); assert(false); exit(1); }

void split(const string &line, const string &delimiter, vector<string> &vector_of_strings){
    auto start = 0U;
    auto end = line.find(delimiter);
    while (end != std::string::npos){
        vector_of_strings.push_back( line.substr(start, end - start) );
        start = end + delimiter.length();
        end = line.find(delimiter, start);
    }
    vector_of_strings.push_back( line.substr(start, end) );
}

#endif // _STRING_MANIPULATION_H_

