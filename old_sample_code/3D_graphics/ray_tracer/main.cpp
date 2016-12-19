
/*

TODO:
    Remove calculation of all unecessary Vector3 objects
    make more things static const if they can be
    obj file loader
    Global illumination
    Beam tracing
    Cone tracing
    distributed ray tracer
    
    Phong Illumination
    refraction

*/

#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <chrono>
#include "util.h"

using std::cout;
using std::endl;
using std::string;

void usage() {
    fprintf(stderr, "usage: main <geometric_object_file_name> <light_sources_file_name>\n");
    exit(1);
}

int main(int argc, char* argv[]) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (argc < 3) { 
        usage(); 
    }
    
    char* geometric_primitive_file_name = argv[1];
    char* light_source_file_name = argv[2];
    
    srand((size_t)argv);
    
    vector<GeometricPrimitive*> geometric_primitive_pointers;
    read_objects_from_file(geometric_primitive_file_name, geometric_primitive_pointers);
    
    vector<GeometricPrimitive*> light_source_pointers;
    read_objects_from_file(light_source_file_name, light_source_pointers);
               
    for (int i = 0; i < 10000; i++) {
        PRINT( string("Working on iteration "+to_string(i)) );
    
        ray_tracer(geometric_primitive_pointers, string("output-")+to_string(i)+".png", 
                   i+55.0, // angle
                   Vector3(500-i,-500+i,-1), // eye position
                   light_source_pointers, // vector of light source pointers
                   Vector3(-5.0+i/2.0,5.0-i/2.0,-10.0+i/2.0).normalize(), // look direction
                   Vector3(0,1,0).normalize(), // up direction
                   1024, // I_dim
                   1.0, // viewport_distance
                   3, // number of ray bounces
                   100*i // number of light rays per pixel
                   );
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    cout << "Total Run Time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(end_time-start_time).count()) / (pow(10.0,9.0)) << " seconds." << endl;
    
    return 0;
}

