
/*

This is a naive nearest neighbor patch correspondence 
algorithm implementation with a complexity of O(n^2).

It is provided to demonstrate the performance improvement 
and accuracy loss we get from using PatchMatch.

This depends on the patchmatch.h header provided with 
the patchmatch implementation here: 
    https://github.com/nguyenp13/SampleCode/tree/master/image_processing/patchmatch

*/

#include "../../util/array.h"
#include "../../util/pfm.h"
#include "../patchmatch/patchmatch.h"

#define X_COORD 0
#define Y_COORD 1
#define D_COORD 2

using std::cout;
using std::endl;
using std::string;

void usage() {
    fprintf(stderr, "\n"
                    "usage: main <input_image_a>.png <input_image_b>.png <output_file>.pfm <patch_dim>\n"
                    "\n"
                    "Naive nearest neighbor patch correspondence algorithm. Intended to demonstate the performance improvement\n"
                    "given by using PatchMatch implementation. See http://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/index.php\n"
                    "for further explanation. <output_file>.pfm will be a .pfm file containing the nearest \n"
                    "neighbor field. <patch_dim> is an integer representing the patch height and width.\n"
                    "\n"
           );
    exit(1);
}

int main(int argc, char* argv[]) {
    
    if (argc < 5) {
        usage();
    }
    
    static const char* A_name = argv[1];
    static const char* B_name = argv[2];
    static const char* output_name = argv[3];
    static const int patch_dim = atoi(argv[4]);
    
    static const png::image< png::rgba_pixel > A_image(A_name); 
    static const png::image< png::rgba_pixel > B_image(B_name); 
    static const int A_height = A_image.get_height();
    static const int A_width = A_image.get_width();
    static const int B_height = B_image.get_height();
    static const int B_width = B_image.get_width();
    static const int Ann_height = A_height-patch_dim+1;
    static const int Ann_width = A_width-patch_dim+1;
    static const Array<byte> A(A_image);
    static const Array<byte> B(B_image);
    Array<int> Ann(vector<int>{Ann_height, Ann_width, 3});
    
    NEWLINE;
    PRINT("Parameter Values");
    TEST(A_height);
    TEST(A_width);
    TEST(B_height);
    TEST(B_width);
    TEST(Ann_height);
    TEST(Ann_width);
    TEST(patch_dim);
    
    unsigned long long initial_total_patch_distance = 0;
    #pragma omp parallel for reduction(+:initial_total_patch_distance)
    for (int yy = 0; yy < Ann_height; ++yy) {
        for (int xx = 0; xx < Ann_width; ++xx) {
            initial_total_patch_distance += Ann(yy,xx,D_COORD);
        }
    }
    NEWLINE;
    cout << "Initial Total Patch Distance: " << initial_total_patch_distance << endl;
    cout << "Initial Mean Patch Distance:  " << DOUBLE(initial_total_patch_distance)/DOUBLE(Ann_height*Ann_width) << endl;
    fflush(stdout);
    
    NEWLINE;
    PRINT("Initializing.");
    auto init_start_time = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int y = 0; y < Ann_height; ++y) {
        for (int x = 0; x < Ann_width; ++x) {
            Ann(y,x,D_COORD) = INT_MAX;
        }
    }
    auto init_end_time = std::chrono::high_resolution_clock::now();
    NEWLINE;
    cout << "Total Initialization Time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(init_end_time-init_start_time).count()) / (pow(10.0,9.0)) << " seconds." << endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    NEWLINE;
    PRINT("Nearest neighbor search started.");
    
    for (int a_y = 0; a_y < Ann_height; ++a_y) {
        for (int a_x = 0; a_x < Ann_width; ++a_x) {
            int &current_distance = Ann(a_y,a_x,D_COORD);
            for (int b_y = 0; b_y < B_height-patch_dim+1; ++b_y) {
                for (int b_x = 0; b_x < B_width-patch_dim+1; ++b_x) {
                    int new_distance = patch_distance(A, B, a_y, a_x, b_y, b_x, patch_dim);
                    if (new_distance < current_distance) {
                        Ann(a_y,a_x,X_COORD) = b_x;
                        Ann(a_y,a_x,Y_COORD) = b_y;
                        Ann(a_y,a_x,D_COORD) = new_distance;
                    }
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    NEWLINE;
    unsigned long long total_patch_dist = 0;
    for (int yy = 0; yy < Ann_height; ++yy) {
        for (int xx = 0; xx < Ann_width; ++xx) {
            total_patch_dist += Ann(yy,xx,D_COORD);
        }
    }
    cout << "Final Total Patch Distance: " << total_patch_dist << endl;
    cout << "Final Mean Patch Distance:  " << DOUBLE(total_patch_dist)/DOUBLE(Ann_height*Ann_width) << endl;
    
    float *depth = new float[Ann_height*Ann_width*3];
    for (int y = 0; y < Ann_height; y++) {
        for (int x = 0; x < Ann_width; x++) {
            int i = (Ann_height-1-y)*Ann_width*3+x*3;
            
            depth[i] = FLOAT(Ann(y,x,X_COORD));
            depth[i+1] = FLOAT(Ann(y,x,Y_COORD));
            depth[i+2] = FLOAT(Ann(y,x,D_COORD));
        }
    }
    write_pfm_file3(output_name, depth, Ann_width, Ann_height);
    
    NEWLINE;
    cout << "Total Run Time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(end_time-start_time).count()) / (pow(10.0,9.0)) << " seconds." << endl;
    NEWLINE;
    
    return 0;
}

