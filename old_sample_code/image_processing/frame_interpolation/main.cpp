
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <climits>
#include "util.h"

#define CLAMP_FLOAT(p) ( MIN(1.0,MAX(0.0,(p))) )
#define CLAMP_INT(p) ( MIN(255,MAX(0,(p))) )
#define CLAMP(val,min,max) ( MIN((max),MAX((min),(val))) )
#define PIXEL(r,g,b) ( png::rgba_pixel(CLAMP_INT(r),CLAMP_INT(g),CLAMP_INT(b),255) )

#define X_COORD 0
#define Y_COORD 1
#define D_COORD 2

#define R_COORD 0
#define G_COORD 1
#define B_COORD 2

#define CUT_OFF_DIST DOUBLE(40)
#define ROOT_TWO 1.41421356237309504880168872420969807856967187537694807317667973799073247846210703885038753432764157273501384623091229702492483605585073721264412149709993583141322266592750559275

// clear && clear && make debug && ./main a.png b.png output 100
// convert   -delay 10   -loop 0   out*.png   final.gif

int patch_distance(const Array<byte> &A, const Array<byte> &B, const int &a_y, const int &a_x, const int &b_y, const int &b_x, const int &patch_dim, const double &sigma) {
    ASSERT(A.sizes.size()==3, (string("A must be 3 dimensional (A currently has ")+to_string(A.sizes.size())+" dimensions)").c_str());
    ASSERT(B.sizes.size()==3, (string("B must be 3 dimensional (B currently has ")+to_string(B.sizes.size())+" dimensions)").c_str());
    ASSERT(A.channels()==3, (string("A must have 3 channels (A currently has ")+to_string(A.channels())+" channels)").c_str());
    ASSERT(B.channels()==3, (string("B must have 3 channels (B currently has ")+to_string(B.channels())+" channels)").c_str());
    ASSERT(patch_dim>3, (string("patch_dim (")+to_string(patch_dim)+") must be greater than zero").c_str());
    ASSERT(patch_dim<=A.height(), (string("patch_dim (")+to_string(patch_dim)+") must be less than the height of A ("+to_string(A.height())+")").c_str());
    ASSERT(patch_dim<=A.width(), (string("patch_dim (")+to_string(patch_dim)+") must be less than the width of A ("+to_string(A.width())+")").c_str());
    ASSERT(patch_dim<=B.height(), (string("patch_dim (")+to_string(patch_dim)+") must be less than the height of B ("+to_string(B.height())+")").c_str());
    ASSERT(patch_dim<=B.width(), (string("patch_dim (")+to_string(patch_dim)+") must be less than the width of B ("+to_string(B.width())+")").c_str());
    ASSERT(a_y < A.height(), (string("a_y (")+to_string(a_y)+") is greater than or equal to the height of A ("+to_string(A.height())+")").c_str());
    ASSERT(a_y >= 0, (string("a_y (")+to_string(a_y)+") is less than zero").c_str());
    ASSERT(a_x < A.width(), (string("a_x (")+to_string(a_x)+") is greater than or equal to the width of A ("+to_string(A.width())+")").c_str());
    ASSERT(a_x >= 0, (string("a_x (")+to_string(a_x)+") is less than zero").c_str());
    ASSERT(b_y < B.height(), (string("b_y (")+to_string(b_y)+") is greater than or equal to the height of B ("+to_string(B.height())+")").c_str());
    ASSERT(b_y >= 0, (string("b_y (")+to_string(b_y)+") is less than zero").c_str());
    ASSERT(b_x < B.width(), (string("b_x (")+to_string(b_x)+") is greater than or equal to the width of B ("+to_string(B.width())+")").c_str());
    ASSERT(b_x >= 0, (string("b_x (")+to_string(b_x)+") is less than zero").c_str());
    
    if (SQUARE(a_x-b_x)+SQUARE(a_y-b_y) > SQUARE(CUT_OFF_DIST)) {
        return INT_MAX;
    }
    
    int int_dist = 0; // since int addition is faster than float addition
    
    #pragma omp parallel for reduction(+:int_dist)
    for (int r=0; r<patch_dim; r++) {
        for (int c=0; c<patch_dim; c++) {
            int_dist += SQUARE(INT(A(a_y+r,a_x+c,R_COORD))-INT(B(b_y+r,b_x+c,R_COORD)));
            int_dist += SQUARE(INT(A(a_y+r,a_x+c,G_COORD))-INT(B(b_y+r,b_x+c,G_COORD)));
            int_dist += SQUARE(INT(A(a_y+r,a_x+c,B_COORD))-INT(B(b_y+r,b_x+c,B_COORD)));
        }
    }
    
    ASSERT(int_dist>=0, (string("dist (")+to_string(int_dist)+") is negative").c_str());
    
//    return int_dist*G(SQUARE(a_x-b_x)+SQUARE(a_y-b_y),sigma);
    return int_dist;
}

void patch_match(const Array<byte> &A, const Array<byte> &B, const int &patch_dim, const int &num_iterations, const int &random_search_size_exponent, const int &num_random_search_attempts, const double &spatial_sigma, Array<int> &Ann) { 
    
    static const int &A_height = A.height();
    static const int &A_width = A.width();
    static const int &B_height = B.height();
    static const int &B_width = B.width();
    static const int Ann_height = A_height-patch_dim+1;
    static const int Ann_width = A_width-patch_dim+1;
    Ann.resize(vector<int>{Ann_height, Ann_width, 3});
    
//    #pragma omp parallel for
    for (int y = 0; y < Ann_height; ++y) {
        for (int x = 0; x < Ann_width; ++x) {
//            Ann(y,x,X_COORD) = RAND_INT(0,B_width-patch_dim+1);
//            Ann(y,x,Y_COORD) = RAND_INT(0,B_height-patch_dim+1);
            int xx = INT(x+RAND_INT(-INT(CUT_OFF_DIST/ROOT_TWO), INT(CUT_OFF_DIST/ROOT_TWO)));
            int yy = INT(y+RAND_INT(-INT(CUT_OFF_DIST/ROOT_TWO), INT(CUT_OFF_DIST/ROOT_TWO)));
            Ann(y,x,X_COORD) = CLAMP(xx, 0, Ann_width-1);
            Ann(y,x,Y_COORD) = CLAMP(yy, 0, Ann_height-1);
            Ann(y,x,D_COORD) = patch_distance(A, B, y, x, Ann(y,x,Y_COORD), Ann(y,x,X_COORD), patch_dim, spatial_sigma);
        }
    }
    
    for (int iteration_index = 0; iteration_index < num_iterations; ++iteration_index) {
        int y_start, x_start, y_end, x_end, delta;
        if (MOD(iteration_index,2) == 0) {
            y_start = 0;
            x_start = 0;
            y_end = Ann_height-1;
            x_end = Ann_width-1;
            delta = 1;
        } else {
            y_start = Ann_height-1;
            x_start = Ann_width-1;
            y_end = 0;
            x_end = 0;
            delta = -1;
        }
        for (int y = y_start; y != y_end+delta; y += delta) {
            for (int x = x_start; x != x_end+delta; x += delta) {
                
                int &current_distance = Ann(y,x,D_COORD);
                
                if (y-delta>=0 && y-delta<Ann_height) {
                    int y_neighbor_distance = patch_distance(A, B, y, x, Ann(y-delta,x,Y_COORD), Ann(y-delta,x,X_COORD), patch_dim, spatial_sigma);
                    if (current_distance > y_neighbor_distance) {
                        Ann(y,x,X_COORD) = Ann(y-delta,x,X_COORD);
                        Ann(y,x,Y_COORD) = Ann(y-delta,x,Y_COORD);
                        current_distance = y_neighbor_distance;
                    }
                }
                if (x-delta>=0 && x-delta<Ann_height) {
                    int x_neighbor_distance = patch_distance(A, B, y, x, Ann(y,x-delta,Y_COORD), Ann(y,x-delta,X_COORD), patch_dim, spatial_sigma);
                    if (current_distance > x_neighbor_distance) {
                        Ann(y,x,X_COORD) = Ann(y,x-delta,X_COORD);
                        Ann(y,x,Y_COORD) = Ann(y,x-delta,Y_COORD);
                        current_distance = x_neighbor_distance;
                    }
                }
                
                for (int exponent = random_search_size_exponent; exponent > 0; exponent--) {
                    const int radius = 1<<exponent;
                    const int rs_x_min = MAX(x-radius, 0);
                    const int rs_x_max = MIN(x+radius, Ann_width);
                    const int rs_y_min = MAX(y-radius, 0);
                    const int rs_y_max = MIN(y+radius, Ann_height);
                    for (int random_search_attempt_index=0; random_search_attempt_index<num_random_search_attempts; random_search_attempt_index++){
                        int rs_y = RAND_INT(rs_y_min, rs_y_max);
                        int rs_x = RAND_INT(rs_x_min, rs_x_max);
                        int neighbor_distance = patch_distance(A, B, y, x, Ann(rs_y,rs_x,Y_COORD), Ann(rs_y,rs_x,X_COORD), patch_dim, spatial_sigma);
                        if (current_distance > neighbor_distance) {
                            Ann(y,x,X_COORD) = Ann(rs_y,rs_x,X_COORD);
                            Ann(y,x,Y_COORD) = Ann(rs_y,rs_x,Y_COORD);
                            current_distance = neighbor_distance;
                        }
                    }
                }
            }
        }
    }
}

void usage() {
    fprintf(stderr, "usage: main <input_image_a>.png <input_image_b>.png <output_file_name_prefix> <num_frames> <spatial_sigma> [-patch_dim <patch_dim>] [-num_iterations <num_iterations>] "
                        "[-random_search_size_exponent <random_search_size_exponent>] [-num_random_search_attempts <num_random_search_attempts>]\n"
                    "\n"
                    "Given two frames from a video sequence, we attempt to interpolate <num_frames> frames between\n"
                    "the two. We find patch correspondences using the PatchMatch algorithm (for further details, see\n"
                    "http://gfx.cs.princeton.edu/pubs/Barnes_2011_TPR/index.php). We then use image morphing techniques\n"
                    "introduced by Beier and Neely to interpolate the frames (see http://dl.acm.org/citation.cfm?id=134003\n"
                    "for further explanation).\n"
                    "\n"
                    "<spatial_sigma> is used for our patch distance function. A lower <spatial_sigma> will lead to a lower patch similarity for those patches that are spatially further away.\n"
                    "<patch_dim> specifies the patch height/width to use for PatchMatch.\n"
                    "<num_iterations> specifies the number of iterations of propogation to use for PatchMatch.\n"
                    "<random_search_size_exponent> is used to determine the max radius (max radius = 2^random_search_size_exponent) of the random search in PatchMatch.\n"
                    "<num_random_search_attempts> specifies the number of patches to search during each iteration of the random search in PatchMatch.\n"
                    "\n"
           );
    exit(1);
}

int main(int argc, char* argv[]) {
    
    if (argc < 6) {
        usage();
    }
    
    std::srand( std::time(NULL) );
    
    static const char* A_name = argv[1];
    static const char* B_name = argv[2];
    static const string output_prefix = string(argv[3]);
    static const int num_frames = atoi(argv[4]);
    static const int spatial_sigma = atoi(argv[5]);
    
    static int patch_dim = 25;
    static int num_iterations = 2;
    static int random_search_size_exponent = 4;
    static int num_random_search_attempts = 25;
    
    for (int i = 6; i < argc; i++) {
        if ( !strcmp(argv[i],"-patch_dim") ) {
            patch_dim = atoi(argv[i+1]);
        } else if ( !strcmp(argv[i],"-num_iterations") ) {
            num_iterations = atoi(argv[i+1]);
        } else if ( !strcmp(argv[i],"-random_search_size_exponent") ) {
            random_search_size_exponent = atoi(argv[i+1]);
        } else if ( !strcmp(argv[i],"-num_random_search_attempts") ) {
            num_random_search_attempts = atoi(argv[i+1]);
        } 
    }
    
    NEWLINE;
    PRINT("PatchMatch Parameters:");
    cout << "    "; TEST(patch_dim);
    cout << "    "; TEST(num_iterations);
    cout << "    "; TEST(random_search_size_exponent);
    cout << "    "; TEST(num_random_search_attempts);
    cout << "    "; TEST(spatial_sigma);
    
    static const png::image< png::rgba_pixel > A_image(A_name);
    static const png::image< png::rgba_pixel > B_image(B_name);
    static const Array<byte> A(A_image);
    static const Array<byte> B(B_image);
    auto Ann = Array<int>();
    auto Bnn = Array<int>();
    
    ASSERT(A.height() == B.height(), "A and B must have the same height");
    ASSERT(A.width() == B.width(), "A and B must have the same width");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    NEWLINE;
    PRINT("Calculating nearest neighbor field for image A.");
    patch_match(A, B, patch_dim, num_iterations, random_search_size_exponent, num_random_search_attempts, spatial_sigma, Ann);
    PRINT("Calculating nearest neighbor field for image B.");
    patch_match(B, A, patch_dim, num_iterations, random_search_size_exponent, num_random_search_attempts, spatial_sigma, Bnn);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    NEWLINE;
    PRINT("Starting frame interpolation.");
    NEWLINE;
    
    int output_height = Ann.height();
    int output_width = Ann.width();
    for (int i = 0; i < num_frames+2; i++) {
        double t = DOUBLE(i)/DOUBLE(num_frames+1);
        cout << "Working on frame " << i << " with t=" << t << endl;
        png::image< png::rgba_pixel > output_image(output_width, output_height);
        for (int y=0; y<output_height; y++) {
            for (int x=0; x<output_width; x++) {
                int a_x = std::round(lerp(x, Bnn(y,x,X_COORD), t));
                int a_y = std::round(lerp(y, Bnn(y,x,Y_COORD), t));
                int b_x = std::round(lerp(x, Ann(y,x,X_COORD), 1.0-t));
                int b_y = std::round(lerp(y, Ann(y,x,Y_COORD), 1.0-t));
                
                output_image[y][x] = PIXEL( lerp(INT(A(a_y,a_x,R_COORD)), INT(B(b_y,b_x,R_COORD)), t), 
                                            lerp(INT(A(a_y,a_x,G_COORD)), INT(B(b_y,b_x,G_COORD)), t), 
                                            lerp(INT(A(a_y,a_x,B_COORD)), INT(B(b_y,b_x,B_COORD)), t) );
            }
        }
        std::stringstream image_index;
        image_index << std::setw(10) << std::setfill('0') << i;
        output_image.write((string(output_prefix)+image_index.str()+".png").c_str());
    }
    NEWLINE;
    PRINT("Frame interpolation complete.");
    
    NEWLINE;
    cout << "Total Run Time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(end_time-start_time).count()) / (pow(10.0,9.0)) << " seconds." << endl;
    NEWLINE;
    
    return 0;
}

