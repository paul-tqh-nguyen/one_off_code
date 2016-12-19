
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <ostream>
#include <cassert>
#include <png++/png.hpp>

#define QUIT exit(1);

#define INT(x) ((int)(x))
#define DOUBLE(x) ((double)(x))
#define FLOAT(x) ((float)(x))
#define BYTE(x) ((unsigned char)(x))

#define SQUARE(x) ((x)*(x))
#define MOD(a,b) ( ((((int)a)%((int)b))+((int)b))%((int)b) )
#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))
#define RAND_FLOAT (((double) rand()) / (double)RAND_MAX)
#define RAND_INT(lower, higher) (MOD(rand(),higher-lower)+lower)

#define PI (3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709)

#define DEBUG BUILD_DEBUG
#define ASSERT(x, msg) if (DEBUG && !(x)) { fprintf(stderr, "Assertion failed in %s(%d): %s\n", __FILE__, __LINE__, msg); assert(false); exit(1); }

#define TEST(x) cout << (#x) << ": " << x << endl;
#define PRINT(x) cout << x << endl;
#define DIVIDER cout << "=======================================================================================================================================================================" << endl;
#define NEWLINE cout << endl;

#define R_COORD 0
#define G_COORD 1
#define B_COORD 2

typedef unsigned char byte;

using std::string;
using std::to_string;
using std::ostream;
using std::vector;
using std::cout;
using std::endl;

template <class real>
real lerp(const real &a, const real &b, const double &t) {
    ASSERT(t>=0 && t<=1, (string("Interpolation factor (")+to_string(t)+") must be in interval [0.0,1.0]").c_str());
    return a*t+(1-t)*b;
}

template<class real>
class Array {
    public:
        real *data;
        vector<int> sizes;
        vector<int> stride;
        int nelems;
        
        void resize(const vector<int> &sizes_) {
            if (sizes == sizes_) { return; }
            
            delete[] data;
            sizes = sizes_;
            
            stride.resize(sizes.size());
            nelems = 1;
            for (int i = stride.size()-1; i >= 0; i--) {
                stride[i] = nelems;
                nelems *= sizes[i];
            }
            data = new real[nelems];
        }
        
        void assign(const Array &other) {
            resize(other.sizes);
            #pragma omp parallel for
            for(unsigned i=0; i < nelems; i++){
                data[i] = other.data[i];
            }
        }
        
        Array() :data(NULL) {
            resize(vector<int>{1});
        }
        
        Array(const Array &other) :data(NULL) {
            assign(other);
        }
        
        Array(const vector<int> &sizes_) :data(NULL) {
            resize(sizes_);
        }
        
        Array(const png::image< png::rgba_pixel > &png_image) :data(NULL) {
            int h = png_image.get_height();
            int w = png_image.get_width();
            resize(vector<int>{h, w, 3});
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    (*this)(y,x,R_COORD) = (real)(png_image[y][x].red);
                    (*this)(y,x,G_COORD) = (real)(png_image[y][x].green);
                    (*this)(y,x,B_COORD) = (real)(png_image[y][x].blue);
                }
            }
        }
        
        ~Array() {
            delete[] data;
        }
        
        void clear(const real &val=0) {
            #pragma omp parallel for
            for(unsigned i=0; i<nelems; i++){
                data[i]=val;
            }
        }
        
        real sum() {
            real sum = 0;
            for(unsigned i=0; i<nelems; i++){
                sum += data[i];
            }
            return sum;
        }
        
        real product() {
            real product = 0;
            for(unsigned i=0; i<nelems; i++){
                product *= data[i];
            }
            return product;
        }
        
        Array& normalize() {
            real sum = this->sum();
            #pragma omp parallel for
            for(unsigned i=0; i<nelems; i++){
                data[i] /= sum;
            }
            return (*this);
        }
        
        int height() const {
            ASSERT(sizes.size() >= 2, "Need at least 2 dimensions to get height");
            return sizes[0];
        }

        int width() const {
            ASSERT(sizes.size() >= 2, "Need at least 2 dimensions to get width");
            return sizes[1];
        }

        int channels() const {
            return (sizes.size()<3) ? 1 : sizes[2];
        }
        
        int dimensions() const {
            return sizes.size();
        }
        
        string str() const {
            string ans("[");
            if (dimensions() == 1) {
                for (int i = 0; i < nelems; i++) {
                    ans += to_string(data[i]);
                    if (i < nelems - 1) { ans += ", "; }
                }
            } else if (dimensions() == 2) {
                for (int y = 0; y < height(); y++) {
                    ans += "[";
                    for (int x = 0; x < width(); x++) {
                        ans += to_string((*this)(y,x));
                        if (x < width()-1) { ans += ", "; }
                    }
                    ans += "]";
                    if (y < height()-1) { ans += ",\n "; }
                }
            }else if (dimensions() == 3) {
                for (int z = 0; z < channels(); z++) {
                    ans += "\nChannel "+to_string(z)+":\n ";
                    for (int y = 0; y < height(); y++) {
                        ans += "[";
                        for (int x = 0; x < width(); x++) {
                            ans += to_string((*this)(y,x,z));
                            if (x < width()-1) { ans += ", "; }
                        }
                        ans += "]";
                        if (y < height()-1) { ans += ",\n "; }
                    }
                    ans += "\n";
                }
            } else {
                ASSERT(false, "Array::str() not implement for dimension > 3");
            }
            ans += "]";
            return ans;
        }
        
        real& operator()(int v0) const {
            ASSERT(sizes.size() == 1, (string("1D lookup in array of dimensionality")+to_string(sizes.size())).c_str()); 
            ASSERT(v0 >= 0 && v0 < sizes[0], (string("1D lookup out of bounds (index=")+to_string(v0)+", length="+to_string(sizes[0])+")").c_str()); 
            return data[v0];
        }
        
        real& operator()(int v0, int v1) const {
            ASSERT(sizes.size() == 2, (string("2D lookup in array of dimensionality")+to_string(sizes.size())).c_str()); 
            ASSERT(v0 >= 0 && v0 < sizes[0] && v1 >= 0 && v1 < sizes[1], (string("2D lookup out of bounds (row=")+to_string(v0)+", column="+to_string(v1)+", height="+to_string(sizes[0])+", width="+to_string(sizes[1])+")").c_str()); 
            return data[v0*stride[0]+v1];
        }
        
        real& operator()(int v0, int v1, int v2) const {
            ASSERT(sizes.size() == 3, (string("3D lookup in array of dimensionality")+to_string(sizes.size())).c_str());
            ASSERT(v0 >= 0 && v0 < sizes[0] && v1 >= 0 && v1 < sizes[1] && v2 >= 0 && v2 < sizes[2], (string("2D lookup out of bounds (row=")+to_string(v0)+", column="+to_string(v1)+", channel="+to_string(v1)+", height="+to_string(sizes[0])+", width="+to_string(sizes[1])+", num_channels="+to_string(sizes[2])+")").c_str()); 
            return data[v0*stride[0]+v1*stride[1]+v2];
        }
};

template<class real>
ostream& operator<<(ostream& os, const Array<real>& obj) {
    os << obj.str();
    return os;
}

// The code below for operating on .pfm files is modified from that of Connelly Barnes
int is_little_endian() {
    if (sizeof(float) != 4) { 
        PRINT("Bad float size."); 
        QUIT;
    }
    byte b[4] = { 255, 0, 0, 0 };
    return *((float*)b) < 1.0;
}

void write_pfm_file3(const char *filename, float *depth, int w, int h) {
    FILE *f = fopen(filename, "wb");
    static const int channels = 3;
    double scale = is_little_endian() ? -1.0 : 1.0;
    fprintf(f, "PF\n%d %d\n%lf\n", w, h, scale);
    for (int i = 0; i < w*h*channels; i++) {
        float d = depth[i];
        fwrite((void *) &d, 1, 4, f);
    }
    fclose(f);
}

