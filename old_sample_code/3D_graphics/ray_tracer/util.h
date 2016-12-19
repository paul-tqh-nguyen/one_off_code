
/*

TODO:
    
    make sure stream operator works for all classes for TEST() printing
    
    matrix 3
        matrix by matrix elemwise methods and functions
        functions and methods with reals
    
    change all non-mutator methods to const
    
    more matrix 3 operations to the bunch of operations we have below
        and the reassignment operators
    
    Make sure we dont' ahve to do a Vecor3(input vector reference) for our inits of triangle and such
        bc we don't know if saying A = A0 makes a a copy of A0 or makes A refer to teh same thing as A0
    
    test more functions
    
    write more color_int methods
    
    make sure to get rid of all template reals where you could just cast to doubles
    
    better asserts in sphere intersection calculation
    
    fix all constructors to use initializer lists
    
    make all str() functions match
        mske them print out color too
        
    add const suffix to all methods that should be const
    
    make all initializers have const attributes via initialization lists
    
    do a length_squared method and replace all comparisons of length with length squared
        also do a distance squared method and replace all distance comparisons with distance_squared comparisons
    
*/

#include <vector>
#include <cmath>
#include <cstring>
#include <typeinfo>
#include <iostream>
#include <cassert>
#include <omp.h>
#include <png++/png.hpp>

#define QUIT exit(1);

#define INT(x) ((int)(x))
#define DOUBLE(x) ((double)(x))
#define BYTE(x) ((unsigned char)(x))

#define EPSILON (1e-10)
#define PI (3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709)
#define RAND_FLOAT (((double) rand()) / (double)RAND_MAX)

#define SQUARE(x) ((x)*(x))
#define MOD(a,b) ( ((((int)a)%((int)b))+((int)b))%((int)b) )
#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))
#define CLAMP_FLOAT(p) ( MIN(1.0,MAX(0.0,p)) )
#define CLAMP_INT(p) ( MIN(255,MAX(0,p)) )

#define PIXEL(r,g,b) ( png::rgb_pixel(CLAMP_INT(r),CLAMP_INT(g),CLAMP_INT(b)) )

#define FAST_2x2_DETERMINANT(a,b,c,d) ((a)*(d)-(b)*(c))
#define FAST_3_TUPLE_PROD(a,b,c) ((a)*(b)*(c))

#define DEBUG BUILD_DEBUG
#define ASSERT(x, msg) if (DEBUG && !(x)) { fprintf(stderr, "Assertion failed in %s(%d): %s\n", __FILE__, __LINE__, msg); assert(false); exit(1); }

#define TEST(x) cout << (#x) << ": " << x << endl;
#define PRINT(x) cout << x << endl;
#define DIVIDER cout << "=======================================================================================================================================================================" << endl;
#define NEWLINE cout << endl;

using std::to_string;
using std::string;
using std::vector;
using std::ostream;
using std::rand;
using std::cout;
using std::endl;

//static double world_x;
//static double world_y;
static double progress = 0;

/* Misc. Functions */

template <class type1, class type2>
bool same_type(const type1 &a, const type2 &b) {
    return typeid(a) == typeid(b);
}

template <class real>
real lerp(const real &a, const real &b, const double &t) {
    ASSERT(t>=0 && t<=1, (string("Interpolation factor (")+to_string(t)+") must be in interval [0.0,1.0]").c_str());
    return a*t+(1-t)*b;
}

/* Geometry Classes */

class Vertex { 
    public:
        double x, y, z;
        
        string str() const {
            string s;
            auto ans = string("Vertex(");
            s = to_string(this->x);
            s.resize(8);
            ans += s;
            ans += ", ";
            s = to_string(this->y);
            s.resize(8);
            ans += s;
            ans += ", ";
            s = to_string(this->z);
            s.resize(8);
            ans += s;
            ans += ")";
            return ans;
        }
        
        Vertex() {
            x = 0.0;
            y = 0.0;
            z = 0.0;
        }
        
        Vertex(const Vertex &other) {
            this->x = other.x;
            this->y = other.y;
            this->z = other.z;
        }
        
        Vertex(const double &x0, const double &y0, const double &z0) {
            x = x0;
            y = y0;
            z = z0;
        }
        
        template <class real>
        Vertex(const vector<real> &v) {
            ASSERT(v.size() == 3, (string("vector input is of size not equal to 3 (has size ")+to_string(v.size())+" )").c_str());
            x = v[0];
            y = v[1];
            z = v[2];
        }
        
        template <class real>
        Vertex(const real* &array) {
            ASSERT( sizeof(array)/sizeof(array[0]) == 3, (string("array input is of size not equal to 3 (has size ")+to_string(sizeof(array)/sizeof(array[0]))+")").c_str());
            x = array[0];
            y = array[1];
            z = array[2];
        }
        
        double& operator[] (const int &index) { 
            ASSERT(index < 3, (string("Index is greater than three (index = ")+to_string(index)).c_str());
            ASSERT(index >= 0, (string("Index is less than zero(index = ")+to_string(index)).c_str());
            if (index == 0) {
                return x;
            } else if (index == 1) {
                return y;
            } else if (index == 2) {
                return z;
            }
        }
        
        bool operator==(const Vertex &other) const {
            return this->x == other.x && this->y == other.y && this->z == other.z;
        }
};

class Vector3 : public Vertex { 
    public:
        using Vertex::x;
        using Vertex::y;
        using Vertex::z;
        
        string str() const {
            string s;
            auto ans = string("Vector3(");
            s = to_string(this->x);
            s.resize(8);
            ans += s;
            ans += ", ";
            s = to_string(this->y);
            s.resize(8);
            ans += s;
            ans += ", ";
            s = to_string(this->z);
            s.resize(8);
            ans += s;
            ans += ")";
            return ans;
        }
        
        Vector3() : Vertex() {}
        
        Vector3(const Vertex &other) : Vertex(other) {}
        
        Vector3(const Vector3 &other) {
            this->x = other.x;
            this->y = other.y;
            this->z = other.z;
        }
        
        Vector3(const double &x0, const double &y0, const double &z0) : Vertex(x0,y0,z0) {}
        
        template <class real>
        Vector3(const vector<real> &v) : Vertex(v){}
        
        template <class real>
        Vector3(const real* &array) : Vertex(array){}
        
        /* Assignment Operations with other Vector3 objects */
        
        Vector3& operator*=(const Vector3 &other){
            x *= other.x; 
            y *= other.y; 
            z *= other.z;
            return *this;
        }
        
        Vector3& operator+=(const Vector3 &other){
            x += other.x; 
            y += other.y; 
            z += other.z;
            return *this;
        }
        
        Vector3& operator-=(const Vector3 &other){
            x -= other.x; 
            y -= other.y; 
            z -= other.z;
            return *this;
        }
        
        Vector3& operator/=(const Vector3 &other){
            x /= other.x; 
            y /= other.y; 
            z /= other.z;
            return *this;
        }
        
        /* Assignment Operations with real values */
        
        template <class real>
        Vector3& operator*=(const real &num){
            x *= num; 
            y *= num; 
            z *= num;
            return *this;
        }
        
        template <class real>
        Vector3& operator+=(const real &num){
            x += num; 
            y += num; 
            z += num;
            return *this;
        }
        
        template <class real>
        Vector3& operator-=(const real &num){
            x -= num; 
            y -= num; 
            z -= num;
            return *this;
        }
        
        template <class real>
        Vector3& operator/=(const real &num){
            ASSERT(num != 0, (string("Division by zero (")+this->str()+"/"+to_string(num)+")").c_str());
            x /= num; 
            y /= num; 
            z /= num;
            return *this;
        }
        
        /* Methods and Functions */
        
        double prod() const {
            return x*y*z;
        }
        
        double sum() const {
            return x+y+z;
        }
        
        double length() const {
            return sqrt( SQUARE(x)+SQUARE(y)+SQUARE(z) );
        }
        
        Vector3& normalize(){
            const double length = this->length();
            x /= length;
            y /= length;
            z /= length;
            ASSERT( fabs(this->length() - 1.0) < EPSILON, (string("Length (")+to_string(this->length())+") after normalization is not equal to one ("+this->str()+")").c_str());
            return *this;
        }
};

class Matrix2 { 
    public:
        double data[4];
        
        string str() const {
            auto ans = string("Matrix2(");
            ans += "[" + to_string(data[0]) + ", " + to_string(data[1]) + "], ";
            ans += "[" + to_string(data[2]) + ", " + to_string(data[3]) + "]";
            ans += ")";
            return ans;
        }
        
        void clear(double clear_val){
            for(int i=0; i<4; i++){
                data[i] = clear_val;
            }
        }
        
        Matrix2(){
            this->clear(0);
        }
        
        Matrix2(const double &v00, const double &v01,
                const double &v10, const double &v11){
            data[0] = v00; data[1] = v01;
            data[2] = v10; data[3] = v11;
        }
        
        template <class real>
        Matrix2(const vector<real> &v){
            ASSERT(v.size() == 4, (string("vector input is of size not equal to 4 (has size ")+to_string(v.size())+" )").c_str());
            data[0] = v[0]; data[1] = v[1];
            data[2] = v[2]; data[3] = v[3];
        }
        
        template <class real>
        Matrix2(const real* &array){
            ASSERT( sizeof(array)/sizeof(array[0]) == 4, (string("array input is of size not equal to 4 (has size ")+to_string(sizeof(array)/sizeof(array[0]))+")").c_str());
            data[0] = array[0]; data[1] = array[1];
            data[2] = array[2]; data[3] = array[3];
        }
        
        double& operator() (const int &r, const int &c) { 
            ASSERT(r < 2, (string("Row is greater than 2 (index = ")+to_string(r)+")").c_str());
            ASSERT(r >= 0, (string("Row is less than zero(index = ")+to_string(r)+")").c_str());
            ASSERT(c < 2, (string("Column is greater than 2 (index = ")+to_string(c)+")").c_str());
            ASSERT(c >= 0, (string("Column is less than zero(index = ")+to_string(c)+")").c_str());
            return data[r*2+c];
        }
        
        void transpose() {
            double temp = data[1];
            data[1] = data[2];
            data[2] = temp;
        }
        
        double determinant() {
            return data[0]*data[3]-data[1]*data[2];
        }
        
        bool is_invertible() { 
            return this->determinant() != 0;
        }
        
        void invert() {
            double det = this->determinant();
            ASSERT(det != 0, "Zero determinant, i.e. this 2x2 matrix is not invertible");
            
            double temp = data[0];
            data[0] = -data[3]/det;
            data[3] = -temp/det;
            
            temp = data[1];
            data[1] = data[2]/det;
            data[2] = temp/det;
        }
        
        double prod(){
            return data[0]*data[1]*data[2]*data[3];
        }
        
        double sum(){
            return data[0]+data[1]+data[2]+data[3];
        }
};

class Matrix3 { 
    public:
        Vector3 r0;
        Vector3 r1;
        Vector3 r2;
        
        string str() const {
            auto ans = string("Matrix3(");
            ans += "[" + to_string(r0.x) + ", " + to_string(r0.y) + ", " + to_string(r0.z) + "], ";
            ans += "[" + to_string(r1.x) + ", " + to_string(r1.y) + ", " + to_string(r1.z) + "], ";
            ans += "[" + to_string(r2.x) + ", " + to_string(r2.y) + ", " + to_string(r2.z) + "]";
            ans += ")";
            return ans;
        }
        
        void clear(double clear_val){
            r0 = Vector3(clear_val,clear_val,clear_val);
            r1 = Vector3(clear_val,clear_val,clear_val);
            r2 = Vector3(clear_val,clear_val,clear_val);
        }
        
        Matrix3(){
            this->clear(0);
        }
        
        Matrix3(const Vertex &r0_, const Vertex &r1_, const Vertex &r2_){
            r0 = Vector3(r0_.x, r0_.y, r0_.z);
            r1 = Vector3(r1_.x, r1_.y, r1_.z);
            r2 = Vector3(r2_.x, r2_.y, r2_.z);
        }
        
        Matrix3(const Matrix3 &other) : r0(other.r0), r1(other.r1), r2(other.r2) {}
        
        Matrix3(const double &v00, const double &v01, const double &v02, 
                const double &v10, const double &v11, const double &v12, 
                const double &v20, const double &v21, const double &v22 ){
            r0 = Vector3(v00, v01, v02);
            r1 = Vector3(v10, v11, v12);
            r2 = Vector3(v20, v21, v22);
        }
        
        double& operator() (const int &r, const int &c) { 
            ASSERT(r < 3, (string("Row is greater than 3 (index = ")+to_string(r)+")").c_str());
            ASSERT(r >= 0, (string("Row is less than zero(index = ")+to_string(r)+")").c_str());
            ASSERT(c < 3, (string("Column is greater than 3 (index = ")+to_string(c)+")").c_str());
            ASSERT(c >= 0, (string("Column is less than zero(index = ")+to_string(c)+")").c_str());
            
            Vertex &row = r==0 ? r0 : (r == 1 ? r1 : r2);
            
            return row[c];
        }
        
        template <class real>
        Matrix3(const vector<real> &v){
            ASSERT(v.size() == 9, (string("vector input is of size not equal to 9 (has size ")+to_string(v.size())+" )").c_str());
            r0 = Vector3(v[0], v[1], v[2]);
            r1 = Vector3(v[3], v[4], v[5]);
            r2 = Vector3(v[6], v[7], v[8]);
        }
        
        template <class real>
        Matrix3(const real* &array){
            ASSERT( sizeof(array)/sizeof(array[0]) == 9, (string("array input is of size not equal to 9 (has size ")+to_string(sizeof(array)/sizeof(array[0]))+")").c_str());
            r0 = Vector3(array[0], array[1], array[2]);
            r1 = Vector3(array[3], array[4], array[5]);
            r2 = Vector3(array[6], array[7], array[8]);
        }
        
        void transpose() {
            double temp;
            for(int r=0; r<3; r++){
                for(int c=0; c<3; c++){
                    if (r < c) {
                        temp = (*this)(r,c);
                        (*this)(r,c) = (*this)(c,r);
                        (*this)(c,r) = temp;
                    }
                }
            }
        }
        
        double determinant() {
            const double &a = (*this)(0,0);
            const double &b = (*this)(0,1);
            const double &c = (*this)(0,2);
            const double &d = (*this)(1,0);
            const double &e = (*this)(1,1);
            const double &f = (*this)(1,2);
            const double &g = (*this)(2,0);
            const double &h = (*this)(2,1);
            const double &i = (*this)(2,2);
            
            return a*FAST_2x2_DETERMINANT(e,f,h,i)-b*FAST_2x2_DETERMINANT(d,f,g,i)+c*FAST_2x2_DETERMINANT(d,e,g,h);
        }
        
        bool is_invertible() { 
            return this->determinant() != 0;
        }
        
        void invert() { 
            const double &a = (*this)(0,0);
            const double &b = (*this)(0,1);
            const double &c = (*this)(0,2);
            const double &d = (*this)(1,0);
            const double &e = (*this)(1,1);
            const double &f = (*this)(1,2);
            const double &g = (*this)(2,0);
            const double &h = (*this)(2,1);
            const double &i = (*this)(2,2);
            
            Matrix3 adjugate = Matrix3();
            
            adjugate(0,0) =  FAST_2x2_DETERMINANT(e,f,h,i);
            adjugate(0,1) = -FAST_2x2_DETERMINANT(d,f,g,i);
            adjugate(0,2) =  FAST_2x2_DETERMINANT(d,e,g,h);
            adjugate(1,0) = -FAST_2x2_DETERMINANT(b,c,h,i);
            adjugate(1,1) =  FAST_2x2_DETERMINANT(a,c,g,i);
            adjugate(1,2) = -FAST_2x2_DETERMINANT(a,b,g,h);
            adjugate(2,0) =  FAST_2x2_DETERMINANT(b,c,e,f);
            adjugate(2,1) = -FAST_2x2_DETERMINANT(a,c,d,f);
            adjugate(2,2) =  FAST_2x2_DETERMINANT(a,b,d,e);
            
            adjugate.transpose();
            
            ASSERT(this->determinant() != 0, "Cannot invert matrix because it is singular/degenerate, i.e. the determinant is zero");
            
            adjugate.r0 /= this->determinant();
            adjugate.r1 /= this->determinant();
            adjugate.r2 /= this->determinant();
            
            this->r0 = Vector3(adjugate.r0);
            this->r1 = Vector3(adjugate.r1);
            this->r2 = Vector3(adjugate.r2);
        }
        
        double prod(){
            return r0.prod()*r1.prod()*r1.prod();
        }
        
        double sum(){
            return r0.sum()+r1.sum()+r1.sum();
        }
};

/* Vector3 Operations with other Vector3 objects */

double distance(const Vertex &a, const Vertex &b){
    return sqrt( SQUARE(a.x-b.x)+SQUARE(a.y-b.y)+SQUARE(a.z-b.z) );
}

Vector3 cross_product(const Vertex &a, const Vertex &b){
    return Vector3( a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x );
}

double dot_product(const Vertex &a, const Vertex &b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

bool are_parallel(const Vertex &a, const Vertex &b){
    static const Vector3 zero_vector = Vector3();
    return cross_product(a, b) == zero_vector;
}

bool are_orthogonal(const Vertex &a, const Vertex &b){
    return dot_product(a, b) == 0;
}

Vector3 operator-(const Vertex& vec) {
    return Vector3(-vec.x, -vec.y, -vec.z);
}

Vector3 operator*(const Vertex &a, const Vertex &b){
    return Vector3(a.x * b.x, a.y * b.y, a.z * b.z);
}

Vector3 operator+(const Vertex &a, const Vertex &b){
    return Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vector3 operator-(const Vertex &a, const Vertex &b){
    return Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vector3 operator/(const Vertex &a, const Vertex &b){
    ASSERT(b.x != 0 && b.y != 0 && b.z != 0, (string("Division by zero (")+a.str()+"/"+b.str()+")").c_str());
    return Vector3(a.x / b.x, a.y / b.y, a.z / b.z);
}

/* Vector3 Operations with double values (should work with ints and similar types also) */

Vector3 operator*(const Vertex &vec, const double &num){
    return Vector3(vec.x * num, vec.y * num, vec.z * num);
}

Vector3 operator+(const Vertex &vec, const double &num){
    return Vector3(vec.x + num, vec.y + num, vec.z + num);
}

Vector3 operator-(const Vertex &vec, const double &num){
    return Vector3(vec.x - num, vec.y - num, vec.z - num);
}

Vector3 operator/(const Vertex &vec, const double &num){
    ASSERT(num != 0, (string("Division by zero (")+vec.str()+"/"+to_string(num)+")").c_str());
    return Vector3(vec.x / num, vec.y / num, vec.z / num);
}

Vector3 operator*(const double &num, const Vertex &vec){
    return Vector3(num * vec.x, num * vec.y, num * vec.z);
}

Vector3 operator+(const double &num, const Vertex &vec){
    return Vector3(num + vec.x, num + vec.y, num + vec.z);
}

Vector3 operator-(const double &num, const Vertex &vec){
    return Vector3(num - vec.x, num - vec.y, num - vec.z);
}

Vector3 operator/(const double &num, const Vertex &vec){
    ASSERT(vec.x != 0 && vec.y != 0 && vec.z != 0, (string("Division by zero (")+to_string(num)+"/"+vec.str()+")").c_str());
    return Vector3(num / vec.x, num / vec.y, num / vec.z);
}

/* Operations between Vector3 objects and Matrix3 objects */

Vector3 operator*(const Matrix3 &mat, const Vertex &vec){
    return Vector3( (mat.r0*vec).sum(), (mat.r1*vec).sum(), (mat.r2*vec).sum() );
}

class Ray {
    public:
        Vertex position;
        Vector3 direction;
        
        string str() const {
            auto ans = string("Ray(position=");
            ans += position.str();
            ans += ", direction=";
            ans += direction.str();
            ans += ")";
            return ans;
        }
        
        Ray() : position(0,0,0), direction(0,0,-1) {
            ASSERT(direction.length() - 1 < EPSILON, "Ray direction is unnormalized");
        }
        
        Ray(const Vertex &position0, const Vector3 &direction0) : position(position0), direction(direction0) { 
            direction.normalize();
            ASSERT(direction.length() - 1 < EPSILON, "Ray direction is unnormalized");
        }
        
        Ray(const Ray &other) : position(other.position), direction(other.direction) {
            direction.normalize();
        }
        
        Ray(const double &position_x, const double &position_y, const double &position_z, 
            const double &direction_x, const double &direction_y, const double &direction_z ) : position(position_x, position_y, position_z), direction(direction_x, direction_y, direction_z) { 
            direction.normalize();
            ASSERT(direction.length() - 1 < EPSILON, "Ray direction is unnormalized");
        }
        
        Vector3 value_at(const double &t) const { 
            ASSERT(direction.length() - 1 < EPSILON, "Ray direction is unnormalized");
            return position+direction*t;
        }
        
        bool operator==(Ray &other) {
            this->direction.normalize();
            other.direction.normalize();
            return this->position == other.position && this->direction == other.direction;
        }
};

Ray operator-(const Ray& r) {
    return Ray(r.position, -r.direction);
}

class color_int { 
    public:
        unsigned char r,g,b;
        
        string str() const {
            auto ans = string("color_int(r=");
            ans += to_string(r);
            ans += ", g=";
            ans += to_string(g);
            ans += ", b=";
            ans += to_string(b);
            ans += ")";
            return ans;
        }
        color_int() : r(0), g(0), b(0) {} 
        
        color_int(const unsigned char &r0, const unsigned char &g0, const unsigned char &b0) : r(r0), g(g0), b(b0) {} 
        
        color_int(const color_int &other) : r(other.r), g(other.g), b(other.b){}
        
        static color_int random() {
            color_int ans( MOD(rand(), 256), MOD(rand(), 256), MOD(rand(), 256) );
            return ans;
        } 
        
        bool operator==(color_int &other) {
            return this->r == other.r && this->g == other.g && this->b == other.b;
        }
};

static const color_int COLOR_INT_WHITE = color_int((unsigned char)255,(unsigned char)255,(unsigned char)255);
static const color_int COLOR_INT_BLACK = color_int();

class light_point { 
    public:
        Vertex position;
        double red_intensity, green_intensity, blue_intensity;
        
        string str() const {
            auto ans = string("light_point(position=");
            ans += position.str();
            ans += ", red_intensity=";
            ans += to_string(red_intensity);
            ans += ", green_intensity=";
            ans += to_string(green_intensity);
            ans += ", blue_intensity=";
            ans += to_string(blue_intensity);
            ans += ")";
            return ans;
        }
        
        light_point() : position(0,0,0), red_intensity(1.0), green_intensity(1.0), blue_intensity(1.0) {}
        
        light_point(const Vertex &position0, const double &red_intensity0, const double &green_intensity0, const double &blue_intensity0) 
                  : position(position0), red_intensity(red_intensity0), green_intensity(green_intensity0), blue_intensity(blue_intensity0) {
            ASSERT(red_intensity>=0 && red_intensity<=1, (string("red_intensity (")+to_string(red_intensity)+") must be in interval [0.0,1.0]").c_str());
            ASSERT(green_intensity>=0 && green_intensity<=1, (string("green_intensity (")+to_string(green_intensity)+") must be in interval [0.0,1.0]").c_str());
            ASSERT(blue_intensity>=0 && blue_intensity<=1, (string("blue_intensity (")+to_string(blue_intensity)+") must be in interval [0.0,1.0]").c_str());
        }
        
        light_point(const light_point &other) 
                  : position(other.position), 
                    red_intensity(other.red_intensity),
                    green_intensity(other.green_intensity),
                    blue_intensity(other.blue_intensity) {
            ASSERT(red_intensity>=0 && red_intensity<=1, (string("red_intensity (")+to_string(red_intensity)+") must be in interval [0.0,1.0]").c_str());
            ASSERT(green_intensity>=0 && green_intensity<=1, (string("green_intensity (")+to_string(green_intensity)+") must be in interval [0.0,1.0]").c_str());
            ASSERT(blue_intensity>=0 && blue_intensity<=1, (string("blue_intensity (")+to_string(blue_intensity)+") must be in interval [0.0,1.0]").c_str());
        }
        
        light_point(const double &position_x, const double &position_y, const double &position_z, 
                    const double &red_intensity0, const double &green_intensity0, const double &blue_intensity0 ) 
                  : position(position_x, position_y, position_z), 
                    red_intensity(red_intensity0), green_intensity(green_intensity0), blue_intensity(blue_intensity0) {
            ASSERT(red_intensity>=0 && red_intensity<=1, (string("red_intensity (")+to_string(red_intensity)+") must be in interval [0.0,1.0]").c_str());
            ASSERT(green_intensity>=0 && green_intensity<=1, (string("green_intensity (")+to_string(green_intensity)+") must be in interval [0.0,1.0]").c_str());
            ASSERT(blue_intensity>=0 && blue_intensity<=1, (string("blue_intensity (")+to_string(blue_intensity)+") must be in interval [0.0,1.0]").c_str());
        }
        
        bool operator==(light_point &other) {
            return this->position == other.position && this->red_intensity == other.red_intensity && this->green_intensity == other.green_intensity && this->blue_intensity == other.blue_intensity;
        }
};

class GeometricPrimitive{
    public:
        color_int color;
        double reflection_factor;
        double transmission_factor;
//        double k_a; // ambient reflection coefficient
//        double k_d; // diffuse reflection coefficient
//        double k_s; // specular reflection coeffient
//        double k_r; // reflective coefficient, i.e. shininess constant
        virtual bool intersects_ray(const Ray &ray, double &t, Vector3 &normal) const = 0;
        virtual Vertex get_random_point() const = 0;
        
        GeometricPrimitive() : color(COLOR_INT_WHITE), reflection_factor(0.2), transmission_factor(0.2) {}
        
        GeometricPrimitive(const unsigned char &r0, const unsigned char &g0, const unsigned char &b0, 
                           const double &reflection_factor0, const double &transmission_factor0) 
                         : color(r0, g0, b0), reflection_factor(reflection_factor0), transmission_factor(transmission_factor0) {}
        
        GeometricPrimitive(const color_int &color0, const double &reflection_factor0, const double &transmission_factor0) 
                         : color(color0), reflection_factor(reflection_factor0), transmission_factor(transmission_factor0) {}

};

class Triangle : public GeometricPrimitive{ 
    public:
        Vertex A, B, C;
        Vector3 AB, AC;
        Vector3 normal;
        using GeometricPrimitive::color;
        using GeometricPrimitive::reflection_factor;
        using GeometricPrimitive::transmission_factor;
//        using GeometricPrimitive::k_a;
//        using GeometricPrimitive::k_d;
//        using GeometricPrimitive::k_s;
//        using GeometricPrimitive::k_r;
        
    public:
        string str() const {
            auto ans = string("Triangle(position=");
            ans += A.str();
            ans += ", ";
            ans += B.str();
            ans += ", ";
            ans += C.str();
            ans += ", normal=";
            ans += normal.str();
            ans += ", reflection_factor=";
            ans += to_string(reflection_factor);
            ans += ", transmission_factor=";
            ans += to_string(transmission_factor);
//            ans += ", k_a=";
//            ans += to_string(k_a);
//            ans += ", k_d=";
//            ans += to_string(k_d);
//            ans += ", k_s=";
//            ans += to_string(k_s);
//            ans += ", k_r=";
//            ans += to_string(k_r);
            ans += ")";
            return ans;
        }
        
        Triangle() : A(0,0,0), B(1,0,0), C(0,1,0), AB(B.x-A.x, B.y-A.y, B.z-A.z), AC(C.x-A.x, C.y-A.y, C.z-A.z), GeometricPrimitive() {
            normal = cross_product(AC, AB).normalize();
        }
        
        Triangle(const Triangle &other) : A(other.A), B(other.B), C(other.C), 
                                          normal(cross_product(AC, AB).normalize()), 
                                          AB(B.x-A.x, B.y-A.y, B.z-A.z), 
                                          AC(C.x-A.x, C.y-A.y, C.z-A.z), 
                                          GeometricPrimitive(other.color, other.reflection_factor, other.transmission_factor) {}
        
        Triangle(const Vertex &A0, const Vertex &B0, const Vertex &C0, const color_int &new_color, 
                 const double &reflection_factor0, const double &transmission_factor0
                ) 
               : A(A0), B(B0), C(C0), 
                 AB(B.x-A.x, B.y-A.y, B.z-A.z), AC(C.x-A.x, C.y-A.y, C.z-A.z), 
                 normal(cross_product(AC, AB).normalize()), 
                 GeometricPrimitive(new_color, reflection_factor0, transmission_factor0) {}
        
        Triangle(const double &A0_x, const double &A0_y, const double &A0_z, 
                 const double &B0_x, const double &B0_y, const double &B0_z, 
                 const double &C0_x, const double &C0_y, const double &C0_z, const color_int &new_color, 
                 const double &reflection_factor0, const double &transmission_factor0
                ) 
               : A(A0_x,A0_y,A0_z), B(B0_x,B0_y,B0_z), C(C0_x,C0_y,C0_z),
                 AB(B.x-A.x, B.y-A.y, B.z-A.z), AC(C.x-A.x, C.y-A.y, C.z-A.z),
                 normal(cross_product(AC, AB).normalize()), 
                 GeometricPrimitive(new_color, reflection_factor0, transmission_factor0) {}
        
        Triangle(const vector<Vertex> &v, const color_int &new_color,
                 const double &reflection_factor0, const double &transmission_factor0
                )  
               : A(v[0]), B(v[1]), C(v[2]), 
                 AB(B.x-A.x, B.y-A.y, B.z-A.z), AC(C.x-A.x, C.y-A.y, C.z-A.z), 
                 normal(cross_product(AC, AB).normalize()), 
                 GeometricPrimitive(new_color, reflection_factor0, transmission_factor0) {
            ASSERT(v.size() == 3, (string("vector input is of size not equal to 3 (has size ")+to_string(v.size())+" )").c_str());
        }
        
        Vertex& operator[] (const int &index) { 
            ASSERT(index < 3, (string("Index is greater than three (index = ")+to_string(index)).c_str());
            ASSERT(index >= 0, (string("Index is less than zero(index = ")+to_string(index)).c_str());
            if (index == 0) {
                return A;
            } else if (index == 1) {
                return B;
            } else if (index == 2) {
                return C;
            }
        }
        
        bool operator==(const Triangle &other) const {
            return (this->normal == other.normal || this->normal == -1*other.normal ) &&
                 ( ( this->A == other.A && this->B == other.B && this->C == other.C )
                || ( this->A == other.A && this->B == other.C && this->C == other.B )
                || ( this->A == other.B && this->B == other.A && this->C == other.C )
                || ( this->A == other.B && this->B == other.C && this->C == other.A )
                || ( this->A == other.C && this->B == other.B && this->C == other.A )
                || ( this->A == other.C && this->B == other.A && this->C == other.B ) );
        }
        
        bool intersects_ray(const Ray &ray, double &t, Vector3 &normal) const {
            // See for details: http://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
            Matrix3 mat = Matrix3( -ray.direction.x , B.x-A.x , C.x-A.x , 
                                   -ray.direction.y , B.y-A.y , C.y-A.y , 
                                   -ray.direction.z , B.z-A.z , C.z-A.z );
            
            ASSERT(mat.is_invertible(), "Matrix for triangle ray intersection is invertible"); 
            
            mat.invert();
            
            Vector3 vec = Vector3( ray.position.x-A.x , ray.position.y-A.y , ray.position.z-A.z );
            
            Vector3 tuv = mat*vec;
            
            t = tuv[0];
            const double &u = tuv[1];
            const double &v = tuv[2];
            
            normal = Vector3(this->normal);
            
            // We hit and edge if u+v == 1 or if v == 0 or u == 0 and an intersection takes place
            
            return 0 <= u && u<=1 && 0 <= v && v<=1 && u+v<=1 && t >=0 && fabs(t) > EPSILON;
        }
        
        Vertex get_random_point() const {
            return lerp(lerp(A,B,RAND_FLOAT),lerp(B,C, RAND_FLOAT), RAND_FLOAT);
        }
};

class Sphere : public GeometricPrimitive { 
    public:
        Vertex center;
        double radius;
        using GeometricPrimitive::color;
        using GeometricPrimitive::reflection_factor;
        using GeometricPrimitive::transmission_factor;
//        using GeometricPrimitive::k_a;
//        using GeometricPrimitive::k_d;
//        using GeometricPrimitive::k_s;
//        using GeometricPrimitive::k_r;
        
        string str() const {
            auto ans = string("Sphere(center=");
            ans += center.str();
            ans += ", radius=";
            ans += to_string(radius);
            ans += ", reflection_factor=";
            ans += to_string(reflection_factor);
            ans += ", transmission_factor=";
            ans += to_string(transmission_factor);
//            ans += ", k_a=";
//            ans += to_string(k_a);
//            ans += ", k_d=";
//            ans += to_string(k_d);
//            ans += ", k_s=";
//            ans += to_string(k_s);
//            ans += ", k_r=";
//            ans += to_string(k_r);
            ans += ")";
            return ans;
        }
        Sphere() : center(0,0,0), radius(1.0), GeometricPrimitive() {}
        
        Sphere(const Sphere &other) : center(other.center), radius(other.radius), GeometricPrimitive(other.color, other.reflection_factor, other.transmission_factor) {} 
        
        Sphere(const Vertex &center0, const double &radius0, const color_int &new_color, 
               const double &reflection_factor0, const double &transmission_factor0) 
             : center(center0), radius(radius0), GeometricPrimitive(new_color, reflection_factor0, transmission_factor0) {} 
        
        Sphere(const double &x0, const double &y0, const double &z0, const double &radius0, const color_int &new_color,
               const double &reflection_factor0, const double &transmission_factor0) 
             : center(x0,y0,z0), radius(radius0), GeometricPrimitive(new_color, reflection_factor0, transmission_factor0) {}
        
        bool operator==(const Sphere &other) const {
            return this->center == other.center && this->radius == other.radius;
        }
        
        bool intersects_ray(const Ray &ray, double &t, Vector3 &normal) const {
            // See for details: https://www.youtube.com/watch?v=ARB3e0kjVoY
            
            const Vector3 &P0 = ray.position;
            const Vector3 &P1 = ray.direction;
            const Vector3 C = Vector3(this->center);
            const double r2 = SQUARE(this->radius);
            
            const Vector3 P0_minus_C = P0-C;
            
            const double a = dot_product(P1, P1);
            const double b = 2*dot_product(P1,P0_minus_C);
            const double c = dot_product(P0_minus_C, P0_minus_C)-r2;
            
            const double discriminant = SQUARE(b)-4*a*c;
            
            if (discriminant < 0) { // no real roots
                return false;
            } else if (discriminant == 0){ // one solution, i.e. the ray is tangent to the sphere
                t = (-b+sqrt(discriminant))/(2*a);
                if (fabs(t) < EPSILON) { 
                    return false; 
                }
                normal = ray.value_at(t)-Vector3(this->center);
                normal.normalize();
                return true;
            } else { // two solutions
                const double sol1 = (-b+sqrt(discriminant))/(2*a);
                const double sol2 = (-b-sqrt(discriminant))/(2*a);
                if (sol1 < 0 && sol2 < 0){ // both negative implies sphere is behind
                    return false;
                } else if (sol1 >= 0 && sol2 >= 0) { // both positive implies we hit it and come out the other side, i.e. we hit the outside of the sphere
                    t = MIN(sol1, sol2);
                    normal = ray.value_at(t)-Vector3(this->center);
                    if (fabs(t) < EPSILON) { 
                        return false; 
                    }
                    normal.normalize();
                    return true;
                } else { // one positive and one negative, i.e. we're inside the sphere
                    if (sol1>=0 && sol2<0) {
                        t = sol1;
                    } else {
                        ASSERT(sol1<0 && sol2>=0, "Math sanity check failed");
                        t = sol2;
                    }
                    if (fabs(t) < EPSILON) { 
                        return false; 
                    }
                    normal = Vector3(this->center)-ray.value_at(t); // we have to use a different normal if we're bouncing off the inside
                    normal.normalize();
                    return true;
                }
            }
        }
        
        Vertex get_random_point() const {
            return Vector3(RAND_FLOAT-0.5, RAND_FLOAT-0.5, RAND_FLOAT-0.5).normalize()*radius+this->center;
        }
};

ostream& operator<<(ostream& os, const Vertex& obj) {
    os << obj.str();
    return os;
}

ostream& operator<<(ostream& os, const Vector3& obj) {
    os << obj.str();
    return os;
}

ostream& operator<<(ostream& os, const Matrix2& obj) {
    os << obj.str();
    return os;
}

ostream& operator<<(ostream& os, const Matrix3 &obj) {
    os << obj.str();
    return os;
}

ostream& operator<<(ostream& os, const Ray& obj) {
    os << obj.str();
    return os;
}

ostream& operator<<(ostream& os, const Triangle& obj) {
    os << obj.str();
    return os;
}

ostream& operator<<(ostream& os, const Sphere& obj) {
    os << obj.str();
    return os;
}

ostream& operator<<(ostream& os, const color_int& obj) {
    os << obj.str();
    return os;
}

/*  String Manipulation and File Reading */

void split(const string &line, const string &delimiter, vector<string> &vector_of_strings, bool skip_empty=false) {
    auto start = 0U;
    auto end = line.find(delimiter);
    while (end != std::string::npos){
        if (!skip_empty || start != end) {
            vector_of_strings.push_back( line.substr(start, end - start) );
        } 
        start = end + delimiter.length();
        end = line.find(delimiter, start);
    }
    vector_of_strings.push_back( line.substr(start, end) );
}

void write_file(const char* filename, string &input) { 
    std::ofstream out(filename);
    out << input;
    out.close();
}

void read_file(const char* filename, string &output) { 
    FILE* f = fopen(filename, "r");
    char* buffer;
    if (!f) {
        fprintf(stderr, "Unable to open %s for reading\n", filename);
        return;
    }
    
    fseek(f, 0, SEEK_END);
    int length = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    buffer = (char*)malloc(length+1);
    length = fread((void*)buffer, 1, length, f);
    fclose(f);
    buffer[length] = '\0';
    
    output = string(buffer);
    
    free(buffer);
}

void read_objects_from_file(char* const &geometric_object_file_name, vector<GeometricPrimitive*> &geometric_primitive_pointers) {
    cout << endl;
    cout << "Parsing " << string(geometric_object_file_name) << "." << endl;
    
    string file_contents;
    read_file(geometric_object_file_name, file_contents);
    
    vector<string> lines;
    split(file_contents, string("\n"), lines, false);
    
    int triangle_count = 0;
    int sphere_count = 0;
    
    for(int i = 0; i < lines.size(); i++){
        vector<string> current_line_elements;
        split(lines[i], string(" "), current_line_elements, true);
        if (current_line_elements[0] == string("triangle")){
            if (current_line_elements.size() < 10) {
                cout << "Could not parse line " << i << ": " << lines[i] << endl;
            }
            unsigned char red = atoi(current_line_elements[1].c_str());
            unsigned char green = atoi(current_line_elements[2].c_str());
            unsigned char blue = atoi(current_line_elements[3].c_str());
            
            double reflection_factor = atof(current_line_elements[4].c_str());
            double transmission_factor = atof(current_line_elements[5].c_str());
            
            double A_x = atof(current_line_elements[6].c_str());
            double A_y = atof(current_line_elements[7].c_str());
            double A_z = atof(current_line_elements[8].c_str());
            
            double B_x = atof(current_line_elements[9].c_str());
            double B_y = atof(current_line_elements[10].c_str());
            double B_z = atof(current_line_elements[11].c_str());
            
            double C_x = atof(current_line_elements[12].c_str());
            double C_y = atof(current_line_elements[13].c_str());
            double C_z = atof(current_line_elements[14].c_str());
            
            GeometricPrimitive* pointer = new Triangle(A_x,A_y,A_z,B_x,B_y,B_z,C_x,C_y,C_z, color_int(red,green,blue), reflection_factor, transmission_factor);
            geometric_primitive_pointers.push_back(pointer);
            triangle_count++;
        } else if (current_line_elements[0] == string("sphere")){
            if (current_line_elements.size() < 5) {
                cout << "Could not parse line " << i << ": " << lines[i] << endl;
            }
            unsigned char red = atoi(current_line_elements[1].c_str());
            unsigned char blue = atoi(current_line_elements[2].c_str());
            unsigned char green = atoi(current_line_elements[3].c_str());
            
            double reflection_factor = atof(current_line_elements[4].c_str());
            double transmission_factor = atof(current_line_elements[5].c_str());
            
            double center_x = atof(current_line_elements[6].c_str());
            double center_y = atof(current_line_elements[7].c_str());
            double center_z = atof(current_line_elements[8].c_str());
            double radius = atof(current_line_elements[9].c_str());
            
            GeometricPrimitive* pointer = new Sphere(center_x, center_y, center_z, radius, color_int(red,green,blue), reflection_factor, transmission_factor);
            geometric_primitive_pointers.push_back(pointer);
            sphere_count++;
        } 
    }
    
    cout << "Done parsing " << string(geometric_object_file_name) << ". Got " << sphere_count+triangle_count << " geometric object" 
         << ((sphere_count+triangle_count==1) ? "" : "s") << " ("
         << sphere_count << " sphere" << ((sphere_count==1) ? "" : "s") << " and " << triangle_count << " triangle" 
         << ((triangle_count==1) ? "" : "s") << ")." << endl;
    cout << endl;
}

bool shoot_ray(vector<GeometricPrimitive*> const &geometric_primitive_pointers, Ray const &ray, 
              double &min_dist, int &closest_geometric_primitive_pointer_index,
              Vector3 &final_normal, int const &depth, color_int &final_color){
    min_dist = DBL_MAX;
    closest_geometric_primitive_pointer_index = -1;
    for(int geometric_primitive_pointer_index=0; geometric_primitive_pointer_index<geometric_primitive_pointers.size(); geometric_primitive_pointer_index++) {
        double t;
        Vector3 normal;
        if (geometric_primitive_pointers[geometric_primitive_pointer_index]->intersects_ray(ray,t,normal)) {
            if (t<min_dist) {
                min_dist = t;
                final_normal = Vector3(normal);
                closest_geometric_primitive_pointer_index = geometric_primitive_pointer_index;
            }
        }
    }
    
    if (closest_geometric_primitive_pointer_index == -1) {
        return false;
    } else {
        double reflection_factor = geometric_primitive_pointers[closest_geometric_primitive_pointer_index]->reflection_factor;
        double transmission_factor = geometric_primitive_pointers[closest_geometric_primitive_pointer_index]->transmission_factor;
        ASSERT(reflection_factor+transmission_factor <= 1, "Reflection and transmission factors >= 1");
        if (depth > 1 /*|| reflection_factor==1.0*/){
            ASSERT(final_normal.length()-1<EPSILON, "Normal has length > 1");
            
            Vector3 reflected_ray_direction = ray.direction-2*dot_product(ray.direction, final_normal)*final_normal;
            Ray reflected_ray(ray.value_at(min_dist), reflected_ray_direction);
            
            double relefted_ray_min_dist;
            int reflected_ray_closest_geometric_primitive_pointer_index;
            Vector3 reflected_ray_hit_point_normal;
            color_int reflected_ray_final_color;
            
            if(!shoot_ray(geometric_primitive_pointers, reflected_ray, relefted_ray_min_dist, reflected_ray_closest_geometric_primitive_pointer_index, reflected_ray_hit_point_normal, depth-1, reflected_ray_final_color)){
                final_color.r = geometric_primitive_pointers[closest_geometric_primitive_pointer_index]->color.r;
                final_color.g = geometric_primitive_pointers[closest_geometric_primitive_pointer_index]->color.g;
                final_color.b = geometric_primitive_pointers[closest_geometric_primitive_pointer_index]->color.b;
                return true;
            }
            
            final_color.r = BYTE(
                                DOUBLE(geometric_primitive_pointers[closest_geometric_primitive_pointer_index]->color.r*(1-reflection_factor))
                              + DOUBLE(reflected_ray_final_color.r*reflection_factor)
                                );
            final_color.g = BYTE(
                                DOUBLE(geometric_primitive_pointers[closest_geometric_primitive_pointer_index]->color.g*(1-reflection_factor))
                              + DOUBLE(reflected_ray_final_color.g*reflection_factor)
                            );
            final_color.b = BYTE(
                                DOUBLE(geometric_primitive_pointers[closest_geometric_primitive_pointer_index]->color.b*(1-reflection_factor))
                              + DOUBLE(reflected_ray_final_color.b*reflection_factor)
                            );
            return true;
        } else {
            final_color.r = geometric_primitive_pointers[closest_geometric_primitive_pointer_index]->color.r;
            final_color.g = geometric_primitive_pointers[closest_geometric_primitive_pointer_index]->color.g;
            final_color.b = geometric_primitive_pointers[closest_geometric_primitive_pointer_index]->color.b;
            return true;
        }
    }
}

void ray_tracer(vector<GeometricPrimitive*> const &geometric_primitive_pointers, string const &output_file_name, 
                double const &theta_degrees,
                Vector3 const &eye_position,
                vector<GeometricPrimitive*> const &light_source_pointers,
                Vector3 const &look_direction,
                Vector3 const &up_direction,
                int const &I_dim,
                double const &viewport_distance, 
                int const &num_ray_bounces,
                int const &num_light_rays_per_pixel
                ) {
    
    ASSERT(theta_degrees < 90, "theta_degrees (half the angle of view) must be < 90");
    
    double theta = theta_degrees*(PI/180.0); // half the angle of view in radians
    Vector3 left_direction = cross_product(up_direction, look_direction);
    Vector3 right_direction = cross_product(look_direction, up_direction);
    ASSERT(right_direction == -left_direction, "Right and left directions are not opposites");
    Vector3 down_direction = -up_direction;
    
    ASSERT( look_direction.length()-1<EPSILON, "look_direction is unnormalized" );
    ASSERT( up_direction.length()-1<EPSILON, "up_direction is unnormalized" );
    ASSERT( down_direction.length()-1<EPSILON, "down_direction is unnormalized" );
    ASSERT( left_direction.length()-1<EPSILON, "left_direction is unnormalized" );
    ASSERT( right_direction.length()-1<EPSILON, "right_direction is unnormalized" );
    
    double dist_from_center = tan(theta)*viewport_distance;
    
    Vector3 upper_left_viewport_position  = Vector3(eye_position)
                                          + viewport_distance * look_direction
                                          + dist_from_center  * left_direction
                                          + dist_from_center  * up_direction;
    Vector3 upper_right_viewport_position = Vector3(eye_position)
                                          + viewport_distance * look_direction
                                          + dist_from_center  * right_direction
                                          + dist_from_center  * up_direction;
    Vector3 lower_left_viewport_position  = Vector3(eye_position)
                                          + viewport_distance * look_direction
                                          + dist_from_center  * left_direction
                                          + dist_from_center  * down_direction;
    Vector3 lower_right_viewport_position = Vector3(eye_position)
                                          + viewport_distance * look_direction
                                          + dist_from_center  * right_direction
                                          + dist_from_center  * down_direction;
    
    png::image< png::rgb_pixel > I(I_dim, I_dim);
    
    double delta = distance(upper_left_viewport_position, upper_right_viewport_position)/(float)I_dim;
    
    #pragma omp parallel for
    for (int y = 0; y < I_dim; ++y) {
        progress = MAX(DOUBLE(y)/DOUBLE(I_dim), progress);
        PRINT( string("    Progress: ")+to_string(progress) );
        for (int x = 0; x < I_dim; ++x) {
//            world_x = x;
//            world_y = y;
            double final_red = 0.0;
            double final_green = 0.0;
            double final_blue = 0.0;
            for (int ray_num = 0; ray_num < num_light_rays_per_pixel; ray_num++) {
                static const double w1 = 0.5;
                static const double w2 = 0.5;
                static const double w3 = 0.5;
                
                Vector3 upper_left_pixel_direction  = upper_left_viewport_position  - Vector3(eye_position)
                                                    + (x  )*delta * right_direction
                                                    + (y  )*delta * down_direction;
                Vector3 upper_right_pixel_direction = upper_left_viewport_position  - Vector3(eye_position)
                                                    + (x+1)*delta * right_direction
                                                    + (y  )*delta * down_direction;
                Vector3 lower_left_pixel_direction  = upper_left_viewport_position  - Vector3(eye_position)
                                                    + (x  )*delta * right_direction
                                                    + (y+1)*delta * down_direction;
                Vector3 lower_right_pixel_direction = upper_left_viewport_position  - Vector3(eye_position)
                                                    + (x+1)*delta * right_direction
                                                    + (y+1)*delta * down_direction;
                
                Vector3 ray_direction = lerp(lerp(upper_left_pixel_direction, upper_right_pixel_direction, w1),lerp(lower_left_pixel_direction, lower_right_pixel_direction, w2), w3);
                Ray ray(eye_position, ray_direction);
                
                double min_dist_to_eye;
                int closest_geometric_primitive_to_eye_pointer_index;
                Vector3 normal;
                color_int object_color = color_int();
                
                if (shoot_ray(geometric_primitive_pointers, ray, min_dist_to_eye, closest_geometric_primitive_to_eye_pointer_index, normal, num_ray_bounces, object_color)) { // shoot ray to see if it hits anything
                    double min_dist_to_light;
                    int closest_geometric_primitive_to_light_pointer_index;
                    Vector3 ray_obj_intersection_point = ray.value_at(min_dist_to_eye);
                    
//                    static const double i_a_r = 255.0;
//                    static const double i_a_g = 255.0;
//                    static const double i_a_b = 255.0;
                    
//                    double red   = geometric_primitive_pointers[closest_geometric_primitive_to_eye_pointer_index]->k_a * i_a_r;
//                    double green = geometric_primitive_pointers[closest_geometric_primitive_to_eye_pointer_index]->k_a * i_a_g;
//                    double blue  = geometric_primitive_pointers[closest_geometric_primitive_to_eye_pointer_index]->k_a * i_a_b;
                    
                    double red   = 0;
                    double green = 0;
                    double blue  = 0;
                    
                    for (int light_source_pointer_index = 0; light_source_pointer_index < light_source_pointers.size(); light_source_pointer_index++) {
                        auto light_ray_position = light_source_pointers[light_source_pointer_index]->get_random_point();
                        Vector3 light_ray_direction(ray_obj_intersection_point - light_ray_position);
                        Ray light_ray(light_ray_position, light_ray_direction);
                        Vector3 dummy_normal;
                        color_int dummy_color = color_int();
                        
                        shoot_ray(geometric_primitive_pointers, light_ray, min_dist_to_light, closest_geometric_primitive_to_light_pointer_index, dummy_normal, 1, dummy_color);
                        
                        if ( (light_ray.value_at(min_dist_to_light)-ray.value_at(min_dist_to_eye)).length() < EPSILON ) { // check if the object is in shadow
                            
//                            double i_d_r   = DOUBLE(geometric_primitive_pointers[closest_geometric_primitive_to_eye_pointer_index]->color.r);
//                            double i_d_g = DOUBLE(geometric_primitive_pointers[closest_geometric_primitive_to_eye_pointer_index]->color.g);
//                            double i_d_b  = DOUBLE(geometric_primitive_pointers[closest_geometric_primitive_to_eye_pointer_index]->color.b);
                           
//                            static const double i_s_r = 255.0;
//                            static const double i_s_g = 255.0;
//                            static const double i_s_b = 255.0;
                            
//                            double &k_d  = geometric_primitive_pointers[closest_geometric_primitive_to_eye_pointer_index]->k_d;
//                            double &k_s  = geometric_primitive_pointers[closest_geometric_primitive_to_eye_pointer_index]->k_s;
//                            double &k_r  = geometric_primitive_pointers[closest_geometric_primitive_to_eye_pointer_index]->k_r;
                            
                            ASSERT(fabs(normal.length()-1)<EPSILON, "Normal must be normalized");
                            ASSERT(fabs(ray.direction.length()-1)<EPSILON, "Ray direction must be normalized");
                            
//                            double Lm_dot_N = dot_product(-light_ray_direction,normal);
//                            double Rm_dot_V = dot_product(2.0*Lm_dot_N*normal+light_ray_direction,-ray_direction);
                            
//                            red   += k_d*(Lm_dot_N*i_d_r) + k_s*pow(Rm_dot_V,k_r)*i_s_r;
//                            green += k_d*(Lm_dot_N*i_d_g) + k_s*pow(Rm_dot_V,k_r)*i_s_g;
//                            blue  += k_d*(Lm_dot_N*i_d_b) + k_s*pow(Rm_dot_V,k_r)*i_s_b;
                            
//                            red   += i_d_r;
//                            green += i_d_g;
//                            blue  += i_d_b;
                            
                            red   += object_color.r;
                            green += object_color.g;
                            blue  += object_color.b;
                            break; // don't double light z
                        }
                    }
                    final_red += red;
                    final_green += green;
                    final_blue += blue;
                }
            }
            
            final_red   /= DOUBLE(num_light_rays_per_pixel);
            final_green /= DOUBLE(num_light_rays_per_pixel);
            final_blue  /= DOUBLE(num_light_rays_per_pixel);
            
            I[y][x] = PIXEL(INT(final_red), INT(final_green), INT(final_blue));
        }
    }
    I.write( output_file_name.c_str() );
}

