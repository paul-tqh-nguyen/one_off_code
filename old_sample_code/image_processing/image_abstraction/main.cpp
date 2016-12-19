
/*

TODO:
    add command line parsing code to util.h
    
    add comments that we are using a .settings file since this method has a large number of parameters
    
    parallelize the code
    
    convert everything from doubles back to bytes
    
    make the sobel filter separable in get_gradient
    
    make everything parallel
    
    remove commented out code
    
    change gradient normalizatoin after ETF into a func
    
    Make readme reflect that we're using floats instead of doubles and byes instead of ints and even shorts sometimes just because we want speed, notice that we implemented the accelerated version of the code. 

*/

#include <chrono>
#include <sstream>
#include "../../util/array.h"
#include "../../util/util.h"

#define X_COORD 0
#define Y_COORD 1
#define Z_COORD 2

#define R_COORD 0
#define G_COORD 1
#define B_COORD 2

using std::cout;
using std::endl;
using std::string;

void getFBL(const Array<double> &I_0, const Array<double> &etf, const double &delta_m, const double &delta_n, const double &sigma_e,
            const double &r_e, const double &sigma_g, const double &r_g, const Array<double> &g, const int &numFBLIterations,
            Array<double> &FBL) { 
    int height0 = I_0.height();
    int width0 = I_0.width();
    auto I = Array<double>(I_0.sizes);
    for (int y = 0; y < I_0.height(); y++){
        for (int x = 0; x < I_0.width(); x++){
            for (int z = 0; z < I_0.channels(); z++){
                I(y,x,z) = DOUBLE( I_0(y,x,z) );
            }
        }
    }
    png::image< png::rgba_pixel > output_image(I.width(), I.height());
    auto Ce = Array<double>(I_0.sizes);
    auto Cg = Array<double>(I_0.sizes);
    int S = ceil(sigma_e*2);
    int T = ceil(sigma_g*2);
    
    for(int i = 0; i < numFBLIterations; i++){
        Ce.clear();
        for (int y = 0; y < I.height(); y++){
            for (int x = 0; x < I.width(); x++){
                double sum_weights = 0.0;
                for (int s_sign = -1; s_sign==-1 || s_sign==1 ; s_sign+=2){
                    double c_x[] = {DOUBLE(x), DOUBLE(y)};
                    for(int s_abs = 0; s_abs <= S; s_abs++){
                        if (s_abs == 0 && s_sign == 1){
                            continue;
                        }
                        int s = s_sign*s_abs;
                        
                        int c_x_x_orig = round(c_x[X_COORD]);
                        int c_x_y_orig = round(c_x[Y_COORD]);
                        
                        vector<double> c_xs = vector<double>(I.channels());
                        for (int z = 0; z < I.channels(); z++) {
                            c_xs[z] = I(c_x_y_orig, c_x_x_orig, z);
                        }
                        
                        double temp = 0.0;
                        for (int z = 0; z < I.channels(); z++) {
                            temp += SQUARE(I(y,x,z)-c_xs[z]);
                        }
                        double weight = G(s,sigma_e) * G(sqrt(temp), r_e);
                        
                        for (int z = 0; z < I.channels(); z++) {
                            Ce(y,x,z) += c_xs[z] * weight;
                        }
                        
                        sum_weights += weight;
                        
                        //if (c_x_x_orig == 0 && c_x_y_orig == 0) { printf("%d %d %d %d %f %f\n", x, y, c_x_x_orig, c_x_y_orig, etf(c_x_y_orig,c_x_x_orig,X_COORD), etf(c_x_y_orig,c_x_x_orig,Y_COORD)); }
                        c_x[X_COORD] += s_sign*delta_m*etf(c_x_y_orig,c_x_x_orig,X_COORD);
                        c_x[Y_COORD] += s_sign*delta_m*etf(c_x_y_orig,c_x_x_orig,Y_COORD);
                        
                        if ( round(c_x[X_COORD]) < 0 || round(c_x[X_COORD]) > I.width()-1 || round(c_x[Y_COORD]) < 0 || round(c_x[Y_COORD]) > I.height()-1){
                            break;
                        }
                    }
                }
                for (int channel = 0; channel < I.channels(); channel++) {
                    Ce(y,x,channel) /= sum_weights;
                }
            }
        }
        I.assign(Ce);
//        for (int y = 0; y < I.height(); y++) {
//            for (int x = 0; x < I.width(); x++) {
//                output_image[y][x] = PIXEL(I(y,x,R_COORD), I(y,x,G_COORD), I(y,x,B_COORD));
//            }
//        }
//        output_image.write(string("asd")+to_string(numFBLIterations)+"C_e.png");
        
        Cg.clear();
        for (int y = 0; y < I.height() ; y++){
            for (int x = 0; x < I.width() ; x++){
                double sum_weights = 0.0;
                for (int t_sign = -1; t_sign==-1 || t_sign==1 ; t_sign+=2){
                    double l_x[] = {DOUBLE(x), DOUBLE(y)};
                    for(int t_abs = 0; t_abs <= T; t_abs++){
                        if (t_abs == 0 && t_sign == 1){
                            continue;
                        }
                        
                        int t = t_sign*t_abs;
                        
                        vector<double> l_xt(I.channels());
                        for (int z = 0; z < I.channels(); z++) {
                            l_xt[z] = I(round(l_x[Y_COORD]), round(l_x[X_COORD]), z);
                        }
                        
                        double temp = 0.0;
                        for (int z = 0; z < I.channels(); z++) {
                            temp += SQUARE(I(y,x,z)-l_xt[z]);
                        }
                        double weight = G(t,sigma_g) * G(sqrt(temp), r_g);
                        
                        for (int z = 0; z < I.channels(); z++) {
                            Cg(y,x,z) += l_xt[z] * weight;
                        }
                        sum_weights += weight;
                        
                        int lx_x_orig = round(l_x[X_COORD]);
                        int lx_y_orig = round(l_x[Y_COORD]);
                        
                        l_x[X_COORD] += t_sign*delta_n*g(lx_y_orig,lx_x_orig,X_COORD);
                        l_x[Y_COORD] += t_sign*delta_n*g(lx_y_orig,lx_x_orig,Y_COORD);

                        if ( round(l_x[X_COORD]) < 0 || round(l_x[X_COORD]) > I.width()-1 || round(l_x[Y_COORD]) < 0 || round(l_x[Y_COORD]) > I.height()-1){
                            break;
                        }
                    }
                }
                for (int channel = 0; channel < I.channels(); channel++) { 
                    Cg(y,x,channel) /= sum_weights;
                }
            }
        }
        I.assign(Cg);
//        for (int y = 0; y < I.height(); y++) {
//            for (int x = 0; x < I.width(); x++) {
//                output_image[y][x] = PIXEL(I(y,x,R_COORD), I(y,x,G_COORD), I(y,x,B_COORD));
//            }
//        }
//        output_image.write(string("asd")+to_string(numFBLIterations)+"C_g.png");
    }
    
    FBL.assign(I);
}

void filterFDOG(Array<double> I_grayscale, const Array<double> &etf, const double &delta_m, const double &delta_n, const double &sigma_m, const double &sigma_c, const double &p, const Array<double> &g, const double &binary_threshold, const int &numFDOGIterations, const int &kernel_dim, const double &blur_kernel_sigma, Array<double> &H){
    double sigma_s = 1.6*sigma_c;
    H.resize(vector<int>{I_grayscale.height(),I_grayscale.width()});
    Array<double> H_g = Array<double>(H.sizes); // For Separable implementation
    int S = ceil(2*sigma_m); 
    int T = ceil(2*sigma_s);
    
    auto blur_kernel = Array<double>(vector<int>{kernel_dim, kernel_dim});
    int center_coord = kernel_dim/2;
    for(int r = 0; r < kernel_dim; r++){
        for(int c = 0; c <=r; c++){
            blur_kernel(r,c) = 1.0/(2*PI*blur_kernel_sigma*blur_kernel_sigma) * exp(-((r-center_coord)*(r-center_coord)+(c-center_coord)*(c-center_coord))/(2*blur_kernel_sigma*blur_kernel_sigma));
            blur_kernel(c,r) = blur_kernel(r,c);
        }
    }
    blur_kernel.normalize();
    
    for(int iteration = 0; iteration < numFDOGIterations; iteration++){
        
        auto temp = Array<double>(I_grayscale);
        convolution_filter(blur_kernel, temp, I_grayscale);
        
        H.clear();
        H_g.clear();
        for (int y = 0; y < I_grayscale.height() ; y++){
            for (int x = 0; x < I_grayscale.width() ; x++){
                for(int t_sign = -1; t_sign==-1 || t_sign==1 ; t_sign+=2){
                    for(int t_abs = 0; t_abs <= T; t_abs++){
                        if (t_abs == 0 && t_sign == 1){
                            continue;
                        }
                        int t = t_sign*t_abs;
                        int l_xst_xcoordinate = round(x+t*delta_n*g(y,x,X_COORD));
                        int l_xst_ycoordinate = round(y+t*delta_n*g(y,x,Y_COORD));
                        if ( l_xst_xcoordinate < 0 || l_xst_xcoordinate > I_grayscale.width()-1 || l_xst_ycoordinate < 0 || l_xst_ycoordinate > I_grayscale.height()-1){
                            continue;
                        }
                        double l_xst = I_grayscale( l_xst_ycoordinate , l_xst_xcoordinate);
                        
                        H_g(y,x) += l_xst * (G(t,sigma_c)-p*G(t,sigma_s));
                    }
                }
            }
        }
        for (int y = 0; y < I_grayscale.height() ; y++){
            for (int x = 0; x < I_grayscale.width() ; x++){
                for (int s_sign = -1; s_sign==-1 || s_sign==1 ; s_sign+=2){
                    double c_x[] = {(double)x, (double)y};
                    for(int s_abs = 0; s_abs <= S; s_abs++){
                        if (s_abs == 0 && s_sign == 1){
                            continue;
                        }
                        int s = s_sign*s_abs;
                        
                        int c_x_x_orig = round(c_x[X_COORD]);
                        int c_x_y_orig = round(c_x[Y_COORD]);
                        
                        H(y,x) += H_g(c_x_y_orig,c_x_x_orig) * G(s,sigma_m);
                        
                        c_x[X_COORD] += s_sign*delta_m*etf(c_x_y_orig,c_x_x_orig,X_COORD);
                        c_x[Y_COORD] += s_sign*delta_m*etf(c_x_y_orig,c_x_x_orig,Y_COORD);
                        
                        if ( round(c_x[X_COORD]) < 0 || round(c_x[X_COORD]) > I_grayscale.width()-1 || round(c_x[Y_COORD]) < 0 || round(c_x[Y_COORD]) > I_grayscale.height()-1){
                            break;
                        }
                    }
                }
            }
        }
        
        for (int r = 0; r < I_grayscale.height() ; r++){
            for (int c = 0; c < I_grayscale.width() ; c++){
                H(r,c) = !(H(r,c) < 0 && 1+tanh(H(r,c)) < binary_threshold);
                if (iteration != numFDOGIterations-1){
                    I_grayscale(r,c) *= H(r,c);
                }
            }
        }
    }
}

void getETF(const Array<double> &I_grayscale, const double &mu, const int &numETFIterations, const Array<double> &g, Array<double> &etf) {
    Array<double> t = Array<double>(vector<int>{I_grayscale.height(),I_grayscale.width(),2});
    for (int y = 0; y < I_grayscale.height() ; y++){
    	for (int x = 0; x < I_grayscale.width() ; x++){
    	    t(y,x,X_COORD) = -g(y,x,Y_COORD);
    	    t(y,x,Y_COORD) = g(y,x,X_COORD);
    	}
	}
    
    etf.resize(t.sizes);
    etf.clear(0);
	
    for (int i = 0; i < numETFIterations*2 ; i++){ // Twice as many iterations since we're using a separable filter now and have to do two directions
        for (int x_r = 0; x_r < I_grayscale.height() ; x_r++){
            for (int x_c = 0; x_c < I_grayscale.width() ; x_c++){
                //*/
                int maxDimensionLength = I_grayscale.width();
                int x = x_c;
                if (IS_ODD(numETFIterations)){ // Do the x-direction on the odd iterations
                    // We have y corresponding to y_c and and x_r corresponding to y_r
                    maxDimensionLength = I_grayscale.width();
                    x = x_c;
                } else { // Do the y-direction on the even iterations
                    maxDimensionLength = I_grayscale.height();
                    x = x_r;
                }
                for (int y = MAX(0,x-mu); y < MIN(x+mu,maxDimensionLength) ; y++){
                    int y_c;
                    int y_r;
                    if (IS_ODD(numETFIterations)){ // Do the x-direction on the odd iterations
                        y_c = y;
                        y_r = x_r;
                    } else { // Do the y-direction on the even iterations
                        y_c = x_c;
                        y_r = y;
                    }
                    bool w_s = SQUARE(x_r-y_r)+SQUARE(x_c-y_c) < mu * mu;
                    
                    double g_hat_x = sqrt( SQUARE(g(x_r,x_c,X_COORD)) +SQUARE(g(x_r,x_c,Y_COORD)) );
                    double g_hat_y = sqrt( SQUARE(g(y_r,y_c,X_COORD)) +SQUARE(g(y_r,y_c,Y_COORD)) );
                    
                    double w_m = (g_hat_y-g_hat_x+1)/2;
                    
                    double tangentDotProduct = t(x_r,x_c,X_COORD)*t(y_r,y_c,X_COORD)+t(x_r,x_c,Y_COORD)*t(y_r,y_c,Y_COORD);
                    
                    double w_d = fabs(tangentDotProduct);
                    
                    double phi = (tangentDotProduct>0)*2-1;
                    
                    etf(x_r,x_c,X_COORD) += t(y_r,y_c,X_COORD)*phi*w_s*w_m*w_d;
                    etf(x_r,x_c,Y_COORD) += t(y_r,y_c,Y_COORD)*phi*w_s*w_m*w_d;
                    
//                    DIVIDER;
//                    TEST( etf(x_r,x_c,X_COORD) );
//                    TEST( etf(x_r,x_c,Y_COORD) );
//                    TEST( t(y_r,y_c,X_COORD) );
//                    TEST( t(y_r,y_c,Y_COORD) );
                }
            }
        }
        // Normalize the etf
        //double etf_max_length = 0.0;
        //for (int y = 0; y < I_grayscale.height() ; y++){ // Find Max
        //	for (int x = 0; x < I_grayscale.width() ; x++){
        //	    etf_max_length = MAX(etf_max_length, sqrt(etf(y,x,X_COORD)*etf(y,x,X_COORD) + etf(y,x,Y_COORD)*etf(y,x,Y_COORD)));
        //	}
	    //}
        for (int y = 0; y < I_grayscale.height() ; y++){ 
        	for (int x = 0; x < I_grayscale.width() ; x++){
                double etf_current_length = sqrt(etf(y,x,X_COORD)*etf(y,x,X_COORD) + etf(y,x,Y_COORD)*etf(y,x,Y_COORD));
                if (etf_current_length == 0) { etf_current_length++; }
        	    etf(y,x,X_COORD) = etf(y,x,X_COORD)/etf_current_length;
        	    etf(y,x,Y_COORD) = etf(y,x,Y_COORD)/etf_current_length;
        	    
//        	    if ((x == 5 && y == 0) || (x==213 && y==210)) {
//            	    TEST( etf(y,x,X_COORD) )
//            	    TEST( etf(y,x,Y_COORD) )
//        	    }
                //if (x == 0 && y == 0) { printf("etf(%d, %d) = %f %f\n", x, y, etf(y, x, X_COORD), etf(y, x, Y_COORD)); }
        	}
	    }
	    
        for (int y = 0; y < etf.height() ; y++){ 
        	for (int x = 0; x < etf.width() ; x++){
//        	    TEST( etf(y,x,X_COORD) )
//        	    TEST( etf(y,x,Y_COORD) )
//        	    if (etf(y,x,X_COORD) == 0 && etf(y,x,Y_COORD) == 0) {
//            	    DIVIDER;
//            	    TEST( x )
//            	    TEST( y )
//            	    QUIT;
//        	    }
        	}
    	}
        //PRINT( etf.str() );
        // why are certain parts of the etf zero?
        /////////////////////////////////////// PROBLEM ///////////////////////////////////////////////////////////////
        
	    // Prep for next iteration only if we are not on the last iteration
        if (i != numETFIterations*2-1){
            for (int z = 0; z < t.channels(); z++) {
                for (int y = 0; y < t.height(); y++) {
                    for (int x = 0; x < t.width(); x++) {
                        t(y,x,z) = etf(y,x,z);
                    }
                }
            }
            etf.clear();
        }
    }
//    PRINT( etf.str() );
    //printf("etf(0, 0) = %f %f\n", etf(0, 0, X_COORD), etf(0, 0, Y_COORD));
}

void getGradient(const Array<double> &I_grayscale, Array<double> &g) { // Get the gradient map g = (g_x , g_y)
    
    ASSERT(I_grayscale.sizes.size()==2, "I_grayscale must be a 2D image");
    
    // x_kernel: 
    // [[ -1, 0, 1 ], 
    //  [ -2, 0, 2 ], 
    //  [ -1, 0, 1 ]] 
    auto x_kernel = Array<double>(vector<int>{3, 3});
    x_kernel(0,0) = -1;
    x_kernel(0,1) = 0;
    x_kernel(0,2) = 1;
    x_kernel(1,0) = -2;
    x_kernel(1,1) = 0;
    x_kernel(1,2) = 2;
    x_kernel(2,0) = -1;
    x_kernel(2,1) = 0;
    x_kernel(2,2) = 1;
    
    // y_kernel:
    // [[ -1, -2, -1 ],
    //  [  0,  0,  0 ],
    //  [  1,  2,  1 ]]
    auto y_kernel = Array<double>(vector<int>{3, 3});
    y_kernel(0,0) = -1;
    y_kernel(0,1) = -2;
    y_kernel(0,2) = -1;
    y_kernel(1,0) = 0;
    y_kernel(1,1) = 0;
    y_kernel(1,2) = 0;
    y_kernel(2,0) = 1;
    y_kernel(2,1) = 2;
    y_kernel(2,2) = 1;
    
    auto temp = Array<short>();
    g.resize(vector<int>{I_grayscale.height(), I_grayscale.width(), 2});
    
    convolution_filter(x_kernel,I_grayscale,temp);
    for (int y = 0; y < I_grayscale.height() ; y++){
    	for (int x = 0; x < I_grayscale.width() ; x++){
	        g(y,x,X_COORD) = temp(y,x);
//	        if ((y-1>0) && (x-1>0) && (x+1<I_grayscale.width()) && (y+1<I_grayscale.height())){
//	            TEST( FLOAT(I_grayscale(y-1,x-1)))
//	            TEST( FLOAT(I_grayscale(y-1,x)))
//	            TEST( FLOAT(I_grayscale(y-1,x+1)))
//	            TEST( FLOAT(I_grayscale(y+1,x-1)))
//	            TEST( FLOAT(I_grayscale(y+1,x)))
//	            TEST( FLOAT(I_grayscale(y+1,x+1)))
//	        }
//	        TEST( g(y,x,X_COORD) );
    	}
	}
    
    convolution_filter(y_kernel,I_grayscale,temp);
    for (int y = 0; y < I_grayscale.height() ; y++){
    	for (int x = 0; x < I_grayscale.width() ; x++){
	        g(y,x,Y_COORD) = temp(y,x);
//	        if (g(y,x,Y_COORD) == 0 && g(y,x,X_COORD) == 0) {
//	            TEST( y );
//	            TEST( x );
//	            TEST( g(y,x,X_COORD) );
//	            TEST( g(y,x,Y_COORD) );
//	        }
//	        NEWLINE;
//	        NEWLINE;
//	        NEWLINE;
//	        TEST( (short)(1293.4213) );
//	        TEST( (double)(165.13) );
//	        TEST( (short)(1293.4213) * (double)(165.13) );
//	        TEST( typeid( (short)(1293.4213)*(double)(165.13) ).name() );
//	        TEST( UCHAR_MAX );
//	        TEST( CHAR_MAX );
//	        TEST( CHAR_MIN );
//	        TEST( SHRT_MAX );
//	        TEST( SHRT_MIN );
//            TEST( FLOAT( g(y,x,X_COORD) ));
//            TEST( FLOAT( g(y,x,Y_COORD) ));
//            TEST( INT(sizeof(unsigned short)) );
//            TEST( INT(sizeof(char)) );
//            TEST( INT(sizeof(short)) );
//            TEST( INT(sizeof(unsigned char)) );
    	}
	}
    
//    TEST( g.str() );
    
    // Do not normalize gradient until after ETF
}

void getParametersFromFile(const char* const &settings_file_name, 
    double &mu, int &numETFIterations, 
    double &delta_m, double &delta_n, double &sigma_m, double &sigma_c, double &p, 
    double &binary_threshold, int &numFDOGIterations, int &kernel_dim, double &blur_kernel_sigma,
    double &sigma_e, double &r_e, double &sigma_g, double &r_g, int &numFBLIterations){
    
    string file_contents;
    read_file(settings_file_name, file_contents);
    vector<string> lines;
    split(file_contents, string("\n"), lines, false);
    
    for(int i = 0; i < lines.size(); i++){
        vector<string> line_elements;
        split(lines[i], string(" "), line_elements, true);
        if (line_elements.size() < 2) {
            PRINT( string("line ")+to_string(i+1)+" is invalid." );
            continue;
        }
        string varName = line_elements[0];
        if (varName == string("mu")){
            mu = atof(line_elements[1].c_str());
        } else if (varName ==  string("numETFIterations")){
            numETFIterations = atoi(line_elements[1].c_str());
        } else if (varName ==  string("delta_m")){
            delta_m = atof(line_elements[1].c_str());
        } else if (varName ==  string("delta_n")){
            delta_n = atof(line_elements[1].c_str());
        } else if (varName ==  string("sigma_m")){
            sigma_m = atof(line_elements[1].c_str());
        } else if (varName ==  string("sigma_c")){
            sigma_c = atof(line_elements[1].c_str());
        } else if (varName ==  string("p")){
            p = atof(line_elements[1].c_str());
        } else if (varName ==  string("binary_threshold")){
            binary_threshold = atof(line_elements[1].c_str());
        } else if (varName ==  string("numFDOGIterations")){
            numFDOGIterations = atoi(line_elements[1].c_str());
        } else if (varName ==  string("kernel_dim")){
            kernel_dim = atoi(line_elements[1].c_str());
        } else if (varName ==  string("blur_kernel_sigma")){
            blur_kernel_sigma =atof(line_elements[1].c_str());
        } else if (varName ==  string("sigma_e")){
            sigma_e = atof(line_elements[1].c_str());
        } else if (varName ==  string("r_e")){
            r_e = atof(line_elements[1].c_str());
        } else if (varName ==  string("sigma_g")){
            sigma_g = atof(line_elements[1].c_str());
        } else if (varName ==  string("r_g")){
            r_g = atof(line_elements[1].c_str());
        } else if (varName ==  string("numFBLIterations")){
            numFBLIterations = atoi(line_elements[1].c_str());
        } else {
            PRINT( string("line ")+to_string(i+1)+" is invalid." );
        }
    }
}

void usage() {
    fprintf(stderr, "usage: img_abstraction <settings_file_name>.settings <input_image_name>.png <output_image_name>.png\n"
                    "\n"
                    "Reads parameter values from each line in the form of \"[varName] [varValue]\"\n"
                    "\n"
                    "See Kang's Flow-Based Image Abstraction Paper for parameter explanations.\n"
                    "Parameters are: \n"
                    "   double mu\n"
                    "   int numETFIterations\n"
                    
                    "   double delta_m\n"
                    "   double delta_n\n"
                    "   double sigma_m\n"
                    "   double sigma_c\n"
                    "   double p\n"
                    "   double binary_threshold\n"
                    "   int numFDOGIterations\n"
                    
                    "   double sigma_e\n"
                    "   double r_e\n"
                    "   double sigma_g\n"
                    "   double r_g\n"
                    "   int numFBLIterations\n"
                    "\n"
                );
    exit(1);
}

int main(int argc, char *argv[]) {
    
    if (argc < 4) { usage(); }
    
    char* settings_file_name = argv[1];
    char* input_image_file_name = argv[2];
    char* output_image_file_name = argv[3];
    
    png::image< png::rgba_pixel > input_image(input_image_file_name);
    static const unsigned I_height = input_image.get_height();
    static const unsigned I_width = input_image.get_width();
    png::image< png::rgba_pixel > output_image(I_width, I_height);
    
    cout << "Running image abstraction on " << input_image_file_name << " (" << I_width*I_height << " pixels)." << endl << endl;
    
    /* Default parameter values if not provided from .settings file */
    double mu = 5.0;
    int numETFIterations = 3; 
    
    double delta_m = 1.0; 
    double delta_n = 1.0;
    double sigma_m = 1.0; 
    double sigma_c = 1.0; 
    double p = 1.0; 
    double binary_threshold = 1.0; 
    int numFDOGIterations = 3; 
    int kernel_dim = 13;
    double blur_kernel_sigma = 2.0;
    
    double sigma_e = 8.0; 
    double r_e = 5.0; 
    double sigma_g = 8.0; 
    double r_g = 5.0; 
    int numFBLIterations = 2;
    
    getParametersFromFile(settings_file_name, mu,numETFIterations,delta_m,delta_n,sigma_m,sigma_c,p,binary_threshold,numFDOGIterations, kernel_dim,blur_kernel_sigma,sigma_e,r_e,sigma_g,r_g,numFBLIterations);
    
    NEWLINE;
    PRINT("Parameter Values");
    TEST(mu);
    TEST(numETFIterations);
    TEST(delta_m);
    TEST(delta_n);
    TEST(sigma_m);
    TEST(sigma_c);
    TEST(p);
    TEST(binary_threshold);
    TEST(numFDOGIterations);
    TEST(kernel_dim);
    TEST(blur_kernel_sigma);
    TEST(sigma_e);
    TEST(r_e);
    TEST(sigma_g);
    TEST(r_g);
    TEST(numFBLIterations);
    NEWLINE;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto I = Array<double>(input_image);
    
    auto I_grayscale = Array<double>(I).rgb2gray();
    
    auto g = Array<double>();
    auto etf = Array<double>();
    auto H = Array<double>();
    auto FBL = Array<double>();
    auto J = Array<double>();
    
    getGradient(I_grayscale, g);
    
    // Optional gaussian blurring
    auto kernel = Array<double>();
    get_gaussian_kernel(9, 20, kernel); 
    convolution_filter(kernel, I_grayscale, J);
    
    getETF(I_grayscale, mu, numETFIterations, g, etf);
    
    // Normalize gradients g after ETF computation
    for (int y = 0; y < g.height(); y++) {
        for (int x = 0; x < g.width(); x++) {
    	    double g_current_length = sqrt(g(y,x,X_COORD)*g(y,x,X_COORD) + g(y,x,Y_COORD)*g(y,x,Y_COORD));
            if (g_current_length) {
                // g must be an array of floats since we normalize it eventually
                g(y,x,X_COORD) = g(y,x,X_COORD)/g_current_length;
                g(y,x,Y_COORD) = g(y,x,Y_COORD)/g_current_length;
            }
        }
    }
    filterFDOG(I_grayscale,etf, delta_m, delta_n, sigma_m, sigma_c, p, g, binary_threshold, numFDOGIterations, kernel_dim, blur_kernel_sigma, H);
    
    getFBL(I,etf, delta_m, delta_n, sigma_e, r_e, sigma_g, r_g, g, numFBLIterations, FBL);
    
    for (int y = 0; y < I.height(); y++) {
        for (int x = 0; x < I.width(); x++) {
            output_image[y][x] = PIXEL(FBL(y,x,R_COORD)*H(y,x), FBL(y,x,G_COORD)*H(y,x), FBL(y,x,B_COORD)*H(y,x));
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    cout << "Total Run Time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(end_time-start_time).count()) / (pow(10.0,9.0)) << " seconds." << endl;
    
    output_image.write(output_image_file_name);
    
    return 0;	
}

