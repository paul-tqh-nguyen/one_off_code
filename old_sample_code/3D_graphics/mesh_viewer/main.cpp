/*

clear && make debug && ./main 

*/

#include <GL/glut.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include "helper_libraries/vecmath.h"
#include "helper_libraries/list.h"
#include "helper_libraries/string_manipulation.h"

#define NUM_COLORS 4

using std::vector;
using std::string;
using std::cout;
using std::endl;

// Global Variables
vector<Vector3f> vector_of_points; // vector of points
vector<Vector3f> vector_of_normals; // vector of normals
vector<vector<unsigned int> > vector_of_faces; // vector of faces (a face is just a list of indices into the point and vector lists that specify points)
unsigned int color_index = 0; // what color we want our teapot to be
GLfloat light_pos_delta = 0.05f;
GLfloat light_pos_x = 0; // x position of the light
GLfloat light_pos_y = 0; // y position of the light
GLfloat colors[NUM_COLORS][4] = {{0.5, 0.5, 0.9, 1.0}, // Array of colors
                                 {0.9, 0.5, 0.5, 1.0},
                                 {0.5, 0.9, 0.3, 1.0},
                                 {0.3, 0.8, 0.9, 1.0}};

// Convenience Functions 
inline void glVertex(const Vector3f &a){ glVertex3f(a[0],a[1],a[2]); }
inline void glNormal(const Vector3f &a){ glNormal3f(a[0],a[1],a[2]); }

void keyboardFunc( unsigned char key, int x, int y ) { // to deal with normal keys, e.g. alphabets
    switch(key){
        case 27: // Escape key
            exit(0);
            break;
        case 'c':
		    color_index = (color_index+1)%4; 
            break;
        default:
            cout << "Unhandled key press " << key << "." << endl;        
    }
    glutPostRedisplay(); // refresh the screen
}

void specialFunc( int key, int x, int y ){ // to deal with special keys, e.g. arrows
    switch ( key )
    {
    case GLUT_KEY_UP:
        light_pos_y += light_pos_delta;
		break;
    case GLUT_KEY_DOWN:
        light_pos_y -= light_pos_delta;
		break;
    case GLUT_KEY_LEFT:
        light_pos_x -= light_pos_delta;
		break;
    case GLUT_KEY_RIGHT:
        light_pos_x += light_pos_delta;
		break;
    }
    
    glutPostRedisplay(); // refersh screen
}

void drawScene(void){ // displays everything
    int i;
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// Clear the rendering window
    
    // Rotate the image
    glMatrixMode( GL_MODELVIEW );  // Current matrix affects objects positions
    glLoadIdentity();              // Initialize to the identity
    
    gluLookAt(0.0, 0.0, 5.0,  // Position the camera at [0,0,5]
              0.0, 0.0, 0.0,  // look at [0,0,0]
              0.0, 1.0, 0.0); // up direction as [0,1,0]
    
    // Set material properties of object
    
	// Here we use the first color entry as the diffuse color
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, colors[color_index]);

	// Define specular color and shininess
    GLfloat specColor[] = {1.0, 1.0, 1.0, 1.0};
    GLfloat shininess[] = {100.0};
    
	// Note that the specular color and shininess can stay constant
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specColor);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
    
    // Set light properties
    
    GLfloat Lt0diff[] = {1.0,1.0,1.0,1.0}; // set light color
	GLfloat Lt0pos[] = {light_pos_x, light_pos_y, 5.0f, 1.0f}; // set light position

    glLightfv(GL_LIGHT0, GL_DIFFUSE, Lt0diff);
    glLightfv(GL_LIGHT0, GL_POSITION, Lt0pos);
    
	// Display whatever faces are in vector_of_faces
	for(int i = 0; i < vector_of_faces.size(); i++){
	    auto face = vector_of_faces[i];
	    glBegin(GL_TRIANGLES);
	    for(int j = 0; j < face.size()/2; j++){
            glNormal3d(vector_of_normals[face[j+3]][0], vector_of_normals[face[j+3]][1], vector_of_normals[face[3+j]][2]);
            glVertex3d( vector_of_points[face[j]  ][0],  vector_of_points[face[j]  ][1],  vector_of_points[face[j]  ][2]);
        }
        glEnd();
    }
    
    // Dump the image to the screen.
    glutSwapBuffers();
}


void initRendering(){ // Initialize OpenGL rendering modes
    glEnable(GL_DEPTH_TEST);   // Depth testing must be turned on
    glEnable(GL_LIGHTING);     // Enable lighting calculations
    glEnable(GL_LIGHT0);       // Turn on light #0.
}

void reshapeFunc(int w, int h) { // called when window is resized
    if (w > h) { // use  largest square viewport possible
        glViewport((w - h) / 2, 0, h, h);
    } else {
        glViewport(0, (h - w) / 2, w, w);
    }

    // Set up a perspective view, with square aspect ratio
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // 50 degree fov, uniform aspect ratio, near = 1, far = 100
    gluPerspective(50.0, 1.0, 1.0, 100.0);
}

void loadInput(string obj_mesh_file_name){ // Parse the .obj mesh
    std::ifstream input( obj_mesh_file_name );
    string line;
    while ( getline( input, line ) ) { // while there's another line
        vector<string> items;
        split(line, string(" "), items);
        if (items[0] == string("v")) { 
            ASSERT(items.size() == 4, "line in .obj file starts with token \"v\" with less/more than four elements");
            vector_of_points.push_back( Vector3f(atof(items[1].c_str()),atof(items[2].c_str()),atof(items[3].c_str())) );
        } else if (items[0] == string("vn")) { 
            ASSERT(items.size() == 4, "line in .obj file starts with token \"vn\" with less/more than four elements");
            vector_of_normals.push_back( Vector3f(atof(items[1].c_str()),atof(items[2].c_str()),atof(items[3].c_str())) );
        } else if (items[0] == string("f")) { 
            ASSERT(items.size() == 4, "line in .obj file starts with token \"f\" with less/more than four elements");
            vector<unsigned int> face;
            vector<string> v1;
            vector<string> v2;
            vector<string> v3;
            split(items[1], string("/"), v1);
            split(items[2], string("/"), v2);
            split(items[3], string("/"), v3);
            face.push_back( atoi(v1[0].c_str())-1 ); // point indices
            face.push_back( atoi(v2[0].c_str())-1 );
            face.push_back( atoi(v3[0].c_str())-1 );
            face.push_back( atoi(v1[2].c_str())-1 ); // normal indices
            face.push_back( atoi(v2[2].c_str())-1 );
            face.push_back( atoi(v3[2].c_str())-1 );
            vector_of_faces.push_back(face);
        }
    }
}

void usage() {
    fprintf(stderr, "usage: main <obj-mesh-file-name>.obj\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    
    if (argc == 1) { usage(); }
    
    auto obj_mesh_file_name = string(argv[1]);
    
    loadInput(obj_mesh_file_name);

    glutInit(&argc,argv);
    
    // We're going to animate it, so double buffer 
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );

    // Initial parameters for window position and size
    glutInitWindowPosition( 60, 60 );
    glutInitWindowSize( 360, 360 );
    glutCreateWindow("mesh_viewer");

    // Initialize OpenGL parameters.
    initRendering();

    // Set up callback functions for key presses
    glutKeyboardFunc(keyboardFunc); // Handles "normal" ascii symbols
    glutSpecialFunc(specialFunc);   // Handles "special" keyboard keys

     // Set up the callback function for resizing windows
    glutReshapeFunc( reshapeFunc );

    // Call this whenever window needs redrawing
    glutDisplayFunc( drawScene ); // specify function to redraw the window

    glutMainLoop( ); // main loop (never returns or quits)
    
    return 0;
}

