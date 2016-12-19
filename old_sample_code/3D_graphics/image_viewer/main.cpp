/*

Following Tutorial at: 
http://duriansoftware.com/joe/An-intro-to-modern-OpenGL.-Chapter-3:-3D-transformation-and-projection.html

*/

#include <GL/glew.h>
#include <GL/glut.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <assert.h>
#include "util.h"

using std::cout;
using std::endl;
using std::string;

#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))
#define CLAMP(x) MIN(MAX((x),0.0),1.0)

#define DEBUG BUILD_DEBUG
#define ASSERT(x, msg) if (DEBUG && !(x)) { fprintf(stderr, "Assertion failed in %s(%d): %s\n", __FILE__, __LINE__, msg); assert(false); exit(1); }

#define TEST(x) cout << (#x) << ": " << x << endl;

static struct {
    GLuint vertex_buffer, element_buffer;
    GLuint textures[2];
    
    GLuint vertex_shader, fragment_shader, program;
    
    struct {
        GLint timer;
        GLint textures[2];
    } uniforms;

    struct {
        GLint position;
    } attributes;

    GLfloat timer;
} g_resources;

static const GLfloat g_vertex_buffer_data[] = { 
    -1.0f, -1.0f, 0.0f, 1.0f, 
     1.0f, -1.0f, 0.0f, 1.0f, 
    -1.0f,  1.0f, 0.0f, 1.0f, 
     1.0f,  1.0f, 0.0f, 1.0f
};

static const GLushort g_element_buffer_data[] = { 0, 1, 2, 3 };

static GLuint make_buffer(GLenum target, const void *buffer_data, GLsizei buffer_size) {
    GLuint buffer; // OpenGL object names are just GLints
    glGenBuffers(1, &buffer); // We use glGenBuffers to get the next name available
    glBindBuffer(target, buffer); // We need to bind that name to some target (type)
    glBufferData(target, buffer_size, buffer_data, GL_STATIC_DRAW); // allocate space in CPU or GPU memory for the target
        // target is what we're allocating space for (the type, specified by a GLuint)
            // e.g. an element array or vertex array
        // buffer size is how many bytes we need to allocate 
        // buffer_data is what we need to put in there 
        // the last arg is a usage hint that specifies how we will use this 
            // (so OpenGL can decide the best place to put it, which is either in CPU or GPU memory)
            // doesn't constrain usage, just impacts performance
    return buffer;
}

static GLuint make_texture(const char *filename) {
    GLuint texture;
    int width, height;
    void *pixels = read_tga(filename, &width, &height);

    if (!pixels) { 
        return 0; // 0 is OpenGL's null object name, which we will return if we fail
    }
    
    glGenTextures(1, &texture); // We use glGenTextures to get the next name available for the texture
    glBindTexture(GL_TEXTURE_2D, texture); // Need to bind name to some target (type), i.e. a 2D texture/image, i.e. GL_TEXTURE_2D
        // OpenGL also supports 1D ad 3D textures
        // textures are handled different than buffers in GPU memory, which is why there's a special bind texture method
    
    // glTexParameteri specifies how the textures are sampled when we sample between texture coordinates or outside of the range
    // (s,t,r) coordinates are used in texture space instead of (x,y,z) to avoid confusion
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
        // GL_TEXTURE_MIN_FILTER says we need to figure out how to operate when a pixel maps to a bunch of texture coordinates
        // GL_LINEAR specifies that we should take a weighted average of the four closest points to the center
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // GL_TEXTURE_MAG_FILTER says we need to figure out how to operate when a pixel maps in between a bunch of texture coordinates
        // GL_LINEAR specifies that we should take a weighted average of the four closest points to the center
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE); 
        // GL_TEXTURE_WRAP_S specifies how we operate when the texture map goes outside of the [0,1] range  in the S direction 
        // CLAMP_TO_EDGE says to just clip it, i.e. set to 0 if less than zero and to one if bigger than one
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
        // GL_TEXTURE_WRAP_T specifies how we operate when the texture map goes outside of the [0,1] range  in the T direction 
        // CLAMP_TO_EDGE says to just clip it, i.e. set to 0 if less than zero and to one if bigger than one 
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, pixels );
        // Similar to glBufferData
            // Textures can have many levels of detail. OpenGl can sample from many different mipmaps, i.e. versions of the same
            // texture at different resolutions. We will only provide the base level of detail here (we interpolate as specified 
            // above when we need different resolutions)
        // GL_TEXTURE_2D is the target, says we're making a 2D texture
        // 0 is the level of detail since we only provide one level of detail at level 0 of the mipmap hierarchy
        // GL_RGB8 specifies taht we're just using 24 bits for each pixel, i.e. 8 bits for each color channel RGB
        // width and height specify the number of elements along the s adn t axes
        // 0 specifies the number of elements for the border
        // GL_GBR specifies the component order of our pixels (since the TGC format stores things as blue, gree, red, we use GL_BGR)
        // GL_UNSIGNED_BYTE specifies what type the pixels are in
        // pixels is a pointer to our data (the data should be in the format we specified with all these other arguments)
        
    free(pixels);
    return texture;
}

static void show_info_log(GLuint object, PFNGLGETSHADERIVPROC glGet__iv, PFNGLGETSHADERINFOLOGPROC glGet__InfoLog) {
    // The PFNGL* function pointer type names are provided by GLEW
    // object is the shader that didn't compile
    
    GLint log_length;
    char* log;

    glGet__iv(object, GL_INFO_LOG_LENGTH, &log_length); 
        // get the length of the log text
    log = (char*) malloc(log_length);
    glGet__InfoLog(object, log_length, NULL, log);
        // get the log text
    fprintf(stderr, "%s", log);
    free(log);
}

static GLuint make_shader(GLenum type, const char *filename) {
    // Cannot pre-compile a shader binary, is compiled everytime from the source
    // We save our source code for the shader here so we can change it without recompiling our code, we just recompile the shader source
    GLint length;
    GLchar *source = (GLchar*) file_contents(filename, &length); 
        // read contents of the GLSL source file 
        // file_contents is a util.h func
    GLuint shader;
    GLint shader_ok;
    
    if (!source) {
        return 0;
    }
    
    shader = glCreateShader(type);
        // Create shader here
        // type is either GL_VERTEX_SHADER or GL_FRAGMENT_SHADER
        // We don't need to bind the shader to a target/type like we need to do with textures or buffers
    glShaderSource(shader, 1, (const GLchar**)&source, &length);
        // we feed in the source code into OpenGL
    free(source);
    glCompileShader(shader);
        // we tell OpenGL to compile the source code we just fed it
        // creates a bunch of object files that we will use later
    
    glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_ok);
        // OpenGL keeps a log of warnings and errors during compilation
        // We're asking for the compilation status (GL_COMPILE_STATUS) of shader and storing the value into shader_ok
    if (!shader_ok) { // if compilation failed, show error info
        fprintf(stderr, "Failed to compile %s:\n", filename);
        show_info_log(shader, glGetShaderiv, glGetShaderInfoLog);
            // passing in glGetShaderiv and glGetShaderInfoLog functions
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint make_program(GLuint vertex_shader, GLuint fragment_shader) {
    // shaders are compiled into object files
    // program objects are the final program after linking them
    GLint program_ok;
    
    GLuint program = glCreateProgram();
        // define the program
    glAttachShader(program, vertex_shader);
        // link vertex shader
    glAttachShader(program, fragment_shader);
        // link fragment shader
    glLinkProgram(program);
        // link them all to the program
    
    glGetProgramiv(program, GL_LINK_STATUS, &program_ok);
        // make sure everything linked and compiled ok
    if (!program_ok) {
        fprintf(stderr, "Failed to link shader program:\n");
        show_info_log(program, glGetProgramiv, glGetProgramInfoLog);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

static int make_resources(const char *vertex_shader_file) { 
    // Make buffers
    g_resources.vertex_buffer  = make_buffer(GL_ARRAY_BUFFER, g_vertex_buffer_data, sizeof(g_vertex_buffer_data) );
    g_resources.element_buffer = make_buffer(GL_ELEMENT_ARRAY_BUFFER, g_element_buffer_data, sizeof(g_element_buffer_data) );
    
    // Make textures
    g_resources.textures[0] = make_texture("hello1.tga");
    g_resources.textures[1] = make_texture("hello2.tga");
    
    if (g_resources.textures[0] == 0 || g_resources.textures[1] == 0) {
        return 0;
    }
    
    /* Make shaders ... */
    g_resources.vertex_shader = make_shader(GL_VERTEX_SHADER, vertex_shader_file);
    if (g_resources.vertex_shader == 0) {
        return 0;
    }
    
    g_resources.fragment_shader = make_shader(GL_FRAGMENT_SHADER, "hello-gl.f.glsl");
    if (g_resources.fragment_shader == 0) {
        return 0;
    }
    
    g_resources.program = make_program(g_resources.vertex_shader, g_resources.fragment_shader);
    if (g_resources.program == 0) {
        return 0;
    }
    
    // GLSL linker gives a GLint to all uniform, attribute, and varying values
    // we will need these GLint values to assign to our uniform, attribute, and varying values later
    g_resources.uniforms.timer = glGetUniformLocation(g_resources.program, "timer");
    g_resources.uniforms.textures[0] = glGetUniformLocation(g_resources.program, "textures[0]");
    g_resources.uniforms.textures[1] = glGetUniformLocation(g_resources.program, "textures[1]");
    g_resources.attributes.position = glGetAttribLocation(g_resources.program, "position");
    // all these things on the left are GLuint handles, the things on the right get the GLuint handle for the
        // variable in the GLSL code associated with the variable name in the quotes
    
    return 1;
}

static void update_timer(void) { 
    // this is the func that glutIdleFunc calls
    int milliseconds = glutGet(GLUT_ELAPSED_TIME);
        // get the ellapsed time in milliseconds
    g_resources.timer = (float) milliseconds * 0.001f;
        // update timer
    glutPostRedisplay();
        // Tells OpenGL we need to redisplay
}

static void render(void) { 
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // set the clearing color to white
    glClear(GL_COLOR_BUFFER_BIT); // clear the buffer to the clearing color
    
    glUseProgram(g_resources.program);
        // tell OpenGL what program to use for shading
    
    // We now need to start assigning variables and such to give params to our program/shader
    glUniform1f(g_resources.uniforms.timer, g_resources.timer);
        // assigned our timer (g_resources.timer) to the location specified by g_resources.uniforms.timer
        // similar functions for assigning uniform variables have format glUniform{dim}{type}
            // type is either i for int or f for float
        // g_resources.uniforms.timer is the GLuint handle for our uniform variable that determines the fade factor
    
    // OpenGL has a limited number of texture units available (these are units that supply the data from the texture we tell it to)
    glActiveTexture(GL_TEXTURE0);
        // sets the active texture unit
    glBindTexture(GL_TEXTURE_2D, g_resources.textures[0]);
        // We have to bind our texture objects to a texture unit (the one we just specified with glActiveTexture), which is done here
    glUniform1i(g_resources.uniforms.textures[0], 0);
        // g_resources.uniforms.textures[0] is a GLint handle for one of our uniform params. We assign 0 to it to specify that 
        // g_resources.uniforms.textures[0] will get data from texture unit 0
    
    // same as above
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, g_resources.textures[1]);
    glUniform1i(g_resources.uniforms.textures[1], 1);
    
    glBindBuffer(GL_ARRAY_BUFFER, g_resources.vertex_buffer);
    glVertexAttribPointer(g_resources.attributes.position, 4, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*4, (void*)0);
        // we need to tell OpenGL the format of our data for our attribute handle
        // g_resources.attributes.position refers to which attribute/where to read from
        // 4 is the size/number of coordinates per vertex
        // GL_FLOAT specifies the type of our vertices
        // GL_FALSE refers to whether or not we should normalize our vertices (i.e. map the int values from [0,255] to [0.0,1.0], but we don't need this since we're usign floats)
        // sizeof(GLfloat)*4 refers to our stride, i.e. the number of bytes between our variables
        // (void*)0 refers to our array buffer offset
            // For historic reasons, the offset is passed as a pointer, but is used as an int, so we cast to void*. 
    glEnableVertexAttribArray(g_resources.attributes.position);
        // Tell OpenGL to read attributes from this handle
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_resources.element_buffer);
        // Need to bind the element buffer (I thought we already did this in make_resources?)
    glDrawElements(GL_TRIANGLE_STRIP,  4, GL_UNSIGNED_SHORT, (void*)0);
        // GL_TRIANGLE_STRIP specifies in what way to render our elements
        // 4 specifies how many vertices
        // GL_UNSIGNED_SHORT specifies the type of our elements
        // (void*)0 is our array buffer offset
    
    glDisableVertexAttribArray(g_resources.attributes.position);
        // Must disable the current attributes since OpenGL is a state machine
        // We don't want other OpenGL code using this data
    
    glutSwapBuffers(); // swaps the buffers (only works if we're using a double buffer via glutInitDisplayMode(GLUT_DOUBLE)
}

void usage() {
    fprintf(stderr, "usage: main <shader_file> \n");
    exit(1);
}

int main(int argc, char** argv) {
    
    if (argc < 2) {
        usage();
    }
    
    char* shader_file_name = argv[1];
    
    glutInit(&argc, argv); 
        // glut searches the command line params for args that specifically apply to glut
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
        // glutInitDisplayMode sets parameters for the display mode (i.e. let's us choose display options)
        // GLUT_DOUBLE says we will have two buffers in our frame buffer that our display will alternate between to make things look like 
            // they move smoothly
        // GLUT_RGB says the window will display color 
    glutInitWindowSize(400, 300);
        // glutInitWindowSize Specifies window dimensions
        // void glutInitWindowSize(int width, int height);
    glutCreateWindow("Hello World");
        // actually creates the window and makes it pop up
        // the string param specifies the window's name, i.e. what shows up at the top bar of the window
    glutDisplayFunc(&render);
        // glutDisplayFunc used to specify a function (in this case render ) that is to be called everytie we want to redraw the window
        // the display call
    glutIdleFunc(&update_timer);
        // glutIdleFunc used to specify a func to be called when window system events are not beign performed to maintain 
            // background processes like animation etc.
    glewInit();
        // GLEW is just a fancy cross platform library that loads fancy extensions of openGL for us
        // glewInit loads a bunch of fancy functions and extensions (we don't need to worry about what it does for now)
    ASSERT(GLEW_VERSION_2_0, "OpenGL 2.0 not available\n");
        // We want to make sure we're using OpenGL 2.0
    
    if (!make_resources(shader_file_name)) { 
        // make_resources is a function that we write that loads all of the resources we need
            // it manipulates a bunch of global variables
        fprintf(stderr, "Failed to load resources\n");
        return 1;
    }
    
    glutMainLoop();
        // glutMainLoop enters the GLUT event processing loop. It never returns. 
        // It just runs all the fancy functions we told it to above
    
    return 0;
}

