#include <GL/glew.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <assert.h>

#define TEST(x) cout << (#x) << ": " << x << endl;

#define DEBUG BUILD_DEBUG
#define ASSERT(x, msg) if (DEBUG && !(x)) { fprintf(stderr, "Assertion failed in %s(%d): %s\n", __FILE__, __LINE__, msg); assert(false); exit(1); }

using std::cout;
using std::endl;

char* read_file(const char* filename, GLint* length) { // returns content of the file as a char*
    FILE* f = fopen(filename, "r");
    char* buffer;
    if (!f) {
        fprintf(stderr, "Unable to open %s for reading\n", filename);
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    buffer = (char*)malloc(*length+1);
    *length = fread((void*)buffer, 1, *length, f);
    fclose(f);
    buffer[*length] = '\0';
    
    return buffer;
}

void print_log(GLuint object) { // prints shader compile error messages
    GLint log_length = 0;
    if (glIsShader(object)) {
        glGetShaderiv(object, GL_INFO_LOG_LENGTH, &log_length);
    } else if (glIsProgram(object)) {
        glGetProgramiv(object, GL_INFO_LOG_LENGTH, &log_length);
    } else {
        fprintf(stderr, "printlog: Not a shader or a program\n");
        return;
    }

    char* log = (char*)malloc(log_length);

    if (glIsShader(object)) {
        glGetShaderInfoLog(object, log_length, NULL, log);
    } else if (glIsProgram(object)) {
        glGetProgramInfoLog(object, log_length, NULL, log);
    }

    fprintf(stderr, "%s", log);
    free(log);
}

GLuint create_shader(const char* filename, GLenum type) {
    GLint source_length;
    const GLchar* source = read_file(filename, &source_length);
    
    GLuint shader = glCreateShader(type);
    const GLchar* sources[2] = { // select the version here and concat the rest of the source after
        #ifdef GL_ES_V2ERSION_2_0
            "#version 100\n"
            "#define GLES2\n",
        #else
            "#version 120\n",
        #endif
            source 
    };
    glShaderSource(shader, 2, sources, NULL);
    free((void*)source);
    
    glCompileShader(shader);
    GLint compile_ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_ok);
    if (compile_ok == GL_FALSE) {
        fprintf(stderr, "%s:", filename);
        print_log(shader);
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader) {
    GLint link_ok = GL_FALSE; 
    
    GLint program = glCreateProgram(); 
    glAttachShader(program, vertex_shader); 
    glAttachShader(program, fragment_shader); 
    glLinkProgram(program); 
    glGetProgramiv(program, GL_LINK_STATUS, &link_ok); 
    
    if (!link_ok) { 
        fprintf(stderr, "glLinkProgram:"); 
        print_log(program); 
        return 0; 
    } 
    
    return program; 
}

