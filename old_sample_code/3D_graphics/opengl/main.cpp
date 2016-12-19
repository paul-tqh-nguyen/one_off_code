/*

Following these tutorials: 
http://en.wikibooks.org/wiki/OpenGL_Programming

*/

#include <cstdio>
#include <cstdlib>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "util.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct attributes {
  GLfloat coord3d[3];
  GLfloat v_color[3];
};

/* Globals */
GLint uniform_m_transform;
GLuint vbo_triangle;
GLint uniform_fade;
GLint attribute_coord3d, attribute_v_color;
GLint program; 

int init_resources(void) { // Make our shaders and program
    struct attributes triangle_attributes[] = {
        {{ 0.0,  0.8, 0.0}, {1.0, 1.0, 0.0}},
        {{-0.8, -0.8, 0.0}, {0.0, 0.0, 1.0}},
        {{ 0.8, -0.8, 0.0}, {1.0, 0.0, 0.0}}
    };
    
    glGenBuffers(1, &vbo_triangle); // get buffer handle
    glBindBuffer(GL_ARRAY_BUFFER, vbo_triangle); // tell OpenGL to set vbo_triangle as the current active vertex buffer
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangle_attributes), triangle_attributes, GL_STATIC_DRAW); // put our data into the buffer
    
    
    // Compile Vertex Shader
    GLuint vertex_shader, fragment_shader;
    if ((vertex_shader = create_shader("shader.v.glsl", GL_VERTEX_SHADER))   == 0) {
        return 0;
    }
    if ((fragment_shader = create_shader("shader.f.glsl", GL_FRAGMENT_SHADER)) == 0) {
        return 0;
    }

    // Compile Program
    if ((program = create_program(vertex_shader, fragment_shader)) == 0) {
        return 0;
    }; 
    
    // Bind Variables
    attribute_v_color = glGetAttribLocation(program, "v_color");
    if (attribute_v_color == -1) {
        fprintf(stderr, "Could not bind attribute %s\n", "v_color");
        return 0;
    }
    
    attribute_coord3d = glGetAttribLocation(program, "coord3d");
    if (attribute_coord3d == -1) {
        fprintf(stderr, "Could not bind attribute %s\n", "coord3d");
        return 0;
    }
    
    uniform_fade = glGetUniformLocation(program, "fade");
    if (uniform_fade == -1) {
        fprintf(stderr, "Could not bind uniform %s\n", "fade");
        return 0;
    }
    
    return 1;
}

void onDisplay() { // draw what we need to draw
    glClearColor(1.0, 1.0, 1.0, 1.0); // set clear color to white
    glClear(GL_COLOR_BUFFER_BIT); // clear background to clear color
    
    glUseProgram(program);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo_triangle);
    glEnableVertexAttribArray(attribute_coord3d);
    glVertexAttribPointer(attribute_coord3d, 3, GL_FLOAT, GL_FALSE, sizeof(struct attributes), 0 );
    
    glEnableVertexAttribArray(attribute_v_color);
    glVertexAttribPointer(attribute_v_color, 3, GL_FLOAT, GL_FALSE, sizeof(attributes), (GLvoid*) offsetof(struct attributes, v_color));
    
    //glUniform1f(uniform_fade, 0.5);
    
    glDrawArrays(GL_TRIANGLES, 0, 3);
        // GL_TRIANGLES tells it in what way to draw the vertices, in this case we tell it to take each 3 adn draw a triangle
        // 0 specifies the index to start at
        // 3 specifies how many vertices to take
    glDisableVertexAttribArray(attribute_coord3d);
    glDisableVertexAttribArray(attribute_v_color);
        // Once we are done wit these attribute values, we want to tell OpenGL to not use them anymore so that no other shader that's 
        // not supposed to use them tries to use them
 
    glutSwapBuffers(); // swap our buffers
} 

void free_resources() { 
    glDeleteBuffers(1, &vbo_triangle);
    glDeleteProgram(program);
} 

void idle() { 
    float move = sinf(glutGet(GLUT_ELAPSED_TIME) / 1000.0 * (2*3.14) / 5); // -1<->+1 every 5 seconds
    float angle = glutGet(GLUT_ELAPSED_TIME) / 1000.0 * 45;  // 45° per second
    glm::vec3 axis_z(0, 0, 1);
    glm::mat4 m_transform = glm::translate(glm::mat4(1.0f), glm::vec3(move, 0.0, 0.0))
        * glm::rotate(glm::mat4(1.0f), angle, axis_z);
    
    float cur_fade = sinf(glutGet(GLUT_ELAPSED_TIME) / 1000.0 * (2*M_PI) / 5) / 2 + 0.5; // 0->1->0 every 5 seconds
    
    glUseProgram(program);
    glUniform1f(uniform_fade, cur_fade);
    glUniformMatrix4fv(uniform_m_transform, 1, GL_FALSE, glm::value_ptr(m_transform));
    glutPostRedisplay();
} 

int main(int argc, char* argv[]) {
    
    // GLUT Initialization
    glutInit(&argc, argv);
    glutInitContextVersion(2,0);
    glutInitDisplayMode(GLUT_RGBA|GLUT_ALPHA|GLUT_DOUBLE|GLUT_DEPTH);
    glutInitWindowSize(640, 480);
    glutCreateWindow("My First Triangle");
    glutIdleFunc(idle);
    // Enable alpha
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // GLEW Initialization
    GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK) {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(glew_status));
        return EXIT_FAILURE;
    }
    
    if (1 == init_resources()) { // only run if we initialized everything successfully
        glutDisplayFunc(onDisplay);
        glutMainLoop();
    }
    
    free_resources();
    return EXIT_SUCCESS;
}

