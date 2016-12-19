/* 

Written in GLSL, i.e. OpenGL's Shading Language

Vertex Shader

Doesn't have many of C's types and doesn't have pointers
    Has bool, int, and float
    Adds a bunch of vector and matrix types of up to four elements
    vec2 and vec4 are vectors of floats
    Vector can have a single scalar value (same value put into all slots of vector)
    Vectors can also string a bunch of single and vector values intoa  big vector
    
The shaders communicate using a bunch of special globals
    Three typees of variables:
        uniform: 
            Comes from uniform state
        attribute:
            Supplies per vertex values/inputs from vertex array
        varying:
            The shader assigned it's per vertex outputs to these verying variables

*/

#version 110 // Says we're using GLSL version 1.10

uniform float timer;

attribute vec4 position;

varying vec2 texcoord;
varying float fade_factor;

void main() {
    gl_Position = position;
        // gl_Position is a predefined GLSL variable for assiging to the screen space
        // gl_Position takes a vec4
    texcoord = position.xy * vec2(0.5) + vec2(0.5);
        // we can use .y, .y, .z, .w to get the first four elements of a vec4
        // we can string them together, e.g. position.xy, to get a vec2 of (position.x, potiison.y)
    fade_factor = sin(timer) * 0.5 + 0.5;
}

