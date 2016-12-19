/*

Fragment shader treats varying variables as inputs. 

Varying variables here correspond to varying variables of the same name in the vertex shader.

Uses same unifrom variables as vertex shader.

Cannot declare attribute variables

*/

#version 110

uniform sampler2D textures[2];

varying float fade_factor;
varying vec2 texcoord;

void main() {
    gl_FragColor = mix(texture2D(textures[0], texcoord), texture2D(textures[1], texcoord), fade_factor);
        // takes a vec4 and uses those values for RGBA
        // texture2D function says to sample texture[i] at the coordinate texcoord (which is an input since it's a varying var)
        // mix interpolated between the two values it's given according to the fade_factor (which is a uniform variable)
        // This is more or less just cross fading between our two images
}
