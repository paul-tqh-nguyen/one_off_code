
uniform float fade;

varying vec3 f_color;

void main(void) {
    
    if (mod(gl_FragCoord.y, 30.0) > 15) {
        gl_FragColor = vec4(f_color.xyz,1.0);
    } else {
        gl_FragColor = vec4(f_color.xyz,fade);
    }
}

