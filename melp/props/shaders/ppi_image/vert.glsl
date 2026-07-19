#version 330

uniform sampler2D Texture;

in vec3 point;
in vec2 im_coords;
in float opacity;

out vec2 v_im_coords;
out float v_opacity;

#INSERT emit_gl_Position.glsl

void main() {
    v_im_coords = im_coords;
    v_opacity = opacity;
    emit_gl_Position(point);
}
