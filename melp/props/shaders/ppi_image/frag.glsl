#version 330

uniform sampler2D Texture;
uniform sampler2D AzimuthTexture;
uniform float scan_start_deg;
uniform float scan_progress_deg;
uniform float data_opacity;

in vec2 v_im_coords;
in float v_opacity;

out vec4 frag_color;

void main() {
    frag_color = texture(Texture, v_im_coords);

    float azimuth_deg = 360.0 * texture(AzimuthTexture, v_im_coords).r;

    float relative_deg = mod(azimuth_deg - scan_start_deg + 360.0, 360.0);
    float visible = step(relative_deg, clamp(scan_progress_deg, 0.0, 360.0));
    frag_color.a *= v_opacity * data_opacity * visible;
}
