#version 330

uniform vec3 beam_origin;
uniform vec3 beam_direction;
uniform float beam_range;
uniform float beam_slope;
uniform float beam_inside_opacity;
uniform float beam_outside_opacity;
uniform float beam_highlight_feather;
uniform float beam_highlight_strength;

in vec4 v_color;
in vec3 v_point;

out vec4 frag_color;

void main(){
    vec3 offset = v_point - beam_origin;
    float axial = dot(offset, beam_direction);
    vec3 radial_vector = offset - axial * beam_direction;
    float radial = length(radial_vector);

    float cone_margin = axial * beam_slope - radial;
    float signed_margin = min(
        min(axial, beam_range - axial),
        cone_margin
    );
    float membership = (
        beam_highlight_feather > 0.0
        ? smoothstep(
            -beam_highlight_feather,
            beam_highlight_feather,
            signed_margin
        )
        : step(0.0, signed_margin)
    );
    float masked_opacity = mix(
        beam_outside_opacity,
        beam_inside_opacity,
        membership
    );

    frag_color = v_color;
    frag_color.a *= mix(
        1.0,
        masked_opacity,
        clamp(beam_highlight_strength, 0.0, 1.0)
    );
}
