#version 130 

uniform sampler2D sph_pos;

varying in vec3 pos;
varying in mat3 rot;
varying in float radius;
varying in float model;

void main()
{
	vec4 tex = texture2D( sph_pos , gl_TexCoord[0].st );

	tex.xyz = rot * tex.xyz;

	gl_FragData[1].xyz = tex.xyz;      // normal vector
	gl_FragData[1].w   = model;        // model id

	tex.xyz *= radius;

	gl_FragData[0].xyz = pos + tex.xyz;
	gl_FragData[0].a   = tex.a;	   // draw or not draw this fragment

	gl_FragData[2].xyz = vec3(.1,.1,.1); // ambient light?
}

