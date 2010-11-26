#version 130 

uniform sampler2D sph_pos;

varying in vec3 pos;
varying in mat3 rot;
varying in float radius;

varying in vec4 mater1;
varying in vec4 mater2;

void main()
{
	vec4 tex = texture2D( sph_pos , gl_TexCoord[0].st );

	tex.xyz = rot * tex.xyz;           // rotate normal

	gl_FragData[1].xyz = tex.xyz;      // normal vector
	gl_FragData[1].w   = 0;        // model id (deprecated)

	tex.xyz *= radius;

	gl_FragData[0].xyz = pos + tex.xyz;
	gl_FragData[0].a   = tex.a;	   // draw or not draw this fragment

	gl_FragData[2]     = mater1;
	gl_FragData[3]     = mater2;
}

