#version 130 

//uniform sampler2D sph_pos;

uniform sampler2D angles;
uniform sampler2D normals;

in float phi;
in float lambda;

in vec3 pos;
in mat3 rot;
in float radius;

in vec4 mater1;
in vec4 mater2;

const float PI = 3.1415926;

void main()
{
//	vec4 tex = texture2D( sph_pos , gl_TexCoord[0].st );

//	tex.xyz = rot * tex.xyz;           // rotate normal

	vec3 ang = texture2D( angles  , gl_TexCoord[0].st ).rgb;

	ang.x += lambda;    // latitude
	ang.y += phi;       // longitude

//	if( ang.y > PI/2.0f ) {
//		ang.y = PI - ang.y;
//		ang.x =-PI - ang.x;
//	}

//	if( ang.y <-PI/2.0f ) {
//		ang.y =-PI - ang.y;
//		ang.x =-PI - ang.x;
//	}

//	if( ang.x > PI )
//		ang.x =-2*PI + ang.x;

//	if( ang.x <-PI )
//		ang.x = 2*PI + ang.x;

//	ang.x *= abs( cos( ang.y ) );

	float cp = cos( ang.y );
	ang.x *= cp;
	vec3 norm = vec3( -sin(ang.x)*cp , -sin(ang.y) , cos(ang.x)*cp );

//	ang.xy = ang.xy / vec2(PI*2,PI) + vec2(.5);

//	vec3 norm= texture2D( normals , ang.st ).rgb;

	gl_FragData[1].xyz = norm;      // normal vector
	gl_FragData[1].w   = 0;        // model id (deprecated)

	norm *= radius;

	gl_FragData[0].xyz = pos + norm;
	gl_FragData[0].a   = ang.z;	   // draw or not draw this fragment

	gl_FragData[2]     = mater1;
	gl_FragData[3]     = mater2;
}

