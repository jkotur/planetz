#version 130 

/**
 *  buffers   |          values 
 * -----------+--------+--------+--------+----------
 *  FragData0 | pos.x  | pos.y  | pos.z  | alpha
 *  FragData1 | norm.x | norm.y | norm.z | material
 *  FragData2 | col.x  | col.y  | col.b  | alpha
 *  FragData3 | ke     | ka     | kd     | ks
 */

/**
 *          |           values
 * ---------+---------+--------+--------+---------
 *  mater1  |    r    |   g    |   b    |  ke 
 *  mater2  |    ka   |   kd   |   ks   |  alpha
 */

uniform sampler2D sph_pos;

//uniform sampler2D anglesTex;
uniform sampler2D normalsTex;
uniform sampler2D textureTex;

//in float phi;
//in float lambda;

in vec3 pos;
in mat3 rot;
in mat3 nrot;
in float radius;

in vec4 mater1;
in vec4 mater2;

const float PI = 3.1415926;

vec2 texture_st_v2( vec2 angles );
vec2 texture_st_f2( float ax , float ay );

void main()
{
	vec4 norm = texture2D( sph_pos , gl_TexCoord[0].st );

	norm.xyz = nrot * norm.xyz;           // rotate normal

	vec3 ntex = rot * norm.xyz;
	vec2 angles = vec2( -atan(ntex.z,ntex.x)+PI/2.0f , asin(ntex.y) );

	vec4 tex = texture2D( textureTex , texture_st_v2(angles) );
//	tex = vec4(1); // temporary for testing

	gl_FragData[1].xyz = norm.xyz;      // normal vector
	gl_FragData[1].w   = 0;        // model id (deprecated)

	norm *= radius;

	gl_FragData[0].xyz = pos + norm.xyz;
	gl_FragData[0].a   = norm.w;	   // draw or not draw this fragment

	gl_FragData[2]     = vec4( mater1.rgb * tex.rgb , mater2.a * tex.a );
	gl_FragData[3]     = vec4( mater1.a , mater2.rgb );
}

vec2 texture_st_f2( float ax , float ay )
{
	if( ay > PI/2.0f ) {
		ay = PI - ay;
		ax =-PI - ax;
	}

	if( ay <-PI/2.0f ) {
		ay =-PI - ay;
		ax =-PI - ax;
	}

	if( ax > PI )
		ax =-2*PI + ax;

	if( ax <-PI )
		ax = 2*PI + ax;

	ax *= abs( cos( ay ) );

	return vec2(.5) - vec2(ax,ay) / vec2(PI*2,PI);
}

vec2 texture_st_v2( vec2 angles )
{
	return texture_st_f2( angles.x , angles.y );
}

