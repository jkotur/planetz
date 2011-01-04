#version 130 
#extension GL_EXT_gpu_shader4 : enable

/**
 *  buffers   |          values 
 * -----------+--------+--------+--------+----------
 *  FragData0 | pos.x  | pos.y  | pos.z  | alpha
 *  FragData1 | norm.x | norm.y | norm.z | acol.r
 *  FragData2 | col.x  | col.y  | col.b  | acol.g
 *  FragData3 | ke     | ka     | kd     | acol.b
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
uniform sampler2DArray texturesTex;

uniform int iftextures = 0;
uniform int ifnormals  = 0;

//in float phi;
//in float lambda;

in vec3 pos;
in mat3 rot;
in mat3 nrot;
in float radius;
in float atmRadius;
in vec3 atmColor;

in float texId;

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

//	angles /= PI;
//	angles += .5;
//	angles.t = 1-angles.t;

	vec4 tex;
	if( iftextures == 1 )
		tex = texture2DArray( texturesTex,vec3(texture_st_v2(angles),texId));
	else	tex = vec4(1);

	if( ifnormals == 1 )
		gl_FragData[2] = vec4( norm.xyz             , atmColor.g );
	else	gl_FragData[2] = vec4( mater1.rgb * tex.rgb , atmColor.g );

	gl_FragData[1] = vec4( norm.xyz , atmColor.r );

	norm.xyz *= radius;

	gl_FragData[0] = vec4( pos + norm.xyz , norm.w );

	gl_FragData[3] = vec4( mater1.a , mater2.rg , atmColor.b );
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

	ax *= abs( cos( ay ) - 0.01 ); // nasty bugfix for texture zip

	return vec2(.5) - vec2(ax,ay) / vec2(PI*2,PI);
}

vec2 texture_st_v2( vec2 angles )
{
	return texture_st_f2( angles.x , angles.y );
}

