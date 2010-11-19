#version 130 
#extension GL_EXT_geometry_shader4 : enable

uniform float ratio;

in float radiuses[];
in float  models[];

varying out vec3 pos;
varying out mat3 rot;
varying out float radius;
varying out float model;

void main(void)
{
	radius = radiuses[0];
	model  = models  [0];
	pos    = gl_PositionIn[0].xyz;

	float r2 = radius / 2.0;

	float lenx = length( gl_PositionIn[0].xz );
	float cosx = gl_PositionIn[0].z / lenx;
	float sinx = gl_PositionIn[0].x / lenx;

	mat3 rotx  = mat3( cosx ,  0   ,-sinx ,
	                    0   ,  1   ,  0   ,
			   sinx ,  0   , cosx );

	float leny = length( gl_PositionIn[0].yz );
	float cosy = gl_PositionIn[0].z / leny;
	float siny = gl_PositionIn[0].y / leny;

	mat3 roty  = mat3(  1   ,  0   ,  0   ,
	                    0   , cosy ,-siny ,
			    0   , siny , cosy );

	mat3 rotyp = mat3(  1   ,  0   ,  0   ,
	                    0   , cosy , siny ,
			    0   ,-siny , cosy );

	vec4 u = vec4( roty * vec3(0,radius,0) , 0 );
	vec4 r = vec4( rotx * vec3(radius,0,0) , 0 );

	rot = rotx * rotyp;

	// upper right
	gl_Position    = gl_PositionIn[0] + u + r;
	gl_Position    = gl_ProjectionMatrix * gl_Position;
	gl_TexCoord[0] = vec4( 1 , 1 , 0 , 0 );
	EmitVertex();
	// upper left
	gl_Position    = gl_PositionIn[0] + u - r;
	gl_Position    = gl_ProjectionMatrix * gl_Position;
	gl_TexCoord[0] = vec4( 0 , 1 , 0 , 0 );
	EmitVertex();
	// lower right
	gl_Position    = gl_PositionIn[0] - u + r;
	gl_Position    = gl_ProjectionMatrix * gl_Position;
	gl_TexCoord[0] = vec4( 1 , 0 , 0 , 0 );
	EmitVertex();
	// lower left
	gl_Position    = gl_PositionIn[0] - u - r;
	gl_Position    = gl_ProjectionMatrix * gl_Position;
	gl_TexCoord[0] = vec4( 0 , 0 , 0 , 0 );
	EmitVertex();
	EndPrimitive();
}

