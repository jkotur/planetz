#version 130 
#extension GL_EXT_geometry_shader4 : enable

in vec3 colors[];
in float radiuses[];

out vec3 pos;
out mat3 nrot;
out float radius;
out vec3 color;

mat3 faceme( vec3 pos );

void main()
{
	if( radiuses[0] <= 0.0f ) return;

	pos    = gl_PositionIn[0].xyz;
	radius = radiuses[0];
	color  = colors[0];

	if( -pos.z < radius ) return;

	nrot = faceme( pos );

	vec4 u = vec4( nrot * vec3(0,radius,0) , 0 );
	vec4 r = vec4( nrot * vec3(radius,0,0) , 0 );

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

mat3 faceme( vec3 pos )
{
	float lenx = length( pos.xz );
	float cosx = gl_PositionIn[0].z / lenx;
	float sinx = gl_PositionIn[0].x / lenx;

	mat3 rotx  = mat3( cosx ,  0   ,-sinx ,
	                    0   ,  1   ,  0   ,
			   sinx ,  0   , cosx );

	float leny = length( pos.yz );
	float cosy = gl_PositionIn[0].z / leny;
	float siny = gl_PositionIn[0].y / leny;

	mat3 roty = mat3(  1   ,  0   ,  0   ,
	                   0   , cosy , siny ,
			   0   ,-siny , cosy );

	return rotx * roty;
}

