#version 130 
#extension GL_EXT_geometry_shader4 : enable

uniform float ratio;

in float radiuses[];

void main(void)
{
	float radius = radiuses[0];
	float r2 = radius / 2.0;

	gl_Position = gl_PositionIn[0] + vec4(r2,r2*ratio,0,0);
	gl_FrontColor    = vec4( 1 , 0 , 0 , 0 );
	EmitVertex();
	gl_Position = gl_Position + vec4(-radius,0,0,0);
	gl_FrontColor    = vec4( 0 , 1 , 0 , 0 );
	EmitVertex();
	gl_Position = gl_Position + vec4( radius,-radius*ratio,0,0);
	gl_FrontColor    = vec4( 1 , 1 , 0 , 0 );
	EmitVertex();
	gl_Position = gl_Position + vec4(-radius,0,0,0);
	gl_FrontColor    = vec4( 0 , 0 , 1 , 0 );
	EmitVertex();
	EndPrimitive();
}

