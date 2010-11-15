#version 130 
#extension GL_EXT_geometry_shader4 : enable

uniform float ratio;

void main(void)
{
	gl_Position = gl_PositionIn[0] + vec4(.5,.5*ratio,0,0);
	gl_FrontColor    = vec4( 1 , 0 , 0 , 1 );
	EmitVertex();
	gl_Position = gl_Position + vec4(-1.0,0,0,0);
	gl_FrontColor    = vec4( 0 , 1 , 0 , 1 );
	EmitVertex();
	gl_Position = gl_Position + vec4( 1.0,-1.0*ratio,0,0);
	gl_FrontColor    = vec4( 1 , 1 , 0 , 1 );
	EmitVertex();
	gl_Position = gl_Position + vec4(-1.0,0,0,0);
	gl_FrontColor    = vec4( 0 , 0 , 1 , 1 );
	EmitVertex();
	EndPrimitive();
}

