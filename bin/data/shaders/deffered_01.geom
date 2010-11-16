#version 130 
#extension GL_EXT_geometry_shader4 : enable

uniform float ratio;

in float radiuses[];

void main(void)
{
	float radius = radiuses[0];
	float r2 = radius / 2.0;

	// upper right
	gl_Position    = gl_PositionIn[0] + vec4(r2,r2*ratio,0,0);
	gl_TexCoord[0] = vec4( 1 , 1 , 0 , 0 );
	gl_FrontColor  = vec4( .75 ,.5 , 0 , 0 );
	EmitVertex();
	// upper left
	gl_Position = gl_Position + vec4(-radius,0,0,0);
	gl_TexCoord[0] = vec4( 0 , 1 , 0 , 0 );
	gl_FrontColor  = vec4( .75 ,.5 , 0 , 0 );
	EmitVertex();
	// lower right
	gl_Position = gl_Position + vec4( radius,-radius*ratio,0,0);
	gl_TexCoord[0] = vec4( 1 , 0 , 0 , 0 );
	gl_FrontColor  = vec4( .75 ,.5 , 0 , 0 );
	EmitVertex();
	// lower left
	gl_Position = gl_Position + vec4(-radius,0,0,0);
	gl_TexCoord[0] = vec4( 0 , 0 , 0 , 0 );
	gl_FrontColor  = vec4( .75 ,.5 , 0 , 0 );
	EmitVertex();
	EndPrimitive();
}

