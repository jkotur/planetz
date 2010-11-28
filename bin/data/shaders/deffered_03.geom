
varying out vec3 lightColor;
varying out vec3 lightPos;

varying in bool is_light[];

void main(void)
{
	if( !is_light[0] ) return;

	lightPos = gl_PositionIn[0].xyz;

	// upper right
	gl_Position    = vec4( 1 , 1 , 0 , 1 );
	gl_TexCoord[0] = vec4( 1 , 1 , 0 , 0 );
	EmitVertex();
	// upper left
	gl_Position    = vec4(-1 , 1 , 0 , 1 );
	gl_TexCoord[0] = vec4( 0 , 1 , 0 , 0 );
	EmitVertex();
	// lower right
	gl_Position    = vec4( 1 ,-1 , 0 , 1 );
	gl_TexCoord[0] = vec4( 1 , 0 , 0 , 0 );
	EmitVertex();
	// lower left
	gl_Position    = vec4(-1 ,-1 , 0 , 1 );
	gl_TexCoord[0] = vec4( 0 , 0 , 0 , 0 );
	EmitVertex();
	EndPrimitive();
}

