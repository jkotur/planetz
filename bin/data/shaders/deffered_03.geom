#version 130
#extension GL_EXT_geometry_shader4 : enable

/**
 *          |           values
 * ---------+---------+--------+--------+---------
 *  mater1  |    r    |   g    |   b    |  ke 
 *  mater2  |    ka   |   kd   |   ks   |  alpha
 */

uniform sampler1D materials;

in float em[];
in int models[];
in float radiuses[];

out float ke;
out vec3 lightColor;
out vec3 lightPos;

// this magic const should corespond to magic const in fragment shader
const float lightStdRange = 50000.0;

void main(void)
{
	// TODO: move color to vertex buffer just like emissive
	//if( em[0] <= 0.0 ) return;

	if( radiuses[0] < 0.0001 ) return;

	vec4 mat = texelFetch(materials,models[0],0);
	if( mat.a <= 0.0 ) return;

	ke = mat.a;

	float lmax = lightStdRange * ke;

	lightPos   = gl_PositionIn[0].xyz;
	lightColor = mat.rgb;

	float lz = 0;
	if( lightPos.z > lmax ) return;
	if( lightPos.z >= -.1 ) lz = -lightPos.z-.1; // -.1 == 0-epsilon

	// upper right
	gl_Position    = vec4(lightPos.xyz + vec3( lmax , lmax , lz ) , 1 );
	gl_Position    = gl_ProjectionMatrix * gl_Position;
	gl_Position.z  = 0;
	gl_TexCoord[0].st=(vec2(gl_Position.x,gl_Position.y)/gl_Position.w+1)/ 2.0f;
	gl_Position /= gl_Position.w;
	if( gl_Position.x > 1 ) { gl_Position.x = 1; gl_TexCoord[0].s = 1; }
	if( gl_Position.y > 1 ) { gl_Position.y = 1; gl_TexCoord[0].t = 1; }
	EmitVertex();
	// upper left
	gl_Position    = vec4(lightPos.xyz + vec3(-lmax , lmax , lz ) , 1 );
	gl_Position    = gl_ProjectionMatrix * gl_Position;
	gl_Position.z  = 0;
	gl_TexCoord[0].st=(vec2(gl_Position.x,gl_Position.y)/gl_Position.w+1)/ 2.0f;
	gl_Position /= gl_Position.w;
	if( gl_Position.x > 1 ) { gl_Position.x = 1; gl_TexCoord[0].s = 1; }
	if( gl_Position.y <-1 ) { gl_Position.y =-1; gl_TexCoord[0].t = 0; }
	EmitVertex();
	// lower right
	gl_Position    = vec4(lightPos.xyz + vec3( lmax ,-lmax , lz ) , 1 );
	gl_Position    = gl_ProjectionMatrix * gl_Position;
	gl_Position.z  = 0;
	gl_TexCoord[0].st=(vec2(gl_Position.x,gl_Position.y)/gl_Position.w+1)/ 2.0f;
	gl_Position /= gl_Position.w;
	if( gl_Position.x <-1 ) { gl_Position.x =-1; gl_TexCoord[0].s = 0; }
	if( gl_Position.y > 1 ) { gl_Position.y = 1; gl_TexCoord[0].t = 1; }
	EmitVertex();
	// lower left
	gl_Position    = vec4(lightPos.xyz + vec3(-lmax ,-lmax , lz ) , 1 );
	gl_Position    = gl_ProjectionMatrix * gl_Position;
	gl_Position.z  = 0;
	gl_TexCoord[0].st=(vec2(gl_Position.x,gl_Position.y)/gl_Position.w+1)/ 2.0f;
	gl_Position /= gl_Position.w;
	if( gl_Position.x <-1 ) { gl_Position.x =-1; gl_TexCoord[0].s = 0; }
	if( gl_Position.y <-1 ) { gl_Position.y =-1; gl_TexCoord[0].t = 0; }
	EmitVertex();
	EndPrimitive();
}

