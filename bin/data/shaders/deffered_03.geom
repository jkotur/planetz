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

out float ke;
out vec3 lightColor;
out vec3 lightPos;

const float lightStdRange = 5000.0;

mat3 faceme( vec3 pos );

void main(void)
{
	// TODO: move color to vertex buffer just like emissive
	//if( em[0] <= 0.0 ) return;

	vec4 mat = texelFetch(materials,models[0],0);
	if( mat.a <= 0.0 ) return;

	ke = mat.a;

	float lmax = lightStdRange * ke;

	lightPos   = gl_PositionIn[0].xyz;
	lightColor = mat.rgb;

	if( lightPos.z > lmax ) return;

	if( lightPos.z <-lmax ) {
		// upper right
		gl_Position    = vec4(lightPos.xyz + vec3( lmax , lmax , 0 ) , 1 );
		gl_Position    = gl_ProjectionMatrix * gl_Position;
		gl_Position.z  = 0;
		gl_TexCoord[0] = (vec4( gl_Position.x , gl_Position.y , 0 , 0 )/gl_Position.w + 1 ) / 2.0f;
		EmitVertex();
		// upper left
		gl_Position    = vec4(lightPos.xyz + vec3(-lmax , lmax , 0 ) , 1 );
		gl_Position    = gl_ProjectionMatrix * gl_Position;
		gl_Position.z  = 0;
		gl_TexCoord[0] = (vec4( gl_Position.x , gl_Position.y , 0 , 0 )/gl_Position.w + 1 ) / 2.0f;
		EmitVertex();
		// lower right
		gl_Position    = vec4(lightPos.xyz + vec3( lmax ,-lmax , 0 ) , 1 );
		gl_Position    = gl_ProjectionMatrix * gl_Position;
		gl_Position.z  = 0;
		gl_TexCoord[0] = (vec4( gl_Position.x , gl_Position.y , 0 , 0 )/gl_Position.w + 1 ) / 2.0f;
		EmitVertex();
	      // lower left
		gl_Position    = vec4(lightPos.xyz + vec3(-lmax ,-lmax , 0 ) , 1 );
		gl_Position    = gl_ProjectionMatrix * gl_Position;
		gl_Position.z  = 0;
		gl_TexCoord[0].st = (vec2( gl_Position.x , gl_Position.y )/gl_Position.w + 1 ) / 2.0f;
		EmitVertex();
		EndPrimitive();
	} else {
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
}

