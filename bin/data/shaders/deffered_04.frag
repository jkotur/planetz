
/**
 *  buffers   |          values 
 * -----------+--------+--------+--------+----------
 *  FragData0 | norm.x | norm.y | norm.z | alpha
 *  FragData1 | pos.x  | pos.y  | pos.z  | 0
 *  FragData2 | col.x  | col.y  | col.b  | 0
 */

uniform sampler2D texture;

in vec3 pos;
in mat3 nrot;
in float radius;
in vec3 color;

void main()
{
	vec4 norm = texture2D( texture , gl_TexCoord[0].st );

	norm.xyz = nrot * norm.xyz;

	gl_FragData[0].rgba = norm;

	norm.xyz *= radius;

	gl_FragData[1].rgb  = pos + norm.xyz;
	gl_FragData[1].a    = 1.0;
	gl_FragData[2].rgba = vec4( color , norm.a );

	gl_FragDepth = gl_FragCoord.z+.0001;
}

