
/**
 *  buffers |          values 
 * ---------+--------+--------+--------+----------
 *  gdat1   | pos.x  | pos.y  | pos.z  | alpha
 *  gdat2   | norm.x | norm.y | norm.z | material
 *  gdat3   | col.x  | col.y  | col.b  | alpha
 *  gdat4   | ke     | ka     | kd     | ks
 */

uniform sampler2D gbuff1;
uniform sampler2D gbuff2;
uniform sampler2D gbuff3;
uniform sampler2D gbuff4;

uniform float brightness;

void main()
{	
	vec4 gdat1 = texture2D( gbuff1 , gl_TexCoord[0].st );
	vec4 gdat2 = texture2D( gbuff2 , gl_TexCoord[0].st );
	vec4 gdat3 = texture2D( gbuff3 , gl_TexCoord[0].st );
	vec4 gdat4 = texture2D( gbuff4 , gl_TexCoord[0].st );

	gl_FragColor.rgb = gdat3.rgb*(gdat4.x + gdat4.y * brightness);
	gl_FragColor.a = gdat1.a;
}

