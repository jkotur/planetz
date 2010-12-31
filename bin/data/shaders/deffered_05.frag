
/**
 *  buffers |          values 
 * ---------+--------+--------+--------+-------
 *  gdat1   | norm.x | norm.y | norm.z | alpha
 *  gdat2   | pos.x  | pos.y  | pos.z  | 0
 *  gdat3   | col.x  | col.y  | col.b  | 0
 */

uniform sampler2D gbuff1;
uniform sampler2D gbuff2;
uniform sampler2D gbuff3;

varying in float ke;
varying in vec3 lightPos;
varying in vec3 lightColor;

void main()
{	
	vec4 gdat1 = texture2D( gbuff1 , gl_TexCoord[0].st );

	gl_FragColor.a = gdat1.a;
	if( gdat1.a <= .0 ) return;

	vec4 gdat2 = texture2D( gbuff2 , gl_TexCoord[0].st );
	vec4 gdat3 = texture2D( gbuff3 , gl_TexCoord[0].st );

	vec3 viewDir =-normalize(gdat1.xyz);

	vec3 lightDir = lightPos - gdat2.xyz; // lightpos - pos

	float dist = length(lightDir);
	dist /= 50;
	dist /= ke;
        lightDir = normalize(lightDir);

	float i = dot(lightDir, gdat1.xyz);

	vec3 atm = gdat3.rgb * clamp( i + .4 , 0.0 , 1.0 ) / dist;

	gl_FragColor.rgb = atm * 10 * gdat3.a * lightColor;
	gl_FragColor.a   = 1;
}

