
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

uniform int planes = 0;

varying in float ke;
varying in vec3 lightPos;
varying in vec3 lightColor;

void main()
{	
	vec4 gdat1 = texture2D( gbuff1 , gl_TexCoord[0].st );
	vec4 gdat2 = texture2D( gbuff2 , gl_TexCoord[0].st );
	vec4 gdat3 = texture2D( gbuff3 , gl_TexCoord[0].st );
	vec4 gdat4 = texture2D( gbuff4 , gl_TexCoord[0].st );

	vec3 viewDir =-normalize(gdat1.xyz);

	vec3 lightDir = lightPos - gdat1.xyz; // lightpos - pos

	float dist = length(lightDir);
//        dist = dist * dist / 10;
	dist /= 5; // magic constans
	dist /= ke;
        lightDir = normalize(lightDir);

	float i = clamp(dot(lightDir, gdat2.xyz) , 0.0 , 1.0 ); 

	vec3 fd = gdat3.rgb * i / dist;

        vec3 h = normalize(lightDir + viewDir);

        i = pow(clamp(dot(gdat2.xyz, h), 0.0 , 1.0 ), gdat3.w );	

        vec3 fs = gdat3.rgb * i / dist;

	gl_FragColor.rgb = (fd * gdat4.z + fs * gdat4.w) * lightColor;
	gl_FragColor.a = gdat1.a;

/** lights range planes */
        if( planes && gdat1.a <= .0 )
                gl_FragColor.rgba = vec4(.1,.1,.1,.2);
}

