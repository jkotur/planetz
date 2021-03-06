
/**
 *  buffers |          values 
 * ---------+--------+--------+--------+----------
 *  gdat1   | pos.x  | pos.y  | pos.z  | alpha
 *  gdat2   | norm.x | norm.y | norm.z | acol.r
 *  gdat3   | col.x  | col.y  | col.b  | acol.g
 *  gdat4   | ke     | ka     | kd     | acol.b
 */

uniform sampler2D gbuff1;
uniform sampler2D gbuff2;
uniform sampler2D gbuff3;
uniform sampler2D gbuff4;

uniform int planes;

varying in float ke;
varying in vec3 lightPos;
varying in vec3 lightColor;

void main()
{	
	vec4 gdat1 = texture2D( gbuff1 , gl_TexCoord[0].st );

	gl_FragColor.a = gdat1.a;
        if( bool(planes) && gdat1.a <= .0 ) gl_FragColor.rgba = vec4(.1,.1,.1,.2);
	if(           gdat1.a <= .0 ) return;

	vec4 gdat2 = texture2D( gbuff2 , gl_TexCoord[0].st );
	vec4 gdat3 = texture2D( gbuff3 , gl_TexCoord[0].st );
	vec4 gdat4 = texture2D( gbuff4 , gl_TexCoord[0].st );

	vec3 viewDir =-normalize(gdat1.xyz);

	vec3 lightDir = lightPos - gdat1.xyz; // lightpos - pos

	float dist = length(lightDir);
//        dist = dist * dist / 50000.0;
	dist /= 50; // magic constans
	dist /= ke;
        lightDir = normalize(lightDir);

	float i = dot(lightDir, gdat2.xyz);

	vec3 fd = gdat3.rgb * clamp( i , 0.0 , 1.0 ) / dist;

	vec3 atm = vec3( gdat2.a , gdat3.a , gdat4.a ) * .3 * clamp( i + .4 , 0.0 , 1.0 ) / dist;

//        vec3 h = normalize(lightDir + viewDir);

//        i = pow(clamp(dot(gdat2.xyz, h), 0.0 , 1.0 ), gdat1.w );	

//        vec3 fs = gdat3.rgb * i / dist;

//        gl_FragColor.rgb = ((fd + atm) * gdat4.z + fs * gdat4.w) * lightColor;
	gl_FragColor.rgb = ((fd + atm) * gdat4.z) * lightColor;
	gl_FragDepth = gl_FragCoord.z;// - gdat2.z;
}

