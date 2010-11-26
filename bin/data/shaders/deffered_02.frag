
uniform sampler2D gbuff1;
uniform sampler2D gbuff2;
uniform sampler2D gbuff3;
uniform sampler2D gbuff4;

varying in vec3 lightPos;
varying in vec3 lightColor;

void main()
{	
	vec4 gdat1 = texture2D( gbuff1 , gl_TexCoord[0].st );
	vec4 gdat2 = texture2D( gbuff2 , gl_TexCoord[0].st );
	vec4 gdat3 = texture2D( gbuff3 , gl_TexCoord[0].st );
	vec4 gdat4 = texture2D( gbuff4 , gl_TexCoord[0].st );

	gl_FragColor.rgb = gdat3.rgb*(gdat4.x + gdat4.y);
	gl_FragColor.a = gdat1.a;
}

