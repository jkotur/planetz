
uniform sampler2D gbuff1;
uniform sampler2D gbuff2;

varying in vec3 lightPos;
varying in vec3 lightColor;

void main()
{	
	vec4 gdat1 = texture2D( gbuff1 , gl_TexCoord[0].st );
	vec4 gdat2 = texture2D( gbuff2 , gl_TexCoord[0].st );

	vec3 fa = vec3(.1,.05,.0);
	vec3 fe = vec3(.0,.0,.0);

	gl_FragColor.rgb = fa + fe;
	gl_FragColor.a = gdat1.a;
}

