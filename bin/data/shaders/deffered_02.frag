
uniform sampler2D gbuff1;
uniform sampler2D gbuff2;

void main()
{	
	vec4 gdat1 = texture2D( gbuff1 , gl_TexCoord[0].st );
	vec4 gdat2 = texture2D( gbuff2 , gl_TexCoord[0].st );

	vec3 cs = vec3(1,.5,0);
	vec3 cd = vec3(1,.5,0);

	float ps = .01;
	float sh = .01;

	vec3 lpos    = vec3(0);
	vec3 pos     = gdat1.xyz;
	vec3 viewDir = vec3(0,0,1);
	vec3 normal  = gdat2.xyz;

	vec3 lightDir = lpos - pos;

	float dist = length(lightDir);
	dist /= 10;
//        dist = dist * dist / 100;
	lightDir = normalize(lightDir); // NORMALIZE THE VECTOR

	float i = clamp(dot(lightDir, normal) , 0.0 , 1.0 ); 

	vec3 fd = cd * i / dist;

	vec3 h = normalize(lightDir + viewDir);

	i = pow(clamp(dot(normal, h), 0.0 , 1.0 ), sh );	

	vec3 fs = i * cs * ps / dist;

	gl_FragColor.rgb += vec3(.1,.05,.0) + fd + fs;
	gl_FragColor.a = gdat1.a;
}

