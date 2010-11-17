
uniform sampler2D gbuff1;
uniform sampler2D gbuff2;

varying in vec3 lightPos;
varying in vec3 lightColor;

void main()
{	
	vec4 gdat1 = texture2D( gbuff1 , gl_TexCoord[0].st );
	vec4 gdat2 = texture2D( gbuff2 , gl_TexCoord[0].st );

	vec3 cs = vec3(1,.5,0);
	vec3 cd = vec3(1,.5,0);

	float ps = .0;
	float sh = .0;

	vec3 pos     = gdat1.xyz;
	vec3 viewDir =-normalize(pos);
	vec3 normal  = gdat2.xyz;

//        lightPos =  vec3(0,0,0);

	vec3 lightDir = lightPos - pos;

	float dist = length(lightDir);
	dist /= 10;
//        dist = dist * dist / 100;
	lightDir = normalize(lightDir);

	float i = clamp(dot(lightDir, normal) , 0.0 , 1.0 ); 

	vec3 fd = cd * i / dist;

//        vec3 h = normalize(lightDir + viewDir);

//        i = pow(clamp(dot(normal, h), 0.0 , 1.0 ), sh );	

//        vec3 fs = i * cs * ps / dist;

	gl_FragColor.rgb = vec3(.1,.05,.0)/8 + fd;// + fs;
	gl_FragColor.a = gdat1.a;

//        gl_FragColor.rgb = lightDir;
}

