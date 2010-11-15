
varying in  float in_radiuses;

varying out float radiuses;

void main()
{	
	radiuses = in_radiuses;
	gl_Position = ftransform();
//        gl_Position = gl_Vertex;
}
