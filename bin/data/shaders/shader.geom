#version 130 
#extension GL_EXT_geometry_shader4 : enable
//#define GL_RGB32F_ARB 0x8815

//CIRL GPU Geometry Program: Derek Anderson and Robert Luke
// very simple geometry shader

/*
GEOMETRY SHADER DOCUMENTATION

Geometry language built-in outputs:
	varying out vec4 gl_FrontColor;
	varying out vec4 gl_BackColor;
	varying out vec4 gl_FrontSecondaryColor;
	varying out vec4 gl_BackSecondaryColor;
	varying out vec4 gl_TexCoord[]; 	// at most gl_MaxTextureCoords
	varying out float gl_FogFragCoord; 

Geometry language input varying variables:
	varying in vec4 gl_FrontColorIn[gl_VerticesIn];
	varying in vec4 gl_BackColorIn[gl_VerticesIn];
	varying in vec4 gl_FrontSecondaryColorIn[gl_VerticesIn];
	varying in vec4 gl_BackSecondaryColorIn[gl_VerticesIn];
	varying in vec4 gl_TexCoordIn[gl_VerticesIn][]; 	// at most will be// gl_MaxTextureCoords
	varying in float gl_FogFragCoordIn[gl_VerticesIn];
	varying in vec4 gl_PositionIn[gl_VerticesIn];
	varying in float gl_PointSizeIn[gl_VerticesIn];
	varying in vec4 gl_ClipVertexIn[gl_VerticesIn];

Geometry Shader Function 
	
	This section contains functions that are geometry language specific.
	
	Syntax:
	 void EmitVertex(); 
	 void EndPrimitive(); 
	
	Description:
	
	 The function EmitVertex() specifies that a vertex is completed. A vertex is added to the current output primitive using the current values of the varying output variables and the current values of the special built-in output variables gl_PointSize, gl_ClipVertex, gl_Layer, gl_Position and gl_PrimitiveID. The values of any unwritten output variables are undefined. The values of all varying output variables and the special built-in output variables are undefined after a call to EmitVertex(). If a geometry shader, in one invocation, emits more vertices than the valueGEOMETRY_VERTICES_OUT_EXT, these emits may have no effect. 
 
	The function EndPrimitive() specifies that the current output primitive is completed and a new output primitive (of the same type) should be started. This function does not emit a vertex. The effect of EndPrimitive() is roughly equivalent to calling End followed by a new Begin, where the primitive mode is taken from the program object parameter GEOMETRY_OUTPUT_TYPE_EXT. If the output primitive type is POINTS, calling EndPrimitive() is optional.
*/

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

uniform sampler1D models;
const int num = 20;

void main(void)
{
	vec4 pos = vec4(0);
	int i , ii;
	for( i=0; i<num ; i++) {
		for( ii=0 ; ii<3; ii++ ) {
			pos.xyz = texelFetch(models,i*3+ii,0).xyz;
			gl_Position = gl_PositionIn[0] + pos;
			gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
			gl_FrontColor = vec4(1,float(i)/float(num),0,1);
			EmitVertex();
		}
		EndPrimitive();
	}
}

