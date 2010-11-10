#include "sphere_converter.h"

#include <cstdio>
#include <cstring>
#include <set>

#include "sphere.h"

#include "debug/routines.h"

using namespace std;

int unionSize( const Triple& a , const Triple& b )
{
	set<int> both;
	both.insert(a.p1);
	both.insert(a.p2);
	both.insert(a.p3);
	both.insert(b.p1);
	both.insert(b.p2);
	both.insert(b.p3);
	return both.size();
}

void SphereConv::toTriangleStrip( float*vert , float*texCoord )
{
	TODO("End this somtime, problem is whith texturecoords");

	int tcount = sm.get_triangles_count();

	bool**graph = new bool*[tcount];
	for( int i = 0 ; i < tcount; i++ )
	{
		graph[i] = new bool[tcount];
		memset( graph[i] , 0 , sizeof(bool)*tcount );
	}

	
	// create graph matrix
	for(int i = 0; i < tcount ; ++i)
	{
		Triple ti = sm.get_triangle(i);
		for(int ii = 0; ii < tcount ; ++ii)
		{
			Triple tii = sm.get_triangle(ii);
			if( unionSize( ti , tii ) == 4 )
				// two shared vertices
				graph[i][ii] = true;
		}
	}

	fprintf(stderr,"----------\n");
	for(int i = 0; i < tcount ; ++i) {
		for(int ii = 0; ii < tcount ; ++ii)
			fprintf(stderr,"%c, ",graph[i][ii]?'X':' ');
		fprintf(stderr,"\n");
	}
	fprintf(stderr,"----------\n");

	/*
	int curr = 0; // current triangle
	Triple tcurr = sm.get_triangle(0);
	for( int i = 0 ; i < tcount ; i++ ) 
	{
		int nebs[3];

		// find neighbours
		for( int j=0 , k=0 ; j<tcount ; j++ )
			if( graph[i][j] ) {
				ASSERT( k < 3 ); // max 3 neighbours
				nebs[k++] = j;
			}

		Triple tnebs[3];
		for( int k=0 ; i<3 ; k++ )
			tnebs[i] = sm.get_triangle( nebs[0] );

	}
	*/

	for( int i = 0 ; i < tcount; i++ )
		delete[]graph[i];
	delete[]graph;
}

void SphereConv::toTriangles( float*vert , float*texCoord )
{
	Vector3 res;
	for(int i = 0; i < sm.get_triangles_count(); ++i)
	{
		res=sm.get_texture_point( sm.get_texture_triangle(i).p1 );
		texCoord[i*6  ] = res.x;
		texCoord[i*6+1] = res.y;
		res=sm.get_point( sm.get_triangle(i).p1 );
		vert[i*9  ] = res.x;
		vert[i*9+1] = res.y;
		vert[i*9+2] = res.z;
		
		res=sm.get_texture_point( sm.get_texture_triangle(i).p2 );
		texCoord[i*6+2] = res.x;
		texCoord[i*6+3] = res.y;
		res=sm.get_point( sm.get_triangle(i).p2 );
		vert[i*9+3] = res.x;
		vert[i*9+4] = res.y;
		vert[i*9+5] = res.z;
		
		res=sm.get_texture_point( sm.get_texture_triangle(i).p3 );
		texCoord[i*6+4] = res.x;
		texCoord[i*6+5] = res.y;
		res=sm.get_point( sm.get_triangle(i).p3 );
		vert[i*9+6] = res.x;
		vert[i*9+7] = res.y;
		vert[i*9+8] = res.z;
	}
}

