#include <list>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

struct point;

typedef list<point*> pList;
typedef unsigned char uint8_t;

struct point
{
	float x, y, z;
	float m, r;
	float vx, vy, vz;
	uint8_t mid;

	pList neighbours;
	point( float _x, float _y, float _z, float _m, float _r, float _vx=0, float _vy=0, float _vz=0, uint8_t _mid=0)
		: x(_x), y(_y), z(_z), m(_m), r(_r), vx(_vx), vy(_vy), vz(_vz), mid(_mid) {}
	point( const point &p1, const point &p2 )
		: x( (p1.x + p2.x) / 2 )
		, y( (p1.y + p2.y) / 2 )
		, z( (p1.z + p2.z) / 2 )
		, m( (p1.m + p2.m) / 4 )
		, r( (p1.r + p2.r) / 4 )
	{}
};

#define VIter(i, c) for( pList::iterator i = c.begin(); i != c.end(); ++i )
class graph
{
	public:
		graph(){}
		~graph(){ VIter(i, vertices) delete *i; }

		void add( const point &p )
		{
			vertices.push_back( new point( p ) );
		}

		void add_n_connect( const point &p )
		{
			point *pp = new point( p );
			VIter(i, vertices)
			{
				pp->neighbours.push_back( *i );
				(*i)->neighbours.push_back( pp );
			}
			vertices.push_back( pp );
		}

		void print()
		{
			printf("BEGIN TRANSACTION;\n");
			VIter(i, vertices)
			{
				point *p = *i;
				printf("INSERT INTO planets VALUES(%f, %f, %f, %f, %f, %f, %f, %f, %u);\n", 
					p->x, p->y, p->z, p->r, p->m, p->vx, p->vy, p->vz, p->mid);
			}
			printf("COMMIT;\n");
		}

		void iterate(unsigned depth)
		{
			for( unsigned i = 0; i < depth; ++i )
			{
				pList old_v = vertices;
				vertices.clear();
				VIter( v, old_v )
				{
					VIter( n, (*v)->neighbours )
					{
						if( *v < *n )
							add_n_connect( point( **v, **n ) );
					}
				}
				print();
			}
		}

	private:
		pList vertices;
};

void cube_gen(int dist, unsigned edge)
{
	graph g;
	const int half_edge = edge / 2;
	for( int x = 0; x < edge; ++x )
	{
		for( int y = 0; y < edge; ++y )
		{
			for( int z = 0; z < edge; ++z )
			{
				unsigned mass = rand() % (1 << 5) + 1;
				bool star = (0 == rand() % (edge));
				bool sun  = (0 == rand() % (edge * edge));
				if( star ) mass *= 1e2 + ( (rand() % 8) >> 1 );
				if( sun  ) mass *= 1e6;
				g.add( point( dist * (x - half_edge), dist * (y - half_edge), dist * (z - half_edge), mass, pow(mass, 0.3),
					4.f * edge * rand() / RAND_MAX,
					4.f * edge * rand() / RAND_MAX,
					4.f * edge * rand() / RAND_MAX,
					sun ? mass>(1<<4) ? 5 : 4 : ( x + y + z ) % 4
					) );
			}
		}
	}
	for( int i = 0; i < 0; ++i )
	{
		g.add( point( dist * i, dist * i, dist * i, 10, 2, 0,0,0,i%100?i%3:3 ) );
	}
	g.print();
}

void spiral_gen()
{
	graph g;
	for( unsigned i = 0; i < 1337; ++i )
	{
		bool star = (0 == i % 100);
		unsigned j = 10 * i;
		unsigned mass = rand() % 40 + 1;
		if( star ) mass *= 1e3;
		g.add( point(
			j * sin( 1 + i * .04),
			j * sin( 1 + i * .06),
			j * sin( 1 + i * .1),
			mass,
			pow( mass, 0.2 ),
			.01f * rand() / RAND_MAX,
			.01f * rand() / RAND_MAX,
			.01f * rand() / RAND_MAX,
			star ? 5 : i % 4) );
	}
	g.print();
}
#define PI 3.14159265
#define RAND_VEL ((20./mass) * rand() / RAND_MAX)
void disk_gen( unsigned count, float r )
{
	const float concentration = 0.6;
	graph g;
	for( unsigned i = 0; i < count; ++i )
	{
		float act_r = r * pow( 1. * rand() / RAND_MAX, concentration );
		float act_angle = 2 * PI * rand() / RAND_MAX;
		unsigned mass = rand() % (1 << 5) + 1;
		bool star = (0 == rand() % (unsigned)pow(count, 0.7));
		if( star ) mass *= 1e4;
		float x = act_r * sin( act_angle );
		float y = act_r * cos( act_angle );
		float vx = r * y/(17 * (r-act_r + r/2)) + RAND_VEL;
		float vy = r * (-x)/(17 * (r-act_r + r/2)) + RAND_VEL;
		float vz = RAND_VEL;
		g.add( point( x, y, 0, mass, pow(mass, 0.3),
			vx, vy, vz, star ? 5 : rand() % 4 ) );	
	}
	g.print();
}

int main()
{
	srand( time( NULL ) );
	/*g.add_n_connect( point( 100, -50, -20, 10, 10) );
	g.add_n_connect( point( -100, -50, -20, 10, 10) );
	g.add_n_connect( point( 0, 50, -20, 10, 10) );
	g.add_n_connect( point( 0, 0, 80, 10, 10) );
	g.print();
	g.iterate( 4 );*/

//        cube_gen(2500, 20);
	//spiral_gen();
	disk_gen( 20000, 1000 );
	
	return 0;
}
