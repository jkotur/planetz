#include "sphere.h"
#include "../util/logger.h"

using namespace std;

map<int, SphereModel*> Sphere::object_tab;

SphereModel* Sphere::get_obj(int precision)
{
	log_printf(INFO, "Loading sphere...\n");
	if(object_tab.find(precision) == object_tab.end())
		object_tab[precision] = generate(precision);
	log_printf(INFO, "Sphere loaded.\n");
	return object_tab[precision];
}

SphereModel* Sphere::generate(int precision)
{
	log_printf(INFO, "Generating...");
	vcomparer vc;
	map<Vector3, Vector3, vcomparer> *texture_to_sphere = new map<Vector3, Vector3, vcomparer>();
	map<Vector3, set<Vector3, vcomparer>, vcomparer> *flat_adj = new map<Vector3, set<Vector3, vcomparer>, vcomparer>();
	map<Vector3, set<Vector3, vcomparer>, vcomparer> *sphere_adj= new map<Vector3, set<Vector3, vcomparer>, vcomparer>;
	init_icosahedron(texture_to_sphere, sphere_adj, flat_adj);
	for(int i = 0; i < precision; ++i)
	{
		map<Vector3, Vector3, vcomparer> *tmp_tts = new map<Vector3, Vector3, vcomparer>();
		map<Vector3, set<Vector3, vcomparer>, vcomparer> *tmp_fa = new map<Vector3, set<Vector3, vcomparer>, vcomparer>();
		map<Vector3, set<Vector3, vcomparer>, vcomparer> *tmp_sa = new map<Vector3, set<Vector3, vcomparer>, vcomparer>();
		for(map<Vector3, Vector3, vcomparer>::iterator it1 = texture_to_sphere->begin(); it1 != texture_to_sphere->end(); ++it1)
			for(set<Vector3, vcomparer>::iterator it2 = (*flat_adj)[it1->first].begin(); it2 != (*flat_adj)[it1->first].end(); ++it2)
				if(vc(it1->first, *it2))
					for(set<Vector3, vcomparer>::iterator it3 = (*flat_adj)[it1->first].begin(); it3 != (*flat_adj)[it1->first].end(); ++it3)
						if(vc(*it2, *it3))
							for(set<Vector3, vcomparer>::iterator it4 = (*flat_adj)[*it2].begin(); it4 != (*flat_adj)[*it2].end(); ++it4)
								if(*it4 == *it3)
								{
									Vector3 tp[6] = {it1->first, *it2, *it3},
										sp[6] = {(*texture_to_sphere)[tp[0]], (*texture_to_sphere)[tp[1]], (*texture_to_sphere)[tp[2]]};
									tp[3] = (tp[0] + tp[1])/2;
									tp[4] = (tp[1] + tp[2])/2;
									tp[5] = (tp[0] + tp[2])/2;
									sp[3] = (sp[0] + sp[1]); sp[3].normalize();
									sp[4] = (sp[1] + sp[2]); sp[4].normalize();
									sp[5] = (sp[0] + sp[2]); sp[5].normalize();
								
									for(int j = 0; j < 6; ++j)
										(*tmp_tts)[tp[j]] = sp[j];
								
									(*tmp_fa)[tp[0]].insert(tp[3]); (*tmp_fa)[tp[0]].insert(tp[5]);
									(*tmp_fa)[tp[1]].insert(tp[3]); (*tmp_fa)[tp[1]].insert(tp[4]);
									(*tmp_fa)[tp[2]].insert(tp[4]); (*tmp_fa)[tp[2]].insert(tp[5]);
									(*tmp_fa)[tp[3]].insert(tp[0]); (*tmp_fa)[tp[3]].insert(tp[1]); (*tmp_fa)[tp[3]].insert(tp[4]); (*tmp_fa)[tp[3]].insert(tp[5]);
									(*tmp_fa)[tp[4]].insert(tp[1]); (*tmp_fa)[tp[4]].insert(tp[2]); (*tmp_fa)[tp[4]].insert(tp[3]); (*tmp_fa)[tp[4]].insert(tp[5]);
									(*tmp_fa)[tp[5]].insert(tp[0]); (*tmp_fa)[tp[5]].insert(tp[2]); (*tmp_fa)[tp[5]].insert(tp[3]); (*tmp_fa)[tp[5]].insert(tp[4]);
								
									(*tmp_sa)[tp[0]].insert(tp[3]); (*tmp_sa)[tp[0]].insert(tp[5]);
									(*tmp_sa)[tp[1]].insert(tp[3]); (*tmp_sa)[tp[1]].insert(tp[4]);
									(*tmp_sa)[tp[2]].insert(tp[4]); (*tmp_sa)[tp[2]].insert(tp[5]);
									(*tmp_sa)[tp[3]].insert(tp[0]); (*tmp_sa)[tp[3]].insert(tp[1]); (*tmp_sa)[tp[3]].insert(tp[4]); (*tmp_sa)[tp[3]].insert(tp[5]);
									(*tmp_sa)[tp[4]].insert(tp[1]); (*tmp_sa)[tp[4]].insert(tp[2]); (*tmp_sa)[tp[4]].insert(tp[3]); (*tmp_sa)[tp[4]].insert(tp[5]);
									(*tmp_sa)[tp[5]].insert(tp[0]); (*tmp_sa)[tp[5]].insert(tp[2]); (*tmp_sa)[tp[5]].insert(tp[3]); (*tmp_sa)[tp[5]].insert(tp[4]);
								}
		map<Vector3, Vector3, vcomparer> *exchg1;
		map<Vector3, set<Vector3, vcomparer>, vcomparer> *exchg2;
		exchg1 = texture_to_sphere;
		texture_to_sphere = tmp_tts;
		delete exchg1;
		exchg2 = flat_adj;
		flat_adj = tmp_fa;
		delete exchg2;
		exchg2 = sphere_adj;
		sphere_adj = tmp_sa;
		delete exchg2;
	}
	SphereModel* ret = new SphereModel(
		(10 << (precision << 1)) + 2,
		20 << (precision << 1),
		texture_to_sphere->size());
	map<Vector3, int, vcomparer> tvti; //texture vector to int
	map<Vector3, int, vcomparer> svti; //sphere vector to int
	int tid = 0, sid = 0, trid = 0;
	for(map<Vector3, Vector3, vcomparer>::iterator it1 = texture_to_sphere->begin(); it1 != texture_to_sphere->end(); ++it1)
		for(set<Vector3, vcomparer>::iterator it2 = (*flat_adj)[it1->first].begin(); it2 != (*flat_adj)[it1->first].end(); ++it2)
			if(vc(it1->first, *it2))
				for(set<Vector3, vcomparer>::iterator it3 = (*flat_adj)[it1->first].begin(); it3 != (*flat_adj)[it1->first].end(); ++it3)
					if(vc(*it2, *it3))
						for(set<Vector3, vcomparer>::iterator it4 = (*flat_adj)[*it2].begin(); it4 != (*flat_adj)[*it2].end(); ++it4)
							if(*it4 == *it3)
							{
								Vector3 tp[3] = {it1->first, *it2, *it3},
									sp[3] = {(*texture_to_sphere)[tp[0]], (*texture_to_sphere)[tp[1]], (*texture_to_sphere)[tp[2]]};
								for(int i = 0; i < 3; ++i)
								{
									if(tvti.find(tp[i]) == tvti.end())
										tvti[tp[i]] = tid++;
									if(svti.find(sp[i]) == svti.end())
										svti[sp[i]] = sid++;
								}
								ret->triangles[trid] = Triple(svti[sp[0]], svti[sp[1]], svti[sp[2]]);
								ret->texture_triangles[trid++] = Triple(tvti[tp[0]], tvti[tp[1]], tvti[tp[2]]);
							}
	for(map<Vector3, int, vcomparer>::iterator it = tvti.begin(); it!= tvti.end(); ++it)
	{
		ret->texture_points[it->second] = it->first;
	}
	for(map<Vector3, int, vcomparer>::iterator it = svti.begin(); it!= svti.end(); ++it)
	{
		ret->points[it->second] = it->first;
		ret->normals[it->second] = it->first;
	}
	delete texture_to_sphere;
	delete flat_adj;
	delete sphere_adj;
	log_printf(INFO, "Done.\n");
	return ret;
}

void Sphere::init_icosahedron(
	map<Vector3, Vector3, vcomparer>*& tts,
	map<Vector3, set<Vector3, vcomparer>, vcomparer>*& sa,
	map<Vector3, set<Vector3, vcomparer>, vcomparer>*& fa)
{
	const double nrm = 1 / sqrt((5.0 + sqrt(5.0)) / 2.0);
	const double fi = nrm * (sqrt(5.0) + 1) / 2;
	const double edge_s = sqrt(106.0) / 45.0;
	const double edge_v = 2.0 / 9.0;
	
	(*tts)[Vector3(0.2, 0.0 / 9.0, 0)] = Vector3(-nrm, -fi, 0);
	(*tts)[Vector3(0.4, 1.0 / 9.0, 0)] = Vector3(-nrm, -fi, 0);
	(*tts)[Vector3(0.6, 2.0 / 9.0, 0)] = Vector3(-nrm, -fi, 0);
	(*tts)[Vector3(0.8, 3.0 / 9.0, 0)] = Vector3(-nrm, -fi, 0);
	(*tts)[Vector3(1.0, 4.0 / 9.0, 0)] = Vector3(-nrm, -fi, 0);
	(*tts)[Vector3(0.0, 1.0 / 9.0, 0)] = Vector3(nrm, -fi, 0);
	(*tts)[Vector3(1.0, 6.0 / 9.0, 0)] = Vector3(nrm, -fi, 0);
	(*tts)[Vector3(0.2, 2.0 / 9.0, 0)] = Vector3(0, -nrm, fi);
	(*tts)[Vector3(0.4, 3.0 / 9.0, 0)] = Vector3(-fi, 0, nrm);
	(*tts)[Vector3(0.6, 4.0 / 9.0, 0)] = Vector3(-fi, 0, -nrm);
	(*tts)[Vector3(0.8, 5.0 / 9.0, 0)] = Vector3(0, -nrm, -fi);
	(*tts)[Vector3(0.0, 3.0 / 9.0, 0)] = Vector3(fi, 0, nrm);
	(*tts)[Vector3(1.0, 8.0 / 9.0, 0)] = Vector3(fi, 0, nrm);
	(*tts)[Vector3(0.2, 4.0 / 9.0, 0)] = Vector3(0, nrm, fi);
	(*tts)[Vector3(0.4, 5.0 / 9.0, 0)] = Vector3(-nrm, fi, 0);
	(*tts)[Vector3(0.6, 6.0 / 9.0, 0)] = Vector3(0, nrm, -fi);
	(*tts)[Vector3(0.8, 7.0 / 9.0, 0)] = Vector3(fi, 0, -nrm);
	(*tts)[Vector3(0.0, 5.0 / 9.0, 0)] = Vector3(nrm, fi, 0);
	(*tts)[Vector3(0.2, 6.0 / 9.0, 0)] = Vector3(nrm, fi, 0);
	(*tts)[Vector3(0.4, 7.0 / 9.0, 0)] = Vector3(nrm, fi, 0);
	(*tts)[Vector3(0.6, 8.0 / 9.0, 0)] = Vector3(nrm, fi, 0);
	(*tts)[Vector3(0.8, 9.0 / 9.0, 0)] = Vector3(nrm, fi, 0);
	
	for(map<Vector3, Vector3, vcomparer>::iterator it1 = tts->begin(); it1 != tts->end(); ++it1)
		for(map<Vector3, Vector3, vcomparer>::iterator it2 = tts->begin(); it2 != tts->end(); ++it2)
			for(map<Vector3, Vector3, vcomparer>::iterator it3 = tts->begin(); it3 != tts->end(); ++it3)
			{
				if(triangle_test(it1->first, it2->first, it3->first, edge_v, edge_s, tts))
				{
					(*fa)[it1->first].insert(it2->first);
					(*fa)[it1->first].insert(it3->first);
					(*sa)[it1->second].insert(it2->second);
					(*sa)[it1->second].insert(it3->second);
				}
			}
}

bool Sphere::triangle_test(Vector3 v1, Vector3 v2, Vector3 v3, double edge_v, double edge_s, map<Vector3, Vector3, vcomparer>*& tts)
{
	return 
		((abs(v1.x - v2.x) < VECTOREPSILON &&
		abs(abs(v1.y - v2.y) - edge_v) < VECTOREPSILON &&
		v1.distance(v3) - edge_s < VECTOREPSILON &&
		v2.distance(v3) - edge_s < VECTOREPSILON) ||
		(abs(v2.x - v3.x) < VECTOREPSILON &&
		abs(abs(v2.y - v3.y) - edge_v) < VECTOREPSILON &&
		v2.distance(v1) - edge_s < VECTOREPSILON &&
		v3.distance(v1) - edge_s < VECTOREPSILON) ||
		(abs(v3.x - v1.x) < VECTOREPSILON &&
		abs(abs(v3.y - v1.y) - edge_v) < VECTOREPSILON &&
		v3.distance(v2) - edge_s < VECTOREPSILON &&
		v1.distance(v2) - edge_s < VECTOREPSILON)) &&
		((*tts)[v1] != (*tts)[v2] &&
		(*tts)[v1] != (*tts)[v3] &&
		(*tts)[v2] != (*tts)[v3]);
}
