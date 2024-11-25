/*
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
*/

#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <cmath>

using namespace std;

struct Vertex {
    int x, y, z; // discretized coordinates
    int i; // index
    int m = 0; // visited mark
    
    Vertex() {}
    Vertex(int x, int y, int z, int i=-1) : x(x), y(y), z(z), i(i) {}
    Vertex(float x, float y, float z, int discrete_bins, int i = -1) {
        this->x = min(int((x + 1) * discrete_bins / 2), discrete_bins - 1);
        this->y = min(int((y + 1) * discrete_bins / 2), discrete_bins - 1);
        this->z = min(int((z + 1) * discrete_bins / 2), discrete_bins - 1);
        this->i = i;
    }

    // undiscretize
    vector<float> undiscrete(int discrete_bins) {
        return {
            float((float(x) + 0.5) / discrete_bins * 2 - 1), 
            float((float(y) + 0.5) / discrete_bins * 2 - 1), 
            float((float(z) + 0.5) / discrete_bins * 2 - 1)
        };
    }

    // operators
    Vertex operator+(const Vertex& v) const {
        return Vertex(x + v.x, y + v.y, z + v.z);
    }
    Vertex operator-(const Vertex& v) const {
        return Vertex(x - v.x, y - v.y, z - v.z);
    }
    bool operator==(const Vertex& v) const {
        return x == v.x && y == v.y && z == v.z;
    }
    bool operator<(const Vertex& v) const {
        // y-z-x order
        return y < v.y || (y == v.y && z < v.z) || (y == v.y && z == v.z && x < v.x);
    }
};

struct Vector3f {
    float x, y, z;
    Vector3f() {}
    Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}
    Vector3f(const Vertex& v) : x(v.x), y(v.y), z(v.z) {}
    Vector3f(const Vertex& v1, const Vertex& v2) : x(v2.x - v1.x), y(v2.y - v1.y), z(v2.z - v1.z) {}
    Vector3f operator+(const Vector3f& v) const {
        return Vector3f(x + v.x, y + v.y, z + v.z);
    }
    Vector3f operator-(const Vector3f& v) const {
        return Vector3f(x - v.x, y - v.y, z - v.z);
    }
    Vector3f operator*(float s) const {
        return Vector3f(x * s, y * s, z * s);
    }
    Vector3f operator/(float s) const {
        return Vector3f(x / s, y / s, z / s);
    }
    bool operator==(const Vector3f& v) const {
        return x == v.x && y == v.y && z == v.z;
    }
    bool operator<(const Vector3f& v) const {
        // y-z-x order
        return y < v.y || (y == v.y && z < v.z) || (y == v.y && z == v.z && x < v.x);
    }
    Vector3f cross(const Vector3f& v) const {
        return Vector3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    float dot(const Vector3f& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    float norm() const {
        return sqrt(x * x + y * y + z * z);
    }
    Vector3f normalize() const {
        float n = norm();
        return Vector3f(x / n, y / n, z / n);
    }
};

struct Facet; // pre-declare

struct HalfEdge {
    Vertex* v = NULL; // opposite vertex
    Vertex* s = NULL; // start vertex
    Vertex* e = NULL; // end vertex
    Facet* t = NULL; // triangle
    HalfEdge* n = NULL; // next half edge
    HalfEdge* p = NULL; // previous half edge
    HalfEdge* o = NULL; // opposite half edge (maybe NULL if at boundary!)
    int i = -1; // index

    // comparison operator
    bool operator<(const HalfEdge& e) const {
        // boundary edge first, then by distance to the opposite vertex
        if (o == NULL) return true;
        else if (e.o == NULL) return false;
        else return Vector3f(*v, *o->v).norm() < Vector3f(*e.v, *e.o->v).norm();
    }
};

struct Facet {
    Vertex* vertices[3];
    HalfEdge* half_edges[3];
    int i = -1; // index
    int ic = -1; // component index
    int m = 0; // visited mark

    Vector3f center; // mass center

    // flip the face orientation (only flip half edges)
    void flip() {
        for (int i = 0; i < 3; i++) {
            swap(half_edges[i]->s, half_edges[i]->e);
            swap(half_edges[i]->n, half_edges[i]->p);
        }
    }

    // comparison operator
    bool operator<(const Facet& f) const {
        // first by connected component, then by center
        if (ic != f.ic) return ic < f.ic;
        else return center < f.center;
    }
};

pair<int, int> edge_key(int a, int b) {
    return a < b ? make_pair(a, b) : make_pair(b, a);
}

class Mesh {
public:

    // mesh data
    vector<Vertex*> verts;
    vector<Facet*> faces;
    bool verbose = false;
    
    // discretization bins 
    int discrete_bins;

    // Euler characteristic: V - E + F = 2 - 2g - b
    // number of bounding loops
    int num_vertices = 0;
    int num_edges = 0;
    int num_faces = 0;
    int num_components = 0;
    bool non_manifold = false;

    Mesh(vector<vector<float>> vertices, vector<vector<int>> triangles, int discrete_bins = 256, bool verbose = false) {

        this->discrete_bins = discrete_bins;
        this->verbose = verbose;

        // discretize vertices (assume in [-1, 1], we won't do error handling in cpp!)
        for (size_t i = 0; i < vertices.size(); i++) {
            Vertex* v = new Vertex(vertices[i][0], vertices[i][1], vertices[i][2], discrete_bins, i);
            verts.push_back(v);
        }
        num_vertices = verts.size();
       
        // build face and edge
        map<pair<int, int>, HalfEdge*> edge2halfedge; // to hold twin half edge
        for (size_t i = 0; i < triangles.size(); i++) {
            vector<int>& triangle = triangles[i];
            Facet* f = new Facet();
            f->i = i;
            // build half edge and link to verts
            for (int j = 0; j < 3; j++) {
                HalfEdge* e = new HalfEdge();
                e->v = verts[triangle[j]];
                e->s = verts[triangle[(j + 1) % 3]];
                e->e = verts[triangle[(j + 2) % 3]];
                e->t = f;
                e->i = j;
                f->vertices[j] = verts[triangle[j]];
                f->half_edges[j] = e;
                // link opposite half edge
                pair<int, int> key = edge_key(triangle[(j + 1) % 3], triangle[(j + 2) % 3]);
                if (edge2halfedge.find(key) == edge2halfedge.end()) {
                    edge2halfedge[key] = e;
                } else {
                    // if this key has already bound two half edges, this mesh is non-manifold!
                    if (edge2halfedge[key] == NULL) {
                        non_manifold = true;
                        // we can do nothing to fix it... treat it as a border edge
                        continue;
                    }
                    // twin half edge
                    e->o = edge2halfedge[key];
                    edge2halfedge[key]->o = e;
                    // using NULL to mark as completed
                    edge2halfedge[key] = NULL;
                }
            }
            // link prev and next half edges
            // assume each face's vertex ordering is counter-clockwise, so next = right, prev = left
            for (int j = 0; j < 3; j++) {
                f->half_edges[j]->n = f->half_edges[(j + 1) % 3];
                f->half_edges[j]->p = f->half_edges[(j + 2) % 3];
            }
            // compute face center
            f->center = Vector3f(
                float(f->vertices[0]->x + f->vertices[1]->x + f->vertices[2]->x) / 3.0,
                float(f->vertices[0]->y + f->vertices[1]->y + f->vertices[2]->y) / 3.0,
                float(f->vertices[0]->z + f->vertices[1]->z + f->vertices[2]->z) / 3.0
            );
            faces.push_back(f);
        }
        num_faces = faces.size();
        num_edges = edge2halfedge.size();

        for (size_t i = 0; i < triangles.size(); i++) {
            Facet* f = faces[i];
            for (int j = 0; j < 3; j++) {
                // boundary edges have no opposite half edge, and their start and end vertices are boundary vertices
                if (f->half_edges[j]->o == NULL) {
                    f->half_edges[j]->s->m = 1;
                    f->half_edges[j]->e->m = 1;
                    if (verbose) cout << "[MESH] Mark boundary vertex for face " << f->i << " : " << f->half_edges[j]->s->i << " -- " << f->half_edges[j]->e->i << endl;
                }
            }
            // sort half edges (this will disturb vertex-half-edge ordering, but we don't use index anyway)
            sort(f->half_edges, f->half_edges + 3, [](const HalfEdge* e1, const HalfEdge* e2) { return *e1 < *e2; });
        }

        // sort faces using center
        sort(faces.begin(), faces.end(), [](const Facet* f1, const Facet* f2) { return *f1 < *f2; });

        // find connected components
        for (size_t i = 0; i < triangles.size(); i++) {
            Facet* f = faces[i];
            if (f->ic == -1) {
                num_components++;
                if (verbose) cout << "[MESH] find connected component " << num_components << endl;
                // recursively mark all connected faces
                queue<Facet*> q;
                q.push(f);
                while (!q.empty()) {
                    Facet* f = q.front();
                    q.pop();
                    if (f->ic != -1) continue;
                    f->ic = num_components;
                    for (int j = 0; j < 3; j++) {
                        HalfEdge* e = f->half_edges[j];
                        if (e->o != NULL && e->o->t->ic == -1) {
                            q.push(e->o->t);
                        }
                    }
                }
            }
        }

        // sort faces again using connected component and center
        sort(faces.begin(), faces.end(), [](const Facet* f1, const Facet* f2) { return *f1 < *f2; });
    }

    ~Mesh() {
        for (Vertex* v : verts) { delete v; }
        for (Facet* f : faces) {
            for (HalfEdge* e : f->half_edges) { delete e; }
            delete f;
        }
    }
};