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
#include <tuple>
#include <queue>
#include <stack>

#include <meto/mesh.h>



class Engine_LR {
public:

    // control operations
    enum OP {
        OP_L = 0, // left face visited, move to right
        OP_R, // right face visited, move to left
        OP_BOM, // begin of a submesh
        OP_NUM, // total number of OPs
    };

    Engine_LR(int discrete_bins=256, bool verbose=false) {
        this->discrete_bins = discrete_bins;
        this->verbose = verbose;
    }

    // quantization vertex from [-1, 1] into [0, discrete_bins-1]
    int discrete_bins;
    bool verbose;

    // results holder
    vector<int> tokens;
    vector<int> face_order;
    vector<int> face_type;
    int num_submesh = 0;
    int num_faces = 0;
    
    // map relative coordinate to non-negative integer
    int offset_coord(int x) {
        if (x < -discrete_bins || x >= discrete_bins) return -1; // out-of-bound value, will handle it later, just mark it as -1
        return x + discrete_bins + OP_NUM;
    }

    int restore_coord(int x) {
        if (x < 0) return x; // out-of-bound value, just keep it for later processing
        return x - discrete_bins - OP_NUM;
    }

    // each compress_face we only mark one face and write one vertex
    void compress_face(HalfEdge* c, bool init=false) {
        // mark face visible
        c->t->m = 1;
        face_order.push_back(c->t->i);
        num_faces++;
       
        // c->o is assured to exist if c is not the init face
        if (!init) {
            // examine face orientation (c and c->o should be opposite)
            if (!(c->s->i == c->o->e->i && c->e->i == c->o->s->i)) {
                // this means triangle c->t is wrongly oriented, we have to fix it for correct encoding
                if (verbose) cout << "[ENCODE] detected conflicting face orientation, flip face: " << c->t->i << endl;
                c->t->flip();
            }
            // parallelogram correction of vertex
            tokens.push_back(offset_coord(c->v->x + c->o->v->x - c->n->v->x - c->p->v->x));
            tokens.push_back(offset_coord(c->v->y + c->o->v->y - c->n->v->y - c->p->v->y));
            tokens.push_back(offset_coord(c->v->z + c->o->v->z - c->n->v->z - c->p->v->z));
        }

        bool tip_visited = c->v->m;
        bool left_visited = c->p->o == NULL || c->p->o->t->m;
        bool right_visited = c->n->o == NULL || c->n->o->t->m;

        if (verbose) {
            cout << "[ENCODE] visit face: " << c->t->i << " (" \
             << c->v->i << (c->v->m ? "v" : "o") << ", " \
             << c->s->i << (c->s->m ? "v" : "o") << ", " \
             << c->e->i << (c->e->m ? "v" : "o") << "), OP = [" \
             << (tip_visited ? "V" : " ") << (left_visited ? "L" : " ") << (right_visited ? "R" : " ") << "]" << endl;
        }

        // decide the OP type
        if (!tip_visited) {
            // OP C, but we'll merge it with OP L
            c->v->m = 1; // mark vertex visible
            tokens.push_back(OP_L);
            face_type.push_back(OP_L);
            compress_face(c->n->o);
        } else if (left_visited && right_visited) { 
            // OP E, but we'll merge it with OP BOM
            face_type.push_back(OP_BOM); // although we don't need a token, we still push to face_type so it has the same length as face_order
            return; // end of recursion
        } else if (left_visited) {
            // OP L (move to right)
            tokens.push_back(OP_L);
            face_type.push_back(OP_L);
            compress_face(c->n->o);
        } else if (right_visited) {
            // OP R (move to left)
            tokens.push_back(OP_R);
            face_type.push_back(OP_R);
            compress_face(c->p->o);
        } else {
            // OP S, but we only do one side now! the other side will be processed as a new submesh.
            tokens.push_back(OP_L);
            face_type.push_back(OP_L);
            compress_face(c->n->o);
            compress_submesh(c->p->o);
        }
    }

    void compress_submesh(HalfEdge* c) {

        // begin of a submesh
        tokens.push_back(OP_BOM);

        if (verbose) cout << "[ENCODE] Submesh start: " << num_submesh++ << endl;

        // init face count
        num_faces = 0;

        // write the first 3 vertices
        tokens.push_back(offset_coord(c->v->x));
        tokens.push_back(offset_coord(c->v->y));
        tokens.push_back(offset_coord(c->v->z));
        tokens.push_back(offset_coord(c->s->x - c->v->x));
        tokens.push_back(offset_coord(c->s->y - c->v->y));
        tokens.push_back(offset_coord(c->s->z - c->v->z));
        tokens.push_back(offset_coord(c->e->x - c->s->x));
        tokens.push_back(offset_coord(c->e->y - c->s->y));
        tokens.push_back(offset_coord(c->e->z - c->s->z));

        // mark vertex as visible
        c->s->m = 1;
        c->e->m = 1;

        // recursive compress
        compress_face(c, true);
    }

    tuple<vector<int>, vector<int>, vector<int>> encode(vector<vector<float>> vertices, vector<vector<int>> triangles) {

        // build mesh
        Mesh* mesh = new Mesh(vertices, triangles, discrete_bins, verbose);
        
        // init
        tokens.clear();
        face_order.clear();
        face_type.clear();
        num_submesh = 0;

        // loop until all faces are visited
        // this can also handle the "hole" and "handle" cases...
        for (size_t i = 0; i < mesh->faces.size(); i++) {
            Facet* f = mesh->faces[i];
            if (f->m) continue;
            // half edges already sorted for heuristics order
            compress_submesh(f->half_edges[0]);
        }

        return make_tuple(tokens, face_order, face_type);
    }

    // decode the tokens
    tuple<vector<vector<float>>, vector<vector<int>>, vector<int>> decode(vector<int> tokens) {
        // mesh 
        vector<vector<float>> vertices;
        vector<vector<int>> faces;
        face_type.clear();

        // keep record of necessary points to restore corodinates
        Vertex v0, v1, v2, v, delta;
        int num_vertices = 0; // number of vertices written
        int num_faces = 0; // number of faces written
        int num_submesh = 0;
        
        for (size_t i = 0; i < tokens.size(); i++) {
            if (tokens[i] == OP_BOM) {
                if (i + 9 >= tokens.size()) {
                    if (verbose) cout << "[DECODE] ERROR: incomplete face at " << i << endl;
                    break;
                }
                if (verbose) {
                    if (num_submesh > 0) cout << "[DECODE] Submesh end: " << num_submesh++ << endl;
                    cout << "[DECODE] Submesh start: " << num_submesh << endl;
                }
                // begin of a submesh: read 3 consecutive vertices
                v0 = {restore_coord(tokens[i+1]), restore_coord(tokens[i+2]), restore_coord(tokens[i+3]), num_vertices++};
                v1 = {v0.x + restore_coord(tokens[i+4]), v0.y + restore_coord(tokens[i+5]), v0.z + restore_coord(tokens[i+6]), num_vertices++};
                v2 = {v1.x + restore_coord(tokens[i+7]), v1.y + restore_coord(tokens[i+8]), v1.z + restore_coord(tokens[i+9]), num_vertices++};
                // write to vertices
                vertices.push_back(v0.undiscrete(discrete_bins));
                vertices.push_back(v1.undiscrete(discrete_bins));
                vertices.push_back(v2.undiscrete(discrete_bins));
                // write the first triangle
                faces.push_back({v0.i, v1.i, v2.i});
                if (i != 0) face_type.push_back(OP_BOM);
                if (verbose) cout << "[DECODE] Add Init face: " << num_faces++ << " = [" << v0.i << ", " << v1.i << ", " << v2.i << "]" << endl;
                // move index
                i += 9;
            } else {
                // tokens[i] should be an OP
                if (tokens[i] >= OP_NUM) {
                    if (verbose) cout << "[DECODE] ERROR: position should be OP at " << i << endl;
                    break;
                }
                // now the following 3 tokens will be the new vertex
                if (i + 3 >= tokens.size()) {
                    if (verbose) cout << "[DECODE] ERROR: incomplete vertex at " << i << endl;
                    break;
                }
                delta = {restore_coord(tokens[i+1]), restore_coord(tokens[i+2]), restore_coord(tokens[i+3])};
                if (tokens[i] == OP_L) {
                    // move to right
                    v = v0 + v2 - v1 + delta;
                    v.i = num_vertices++;
                    vertices.push_back(v.undiscrete(discrete_bins));
                    faces.push_back({v.i, v0.i, v2.i});
                    if (verbose) cout << "[DECODE] Add R face: " << num_faces++ << " = [" << v.i << ", " << v0.i << ", " << v2.i << "]" << endl;
                    // update v0, v1, v2
                    v1 = v0;
                    v0 = v;
                } else if (tokens[i] == OP_R) {
                    // move to left
                    v = v0 + v1 - v2 + delta;
                    v.i = num_vertices++;
                    vertices.push_back(v.undiscrete(discrete_bins));
                    faces.push_back({v.i, v1.i, v0.i});
                    if (verbose) cout << "[DECODE] Add L face: " << num_faces++ << " = [" << v.i << ", " << v1.i << ", " << v0.i << "]" << endl;
                    // update v0, v1, v2
                    v2 = v0;
                    v0 = v;
                }
                face_type.push_back(tokens[i]);
                i += 3;
            }
        }
        face_type.push_back(OP_BOM); // last face
        return make_tuple(vertices, faces, face_type);
    }
};