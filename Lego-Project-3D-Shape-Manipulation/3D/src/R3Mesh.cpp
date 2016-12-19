// Source file for mesh class



// Include files

#include "R3Mesh.h"
#include <iostream>

#define X_COORD 0
#define Y_COORD 1
#define Z_COORD 2

////////////////////////////////////////////////////////////
// METHODS TO IMPLEMENT -- Twist() (1st Project), Taubin, Loop (2nd Project)
////////////////////////////////////////////////////////////

using std::cout;
using std::endl;
using std::srand;
using std::rand;
using std::ifstream;

void R3Mesh::DeleteAllFaces(){
    // Delete all old faces
    vector< R3MeshFace* > facesToBeDeleted;
    for (int whichFace = 0; whichFace < NFaces(); whichFace++) {
        facesToBeDeleted.push_back(Face(whichFace));
    }
    for (unsigned int whichFace = 0; whichFace < facesToBeDeleted.size(); whichFace++) {
        DeleteFace(facesToBeDeleted[whichFace]);
    }
}

void R3Mesh::CreateTriangleAt(double xa, double ya, double za, 
                              double xb, double yb, double zb, 
                              double xc, double yc, double zc){
    
    srand(xa*ya*za*xb*yb*zb*xc*yb*zc); // seed the rand
    
    vector< R3MeshVertex* > facePoints;
    
    R3Point position = R3Point( xa, ya, za );
    R3Vector normal = R3Vector( rand(), rand(), rand() );
    R2Point texcoords = R2Point( rand(), rand() );
    
    facePoints.push_back( (new R3MeshVertex(position, normal, texcoords)) );
    
    position = R3Point( xb, yb, zb );
    normal = R3Vector( rand(), rand(), rand() );
    texcoords = R2Point( rand(), rand() );
    
    facePoints.push_back( (new R3MeshVertex(position, normal, texcoords)) );
    
    position = R3Point( xc, yc, zc );
    normal = R3Vector( rand(), rand(), rand() );
    texcoords = R2Point( rand(), rand() );
    
    facePoints.push_back( (new R3MeshVertex(position, normal, texcoords)) );
    
    vector< R3MeshVertex* > faceVector;
    for (unsigned int i = 0; i < facePoints.size(); i++) {
        R3MeshVertex* v = facePoints[i];
        CreateVertex(v->position, v->normal, v->texcoords);
        faceVector.push_back(vertices.back());
    }
    
    // Create Face
    CreateFace(faceVector);
}

void R3Mesh::CreateSquareAt(double xa, double ya, double za, 
                            double xb, double yb, double zb, 
                            double xc, double yc, double zc, 
                            double xd, double yd, double zd){
    // Assuming Square is in the format:
    //        a b
    //        c d
    
    CreateTriangleAt(xa, ya, za,
                     xb, yb, zb,
                     xc, yc, zc);
    CreateTriangleAt(xb, yb, zb,
                     xc, yc, zc,
                     xd, yd, zd);
    
}

void R3Mesh::CreateBoxAt(double x, double y, double z, double width){
    double hw = width/2; // hw = half_width
    CreateSquareAt(x-hw , y-hw, z-hw,
                   x+hw , y-hw, z-hw,
                   x-hw , y-hw, z+hw,
                   x+hw , y-hw, z+hw);
                   
    CreateSquareAt(x-hw , y-hw, z-hw,
                   x-hw , y-hw, z+hw,
                   x-hw , y+hw, z-hw,
                   x-hw , y+hw, z+hw);
                   
    CreateSquareAt(x-hw , y-hw, z-hw,
                   x+hw , y-hw, z-hw,
                   x-hw , y+hw, z-hw,
                   x+hw , y+hw, z-hw);
                   
    CreateSquareAt(x+hw , y+hw, z+hw,
                   x-hw , y+hw, z+hw,
                   x+hw , y+hw, z-hw,
                   x-hw , y+hw, z-hw);
                   
    CreateSquareAt(x+hw , y+hw, z+hw,
                   x+hw , y+hw, z-hw,
                   x+hw , y-hw, z+hw,
                   x+hw , y-hw, z-hw);
                   
    CreateSquareAt(x+hw , y+hw, z+hw,
                   x-hw , y+hw, z+hw,
                   x+hw , y-hw, z+hw,
                   x-hw , y-hw, z+hw);
                   
}
void R3Mesh::CreateDoubleBoxAt(double x0, double y0, double z0, double x1, double y1, double z1, double width){
    double hw = width/2; // hw = half_width
    double x_min = MIN(x0, x1);
    double y_min = MIN(y0, y1);
    double z_min = MIN(z0, z1);
    double x_max = MAX(x0, x1);
    double y_max = MAX(y0, y1);
    double z_max = MAX(z0, z1);
    
    CreateSquareAt(x_min-hw , y_min-hw, z_min-hw,
                   x_max+hw , y_min-hw, z_min-hw,
                   x_min-hw , y_min-hw, z_max+hw,
                   x_max+hw , y_min-hw, z_max+hw);
                   
    CreateSquareAt(x_min-hw , y_min-hw, z_min-hw,
                   x_min-hw , y_min-hw, z_max+hw,
                   x_min-hw , y_max+hw, z_min-hw,
                   x_min-hw , y_max+hw, z_max+hw);
                   
    CreateSquareAt(x_min-hw , y_min-hw, z_min-hw,
                   x_max+hw , y_min-hw, z_min-hw,
                   x_min-hw , y_max+hw, z_min-hw,
                   x_max+hw , y_max+hw, z_min-hw);
                   
    CreateSquareAt(x_max+hw , y_max+hw, z_max+hw,
                   x_min-hw , y_max+hw, z_max+hw,
                   x_max+hw , y_max+hw, z_min-hw,
                   x_min-hw , y_max+hw, z_min-hw);
                   
    CreateSquareAt(x_max+hw , y_max+hw, z_max+hw,
                   x_max+hw , y_max+hw, z_min-hw,
                   x_max+hw , y_min-hw, z_max+hw,
                   x_max+hw , y_min-hw, z_min-hw);
                   
    CreateSquareAt(x_max+hw , y_max+hw, z_max+hw,
                   x_min-hw , y_max+hw, z_max+hw,
                   x_max+hw , y_min-hw, z_max+hw,
                   x_min-hw , y_min-hw, z_max+hw);
}

void R3Mesh::Voxel(string file_name){
    DeleteAllFaces();
    
    ifstream stream1(file_name.c_str());
    string line;
    
    while (std::getline(stream1, line)) {
        int x = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int y = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int z = atoi( line.substr(0, line.find(" ")).c_str() );
        CreateBoxAt(x,y,z,0.5);
    }
    Update();
}

void R3Mesh::Singles(string file_name){
    Voxel(file_name);
}

void R3Mesh::Doubles(string file_name){
    DeleteAllFaces();
    
    ifstream stream1(file_name.c_str());
    string line;
    
    while (std::getline(stream1, line)) {
        int x0 = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int y0 = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int z0 = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int x1 = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int y1 = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int z1 = atoi( line.substr(0, line.find(" ")).c_str() );
        CreateDoubleBoxAt(x0,y0,z0,x1,y1,z1,0.5);
    }
    Update();
}

void R3Mesh::Lego(string singles_file_name, string doubles_file_name){
    Voxel(singles_file_name);
    
    ifstream stream1(doubles_file_name.c_str());
    string line;
    
    while (std::getline(stream1, line)) {
        int x0 = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int y0 = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int z0 = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int x1 = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int y1 = atoi( line.substr(0, line.find(" ")).c_str() );
        line.erase(0, line.find(" ")+1);
        int z1 = atoi( line.substr(0, line.find(" ")).c_str() );
        CreateDoubleBoxAt(x0,y0,z0,x1,y1,z1,0.5);
        //cout << " " << x0 << " " << y0 << " " << z0 << " " << x1 << " " << y1 << " " << z1 << endl;
    }
    Update();
}

void R3Mesh::
Twist(double angle)
{
  // Twist mesh by an angle, or other simple mesh warping of your choice.
  // See Scale() for how to get the vertex positions, and see bbox for the bounding box.
  
  // FILL IN IMPLEMENTATION HERE

  // Update mesh data structures
  Update();
}

void R3Mesh::Smooth(double factor){
    
    /* Make new copy of vertices */
    vector<R3MeshVertex *> newVertices(vertices);
    
    /* Get edges (ordered point pairs) and put into list */
    vector< vector< R3MeshVertex* > > edgeList; // 2D vector of R3Point values
    
    // iterate through faces
    for (int whichFace = 0; whichFace < NFaces(); whichFace++) { 
        R3MeshFace* face = Face(whichFace); // Current Face 
        R3MeshVertex* vertex1 = face->vertices.back();// Current vertex on current face 
        
        // for all vertices in the current face
        for (unsigned int i = 0; i < face->vertices.size(); i++) {
            R3MeshVertex* vertex2 = face->vertices[i];
            
            vector< R3MeshVertex* > newPair;
            
            // order the points (they are ordered by x first, then y, then z)
            for (int dim = 0; dim < 3; dim++) {
                if ( vertex1->position[dim] < vertex2->position[dim] ) {
                    newPair.push_back(vertex1);
                    newPair.push_back(vertex2);
                    break;
                } else if ( vertex2->position[dim] < vertex1->position[dim] || dim == 2 ) {
                    // if dim == 2, then all the coords are the same, so we might as well put them in any order (this shouldn't ever happen)
                    newPair.push_back(vertex2);
                    newPair.push_back(vertex1);
                    break;
                }
            }
            
            // add edge to edgeList if it's not already in there
            bool alreadyAdded = 0;
            for (unsigned int whichEdge = 0; whichEdge < edgeList.size(); whichEdge++) { 
                alreadyAdded = alreadyAdded || (edgeList[whichEdge][0] == newPair[0] && edgeList[whichEdge][1] == newPair[1]);
            }
            if (!alreadyAdded) {
                edgeList.push_back(newPair);
            }
            
            vertex1 = vertex2;
        }
    }
    
    // for each point
    for (unsigned int whichVertex = 0; whichVertex < vertices.size(); whichVertex++) {
        R3MeshVertex* vertex = vertices[whichVertex];
        
        // iterate through edge list to find neighbors
        vector< R3MeshVertex* > neighbors;
        for (unsigned int whichEdge = 0; whichEdge < edgeList.size(); whichEdge++) {
            if (vertex ==  edgeList[whichEdge][0]) {
                neighbors.push_back(edgeList[whichEdge][1]);
            } else if (vertex ==  edgeList[whichEdge][1]) {
                neighbors.push_back(edgeList[whichEdge][0]);
            }
        }
        
        // compute new value of the point using old vertices
        R3Vector delta_v(0.0,0.0,0.0);
        for (unsigned int whichNeighbor = 0; whichNeighbor < neighbors.size(); whichNeighbor++) { 
            delta_v += (neighbors[whichNeighbor]->position)-(vertex->position);
        }
        delta_v /= neighbors.size();
        
        // save computed value in the new copy of vertices
        newVertices[whichVertex]->position += factor*delta_v;
    }
    
    // copy vertices over
    vertices = newVertices;
    
    Update();
}

void R3Mesh::Taubin(double lambda, double mu, int iters) {
    // Apply Taubin smoothing to the mesh for a given number of iterations. 
    // I suggest to make a single method that smoothes by a positive or negative 
    // amount, and then call Smooth(lambda) and Smooth(mu) in an iterative loop.
    // See Scale() for how to get the vertex positions.

    // FILL IN IMPLEMENTATION 
    for (int i = 0; i < iters; i++){
        Smooth(lambda);
        Smooth(mu);
    }
    
    // Update mesh data structures
    Update();
}

void R3Mesh::Loop() {
    
    // Perform Loop subdivision on a triangular mesh. Faces that are not triangles can be skipped.
    
    // FILL IN IMPLEMENTATION HERE
    
    /* Get edge list of all edges of triangles */
    
    // Each edge corresponds to a new/odd vertex.
    // Edge list will contain vectors of the following format: {p1 p2 p3 p4 p5}
        // p1 and p2 are the points on the edge (i.e the 3/8 weighted points)
        // p3 and p4 are the other 1/8 weighted points
        // p5 is the final new/odd point that we will calculate
    
    vector< vector< R3MeshVertex* > > edgeList; // Vector of R3Point pairs
    // Go through all the triangles
    for (int whichFace = 0; whichFace < NFaces(); whichFace++) {
        R3MeshFace* face = Face(whichFace); // Current Face 
        
        if (face->vertices.size() != 3) { // skip non-triangles.
            continue;
        }
        
        /* Get edges (ordered point pairs) of triangles and put into list */
        R3MeshVertex* vertex1 = face->vertices.back(); // Current vertex on current face
        
        // for all vertices in the current face
        for (int i = 0; i < 3; i++) {
            R3MeshVertex* vertex2 = face->vertices[i];
            
            vector< R3MeshVertex* > newEntry;
            
            // order the points (they are ordered by x first, then y, then z)
            for (int dim = 0; dim < 3; dim++) {
                if ( vertex1->position[dim] < vertex2->position[dim] ) {
                    newEntry.push_back(vertex1);
                    newEntry.push_back(vertex2);
                    break;
                } else if ( vertex2->position[dim] < vertex1->position[dim] || dim == 2 ) {
                    // if dim == 2, then all the coords are the same, so we might as well put them in any order (this shouldn't ever happen)
                    newEntry.push_back(vertex2);
                    newEntry.push_back(vertex1);
                    break;
                }
            }
            
            // add edge to edgeList if it's not already in there
            bool alreadyAdded = 0;
            int alreadyAddedEdgeIndex = -1;
            for (unsigned int whichEdge = 0; whichEdge < edgeList.size(); whichEdge++) { 
                if (edgeList[whichEdge][0] == newEntry[0] && edgeList[whichEdge][1] == newEntry[1]) {
                    alreadyAdded = 1;
                    alreadyAddedEdgeIndex = whichEdge;
                    break;
                }
            }
            if (alreadyAdded) {
                edgeList[alreadyAddedEdgeIndex].push_back(face->vertices[(i+1)%3]); // Add the other edge to the triangle as p4
            } else {
                newEntry.push_back(face->vertices[(i+1)%3]); // Add the other edge to the triangle as p3
                edgeList.push_back(newEntry);
            }
            
            vertex1 = vertex2;
        }
    }
    
    // Currently, each entry in the edge list has 4 points, 2 for the edge or 3/8 points and 2 for the 1/8 points
    // We will add a 5th element to correspond to the new/odd points
    
    // Iterate through the edges to calculate the new/odd vertex
    for (unsigned int whichEdge = 0; whichEdge < edgeList.size(); whichEdge++){
        R3Point position( (3.0/8.0)*(edgeList[whichEdge][0]->position[0] + edgeList[whichEdge][1]->position[0])+(1.0/8.0)*(edgeList[whichEdge][2]->position[0] + edgeList[whichEdge][3]->position[0]),
                          (3.0/8.0)*(edgeList[whichEdge][0]->position[1] + edgeList[whichEdge][1]->position[1])+(1.0/8.0)*(edgeList[whichEdge][2]->position[1] + edgeList[whichEdge][3]->position[1]),
                          (3.0/8.0)*(edgeList[whichEdge][0]->position[2] + edgeList[whichEdge][1]->position[2])+(1.0/8.0)*(edgeList[whichEdge][2]->position[2] + edgeList[whichEdge][3]->position[2])
                          );
        
        R3Vector normal( (3.0/8.0)*(edgeList[whichEdge][0]->normal[X_COORD] + edgeList[whichEdge][1]->normal[X_COORD])+(1.0/8.0)*(edgeList[whichEdge][2]->normal[X_COORD] + edgeList[whichEdge][3]->normal[X_COORD]),
                         (3.0/8.0)*(edgeList[whichEdge][0]->normal[Y_COORD] + edgeList[whichEdge][1]->normal[Y_COORD])+(1.0/8.0)*(edgeList[whichEdge][2]->normal[Y_COORD] + edgeList[whichEdge][3]->normal[Y_COORD]),
                         (3.0/8.0)*(edgeList[whichEdge][0]->normal[Z_COORD] + edgeList[whichEdge][1]->normal[Z_COORD])+(1.0/8.0)*(edgeList[whichEdge][2]->normal[Z_COORD] + edgeList[whichEdge][3]->normal[Z_COORD])
                         );
        R2Point texcoords( (3.0/8.0)*(edgeList[whichEdge][0]->texcoords[X_COORD] + edgeList[whichEdge][1]->texcoords[X_COORD])+(1.0/8.0)*(edgeList[whichEdge][2]->texcoords[X_COORD] + edgeList[whichEdge][3]->texcoords[X_COORD]),
                           (3.0/8.0)*(edgeList[whichEdge][0]->texcoords[Y_COORD] + edgeList[whichEdge][1]->texcoords[Y_COORD])+(1.0/8.0)*(edgeList[whichEdge][2]->texcoords[Y_COORD] + edgeList[whichEdge][3]->texcoords[Y_COORD])
                           );
        
        edgeList[whichEdge].push_back(new R3MeshVertex(position, normal, texcoords));
        cout << edgeList[whichEdge].back()->position[0] << edgeList[whichEdge].back()->position[1] << edgeList[whichEdge].back()->position[2] << endl;
    }
    
    //*/
    // Readjust the old/even vertices
    for (unsigned int whichVertex = 0; whichVertex < vertices.size(); whichVertex++) {
        vector< R3MeshVertex* > newOddNeighbors;
        for (unsigned int whichEdge = 0; whichEdge < edgeList.size(); whichEdge++){
            if (edgeList[whichEdge][0] == vertices[whichVertex] || edgeList[whichEdge][1] == vertices[whichVertex]) {
                newOddNeighbors.push_back(edgeList[whichEdge][4]);
            }
        }
        R3Point newValue(0.0,0.0,0.0);
        for (unsigned int whichNewOddNeighbor = 0; whichNewOddNeighbor < newOddNeighbors.size(); whichNewOddNeighbor++){
            newValue += newOddNeighbors[whichNewOddNeighbor]->position;
        }
        static const double k = 1.0;
        double beta = (newOddNeighbors.size() == 3) ? 3.0/1.60 : 3/(8.0*newOddNeighbors.size());
        newValue *= beta;
        newValue[0] += (vertices[whichVertex]->position[0])*(1.0-k*beta);
        newValue[1] += (vertices[whichVertex]->position[1])*(1.0-k*beta);
        newValue[2] += (vertices[whichVertex]->position[2])*(1.0-k*beta);
        vertices[whichVertex]->position[0] = newValue[0];
        vertices[whichVertex]->position[1] = newValue[1];
        vertices[whichVertex]->position[2] = newValue[2];
    }
    //*/
    
    // Now we need to get rid of the old faces and add new faces
    
    vector< vector< R3MeshVertex* > > facesToBeCreated;
    vector< R3MeshFace* > facesToBeDeleted;
    // for each face
    for (int whichFace = 0; whichFace < NFaces(); whichFace++) {
        R3MeshFace* face = Face(whichFace); // Current Face 
        
        if (face->vertices.size() != 3) { // skip non-triangles.
            continue;
        }
        
        // create three new triangle faces
        vector< vector< R3MeshVertex* > > faceEdges;
        
        // Find edges in edge list that correspond to the face
        for (unsigned int whichEdge = 0; whichEdge < edgeList.size(); whichEdge++) {
            vector< R3MeshVertex* > newEntry;
            if (     (face->vertices[0] == edgeList[whichEdge][0] && face->vertices[1] == edgeList[whichEdge][1]) 
                  || (face->vertices[1] == edgeList[whichEdge][0] && face->vertices[0] == edgeList[whichEdge][1]) 
                  || (face->vertices[1] == edgeList[whichEdge][0] && face->vertices[2] == edgeList[whichEdge][1]) 
                  || (face->vertices[2] == edgeList[whichEdge][0] && face->vertices[1] == edgeList[whichEdge][1]) 
                  || (face->vertices[0] == edgeList[whichEdge][0] && face->vertices[2] == edgeList[whichEdge][1]) 
                  || (face->vertices[2] == edgeList[whichEdge][0] && face->vertices[0] == edgeList[whichEdge][1])  ) {
                newEntry.push_back(face->vertices[0]);
                newEntry.push_back(face->vertices[1]);
                newEntry.push_back(edgeList[whichEdge][4]);
            } 
            if (newEntry.size() != 0) {
                faceEdges.push_back(newEntry);
            }
        }
        
        // create 4 new faces from edges and add to list
        
        for (int i = 0; i < 3; i++) { // make faces for the three outer edges
            vector< R3MeshVertex* > facePoints;
            R3MeshVertex* commonPoint; // the point that the two edges share
            if        (faceEdges[i][0] == faceEdges[(i+1)%3][0] || faceEdges[i][0] == faceEdges[(i+1)%3][1]){
                commonPoint = faceEdges[i][0];
            } else if (faceEdges[i][1] == faceEdges[(i+1)%3][1] || faceEdges[i][1] == faceEdges[(i+1)%3][0]){
                commonPoint = faceEdges[i][1];
            }
            //cout << 3 << endl;
            facePoints.push_back(commonPoint); // common point
            facePoints.push_back(faceEdges[i][2]); // new odd point
            facePoints.push_back(faceEdges[(i+1)%3][2]); // new odd point
            
            facesToBeCreated.push_back(facePoints); //CreateFace(facePoints);
        }
        
        // make face for the inner triangle made of only new/odd points
        vector< R3MeshVertex* > facePoints;
        for (int i = 0; i < 3; i++) {
            facePoints.push_back(faceEdges[i][2]);
        }
        facesToBeCreated.push_back(facePoints);
        //CreateFace(facePoints);
        
        // delete old face
        facesToBeDeleted.push_back(face); //DeleteFace(face); //delete Face(whichFace);
    }
    
    for (unsigned int i = 0; i < facesToBeDeleted.size(); i++) {
        DeleteFace(facesToBeDeleted[i]);
    }
    //*
    for (unsigned int i = 0; i < facesToBeCreated.size(); i++) {
        // Create Vertices
        vector< R3MeshVertex* > faceVector;
        for (unsigned int j = 0; j < facesToBeCreated[i].size(); j++) {
            R3MeshVertex* v = facesToBeCreated[i][j];
            CreateVertex(v->position, v->normal, v->texcoords);
            faceVector.push_back(vertices.back());
        }
        
        // Create Face
        CreateFace(faceVector);
    }
    // Update mesh data structures
    Update();
}


////////////////////////////////////////////////////////////
// MESH CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////

R3Mesh::
R3Mesh(void)
  : bbox(R3null_box)
{
}



R3Mesh::
R3Mesh(const R3Mesh& mesh)
  : bbox(R3null_box)
{
  // Create vertices
  for (int i = 0; i < mesh.NVertices(); i++) {
    R3MeshVertex *v = mesh.Vertex(i);
    CreateVertex(v->position, v->normal, v->texcoords);
  }

  // Create faces
  for (int i = 0; i < mesh.NFaces(); i++) {
    R3MeshFace *f = mesh.Face(i);
    vector<R3MeshVertex *> face_vertices;
    for (unsigned int j = 0; j < f->vertices.size(); j++) {
      R3MeshVertex *ov = f->vertices[j];
      R3MeshVertex *nv = Vertex(ov->id);
      face_vertices.push_back(nv);
    }
    CreateFace(face_vertices);
  }
}



R3Mesh::
~R3Mesh(void)
{
  // Delete faces
  for (int i = 0; i < NFaces(); i++) {
    R3MeshFace *f = Face(i);
    delete f;
  }

  // Delete vertices
  for (int i = 0; i < NVertices(); i++) {
    R3MeshVertex *v = Vertex(i);
    delete v;
  }
}



////////////////////////////////////////////////////////////
// MESH PROPERTY FUNCTIONS
////////////////////////////////////////////////////////////

R3Point R3Mesh::
Center(void) const
{
  // Return center of bounding box
  return bbox.Centroid();
}



double R3Mesh::
Radius(void) const
{
  // Return radius of bounding box
  return bbox.DiagonalRadius();
}



////////////////////////////////////////////////////////////
// MESH PROCESSING FUNCTIONS
////////////////////////////////////////////////////////////

void R3Mesh::
Translate(double dx, double dy, double dz)
{
  // Translate the mesh by adding a 
  // vector (dx,dy,dz) to every vertex

  // This is implemented for you as an example 

  // Create a translation vector
  R3Vector translation(dx, dy, dz);

  // Update vertices
  for (unsigned int i = 0; i < vertices.size(); i++) {
    R3MeshVertex *vertex = vertices[i];
    vertex->position.Translate(translation);
  }

  // Update mesh data structures
  Update();
}




void R3Mesh::
Scale(double sx, double sy, double sz)
{
  // Scale the mesh by increasing the distance 
  // from every vertex to the origin by a factor 
  // given for each dimension (sx, sy, sz)

  // This is implemented for you as an example 

  // Update vertices
  for (unsigned int i = 0; i < vertices.size(); i++) {
    R3MeshVertex *vertex = vertices[i];
    vertex->position[0] *= sx;
    vertex->position[1] *= sy;
    vertex->position[2] *= sz;
  }

  // Update mesh data structures
  Update();
}




void R3Mesh::
Rotate(double angle, const R3Line& axis)
{
  // Rotate the mesh counter-clockwise by an angle 
  // (in radians) around a line axis

  // This is implemented for you as an example 

  // Update vertices
  for (unsigned int i = 0; i < vertices.size(); i++) {
    R3MeshVertex *vertex = vertices[i];
    vertex->position.Rotate(axis, angle);
  }

  // Update mesh data structures
  Update();
}


////////////////////////////////////////////////////////////
// MESH ELEMENT CREATION/DELETION FUNCTIONS
////////////////////////////////////////////////////////////

R3MeshVertex *R3Mesh::
CreateVertex(const R3Point& position, const R3Vector& normal, const R2Point& texcoords)
{
  // Create vertex
  R3MeshVertex *vertex = new R3MeshVertex(position, normal, texcoords);

  // Update bounding box
  bbox.Union(position);

  // Set vertex ID
  vertex->id = vertices.size();

  // Add to list
  vertices.push_back(vertex);

  // Return vertex
  return vertex;
}



R3MeshFace *R3Mesh::
CreateFace(const vector<R3MeshVertex *>& vertices)
{
    //cout << 41 << endl;
  // Create face
  R3MeshFace *face = new R3MeshFace(vertices);
    //cout << 42 << endl;

  // Set face  ID
  face->id = faces.size();
    //cout << 43 << endl;

  // Add to list
  faces.push_back(face);
    //cout << 44 << endl;

  // Return face
  return face;
}



void R3Mesh::
DeleteVertex(R3MeshVertex *vertex)
{
  // Remove vertex from list
  for (unsigned int i = 0; i < vertices.size(); i++) {
    if (vertices[i] == vertex) {
      vertices[i] = vertices.back();
      vertices[i]->id = i;
      vertices.pop_back();
      break;
    }
  }

  // Delete vertex
  delete vertex;
}



void R3Mesh::
DeleteFace(R3MeshFace *face)
{
  // Remove face from list
  for (unsigned int i = 0; i < faces.size(); i++) {
    if (faces[i] == face) {
      faces[i] = faces.back();
      faces[i]->id = i;
      faces.pop_back();
      break;
    }
  }

  // Delete face
  delete face;
}



////////////////////////////////////////////////////////////
// UPDATE FUNCTIONS
////////////////////////////////////////////////////////////

void R3Mesh::
Update(void)
{
  // Update everything
  UpdateBBox();
  UpdateFacePlanes();
  UpdateVertexNormals();
  UpdateVertexCurvatures();
}



void R3Mesh::
UpdateBBox(void)
{
  // Update bounding box
  bbox = R3null_box;
  for (unsigned int i = 0; i < vertices.size(); i++) {
    R3MeshVertex *vertex = vertices[i];
    bbox.Union(vertex->position);
  }
}



void R3Mesh::
UpdateVertexNormals(void)
{
  // Update normal for every vertex
  for (unsigned int i = 0; i < vertices.size(); i++) {
    vertices[i]->UpdateNormal();
  }
}




void R3Mesh::
UpdateVertexCurvatures(void)
{
  // Update curvature for every vertex
  for (unsigned int i = 0; i < vertices.size(); i++) {
    vertices[i]->UpdateCurvature();
  }
}




void R3Mesh::
UpdateFacePlanes(void)
{
  // Update plane for all faces
  for (unsigned int i = 0; i < faces.size(); i++) {
    faces[i]->UpdatePlane();
  }
}



////////////////////////////////////////////////////////////////////////
// I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3Mesh::
Read(const char *filename)
{
  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .ply)\n", filename);
    return 0;
  }

  // Read file of appropriate type
  int status = 0;
  if (!strncmp(extension, ".ray", 4)) 
    status = ReadRay(filename);
  else if (!strncmp(extension, ".off", 4)) 
    status = ReadOff(filename);
  else if (!strncmp(extension, ".jpg", 4)) 
    status = ReadImage(filename);
  else if (!strncmp(extension, ".jpeg", 4)) 
    status = ReadImage(filename);
  else if (!strncmp(extension, ".bmp", 4)) 
    status = ReadImage(filename);
  else if (!strncmp(extension, ".ppm", 4)) 
    status = ReadImage(filename);
  else {
    fprintf(stderr, "Unable to read file %s (unrecognized extension: %s)\n", filename, extension);
    return status;
  }

  // Update mesh data structures
  Update();

  // Return success
  return 1;
}



int R3Mesh::
Write(const char *filename)
{
  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .ply)", filename);
    return 0;
  }

  // Write file of appropriate type
  if (!strncmp(extension, ".ray", 4)) 
    return WriteRay(filename);
  else if (!strncmp(extension, ".off", 4)) 
    return WriteOff(filename);
  else {
    fprintf(stderr, "Unable to write file %s (unrecognized extension: %s)", filename, extension);
    return 0;
  }
}



////////////////////////////////////////////////////////////
// IMAGE FILE INPUT/OUTPUT
////////////////////////////////////////////////////////////

int R3Mesh::
ReadImage(const char *filename)
{
  // Create a mesh by reading an image file, 
  // constructing vertices at (x,y,luminance), 
  // and connecting adjacent pixels into faces. 
  // That is, the image is interpretted as a height field, 
  // where the luminance of each pixel provides its z-coordinate.

  // Read image
  R2Image *image = new R2Image();
  if (!image->Read(filename)) return 0;

  // Create vertices and store in arrays
  R3MeshVertex ***vertices = new R3MeshVertex **[image->Width() ];
  for (int i = 0; i < image->Width(); i++) {
    vertices[i] = new R3MeshVertex *[image->Height() ];
    for (int j = 0; j < image->Height(); j++) {
      double luminance = image->Pixel(i, j).Luminance();
      double z = luminance * image->Width();
      R3Point position((double) i, (double) j, z);
      R2Point texcoords((double) i, (double) j);
      vertices[i][j] = CreateVertex(position, R3zero_vector, texcoords);
    }
  }

  // Create faces
  vector<R3MeshVertex *> face_vertices;
  for (int i = 1; i < image->Width(); i++) {
    for (int j = 1; j < image->Height(); j++) {
      face_vertices.clear();
      face_vertices.push_back(vertices[i-1][j-1]);
      face_vertices.push_back(vertices[i][j-1]);
      face_vertices.push_back(vertices[i][j]);
      face_vertices.push_back(vertices[i-1][j]);
      CreateFace(face_vertices);
    }
  }

  // Delete vertex arrays
  for (int i = 0; i < image->Width(); i++) delete [] vertices[i];
  delete [] vertices;

  // Delete image
  delete image;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////
// OFF FILE INPUT/OUTPUT
////////////////////////////////////////////////////////////

int R3Mesh::
ReadOff(const char *filename)
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "r"))) {
    fprintf(stderr, "Unable to open file %s\n", filename);
    return 0;
  }

  // Read file
  int nverts = 0;
  int nfaces = 0;
  int nedges = 0;
  int line_count = 0;
  int vertex_count = 0;
  int face_count = 0;
  char buffer[1024];
  char header[64];
  while (fgets(buffer, 1023, fp)) {
    // Increment line counter
    line_count++;

    // Skip white space
    char *bufferp = buffer;
    while (isspace(*bufferp)) bufferp++;

    // Skip blank lines and comments
    if (*bufferp == '#') continue;
    if (*bufferp == '\0') continue;

    // Check section
    if (nverts == 0) {
      // Read header keyword
      if (strstr(bufferp, "OFF")) {
        // Check if counts are on first line
        int tmp;
        if (sscanf(bufferp, "%s%d%d%d", header, &tmp, &nfaces, &nedges) == 4) {
          nverts = tmp;
        }
      }
      else {
        // Read counts from second line
        if ((sscanf(bufferp, "%d%d%d", &nverts, &nfaces, &nedges) != 3) || (nverts == 0)) {
          fprintf(stderr, "Syntax error reading header on line %d in file %s\n", line_count, filename);
          fclose(fp);
          return 0;
        }
      }
    }
    else if (vertex_count < nverts) {
      // Read vertex coordinates
      double x, y, z;
      if (sscanf(bufferp, "%lf%lf%lf", &x, &y, &z) != 3) {
        fprintf(stderr, "Syntax error with vertex coordinates on line %d in file %s\n", line_count, filename);
        fclose(fp);
        return 0;
      }

      // Create vertex
      CreateVertex(R3Point(x, y, z), R3zero_vector, R2zero_point);

      // Increment counter
      vertex_count++;
    }
    else if (face_count < nfaces) {
      // Read number of vertices in face 
      int face_nverts = 0;
      bufferp = strtok(bufferp, " \t");
      if (bufferp) face_nverts = atoi(bufferp);
      else {
        fprintf(stderr, "Syntax error with face on line %d in file %s\n", line_count, filename);
        fclose(fp);
        return 0;
      }

      // Read vertex indices for face
      vector<R3MeshVertex *> face_vertices;
      for (int i = 0; i < face_nverts; i++) {
        R3MeshVertex *v = NULL;
        bufferp = strtok(NULL, " \t");
        if (bufferp) v = Vertex(atoi(bufferp));
        else {
          fprintf(stderr, "Syntax error with face on line %d in file %s\n", line_count, filename);
          fclose(fp);
          return 0;
        }

        // Add vertex to vector
        face_vertices.push_back(v);
      }

      // Create face
      CreateFace(face_vertices);

      // Increment counter
      face_count++;
    }
    else {
      // Should never get here
      fprintf(stderr, "Found extra text starting at line %d in file %s\n", line_count, filename);
      break;
    }
  }

  // Check whether read all vertices
  if ((vertex_count != nverts) || (NVertices() < nverts)) {
    fprintf(stderr, "Expected %d vertices, but read %d vertex lines and created %d vertices in file %s\n", 
      nverts, vertex_count, NVertices(), filename);
  }

  // Check whether read all faces
  if ((face_count != nfaces) || (NFaces() < nfaces)) {
    fprintf(stderr, "Expected %d faces, but read %d face lines and created %d faces in file %s\n", 
      nfaces, face_count, NFaces(), filename);
  }

  // Close file
  fclose(fp);

  // Return number of faces read
  return NFaces();
}



int R3Mesh::
WriteOff(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open file %s\n", filename);
    return 0;
  }

  // Write header
  fprintf(fp, "OFF\n");
  fprintf(fp, "%d %d %d\n", NVertices(), NFaces(), 0);

  // Write vertices
  for (int i = 0; i < NVertices(); i++) {
    R3MeshVertex *vertex = Vertex(i);
    const R3Point& p = vertex->position;
    fprintf(fp, "%g %g %g\n", p.X(), p.Y(), p.Z());
    vertex->id = i;
  }

  // Write Faces
  for (int i = 0; i < NFaces(); i++) {
    R3MeshFace *face = Face(i);
    fprintf(fp, "%d", (int) face->vertices.size());
    for (unsigned int j = 0; j < face->vertices.size(); j++) {
      fprintf(fp, " %d", face->vertices[j]->id);
    }
    fprintf(fp, "\n");
  }

  // Close file
  fclose(fp);

  // Return number of faces
  return NFaces();
}



////////////////////////////////////////////////////////////
// RAY FILE INPUT/OUTPUT
////////////////////////////////////////////////////////////

int R3Mesh::
ReadRay(const char *filename)
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "r"))) {
    fprintf(stderr, "Unable to open file %s", filename);
    return 0;
  }

  // Read body
  char cmd[128];
  int polygon_count = 0;
  int command_number = 1;
  while (fscanf(fp, "%s", cmd) == 1) {
    if (!strcmp(cmd, "#vertex")) {
      // Read data
      double px, py, pz;
      double nx, ny, nz;
      double ts, tt;
      if (fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf", &px, &py, &pz, &nx, &ny, &nz, &ts, &tt) != 8) {
        fprintf(stderr, "Unable to read vertex at command %d in file %s", command_number, filename);
        return 0;
      }

      // Create vertex
      R3Point point(px, py, pz);
      R3Vector normal(nx, ny, nz);
      R2Point texcoords(ts, tt);
      CreateVertex(point, normal, texcoords);
    }
    else if (!strcmp(cmd, "#shape_polygon")) {
      // Read data
      int m, nverts;
      if (fscanf(fp, "%d%d", &m, &nverts) != 2) {
        fprintf(stderr, "Unable to read polygon at command %d in file %s", command_number, filename);
        return 0;
      }

      // Get vertices
      vector<R3MeshVertex *> face_vertices;
      for (int i = 0; i < nverts; i++) {
        // Read vertex id
        int vertex_id;
        if (fscanf(fp, "%d", &vertex_id) != 1) {
          fprintf(stderr, "Unable to read polygon at command %d in file %s", command_number, filename);
          return 0;
        }

        // Get vertex
        R3MeshVertex *v = Vertex(vertex_id);
        face_vertices.push_back(v);
      }

      // Create face
      CreateFace(face_vertices);

      // Increment polygon counter
      polygon_count++;
    }
	
    // Increment command number
    command_number++;
  }

  // Close file
  fclose(fp);

  // Return number of faces created
  return polygon_count;
}



int R3Mesh::
WriteRay(const char *filename)
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "w"))) {
    fprintf(stderr, "Unable to open file %s", filename);
    return 0;
  }

  // Write vertices
  for (int i = 0; i < NVertices(); i++) {
    R3MeshVertex *vertex = Vertex(i);
    const R3Point& p = vertex->position;
    const R3Vector& n = vertex->normal;
    const R2Point& t = vertex->texcoords;
    fprintf(fp, "#vertex %g %g %g  %g %g %g  %g %g\n", p.X(), p.Y(), p.Z(), 
      n.X(), n.Y(), n.Z(), t.X(), t.Y());
    vertex->id = i;
  }

  // Write faces
  for (int i = 0; i < NFaces(); i++) {
    R3MeshFace *face = Face(i);
    int nvertices = face->vertices.size();
    fprintf(fp, "#shape_polygon 0 %d ", nvertices);
    for (int j = 0; j < nvertices; j++) {
      R3MeshVertex *v = face->vertices[j];
      fprintf(fp, "%d ", v->id);
    }
    fprintf(fp, "\n");
  }

  // Close file
  fclose(fp);

  // Return number of faces written
  return NFaces();
}



////////////////////////////////////////////////////////////
// MESH VERTEX MEMBER FUNCTIONS
////////////////////////////////////////////////////////////

R3MeshVertex::
R3MeshVertex(void)
  : position(0, 0, 0),
    normal(0, 0, 0),
    texcoords(0, 0),
    curvature(0),
    id(0)
{
}



R3MeshVertex::
R3MeshVertex(const R3MeshVertex& vertex)
  : position(vertex.position),
    normal(vertex.normal),
    texcoords(vertex.texcoords),
    curvature(vertex.curvature),
    id(0)
{
}




R3MeshVertex::
R3MeshVertex(const R3Point& position, const R3Vector& normal, const R2Point& texcoords)
  : position(position),                    
    normal(normal),
    texcoords(texcoords),
    curvature(0),
    id(0)
{
}




double R3MeshVertex::
AverageEdgeLength(void) const
{
  // Return the average length of edges attached to this vertex
  // This feature should be implemented first.  To do it, you must
  // design a data structure that allows O(K) access to edges attached
  // to each vertex, where K is the number of edges attached to the vertex.

  // FILL IN IMPLEMENTATION HERE  (THIS IS REQUIRED)
  // BY REPLACING THIS ARBITRARY RETURN VALUE
  fprintf(stderr, "Average vertex edge length not implemented\n");
  return 0.12345;
}




void R3MeshVertex::
UpdateNormal(void)
{
  // Compute the surface normal at a vertex.  This feature should be implemented
  // second.  To do it, you must design a data structure that allows O(K)
  // access to faces attached to each vertex, where K is the number of faces attached
  // to the vertex.  Then, to compute the normal for a vertex,
  // you should take a weighted average of the normals for the attached faces, 
  // where the weights are determined by the areas of the faces.
  // Store the resulting normal in the "normal"  variable associated with the vertex. 
  // You can display the computed normals by hitting the 'N' key in meshview.

  // FILL IN IMPLEMENTATION HERE (THIS IS REQUIRED)
  // fprintf(stderr, "Update vertex normal not implemented\n");
}




void R3MeshVertex::
UpdateCurvature(void)
{
  // Compute an estimate of the Gauss curvature of the surface 
  // using a method based on the Gauss Bonet Theorem, which is described in 
  // [Akleman, 2006]. Store the result in the "curvature"  variable. 

  // FILL IN IMPLEMENTATION HERE
  // fprintf(stderr, "Update vertex curvature not implemented\n");
}





////////////////////////////////////////////////////////////
// MESH FACE MEMBER FUNCTIONS
////////////////////////////////////////////////////////////

R3MeshFace::
R3MeshFace(void)
  : vertices(),
    plane(0, 0, 0, 0),
    id(0)
{
}



R3MeshFace::
R3MeshFace(const R3MeshFace& face)
  : vertices(face.vertices),
    plane(face.plane),
    id(0)
{
}



R3MeshFace::
R3MeshFace(const vector<R3MeshVertex *>& vertices)
  : vertices(vertices),
    plane(0, 0, 0, 0),
    id(0)
{
    //cout << 51 << endl;
  UpdatePlane();
    //cout << 51 << endl;
}



double R3MeshFace::
AverageEdgeLength(void) const
{
  // Check number of vertices
  if (vertices.size() < 2) return 0;

  // Compute average edge length
  double sum = 0;
  R3Point *p1 = &(vertices.back()->position);
  for (unsigned int i = 0; i < vertices.size(); i++) {
    R3Point *p2 = &(vertices[i]->position);
    double edge_length = R3Distance(*p1, *p2);
    sum += edge_length;
    p1 = p2;
  }

  // Return the average length of edges attached to this face
  return sum / vertices.size();
}



double R3MeshFace::
Area(void) const
{
  // Check number of vertices
  if (vertices.size() < 3) return 0;

  // Compute area using Newell's method (assumes convex polygon)
  R3Vector sum = R3null_vector;
  const R3Point *p1 = &(vertices.back()->position);
  for (unsigned int i = 0; i < vertices.size(); i++) {
    const R3Point *p2 = &(vertices[i]->position);
    sum += p2->Vector() % p1->Vector();
    p1 = p2;
  }

  // Return area
  return 0.5 * sum.Length();
}



void R3MeshFace::
UpdatePlane(void)
{
  // Check number of vertices
  //cout << 61 << endl;
  int nvertices = vertices.size();
  if (nvertices < 3) { 
    plane = R3null_plane; 
    return; 
  }
  //cout << 62 << endl;
  // Compute centroid
  R3Point centroid = R3zero_point;
  for (int i = 0; i < nvertices; i++) {
    //cout << "i: " << i << " " << 621 << " nvertices: " << nvertices << endl;
    //cout << "vertices.size(): " << vertices.size() << endl;
    /*
    //cout << "vertices[i]->position[0]: " << vertices[i]->position[0] << endl;
    //cout << "vertices[i]->position[1]: " << vertices[i]->position[1] << endl;
    //cout << "vertices[i]->position[2]: " << vertices[i]->position[2] << endl;
    */
    centroid += vertices[i]->position;
    //cout << 622 << endl;
  }
  centroid /= nvertices;
  //cout << 63 << endl;
  // Compute best normal for counter-clockwise array of vertices using newell's method
  R3Vector normal = R3zero_vector;
  const R3Point *p1 = &(vertices[nvertices-1]->position);
  //cout << 64 << endl;
  for (int i = 0; i < nvertices; i++) {
    const R3Point *p2 = &(vertices[i]->position);
    normal[0] += (p1->Y() - p2->Y()) * (p1->Z() + p2->Z());
    normal[1] += (p1->Z() - p2->Z()) * (p1->X() + p2->X());
    normal[2] += (p1->X() - p2->X()) * (p1->Y() + p2->Y());
    p1 = p2;
  }
  //cout << 65 << endl;
  // Normalize normal vector
  normal.Normalize();
  
  // Update face plane
  plane.Reset(centroid, normal);
}



