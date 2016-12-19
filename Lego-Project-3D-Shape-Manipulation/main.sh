# obj meshes at http://people.sc.fsu.edu/~jburkardt/data/obj/obj.html 
# SDFGen code from https://github.com/christopherbatty/SDFGen
# Tool for converting from .off to .obj http://www.greentoken.de/onlineconv/ 

# mesh needs to be just a mesh of triangles since SDFGen will segfault on a non-triangle face
# can convert to triangles using blender (http://www.blender.org/)
# follow these instructions (http://stackoverflow.com/questions/5257272/how-to-get-wavefront-obj-file-with-3-faces-traingle)
# SDFGen is kind of buggy and won't work with all meshes

clear && clear

MESH_NAME=gourd
DX=1
PADDING=1
SCALE_FACTOR=6
NUM_CHROMOSOMES=10
NUM_GENERATIONS=10

echo -e "\nScaling "${MESH_NAME}".obj by "${SCALE_FACTOR}"\n"

(cd ./voxelizer && python ./scale.py ../obj-meshes/${MESH_NAME}.obj ${SCALE_FACTOR})

echo -e "\nStarting SDFGen on scaled.obj\n"

(cd ./SDFGen && make) # Compile SDFGen

# Run SDFGen on mesh
./SDFGen/bin/SDFGen ./obj-meshes/scaled.obj ${DX} ${PADDING} #MESH_NAME.sdf should now be in ./obj-meshes/

echo -e "\nStarting voxelizer.py on scaled.sdf\n"

#run voxelizer.py, should print out voxel_positions.voxel file in ./voxelizer that has the xyz positions on each line to represent
# places we need to put blocks
(cd ./voxelizer && python voxelizer.py ../obj-meshes/scaled.sdf ./ ${NUM_CHROMOSOMES} ${NUM_GENERATIONS}) 

echo -e "\nCompiling 3D Shape Manipulator\n"

#(cd 3D && make clean)
(cd 3D && make)

echo -e "Placing Voxels Down"

(cd 3D && ./src/meshpro ./meshes/X.off ./voxels.off -voxel ../voxelizer/voxel_positions.voxel)

echo -e "Displaying Voxels"

#(cd 3D && ./src/meshview ./voxels.off)

echo -e "Placing Singles Down"

(cd 3D && ./src/meshpro ./meshes/X.off ./singles.off -singles ../voxelizer/lego_positions.singles)

echo -e "Displaying Singles"

#(cd 3D && ./src/meshview ./singles.off)

echo -e "Placing Doubles Down"

(cd 3D && ./src/meshpro ./meshes/X.off ./doubles.off -doubles ../voxelizer/lego_positions.doubles)

echo -e "Displaying Doubles"

#(cd 3D && ./src/meshview ./doubles.off)

echo -e "Placing Everything Down"

(cd 3D && ./src/meshpro ./meshes/X.off ./lego.off -lego ../voxelizer/lego_positions.singles ../voxelizer/lego_positions.doubles)

echo -e "Displaying Everything"

#(cd 3D && ./src/meshview ./lego.off)

