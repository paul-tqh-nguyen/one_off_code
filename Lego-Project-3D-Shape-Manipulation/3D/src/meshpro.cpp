// Modified mesh processing starter code, originally by Adam Finkelstein.
// CS 6501 -- 2D/3D Shape Manipulation, 3D Printing



// Include files
#ifdef _WIN32
#include <windows.h>
#endif

#include "R2/R2.h"
#include "R3/R3.h"
#include "R3Mesh.h"

// Program arguments

static char options[] =
"  -help\n"
"  -twist angle\n"
"  -taubin lambda mu iters\n"
"  -loop\n";

static void 
ShowUsage(void)
{
  // Print usage message and exit
  fprintf(stderr, "Usage: meshpro input_mesh output_mesh [  -option [arg ...] ...]\n");
  fprintf(stderr, "%s", options);
  exit(EXIT_FAILURE);
}



static void 
CheckOption(char *option, int argc, int minargc)
{
  // Check if there are enough remaining arguments for option
  if (argc < minargc)  {
    fprintf(stderr, "Too few arguments for %s\n", option);
    ShowUsage();
    exit(-1);
  }
}



int 
main(int argc, char **argv)
{
  // Look for help
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "-help")) {
      ShowUsage();
    }
  }

  // Read input and output mesh filenames
  if (argc < 3)  ShowUsage();
  argv++, argc--; // First argument is program name
  char *input_mesh_name = *argv; argv++, argc--; 
  char *output_mesh_name = *argv; argv++, argc--; 

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  if (!mesh) {
    fprintf(stderr, "Unable to allocate mesh\n");
    exit(-1);
  }

  // Read input mesh
  if (!mesh->Read(input_mesh_name)) {
    fprintf(stderr, "Unable to read mesh from %s\n", input_mesh_name);
    exit(-1);
  }

  // Parse arguments and perform operations 
  while (argc > 0) {
    if (!strcmp(*argv, "-twist")) {
      CheckOption(*argv, argc, 2);
      double angle = atof(argv[1]);
      argv += 2, argc -= 2;
      mesh->Twist(angle);
    }
    else if (!strcmp(*argv, "-taubin")) {
      CheckOption(*argv, argc, 4);
      double lambda = atof(argv[1]);
      double mu = atof(argv[2]);
      double iters = atoi(argv[3]);
      argv += 4, argc -= 4;
      mesh->Taubin(lambda, mu, iters);
    }
    else if (!strcmp(*argv, "-voxel")) {
      CheckOption(*argv, argc, 2);
      string file_name = string(argv[1]);
      argv += 2, argc -= 2;
      mesh->Voxel(file_name);
    }
    else if (!strcmp(*argv, "-singles")) {
      CheckOption(*argv, argc, 2);
      string file_name = string(argv[1]);
      argv += 2, argc -= 2;
      mesh->Singles(file_name);
    }
    else if (!strcmp(*argv, "-doubles")) {
      CheckOption(*argv, argc, 2);
      string file_name = string(argv[1]);
      argv += 2, argc -= 2;
      mesh->Doubles(file_name);
    }
    else if (!strcmp(*argv, "-lego")) {
      CheckOption(*argv, argc, 3);
      string singles_file_name = string(argv[1]); 
      string doubles_file_name = string(argv[2]);
      argv += 3, argc -= 3;
      mesh->Lego(singles_file_name, doubles_file_name);
    }
    else if (!strcmp(*argv, "-loop")) {
      CheckOption(*argv, argc, 1);
      argv += 1, argc -= 1;
      mesh->Loop();
    }
    else {
      // Unrecognized program argument
      fprintf(stderr, "meshpro: invalid option: %s\n", *argv);
      ShowUsage();
    }
  }

  // Write output mesh
  if (!mesh->Write(output_mesh_name)) {
    fprintf(stderr, "Unable to write mesh to %s\n", output_mesh_name);
    exit(-1);
  }

  // Delete mesh
  delete mesh;

  // Return success
  return EXIT_SUCCESS;
}

