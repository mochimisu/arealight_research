
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

//-------------------------------------------------------------------------------
//
//  Soft Shadow example modified from OptiX tutorial
//
//-------------------------------------------------------------------------------



#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <iostream>
#include <GLUTDisplay.h>
#include <ObjLoader.h>
#include <ImageLoader.h>
#include "commonStructs.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <math.h>
#include <time.h>
#include "random.h"

using namespace optix;

static float rand_range(float min, float max)
{
  return min + (max - min) * (float) rand() / (float) RAND_MAX;
}


//-----------------------------------------------------------------------------
//
// Whitted Scene
//
//-----------------------------------------------------------------------------

class Softshadow : public SampleScene
{
  public:
    Softshadow(const std::string& texture_path)
      : SampleScene(), _width(1080u), _height(720u), texture_path( texture_path )
        , _frame_number( 0 ), _keep_trying( 1 )
  {}

    // From SampleScene
    void   initScene( InitialCameraData& camera_data );
    void   trace( const RayGenCameraData& camera_data );
    void   doResize( unsigned int width, unsigned int height );
    void   setDimensions( const unsigned int w, const unsigned int h ) { _width = w; _height = h; }
    Buffer getOutputBuffer();

    virtual bool   keyPressed(unsigned char key, int x, int y);

  private:
    std::string texpath( const std::string& base );
    void resetAccumulation();
    void createGeometry();

    bool _accel_cache_loaded;

    unsigned int  _frame_number;
    unsigned int  _keep_trying;

    Buffer       _brdf;
    Buffer       _occ;
    GeometryGroup geomgroup;

    Buffer _conv_buffer;

    unsigned int _width;
    unsigned int _height;
    std::string   texture_path;
    std::string  _ptx_path;

    float _env_theta;
    float _env_phi;

    uint _blur_occ;
    uint _err_vis;
    uint _view_zmin;

    uint _normal_rpp;
    uint _brute_rpp;
    uint _show_progressive;
    int2 _pixel_radius;

    float _zmin_rpp_scale;
    bool _converged;

    time_t _started_render;

    Buffer testBuf;

    AreaLight * _env_lights;
    uint _show_brdf;
    uint _show_occ;
};


void Softshadow::initScene( InitialCameraData& camera_data )
{
  // set up path to ptx file associated with tutorial number
  std::stringstream ss;
  ss << "ambocc.cu";
  _ptx_path = ptxpath( "ambocc_research", ss.str() );

  // context 
  _context->setRayTypeCount( 2 );
  _context->setEntryPointCount( 1 );
  _context->setStackSize( 8000 );

  _context["max_depth"]->setInt(100);
  _context["radiance_ray_type"]->setUint(0);
  _context["shadow_ray_type"]->setUint(1);
  _context["frame_number"]->setUint( 0u );
  _context["scene_epsilon"]->setFloat( 1.e-3f );
  _context["importance_cutoff"]->setFloat( 0.01f );
  _context["ambient_light_color"]->setFloat( 0.31f, 0.33f, 0.28f );

  _context["output_buffer"]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, _width, _height) );

  //[bmw] my stuff
  //seed rng
  //(i have no idea if this is right)
  //fix size later
  Buffer shadow_rng_seeds = _context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT2, _width, _height);
  _context["shadow_rng_seeds"]->set(shadow_rng_seeds);
  uint2* seeds = reinterpret_cast<uint2*>( shadow_rng_seeds->map() );
  for(unsigned int i = 0; i < _width * _height; ++i )
    seeds[i] = random2u();
  shadow_rng_seeds->unmap();

  _context["numAvg"]->setUint(1);

  // Accumulation buffer
  _brdf = _context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  _context["brdf"]->set( _brdf );

  // Occlusion accumulation buffer
  _occ = _context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4, _width, _height );
  _context["occ"]->set( _occ );

  // gauss values
  Buffer gauss_lookup = _context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, 65);
  _context["gaussian_lookup"]->set( gauss_lookup );

  float* lookups = reinterpret_cast<float*>( gauss_lookup->map() );
  const float gaussian_lookup[65] = { 0.85, 0.82, 0.79, 0.76, 0.72, 0.70, 0.68,
    0.66, 0.63, 0.61, 0.59, 0.56, 0.54, 0.52,
    0.505, 0.485, 0.46, 0.445, 0.43, 0.415, 0.395,
    0.38, 0.365, 0.35, 0.335, 0.32, 0.305, 0.295,
    0.28, 0.27, 0.255, 0.24, 0.23, 0.22, 0.21,
    0.2, 0.19, 0.175, 0.165, 0.16, 0.15, 0.14,
    0.135, 0.125, 0.12, 0.11, 0.1, 0.095, 0.09,
    0.08, 0.075, 0.07, 0.06, 0.055, 0.05, 0.045,
    0.04, 0.035, 0.03, 0.02, 0.018, 0.013, 0.008,
    0.003, 0.0 };

  /*
     const float gaussian_lookup[65] = { 0.86466, 0.86418,
     0.86271, 0.86028, 0.85688, 0.85253, 0.84724, 0.84102, 0.83390,
     0.82589, 0.81701, 0.80729, 0.79677, 0.78546, 0.77340, 0.76062,
     0.74716, 0.73306, 0.71834, 0.70306, 0.68724, 0.67094, 0.65419,
     0.63703, 0.61950, 0.60166, 0.58353, 0.56517, 0.54661, 0.52789,
     0.50905, 0.49014, 0.47120, 0.45225, 0.43334, 0.41450, 0.39576,
     0.37716, 0.35873, 0.34050, 0.32250, 0.30474, 0.28727, 0.27008,
     0.25322, 0.23670, 0.22053, 0.20473, 0.18932, 0.17430, 0.15969,
     0.14549, 0.13172, 0.11837, 0.10546, 0.09297, 0.08093, 0.06932,
     0.05815, 0.04740, 0.03709, 0.02719, 0.01772, 0.00866, 0.00000 };
     */

  for(int i=0; i<65; i++) {
    lookups[i] = gaussian_lookup[i];
  }

  gauss_lookup->unmap();

  // world space buffer
  Buffer world_loc = _context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  _context["world_loc"]->set( world_loc );

  Buffer n = _context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  _context["n"]->set( n );

  Buffer obj_id = _context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_INT, _width, _height );
  _context["obj_id_buf"]->set( obj_id );
  
  Buffer err_buf = _context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_INT, _width, _height );
  _context["err_buf"]->set( err_buf );

  
  Buffer dist_scale = _context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
  _context["dist_scale"]->set( dist_scale );

  _blur_occ = 1;
  _context["blur_occ"]->setUint(_blur_occ);

  _err_vis = 0;
  _context["err_vis"]->setUint(_err_vis);

  _view_zmin = 0;
  _context["view_zmin"]->setUint(_view_zmin);

  _show_progressive = 0;
  _context["show_progressive"]->setUint(_show_progressive);

  _show_brdf = 0;
  _context["show_brdf"]->setUint(_show_brdf);

  _show_occ = 1;
  _context["show_occ"]->setUint(_show_occ);

  _normal_rpp = 6;
  _brute_rpp = 6;

  _context["normal_rpp"]->setUint(_normal_rpp);
  _context["brute_rpp"]->setUint(_brute_rpp);

  _zmin_rpp_scale = 1;
  _context["zmin_rpp_scale"]->setFloat(_zmin_rpp_scale);

  _pixel_radius = make_int2(10,10);
  _context["pixel_radius"]->setInt(_pixel_radius);

  // Ray gen program
  std::string camera_name;
  camera_name = "pinhole_camera";

  Program ray_gen_program = _context->createProgramFromPTXFile( _ptx_path, camera_name );
  _context->setRayGenerationProgram( 0, ray_gen_program );

  // Exception / miss programs
  Program exception_program = _context->createProgramFromPTXFile( _ptx_path, "exception" );
  _context->setExceptionProgram( 0, exception_program );
  _context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );

  //Blur program (i hope)
  //Program blur_program = _context->createProgramFromPTXFile( _ptx_path, "gaussianBlur" );


  std::string miss_name;
  miss_name = "miss";
  _context->setMissProgram( 0, _context->createProgramFromPTXFile( _ptx_path, miss_name ) );
  const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
  _context["bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );

  // Area lights

  AreaLight lights[] = {
    { make_float3(-10.0f, 15.0f, -16.0f),
      make_float3(0.0f, 15.0f, -16.0f),
      make_float3(10.0f, 15.0f, -16.0f),
      make_float3(1.0f, 1.0f, 1.0f)
    }
  };
  _env_lights = lights;
  Buffer light_buffer = _context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(AreaLight));
  //light_buffer->setSize( sizeof(_env_lights)/sizeof(_env_lights[0]) );
  //memcpy(light_buffer->map(), _env_lights, sizeof(_env_lights));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  _context["lights"]->set(light_buffer);

/*
  // Lights for non IBL  
  BoxLight lights[] = {
    { make_float3( -10.0f, 55.0f, -16.0f ),
      make_float3( 0.0f, 65.0f, -16.0f ),
      make_float3( 1.0f, 1.0f, 1.0f ),
      1 
    }
  };
  Buffer light_buffer = _context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(BoxLight));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  _context["lights"]->set(light_buffer);
  */

  // Set up camera
  camera_data = InitialCameraData( make_float3( 7.0f, 9.2f, -6.0f ), // eye
      make_float3( 0.0f, 4.0f,  0.0f ), // lookat
      make_float3( 0.0f, 1.0f,  0.0f ), // up
      60.0f );                          // vfov

  _context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  _context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  _context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  _context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  _env_theta = 0.0f;
  _env_phi = 0.0f;
  _context["env_theta"]->setFloat(_env_theta);
  _context["env_phi"]->setFloat(_env_phi);

  //Load OBJ
  geomgroup = _context->createGeometryGroup();
  ObjLoader* loader = 0;
  std::string obj_file;
  obj_file = "plant.obj";
  //obj_file = "bunny.obj";
  //obj_file = "sphere.obj";

  //just for an ex

  //Material
  Material mat = _context->createMaterial();
  mat->setClosestHitProgram(0, _context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  mat->setAnyHitProgram(1, _context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));

  mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  mat["Kd"]->setFloat( .40f, 0.8f, .23f );
  mat["Ks"]->setFloat( 1.0f, 1.0f, 1.0f );
  mat["phong_exp"]->setFloat( 88 );
  mat["reflectivity"]->setFloat( 0.05f, 0.05f, 0.05f );
  mat["reflectivity_n"]->setFloat( 0.2f, 0.2f, 0.2f );  
  mat["obj_id"]->setInt(10);

  //Matrix4x4 obj_xform = Matrix4x4::identity()* Matrix4x4::translate(make_float3(0,3,0)) * Matrix4x4::rotate(0, make_float3(1,0,0)) * Matrix4x4::scale(make_float3(3.0));
  Matrix4x4 obj_xform = Matrix4x4::identity() * Matrix4x4::scale(make_float3(1.0));


  loader = new ObjLoader( texpath(obj_file).c_str(), _context, geomgroup, mat );
  loader->load(obj_xform);  

  //Use kd tree (default by ObjLoder is SBVH)
  //doesnt work for some reason..?
  /*
     Acceleration accel = geomgroup->getAcceleration();
     accel->setBuilder("TriangleKdTree");
     accel->setTraverser("KdTree");
     accel->setProperty( "vertex_buffer_name", "vertex_buffer" );
     accel->setProperty( "index_buffer_name", "vindex_buffer" );
     */


  // Populate scene hierarchy
  createGeometry();

  _context["top_object"]->set( geomgroup );
  _context["top_shadower"]->set( geomgroup );





  //Initialize progressive accumulation
  resetAccumulation();

  // Prepare to run
  _context->validate();
  _context->compile();
}


Buffer Softshadow::getOutputBuffer()
{

  //return _context["output_buffer"]->getBuffer();
  return _context["dist_scale"]->getBuffer();
}


void Softshadow::trace( const RayGenCameraData& camera_data )
{
  _frame_number ++;

  if(_camera_changed) {
    _context["numAvg"]->setUint(1);
    _camera_changed = false;
    resetAccumulation();
  }
  _context["eye"]->setFloat( camera_data.eye );
  _context["U"]->setFloat( camera_data.U );
  _context["V"]->setFloat( camera_data.V );
  _context["W"]->setFloat( camera_data.W );

  // do i need to reseed?
  Buffer shadow_rng_seeds = _context["shadow_rng_seeds"]->getBuffer();
  uint2* seeds = reinterpret_cast<uint2*>( shadow_rng_seeds->map() );
  for(unsigned int i = 0; i < _width * _height; ++i )
    seeds[i] = random2u();
  shadow_rng_seeds->unmap();

  Buffer buffer = _context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );
  _context["frame"]->setUint( _frame_number );


  _context->launch( 0, static_cast<unsigned int>(buffer_width),
      static_cast<unsigned int>(buffer_height) );

  if (_frame_number == _normal_rpp) {
    time_t end;
    time(&end);
    std::cout << "Blurred rays done: " << (float)difftime(end,_started_render) << "s" << std::endl;
  }


  if (_frame_number == _brute_rpp) {
    time_t end;
    time(&end);
    std::cout << "Brute force Done: " << difftime(end,_started_render) << "s" << std::endl;
  }

  if (_frame_number == _brute_rpp+1) {
    time_t end;
    time(&end);
    std::cout << "Total render done (including blur): " << difftime(end,_started_render) << "s" << std::endl;
  }

  /*
     unsigned int unconverged = 0;
     unsigned int * conv = reinterpret_cast<unsigned int*>( _conv_buffer->map() );
     for (unsigned int i = 0; i < _width * _height; ++i)
     unconverged += conv[i];
     _conv_buffer->unmap();
     std::cout << "Unconverged: " << unconverged << std::endl;
     */
}


void Softshadow::doResize( unsigned int width, unsigned int height )
{
  // output buffer handled in SampleScene::resize
}

std::string Softshadow::texpath( const std::string& base )
{
  return texture_path + "/" + base;
}

float4 make_plane( float3 n, float3 p )
{
  n = normalize(n);
  float d = -dot(n, p);
  return make_float4( n, d );
}

void Softshadow::resetAccumulation()
{
  _frame_number = 0;
  _context["frame"]->setUint( _frame_number );
  _converged = false;
  time(&_started_render);
}


bool Softshadow::keyPressed(unsigned char key, int x, int y) {
  float delta = 0.01f;

  switch(key) {
    case 'U':
    case 'u':
      break;
    case 'J':
    case 'j':
      break;
    case 'I':
    case 'i':
      break;
    case 'K':
    case 'k':
      break;
      
    case 'O':
    case 'o':
      break;
    case 'L':
    case 'l':
      break;

    case 'M':
    case 'm':
      _show_brdf = 1-_show_brdf;
      _context["show_brdf"]->setUint(_show_brdf);
      if (_show_brdf)
        std::cout << "BRDF: On" << std::endl;
      else
        std::cout << "BRDF: Off" << std::endl;
      return true;
    case 'N':
    case 'n':
      _show_occ = 1-_show_occ;
      _context["show_occ"]->setUint(_show_occ);
      if (_show_occ)
        std::cout << "Occlusion Display: On" << std::endl;
      else
        std::cout << "Occlusion Display: Off" << std::endl;
      return true;





    case 'Q':
    case 'q':
      _env_theta += delta;
      _context["env_theta"]->setFloat(_env_theta);
      _camera_changed = true;
      return true;
    case 'W':
    case 'w':
      _env_theta -= delta;
      _context["env_theta"]->setFloat(_env_theta);
      _camera_changed = true;
      return true;
    case 'A':
    case 'a':
      std::cout << _frame_number << " Rays." << std::endl;
      return true;
    case 'B':
    case 'b':
      _blur_occ = 1-_blur_occ;
      _context["blur_occ"]->setUint(_blur_occ);
      if (_blur_occ)
        std::cout << "Blur: On" << std::endl;
      else
        std::cout << "Blur: Off" << std::endl;

      return true;
    case 'E':
    case 'e':
      _err_vis = 1-_err_vis;
      _context["err_vis"]->setUint(_err_vis);
      if (_err_vis)
        std::cout << "Err vis: On" << std::endl;
      else
        std::cout << "Err vis: Off" << std::endl;
      return true;
    case 'Z':
    case 'z':
      _view_zmin = 1-_view_zmin;
      _context["view_zmin"]->setUint(_view_zmin);
      if (_view_zmin)
        std::cout << "View ZMin: On" << std::endl;
      else
        std::cout << "View ZMin: Off" << std::endl;
      return true;
    case 'P':
    case 'p':
      _show_progressive = 1-_show_progressive;
      _context["show_progressive"]->setUint(_show_progressive);
      if (_show_progressive)
        std::cout << "Blur progressive: On" << std::endl;
      else
        std::cout << "Blur progressive: Off" << std::endl;
      return true;
    case '.':
      if(_pixel_radius.x > 1000)
        return true;
      _pixel_radius += make_int2(1,1);
      _context["pixel_radius"]->setInt(_pixel_radius);
      std::cout << "Pixel radius now: " << _pixel_radius.x << "," << _pixel_radius.y << std::endl;
      return true;
    case ',':
      if (_pixel_radius.x < 2)
        return true;
      _pixel_radius -= make_int2(1,1);
      _context["pixel_radius"]->setInt(_pixel_radius);
      std::cout << "Pixel radius now: " << _pixel_radius.x << "," << _pixel_radius.y << std::endl;
      return true;
  }
  return false;
}

void Softshadow::createGeometry()
{
  std::string box_ptx( ptxpath( "ambocc_research", "box.cu" ) ); 
  Program box_bounds = _context->createProgramFromPTXFile( box_ptx, "box_bounds" );
  Program box_intersect = _context->createProgramFromPTXFile( box_ptx, "box_intersect" );

  // Create box
  Geometry box = _context->createGeometry();
  box->setPrimitiveCount( 1u );
  box->setBoundingBoxProgram( box_bounds );
  box->setIntersectionProgram( box_intersect );
  box["boxmin"]->setFloat( -2.0f, 0.0f, -2.0f );
  box["boxmax"]->setFloat(  2.0f, 7.0f,  2.0f );

  // Create chull
  Geometry chull = 0;

  // Sphere
  std::string sph_ptx( ptxpath( "ambocc_research", "sphere.cu" ) ); 
  Program sph_bounds = _context->createProgramFromPTXFile( sph_ptx, "bounds" );
  Program sph_intersect = _context->createProgramFromPTXFile( sph_ptx, "intersect" );

  Geometry sphere = _context->createGeometry();
  sphere->setBoundingBoxProgram( sph_bounds );
  sphere->setIntersectionProgram( sph_intersect );
  sphere->setPrimitiveCount( 1u );
  sphere["sphere"]->setFloat( 0, 5, 0, 4 );

  


  // Floor geometry
  std::string pgram_ptx( ptxpath( "ambocc_research", "parallelogram.cu" ) );
  Geometry parallelogram = _context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setBoundingBoxProgram( _context->createProgramFromPTXFile( pgram_ptx, "bounds" ) );
  parallelogram->setIntersectionProgram( _context->createProgramFromPTXFile( pgram_ptx, "intersect" ) );
  float3 anchor = make_float3( -16.0f, 0.01f, -16.0f );
  float3 v1 = make_float3( 32.0f, 0.0f, 0.0f );
  float3 v2 = make_float3( 0.0f, 0.0f, 32.0f );
  float3 normal = cross( v2, v1 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  float4 plane = make_float4( normal, d );
  parallelogram["plane"]->setFloat( plane );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["anchor"]->setFloat( anchor );

  // Materials
  std::string box_chname;
  box_chname = "closest_hit_radiance3";

  Material box_matl = _context->createMaterial();
  Program box_ch = _context->createProgramFromPTXFile( _ptx_path, box_chname );
  box_matl->setClosestHitProgram( 0, box_ch );
  Program box_ah = _context->createProgramFromPTXFile( _ptx_path, "any_hit_shadow" );
  box_matl->setAnyHitProgram( 1, box_ah );

  box_matl["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  box_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f );
  box_matl["Ks"]->setFloat( 1.0f, 1.0f, 1.0f );
  box_matl["phong_exp"]->setFloat( 88 );
  box_matl["reflectivity"]->setFloat( 0.05f, 0.05f, 0.05f );
  box_matl["reflectivity_n"]->setFloat( 0.2f, 0.2f, 0.2f );
  box_matl["obj_id"]->setInt(1);


  
  std::string sph_chname;
  sph_chname = "closest_hit_radiance3";
  Material sph_matl = _context->createMaterial();
  Program sph_ch = _context->createProgramFromPTXFile( _ptx_path, sph_chname );
  sph_matl->setClosestHitProgram( 0, sph_ch );
  Program sph_ah = _context->createProgramFromPTXFile( _ptx_path, "any_hit_shadow" );
  sph_matl->setAnyHitProgram( 1, sph_ah );

  sph_matl["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  sph_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f );
  sph_matl["Ks"]->setFloat( 1.0f, 1.0f, 1.0f );
  sph_matl["phong_exp"]->setFloat( 88 );
  sph_matl["reflectivity"]->setFloat( 0.05f, 0.05f, 0.05f );
  sph_matl["reflectivity_n"]->setFloat( 0.2f, 0.2f, 0.2f );
  sph_matl["obj_id"]->setInt(1);

  std::string floor_chname;
  floor_chname = "closest_hit_radiance3";
  Material floor_matl = _context->createMaterial();
  Program floor_ch = _context->createProgramFromPTXFile( _ptx_path, floor_chname );
  floor_matl->setClosestHitProgram( 0, floor_ch );

  Program floor_ah = _context->createProgramFromPTXFile( _ptx_path, "any_hit_shadow" );
  floor_matl->setAnyHitProgram( 1, floor_ah );

  floor_matl["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  //floor_matl["Kd"]->setFloat( 194/255.f*.6f, 186/255.f*.6f, 151/255.f*.6f );
  //floor_matl["Ks"]->setFloat( 0.4f, 0.4f, 0.4f );
  floor_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f );
  floor_matl["Ks"]->setFloat( 1.0f, 1.0f, 1.0f );
  floor_matl["reflectivity"]->setFloat( 0.1f, 0.1f, 0.1f );
  floor_matl["reflectivity_n"]->setFloat( 0.05f, 0.05f, 0.05f );
  floor_matl["phong_exp"]->setFloat( 88 );
  floor_matl["tile_v0"]->setFloat( 0.25f, 0, .15f );
  floor_matl["tile_v1"]->setFloat( -.15f, 0, 0.25f );
  floor_matl["crack_color"]->setFloat( 0.1f, 0.1f, 0.1f );
  floor_matl["crack_width"]->setFloat( 0.02f );
  floor_matl["obj_id"]->setInt(2);


  // Glass material
  Material glass_matl;
  if( chull.get() ) {
    Program glass_ch = _context->createProgramFromPTXFile( _ptx_path, "glass_closest_hit_radiance" );
    std::string glass_ahname;
    glass_ahname = "any_hit_shadow";
    Program glass_ah = _context->createProgramFromPTXFile( _ptx_path, glass_ahname );
    glass_matl = _context->createMaterial();
    glass_matl->setClosestHitProgram( 0, glass_ch );
    glass_matl->setAnyHitProgram( 1, glass_ah );

    glass_matl["importance_cutoff"]->setFloat( 1e-2f );
    glass_matl["cutoff_color"]->setFloat( 0.34f, 0.55f, 0.85f );
    glass_matl["fresnel_exponent"]->setFloat( 3.0f );
    glass_matl["fresnel_minimum"]->setFloat( 0.1f );
    glass_matl["fresnel_maximum"]->setFloat( 1.0f );
    glass_matl["refraction_index"]->setFloat( 1.4f );
    glass_matl["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    glass_matl["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    glass_matl["refraction_maxdepth"]->setInt( 100 );
    glass_matl["reflection_maxdepth"]->setInt( 100 );
    float3 extinction = make_float3(.80f, .89f, .75f);
    glass_matl["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
    glass_matl["shadow_attenuation"]->setFloat( 0.4f, 0.7f, 0.4f );
  }

  // Create GIs for each piece of geometry
  std::vector<GeometryInstance> gis;
  gis.push_back( _context->createGeometryInstance( box, &box_matl, &box_matl+1 ) );
  gis.push_back( _context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );
  //gis.push_back( _context->createGeometryInstance( sphere, &sph_matl, &sph_matl+1 ) );
  if(chull.get())
    gis.push_back( _context->createGeometryInstance( chull, &glass_matl, &glass_matl+1 ) );

  // Place all in group
  int ct = geomgroup->getChildCount();
  geomgroup->setChildCount( ct + 1 );
  geomgroup->setChild( ct, gis[1] );

  GeometryGroup geometrygroup = _context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  geometrygroup->setChild( 0, gis[0] );
  geometrygroup->setChild( 1, gis[1] );
  //geometrygroup->setChild( 2, gis[2] );
  if(chull.get())
    geometrygroup->setChild( 3, gis[3] );
  geometrygroup->setAcceleration( _context->createAcceleration("NoAccel","NoAccel") );

  _context["top_object"]->set( geometrygroup );
  _context["top_shadower"]->set( geometrygroup );
}


//-----------------------------------------------------------------------------
//
// Main driver
//
//-----------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -t  | --texture-path <path>                Specify path to texture directory\n"
    << "        --dim=<width>x<height>               Set image dimensions\n"
    << std::endl;

  if ( doExit ) exit(1);
}


int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  unsigned int width = 1080u, height = 720u;
  //unsigned int width = 1600u, height = 800u;

  std::string texture_path;
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else if ( arg.substr( 0, 6 ) == "--dim=" ) {
      std::string dims_arg = arg.substr(6);
      if ( sutilParseImageDimensions( dims_arg.c_str(), &width, &height ) != RT_SUCCESS ) {
        std::cerr << "Invalid window dimensions: '" << dims_arg << "'" << std::endl;
        printUsageAndExit( argv[0] );
      }
    } else if ( arg == "-t" || arg == "--texture-path" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      texture_path = argv[++i];
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  if( texture_path.empty() ) {
    texture_path = std::string( sutilSamplesDir() ) + "/ambocc_research/data";
  }

  std::stringstream title;
  title << "ambocc";
  try {
    Softshadow scene(texture_path);
    scene.setDimensions( width, height );
    //dont time out progressive
    GLUTDisplay::setProgressiveDrawingTimeout(0.0);
    GLUTDisplay::run( title.str(), &scene, GLUTDisplay::CDProgressive );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
  return 0;
}
