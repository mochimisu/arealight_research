/*
* arealight.cpp
* Area Light Filtering
* Adapted from NVIDIA OptiX Tutorial
* Brandon Wang, Soham Mehta, Ravi Ramamoorthi
*/

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
#include <limits>
#include "random.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <Mouse.h>


using namespace optix;

class Arealight : public SampleScene
{
public:
  Arealight(const std::string& texture_path)
    : SampleScene(), _width(1080u), _height(720u), texture_path( texture_path )
    , _frame_number( 0 ), _keep_trying( 1 )
  {
  }

  // From SampleScene
  void   initScene( InitialCameraData& camera_data );
  void   trace( const RayGenCameraData& camera_data );
  void   doResize( unsigned int width, unsigned int height );
  void   setDimensions( const unsigned int w, const unsigned int h ) 
  { 
    _width = w; _height = h; 
  }
  Buffer getOutputBuffer();

  virtual bool   keyPressed(unsigned char key, int x, int y);

private:
  std::string texpath( const std::string& base );
  void resetAccumulation();
  void createGeometry();

  unsigned int  _frame_number;
  unsigned int  _keep_trying;

  Buffer       _brdf;
  Buffer       _vis;
  GeometryGroup geomgroup;

  unsigned int _width;
  unsigned int _height;
  std::string   texture_path;
  std::string  _ptx_path;

  uint _blur_occ;
  uint _blur_wxf;
  uint _view_mode;

  uint _normal_rpp;
  uint _brute_rpp;
  uint _max_rpp_pass;
  int2 _pixel_radius;
  int2 _pixel_radius_wxf;


  AreaLight * _env_lights;
  uint _show_brdf;
  uint _show_occ;
  float _sigma;

  Buffer light_buffer;

  float _anim_t;
  double _previous_frame_time;
  bool _is_anim;
  bool _move_light;
  bool _move_camera;

  Group _top_grp;

};

Arealight* _scene;
int output_num = 0;

void Arealight::initScene( InitialCameraData& camera_data )
{
  _anim_t = 0;
  sutilCurrentTime(&_previous_frame_time);
  _is_anim = true;
  _move_light = true;
  _move_camera = false;
  // set up path to ptx file associated with tutorial number
  std::stringstream ss;
  ss << "arealight.cu";
  _ptx_path = ptxpath( "arealight", ss.str() );

  // context 
  m_context->setRayTypeCount( 2 );
  m_context->setEntryPointCount( 7 );
  m_context->setStackSize( 8000 );

  m_context["max_depth"]->setInt(100);
  m_context["radiance_ray_type"]->setUint(0);
  m_context["shadow_ray_type"]->setUint(1);
  m_context["frame_number"]->setUint( 0u );
  m_context["scene_epsilon"]->setFloat( 1.e-3f );
  m_context["importance_cutoff"]->setFloat( 0.01f );
  m_context["ambient_light_color"]->setFloat( 0.31f, 0.33f, 0.28f );

  m_context["output_buffer"]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4,
        _width, _height) );

  Buffer shadow_rng_seeds = m_context->createBuffer(RT_BUFFER_INPUT, 
      RT_FORMAT_UNSIGNED_INT2, _width, _height);
  m_context["shadow_rng_seeds"]->set(shadow_rng_seeds);
  uint2* seeds = reinterpret_cast<uint2*>( shadow_rng_seeds->map() );
  for(unsigned int i = 0; i < _width * _height; ++i )
    seeds[i] = random2u();
  shadow_rng_seeds->unmap();

  // BRDF buffer
  _brdf = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3,
      _width, _height );
  m_context["brdf"]->set( _brdf );

  // Occlusion buffer
  _vis = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
      RT_FORMAT_FLOAT3, _width, _height );
  m_context["vis"]->set( _vis );

  // Blurred (on one dimension) cclusion accumulation buffer
  Buffer _occ_blur = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT |
      RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
  m_context["vis_blur1d"]->set( _occ_blur );

  // samples per pixel buffer
  Buffer spp = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, 
      RT_FORMAT_FLOAT, _width, _height );
  m_context["spp"]->set( spp );

  // current samples per pixel buffer
  Buffer spp_cur = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT,
      RT_FORMAT_FLOAT, _width, _height );
  m_context["spp_cur"]->set( spp_cur );

  Buffer slope = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT |
      RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT2, _width, _height );
  m_context["slope"]->set( slope );

  // gauss values
  Buffer gauss_lookup = m_context->createBuffer( RT_BUFFER_INPUT, 
      RT_FORMAT_FLOAT, 65);
  m_context["gaussian_lookup"]->set( gauss_lookup );

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
  const float exp_lookup[60] = {1.0000,    0.9048,    0.8187,    0.7408,    
    0.6703,    0.6065,    0.5488,    0.4966,    0.4493,    0.4066,   
    0.3679,    0.3329,    0.3012,    0.2725,    0.2466,    0.2231,   
    0.2019,    0.1827,    0.1653,    0.1496,    0.1353,    0.1225,    
    0.1108,    0.1003,    0.0907,    0.0821,    0.0743,   0.0672,    
    0.0608,    0.0550,    0.0498,    0.0450,    0.0408,    0.0369,    
    0.0334,    0.0302,    0.0273,    0.0247,    0.0224,    0.0202,   
    0.0183,    0.0166,    0.0150,    0.0136,    0.0123,    0.0111,    
    0.0101,    0.0091,    0.0082,    0.0074,    0.0067,    0.0061,    
    0.0055,    0.0050,    0.0045,    0.0041,    0.0037,    0.0033,    
    0.0030,    0.0027 };

  for(int i=0; i<65; i++) {
    lookups[i] = gaussian_lookup[i];
  }
  gauss_lookup->unmap();

  // world space buffer
  Buffer world_loc = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | 
      RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  m_context["world_loc"]->set( world_loc );

  Buffer n = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | 
      RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  m_context["n"]->set( n );

  Buffer filter_n = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | 
      RT_BUFFER_GPU_LOCAL, RT_FORMAT_INT, _width, _height );
  m_context["use_filter_n"]->set( filter_n );

  Buffer filter_occ = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | 
      RT_BUFFER_GPU_LOCAL, RT_FORMAT_INT, _width, _height );
  m_context["use_filter_occ"]->set( filter_occ );

  Buffer filter_occ_filter1d = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT 
      | RT_BUFFER_GPU_LOCAL, RT_FORMAT_INT, _width, _height );
  m_context["use_filter_occ_filter1d"]->set( filter_occ_filter1d );

  Buffer s1s2_blur1d = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT
      | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT2, _width, _height );
  m_context["slope_filter1d"]->set( s1s2_blur1d );

  Buffer spp_blur1d = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT
      | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
  m_context["spp_filter1d"]->set( spp_blur1d );

  Buffer dist_to_light = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT
      | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
  m_context["dist_to_light"]->set( dist_to_light );

  Buffer proj_d = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT
      | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
  m_context["proj_d"]->set( proj_d );

  Buffer obj_id = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT
      | RT_BUFFER_GPU_LOCAL, RT_FORMAT_INT, _width, _height );
  m_context["obj_id_b"]->set( obj_id );

  _blur_occ = 1;
  m_context["blur_occ"]->setUint(_blur_occ);

  _blur_wxf = 0;
  m_context["blur_wxf"]->setUint(_blur_wxf);

  _view_mode = 0;
  m_context["view_mode"]->setUint(_view_mode);

  _show_brdf = 1;
  m_context["show_brdf"]->setUint(_show_brdf);

  _show_occ = 1;
  m_context["show_occ"]->setUint(_show_occ);

  m_context["image_dim"]->setUint(make_uint2(_width, _height));


  _normal_rpp = 3;
  _brute_rpp = 2000;
  _max_rpp_pass = 8;
  float spp_mu = 2;

  m_context["normal_rpp"]->setUint(_normal_rpp);
  m_context["brute_rpp"]->setUint(_brute_rpp);
  m_context["max_rpp_pass"]->setUint(_max_rpp_pass);

  m_context["spp_mu"]->setFloat(spp_mu);

  _pixel_radius = make_int2(10,10);
  m_context["pixel_radius"]->setInt(_pixel_radius);

  // Sampling program
  std::string camera_name;
  camera_name = "pinhole_camera_initial_sample";

  Program ray_gen_program = m_context->createProgramFromPTXFile( _ptx_path,
      camera_name );
  m_context->setRayGenerationProgram( 0, ray_gen_program );

  // continual Sampling
  std::string continue_sampling = "pinhole_camera_continue_sample";

  Program continue_sampling_program = m_context->createProgramFromPTXFile(
      _ptx_path, continue_sampling );
  m_context->setRayGenerationProgram( 6, continue_sampling_program );

  // Occlusion Filter programs
  std::string first_pass_occ_filter_name = "occlusion_filter_first_pass";
  Program first_occ_filter_program = m_context->createProgramFromPTXFile(
      _ptx_path, first_pass_occ_filter_name );
  m_context->setRayGenerationProgram( 2, first_occ_filter_program );
  std::string second_pass_occ_filter_name = "occlusion_filter_second_pass";
  Program second_occ_filter_program = m_context->createProgramFromPTXFile( 
      _ptx_path, second_pass_occ_filter_name );
  m_context->setRayGenerationProgram( 3, second_occ_filter_program );

  // S1, S2 Filter programs
  std::string first_pass_s1s2_filter_name = "s1s2_filter_first_pass";
  Program first_s1s2_filter_program = m_context->createProgramFromPTXFile( 
      _ptx_path, first_pass_s1s2_filter_name );
  m_context->setRayGenerationProgram( 4, first_s1s2_filter_program );
  std::string second_pass_s1s2_filter_name = "s1s2_filter_second_pass";
  Program second_s1s2_filter_program = m_context->createProgramFromPTXFile( 
      _ptx_path, second_pass_s1s2_filter_name );
  m_context->setRayGenerationProgram( 5, second_s1s2_filter_program );


  // Display program
  std::string display_name;
  display_name = "display_camera";

  Program display_program = m_context->createProgramFromPTXFile( _ptx_path,
      display_name );
  m_context->setRayGenerationProgram( 1, display_program );

  // Exception / miss programs
  Program exception_program = m_context->createProgramFromPTXFile( _ptx_path, 
      "exception" );
  m_context->setExceptionProgram( 0, exception_program );
  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );

  std::string miss_name;
  miss_name = "miss";
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( _ptx_path, 
        miss_name ) );
  const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
  m_context["bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );

  // grids2
  float3 pos = make_float3(-4.5, 16, 8);
  float3 pos1 = make_float3(1.5, 16, 8);
  float3 pos2 = make_float3(-4.5, 21.8284, 3.8284);
  float3 axis1 = pos1-pos;
  float3 axis2 = pos2-pos;

  float3 norm = cross(axis1,axis2);

  AreaLight lights[] = {
    { pos,
    pos1,
    pos2,
    make_float3(1.0f, 1.0f, 1.0f)
    }
  };

  float3 normed_norm = normalize(norm);
  m_context["lightnorm"]->setFloat(normed_norm);

  _sigma = sqrt(length(norm)/4.0f);

  m_context["light_sigma"]->setFloat(_sigma);

  _env_lights = lights;
  light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(AreaLight));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context["lights"]->set(light_buffer);


  // Set up camera
  camera_data = InitialCameraData( make_float3( -4.5f, 2.5f, 5.5f ), // eye
    make_float3( 0.0f, 0.5f,  0.0f ), // lookat
    make_float3( 0.0f, 1.0f,  0.0f ), // up
    60 );                             // vfov

  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );


  // Populate scene hierarchy
  createGeometry();

  //Initialize progressive accumulation
  resetAccumulation();

  // Prepare to run
  m_context->validate();
  m_context->compile();
}


Buffer Arealight::getOutputBuffer()
{

  return m_context["output_buffer"]->getBuffer();
}

void Arealight::trace( const RayGenCameraData& camera_data )
{
  _frame_number ++;

  if(m_camera_changed) {
    m_context["numAvg"]->setUint(1);
    m_camera_changed = false;
    resetAccumulation();
  }


  double t;
  if(GLUTDisplay::getContinuousMode() != GLUTDisplay::CDNone) {
    sutilCurrentTime(&t);
  } else {
    t = _previous_frame_time;
  }
  
  double time_elapsed = t - _previous_frame_time;
  _previous_frame_time = t;

  if (_is_anim) {
    _anim_t += 0.7 * time_elapsed;
  }

  if (_move_light && _is_anim) {
    float3 d = make_float3(0.15*sin(_anim_t/1.0),0.2*cos(_anim_t/1.0),
        0.1*sin(_anim_t/1.2));
    AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
    lights[0].v1 -= d;
    lights[0].v2 -= d;
    lights[0].v3 -= d;

    light_buffer->unmap();
  }


  if (_move_camera && _is_anim) {
    float3 eye, u, v, w;
    eye.x = (float) (camera_data.eye.x * sin(_anim_t));
    eye.y = (float)( 0.2 + camera_data.eye.y + cos( _anim_t*1.5 ) );
    eye.z = (float)( 0.5+camera_data.eye.z*cos( _anim_t ) );
    float3 lookat = make_float3(0);

    PinholeCamera pc(eye, lookat, make_float3(0,1,0), 60.f, 60.f/(640.0/480.0));
    pc.getEyeUVW( eye, u, v, w );
    m_context["eye"]->setFloat( eye );
    m_context["U"]->setFloat( u );
    m_context["V"]->setFloat( v );
    m_context["W"]->setFloat( w );
  } else {

    m_context["eye"]->setFloat( camera_data.eye );
    m_context["U"]->setFloat( camera_data.U );
    m_context["V"]->setFloat( camera_data.V );
    m_context["W"]->setFloat( camera_data.W );
  }

  Buffer shadow_rng_seeds = m_context["shadow_rng_seeds"]->getBuffer();
  uint2* seeds = reinterpret_cast<uint2*>( shadow_rng_seeds->map() );
  for(unsigned int i = 0; i < _width * _height; ++i )
    seeds[i] = random2u();
  shadow_rng_seeds->unmap();

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );
  m_context["frame"]->setUint( _frame_number );

  int num_resample = ceil((float)_brute_rpp * _brute_rpp 
      / (_max_rpp_pass * _max_rpp_pass));

  //Initial 16 Samples
  m_context->launch( 0, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  //Filter s1,s2
  m_context->launch( 4, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  m_context->launch( 5, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  //Resample
  m_context->launch( 6, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  //Filter occlusion
  m_context->launch( 2, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  m_context->launch( 3, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  //Display
 
  m_context->launch( 1, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
}


void Arealight::doResize( unsigned int width, unsigned int height )
{
  // output buffer handled in SampleScene::resize
}

std::string Arealight::texpath( const std::string& base )
{
  return texture_path + "/" + base;
}

float4 make_plane( float3 n, float3 p )
{
  n = normalize(n);
  float d = -dot(n, p);
  return make_float4( n, d );
}

void Arealight::resetAccumulation()
{
  _frame_number = 0;
  m_context["frame"]->setUint( _frame_number );
}


bool Arealight::keyPressed(unsigned char key, int x, int y) {
  float delta = 0.5f;

  Buffer spp;
  switch(key) {
  case 'U':
  case 'u':
    {
      float3 d = make_float3(delta,0,0);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 += d;
      lights[0].v2 += d;
      lights[0].v3 += d;

      light_buffer->unmap();

      return true;
    }
  case 'J':
  case 'j':
    {
      float3 d = make_float3(delta,0,0);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 -= d;
      lights[0].v2 -= d;
      lights[0].v3 -= d;

      light_buffer->unmap();

      return true;
    }
  case 'I':
  case 'i':
    {
      float3 d = make_float3(0,delta,0);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 += d;
      lights[0].v2 += d;
      lights[0].v3 += d;

      light_buffer->unmap();

      return true;
    }
  case 'K':
  case 'k':
    {
      float3 d = make_float3(0,delta,0);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 -= d;
      lights[0].v2 -= d;
      lights[0].v3 -= d;

      light_buffer->unmap();

      return true;
    }

  case 'O':
  case 'o':
    {
      float3 d = make_float3(0,0,delta);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 += d;
      lights[0].v2 += d;
      lights[0].v3 += d;

      light_buffer->unmap();

      return true;
    }
  case 'L':
  case 'l':
    {
      float3 d = make_float3(0,0,delta);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 -= d;
      lights[0].v2 -= d;
      lights[0].v3 -= d;

      light_buffer->unmap();

      return true;
    }

  case 'M':
  case 'm':
    _show_brdf = 1-_show_brdf;
    m_context["show_brdf"]->setUint(_show_brdf);
    if (_show_brdf)
      std::cout << "BRDF: On" << std::endl;
    else
      std::cout << "BRDF: Off" << std::endl;
    return true;
  case 'N':
  case 'n':
    _show_occ = 1-_show_occ;
    m_context["show_occ"]->setUint(_show_occ);
    if (_show_occ)
      std::cout << "Occlusion Display: On" << std::endl;
    else
      std::cout << "Occlusion Display: Off" << std::endl;
    return true;



  case 'V':
  case 'v':
    {
      std::cout << "SPP stats" << std:: endl;
      Buffer cur_spp = m_context["spp_cur"]->getBuffer();
      Buffer brdf = m_context["brdf"]->getBuffer();
      spp = m_context["spp"]->getBuffer();
      float min_cur_spp = 10000000.0;
      float max_cur_spp = 0.0;
      float avg_cur_spp = 0.0;
      float min_spp = 100000000.0; 
      float max_spp = 0.0;
      float avg_spp = 0.0;
      float* spp_arr = reinterpret_cast<float*>( spp->map() );
      float3* brdf_arr = reinterpret_cast<float3*>( brdf->map() );
      int num_avg = 0;

      int num_low = 0;
      for(unsigned int j = 0; j < _height; ++j ) {
        for(unsigned int i = 0; i < _width; ++i ) {
          float cur_brdf_x = brdf_arr[i+j*_width].x;
          if (cur_brdf_x > -1) {
            float cur_spp_val = spp_arr[i+j*_width];
            if (cur_spp_val > -0.001) {
              min_spp = min(min_spp,cur_spp_val);
              max_spp = max(max_spp,cur_spp_val);
              avg_spp += cur_spp_val;
              num_avg++;

            }
          } 
        }
      }
      spp->unmap();
      avg_spp /= num_avg;
      uint2 err_loc;
      uint2 err_first_loc;
      bool first_loc_set = false;
      int num_cur_avg = 0;
      float* cur_spp_arr = reinterpret_cast<float*>( cur_spp->map() );
      for(unsigned int j = 0; j < _height; ++j ) {
        for(unsigned int i = 0; i < _width; ++i ) {
          float cur_brdf_x = brdf_arr[i+j*_width].x;
          if (cur_brdf_x > -1) {
            float cur_spp_val = cur_spp_arr[i+j*_width];
            if (cur_spp_val > -0.001) {
              min_cur_spp = min(min_cur_spp,cur_spp_val);
              max_cur_spp = max(max_cur_spp,cur_spp_val);
              avg_cur_spp += cur_spp_val;
              num_cur_avg++;
            }
          }
        }
      }
      cur_spp->unmap();
      brdf->unmap();
      avg_cur_spp /= num_cur_avg;

      std::cout << "Minimum SPP: " << min_cur_spp << std::endl;
      std::cout << "Maximum SPP: " << max_cur_spp << std::endl;
      std::cout << "Average SPP: " << avg_cur_spp << std::endl;
      std::cout << "Minimum Theoretical SPP: " << min_spp << std::endl;
      std::cout << "Maximum Theoretical SPP: " << max_spp << std::endl;
      std::cout << "Average Theoretical SPP: " << avg_spp << std::endl;
      return true;
    }
  case 'C':
  case 'c':
    _sigma += 0.1;
    m_context["light_sigma"]->setFloat(_sigma);
    std::cout << "Light sigma is now: " << _sigma << std::endl;
    m_camera_changed = true;
    return true;
  case 'X':
  case 'x':
    _sigma -= 0.1;
    m_context["light_sigma"]->setFloat(_sigma);
    std::cout << "Light sigma is now: " << _sigma << std::endl;
    m_camera_changed = true;
    return true;

  case 'A':
  case 'a':
    std::cout << _frame_number << " frames." << std::endl;
    return true;
  case 'B':
  case 'b':
    _blur_occ = 1-_blur_occ;
    m_context["blur_occ"]->setUint(_blur_occ);
    if (_blur_occ)
      std::cout << "Blur: On" << std::endl;
    else
      std::cout << "Blur: Off" << std::endl;
    return true;

  case 'H':
  case 'h':
    _blur_wxf = 1-_blur_wxf;
    m_context["blur_wxf"]->setUint(_blur_wxf);
    if (_blur_wxf)
      std::cout << "Blur Omega x f: On" << std::endl;
    else
      std::cout << "Blur Omega x f: Off" << std::endl;
    return true;
  case 'Z':
    _view_mode = (_view_mode+9)%11;
  case 'z':
    _view_mode = (_view_mode+1)%11;
    m_context["view_mode"]->setUint(_view_mode);
    switch(_view_mode) {
    case 0:
      std::cout << "View mode: Normal" << std::endl;
      break;
    case 1:
      std::cout << "View mode: Occlusion Only" << std::endl;
      break;
    case 2:
      std::cout << "View mode: Scale" << std::endl;
      break;
    case 3:
      std::cout << "View mode: Current SPP" << std::endl;
      break;
    case 4:
      std::cout << "View mode: Theoretical SPP" << std::endl;
      break;
    case 5:
      std::cout << "Use filter (normals)" << std::endl;
      break;
    case 6:
      std::cout << "Use filter (unoccluded)" << std::endl;
      break;
    case 7:
      std::cout << "View unconverged pixels" << std::endl;
      break;
    case 8:
      std::cout << "View Object ID" << std::endl;
      break;
    case 9:
      std::cout << "View s1" << std::endl;
      break;
    case 10:
      std::cout << "View s2" << std::endl;
        break;
    default:
      std::cout << "View mode: Unknown" << std::endl;
      break;
    }
    return true;
  case '.':
    if(_pixel_radius.x > 1000)
      return true;
    _pixel_radius += make_int2(1,1);
    m_context["pixel_radius"]->setInt(_pixel_radius);
    std::cout << "Pixel radius now: " << _pixel_radius.x << "," 
      << _pixel_radius.y << std::endl;
    return true;
  case ',':
    if (_pixel_radius.x < 2)
      return true;
    _pixel_radius -= make_int2(1,1);
    m_context["pixel_radius"]->setInt(_pixel_radius);
    std::cout << "Pixel radius now: " << _pixel_radius.x << "," 
      << _pixel_radius.y << std::endl;
    return true;
  case 'S':
  case 's':
    {
      std::stringstream fname;
      fname << "output_";
      fname << std::setw(7) << std::setfill('0') << output_num;
      fname << ".ppm";
      Buffer output_buf = _scene->getOutputBuffer();
      sutilDisplayFilePPM(fname.str().c_str(), output_buf->get());
      output_num++;
      std::cout << "Saved file" << std::endl;
      return true;
    }
  case 'Y':
  case 'y':
    return true;

  case 'R':
  case 'r':
    sutilCurrentTime(&_previous_frame_time);
    _anim_t = 0;
    return true;
  case 'T':
  case 't':
    _is_anim = !_is_anim;
    return true;
  case '[':
    _move_light = !_move_light;
    return true;
  case ']':
    _move_camera = !_move_camera;
    return true;
  }
  return false;

}

void appendGeomGroup(GeometryGroup& target, GeometryGroup& source)
{
  int ct_target = target->getChildCount();
  int ct_source = source->getChildCount();
  target->setChildCount(ct_target+ct_source);
  for(int i=0; i<ct_source; i++)
    target->setChild(ct_target + i, source->getChild(i));
}
//grids2
void Arealight::createGeometry()
{
  //Intersection programs
  Program closest_hit = m_context->createProgramFromPTXFile(_ptx_path, 
      "closest_hit_radiance3");
  Program any_hit = m_context->createProgramFromPTXFile(_ptx_path, 
      "any_hit_shadow");

  //Make some temp geomgroups
  _top_grp = m_context->createGroup();
  GeometryGroup floor_geom_group = m_context->createGeometryGroup();
  GeometryGroup grid1_geom_group = m_context->createGeometryGroup();
  GeometryGroup grid3_geom_group = m_context->createGeometryGroup();
  GeometryGroup grid2_geom_group = m_context->createGeometryGroup();

  //Set some materials
  Material floor_mat = m_context->createMaterial();
  floor_mat->setClosestHitProgram(0, closest_hit);
  floor_mat->setAnyHitProgram(1, any_hit);
  floor_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  floor_mat["Kd"]->setFloat( 0.87402f, 0.87402f, 0.87402f );
  floor_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  floor_mat["phong_exp"]->setFloat( 100.0f );
  floor_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  floor_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  floor_mat["obj_id"]->setInt(10);

  Material grid1_mat = m_context->createMaterial();
  grid1_mat->setClosestHitProgram(0, closest_hit);
  grid1_mat->setAnyHitProgram(1, any_hit);
  grid1_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid1_mat["Kd"]->setFloat( 0.72f, 0.100741f, 0.09848f );
  grid1_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid1_mat["phong_exp"]->setFloat( 100.0f );
  grid1_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid1_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid1_mat["obj_id"]->setInt(11);

  Material grid2_mat = m_context->createMaterial();
  grid2_mat->setClosestHitProgram(0, closest_hit);
  grid2_mat->setAnyHitProgram(1, any_hit);
  grid2_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid2_mat["Kd"]->setFloat( 0.0885402f, 0.77f, 0.08316f );
  grid2_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid2_mat["phong_exp"]->setFloat( 100.0f );
  grid2_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid2_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid2_mat["obj_id"]->setInt(12);

  Material grid3_mat = m_context->createMaterial();
  grid3_mat->setClosestHitProgram(0, closest_hit);
  grid3_mat->setAnyHitProgram(1, any_hit);
  grid3_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid3_mat["Kd"]->setFloat( 0.123915f, 0.192999f, 0.751f );
  grid3_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid3_mat["phong_exp"]->setFloat( 100.0f );
  grid3_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid3_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid3_mat["obj_id"]->setInt(13);

  //Transformations
  Matrix4x4 floor_xform = Matrix4x4::identity();
  float *floor_xform_m = floor_xform.getData();
  floor_xform_m[0] = 4.0;
  floor_xform_m[10] = 4.0;
  floor_xform_m[12] = 0.0778942;
  floor_xform_m[14] = 0.17478;

  Matrix4x4 grid1_xform = Matrix4x4::identity();
  float *grid1_xform_m = grid1_xform.getData();
  grid1_xform_m[0] = 0.75840;
  grid1_xform_m[1] = 0.6232783;
  grid1_xform_m[2] = -0.156223;
  grid1_xform_m[3] = 0.0;
  grid1_xform_m[4] = -0.465828;
  grid1_xform_m[5] = 0.693876;
  grid1_xform_m[6] = 0.549127;
  grid1_xform_m[7] = 0.0;
  grid1_xform_m[8] = 0.455878;
  grid1_xform_m[9] = -0.343688;
  grid1_xform_m[10] = 0.821008;
  grid1_xform_m[11] = 0.0;
  grid1_xform_m[12] = 2.18526;
  grid1_xform_m[13] = 1.0795;
  grid1_xform_m[14] = 1.23179;
  grid1_xform_m[15] = 1.0;

  Matrix4x4 grid2_xform = Matrix4x4::identity();
  float *grid2_xform_m = grid2_xform.getData();
  grid2_xform_m[0] = 0.893628;
  grid2_xform_m[1] = 0.203204;
  grid2_xform_m[2] = -0.40017;
  grid2_xform_m[3] = 0.0;
  grid2_xform_m[4] = 0.105897;
  grid2_xform_m[5] = 0.770988;
  grid2_xform_m[6] = 0.627984;
  grid2_xform_m[7] = 0.0;
  grid2_xform_m[8] = 0.436135;
  grid2_xform_m[9] = -0.603561;
  grid2_xform_m[10] = 0.667458;
  grid2_xform_m[11] = 0.0;
  grid2_xform_m[12] = 0.142805;
  grid2_xform_m[13] = 1.0837;
  grid2_xform_m[14] = 0.288514;
  grid2_xform_m[15] = 1.0;



  Matrix4x4 grid3_xform = Matrix4x4::identity();
  float *grid3_xform_m = grid3_xform.getData();
  grid3_xform_m[0] = 0.109836;
  grid3_xform_m[1] = 0.392525;
  grid3_xform_m[2] = -0.913159;
  grid3_xform_m[3] = 0.0;
  grid3_xform_m[4] = 0.652392;
  grid3_xform_m[5] = 0.664651;
  grid3_xform_m[6] = 0.364174;
  grid3_xform_m[7] = 0.0;
  grid3_xform_m[8] = 0.74988;
  grid3_xform_m[9] = -0.635738;
  grid3_xform_m[10] = -0.183078;
  grid3_xform_m[11] = 0.0;
  grid3_xform_m[12] = -2.96444;
  grid3_xform_m[13] = 1.86879;
  grid3_xform_m[14] = 1.00696;
  grid3_xform_m[15] = 1.0;

  floor_xform = floor_xform.transpose();
  grid1_xform = grid1_xform.transpose();
  grid2_xform = grid2_xform.transpose();
  grid3_xform = grid3_xform.transpose();

  //Load the OBJ's
  ObjLoader * floor_loader = new ObjLoader(texpath("grids2/floor.obj").c_str(),
      m_context, floor_geom_group, floor_mat );
  floor_loader->load(floor_xform);
  ObjLoader * grid1_loader = new ObjLoader(texpath("grids2/grid1.obj").c_str(),
      m_context, grid1_geom_group, grid1_mat );
  grid1_loader->load(grid1_xform);
  ObjLoader * grid3_loader = new ObjLoader(texpath("grids2/grid3.obj").c_str(),
      m_context, grid3_geom_group, grid3_mat );
  grid3_loader->load(grid3_xform);
  ObjLoader * grid2_loader = new ObjLoader(texpath("grids2/grid2.obj").c_str(),
      m_context, grid2_geom_group, grid2_mat );
  grid2_loader->load(grid2_xform);



  //Make one big geom group
  GeometryGroup geom_group = m_context->createGeometryGroup();

  geom_group->setChildCount(0);

  Transform _trans = m_context->createTransform();
  Transform _trans2 = m_context->createTransform();

  grid2_geom_group->getAcceleration()->setProperty("refit", "1");
  grid3_geom_group->getAcceleration()->setProperty("refit", "1");

  Geometry _anim_geom = grid2_geom_group->getChild(0)->getGeometry();
  GeometryGroup _anim_geom_group = grid2_geom_group;
  

  optix::Matrix4x4 test = optix::Matrix4x4::translate(make_float3(0,0,0));
  _trans->setMatrix( false, test.getData(), 0);
  _trans->setChild( grid2_geom_group );

  _trans2->setMatrix( false, test.getData(), 0);
  _trans2->setChild( grid3_geom_group );


  appendGeomGroup(geom_group, floor_geom_group);
  appendGeomGroup(geom_group, grid1_geom_group);
  geom_group->setAcceleration( m_context->createAcceleration("Bvh", "Bvh") );

  _top_grp->setChildCount(3);
  _top_grp->setChild(0, geom_group);
  _top_grp->setChild(1, _trans);
  _top_grp->setChild(2, _trans2);

  _top_grp->setAcceleration(m_context->createAcceleration("Bvh", "Bvh") );
  _top_grp->getAcceleration()->setProperty("refit", "1");

  //Set the geom group
  m_context["top_object"]->set( _top_grp );
  m_context["top_shadower"]->set( _top_grp );

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
    << "  -h  | --help                               Print this  message\n"
    << "  -t  | --texture-path <path>                Specify texture dir\n"
    << "        --dim=<width>x<height>               Set image dimensions\n"
    << std::endl;

  std::cout
    << "Key bindings:" << std::endl
    << "z/Z: Cycle through visualizations" << std::endl
    << "s: Save current frame" << std::endl
    << "t: Toggle all animation" << std::endl
    << "[: Toggle moving light" << std::endl
    << "]: Toggle moving camera" << std::endl

    << "c/x: Increase/decrease light sigma" << std::endl
    << "a: Current frame number" << std::endl
    << "b: Toggle occlusion filtering" << std::endl
    << "h: Toggle omega_x_f filtering" << std::endl

    << "./,: Increase/decrease pixel radius" << std::endl
    << "v: Output SPP stats" << std::endl
    << "m: Toggle BRDF display" << std::endl
    << "n: Toggle Occlusion" << std::endl
    << "u/j, i/k, o/l: Increase/decrease light in x,y,z" << std::endl
    << std::endl;

  if ( doExit ) exit(1);
}


int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  unsigned int width = 1080u, height = 720u;

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
    texture_path = std::string( sutilSamplesDir() ) + "/arealight_research/data";
  }


  std::stringstream title;
  title << "arealight";
  try {
    _scene = new Arealight(texture_path);
    _scene->setDimensions( width, height );
    GLUTDisplay::setProgressiveDrawingTimeout(0.0);
    GLUTDisplay::run( title.str(), _scene, GLUTDisplay::CDProgressive ); 
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
  return 0;
}
