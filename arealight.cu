/*
* arealight.cu
* Area Light Filtering
* Adapted from NVIDIA OptiX Tutorial
* Brandon Wang, Soham Mehta
*/

#include "arealight.h"
struct MatrixValues
{
  float visibility;
  float d1;
  float d2min;
  float d2max;
  float3 world_loc;
  float proj_dist;
};


rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type , , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );

rtDeclareVariable(float,          light_sigma, , );

// Create ONB from normal.  Resulting W is Parallel to normal
__device__ __inline__ void createONB( const optix::float3& n,
  optix::float3& U,
  optix::float3& V,
  optix::float3& W )
{
  using namespace optix;

  W = normalize( n );
  U = cross( W, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( fabsf( U.x) < 0.001f && fabsf( U.y ) < 0.001f && fabsf( U.z ) < 0.001f  )
    U = cross( W, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( W, U );
}

// Create ONB from normalalized vector
__device__ __inline__ void createONB( const optix::float3& n,
  optix::float3& U,
  optix::float3& V)
{
  using namespace optix;
  U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( dot(U, U) < 1.e-3f )
    U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( n, U );
}

// sample hemisphere with cosine density
__device__ __inline__ void sampleUnitHemisphere( const optix::float2& sample,
  const optix::float3& U,
  const optix::float3& V,
  const optix::float3& W,
  optix::float3& point )
{
  using namespace optix;

  float phi = 2.0f * M_PIf*sample.x;
  float r = sqrt( sample.y );
  float x = r * cos(phi);
  float y = r * sin(phi);
  float z = 1.0f - x*x -y*y;
  z = z > 0.0f ? sqrt(z) : 0.0f;

  point = x*U + y*V + z*W;
}

// HeatMap visualization
__device__ __inline__ float3 heatMap(float val) {
  float fraction;
  if (val < 0.0f)
    fraction = -1.0f;
  else if (val > 1.0f)
    fraction = 1.0f;
  else
    fraction = 2.0f * val - 1.0f;

  if (fraction < -0.5f)
    return make_float3(0.0f, 2*(fraction+1.0f), 1.0f);
  else if (fraction < 0.0f)
    return make_float3(0.0f, 1.0f, 1.0f - 2.0f * (fraction + 0.5f));
  else if (fraction < 0.5f)
    return make_float3(2.0f * fraction, 1.0f, 0.0f);
  else
    return make_float3(1.0f, 1.0f - 2.0f*(fraction - 0.5f), 0.0f);
}

rtBuffer<float, 1>              gaussian_lookup;


// Our Gaussian Filter, based on w_xf
__device__ __inline__ float gaussFilter(float distsq, float wxf)
{
  float sample = distsq*wxf*wxf;
  if (sample > 0.9999) {
    return 0.0;
  }

  return exp(-3*sample);
  //return exp(-4*sample);
  //return exp(-2*sample);
  //return exp(-0.5*sample);
}


//marsaglia polar method
__device__ __inline__ float2 randomGauss(float center, float std_dev, float2 sample)
{
  float u,v,s;
  u = sample.x * 2 - 1;
  v = sample.y * 2 - 1;
  s = u*u + v*v;
  float2 result = make_float2(
    center+std_dev*v*sqrt(-2.0*log(s)/s),
    center+std_dev*u*sqrt(-2.0*log(s)/s));
  return result;
}

//
// Pinhole camera implementation
//
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtBuffer<uchar4, 2>              output_buffer;
rtBuffer<MatrixValues, 2>              matrix_vals;

rtDeclareVariable(float3, bg_color, , );

rtBuffer<float3, 2>               brdf;
//divided occlusion, undivided occlusion, wxf, num_samples
rtBuffer<float3, 2>               vis;

rtDeclareVariable(uint,           frame, , );
rtDeclareVariable(uint,           view_mode, , );

rtDeclareVariable(uint,           normal_rpp, , );
rtDeclareVariable(uint,           brute_rpp, , );
rtDeclareVariable(uint,           max_rpp_pass, , );
rtDeclareVariable(uint,           show_progressive, , );
rtDeclareVariable(float,          zmin_rpp_scale, , );
rtDeclareVariable(int2,           pixel_radius, , );
rtDeclareVariable(int2,           pixel_radius_wxf, , );

rtDeclareVariable(uint,           show_brdf, , );
rtDeclareVariable(uint,           show_occ, , );

rtDeclareVariable(float,          max_disp_val, , );
rtDeclareVariable(float,          min_disp_val, , );

rtDeclareVariable(float,          spp_mu, , );


RT_PROGRAM void pinhole_camera_initial_sample() {
  // Find direction to shoot ray in
  size_t2 screen = output_buffer.size();

  //capture only a scan line

  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
  //125 is an "interesting" portion of the balance scene
  if (show_brdf)
    d = make_float2(launch_index.x, 125./480. * screen.y) / make_float2(screen) * 2.f - 1.f;
  //now hack together a matrix, using launch_index.x to stratify the light.



  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  PerRayData_radiance prd;

  // Initialize the stuff we use in later passes
  vis[launch_index] = make_float3(1.0, 0.0, 0.0);
  brdf[launch_index] = make_float3(0,0,0);

  optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon);
  prd.first_pass = true;
  prd.sqrt_num_samples = normal_rpp;
  prd.unavg_vis = 0.0f;
  prd.vis_weight_tot = 0.0f;
  prd.hit_shadow = false;
  prd.use_filter_n = false;
  prd.hit = false;
  prd.obj_id = -1;

  rtTrace(top_object, ray, prd);


  if (!prd.hit) {
    vis[launch_index] = make_float3(1,1,0);
    brdf[launch_index].x = -2;
    return;
  }

  brdf[launch_index] = prd.brdf;

  vis[launch_index].x = 1;


  if (prd.hit_shadow && prd.vis_weight_tot > 0.01) {
    vis[launch_index].x = prd.unavg_vis/prd.vis_weight_tot;
  }
  vis[launch_index].y = prd.unavg_vis;
  vis[launch_index].z = prd.vis_weight_tot;
}

RT_PROGRAM void display_camera() {
  float3 cur_vis = vis[launch_index];
  float blurred_vis = cur_vis.x;

  if (brdf[launch_index].x < -1.0f) {
    output_buffer[launch_index] = make_color( bg_color );
    return;
  }

  float3 brdf_term = make_float3(1);
  float vis_term = 1;
  if (show_brdf)
    brdf_term = brdf[launch_index];
  brdf_term = brdf[launch_index];
  if (show_occ)
    vis_term = blurred_vis;
  output_buffer[launch_index] = make_color( vis_term * brdf_term);

}

rtBuffer<AreaLight>        lights;
rtDeclareVariable(float3,        lightnorm, , );

//
// Returns solid color for miss rays
//
RT_PROGRAM void miss()
{
  prd_radiance.brdf = bg_color;
  prd_radiance.hit_shadow = false;
}

//
// Terminates and fully attenuates ray after any hit
//
RT_PROGRAM void any_hit_shadow()

{
  prd_shadow.attenuation = make_float3(0);
  prd_shadow.hit = true;

  prd_shadow.distance_min = min(prd_shadow.distance_min, t_hit);
  prd_shadow.distance_max = max(prd_shadow.distance_max, t_hit);

  rtIgnoreIntersection();
}


//
// Phong surface shading with shadows
//
rtDeclareVariable(float3,   Ka, , );
rtDeclareVariable(float3,   Ks, , );
rtDeclareVariable(float,    phong_exp, , );
rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(int,      obj_id, , );
rtDeclareVariable(float3,   ambient_light_color, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float3, reflectivity, , );
rtDeclareVariable(float, importance_cutoff, , );
rtDeclareVariable(int, max_depth, , );
rtDeclareVariable(unsigned int,          matrix_samp_mult, , );

//asdf
rtBuffer<uint2, 2> shadow_rng_seeds;

RT_PROGRAM void closest_hit_radiance3()
{
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 color = Ka * ambient_light_color;

  float3 hit_point = ray.origin + t_hit * ray.direction;
  prd_radiance.t_hit = t_hit;
  prd_radiance.world_loc = hit_point;
  prd_radiance.hit = true;
  prd_radiance.n = ffnormal;
  prd_radiance.obj_id = obj_id;

  uint2 seed = shadow_rng_seeds[launch_index];


  //Assume 1 light for now
  AreaLight light = lights[0];
  float3 lx = light.v2 - light.v1;
  float3 ly = light.v3 - light.v1;
  float3 lo = light.v1;
  //float3 lc = light.color;
  float3 colorAvg = make_float3(0,0,0);

  //phong values
  float3 light_center = (0.5 * lx + 0.5 * ly) +lo;
  float3 to_light = light_center - hit_point;
  float dist_to_light = sqrt(to_light.x*to_light.x + to_light.y*to_light.y + to_light.z*to_light.z);
  prd_radiance.dist_to_light = dist_to_light;
  if(prd_radiance.first_pass) {
    float3 L = normalize(to_light);
    float nDl = max(dot( ffnormal, L ),0.0f);
    float3 H = normalize(L - ray.direction);
    float nDh = max(dot( ffnormal, H ),0.0f);
    //temporary - white light
    float3 Lc = make_float3(1,1,1);
    color += Kd * nDl * Lc;// * strength;
    if (nDh > 0)
      color += Ks * pow(nDh, phong_exp);
    prd_radiance.brdf = color;
  }

  //Stratify x
  //for(int i=0; i<prd_radiance.sqrt_num_samples; ++i) {
  //jk dont stratify x
  for(int i=0; i<1; ++i) {
    seed.x = rot_seed(seed.x, i);

    //Stratify y
    //for(int j=0; j<matrix_vals.size().y; ++j) {
    for(int j=0; j<matrix_samp_mult; ++j) {
      seed.y = rot_seed(seed.y, j);

      float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );
      sample.x = (sample.x+((float)i))/prd_radiance.sqrt_num_samples;
      sample.y = (sample.y+((float)j))/prd_radiance.sqrt_num_samples;

      sample.x = 0.5;
      sample.y = (float)(launch_index.y*matrix_samp_mult+j) / (matrix_vals.size().y);

      float3 target = (sample.x * lx + sample.y * ly) + lo;

      float strength = exp( -0.5 * ((light_center.x - target.x) * (light_center.x - target.x) \
        + (light_center.y - target.y) * (light_center.y - target.y) \
        + (light_center.z - target.z) * (light_center.z - target.z)) \
        / ( light_sigma * light_sigma));

      float3 sampleDir = normalize(target - hit_point);

      if(dot(ffnormal, sampleDir) > 0.0f) {
        prd_radiance.use_filter_n = true;
        prd_radiance.vis_weight_tot += strength;

        // SHADOW
        //cast ray and check for shadow
        PerRayData_shadow shadow_prd;
        shadow_prd.attenuation = make_float3(strength);
        shadow_prd.distance_max = 0;
        shadow_prd.distance_min = dist_to_light;
        shadow_prd.hit = false;
        optix::Ray shadow_ray ( hit_point, sampleDir, shadow_ray_type, 0.001);//scene_epsilon );
        rtTrace(top_shadower, shadow_ray, shadow_prd);

        MatrixValues cur_m_val;
        cur_m_val.visibility = strength;
        cur_m_val.d1 = 0;
        cur_m_val.d2min = 0;
        cur_m_val.d2max = 0;

        cur_m_val.world_loc = hit_point;
        
        size_t2 screen = output_buffer.size();
        //vfov for balance is 60
        const float vfov = 60.f;
        cur_m_val.proj_dist = 2.f/screen.y * t_hit * tan(vfov/2.f*M_PI/180.f);

        if(shadow_prd.hit) {
          prd_radiance.hit_shadow = true;
          float d2min = dist_to_light - shadow_prd.distance_max;
          float d2max = dist_to_light - shadow_prd.distance_min;
          if (shadow_prd.distance_max < 0.000000001)
            d2min = dist_to_light;
          cur_m_val.visibility = 0;
          cur_m_val.d1 = dist_to_light;
          cur_m_val.d2min = d2min;
          cur_m_val.d2max = d2max;

          float s1 = dist_to_light/d2min - 1.0;
          float s2 = dist_to_light/d2max - 1.0;

          prd_radiance.s1 = max(prd_radiance.s1, s1);
          prd_radiance.s2 = min(prd_radiance.s2, s2);
        } else {
          prd_radiance.unavg_vis += strength;
        }

        matrix_vals[make_uint2(launch_index.x, launch_index.y*matrix_samp_mult + j)] = cur_m_val;
      }

    }
  }

  shadow_rng_seeds[launch_index] = seed;

}


//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
