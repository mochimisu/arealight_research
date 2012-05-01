/*
 * arealight.cu
 * Area Light Filtering
 * Adapted from NVIDIA OptiX Tutorial
 * Brandon Wang, Soham Mehta
 */

#include "arealight.h"

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

__device__ __inline__ float gaussFilter(float distsq, float wxf)
{
  float sample = distsq*wxf*wxf;
  if (sample > 0.9999) {
    return 0.0;
  }
  float scaled = sample*64;
  int index = (int) scaled;
  float weight = scaled - index;
  return (1.0 - weight) * gaussian_lookup[index] + weight * gaussian_lookup[index + 1];
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

rtDeclareVariable(float3, bg_color, , );

rtBuffer<float3, 2>               brdf;
//divided occlusion, undivided occlusion, zmin, num_samples
rtBuffer<float4, 2>               occ;
rtBuffer<float, 2>                occ_blur1d;
rtBuffer<float3, 2>               world_loc;
rtBuffer<float3, 2>               n;
rtBuffer<float, 2>                dist_scale;
rtBuffer<float2, 2>               zdist;
rtBuffer<float, 2>               wxf_blur1d;
rtBuffer<float, 2>                spp;
rtBuffer<float, 2>                spp_cur;
//s1,s2
rtBuffer<float2, 2>               slope;
rtBuffer<uint, 2>                 use_filter;
rtDeclareVariable(uint,           frame, , );
rtDeclareVariable(uint,           blur_occ, , );
rtDeclareVariable(uint,           blur_wxf, , );
rtDeclareVariable(uint,           err_vis, , );
rtDeclareVariable(uint,						view_mode, , );

rtDeclareVariable(uint,           normal_rpp, , );
rtDeclareVariable(uint,           brute_rpp, , );
rtDeclareVariable(uint,           max_rpp_pass, , );
rtDeclareVariable(uint,           show_progressive, , );
rtDeclareVariable(float,          zmin_rpp_scale, , );
rtDeclareVariable(int2,           pixel_radius, , );
rtDeclareVariable(int2,           pixel_radius_wxf, , );

rtDeclareVariable(uint,           show_brdf, , );
rtDeclareVariable(uint,           show_occ, , );

__device__ __inline__ float computeSpp( float t_hit,
  float s1, float s2, float wxf ) {
    //Currently assuming fov of 60deg, height of 720p, 1:1 aspect
    float d = 1.0/360.0 * (t_hit*tan(30.0*M_PI/180.0));
    float spp_t_1 = (1+d*wxf);
    float spp_t_2 = (1+s1/s2);
    float spp = 4*spp_t_1*spp_t_1*spp_t_2*spp_t_2;
    return spp;
}

RT_PROGRAM void pinhole_camera_initial_sample() {
  size_t2 screen = output_buffer.size();
  
  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  PerRayData_radiance prd;

  prd.sqrt_num_samples = normal_rpp;

  //Initialize the stuff we use in later passes
  occ[launch_index] = make_float4(1.0, 0.0, 100000.0, 0.0);
  prd.brdf = true;
  zdist[launch_index] = make_float2(10000.0, 0.0);
  slope[launch_index] = make_float2(0.0, 10000.0);
  float current_spp = normal_rpp * normal_rpp;
  //spp[launch_index] = current_spp;
  //spp_cur[launch_index] = current_spp;
  use_filter[launch_index] = false;

  optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon);
  prd.importance = 1.0f;
  prd.unavg_occ = 0.0f;
  prd.depth = 0.0f;
  prd.hit = false;
  prd.d2min = zdist[launch_index].x;
  prd.d2max = zdist[launch_index].y;
  prd.s1 = slope[launch_index].x;
  prd.s2 = slope[launch_index].y;
  prd.use_filter = false;
  prd.wxf = 0.0f;

  rtTrace(top_object, ray, prd);

  if (!prd.hit) {
    occ[launch_index] = make_float4(1,1,-1,0);
    use_filter[launch_index] = false;
    spp[launch_index] = 0;
    return;
  }
  float theoretical_spp = 0;
  if(prd.hit_shadow)
    theoretical_spp = computeSpp(prd.t_hit, prd.s1, prd.s2, prd.wxf);

  world_loc[launch_index] = prd.world_loc;
  brdf[launch_index] = prd.result;
  n[launch_index] = normalize(prd.n);
  dist_scale[launch_index] = prd.dist_scale;

  zdist[launch_index] = make_float2(prd.d2min, prd.d2max);
  slope[launch_index] =make_float2(prd.s1, prd.s2);
  use_filter[launch_index] = prd.use_filter;
  if (!prd.use_filter)
    occ[launch_index].x = 0;

  spp_cur[launch_index] = current_spp;
  spp[launch_index] = min(theoretical_spp, (float) brute_rpp * brute_rpp);

  if (prd.hit_shadow && prd.num_occ > 0.01) {
    occ[launch_index].x = prd.unavg_occ/prd.num_occ;
    occ[launch_index].y = prd.unavg_occ;
  }
  occ[launch_index].z = prd.wxf;
  occ[launch_index].w = prd.num_occ;
}

RT_PROGRAM void pinhole_camera_continue_sample() {  size_t2 screen = output_buffer.size();

  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  PerRayData_radiance prd;

  prd.importance = 1.0f;
  prd.unavg_occ = 0.0f;
  prd.depth = 0.0f;
  prd.hit = false;
  prd.d2min = zdist[launch_index].x;
  prd.d2max = zdist[launch_index].y;
  prd.s1 = slope[launch_index].x;
  prd.s2 = slope[launch_index].y;
  prd.use_filter = false;
  prd.wxf = occ[launch_index].z;
  optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon);

  float target_spp = spp[launch_index];
  target_spp = 100000.0;
  float cur_spp = spp_cur[launch_index];

  // Compute spp difference
  if (cur_spp < target_spp ) {
    int new_samp = min((int) (target_spp - cur_spp), (int) max_rpp_pass);
    int sqrt_samp = ceil(sqrt((float)new_samp));
    prd.sqrt_num_samples = sqrt_samp;
    cur_spp = cur_spp + sqrt_samp * sqrt_samp;

    spp_cur[launch_index] = cur_spp;

    rtTrace(top_object, ray, prd);
    occ[launch_index].z = prd.wxf;
    occ[launch_index].w += prd.num_occ;
    if (prd.hit_shadow && prd.num_occ > 0.01) {
      occ[launch_index].y += prd.unavg_occ;
      occ[launch_index].x = occ[launch_index].y/occ[launch_index].w;
    }
  }

}


RT_PROGRAM void display_camera() {
  float4 cur_occ = occ[launch_index];
  float blurred_occ = cur_occ.x;

  if (cur_occ.z < 0.0f) {
    output_buffer[launch_index] = make_color( bg_color );
    return;
  }

  float3 brdf_term = make_float3(1);
  float occ_term = 1;
  if (show_brdf)
    brdf_term = brdf[launch_index];
  if (show_occ)
    occ_term = blurred_occ;
  if (view_mode) {
    if (view_mode == 1)
      //Occlusion only
      output_buffer[launch_index] = make_color( make_float3(blurred_occ) );
    if (view_mode == 2) 
      //Scale
      //output_buffer[launch_index] = make_color( make_float3(scale) );
      output_buffer[launch_index] = make_color( heatMap(1/(cur_occ.z/light_sigma)*5.0) );
    if (view_mode == 3) 
      //Current SPP
      //output_buffer[launch_index] = make_color( make_float3(spp_cur[launch_index]) / 100.0 );
      output_buffer[launch_index] = make_color( heatMap(spp_cur[launch_index] / 100.0 ) );
    if (view_mode == 4) 
      //Theoretical SPP
      //output_buffer[launch_index] = make_color( make_float3(spp[launch_index]) / 100.0 );
      output_buffer[launch_index] = make_color( heatMap(spp[launch_index] / 100.0 ) );
  } else
    output_buffer[launch_index] = make_color( occ_term * brdf_term);


}

__device__ __inline__ void occlusionFilter( float& blurred_occ_sum,
  float& sum_weight, const optix::float3& cur_world_loc, float3 cur_n,
  float wxf, int i, int j, const optix::size_t2& buf_size, 
  unsigned int pass ) {
    const float dist_scale_threshold = 1.0f;
    const float dist_threshold = 1.0f;
    const float angle_threshold = 20.0f * M_PI/180.0f;
    if (i > 0 && i < buf_size.x && j > 0 && j <buf_size.y) {
      uint2 target_index = make_uint2(i,j);
      float4 target_occ = occ[target_index];
      float target_wxf = target_occ.z;
      if (target_wxf > 0 && abs(wxf - target_wxf) < dist_scale_threshold &&
         use_filter[target_index] ) {
          float3 target_loc = world_loc[target_index];
          float3 diff = cur_world_loc - target_loc;
          float distancesq = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
          if (distancesq < 1.0) {
            float3 target_n = n[target_index];
            if (acos(dot(target_n, cur_n)) < angle_threshold) {
              float weight = gaussFilter(distancesq, wxf);
              float target_occ_val = target_occ.x;
              if (pass == 1) {
                target_occ_val = occ_blur1d[target_index];
              }
              blurred_occ_sum += weight * target_occ_val;
              sum_weight += weight;
            }
          }
      }
    }
}

RT_PROGRAM void occlusion_filter_first_pass() {
  float4 cur_occ = occ[launch_index];
  float wxf = cur_occ.z;
  float blurred_occ = cur_occ.x;
  if (wxf < 0.0) {
    occ_blur1d[launch_index] = blurred_occ;
    return;
  }
  
  if (blur_occ) {

    float blurred_occ_sum = 0.0f;
    float sum_weight = 0.0f;

    float3 cur_world_loc = world_loc[launch_index];
    float3 cur_n = n[launch_index];
    size_t2 buf_size = occ.size();

    for (int i = -pixel_radius.x; i < pixel_radius.x; i++) {
      occlusionFilter(blurred_occ_sum, sum_weight, cur_world_loc, cur_n, wxf,
        launch_index.x+i, launch_index.y, buf_size, 0);
    }

    if (sum_weight > 0.0f)
      blurred_occ = blurred_occ_sum / sum_weight;
    else
      blurred_occ = blurred_occ_sum;
  }

  occ_blur1d[launch_index] = blurred_occ;
}

RT_PROGRAM void occlusion_filter_second_pass() {
  float4 cur_occ = occ[launch_index];
  float wxf = cur_occ.z;
  float blurred_occ = occ_blur1d[launch_index];

  if (blur_occ) {
    if (wxf <= 0.0) {
      occ[launch_index].x = blurred_occ;
      return;
    }

    float blurred_occ_sum = 0.0f;
    float sum_weight = 0.0f;

    float3 cur_world_loc = world_loc[launch_index];
    float3 cur_n = n[launch_index];
    size_t2 buf_size = occ.size();

    for (int j = -pixel_radius.y; j < pixel_radius.y; j++) {
      occlusionFilter(blurred_occ_sum, sum_weight, cur_world_loc, cur_n, wxf,
        launch_index.x, launch_index.y+j, buf_size, 1);
    }

    if (sum_weight > 0.0f)
      blurred_occ = blurred_occ_sum / sum_weight;
    else
      blurred_occ = blurred_occ_sum;
  }

  occ[launch_index].x = blurred_occ;
}

__device__ __inline__ void wxfFilterBilateral( float& blurred_wxf_sum,
  float& sum_weight, float cur_wxf, unsigned int i,
  unsigned int j, const::optix::size_t2& buf_size, unsigned int pass = 0) {
  const float k = 0.5; //Guessing...

    if (i > 0 && i < buf_size.x && j > 0 && j < buf_size.y) {
      uint2 target_index = make_uint2(i,j);
      if (occ[target_index].z < 0)
        return;
      float target_wxf;
      if (pass == 0) {
        target_wxf = occ[target_index].z;
      } else {
        target_wxf = wxf_blur1d[target_index];
      }
      if (target_wxf > 10000.0)
        return;
      float wxf_diff = target_wxf - cur_wxf;
      float weight = 1.0; // exp(-(wxf_diff*wxf_diff)/(k*k));
      blurred_wxf_sum += weight * target_wxf;
      sum_weight += weight;
    }
}

RT_PROGRAM void wxf_filter_first_pass() {
  float blurred_wxf = occ[launch_index].z;

  if (blur_wxf && blurred_wxf > 0) {
    if (blurred_wxf > 10000.0) {
      blurred_wxf = 0.5f;
    }
    float blurred_wxf_sum = 0.0f;
    float sum_weight = 0.0f;
    size_t2 buf_size = occ.size(); 

    for (int i = -pixel_radius_wxf.x; i < pixel_radius_wxf.x; i++) {
      wxfFilterBilateral(blurred_wxf_sum, sum_weight, blurred_wxf,
        launch_index.x+i, launch_index.y, buf_size, 0);
    }
    if (sum_weight > 0.0001f)
      blurred_wxf = blurred_wxf_sum / sum_weight;
    else
      blurred_wxf = blurred_wxf_sum;
  }

  wxf_blur1d[launch_index] = blurred_wxf;
}
RT_PROGRAM void wxf_filter_second_pass() {
  float blurred_wxf = wxf_blur1d[launch_index];

  if (blur_wxf && blurred_wxf > 0) {    if (blurred_wxf > 10000.0) {
      blurred_wxf = 0.5f;
    }
    float blurred_wxf_sum = 0.0f;
    float sum_weight = 0.0f;
    size_t2 buf_size = occ.size(); 

    for (int j = -pixel_radius_wxf.y; j < pixel_radius_wxf.y; j++) {
      wxfFilterBilateral(blurred_wxf_sum, sum_weight, blurred_wxf,
        launch_index.x, launch_index.y+j, buf_size, 1);
    }
    if (sum_weight > 0.00001f)
      blurred_wxf = blurred_wxf_sum / sum_weight;
    else
      blurred_wxf = blurred_wxf_sum;
  }

  occ[launch_index].z = blurred_wxf;
}

//
// Returns solid color for miss rays
//
RT_PROGRAM void miss()
{
  prd_radiance.result = bg_color;
  prd_radiance.hit_shadow = false;
  //prd_radiance.shadow_intersection = 100000.0f;
}

//
// Terminates and fully attenuates ray after any hit
//
RT_PROGRAM void any_hit_shadow()

{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);
  prd_shadow.hit = true;
  //prd_shadow.distance = t_hit;
  
  prd_shadow.distance_min = min(prd_shadow.distance_min, t_hit);
  prd_shadow.distance_max = max(prd_shadow.distance_max, t_hit);

  rtIgnoreIntersection();

  //rtTerminateRay();
}


//
// Phong surface shading with shadows
//
rtDeclareVariable(float3,   Ka, , );
rtDeclareVariable(float3,   Ks, , );
rtDeclareVariable(float,    phong_exp, , );
rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(float3,   ambient_light_color, , );
rtBuffer<AreaLight>        lights;
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float3, reflectivity, , );
rtDeclareVariable(float, importance_cutoff, , );
rtDeclareVariable(int, max_depth, , );

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
  //occlusion values
  unsigned int occlusion = 0;

  prd_radiance.dist_scale = 1.0/(t_hit*tan(30.0*M_PI/180.0)); 
      //divide by 360 to get absolute between pixel and image


  uint2 seed = shadow_rng_seeds[launch_index];
  //seed.x = rot_seed(seed.x, frame);
  //seed.y = rot_seed(seed.y, frame);
  //float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );

  //shadow_rng_seeds[launch_index] = seed;


  prd_radiance.wxf = 0.0f;
  //Assume 1 light for now
  AreaLight light = lights[0];
  float3 lx = light.v2 - light.v1;
  float3 ly = light.v3 - light.v1;
  float3 lo = light.v1;
  //float3 lc = light.color;
  float3 colorAvg = make_float3(0,0,0);

  //phong values
  //assuming ambocc for now
  //THIS IS WRONG, FIX IT SOON
  /*
     if (prd_radiance.brdf) {

     float3 H = normalize(ffnormal - ray.direction);
     float nDh = dot( ffnormal, H );
     if (nDh > 0)
     colorAvg += Ks * pow(nDh, phong_exp);
     }
   */
  if(prd_radiance.brdf) {
    float3 phong_target = 0.5 * lx + 0.5 * ly + lo;
    float3 phong_dir = normalize( phong_target - hit_point );

    float3 L = normalize(phong_target - hit_point);
    float nDl = dot( ffnormal, L );
    float3 H = normalize(phong_dir - ray.direction);
    float nDh = dot( ffnormal, H );
    //temporary - white light
    float3 Lc = make_float3(1,1,1);
    color += Kd * nDl * Lc;// * strength;
    if (nDh > 0)
      color += Ks * pow(nDh, phong_exp);
  }

  //hardcoded sigma for now (for light intensity)

  //Stratify x
  float num_occ = 0;
  float occ_strength_tot = 0.0;
  float distance_summed = 0.0;
  bool hit_shadow = false;
  float3 light_center = (0.5 * lx + 0.5 * ly) +lo;
  for(int i=0; i<prd_radiance.sqrt_num_samples; ++i) {
    seed.x = rot_seed(seed.x, i);

    //Stratify y
    for(int j=0; j<prd_radiance.sqrt_num_samples; ++j) {
      seed.y = rot_seed(seed.y, j);

      float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );
      sample.x = (sample.x+((float)i))/prd_radiance.sqrt_num_samples;
      sample.y = (sample.y+((float)j))/prd_radiance.sqrt_num_samples;

      
      float strength = exp(-0.5*((sample.x-0.5)*(sample.x-0.5) \
        + (sample.y-0.5) * (sample.y-0.5)) \
        / (2 * light_sigma * light_sigma));
      
      strength = 1;

      float3 target = (sample.x * lx + sample.y * ly) + lo;

      /*
      float strength = exp( -0.5 * ((sample.x - target.x) * (sample.x - target.x) \
        + (sample.y - target.y) * (sample.y - target.y)) \
        / ( 2 * light_sigma * light_sigma));
      */

      float3 sampleDir = normalize(target - hit_point);

      float3 distancevectoocc = target-hit_point;
      float distancetolight = sqrt(distancevectoocc.x*distancevectoocc.x + distancevectoocc.y*distancevectoocc.y + distancevectoocc.z * distancevectoocc.z);
      distance_summed += distancetolight;

      if(dot(ffnormal, sampleDir) > 0.0f) {
        prd_radiance.use_filter = true;
        num_occ += strength;


        // SHADOW
        //cast ray and check for shadow
        PerRayData_shadow shadow_prd;
        shadow_prd.attenuation = make_float3(strength);
        shadow_prd.distance_max = 0;
        shadow_prd.distance_min = distancetolight;
        shadow_prd.hit = false;
        optix::Ray shadow_ray ( hit_point, sampleDir, shadow_ray_type, 0.001);//scene_epsilon );
        rtTrace(top_shadower, shadow_ray, shadow_prd);
        occlusion += shadow_prd.attenuation.x;
 
        if(shadow_prd.hit) {
          hit_shadow = true;
          float d2min = distancetolight - shadow_prd.distance_max;
          float d2max = distancetolight - shadow_prd.distance_min;
          if (shadow_prd.distance_max < 0.000000001)
            d2min = distancetolight;

          prd_radiance.d2min = d2min;
          prd_radiance.d2max = d2max;

          float scale = distancetolight/d2min - 1;
          float wxf = 1/(scale*light_sigma);
          float wxfsq = wxf*wxf;
          prd_radiance.wxf = max(wxf,prd_radiance.wxf);
        }
      }

    }
  }

  shadow_rng_seeds[launch_index] = seed;
  distance_summed /= prd_radiance.sqrt_num_samples * prd_radiance.sqrt_num_samples;

  /*
     float importance = prd_radiance.importance * optix::luminance( reflectivity );

     if( importance > importance_cutoff && prd_radiance.depth < max_depth) {
     PerRayData_radiance refl_prd;
     refl_prd.importance = importance;
     refl_prd.depth = prd_radiance.depth+1;
     float3 R = reflect( ray.direction, ffnormal );
     optix::Ray refl_ray( hit_point, R, radiance_ray_type, scene_epsilon );
     rtTrace(top_object, refl_ray, refl_prd);
  //color += reflectivity * refl_prd.result;
  }*/

  float s1 = distance_summed/prd_radiance.d2min - 1.0;
  float s2 = distance_summed/prd_radiance.d2max - 1.0;
  s1 = max(prd_radiance.s1,s1);
  s2 = min(prd_radiance.s2,s2);
  prd_radiance.s1 = s1;
  prd_radiance.s2 = s2;


  if(!hit_shadow) {
    //prd_radiance.spp = 0;
    prd_radiance.wxf = 0.1f; //100000.0f;
  }
  prd_radiance.hit_shadow = hit_shadow;

  prd_radiance.unavg_occ = occlusion;
  prd_radiance.num_occ = num_occ;
  prd_radiance.result = color;
}


//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
